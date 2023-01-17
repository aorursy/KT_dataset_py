import os

from os.path import join



import numpy as np

import pandas as pd



from scipy.spatial import distance_matrix



import torch

from torch import nn

from torch.utils.data import Dataset, DataLoader



try:

    import dgl

except:

    !pip install dgl

    import dgl

    

import warnings

warnings.filterwarnings('ignore')
"""

layers-script from dgl-schnet implementation:

https://github.com/dmlc/dgl/blob/master/examples/pytorch/schnet/layers.py

"""



import torch as th

import numpy as np

import torch.nn as nn

import dgl.function as fn

from torch.nn import Softplus





class AtomEmbedding(nn.Module):

    """

    Convert the atom(node) list to atom embeddings.

    The atom with the same element share the same initial embeddding.

    """



    def __init__(self, dim=128, type_num=100, pre_train=None):

        """

        Randomly init the element embeddings.

        Args:

            dim: the dim of embeddings

            type_num: the largest atomic number of atoms in the dataset

            pre_train: the pre_trained embeddings

        """

        super().__init__()

        self._dim = dim

        self._type_num = type_num

        if pre_train is not None:

            self.embedding = nn.Embedding.from_pretrained(pre_train,

                                                          padding_idx=0)

        else:

            self.embedding = nn.Embedding(type_num, dim, padding_idx=0)



    def forward(self, g, p_name="node"):

        """Input type is dgl graph"""

        atom_list = g.ndata["node_type"]

        g.ndata[p_name] = self.embedding(atom_list)

        return g.ndata[p_name]





class EdgeEmbedding(nn.Module):

    """

    Convert the edge to embedding.

    The edge links same pair of atoms share the same initial embedding.

    """



    def __init__(self, dim=128, edge_num=3000, pre_train=None):

        """

        Randomly init the edge embeddings.

        Args:

            dim: the dim of embeddings

            edge_num: the maximum type of edges

            pre_train: the pre_trained embeddings

        """

        super().__init__()

        self._dim = dim

        self._edge_num = edge_num

        if pre_train is not None:

            self.embedding = nn.Embedding.from_pretrained(pre_train,

                                                          padding_idx=0)

        else:

            self.embedding = nn.Embedding(edge_num, dim, padding_idx=0)



    def generate_edge_type(self, edges):

        """

        Generate the edge type based on the src&dst atom type of the edge.

        Note that C-O and O-C are the same edge type.

        To map a pair of nodes to one number, we use an unordered pairing function here

        See more detail in this disscussion:

        https://math.stackexchange.com/questions/23503/create-unique-number-from-2-numbers

        Note that, the edge_num should larger than the square of maximum atomic number

        in the dataset.

        """

        atom_type_x = edges.src["node_type"]

        atom_type_y = edges.dst["node_type"]



        return {

            "type":

            atom_type_x * atom_type_y +

            (th.abs(atom_type_x - atom_type_y) - 1)**2 / 4

        }



    def forward(self, g, p_name="edge_f"):

        g.apply_edges(self.generate_edge_type)

        g.edata[p_name] = self.embedding(g.edata["type"])

        return g.edata[p_name]





class ShiftSoftplus(Softplus):

    """

    Shiftsoft plus activation function:

        1/beta * (log(1 + exp**(beta * x)) - log(shift))

    """



    def __init__(self, beta=1, shift=2, threshold=20):

        super().__init__(beta, threshold)

        self.shift = shift

        self.softplus = Softplus(beta, threshold)



    def forward(self, input):

        return self.softplus(input) - np.log(float(self.shift))





class RBFLayer(nn.Module):

    """

    Radial basis functions Layer.

    e(d) = exp(- gamma * ||d - mu_k||^2)

    default settings:

        gamma = 10

        0 <= mu_k <= 30 for k=1~300

    """



    def __init__(self, low=0, high=30, gap=0.1, dim=1):

        super().__init__()

        self._low = low

        self._high = high

        self._gap = gap

        self._dim = dim



        self._n_centers = int(np.ceil((high - low) / gap))

        centers = np.linspace(low, high, self._n_centers)

        self.centers = th.tensor(centers, dtype=th.float, requires_grad=False)

        self.centers = nn.Parameter(self.centers, requires_grad=False)

        self._fan_out = self._dim * self._n_centers



        self._gap = centers[1] - centers[0]



    def dis2rbf(self, edges):        

        dist = edges.data["distance"]

        radial = dist - self.centers

        coef = -1 / self._gap

        rbf = th.exp(coef * (radial**2))

        return {"rbf": rbf}



    def forward(self, g):

        """Convert distance scalar to rbf vector"""

        g.apply_edges(self.dis2rbf)

        return g.edata["rbf"]





class CFConv(nn.Module):

    """

    The continuous-filter convolution layer in SchNet.

    One CFConv contains one rbf layer and three linear layer

        (two of them have activation funct).

    """



    def __init__(self, rbf_dim, dim=64, act="sp"):

        """

        Args:

            rbf_dim: the dimsion of the RBF layer

            dim: the dimension of linear layers

            act: activation function (default shifted softplus)

        """

        super().__init__()

        self._rbf_dim = rbf_dim

        self._dim = dim



        self.linear_layer1 = nn.Linear(self._rbf_dim, self._dim)

        self.linear_layer2 = nn.Linear(self._dim, self._dim)



        if act == "sp":

            self.activation = nn.Softplus(beta=0.5, threshold=14)

        else:

            self.activation = act



    def update_edge(self, edges):

        rbf = edges.data["rbf"]

        h = self.linear_layer1(rbf)

        h = self.activation(h)

        h = self.linear_layer2(h)

        return {"h": h}



    def forward(self, g):

        g.apply_edges(self.update_edge)

        g.update_all(message_func=fn.u_mul_e('new_node', 'h', 'neighbor_info'),

                     reduce_func=fn.sum('neighbor_info', 'new_node'))

        return g.ndata["new_node"]





class Interaction(nn.Module):

    """

    The interaction layer in the SchNet model.

    """



    def __init__(self, rbf_dim, dim):

        super().__init__()

        self._node_dim = dim

        self.activation = nn.Softplus(beta=0.5, threshold=14)

        self.node_layer1 = nn.Linear(dim, dim, bias=False)

        self.cfconv = CFConv(rbf_dim, dim, act=self.activation)

        self.node_layer2 = nn.Linear(dim, dim)

        self.node_layer3 = nn.Linear(dim, dim)



    def forward(self, g):



        g.ndata["new_node"] = self.node_layer1(g.ndata["node"])

        cf_node = self.cfconv(g)

        cf_node_1 = self.node_layer2(cf_node)

        cf_node_1a = self.activation(cf_node_1)

        new_node = self.node_layer3(cf_node_1a)

        g.ndata["node"] = g.ndata["node"] + new_node

        return g.ndata["node"]





class VEConv(nn.Module):

    """

    The Vertex-Edge convolution layer in MGCN which take edge & vertex features

    in consideratoin at the same time.

    """



    def __init__(self, rbf_dim, dim=64, update_edge=True):

        """

        Args:

            rbf_dim: the dimension of the RBF layer

            dim: the dimension of linear layers

            update_edge: whether update the edge emebedding in each conv-layer

        """

        super().__init__()

        self._rbf_dim = rbf_dim

        self._dim = dim

        self._update_edge = update_edge



        self.linear_layer1 = nn.Linear(self._rbf_dim, self._dim)

        self.linear_layer2 = nn.Linear(self._dim, self._dim)

        self.linear_layer3 = nn.Linear(self._dim, self._dim)



        self.activation = nn.Softplus(beta=0.5, threshold=14)



    def update_rbf(self, edges):

        rbf = edges.data["rbf"]

        h = self.linear_layer1(rbf)

        h = self.activation(h)

        h = self.linear_layer2(h)

        return {"h": h}



    def update_edge(self, edges):

        edge_f = edges.data["edge_f"]

        h = self.linear_layer3(edge_f)

        return {"edge_f": h}



    def forward(self, g):

        g.apply_edges(self.update_rbf)

        if self._update_edge:

            g.apply_edges(self.update_edge)



        g.update_all(

            message_func=[

                fn.u_mul_e("new_node", "h", "m_0"),

                fn.copy_e("edge_f", "m_1")],

            reduce_func=[

                fn.sum("m_0", "new_node_0"),

                fn.sum("m_1", "new_node_1")])

        g.ndata["new_node"] = g.ndata.pop("new_node_0") + g.ndata.pop(

            "new_node_1")



        return g.ndata["new_node"]





class MultiLevelInteraction(nn.Module):

    """

    The multilevel interaction in the MGCN model.

    """



    def __init__(self, rbf_dim, dim):

        super().__init__()



        self._atom_dim = dim



        self.activation = nn.Softplus(beta=0.5, threshold=14)



        self.node_layer1 = nn.Linear(dim, dim, bias=True)

        self.edge_layer1 = nn.Linear(dim, dim, bias=True)

        self.conv_layer = VEConv(rbf_dim, dim)

        self.node_layer2 = nn.Linear(dim, dim)

        self.node_layer3 = nn.Linear(dim, dim)



    def forward(self, g, level=1):

        g.ndata["new_node"] = self.node_layer1(g.ndata["node_%s" %

                                                       (level - 1)])

        node = self.conv_layer(g)

        g.edata["edge_f"] = self.activation(self.edge_layer1(

            g.edata["edge_f"]))

        node_1 = self.node_layer2(node)

        node_1a = self.activation(node_1)

        new_node = self.node_layer3(node_1a)



        g.ndata["node_%s" % (level)] = g.ndata["node_%s" %

                                               (level - 1)] + new_node



        return g.ndata["node_%s" % (level)]
class RBFLayerTensor(RBFLayer):

    """

    Same as DGL's RBFLayer only applied to just a tensor (not a DGLGraph-object with edges).

    """



    def forward(self, dist):

        

        radial = dist - self.centers

        coef   = -1 / self._gap

        rbf    = th.exp(coef * (radial**2))

        

        return rbf

    

    

class Interaction_Dense_BN(nn.Module):

    """

    Like DGL's Interaction-layer only with:

        * added batch-normalization

        * dense-shortcut instead of residual shortcut

    @ rbf_dim: dimension of radial_distance_function(distance)

    @ in_dim: dimension of input node states

    @ k_dim: dimension of newly created node-state features

             (equivalent to growth-rate k in DenseNet)

    return: new node hidden states with dimension in_dim + k_dim

    """



    def __init__(self, rbf_dim, in_dim, k_dim):

        super().__init__()

        

        self.activation  = nn.Softplus(beta=0.5, threshold=14)

        self.node_layer1 = nn.Linear(in_dim, in_dim, bias=False)

        self.cfconv      = CFConv(rbf_dim, in_dim, act=self.activation)

        self.node_layer2 = nn.Linear(in_dim, k_dim)

        self.node_layer3 = nn.Linear(k_dim, k_dim)

        self.batch_norm  = nn.BatchNorm1d(k_dim)



    def forward(self, g):



        g.ndata["new_node"] = self.node_layer1(g.ndata["node"])

        cf_node             = self.cfconv(g)

        cf_node_1           = self.node_layer2(cf_node)

        cf_node_1a          = self.activation(cf_node_1)

        new_node            = self.node_layer3(cf_node_1a)

        

        new_features        = self.batch_norm(new_node)

        g.ndata['node']     = torch.cat([g.ndata['node'], new_features], dim=1)

        

        return g.ndata["node"]



class J_Coupling_Regression(nn.Module):

    

    def __init__(self, input_dim, intermediate_dim, output_dim=1):

        """

        @ input_dim: 2 * node-state-dim + additional input

        @ intermediate_dim: dimension of both hidden layers

        @ output_dim:

            * Set to 1 for predicting sc-constant

            * Set to 4 for predicting the 4 sc-contributions:

              The sum up to the sc-constant but may provide more detailed feedback for the model        

        """

        super().__init__()

        

        self.activation = nn.LeakyReLU(inplace=True)

        

        self.mlp = nn.Sequential(

            nn.Linear(input_dim, input_dim // 2),

            self.activation,

            #nn.Dropout(p=0.2),

            nn.Linear(input_dim // 2, intermediate_dim),

            self.activation,

            #nn.Dropout(p=0.1),

            nn.Linear(intermediate_dim, output_dim)

        )

        

    def forward(self, x):

        """

        x is a concatenation of the hidden-states of 2 j-coupled nodes

        and some additional input

        """            

        return self.mlp(x)



        

class Atominator(nn.Module):

    """

    Schnet for feature extraction and regression to predict j-coupling constant

    """

    

    def __init__(self,

                 num_atom_types=6,  # count starts at 1

                 embedding_dim=128,

                 graph_state_dim=64,

                 output_dim=4,

                 n_conv=3,

                 cutoff=5,

                 width=1):

        super().__init__()

        

        self.embedding_layer = AtomEmbedding(type_num=num_atom_types,

                                             dim=embedding_dim)

        

        self.rbf_layer        = RBFLayer(0, cutoff, width)

        self.tensor_rbf_layer = RBFLayerTensor(0, cutoff, width)

        

        self.n_conv = n_conv

        self.conv_layers = nn.ModuleList(

            [Interaction_Dense_BN(self.rbf_layer._fan_out, in_dim=embedding_dim,     k_dim=embedding_dim),

             Interaction_Dense_BN(self.rbf_layer._fan_out, in_dim=embedding_dim * 2, k_dim=embedding_dim),

             Interaction_Dense_BN(self.rbf_layer._fan_out, in_dim=embedding_dim * 3, k_dim=embedding_dim)]

        )

        final_node_state_dim = embedding_dim * 4

        

        self.readout = nn.Sequential(

            nn.Linear(final_node_state_dim, final_node_state_dim // 2),

            nn.ReLU(inplace=True),

            #nn.Dropout(p=0.2),

            nn.Linear(final_node_state_dim // 2, graph_state_dim)

        )

        

        # 2 node-hidden-states + rbf(distance) + graph_state

        reg_input_dim = (final_node_state_dim * 2

                         + self.tensor_rbf_layer._fan_out

                         + graph_state_dim)

        self.target_regression = J_Coupling_Regression(

            input_dim=reg_input_dim,

            intermediate_dim=128,

            output_dim=output_dim

        )

            

    def forward(self,

                g: dgl.DGLGraph,

                j_pairs: np.array):

        """

        @ g: molecule-graph

        @ j_pairs: (i, j, distance) where i, j are node-indices

        """

                

        self.embedding_layer(g)

        self.rbf_layer(g)

                

        for idx in range(self.n_conv):

            self.conv_layers[idx](g)

            

        node_state_sum = graph.ndata['node'].sum(dim=0)

        graph_state = self.readout(node_state_sum)

        

        concatentations = []

        for id_, i, j, dist in j_pairs:

            rbf_dist  = self.tensor_rbf_layer(torch.tensor([dist]).to(DEVICE))

            concatentations.append(torch.cat([g.ndata['node'][int(i)],

                                              g.ndata['node'][int(j)],

                                              rbf_dist,

                                              graph_state]

                                            ))

        

        concat_batch = torch.stack(concatentations, dim=0)

        y = self.target_regression(concat_batch)

        

        return y  # estimated of j-coupling constant for all coupled atoms
# define all global variables:



DATA_DIR = '../input/champs-scalar-coupling'

ATOM2ENUM = {

    'H': 1,  # start at 1 just to be sure in cas 0 is a default embedding in DGL

    'C': 2,

    'N': 3,

    'O': 4,

    'F': 5

}

J_TYPE = '1JHN'

DEVICE = None  # using only CPU here
def train_val_split(df: pd.DataFrame, val_fraction=0.2):

    """ Split by molecule. """

    molecules     = df.molecule_name.unique().tolist()

    val_molecules = np.random.choice(molecules,

                                     size=int(val_fraction * len(molecules)),

                                     replace=False)

    val_set   = df.query('molecule_name in @val_molecules')

    train_set = df.query('molecule_name not in @val_molecules')

    return train_set, val_set





def load_dataset(j_type=None):

    

    train = pd.merge(pd.read_csv(join(DATA_DIR, 'train.csv')),

                     pd.read_csv(join(DATA_DIR, 'scalar_coupling_contributions.csv')),

                     on=['molecule_name', 'atom_index_0', 'atom_index_1', 'type'])

    test  = pd.read_csv(join(DATA_DIR, 'test.csv'))

    

    if j_type is not None:

        train = train.query('type == @j_type')

        test  = test.query('type == @j_type')

        

    train, valid = train_val_split(train)

    

    return train, valid, test





structures_df = pd.read_csv(join(DATA_DIR, 'structures.csv'))

structures_df.index = structures_df.molecule_name

structures_df = structures_df.drop('molecule_name', axis=1)



train, valid, test = load_dataset(j_type=J_TYPE)



print(f'train: {train.shape}')

print(f'validation: {valid.shape}')

print(f'test: {test.shape}')

train.head()
%%time



class Molecule_Dataset(Dataset):

    

    def __init__(self,

                 structures: pd.DataFrame,

                 targets: pd.DataFrame,

                 adj_cutoff=3,  # create a edges between atoms within this distance

                 train=True,

                 limit=None):

        """

        For each molecule, save in a list:

            * all information required to create a molecule graph

              (The graph has to be created on the fly to avoid memory leakage)

            * all information required for j-coupling regression (atom indices, distance)

        """

        

        self.molecule_list = []

        self.num_j_couplings = len(targets)

        

        self.atom_counts       = []

        self.j_coupling_counts = []



        for i, (mol_name, group_df) in enumerate(targets.groupby('molecule_name')):



            struct_df = structures.loc[mol_name]

            

            self.atom_counts.append(len(struct_df))

            self.j_coupling_counts.append(len(group_df))

  

            atom_types  = struct_df.atom.map(ATOM2ENUM).values

            coords      = struct_df[['x', 'y', 'z']].values

            dist_matrix = distance_matrix(coords, coords)

            adj_matrix  = np.multiply(dist_matrix <= adj_cutoff,  dist_matrix > 0)

            edges       = np.where(adj_matrix > 0)

            distances   = torch.tensor(dist_matrix[edges].tolist())



            graph_input = (atom_types, edges, distances)

            

            ids = group_df.id.values

            a0_idx  = group_df.atom_index_0.values

            a1_idx  = group_df.atom_index_1.values

            j_dists = dist_matrix[a0_idx, a1_idx]

            j_pairs = np.concatenate([np.expand_dims(ids,     axis=1),

                                      np.expand_dims(a0_idx,  axis=1),

                                      np.expand_dims(a1_idx,  axis=1),

                                      np.expand_dims(j_dists, axis=1)],

                                     axis=1)

            

            if train:

                sc_contributions = ['fc', 'sd', 'pso', 'dso']  # sum up to sc-constant

                y = group_df[sc_contributions].values

                self.molecule_list.append( [graph_input, (j_pairs, y)] )

            else:

                self.molecule_list.append( [graph_input, (j_pairs, )] )



            if i == limit:

                break

                

        self.num_molecules = len(self.molecule_list)

        self.batch_sizes   = len(set(zip(self.atom_counts, self.j_coupling_counts)))

        print(f'initialized dataset with {self.num_molecules} and {self.num_j_couplings} j-couplings.')

        

    @staticmethod

    def get_graph(atom_types, edges, distances) -> dgl.DGLGraph:

        """

        Create graph on the fly.

        Delete it after passing through the net.

        Required to prevent memory leak. Somehow DGLGraph does not release cuda-memory...

        """

        g = dgl.DGLGraph()

        g.add_nodes(len(atom_types))

        g.ndata['node_type'] = torch.LongTensor(atom_types)

        

        g.add_edges(edges[0].tolist(), edges[1].tolist())

        g.edata['distance'] = distances.view(-1, 1)

        

        return g

    

    def __len__(self):

        return len(self.molecule_list)

    

    def __getitem__(self, i):

        graph_input, target_infos = self.molecule_list[i]

        graph = self.get_graph(*graph_input)

        return graph, target_infos





ds_train = Molecule_Dataset(structures_df, train)

ds_valid = Molecule_Dataset(structures_df, valid)
net = Atominator()

net.train()

net.to(DEVICE)



optimizer = torch.optim.Adam(net.parameters(), lr=0.005, weight_decay=0)  #1e-6)

scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.97)



loss_function = nn.L1Loss()
%%time



num_epochs=10

print('epoch\ttrain\tvalidation\tlearning-rate')



for epoch in range(num_epochs):

    

    train_loss   = []

    running_loss = []

    

    random_indices = np.random.choice(range(len(ds_train)), size=len(ds_train), replace=False)

    for i in range(len(ds_train)):



        random_i = random_indices[i]

        graph, (j_pairs, y) = ds_train[random_i]



        net.train()

        optimizer.zero_grad()



        graph.ndata['node_type'] = graph.ndata['node_type'].to(DEVICE)

        graph.edata['distance']  = graph.edata['distance'].to(DEVICE)

        

        y_hat = net(graph, j_pairs=j_pairs)



        y_truth = torch.tensor(y).float().to(DEVICE)

        loss = loss_function(y_hat, target=y_truth)

        loss.backward()

        optimizer.step()

        

        # multiply times 4 to obtain error of sum of the 4 sc-contributions:

        train_loss.append(np.log(loss.item() * 4))

        running_loss.append(np.log(loss.item() * 4))

        

        # free GPU-memory:

        del y_truth, y_hat

        del graph



        #if i and i % 2500 == 0:

        #    print(f'{i}\t{np.mean(running_loss):.2f}')

        #    running_loss = []



    validation_loss = []

    for  graph, (j_pairs, y) in ds_valid:



        graph.ndata['node_type'] = graph.ndata['node_type'].to(DEVICE)

        graph.edata['distance']  = graph.edata['distance'].to(DEVICE)

        

        net.eval()

        y_hat = net(graph, j_pairs=j_pairs)

        

        # sum up sc-contributions to obtain the sc-constant:

        sc_truth = torch.tensor(y).float().sum(dim=1).to(DEVICE)

        sc_pred  = y_hat.sum(dim=1)

        

        loss = loss_function(sc_pred, target=sc_truth)

        validation_loss.append(np.log(loss.item()))

        

        # free GPU-memory:

        del sc_truth, y_hat

        del graph



    scheduler.step()

    current_lr = optimizer.param_groups[0]['lr']

    print(f'{epoch}:\t{np.mean(train_loss):.4f}\t{np.mean(validation_loss):.4f}\t{current_lr}')
%%time



ds_test = Molecule_Dataset(structures_df, test, train=False)



predictions = []

id2prediction = {}



for  graph, (j_pairs, ) in ds_test:



    graph.ndata['node_type'] = graph.ndata['node_type'].to(DEVICE)

    graph.edata['distance']  = graph.edata['distance'].to(DEVICE)

        

    net.eval()

    y_hat = net(graph, j_pairs=j_pairs)

        

    # sum up sc-contributions to obtain the sc-constant:

    sc_pred  = y_hat.sum(dim=1).detach().cpu().numpy().tolist()

    predictions.extend(sc_pred)

        

    # free GPU-memory:

    del graph

    del y_hat
assert len(predictions) == len(test)

test['scalar_coupling_constant'] = predictions



test[['id', 'scalar_coupling_constant']].to_csv(f'submission_{J_TYPE}.csv', index=False)