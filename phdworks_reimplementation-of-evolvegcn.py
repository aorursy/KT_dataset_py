import math

import numpy as np

import pandas as pd

import torch

from torch.nn.parameter import Parameter

from torch.nn import functional as F

import torch.nn as nn

from torch.utils.data import Dataset

from sklearn.metrics import confusion_matrix
class bitcoin_dataset():

    def __init__(self,edges):



        num_nodes = edges[:,[0,1]].unique().size(0)

        timesteps = edges[:,3]

        aggr_time = 1200000

        timesteps = timesteps - timesteps.min()

        timesteps = timesteps // aggr_time

        self.max_time = timesteps.max()

        self.min_time = timesteps.min()

        edges[:,3] = timesteps   # 'TimeStep': 3



        ratings = edges[:,2] #'Weight': 2

        pos_indices = ratings > 0

        neg_indices = ratings <= 0

        ratings[pos_indices] = 1

        ratings[neg_indices] = -1  

        edges[:,2] = ratings





        #add the reversed link to make the graph undirected

        edges = torch.cat([edges,edges[:,[1, 0, 2, 3]]])



        #separate classes

        sp_indices = edges[:,[0,1,3]].t()

        sp_values = edges[:,2]



        neg_mask = sp_values == -1

        neg_sp_indices = sp_indices[:,neg_mask]

        neg_sp_values = sp_values[neg_mask]

        neg_sp_edges = torch.sparse.LongTensor(neg_sp_indices, neg_sp_values,

                                               torch.Size([num_nodes, num_nodes, self.max_time+1])).coalesce()

 

        pos_mask = sp_values == 1

        pos_sp_indices = sp_indices[:,pos_mask]

        pos_sp_values = sp_values[pos_mask]



        pos_sp_edges = torch.sparse.LongTensor(pos_sp_indices, pos_sp_values,

                                               torch.Size([num_nodes, num_nodes, self.max_time+1])).coalesce()



        #scale positive class to separate after adding

        pos_sp_edges *= 1000



        #we substract the neg_sp_edges to make the values positive

        sp_edges = (pos_sp_edges - neg_sp_edges).coalesce()

        #separating negs and positive edges per edge/timestamp

        vals = sp_edges._values()

        neg_vals = vals%1000

        pos_vals = vals//1000

        #We add the negative and positive scores and do majority voting

        vals = pos_vals - neg_vals

        #creating labels new_vals -> the label of the edges

        new_vals = torch.zeros(vals.size(0),dtype=torch.long)

        new_vals[vals>0] = 1

        new_vals[vals<=0] = 0

        indices_labels = torch.cat([sp_edges._indices().t(),new_vals.view(-1,1)],dim=1)

        vals = pos_vals + neg_vals

        

        self.edges = {'idx': indices_labels, 'vals': vals}

        self.num_nodes = num_nodes

        self.num_classes = 2
path = "../input/bitcoin-otc-trust-weighted-signed-network/soc-sign-bitcoinotc.csv"

with open(path) as f:

    lines = f.read().splitlines()

    

# 'FromNodeId': 0, 'ToNodeId': 1, 'Weight': 2,'TimeStep': 3

edges = [[float(r) for r in row.split(',')] for row in lines]

edges = torch.tensor(edges,dtype = torch.long)



#make_contigous_node_ids

new_edges = edges[:,[0,1]]

_, new_edges = new_edges.unique(return_inverse=True)

edges[:,[0,1]] = new_edges





dataset = bitcoin_dataset(edges)

class Cross_Entropy(torch.nn.Module):

    def __init__(self, class_weights ):

        super().__init__()

        self.weights = torch.tensor(class_weights).to('cuda')



    def forward(self,logits,labels):

        labels = labels.view(-1,1)

        alpha = self.weights[labels].view(-1,1)

        

        m,_ = torch.max(logits,dim=1)

        m = m.view(-1,1)

        sum_exp = torch.sum(torch.exp(logits-m),dim=1, keepdim=True)

        lossumexp_output = m + torch.log(sum_exp)

        

        loss = alpha * (- logits.gather(-1,labels) + lossumexp_output)

        return loss.mean()

    

comp_loss = Cross_Entropy(class_weights = [ 0.8, 0.2]).to('cuda')
class Classifier(torch.nn.Module):

    def __init__(self, out_features=2, in_features = None):

        super(Classifier,self).__init__()

        num_feats = in_features

        self.mlp = torch.nn.Sequential(torch.nn.Linear(in_features = num_feats,out_features = 100),

                                       torch.nn.ReLU(),

                                       torch.nn.Linear(in_features = 100,out_features = out_features))



    def forward(self,x):

        return self.mlp(x)
class GRCU(torch.nn.Module):

    def __init__(self,in_feats,out_feats ):

        super().__init__()

        

        self.W_update = Parameter(torch.Tensor(in_feats,in_feats))

        self.U_update = Parameter(torch.Tensor(in_feats,in_feats))

        self.b_update = Parameter(torch.zeros(in_feats,out_feats))   

        

        self.W_reset = Parameter(torch.Tensor(in_feats,in_feats))

        self.U_reset = Parameter(torch.Tensor(in_feats,in_feats))

        self.b_reset = Parameter(torch.zeros(in_feats,out_feats))   

        

        self.W_h_cap = Parameter(torch.Tensor(in_feats,in_feats))

        self.U_h_cap = Parameter(torch.Tensor(in_feats,in_feats))

        self.b_h_cap = Parameter(torch.zeros(in_feats,out_feats))   

        

        self.scorer = Parameter(torch.Tensor(in_feats,1))   # feats = rows 

        

        self.GCN_init_weights = Parameter(torch.Tensor(in_feats,out_feats))



        self.reset_param(self.W_update)

        self.reset_param(self.U_update)

        self.reset_param(self.W_reset)

        self.reset_param(self.U_reset)

        self.reset_param(self.W_h_cap)

        self.reset_param(self.U_h_cap)

        self.reset_param(self.scorer)

        self.reset_param(self.GCN_init_weights)



        self.k = out_feats   

    def reset_param(self,t):

        #Initialize based on the number of columns

        stdv = 1. / math.sqrt(t.size(1))

        t.data.uniform_(-stdv,stdv)

        

        

    def forward(self,A_list,node_embs_list,mask_list):

        GCN_weights = self.GCN_init_weights

        out_seq = []

        for t,Ahat in enumerate(A_list):

            node_embs = node_embs_list[t]

            x =node_embs.matmul(self.scorer) / self.scorer.norm()

            scores = node_embs.matmul(self.scorer) / self.scorer.norm() + mask_list[t]    # node_embs is prev_Z 



            vals, topk_indices = scores.view(-1).topk(self.k)



            topk_indices = topk_indices[vals > -float("Inf")]



            if topk_indices.size(0) < self.k:

                pad = torch.ones(self.k - topk_indices.size(0), dtype=torch.long, device = 'cuda') * topk_indices[-1]

                topk_indices = torch.cat([topk_indices,pad])







            if isinstance(node_embs, torch.sparse.FloatTensor) or isinstance(node_embs, torch.cuda.sparse.FloatTensor):

                node_embs = node_embs.to_dense()



            z_topk = node_embs[topk_indices] * torch.nn.Tanh()(scores[topk_indices].view(-1,1))

            z_topk = z_topk.t()



            update = torch.nn.Sigmoid()(self.W_update.matmul(z_topk) + self.U_update.matmul(GCN_weights) + self.b_update)

            reset = torch.nn.Sigmoid()(self.W_reset.matmul(z_topk) + self.U_reset.matmul(GCN_weights) + self.b_reset)



            h_cap = reset * GCN_weights   # GCN_weights is prev_Q

            h_cap = torch.nn.Tanh()(self.W_h_cap.matmul(z_topk) + self.U_h_cap.matmul(h_cap) + self.b_h_cap)



            GCN_weights = (1 - update) * GCN_weights + update * h_cap

            node_embs = torch.nn.RReLU()(Ahat.matmul(node_embs.matmul(GCN_weights)))

            out_seq.append(node_embs)

            

        return out_seq
class data_split(Dataset):

    def __init__(self, dataset, num_hist_steps, start, end):

        self.start = start

        self.end = end

        

        self.data = dataset

        #max_time for link pred should be one before

        self.max_time = dataset.max_time

        self.adj_mat_time_window = 1

        self.num_hist_steps = num_hist_steps



    def __len__(self):

        return self.end-self.start



    def __getitem__(self,sample_time):

        return self.get_sample(self.start + sample_time)

    

    def get_sample(self,sample_time):  #time

        hist_adj_list = []

        hist_ndFeats_list = []

        hist_mask_list = []

        for i in range(sample_time - self.num_hist_steps, sample_time+1):

            

            id3 = self.data.edges['idx']

            subset = id3[:,2] == i                          # 'time': 2

            print("subset",torch.sum(subset))

            id4 = self.data.edges['idx'][subset][:,[0, 1]]   #'source': 0,'target': 1,'

            val3 = self.data.edges['vals'][subset]

            out = torch.sparse.FloatTensor(id4.t(),val3).coalesce()



            id5 = out._indices().t()

            val4 = out._values()

            cur_adj = {'idx': id5, 'vals': val4}

            print("val4.shape",val4.shape)

#------------------------------------------------------------------

            

            node_mask = torch.zeros(self.data.num_nodes) - float("Inf")

            non_zero = cur_adj['idx'].unique()

            node_mask[non_zero] = 0

#----------------------------------------------------

            num_nodes = self.data.num_nodes

            

            new_vals = torch.ones(cur_adj['idx'].size(0))

            new_adj = {'idx':cur_adj['idx'], 'vals': new_vals}

            tensor_size = torch.Size([num_nodes, num_nodes])

            new_adj = torch.sparse.LongTensor(new_adj['idx'].t(), new_adj['vals'].type(torch.long), tensor_size)

            degs_out = new_adj.matmul(torch.ones(num_nodes,1,dtype = torch.long))            

            degs_out = {'idx': torch.cat([torch.arange(num_nodes).view(-1,1), degs_out.view(-1,1)],dim=1),'vals': torch.ones(num_nodes)}



#---------------------------------------------------------------------------------------------------------------------------------------------

            max_deg_out = []

            for t in range(dataset.min_time, dataset.max_time):

                

                subset = dataset.edges['idx'][:,2] <= t   #'time': 2

                subset = subset * (dataset.edges['idx'][:,2] > (t -  self.adj_mat_time_window))



                out = torch.sparse.FloatTensor(dataset.edges['idx'][subset][:,[0, 1]].t(),

                                               dataset.edges['vals'][subset]).coalesce()                  #'source': 0,'target': 1



                cur_adj1 =  {'idx': out._indices().t(),

                            'vals': torch.ones(out._indices().t().size(0),dtype=torch.long)}



                tensor_size = torch.Size([num_nodes,num_nodes])

                cur_adj1 = torch.sparse.LongTensor(cur_adj1['idx'].t(), cur_adj1['vals'].type(torch.long), tensor_size)

                max_deg_out.append(cur_adj1.matmul(torch.ones(num_nodes,1,dtype = torch.long)).max())



            max_deg = int(torch.stack(max_deg_out).max()) + 1



            tensor_size = torch.Size([num_nodes,max_deg])

            degs_out = torch.sparse.LongTensor(degs_out['idx'].t(), degs_out['vals'].type(torch.long), tensor_size)

            node_feats = {'idx': degs_out._indices().t(),'vals': degs_out._values()}

            



            hist_ndFeats_list.append(node_feats)

            hist_mask_list.append(node_mask)

            

#----------------------------------------------------------------------------------------------------------------------------------------------

            sp_tensor = torch.sparse.FloatTensor(cur_adj['idx'].t(),cur_adj['vals'].type(torch.float), torch.Size([num_nodes,num_nodes]))

            eye_idx = torch.arange(num_nodes)

            eye_idx = torch.stack([eye_idx,eye_idx],dim=1).t() 

            sparse_eye = torch.sparse.FloatTensor(eye_idx, torch.ones(num_nodes), torch.Size([num_nodes,num_nodes]))



            sp_tensor = sparse_eye + sp_tensor

            

            

            degree = torch.sparse.sum(sp_tensor,dim=1).to_dense()

            di = degree[sp_tensor._indices()[0]]

            dj = degree[sp_tensor._indices()[1]]

            new_val = sp_tensor._values() * ((di * dj) ** -0.5)  

            

            hist_adj_list.append( {'idx': sp_tensor._indices().t(), 'vals': new_val})

#--------------------------------------------------------------------------------------------------------------------------------------------------

        subset =  self.data.edges['idx'][:,2] == sample_time  #'time': 2

        node_indices = self.data.edges['idx'][subset][:,[0,1]].t()   #'source': 0,'target': 1

        true_labels = self.data.edges['idx'][subset][:,3] #'label':3 







        for j,adj in enumerate(hist_adj_list):

            tensor_size = torch.Size([dataset.num_nodes, dataset.num_nodes])



            hist_adj_list[j] = torch.sparse.FloatTensor(adj['idx'].t(), adj['vals'].type(torch.float), tensor_size).to('cuda')



            tensor_size = torch.Size([dataset.num_nodes,max_deg])

            hist_ndFeats_list[j] = torch.sparse.FloatTensor(hist_ndFeats_list[j]['idx'].t(), 

                                                            hist_ndFeats_list[j]['vals'].type(torch.float), tensor_size).to('cuda')



            hist_mask_list[j] = hist_mask_list[j].view(-1,1).to('cuda')



        return (t, hist_adj_list, hist_ndFeats_list, node_indices,true_labels.to('cuda'), hist_mask_list, max_deg)

    
classifier = Classifier(in_features = 100, out_features = 2).to('cuda')



grcu_input = GRCU(in_feats = 145, out_feats =50).to('cuda')

grcu_output = GRCU(in_feats = 50, out_feats =50 ).to('cuda')



grcu_input_opt = torch.optim.Adam(grcu_input.parameters(), lr = .001) 

grcu_output_opt = torch.optim.Adam(grcu_output.parameters(), lr = .001) 

classifier_opt = torch.optim.Adam( classifier.parameters(), lr = .001)





grcu_input_opt.zero_grad()

grcu_output_opt.zero_grad()

classifier_opt.zero_grad()

torch.set_grad_enabled(True)
train_splitter = data_split(dataset,num_hist_steps = 10, start= 10, end = 95)

t, hist_adj_list, hist_ndFeats_list, node_indices, true_labels, hist_mask_list, max_deg  = train_splitter[79]
# class weights 

torch.sum(true_labels).item()/len(true_labels)
for _ in range(100):

    Nodes_list = grcu_input(hist_adj_list,hist_ndFeats_list,hist_mask_list)

    Nodes_list = grcu_output(hist_adj_list,Nodes_list,hist_mask_list)

    nodes_embs = Nodes_list[-1]



    cls_input = torch.cat([nodes_embs[node_indices[0]],nodes_embs[node_indices[0]]],dim=1)





    predictions = classifier(cls_input)

    predicted_classes = predictions.argmax(dim=1)

    print("confusion matrix \n",confusion_matrix(predicted_classes.cpu().detach().numpy(),true_labels.cpu()))



    loss = comp_loss(predictions,true_labels)

    print(loss)

    print('---------------------------------------')



    loss.backward()

    grcu_input_opt.step()

    grcu_output_opt.step()

    classifier_opt.step()



    grcu_input_opt.zero_grad()

    grcu_output_opt.zero_grad()

    classifier_opt.zero_grad()