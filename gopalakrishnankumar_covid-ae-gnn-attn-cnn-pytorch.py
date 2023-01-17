!pip -q install git+https://github.com/ildoonet/pytorch-gradual-warmup-lr.git >> quit
import os, random, glob, gc, json, time

import numpy as np, pandas as pd



import torch, torch.nn as nn

import torch.nn.functional as F



from torch.utils.data import DataLoader, TensorDataset

from tqdm import tqdm

from sklearn.model_selection import KFold, StratifiedKFold


train = pd.read_json("/kaggle/input/stanford-covid-vaccine/train.json",lines=True)

train = train[train.signal_to_noise > 1].reset_index(drop = True)



test  = pd.read_json("/kaggle/input/stanford-covid-vaccine/test.json",lines=True)

test_pub = test[test["seq_length"] == 107]

test_pri = test[test["seq_length"] == 130]

sub = pd.read_csv("/kaggle/input/stanford-covid-vaccine/sample_submission.csv")





As = []

for id in tqdm(train["id"]):

    a = np.load(f"/kaggle/input/stanford-covid-vaccine/bpps/{id}.npy")

    As.append(a)

As = np.array(As)

As_pub = []

for id in tqdm(test_pub["id"]):

    a = np.load(f"/kaggle/input/stanford-covid-vaccine/bpps/{id}.npy")

    As_pub.append(a)

As_pub = np.array(As_pub)

As_pri = []

for id in tqdm(test_pri["id"]):

    a = np.load(f"/kaggle/input/stanford-covid-vaccine/bpps/{id}.npy")

    As_pri.append(a)

As_pri = np.array(As_pri)
targets = list(sub.columns[1:])

print(targets)



y_train = []

seq_len = train["seq_length"].iloc[0]

seq_len_target = train["seq_scored"].iloc[0]

ignore = -10000

ignore_length = seq_len - seq_len_target

for target in targets:

    y = np.vstack(train[target])

    dummy = np.zeros([y.shape[0], ignore_length]) + ignore

    y = np.hstack([y, dummy])

    y_train.append(y)

y = np.stack(y_train, axis = 2)

y.shape
def get_structure_adj(train):

    ## get adjacent matrix from structure sequence

    

    ## here I calculate adjacent matrix of each base pair, 

    ## but eventually ignore difference of base pair and integrate into one matrix

    Ss = []

    for i in tqdm(range(len(train))):

        seq_length = train["seq_length"].iloc[i]

        structure = train["structure"].iloc[i]

        sequence = train["sequence"].iloc[i]



        cue = []

        a_structures = {

            ("A", "U") : np.zeros([seq_length, seq_length]),

            ("C", "G") : np.zeros([seq_length, seq_length]),

            ("U", "G") : np.zeros([seq_length, seq_length]),

            ("U", "A") : np.zeros([seq_length, seq_length]),

            ("G", "C") : np.zeros([seq_length, seq_length]),

            ("G", "U") : np.zeros([seq_length, seq_length]),

        }

        a_structure = np.zeros([seq_length, seq_length])

        for i in range(seq_length):

            if structure[i] == "(":

                cue.append(i)

            elif structure[i] == ")":

                start = cue.pop()

#                 a_structure[start, i] = 1

#                 a_structure[i, start] = 1

                a_structures[(sequence[start], sequence[i])][start, i] = 1

                a_structures[(sequence[i], sequence[start])][i, start] = 1

        

        a_strc = np.stack([a for a in a_structures.values()], axis = 2)

        a_strc = np.sum(a_strc, axis = 2, keepdims = True)

        Ss.append(a_strc)

    

    Ss = np.array(Ss)

    print(Ss.shape)

    return Ss

Ss = get_structure_adj(train)

Ss_pub = get_structure_adj(test_pub)

Ss_pri = get_structure_adj(test_pri)
def get_distance_matrix(As):

    ## adjacent matrix based on distance on the sequence

    ## D[i, j] = 1 / (abs(i - j) + 1) ** pow, pow = 1, 2, 4

    

    idx = np.arange(As.shape[1])

    Ds = []

    for i in range(len(idx)):

        d = np.abs(idx[i] - idx)

        Ds.append(d)



    Ds = np.array(Ds) + 1

    Ds = 1/Ds

    Ds = Ds[None, :,:]

    Ds = np.repeat(Ds, len(As), axis = 0)

    

    Dss = []

    for i in [1, 2, 4]: 

        Dss.append(Ds ** i)

    Ds = np.stack(Dss, axis = 3)

    print(Ds.shape)

    return Ds



Ds = get_distance_matrix(As)

Ds_pub = get_distance_matrix(As_pub)

Ds_pri = get_distance_matrix(As_pri)
## concat adjecent

As = np.concatenate([As[:,:,:,None], Ss, Ds], axis = 3).astype(np.float32)

As_pub = np.concatenate([As_pub[:,:,:,None], Ss_pub, Ds_pub], axis = 3).astype(np.float32)

As_pri = np.concatenate([As_pri[:,:,:,None], Ss_pri, Ds_pri], axis = 3).astype(np.float32)

del Ss, Ds, Ss_pub, Ds_pub, Ss_pri, Ds_pri

As.shape, As_pub.shape, As_pri.shape
## sequence

def return_ohe(n, i):

    tmp = [0] * n

    tmp[i] = 1

    return tmp



def get_input(train):

    ## get node features, which is one hot encoded

    mapping = {}

    vocab = ["A", "G", "C", "U"]

    for i, s in enumerate(vocab):

        mapping[s] = return_ohe(len(vocab), i)

    X_node = np.stack(train["sequence"].apply(lambda x : list(map(lambda y : mapping[y], list(x)))))



    mapping = {}

    vocab = ["S", "M", "I", "B", "H", "E", "X"]

    for i, s in enumerate(vocab):

        mapping[s] = return_ohe(len(vocab), i)

    X_loop = np.stack(train["predicted_loop_type"].apply(lambda x : list(map(lambda y : mapping[y], list(x)))))

    

    mapping = {}

    vocab = [".", "(", ")"]

    for i, s in enumerate(vocab):

        mapping[s] = return_ohe(len(vocab), i)

    X_structure = np.stack(train["structure"].apply(lambda x : list(map(lambda y : mapping[y], list(x)))))

    

    

    X_node = np.concatenate([X_node, X_loop], axis = 2)

    

    ## interaction

    a = np.sum(X_node * (2 ** np.arange(X_node.shape[2])[None, None, :]), axis = 2)

    vocab = sorted(set(a.flatten()))

    print(vocab)

    ohes = []

    for v in vocab:

        ohes.append(a == v)

    ohes = np.stack(ohes, axis = 2)

    X_node = np.concatenate([X_node, ohes], axis = 2).astype(np.float32)

    

    

    print(X_node.shape)

    return X_node



X_node = get_input(train)

X_node_pub = get_input(test_pub)

X_node_pri = get_input(test_pri)
class Attention(nn.Module):

    def __init__(self, in_channels, n_factor):

        super().__init__()



        self.n_factor = n_factor



        self.x_Q = nn.Conv1d(in_channels=in_channels, out_channels=self.n_factor, kernel_size=1, padding=1//2)

        self.x_K = nn.Conv1d(in_channels=in_channels, out_channels=self.n_factor, kernel_size=1, padding=1//2)

        self.x_V = nn.Conv1d(in_channels=in_channels, out_channels=self.n_factor, kernel_size=1, padding=1//2)





    def forward(self, x_inner, x_outer):



        x_Q = self.x_Q(x_inner)  # (N, value_len, heads, head_dim)

        x_K = self.x_K(x_outer)  # (N, key_len, heads, head_dim)

        x_V = self.x_V(x_outer)  # (N, query_len, heads, heads_dim)



        # Einsum does matrix mult. for query*keys for each training example

        # with every other training example, don't be confused by einsum

        # it's just how I like doing matrix multiplication & bmm



        x_Q = x_Q.permute(0, 2, 1)

        x_K = x_K.permute(0, 2, 1)

        x_V = x_V.permute(0, 2, 1)



        #print(x_Q.shape, x_K.shape)

        res = torch.einsum("nqd,nkd->nqk", [x_Q, x_K])

        #print(res.shape)



        # Normalize energy values similarly to seq2seq + attention

        # so that they sum to 1. Also divide by scaling factor for

        # better stability

        attention = torch.softmax(res / (self.n_factor ** (1 / 2)), dim=2)



        #print(attention.shape)



        attention = torch.einsum("nql,nld->nqd", [attention, x_V])

        #.reshape(

        #    N, query_len, self.n_factor

        #)

        

        return attention





class MultiHeadAttention(nn.Module):

    def __init__(self, n_factor, n_head, dropout):

        super().__init__()

        #self.attention = Attention(in_channels, n_factor)

        self.norm1 = nn.LayerNorm(n_factor)



        self.n_factor_head = n_factor // n_head

        self.heads = nn.ModuleList([

            Attention(n_factor, self.n_factor_head) for _ in range(n_head)

        ])



        self.dropout = nn.Dropout(dropout)



    def forward(self, x, y):

        

        att_out = []

        for head in self.heads:

            out = head(x, y)

            att_out.append(out)



        att = torch.cat(att_out, dim=2)

        x = x.permute(0, 2, 1)

        #print(x.shape, att.shape)

        x = x + att

        x = self.norm1(x)

        x = self.dropout(x)



        return x



class  AdjAttn(nn.Module):

    def __init__(self, in_unit, out_unit, n=2, rate=0.1):

        super().__init__()

        self.n = n

        

        self.f1 = Forward(in_unit=in_unit, out_unit=out_unit, kernel=3, rate=rate)

        self.f2 = Forward(in_unit=out_unit, out_unit=out_unit, kernel=3, rate=rate)

        self.f3 = Forward(in_unit=out_unit * self.n, out_unit=out_unit, kernel=3, rate=rate)



    def forward(self, x, adj):

        x_a = x

        x_as = []

        for i in range(self.n):

            if i == 0:

                x_a = self.f1(x_a)

            else:

                x_a = self.f2(x_a)

            x_a = x_a.permute(0, 2, 1)

            #print(x_a.shape, adj.shape)

            x_a = torch.matmul(adj, x_a) ## aggregate neighborhods

            #print(x_a.shape)

            x_a = x_a.permute(0, 2, 1)

            #print(x_a.shape)

            x_as.append(x_a)

        

        if self.n == 1:

            x_a = x_as[0]

        else:

            x_a = torch.cat(x_as, dim=1)

        #print(x_a.shape)

        x_a = self.f3(x_a)

        return x_a



class Res(nn.Module):

    def __init__(self, in_unit, out_unit, kernel, rate=0.1):

        super().__init__()



        self.cnn = nn.Conv1d(in_channels=in_unit, out_channels=out_unit, kernel_size=kernel, stride=1, padding=kernel//2)

        self.norm = nn.LayerNorm(out_unit)

        self.dropout = nn.Dropout(p=rate)



    def forward(self, x):

        h = self.cnn(x)

        h = h.permute(0, 2, 1)

        h = self.norm(h)

        h = h.permute(0, 2, 1)

        h = F.leaky_relu(h)

        h = self.dropout(h)

        return x + h



class Forward(nn.Module):

    def __init__(self, in_unit, out_unit, kernel, rate=0.1):

        super().__init__()



        self.cnn = nn.Conv1d(in_channels=in_unit, out_channels=out_unit, kernel_size=kernel, padding=kernel//2)

        self.norm = nn.LayerNorm(out_unit)

        self.dropout = nn.Dropout(p=rate)



        self.res = Res(out_unit, out_unit, kernel, rate)



    def forward(self, x):

        x = self.cnn(x)

        x = x.permute(0, 2, 1)

        x = self.norm(x)

        x = x.permute(0, 2, 1)

        x = self.dropout(x)

        x = F.leaky_relu(x)

        x = self.res(x)

        return x



class BaseModel(nn.Module):

    def __init__(self, in_dim, out_unit = 128, rate=0.0):

        super().__init__()



        self.adj_learned = nn.Linear(in_features=in_dim, out_features=1)

        

        self.f1 = Forward(in_unit=39, out_unit=128, kernel=3, rate=rate)

        self.f2 = Forward(in_unit=128, out_unit=64, kernel=7, rate=rate)

        self.f3 = Forward(in_unit=64, out_unit=32, kernel=17, rate=rate)

        self.f4 = Forward(in_unit=32, out_unit=16, kernel=31, rate=rate)



        self.f5_64 = Forward(in_unit=240, out_unit=64, kernel=31, rate=rate)

        self.f5_32 = Forward(in_unit=64, out_unit=32, kernel=31, rate=rate)

        

        self.f6_64 = Forward(in_unit=448, out_unit=64, kernel=3, rate=rate)

        self.f6_32 = Forward(in_unit=224, out_unit=32, kernel=3, rate=rate)



        self.adj_attn_64 = AdjAttn(in_unit=240, out_unit=64, n=2)

        self.adj_attn_32 = AdjAttn(in_unit=64, out_unit=32, n=2)

        

        self.multi_head_attention_64 = MultiHeadAttention(n_factor=64, n_head=4, dropout=0.0)

        self.multi_head_attention_32 = MultiHeadAttention(n_factor=32, n_head=4, dropout=0.0)

    

    def forward(self, node, adj):



        node = node.permute(0, 2, 1)

        #print(node.shape, adj.shape)

        # node -> [BS, 107, 39] -> channel first -> [BS, 39, 107]

        # adj -> [BS, 107, 107, 5]



        adj_learned = F.relu(self.adj_learned(adj)) 

        adj_all = torch.cat([adj, adj_learned], dim=3)

        # adj_learned -> [BS, 107, 107, 1]

        # adj_all -> [BS, 107, 107, 6]

        #print(adj_learned.shape, adj_all.shape)



        xs = []

        xs.append(node)

        

        x1 = self.f1(node)

        x2 = self.f2(x1)

        x3 = self.f3(x2)

        x4 = self.f4(x3)

        #print(x1.shape, x2.shape, x3.shape, x4.shape)

        x = torch.cat([x1, x2, x3, x4], dim=1)

        # x -> [BS, 240, 107]

        #print(x.shape)



        for unit in [64, 32]:

            x_as = []

            for i in range(adj_all.shape[3]):

                if unit == 64:

                    #print("64 ----> ", x.shape)

                    x_a = self.adj_attn_64(x, adj_all[:, :, :, i])

                    x_as.append(x_a)

                    x_c = self.f5_64(x)

                if unit == 32:

                    #print("32 ----> " ,x.shape)

                    x_a = self.adj_attn_32(x, adj_all[:, :, :, i])

                    #print(x_a.shape)

                    x_as.append(x_a)

                    x_c = self.f5_32(x)

                    

            if unit == 64:

                x = torch.cat(x_as + [x_c], dim=1)

                x = self.f6_64(x)

                x = self.multi_head_attention_64(x, x)

                x = x.permute(0,2,1)

                xs.append(x)

            if unit == 32:

                #print("32 ----> " ,x.shape)

                x = torch.cat(x_as + [x_c], dim=1)

                #print(x.shape)

                x = self.f6_32(x)

                x = self.multi_head_attention_32(x, x)

                x = x.permute(0,2,1)

                xs.append(x)

        

        x = torch.cat(xs, dim=1)



        return x



class AutoEncoder(nn.Module):

    """

    Denoising auto encoder part

    Node, Adj --> Middle features --> None

    """

    def __init__(self, in_dim, rate=0.0):

        super().__init__()



        self.base = BaseModel(in_dim=in_dim)

        self.f1 = Forward(in_unit=135, out_unit=16, kernel=31, rate=rate)



        self.fc1 = nn.Linear(in_features=16, out_features=39)



    def forward(self, node, adj):

        x = self.base(node, adj)

        #print(x.shape)

        x = self.f1(x)

        x = x.permute(0, 2, 1)

        #print(x.shape)

        p = torch.sigmoid(self.fc1(x))

        

        #loss = torch.mean(20 * node * torch.log(p + 1e-4) + (1 - node) * torch.log(1 - p + 1e-4))



        return p



class FinalModel(nn.Module):

    """

    Regression part

    Node, Adj --> Middle Feature --> Prediction of targets

    """

    def __init__(self, in_dim, rate=0.0):

        super().__init__()



        self.base = BaseModel(in_dim=in_dim)

        self.f2 = Forward(in_unit=135, out_unit=128, kernel=31, rate=rate)



        self.fc2 = nn.Linear(in_features=128, out_features=5)



    

    def forward(self, node, adj):

        x = self.base(node, adj)

        #print(x.shape)

        x = self.f2(x)

        x = x.permute(0, 2, 1)

        #print(x.shape)

        x = self.fc2(x)

        return x
def MCRMSE(y_true, y_pred):

    y_true = y_true[:, :68, :]

    y_pred = y_pred[:, :68, :]

    colwise_mse = torch.mean(torch.square(y_true - y_pred), dim=1)

    return torch.mean(torch.sqrt(colwise_mse), dim=1).mean()



class MCRMSELoss(nn.Module):

    def __init__(self):

        super().__init__()

        self.mse = nn.MSELoss()

    

    def rmse(self, y_actual, y_pred):

        mse = self.mse(y_actual, y_pred)

        return torch.sqrt(mse)

    

    def forward(self, y_actual, y_pred, num_scored=None):

        if num_scored == None:

            num_scored = y_actual.shape[-1]

        score = 0

        for i in range(num_scored):

            score += self.rmse(y_actual[:, :68, i], y_pred[:, :68, i]) / num_scored

        return score



class AverageMeter(object):

    """Computes and stores the average and current value"""



    def __init__(self):

        self.reset()



    def reset(self):

        self.val = 0

        self.avg = 0

        self.sum = 0

        self.count = 0



    def update(self, val, n=1):

        self.val = val

        self.sum += val * n

        self.count += n

        self.avg = self.sum / self.count
def train_epcoh(model, loader, optimizer, criterion, device, epoch):

    losses = AverageMeter()



    model.train()

    t = tqdm(loader)

    for i, (node, adj, y) in enumerate(t):



        #print(d)



        node = node.to(device)

        adj = adj.to(device)

        y = y.to(device)



        pred_y = model(node, adj)

        

        #print(y.shape, pred_y.shape)



        loss = criterion(y, pred_y)



        optimizer.zero_grad()

        loss.backward()

        optimizer.step()



        bs = y.size(0)

        losses.update(loss.item(), bs)



        t.set_description(f"Train E:{epoch} - Loss:{losses.avg:0.5f}")

    

    t.close()

    return losses.avg



def valid_epoch(model, loader, criterion, device, epoch):

    losses = AverageMeter()



    model.eval()



    with torch.no_grad():

        t = tqdm(loader)

        for i, (node, adj, y) in enumerate(t):



            #print(d)



            node = node.to(device)

            adj = adj.to(device)

            y = y.to(device)



            pred_y = model(node, adj)

            

            #print(y.shape, pred_y.shape)

            

            loss = criterion(y, pred_y)



            bs = y.size(0)

            losses.update(loss.item(), bs)



            t.set_description(f"Valid E:{epoch} - Loss:{losses.avg:0.5f}")

        

    t.close()

    return losses.avg



def test_predic(model, loader, device):

    

    predicts = []

    

    model.eval()

    

    with torch.no_grad():

        t = tqdm(loader)

        for i, (node, adj) in enumerate(t):

            

            node = node.to(device)

            adj = adj.to(device)

            

            outs = model(node, adj).cpu().detach().numpy().tolist()

            

            predicts.extend(outs)

    

    return predicts
from torch.optim.lr_scheduler import CosineAnnealingLR

# Fix Warmup Bug

from warmup_scheduler import GradualWarmupScheduler  # https://github.com/ildoonet/pytorch-gradual-warmup-lr





class GradualWarmupSchedulerV2(GradualWarmupScheduler):

    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):

        super(GradualWarmupSchedulerV2, self).__init__(optimizer, multiplier, total_epoch, after_scheduler)

    def get_lr(self):

        if self.last_epoch > self.total_epoch:

            if self.after_scheduler:

                if not self.finished:

                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]

                    self.finished = True

                return self.after_scheduler.get_lr()

            return [base_lr * self.multiplier for base_lr in self.base_lrs]

        if self.multiplier == 1.0:

            return [base_lr * (float(self.last_epoch) / self.total_epoch) for base_lr in self.base_lrs]

        else:

            return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')





AE_model = AutoEncoder(in_dim=5)

AE_model = AE_model.to(device)

optimizer = torch.optim.Adam(AE_model.parameters(), lr=0.001)

criterion = nn.MSELoss()



train_dataset = TensorDataset(torch.tensor(X_node, dtype=torch.float), torch.tensor(As,  dtype=torch.float), torch.tensor(X_node,  dtype=torch.float))

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)



public_dataset = TensorDataset(torch.tensor(X_node_pub, dtype=torch.float), torch.tensor(As_pub,  dtype=torch.float), torch.tensor(X_node_pub,  dtype=torch.float))

public_loader = DataLoader(public_dataset, batch_size=64, shuffle=True)



private_dataset = TensorDataset(torch.tensor(X_node_pri, dtype=torch.float), torch.tensor(As_pri,  dtype=torch.float), torch.tensor(X_node_pri,  dtype=torch.float))

private_loader = DataLoader(private_dataset, batch_size=64, shuffle=True)



for epoch in range(10):

    print("####### Epoch : ", epoch)

    print("####### Train")

    train_loss = train_epcoh(AE_model, train_loader, optimizer, criterion, device, epoch)

    print("####### Public")

    public_loss = train_epcoh(AE_model, public_loader, optimizer, criterion, device, epoch)

    print("####### Private")

    private_loss = train_epcoh(AE_model, private_loader, optimizer, criterion, device, epoch)



torch.save(AE_model.state_dict(), "AE_model.bin")
class args:



    exp_name = "demo"

    output_dir = ""

    sub_name = "demo"



    # Training parameters

    lr = 0.001

    seed = 42

    epochs = 100

    n_folds = 10

    batch_size = 64
args.save_path = os.path.join(args.output_dir, args.exp_name)

os.makedirs(args.save_path, exist_ok=True)



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



public_predictions = [] 

public_ids         = []



private_predictions = []

private_ids         = []



public_dataset = TensorDataset(torch.tensor(X_node_pub, dtype=torch.float), torch.tensor(As_pub,  dtype=torch.float))

public_loader = DataLoader(public_dataset, batch_size=64, shuffle=False, drop_last=False)



private_dataset = TensorDataset(torch.tensor(X_node_pri, dtype=torch.float), torch.tensor(As_pri,  dtype=torch.float))

private_loader = DataLoader(private_dataset, batch_size=64, shuffle=False, drop_last=False)



skf = KFold(args.n_folds, shuffle=True, random_state=42)



for i, (train_index, valid_index) in enumerate(skf.split(X_node, As)):

    print("#"*20)

    print(f"##### Fold : {i}")

    

    args.fold = i



    X_node_tr = X_node[train_index]

    X_node_va = X_node[valid_index]

    As_tr = As[train_index]

    As_va = As[valid_index]

    y_tr = y[train_index]

    y_va = y[valid_index]

    

    train_dataset = TensorDataset(torch.tensor(X_node_tr, dtype=torch.float), torch.tensor(As_tr,  dtype=torch.float), torch.tensor(y_tr,  dtype=torch.float))

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    

    valid_dataset = TensorDataset(torch.tensor(X_node_va, dtype=torch.float), torch.tensor(As_va,  dtype=torch.float), torch.tensor(y_va,  dtype=torch.float))

    valid_loader = DataLoader(valid_dataset, batch_size=64, shuffle=False)

    

    model = FinalModel(in_dim=5)

    model.load_state_dict(torch.load("AE_model.bin"), strict=False)

    model = model.to(device)

    

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    criterion = MCRMSE #MCRMSELoss()

    

    scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, args.epochs)

    scheduler_warmup = GradualWarmupSchedulerV2(optimizer, multiplier=10, total_epoch=1, after_scheduler=scheduler_cosine)

    

    best_loss = 99999



    for epoch in range(args.epochs):



        train_loss = train_epcoh(model, train_loader, optimizer, criterion, device, epoch)

        valid_loss = valid_epoch(model, valid_loader, criterion, device, epoch)

        

        scheduler_warmup.step()    

        if epoch==2: scheduler_warmup.step() # bug workaround 

        

        content = f"""

            {time.ctime()} \n

            Fold:{args.fold}, Epoch:{epoch}, lr:{optimizer.param_groups[0]['lr']:.7}, \n

            Train Loss:{train_loss:0.4f} - Valid Loss:{valid_loss:0.4f} \n

        """

        print(content)



        with open(f'{args.save_path}/log_{args.exp_name}.txt', 'a') as appender:

            appender.write(content + '\n')



        if valid_loss < best_loss:

            print(f"######### >>>>>>> Model Improved from {best_loss} -----> {valid_loss}")

            torch.save(model.state_dict(), os.path.join(args.save_path, f"fold-{args.fold}.bin"))

            best_loss = valid_loss



        torch.save(model.state_dict(), os.path.join(args.save_path, f"last-fold-{args.fold}.bin")) 

    

    public_model = FinalModel(in_dim=5).to(device)

    public_model.load_state_dict(torch.load(os.path.join(args.save_path, f"fold-{args.fold}.bin")))

    

    public_pred = test_predic(public_model, public_loader, device)

    private_pred = test_predic(public_model, private_loader, device)

    

    public_predictions.append(np.array(public_pred).reshape(629 * 107 , 5))

    private_predictions.append(np.array(private_pred).reshape(3005 * 130, 5))

    

    public_ids.append(test_pub.id.values)

    private_ids.append(test_pri.id.values)    
target_cols = ['reactivity', 'deg_Mg_pH10', 'deg_pH10', 'deg_Mg_50C', 'deg_50C']



public_ids1 = [f"{id}_{i}" for id in public_ids[0] for i in range(107)]

private_ids1 = [f"{id}_{i}" for id in private_ids[0] for i in range(130)]



public_preds = np.mean(public_predictions, axis=0)

private_preds = np.mean(private_predictions, axis=0)



public_pred_df = pd.DataFrame(public_preds, columns=target_cols)

public_pred_df["id_seqpos"] = public_ids1



private_pred_df = pd.DataFrame(private_preds, columns=target_cols)

private_pred_df["id_seqpos"] = private_ids1



pred_sub_df = public_pred_df.append(private_pred_df)



pred_sub_df.to_csv(os.path.join(args.save_path, f"{args.sub_name}_submission.csv"), index=False)
pred_sub_df.head()