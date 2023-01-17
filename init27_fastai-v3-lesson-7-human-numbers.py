from fastai.text import *
bs=64
path = untar_data(URLs.HUMAN_NUMBERS)

path.ls()
def readnums(d): return [', '.join(o.strip() for o in open(path/d).readlines())]
train_txt = readnums('train.txt'); train_txt[0][:80]
valid_txt = readnums('valid.txt'); valid_txt[0][-80:]
train = TextList(train_txt, path=path)

valid = TextList(valid_txt, path=path)



src = ItemLists(path=path, train=train, valid=valid).label_for_lm()

data = src.databunch(bs=bs)
train[0].text[:80]
len(data.valid_ds[0][0].data)
data.bptt, len(data.valid_dl)
13017/70/bs
it = iter(data.valid_dl)

x1,y1 = next(it)

x2,y2 = next(it)

x3,y3 = next(it)

it.close()
x1.numel()+x2.numel()+x3.numel()
x1.shape,y1.shape
x2.shape,y2.shape
x1[:,0]
y1[:,0]
v = data.valid_ds.vocab
v.textify(x1[0])
v.textify(y1[0])
v.textify(x2[0])
v.textify(x3[0])
v.textify(x1[1])
v.textify(x2[1])
v.textify(x3[1])
v.textify(x3[-1])
data.show_batch(ds_type=DatasetType.Valid)
data = src.databunch(bs=bs, bptt=3)
x,y = data.one_batch()

x.shape,y.shape
nv = len(v.itos); nv
nh=64
def loss4(input,target): return F.cross_entropy(input, target[:,-1])

def acc4 (input,target): return accuracy(input, target[:,-1])
class Model0(nn.Module):

    def __init__(self):

        super().__init__()

        self.i_h = nn.Embedding(nv,nh)  # green arrow

        self.h_h = nn.Linear(nh,nh)     # brown arrow

        self.h_o = nn.Linear(nh,nv)     # blue arrow

        self.bn = nn.BatchNorm1d(nh)

        

    def forward(self, x):

        h = self.bn(F.relu(self.i_h(x[:,0])))

        if x.shape[1]>1:

            h = h + self.i_h(x[:,1])

            h = self.bn(F.relu(self.h_h(h)))

        if x.shape[1]>2:

            h = h + self.i_h(x[:,2])

            h = self.bn(F.relu(self.h_h(h)))

        return self.h_o(h)
learn = Learner(data, Model0(), loss_func=loss4, metrics=acc4)
learn.fit_one_cycle(6, 1e-4)
class Model1(nn.Module):

    def __init__(self):

        super().__init__()

        self.i_h = nn.Embedding(nv,nh)  # green arrow

        self.h_h = nn.Linear(nh,nh)     # brown arrow

        self.h_o = nn.Linear(nh,nv)     # blue arrow

        self.bn = nn.BatchNorm1d(nh)

        

    def forward(self, x):

        h = torch.zeros(x.shape[0], nh).to(device=x.device)

        for i in range(x.shape[1]):

            h = h + self.i_h(x[:,i])

            h = self.bn(F.relu(self.h_h(h)))

        return self.h_o(h)
learn = Learner(data, Model1(), loss_func=loss4, metrics=acc4)
learn.fit_one_cycle(6, 1e-4)
data = src.databunch(bs=bs, bptt=20)
x,y = data.one_batch()

x.shape,y.shape
class Model2(nn.Module):

    def __init__(self):

        super().__init__()

        self.i_h = nn.Embedding(nv,nh)

        self.h_h = nn.Linear(nh,nh)

        self.h_o = nn.Linear(nh,nv)

        self.bn = nn.BatchNorm1d(nh)

        

    def forward(self, x):

        h = torch.zeros(x.shape[0], nh).to(device=x.device)

        res = []

        for i in range(x.shape[1]):

            h = h + self.i_h(x[:,i])

            h = F.relu(self.h_h(h))

            res.append(self.h_o(self.bn(h)))

        return torch.stack(res, dim=1)
learn = Learner(data, Model2(), metrics=accuracy)
learn.fit_one_cycle(10, 1e-4, pct_start=0.1)
class Model3(nn.Module):

    def __init__(self):

        super().__init__()

        self.i_h = nn.Embedding(nv,nh)

        self.h_h = nn.Linear(nh,nh)

        self.h_o = nn.Linear(nh,nv)

        self.bn = nn.BatchNorm1d(nh)

        self.h = torch.zeros(bs, nh).cuda()

        

    def forward(self, x):

        res = []

        h = self.h

        for i in range(x.shape[1]):

            h = h + self.i_h(x[:,i])

            h = F.relu(self.h_h(h))

            res.append(self.bn(h))

        self.h = h.detach()

        res = torch.stack(res, dim=1)

        res = self.h_o(res)

        return res
learn = Learner(data, Model3(), metrics=accuracy)
learn.fit_one_cycle(20, 3e-3)
class Model4(nn.Module):

    def __init__(self):

        super().__init__()

        self.i_h = nn.Embedding(nv,nh)

        self.rnn = nn.RNN(nh,nh, batch_first=True)

        self.h_o = nn.Linear(nh,nv)

        self.bn = BatchNorm1dFlat(nh)

        self.h = torch.zeros(1, bs, nh).cuda()

        

    def forward(self, x):

        res,h = self.rnn(self.i_h(x), self.h)

        self.h = h.detach()

        return self.h_o(self.bn(res))
learn = Learner(data, Model4(), metrics=accuracy)
learn.fit_one_cycle(20, 3e-3)
class Model5(nn.Module):

    def __init__(self):

        super().__init__()

        self.i_h = nn.Embedding(nv,nh)

        self.rnn = nn.GRU(nh, nh, 2, batch_first=True)

        self.h_o = nn.Linear(nh,nv)

        self.bn = BatchNorm1dFlat(nh)

        self.h = torch.zeros(2, bs, nh).cuda()

        

    def forward(self, x):

        res,h = self.rnn(self.i_h(x), self.h)

        self.h = h.detach()

        return self.h_o(self.bn(res))
learn = Learner(data, Model5(), metrics=accuracy)
learn.fit_one_cycle(10, 1e-2)