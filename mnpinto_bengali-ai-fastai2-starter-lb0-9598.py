!pip install git+https://github.com/fastai/fastai2 

!pip install git+https://github.com/fastai/fastcore
import fastai2

from fastai2.vision.all import *

from sklearn.metrics import recall_score

print(fastai2.__version__)
# Configs

sz = 128

bs = 128

nfolds = 5

fold = 0

train_path = Path('/kaggle/input/grapheme-imgs-128x128')

csv_file = Path('/kaggle/input/iterative-stratification/train_with_fold.csv')

arch = xresnet50
# Load dataframe

df = pd.read_csv(csv_file)

df.drop(columns=['id'], inplace=True)

df.head()
dblock = DataBlock(

  blocks=(ImageBlock(cls=PILImageBW), *(3*[CategoryBlock])),      # one image input and three categorical outputs

  getters=[ColReader('image_id', pref=train_path, suff='.png'),   # image input

           ColReader('grapheme_root'),                            # label 1

           ColReader('vowel_diacritic'),                          # label 2

           ColReader('consonant_diacritic')],                     # label 3

  splitter=IndexSplitter(df.loc[df.fold==fold].index),            # train/validation split

  batch_tfms=[Normalize.from_stats([0.0692], [0.2051]),           # Normalize the images with the specified mean and standard deviation

              *aug_transforms(do_flip=False, size=sz)])           # Add default transformations except for horizontal flip      

dls = dblock.dataloaders(df, bs=bs)                               # Create the dataloaders

dls.n_inp = 1                                                     # Set the number of inputs
# Show an example

dls.show_batch()
# Model 

class Head(Module):

    def __init__(self, nc, n, ps=0.5):

        self.fc = nn.Sequential(*[AdaptiveConcatPool2d(), nn.ReLU(inplace=True), Flatten(),

             LinBnDrop(nc*2, 512, True, ps, nn.ReLU(inplace=True)),

             LinBnDrop(512, n, True, ps)])

        self._init_weight()

        

    def _init_weight(self):

        for m in self.modules():

            if isinstance(m, nn.Conv2d):

                torch.nn.init.kaiming_normal_(m.weight)

            elif isinstance(m, nn.BatchNorm2d):

                m.weight.data.fill_(1.0)

                m.bias.data.zero_()

        

    def forward(self, x):

        return self.fc(x)



class BengaliModel(Module):

    def __init__(self, arch=arch, n=dls.c, pre=True):

        m = arch(pre)

        m = nn.Sequential(*children_and_parameters(m)[:-4])

        conv = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)

        w = (m[0][0].weight.sum(1)).unsqueeze(1)

        conv.weight = nn.Parameter(w)

        m[0][0] = conv

        nc = m(torch.zeros(2, 1, sz, sz)).detach().shape[1]

        self.body = m

        self.heads = nn.ModuleList([Head(nc, c) for c in n])

        

    def forward(self, x):    

        x = self.body(x)

        return [f(x) for f in self.heads]
# Loss function

class Loss_combine(Module):

    def __init__(self, func=F.cross_entropy, weights=[2, 1, 1]):

        self.func, self.w = func, weights



    def forward(self, xs, *ys):

        for i, w, x, y in zip(range(len(xs)), self.w, xs, ys):

            if i == 0: loss = w*self.func(x, y) 

            else: loss += w*self.func(x, y) 

        return loss
# Metrics

class RecallPartial(Metric):

    # based on AccumMetric

    "Stores predictions and targets on CPU in accumulate to perform final calculations with `func`."

    def __init__(self, a=0, **kwargs):

        self.func = partial(recall_score, average='macro', zero_division=0)

        self.a = a



    def reset(self): self.targs,self.preds = [],[]



    def accumulate(self, learn):

        pred = learn.pred[self.a].argmax(dim=-1)

        targ = learn.y[self.a]

        pred,targ = to_detach(pred),to_detach(targ)

        pred,targ = flatten_check(pred,targ)

        self.preds.append(pred)

        self.targs.append(targ)



    @property

    def value(self):

        if len(self.preds) == 0: return

        preds,targs = torch.cat(self.preds),torch.cat(self.targs)

        return self.func(targs, preds)



    @property

    def name(self): return df.columns[self.a+1]

    

class RecallCombine(Metric):

    def accumulate(self, learn):

        scores = [learn.metrics[i].value for i in range(3)]

        self.combine = np.average(scores, weights=[2,1,1])



    @property

    def value(self):

        return self.combine
# Create learner

learn = Learner(dls, BengaliModel(), loss_func=Loss_combine(), cbs=CSVLogger(),

                metrics=[RecallPartial(a=i) for i in range(len(dls.c))] + [RecallCombine()],

                splitter=lambda m: [list(m.body.parameters()), list(m.heads.parameters())])
learn.fit_one_cycle(12, lr_max=slice(1e-3, 1e-2))
learn.recorder.plot_loss()
learn.save('model') 