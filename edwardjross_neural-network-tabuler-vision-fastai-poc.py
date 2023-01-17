from fastai import *

from fastai.vision import *

from fastai.tabular import *

from fastai.callbacks import *
import fastai

fastai.__version__
kappa = KappaScore()

kappa.weights = "quadratic"
from sklearn.metrics import cohen_kappa_score

qwk = partial(cohen_kappa_score, weights="quadratic")
path = Path('../input/petfinder-adoption-prediction')

work_path = Path('../working')
train_df = pd.read_csv(path / 'train/train.csv')

test_df = pd.read_csv(path / 'test/test.csv')
train_df['valid'] = (train_df.RescuerID > 'd')

train_df.valid.mean()
train_df['photo'] = train_df.PetID.apply(lambda x: f'train_images/{x}-1.jpg')

train_df['has_photo'] = train_df.photo.apply(lambda x: (path / x).exists())

test_df['photo'] = test_df.PetID.apply(lambda x: f'test_images/{x}-1.jpg')

test_df['has_photo'] = test_df.photo.apply(lambda x: (path / x).exists())

(~train_df['has_photo']).sum(), len(train_df)
train_df['age'] = train_df.Age.apply(lambda x: f'{x} mth' if x < 12 else f'{min(x//12, 6)} yrs')

train_df['quantity'] = train_df.Quantity.apply(lambda x: min(x, 5))

train_df['fee'] = np.log1p(train_df['Fee'])



test_df['age'] = test_df.Age.apply(lambda x: f'{x} mth' if x < 12 else f'{min(x//12, 6)} yrs')

test_df['quantity'] = test_df.Quantity.apply(lambda x: min(x, 5))

test_df['fee'] = np.log1p(test_df['Fee'])
cat_cols = ['age',

                      'Breed1',

                      'Gender',

                      'Color1',

                      'MaturitySize', 'FurLength',

                      'Vaccinated', 'Dewormed', 'Sterilized', 'Health',

                      'quantity', 'State', 'PhotoAmt']

cont_cols = ['fee']
procs = [FillMissing, Categorify, Normalize]

data_tab = (TabularList

        .from_df(train_df, cat_names=cat_cols, cont_names=cont_cols, procs=procs)

        .split_from_df(col='valid')

        .label_from_df(cols=['AdoptionSpeed'])

        .add_test(TabularList.from_df(test_df, cat_names=cat_cols, cont_names=cont_cols, procs=procs))

        .databunch(bs=32))
tab_learn = tabular_learner(data_tab, layers=[], metrics=[kappa, accuracy])
tab_learn.fit_one_cycle(4, max_lr=1e-2)
tab_pred, _ = tab_learn.get_preds(DatasetType.Test)

tab_valid, tab_targ = tab_learn.get_preds(DatasetType.Valid)
qwk(train_df[train_df.valid].AdoptionSpeed, tab_valid.argmax(1))
if not Path('../working/train_images').exists():

    os.symlink(path /'train_images', '../working/train_images', target_is_directory=True)

if not Path('../working/test_images').exists():

    os.symlink(path / 'test_images', '../working/test_images', target_is_directory=True)
transforms = get_transforms()

data_vis = (ImageList

        .from_df(train_df[train_df.has_photo].copy(), work_path, cols=['photo'])

        .split_from_df('valid')

        .label_from_df(cols=['AdoptionSpeed'])

        .transform(transforms, size=224)

        .add_test(ImageList.from_df(test_df[test_df.has_photo].copy(), path, cols=['photo']))

        # Pytorch error with too many workers

        .databunch(bs=32, num_workers=0)

        .normalize(imagenet_stats)

       )
Path('/tmp/.torch/models/').mkdir(parents=True, exist_ok = True)

shutil.copy('../input/resnet34/resnet34.pth', '/tmp/.torch/models/resnet34-333f7ec4.pth')
vis_learn = cnn_learner(data_vis, models.resnet34, pretrained=True, metrics=[kappa, accuracy])
vis_learn.fit_one_cycle(2, max_lr=3e-3)
vis_learn.unfreeze()

vis_learn.fit_one_cycle(3, max_lr=slice(3e-6, 3e-4))
vis_learn.save('vis')
vis_pred, _ = vis_learn.get_preds(DatasetType.Test)

vis_valid, targ = vis_learn.get_preds(DatasetType.Valid)
valid_mask = train_df.valid & train_df.has_photo
qwk(train_df.AdoptionSpeed[train_df.valid & train_df.has_photo], vis_valid.argmax(1))
vis_mask = train_df[train_df.valid].has_photo

vis_idx = list(np.where(vis_mask))
qwk(train_df.AdoptionSpeed[train_df.valid & train_df.has_photo], (vis_valid+tab_valid[vis_idx]).argmax(1))
tfms = get_transforms()

tfms = [[tfms[0], []], [tfms[0], []]]
procs = [FillMissing, Categorify, Normalize]

list_tab = (TabularList

            .from_df(train_df[train_df.has_photo],

             cat_names=cat_cols,

             cont_names=cont_cols,

             procs=procs))
list_vis = (ImageList

            .from_df(train_df[train_df.has_photo], path, cols=['photo']))
data = (MixedItemList([list_vis, list_tab], path='.', inner_df=train_df[train_df.has_photo])

        .split_from_df('valid')

        .label_from_df(cols=['AdoptionSpeed'])

        .transform(tfms, size=224)

        .databunch(bs=32, num_workers=0))
emb_szs = data_tab.get_emb_szs({})

emb_szs
nt = 100
tab = TabularModel(emb_szs, len(data_tab.cont_names), out_sz=nt, layers=[], bn_final=True)
class Ridealong(nn.Module):

    """Run m on the first element of list, and bring the others along for the ride"""

    def __init__(self, m, posn=0):

        super().__init__()

        self.m = m

        self.posn = posn

        

    def forward(self, *xs):

        x = xs[self.posn]

        if isinstance(x, Tensor): x = [x]

        x = self.m(*x)

        xs = (*xs[:self.posn], x, *xs[self.posn+1:])

        return xs
class CombineVisTab(nn.Module):

    def __init__(self, nf, ts, nc):

        super().__init__()

        self.nf = nf

        self.ts = ts

        self.nc = nc

        self.flat = nn.Sequential(AdaptiveConcatPool2d(), Flatten())

        self.layers = nn.Sequential(*bn_drop_lin(nf + ts, nc))

        

    def forward(self, *xs):

        vis, tab = xs

        x = torch.cat([self.flat(vis), tab], dim=1)

        return self.layers(x)

        
body = create_body(models.resnet34, pretrained=True)

nf = num_features_model(body) * 2
class MultiSeq(nn.Sequential):

    def forward(self, *input):

        for module in self._modules.values():

            input = module(*input)

        return input
model = MultiSeq(

    Ridealong(body[:6]),

    Ridealong(body[6:]),

    MultiSeq(

        Ridealong(tab, 1),

        CombineVisTab(nf, nt, data.c)

    )

)

    

gc.collect()
learn = Learner(data, model, metrics=[accuracy, kappa])
learn = learn.split([model[0], model[1]])
learn.freeze()
learn.lr_find()
learn.recorder.plot()
learn.fit_one_cycle(2, max_lr=slice(1e-3))
learn.save('l1')
learn.unfreeze()
learn.fit_one_cycle(3, max_lr=slice(1e-6, 1e-4))
#sub_df = pd.DataFrame(data={'PetID': test_df.PetID, 'AdoptionSpeed': pred.argmax(1).numpy()})

#sub_df.to_csv('submission.csv', index=False)