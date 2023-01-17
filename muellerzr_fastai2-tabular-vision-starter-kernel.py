!pip install fastai2 pydicom --quiet
from fastai2.tabular.all import *

from fastai2.vision.all import *

from fastai2.medical.imaging import *
path = Path("../input/siim-isic-melanoma-classification")
df = pd.read_csv(path/'train.csv')

test_df = pd.read_csv(path/'test.csv')
df.head()
df['image_name'] = 'train/' + df['image_name'].astype(str)

test_df['image_name'] = 'test/' + test_df['image_name'].astype(str)
cols = ['image_name', 'sex', 'age_approx',

       'anatom_site_general_challenge', 'diagnosis', 'benign_malignant',

       'target']

df = df[cols]

procs = [Categorify, FillMissing, Normalize]

cat_names  = ['sex', 'anatom_site_general_challenge']

cont_names = ['age_approx']

splitter = RandomSplitter(seed=42)

splits = splitter(range_of(df))

to = TabularPandas(df, procs, cat_names, cont_names,

                  y_names='target', y_block=CategoryBlock(),

                  splits=splits)
tab_dl = to.dataloaders(bs=8)
get_x = lambda x:path/f'{x[0]}.dcm'

get_y=ColReader('target')

batch_tfms = aug_transforms(flip_vert=True, max_lighting=0.1, max_zoom=1.05, max_warp=0.)

blocks = (ImageBlock(cls=PILDicom), CategoryBlock(vocab=[0,1]))

melanoma = DataBlock(blocks=blocks,

                   get_x=get_x,

                   splitter=splitter,

                   item_tfms=Resize(128),

                   get_y=ColReader('target'),

                   batch_tfms=batch_tfms)
vis_dl = melanoma.dataloaders(df, bs=8)
type(vis_dl[0])
from fastai2.data.load import _FakeLoader, _loaders
class MixedDL():

    def __init__(self, tab_dl:TabDataLoader, vis_dl:TfmdDL, device='cuda:0'):

        "Stores away `tab_dl` and `vis_dl`, and overrides `shuffle_fn`"

        self.device = device

        tab_dl.shuffle_fn = self.shuffle_fn

        vis_dl.shuffle_fn = self.shuffle_fn

        self.dls = [tab_dl, vis_dl]

        self.count = 0

        self.fake_l = _FakeLoader(self, False, 0, 0)

    

    def __len__(self): return len(self.dls[0])

        

    def shuffle_fn(self, idxs):

        "Generates a new `rng` based upon which `DataLoader` is called"

        if self.count == 0: # if we haven't generated an rng yet

            self.rng = self.dls[0].rng.sample(idxs, len(idxs))

            self.count += 1

            return self.rng

        else:

            self.count = 0

            return self.rng

        

    def to(self, device): self.device = device
vis_dl[0].get_idxs()[:10]
tab_dl[0].get_idxs()[:10]
mixed_dl = MixedDL(tab_dl[0], vis_dl[0])
mixed_dl.dls[0].get_idxs()[:10]
mixed_dl.dls[1].get_idxs()[:10]
@patch

def __iter__(dl:MixedDL):

    "Iterate over your `DataLoader`"

    z = zip(*[_loaders[i.fake_l.num_workers==0](i.fake_l) for i in dl.dls])

    for b in z:

        if dl.device is not None: 

            b = to_device(b, dl.device)

        batch = []

        batch.extend(dl.dls[0].after_batch(b[0])[:2])

        batch.append(dl.dls[1].after_batch(b[1][0]))

        try: # In case the data is unlabelled

            batch.append(b[1][1])

            yield tuple(batch)

        except:

            yield tuple(batch)
@patch

def one_batch(x:MixedDL):

    "Grab a batch from the `DataLoader`"

    with x.fake_l.no_multiproc(): res = first(x)

    if hasattr(x, 'it'): delattr(x, 'it')

    return res
batch = mixed_dl.one_batch()
batch[0]
batch[1]
batch[2]
batch[3]
@patch

def show_batch(x:MixedDL):

    "Show a batch from multiple `DataLoaders`"

    for dl in x.dls:

        dl.show_batch()
mixed_dl.show_batch()
im_test = vis_dl.test_dl(test_df)

tab_test = tab_dl.test_dl(test_df)
test_dl = MixedDL(tab_test, im_test)
!pip install fastinference --quiet
from fastinference.inference import *
#learn.dls.n_inp = 3
# preds = learn.get_preds(dl=test_dl, decoded_loss=True)