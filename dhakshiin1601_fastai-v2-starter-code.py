!pip install git+https://github.com/fastai/fastai_dev             > /dev/null
from fastai2.data.all import *

from fastai2.tabular.core import *

from fastai2.tabular.model import *

from fastai2.optimizer import *

from fastai2.learner import *

from fastai2.metrics import *

from fastai2.callback.all import *
path = Path('/kaggle/input/ashrae-energy-prediction')
train = pd.read_csv(path/'train.csv')

train = train.iloc[:7000]



bldg = pd.read_csv(path/'building_metadata.csv')

weather_train = pd.read_csv(path/"weather_train.csv")
train = train[np.isfinite(train['meter_reading'])]
train.head()
bldg.head()
train = train.merge(bldg, left_on = 'building_id', right_on = 'building_id', how = 'left')
train.head()
weather_train.head()
train = train.merge(weather_train, left_on = ['site_id', 'timestamp'], right_on = ['site_id', 'timestamp'])
del weather_train
train["timestamp"] = pd.to_datetime(train["timestamp"])

train["hour"] = train["timestamp"].dt.hour

train["day"] = train["timestamp"].dt.day

train["weekend"] = train["timestamp"].dt.weekday

train["month"] = train["timestamp"].dt.month
train.drop('timestamp', axis=1, inplace=True)

train['meter_reading'] = np.log1p(train['meter_reading'])
test  = train.copy( deep=True)
cat_vars = ["building_id", "primary_use", "hour", "day", "weekend", "month", "meter"]

cont_vars = ["square_feet", "year_built", "air_temperature", "cloud_coverage",

              "dew_temperature"]

dep_var = 'meter_reading'
procs = [Normalize, Categorify, FillMissing]

splits = RandomSplitter()(range_of(train))
to = TabularPandas(train, procs, cat_vars, cont_vars, y_names=dep_var, splits=splits, is_y_cat=False)
to
to.train
dbch = to.databunch()

dbch.valid_dl.show_batch()
trn_dl = TabDataLoader(to.train, bs=64, num_workers=0, shuffle=True, drop_last=True)

val_dl = TabDataLoader(to.valid, bs=128, num_workers=0)
dbunch = DataBunch(trn_dl, val_dl)

dbunch.valid_dl.show_batch()
def emb_sz_rule(n_cat): 

    "Rule of thumb to pick embedding size corresponding to `n_cat`"

    return min(600, round(1.6 * n_cat**0.56))
def _one_emb_sz(classes, n, sz_dict=None):

    "Pick an embedding size for `n` depending on `classes` if not given in `sz_dict`."

    sz_dict = ifnone(sz_dict, {})

    n_cat = len(classes[n])

    sz = sz_dict.get(n, int(emb_sz_rule(n_cat)))  # rule of thumb

    return n_cat,sz
def get_emb_sz(to, sz_dict=None):

    "Get default embedding size from `TabularPreprocessor` `proc` or the ones in `sz_dict`"

    return [_one_emb_sz(to.procs.classes, n, sz_dict) for n in to.cat_names]
emb_szs = get_emb_sz(to); print(emb_szs)
class TabularModel(Module):

    "Basic model for tabular data."

    def __init__(self, emb_szs, n_cont, out_sz, layers, ps=None, embed_p=0., y_range=None, use_bn=True, bn_final=False):

        ps = ifnone(ps, [0]*len(layers))

        if not is_listy(ps): ps = [ps]*len(layers)

        self.embeds = nn.ModuleList([Embedding(ni, nf) for ni,nf in emb_szs])

        self.emb_drop = nn.Dropout(embed_p)

        self.bn_cont = nn.BatchNorm1d(n_cont)

        n_emb = sum(e.embedding_dim for e in self.embeds)

        self.n_emb,self.n_cont,self.y_range = n_emb,n_cont,y_range

        sizes = [n_emb + n_cont] + layers + [out_sz]

        actns = [nn.ReLU(inplace=True) for _ in range(len(sizes)-2)] + [None]

        _layers = [BnDropLin(sizes[i], sizes[i+1], bn=use_bn and i!=0, p=p, act=a)

                       for i,(p,a) in enumerate(zip([0.]+ps,actns))]

        if bn_final: _layers.append(nn.BatchNorm1d(sizes[-1]))

        self.layers = nn.Sequential(*_layers)

    

    def forward(self, x_cat, x_cont):

        if self.n_emb != 0:

            x = [e(x_cat[:,i]) for i,e in enumerate(self.embeds)]

            x = torch.cat(x, 1)

            x = self.emb_drop(x)

        if self.n_cont != 0:

            x_cont = self.bn_cont(x_cont)

            x = torch.cat([x, x_cont], 1) if self.n_emb != 0 else x_cont

        x = self.layers(x)

        if self.y_range is not None:

            x = (self.y_range[1]-self.y_range[0]) * torch.sigmoid(x) + self.y_range[0]

        return x
model = TabularModel(emb_szs, len(to.cont_names), 1, [1000,500]); model
opt_func = partial(Adam, wd=0.01, eps=1e-5)

learn = Learner(dbunch, model, MSELossFlat(), opt_func=opt_func)
learn.fit_one_cycle(5)
# preds = learn.get_preds(dl = dbunch.train_dl)
# preds = learn.get_preds(dl = dbunch.valid_dl)
# test = pd.read_csv(path/'test.csv')
test.shape
to_test = TabularPandas(test, procs, cat_vars, cont_vars, y_names=dep_var ,  is_y_cat=False)
to_test.train.shape
to_test.valid.shape
dbch_test = to_test.databunch(shuffle_train = False)
preds = learn.get_preds(dl = dbch_test.train_dl)
trn_dl = TabDataLoader(to_test.train, bs=64, num_workers=0, shuffle=False)

tdbunch = DataBunch(trn_dl)
preds = learn.get_preds(dl = tdbunch.train_dl)