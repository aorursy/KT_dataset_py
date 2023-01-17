!pip install fastai --upgrade 
from fastai.tabular.all import *
path = Path("../input/adult-census-income")
path.ls()
df = pd.read_csv(path/'adult.csv')
df.head()
cat_names = ['workclass', 'education', 'marital.status', 'occupation', 'relationship', 'race']
cont_names = ['age', 'fnlwgt', 'education.num']
cat = Categorify()
df.dtypes
to = TabularPandas(df, cat, cat_names)
cats = to.procs.categorify
cats['race']
to.show(max_n=3)
to.cats.head()
!pip install fastbook
cont_names
to = TabularPandas(df, Categorify() , cont_names=cont_names)
#to = TabularPandas(df, NormalizeTab(), cont_names=cont_names)

fm = FillMissing(fill_strategy=FillStrategy.median)
to = TabularPandas(df, fm, cont_names=cont_names)
to.conts.head()
to.cat_names
to.cats.head()
splits = RandomSplitter()(range_of(df))
splits
len(df)
cat_names = ['workclass', 'education', 'marital.status', 'occupation', 'relationship', 'race']
cont_names = ['age', 'fnlwgt', 'education.num']
procs = [Categorify, Normalize]
y_names = 'income'
y_block = CategoryBlock()
to = TabularPandas(df, procs=procs, cat_names=cat_names, cont_names=cont_names,
                   y_names= y_names, y_block=y_block, splits=splits)
dls = to.dataloaders()

dls.show_batch()
trn_dl = TabDataLoader(to.train, bs=64, shuffle=True, drop_last=True)
val_dl = TabDataLoader(to.valid, bs=128)
dls = DataLoaders(trn_dl, val_dl)
dls.show_batch()
to._dbunch_type
dls._dbunch_type


def get_emb_sz(to, sz_dict=None):
    "Get default embedding size from `TabularPreprocessor` `proc` or the ones in `sz_dict`"
    return [_one_emb_sz(to.classes, n, sz_dict) for n in to.cat_names]


def _one_emb_sz(classes, n, sz_dict=None):
    "Pick an embedding size for `n` depending on `classes` if not given in `sz_dict`."
    sz_dict = ifnone(sz_dict, {})
    n_cat = len(classes[n])
    sz = sz_dict.get(n, int(emb_sz_rule(n_cat)))  # rule of thumb
    return n_cat,sz
def emb_sz_rule(n_cat):
    "Rule of thumb to pick embedding size corresponding to `n_cat`"
    return min(600, round(1.6 * n_cat**0.56))

emb_szs = get_emb_sz(to)
emb_szs
to.cat_names
to['workclass'].nunique()
cont_len = len(to.cont_names)
cont_len
batch = dls.one_batch()
len(batch)
batch[0][0], batch[1][0]
net = TabularModel(emb_szs, cont_len, 2, [200, 100])
net
learn = tabular_learner(dls, [200,100], metrics=accuracy)
learn.lr_find()
learn.fit(3, 1e-2)
dls = to.dataloaders(bs=1024)
learn = tabular_learner(dls, [200,100], metrics=accuracy)
learn.lr_find()
learn.fit(3, 1e-2)
dls = to.dataloaders(bs=4096)
learn = tabular_learner(dls, [200,100], metrics=accuracy)
learn.lr_find()
learn.fit_one_cycle(3,1e-2)
learn.export('myModel.pkl')
del learn
learn = load_learner('myModel.pkl')
dl = learn.dls.test_dl(df.iloc[:100])
dl.show_batch()
df2 = df.iloc[:100].drop('income', axis=1)
df2.head()
dl = learn.dls.test_dl(df2)
dl.show_batch()
learn.validate(dl=dl)


dl = learn.dls.test_dl(df.iloc[:100])



learn.validate(dl=dl)

