import seaborn as sns

import matplotlib.pyplot as plt



from fastai.tabular import *

from sklearn.metrics import roc_auc_score



%matplotlib inline
torch.manual_seed(47)



torch.backends.cudnn.deterministic = True

torch.backends.cudnn.benchmark = False



np.random.seed(47)
data_dir = '../input/'

!ls {data_dir}
train_raw = pd.read_csv(f'{data_dir}train.csv')

train_raw.head()
test_raw = pd.read_csv(f'{data_dir}test.csv')

test_raw.head()
train_raw.shape, test_raw.shape
train_raw.isnull().sum().sum(), test_raw.isnull().sum().sum()
sns.countplot(train_raw.target)

plt.show()
train_raw.target.value_counts()
valid_idx = range(len(train_raw)- 20000, len(train_raw))
cont_names = train_raw.columns.tolist()[1:-1]

cont_names.remove('wheezy-copper-turtle-magic')



cat_names = ['wheezy-copper-turtle-magic']



procs = [FillMissing, Categorify, Normalize]
dep_var = 'target'



data = TabularDataBunch.from_df('.', train_raw, dep_var=dep_var, valid_idx=valid_idx, procs=procs,

                                cat_names=cat_names, cont_names=cont_names, test_df=test_raw, bs=2048)
learn = tabular_learner(data, layers=[1000, 750, 500, 300], emb_szs={'wheezy-copper-turtle-magic': 200}, metrics=accuracy, ps=0.65, wd=3e-1)
learn.lr_find()
learn.recorder.plot()
lr = 1e-2

learn.fit_one_cycle(26, lr)
val_preds = learn.get_preds(DatasetType.Valid)

roc_auc_score(train_raw.iloc[valid_idx].target.values, val_preds[0][:,1].numpy())
test_preds = learn.get_preds(DatasetType.Test)
sub_df = pd.read_csv(f'{data_dir}sample_submission.csv')

sub_df.target = test_preds[0][:,1].numpy()

sub_df.head()
sub_df.to_csv('solution.csv', index=False)