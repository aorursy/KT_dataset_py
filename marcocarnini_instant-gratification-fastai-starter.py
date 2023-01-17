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
columns = train_raw.columns[1:-1]

first_name = [i.split("-")[0] for i in columns]

print(set(first_name))

print(len(first_name))

print(len(set(first_name)))
for first in first_name:

    filter_col = [col for col in train_raw if col.startswith(first)]

    test_raw[first+"-mean"] = test_raw.loc[:, filter_col].mean(axis=1)

    train_raw[first+"-mean"] = train_raw.loc[:, filter_col].mean(axis=1)

    test_raw[first+"-std"] = test_raw.loc[:, filter_col].std(axis=1)

    train_raw[first+"-std"] = train_raw.loc[:, filter_col].std(axis=1)
second_name = [i.split("-")[1] for i in columns]

print(set(second_name))

print(len(second_name))

print(len(set(second_name)))
for second in second_name:

    filter_col = [col for col in columns if second==col.split("-")[1]]

    test_raw[second+"-mean"] = test_raw.loc[:, filter_col].mean(axis=1)

    train_raw[second+"-mean"] = train_raw.loc[:, filter_col].mean(axis=1)

    test_raw[second+"-std"] = test_raw.loc[:, filter_col].std(axis=1)

    train_raw[second+"-std"] = train_raw.loc[:, filter_col].std(axis=1)
train_raw.shape
for col in train_raw.columns:

    if (train_raw[col].isnull().sum()>0):

        train_raw.drop([col], axis=1, inplace=True)

        test_raw.drop([col], axis=1, inplace=True)
train_raw.shape
cont_names = train_raw.columns.tolist()

cont_names.remove('id')

cont_names.remove('target')

cont_names.remove('wheezy-copper-turtle-magic')



cat_names = ['wheezy-copper-turtle-magic']



procs = [FillMissing, Categorify, Normalize]
dep_var = 'target'



data = TabularDataBunch.from_df('.', train_raw, dep_var=dep_var, valid_idx=valid_idx, procs=procs,

                                cat_names=cat_names, cont_names=cont_names, test_df=test_raw, bs=2048)
learn = tabular_learner(data, layers=[1000, 750, 500, 300], emb_szs={'wheezy-copper-turtle-magic': 512}, metrics=accuracy, ps=0.65, wd=3e-1)
learn.lr_find()
learn.recorder.plot()
lr = 1e-3

learn.fit_one_cycle(40, lr)
learn.save("stage-1")
learn.recorder.plot_losses()
learn.recorder.plot_metrics()
learn.lr_find()
learn.recorder.plot()
learn.fit_one_cycle(200, 1e-4)
learn.recorder.plot_losses()
learn.recorder.plot_metrics()
learn.save("model-2")
val_preds = learn.get_preds(DatasetType.Valid)

roc_auc_score(train_raw.iloc[valid_idx].target.values, val_preds[0][:,1].numpy())
data = TabularDataBunch.from_df('.', train_raw, dep_var=dep_var, valid_idx=[], procs=procs,

                                cat_names=cat_names, cont_names=cont_names, test_df=test_raw, bs=2048)

learn.data = data
learn.lr_find()

learn.recorder.plot()
learn.fit_one_cycle(200, 1e-4)
test_preds = learn.get_preds(DatasetType.Test)
sub_df = pd.read_csv(f'{data_dir}sample_submission.csv')

sub_df.target = test_preds[0][:,1].numpy()

sub_df.head()
sub_df.to_csv('solution.csv', index=False)
test_raw.to_csv("test_raw.csv", index=False)

train_raw.to_csv("train_raw.csv", index=False)