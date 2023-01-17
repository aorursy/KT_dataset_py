import pandas as pd

from fastai.tabular import *

import fastai
fastai.__version__
# Show all output in cells

from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"
path = Path('/kaggle/input/adult-census-income/')

df = pd.read_csv(path/'adult.csv')



df = df.join(pd.get_dummies(df.income))

df.head(2)
dep_var = '>50K'

cat_names = ['workclass', 'education', 'marital.status', 'occupation', 'relationship', 'race']

cont_names = ['age', 'fnlwgt', 'education.num', 'hours.per.week']

procs = [FillMissing, Categorify, Normalize]
df[dep_var].value_counts()
N = 10_000



idx_test = df.iloc[-N+1:].index # last N rows

idx_val  = df.iloc[-2*N:-N].index # -2N:-N last rows

idx_val, idx_test
df.loc[idx_test, dep_var].value_counts()
test = TabularList.from_df(df.loc[idx_test].copy(), path=path, cat_names=cat_names, cont_names=cont_names)
BS = 64



data = (TabularList.from_df(df, path='/kaggle/working', cat_names=cat_names, cont_names=cont_names, procs=procs)

                           .split_by_idx(idx_val)

                           .label_from_df(cols=dep_var)

                           .add_test(test)

                           .databunch())



data.batch_size = BS



data.show_batch(rows=5)
print('starting training...')

learn = tabular_learner(data, layers=[200,100], metrics=[accuracy, dice, AUROC()])



learn.lr_find()

learn.recorder.plot()
lr = 8e-02

learn.fit_one_cycle(1, lr)
learn.lr_find()

learn.recorder.plot()
learn.fit_one_cycle(3, 1e-05)
probas_test, _ = learn.get_preds(ds_type=DatasetType.Test) # run inference on test using GPU

probas_test = probas_test[:, 1] # only get fraud probability tensor

len(probas_test), len(df.loc[idx_test, dep_var].values)
df.head()
submission_df = pd.DataFrame({'Index':df.loc[idx_test].index.values,

                  dep_var:probas_test}

                )



submission_df.head()



submission_df.to_csv('submission.csv')