from fastai.tabular import *
import pandas as pd

sample_submission = pd.read_csv("../input/cat-in-the-dat-ii/sample_submission.csv")

X_test = pd.read_csv("../input/cat-in-the-dat-ii/test.csv")

train = pd.read_csv("../input/cat-in-the-dat-ii/train.csv")
train.head()
procs = [FillMissing, Categorify, Normalize]

FillMissing.FillStrategy='MEAN'



dep_var = 'target'

cat_names = ['bin_0', 'bin_1', 'bin_2', 'bin_3', 'bin_4', 'nom_0', 'nom_1', 'nom_2', 

             'nom_3', 'nom_4', 'nom_5', 'nom_6', 'nom_7', 'nom_8', 'nom_9',

            'ord_0', 'ord_1', 'ord_2', 'ord_3', 'ord_4', 'ord_5', 'day', 'month']

cont_names = []

X=train[cont_names]

X=train[cat_names]
PATH = Path('/kaggle/input/cat-in-the-dat-ii/')

test = TabularList.from_df(X_test, path=PATH, cat_names=cat_names, cont_names=cont_names)
data = (TabularList.from_df(train, path=PATH,

                            cat_names=cat_names, 

                            cont_names=cont_names,

                            procs=procs)

                           .split_by_idx(valid_idx = range(len(train)-50000, len(train)))

                           .label_from_df(cols=dep_var)

                           .add_test(test)

                           .databunch())

print(data.train_ds.cont_names)

print(data.train_ds.cat_names)
data.show_batch()
learn = tabular_learner(data, layers=[200,100], metrics=accuracy,model_dir="/tmp/model/", ps=0.15)
# learn.lr_find()

# learn.recorder.plot()
learn.fit(1, lr=1e-3)
# learn.lr_find()

# learn.recorder.plot()
lr=1e-03
learn.fit_one_cycle(1, slice(lr))
# learn.lr_find()

# learn.recorder.plot()
lr=1e-03
learn.fit_one_cycle(2, slice(lr),wd=0.3)
learn.recorder.plot_losses()
learn.save('model_1')
preds = learn.get_preds(ds_type=DatasetType.Test)[0][:,1].numpy()
submission = pd.DataFrame({'id':sample_submission['id'],'target':preds})

submission.to_csv('my_submission.csv', header=True, index=False)
submission.describe()