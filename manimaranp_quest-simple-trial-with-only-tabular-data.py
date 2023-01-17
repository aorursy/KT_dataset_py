from fastai.tabular import *

from fastai.metrics import rmse
def reset_seed(seed=42):

    random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

#     tf.set_random_seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.backends.cudnn.deterministic = True

    if torch.cuda.is_available(): 

        torch.cuda.manual_seed(seed)

        torch.cuda.manual_seed_all(seed)



reset_seed()
path = Path('/kaggle/input/google-quest-challenge')

path.ls()
train_data = pd.read_csv(path/'train.csv', index_col=[0])

train_data.head()
test_data = pd.read_csv(path/'test.csv', index_col=[0])

test_data.head()
cat_names = ['category', 'host']

# cont_names = train_data.columns[10:].tolist()

dep_var = train_data.columns[10:].tolist()

procs = [Categorify]
data = (TabularList.from_df(train_data, path=path, cat_names=cat_names, procs=procs)

                           .split_by_rand_pct(0.1)

                           .label_from_df(cols=dep_var, label_cls=FloatList, log=False)

                           .add_test(TabularList.from_df(test_data, path=path, cat_names=cat_names))

                           .databunch())
data.show_batch(rows=5)
learn = tabular_learner(data, layers=[10, 20, 30, 20, 10], metrics=rmse, emb_drop=0.2, ps=[0.2, 0.2, 0.2, 0.2, 0.2])

learn.loss_func=MSELossFlat()
learn.model_dir = Path('/kaggle/working')
learn.lr_find()
learn.recorder.plot(suggestion=True)
learn.fit_one_cycle(10, learn.recorder.min_grad_lr, wd=1e-4)
preds,y = learn.get_preds(DatasetType.Test)
preds.data.numpy()
preds.shape
sample =  pd.read_csv(path/'sample_submission.csv')

# sample.to_csv('submission.csv', index=False)

sample.head()
sample.iloc[:, 1:] = preds.data.numpy()
sample.head()
sample.to_csv('submission.csv', index=False)