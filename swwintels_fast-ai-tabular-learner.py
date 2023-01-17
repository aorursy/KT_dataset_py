from fastai import *

from fastai.tabular import *



from sklearn.metrics import roc_auc_score
df = pd.read_csv('../input/data.csv')
print(df.shape)

df = df.drop(["id","Unnamed: 32"], axis=1)

print(df.shape)
val_size = 100

valid_idx = range(len(df)-val_size, len(df))
procs = [FillMissing, Categorify, Normalize]

dep_var = 'diagnosis'

cat_names = []
def auc_score(y_score,y_true):

    return torch.tensor(roc_auc_score(y_true,y_score[:,1]))
data = TabularDataBunch.from_df("/", df, dep_var, valid_idx=valid_idx, procs=procs, cat_names=cat_names)

print(data.train_ds.cont_names)  # `cont_names` defaults to: set(df)-set(cat_names)-{dep_var}
learn = tabular_learner(data, layers=[200,100], metrics=[accuracy, auc_score])

learn.fit_one_cycle(1, 1e-2)
preds,y,losses = learn.get_preds(with_loss=True)

interp = ClassificationInterpretation(learn, preds, y, losses)

interp.plot_confusion_matrix()