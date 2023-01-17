import torch

print("GPU name:", torch.cuda.get_device_name(0))
from fastai.tabular import * 
train = pd.read_csv("../input/murcia-car-challenge/train.csv", index_col="Id")

test  = pd.read_csv("../input/murcia-car-challenge/test.csv", index_col="Id")

sub   = pd.read_csv("../input/murcia-car-challenge/sampleSubmission.csv", index_col="Id")
procs = [FillMissing, Categorify, Normalize]
valid_idx = range(int(len(train)*0.9), len(train))

valid_idx
train.columns
cat_vars = ['Marca', 'Modelo', 'Provincia', 'Localidad', 'Cambio', 'Combust', 'Puertas', 'Vendedor']

num_vars = ['Año', 'Kms', 'Cv']

target_var   = 'Precio'
data = (TabularList.from_df(train, path=".", cat_names=cat_vars,

                            cont_names=num_vars, procs=procs)

                           .split_by_idx(valid_idx)

                           .label_from_df(cols=target_var, label_cls=FloatList, log=True)#log=True

                           .databunch())



data.add_test(TabularList.from_df(test, path=".", cat_names=cat_vars,

                            cont_names=num_vars, procs=procs))
learn = tabular_learner(data, layers=[100,100],

                        ps=0.5,

                        emb_drop=0.5,

                        metrics=[msle, rmse, r2_score]#[accuracy, AUROC()]

                       )
learn.fit_one_cycle(3, 1e-2)
len(sub)
preds = learn.get_preds(ds_type=DatasetType.Test)



preds_probs  = preds[0][:,0].numpy()

preds_values = preds[1].numpy()
preds_probs
sub['Precio'] = np.expm1(preds_probs) # test_preds.astype(int)

sub.head()
sub.to_csv('sub.csv', header=True, index=True)
! head sub.csv