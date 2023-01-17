!pip install torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html -q
!pip install fastai==2.0.13 -q
from fastai.tabular.all import *
import matplotlib.pyplot as plt
path_train = "../input/osic-pulmonary-fibrosis-progression/train.csv"
path_test = "../input/osic-pulmonary-fibrosis-progression/test.csv"
dataset = pd.read_csv(path_train)
test = pd.read_csv(path_test)
dataset.head()
dataset.isna().sum()

dataset = dataset.astype({'Age': 'float'})
dataset = dataset.astype({'Weeks': 'float'})
dataset = dataset.astype({'FVC': 'float'})
test = dataset.astype({'Age': 'float'})
test = dataset.astype({'Weeks': 'float'})
test = dataset.astype({'FVC': 'float'})
dataset.dtypes
test.dtypes
dataset.shape
dataset.describe().T
fig, (ax1, ax2,ax3) = plt.subplots(1, 3,figsize=(12,6))

ax1.scatter(dataset['Percent'].values, dataset['FVC'].values, label='Percent')
ax2.scatter(dataset['Age'].values, dataset['FVC'].values, label='Age', color='#EF4202')
ax3.scatter(dataset['Sex'].values, dataset['FVC'].values, label='Sex', color='#4A7A6B')

ax1.set_xlabel('Percent')
ax1.set_ylabel('FVC')
ax1.set_title('Percent vs FVC')
ax1.legend()

ax2.set_xlabel('Age')
ax2.set_ylabel('FVC')
ax2.set_title('Age vs FVC')
ax2.legend()

ax3.set_xlabel('Sex')
ax3.set_ylabel('FVC')
ax3.set_title('Sex vs FVC')
ax3.legend()

plt.show()
cat_names=['Sex', 'SmokingStatus']
cont_names = ['Age','Percent','Weeks']
procs = [Categorify, FillMissing, Normalize]
dls = TabularDataLoaders.from_df(dataset, path_train, procs=procs, cat_names=cat_names, cont_names=cont_names, 
                                 y_names="FVC", valid_pct=0.2, bs=64, shuffle_train=False)

dls_test = TabularDataLoaders.from_df(dataset, path_test, procs=procs, cat_names=cat_names, cont_names=cont_names, 
                                 y_names="FVC", valid_pct=0.2, bs=64, shuffle_train=False)
dls.show_batch()
learn = tabular_learner(dls,layers=[64,32],n_out=1, loss_func=F.mse_loss, metrics=rmse)
learn.fit_one_cycle(50,lr_max=0.03)
learn.recorder.plot_loss()
preds, y = learn.get_preds()

error = preds - y
plt.hist(error.detach().numpy(), bins = 25, color='#9999CF',density=False)
plt.axvline(error.mean(), color='k', linestyle='dashed', linewidth=1)
plt.xlabel("Prediction Error [labels]")
_ = plt.ylabel("Count")
import seaborn as sns
# Density Plot and Histogram of validation errors.
sns.distplot(error.detach().numpy(), hist=True, kde=True, 
             bins=30, color = 'darkblue', 
             hist_kws={'edgecolor':'#9999CF'},
             kde_kws={'linewidth': 4})
plt.show()
plt.figure(figsize=(8,10))
plt.axes(aspect='equal')
plt.scatter(y.detach().numpy(),y.detach().numpy(), color='#EF4202',label='Perfect model')
plt.scatter(y.detach().numpy(), preds, alpha=0.4,label='Our model')
plt.xlabel('True Labels')
plt.ylabel('Predicted')
plt.legend()
plt.show()
test_dl = learn.dls.test_dl(test)
sub = learn.get_preds(dl=test_dl)
sub = pd.read_csv(f"{ROOT}/sample_submission.csv")
sub['Patient'] = sub['Patient_Week'].apply(lambda x:x.split('_')[0])
sub
sub['Weeks'] = sub['Patient_Week'].apply(lambda x: int(x.split('_')[-1]))
sub = sub[['Patient', 'Weeks', 'FVC', 'Confidence']]
sub
sub['Patient'].unique()
sub['Patient']
c = sub[sub['Patient'] == 'ID00419637202311204720264']
sub
sub['Age'] = sub['Patient']
sub['Patient'].unique()
for i in range(5):
    sub.loc[sub["Patient"]==test['Patient'][i], "Age"] = test['Age'][i]
    sub.loc[sub["Patient"]==test['Patient'][i], "Sex"] = test['Sex'][i]
sub
