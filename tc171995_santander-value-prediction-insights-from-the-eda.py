#base modules
import numpy as np
import pandas as pd

#visualization
import seaborn as sns
from matplotlib import pyplot as plt

#Scipy
import scipy

#scikit-learn
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD

#LightGbm
import lightgbm

#Model validation
from sklearn.model_selection import cross_validate
from sklearn.metrics.scorer import make_scorer
from IPython.display import set_matplotlib_formats
set_matplotlib_formats('retina')
# Let's import the dataset
train_data  = pd.read_csv("../input/train.csv")
test_data = pd.read_csv("../input/test.csv")
train_data.head()
test_data.head()
train_id = train_data['ID']
train_target = train_data['target']
test_id = test_data['ID']
del train_data['ID']
del train_data['target']
del test_data['ID']
print("The number of columns in train dataset are %i" % len(train_data.columns))
print("The number of rows in train dataset are %i" % len(train_data))
print("The number of columns in test dataset are %i" % len(test_data.columns))
print("The number of rows in test dataset are %i" % len(test_data))
plt.figure(figsize=(15,8))
sns.distplot(train_target)
plt.xlabel("Target",fontsize=14)
plt.title("Histogram-KDE plot of target variable",fontsize=14)
plt.figure(figsize=(15,10))
sns.distplot(np.log(train_target))
plt.xlabel("Target",fontsize=14)
plt.title("Histogram-KDE of log transformation of the target varible",fontsize=14)
box_cox_trans = scipy.stats.boxcox(train_target.values,lmbda=0.1)
box_cox_trans
plt.figure(figsize=(15,10))
sns.distplot(box_cox_trans)
plt.xlabel("Target",fontsize=14)
plt.title("Histogram-KDE plot of box-cox transformation of the target variable",fontsize=14)
train_data.nunique()[train_data.nunique(axis=0)==1]
constant_column_names = train_data.columns[train_data.nunique(axis=0)==1].tolist() #Saving the redundant column names 
train_data.isnull().values.any()
combined_data.dtypes.value_counts()
train_data = train_data.drop(columns = constant_column_names,axis=1)
test_data = test_data.drop(columns = constant_column_names,axis=1)
combined_data = pd.concat([train_data,test_data],axis=0)
print(combined_data.shape)
# combined_data = scale(combined_data)
pca = PCA(n_components=2000)
pca.fit_transform(combined_data)
cumsum_variance = np.cumsum(np.round(pca.explained_variance_ratio_,decimals=4)*100)
plt.plot(cumsum_variance)
plt.xlabel("Number of components")
plt.ylabel("Cumulative percentage of explained variance")
plt.title("Plot of explained variance in percentage")
tsvd = TruncatedSVD(n_components=2000)
tsvd.fit_transform(combined_data)
cumsum_variance = np.cumsum(np.round(tsvd.explained_variance_ratio_,decimals=4)*100)
cumsum_variance
plt.plot(cumsum_variance)
plt.xlabel("Number of components")
plt.ylabel("Cumulative percentage of explained variance")
plt.title("Plot of explained variance in percentage-tsvd")
column_name = []
distinct_values = []
for col in combined_data.columns:
#     if combined_data[col].dtype == 'int64':
    column_name.append(col)
    distinct_values.append(combined_data[col].nunique())
plt.plot(sorted(distinct_values))
count=0
for col in combined_data.columns:
    if combined_data[col].nunique()<=500:
        combined_data[col]=combined_data[col].astype('category')
#         count=count+1
train_data = combined_data[:len(train_data)]
test_data = combined_data[len(train_data):]
# Let's now define the root mean squared logarthmic error
def rmsle(y_pred,y_act):
    y_pred = scipy.special.inv_boxcox(y_pred,0.1)
    y_act = scipy.special.inv_boxcox(y_act,0.1)
    return np.sqrt(np.mean(np.square(np.log(y_pred+1)-np.log(y_act+1))))

scorer = make_scorer(rmsle)
model = lightgbm.LGBMRegressor()
rmsle_scores = cross_validate(model,train_data,box_cox_trans,scoring=scorer,cv=5)
rmsle_scores
model.fit(train_data,box_cox_trans)
test_pred = model.predict(test_data)

test_pred = scipy.special.inv_boxcox(test_pred,0.1)
dat = pd.DataFrame()
dat["ID"] = test_id
dat['target'] = test_pred
dat.to_csv("first_sub_2.csv",index=False)