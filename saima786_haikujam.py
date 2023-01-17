import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from scipy import stats

import numpy as np

from sklearn.feature_selection import mutual_info_classif

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score,roc_auc_score

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import RepeatedStratifiedKFold

from xgboost import XGBClassifier

sns.set()

import warnings

warnings.filterwarnings('ignore')

%matplotlib inline
train_data=pd.read_csv("/kaggle/input/haikujam/BuyAffinity_Train.txt",delimiter='\t')

test_data=pd.read_csv("/kaggle/input/haikujam/BuyAffinity_Test.txt",delimiter='\t')

train_data.head()
train_data.info(verbose=True)
train_data.dtypes
train_data_new=train_data.drop(['F15','F16'],axis=1,inplace=True)

test_data_new=test_data.drop(['F15','F16'],axis=1,inplace=True)
train_data.describe().T
Not_buy = train_data[train_data['C']==0]['C'].count()

buy = train_data[train_data['C']==1]['C'].count()

buy_percent = buy/(buy+Not_buy)

Not_buy_percent = Not_buy/(buy+Not_buy)

print('Total Number of Non-buy items:',Not_buy)

print('Total Number of buy items:',buy)

print('Percentage of Non-buy items:',Not_buy_percent*100,'%')

print('Percentage of buy items:',buy_percent*100,'%')
print(train_data.C.value_counts())

p=train_data.C.value_counts().plot(kind="bar")
print(train_data.isnull().sum())
plt.figure(figsize=(20,15))  # on this line I just set the size of figure to 12 by 10.

p=sns.heatmap(train_data.corr(), annot=True,cmap ='RdYlGn') 
train_data_new=train_data.drop(['F17','F18'],axis=1,inplace=True)

test_data_new=test_data.drop(['Index','F17','F18'],axis=1,inplace=True)
col_names = ['F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9', 'F10', 'F11',

       'F12', 'F13', 'F14', 'F19', 'F20', 'F21',

       'F22']

fig, ax = plt.subplots(len(col_names), figsize=(15,40))



for i, col_val in enumerate(col_names):



    sns.boxplot(y=train_data[col_val], ax=ax[i])

    ax[i].set_title('Box plot - {}'.format(col_val), fontsize=10)

    ax[i].set_xlabel(col_val, fontsize=12)



plt.show()
p = train_data.hist(figsize = (20,20))
def percentile_based_outlier(data, threshold=95):

    diff = (100 - threshold) / 2

    minval, maxval = np.percentile(data, [diff, 100 - diff])

    return (data < minval) | (data > maxval)



col_names = ['F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9', 'F10', 'F11',

       'F12', 'F13', 'F14', 'F19', 'F20', 'F21',

       'F22']



fig, ax = plt.subplots(len(col_names), figsize=(8,40))



for i, col_val in enumerate(col_names):

    x = train_data[col_val][:]

    sns.distplot(x, ax=ax[i], rug=True, hist=False)

    outliers = x[percentile_based_outlier(x)]

    ax[i].plot(outliers, np.zeros_like(outliers), 'ro', clip_on=False)

    ax[i].set_xlabel(col_val, fontsize=8)



plt.show()
q = test_data.hist(figsize = (20,20))
z = np.abs(stats.zscore(train_data))

print(z)
X_z = train_data[(z < 3).all(axis=1)]
X_z.shape

X_z.head()
z_test = np.abs(stats.zscore(test_data))

print(z_test)

z_test_data = test_data[(z_test < 3).all(axis=1)]
z_test_data.shape,X_z.shape
X=X_z.iloc[:,1:-1]

y=X_z['C']

X.columns

X.shape,y.shape
X_test=z_test_data.iloc[:,:]

X.shape,X_test.shape
from sklearn.ensemble import ExtraTreesClassifier
model=ExtraTreesClassifier()

model.fit(X,y)
print(model.feature_importances_)

ranked_features=pd.Series(model.feature_importances_,index=X.columns)
ranked_features.nlargest(18).plot(kind='barh')

plt.show()
# fit scaler on training data

norm = StandardScaler().fit(X)



# transform training data

X_norm = norm.transform(X)



X_normm=pd.DataFrame(X_norm,columns=X.columns)

# transform testing dataabs

#X_test_norm = norm.transform(X_test)
X_tr1,X_val1,y_tr1,y_val1=train_test_split(X_normm,y,test_size=0.33,stratify=y)
classifier=LogisticRegression()

classifier.fit(X_tr1,y_tr1)

y_pred=classifier.predict(X_val1)

y_pred1=classifier.predict_proba(X_val1)

print("Accuracy_score: {}".format(accuracy_score(y_val1,y_pred)))

print("roc_auc_score: {}".format(roc_auc_score(y_val1,y_pred1[:,1])))
print(classification_report(y_val1, y_pred))
X_tr,X_val,y_tr,y_val=train_test_split(X,y,test_size=0.33,stratify=y)
rfc = RandomForestClassifier(max_depth=5,max_features=7)

rfc.fit(X_tr, y_tr)
pred_rfc = rfc.predict(X_val)

print(confusion_matrix(y_val, pred_rfc))

print(classification_report(y_val, pred_rfc))

print(accuracy_score(y_val, pred_rfc))

roc=roc_auc_score(y_val, rfc.predict_proba(X_val)[:,1])

print("roc_auc_score:",roc)
y_pred_test1=classifier.predict(X_test)

res1=pd.DataFrame(y_pred_test1,columns=['Predicted_Value'])
res1.head(10)
res1.to_csv("PredictedResult.csv", header=True)
from matplotlib.ticker import FuncFormatter,MaxNLocator

fig,ax=plt.subplots()

ax=fig.add_axes([0,0,1,1])

ax.grid(True)

ax.xaxis.set_major_locator(plt.MaxNLocator(30))

ax.set_ylabel('PredictedValue')

ax.plot(X_test.index[0:45],y_pred_test1[0:45])

plt.xticks(rotation='vertical')
from numpy import mean

model = XGBClassifier()



model.fit(X_tr,y_tr)

# define evaluation procedure

cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)

# evaluate model

scores = cross_val_score(model, X_tr, y_tr, scoring='roc_auc', cv=cv, n_jobs=-1)

# summarize performance

y_test_pred = model.predict(X_val)

accuracy = accuracy_score(y_test_pred, y_val)

print(accuracy)

print('Mean ROC AUC: %.5f' % mean(scores))
print(classification_report(y_val,y_test_pred))