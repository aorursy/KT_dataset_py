# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns 

import imblearn



%matplotlib inline
df = pd.read_csv("/kaggle/input/minor-project-2020/train.csv")
df.head()
df.info()
df.describe()
df['target'].value_counts()
#all hyperparameters were tuned manually/ no grid search was used
# fig, axs = plt.subplots(ncols=3, nrows=3,figsize=(20,10))

# index = 0

# axs = axs.flatten()

lower_bound={}

upper_bound={}

total_var=0

no_of_items=0

for k,v in df.items():

    if(k=='target' or k=='id'):

        continue

    total_var+=df[k].std()

    no_of_items+=1

average_variance=total_var/no_of_items

col1=[]

for k,v in df.items():

    if(k=='target' or k=='id'):

        continue

    if(df[k].std()<=average_variance*2/3):

        col1.append(k)

print(col1)

df=df.drop(col1,axis=1);

# plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=5.0)
from sklearn.decomposition import PCA

from sklearn.model_selection import train_test_split

y=df['target']

X=df.drop(['target','id'],axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=121)

# pca = PCA(n_components=70)

print(y_train.value_counts())

print(y_test.value_counts())

# X_pca = pca.fit_transform(X_train)

PCA_df = pd.DataFrame(data = X_train)

PCA_df.describe()

PCA_df['target']=y_train
fig, axs = plt.subplots(ncols=4, nrows=5, figsize=(20, 10))

index = 0

axs = axs.flatten()

for k,v in PCA_df.items():

    sns.boxplot(x='target',y=k,data=PCA_df, ax=axs[index])

    index += 1

    if(index>=20):

        break

plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=5.0)

# PCA_df.drop('target',axis=1)
# lower_bound0={}

# upper_bound0={}

# upper_bound1={}

# lower_bound1={}

# PCA_df0=PCA_df[PCA_df['target']==0]

# PCA_df1=PCA_df[PCA_df['target']==1]

# for k,v in PCA_df.items():

#     q1 = PCA_df0[k].quantile(0.25)

#     q3 = PCA_df0[k].quantile(0.75)

#     iqr = q3 - q1 #Interquartile range

#     lower_bound0[k] = q1 - (2*iqr)

#     upper_bound0[k] = q3 + (2*iqr)

#     q1 = PCA_df1[k].quantile(0.25)

#     q3 = PCA_df1[k].quantile(0.75)

#     iqr = q3 - q1 #Interquartile range

#     lower_bound1[k] = q1 - (2*iqr)

#     upper_bound1[k] = q3 + (2*iqr)

# for k,v in PCA_df.items():

#     filter1 = ((PCA_df['target']==0) & (PCA_df[k] >= lower_bound0[k]) & (PCA_df[k] <= upper_bound0[k])) | ((PCA_df['target']==1) & (PCA_df[k] >= lower_bound1[k]) & (PCA_df[k] <= upper_bound1[k]))

#     PCA_df=PCA_df.loc[filter1]  

# print(PCA_df.describe())

# PCA_df['target'].value_counts()
from imblearn.over_sampling import ADASYN,SMOTE,BorderlineSMOTE,SVMSMOTE,KMeansSMOTE

# y_train=PCA_df['target']

X_pca=PCA_df.drop('target',axis=1)

oversample = ADASYN()

X_pca, y_train = oversample.fit_resample(X_pca, y_train)

print(X_pca.describe())

print(y_train.value_counts())
from sklearn import preprocessing

scaler = preprocessing.MinMaxScaler()              #Instantiate the scaler

scaled_X = scaler.fit_transform(X_pca) 
from sklearn.linear_model import LogisticRegression

from xgboost import XGBClassifier

from imblearn.ensemble import BalancedBaggingClassifier,BalancedRandomForestClassifier,RUSBoostClassifier,EasyEnsembleClassifier

from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text

from sklearn.model_selection import GridSearchCV

# weights = {0:1.0, 1:300}

clf = LogisticRegression(random_state=1,class_weight='balanced',max_iter=2000,solver='saga').fit(scaled_X, y_train)

# clf = XGBClassifier().fit(scaled_X, y_train)

# clf=BalancedBaggingClassifier(base_estimator=DecisionTreeClassifier(),

#                                  sampling_strategy='auto',

#                                  replacement=False,

#                                  random_state=0).fit(scaled_X,y_train)

# grid_values = {'C':[0.3,1,3,10],'solver':['lbfgs','saga']}

# clf = GridSearchCV(lr, param_grid = grid_values,scoring = 'roc_auc',verbose=1,).fit(scaled_X,y_train)

# print("tuned hpyerparameters :(best parameters) ",clf.best_params_)

predictions=clf.predict_proba(scaled_X)[:,1]

print(predictions)

# print(clf.classes_)

clf.score(scaled_X,y_train)
# from sklearn.metrics import classification_report, confusion_matrix, plot_confusion_matrix

# print("Confusion Matrix: ")

# print(confusion_matrix(y, predictions))
# print("Classification Report: ")

# print(classification_report(y, predictions))
from sklearn.metrics import roc_curve, auc

plt.style.use('seaborn-pastel')



FPR, TPR, _ = roc_curve(y_train, predictions)

ROC_AUC = auc(FPR, TPR)

print (ROC_AUC)



plt.figure(figsize =[11,9])

plt.plot(FPR, TPR, label= 'ROC curve(area = %0.2f)'%ROC_AUC, linewidth= 4)

plt.plot([0,1],[0,1], 'k--', linewidth = 4)

plt.xlim([0.0,1.0])

plt.ylim([0.0,1.05])

plt.xlabel('False Positive Rate', fontsize = 18)

plt.ylabel('True Positive Rate', fontsize = 18)

plt.title('ROC for Training Set', fontsize= 18)

plt.show()
# val_df=X_test.drop('id',axis=1)

# test_df.drop(cols)

# X_test=X_test.drop(col1,axis=1);

scaler = preprocessing.MinMaxScaler()              #Instantiate the scaler\

# pca_val_df=pca.transform(X_test)

scaled_val_df = scaler.fit_transform(X_test) 

predictions=clf.predict_proba(scaled_val_df)[:,1]
from sklearn.metrics import roc_curve, auc

plt.style.use('seaborn-pastel')



FPR, TPR, _ = roc_curve(y_test, predictions)

ROC_AUC = auc(FPR, TPR)

print (ROC_AUC)



plt.figure(figsize =[11,9])

plt.plot(FPR, TPR, label= 'ROC curve(area = %0.2f)'%ROC_AUC, linewidth= 4)

plt.plot([0,1],[0,1], 'k--', linewidth = 4)

plt.xlim([0.0,1.0])

plt.ylim([0.0,1.05])

plt.xlabel('False Positive Rate', fontsize = 18)

plt.ylabel('True Positive Rate', fontsize = 18)

plt.title('ROC for Validation Set', fontsize= 18)

plt.show()
test_df = pd.read_csv("/kaggle/input/minor-project-2020/test.csv")
test_df.head()

id1=test_df['id']

test_df=test_df.drop('id',axis=1)

test_df=test_df.drop(col1,axis=1)

scaler = preprocessing.MinMaxScaler()              #Instantiate the scaler\

# pca_test_df=pca.transform(test_df)

scaled_test_df = scaler.fit_transform(test_df) 
# test_df.describe()
predictions=clf.predict_proba(scaled_test_df)[:,1]
unique_elements, counts_elements = np.unique(predictions, return_counts=True)

print("Frequency of unique values of the said array:")

print(np.asarray((unique_elements, counts_elements)))



# ans=df['id']

test_df['target']=predictions

test_df['id']=id1

# print(test_df.head())

df2=test_df[['id','target']]

df2.head()
df2.to_csv('/kaggle/working/submission1.csv',index=False)