# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from sklearn.preprocessing import LabelEncoder

import matplotlib.pyplot as plt

import seaborn as sns

import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.

train_data = pd.read_csv('../input/Churn_Modelling.csv')

print("amount of data:",train_data.shape[0])

print("amount of feature",train_data.shape[1])
train_data['Exited'].astype(int).plot.hist()
train_data = train_data.drop(['Surname','RowNumber','CustomerId'], axis=1)

train_data
count_miss_val = train_data.isnull().sum()

count_miss_val.head(10)
train_data.dtypes.value_counts()

train_data.select_dtypes(include = ['object']).apply(pd.Series.nunique, axis = 0)

le = LabelEncoder()

le.fit(train_data['Gender'])

train_data['Gender'] = le.transform(train_data['Gender'])

train_data = pd.get_dummies(train_data)

print("amount of training data: %d, amount of features:%d"% (train_data.shape[0],train_data.shape[1]))
train_data['EstimatedSalary'].describe()

train_data['Balance'].describe()

train_data['Age'].describe()
correlations = train_data.corr()['Exited'].sort_values()

correlations
plt.hist(train_data['Age'], edgecolor = 'k', bins = 25)

plt.title('Age of Client')

plt.xlabel('Age (years)')

plt.ylabel('Count')
plt.figure(figsize = (10, 6))

sns.kdeplot(train_data.loc[train_data['Exited'] == 0, 'Age'], label = 'Exited = 0')

sns.kdeplot(train_data.loc[train_data['Exited'] == 1, 'Age'], label = 'Exited = 1')

plt.xlabel('Age')

plt.ylim(0, 0.06)

plt.ylabel('Density')

plt.title('KDE Distribution of Ages')
train_data["Age"].describe()

age_data = train_data[['Exited', 'Age']]

age_data['Age'] = pd.cut(age_data['Age'], bins = np.linspace(18, 93,num = 15))

age_groups = age_data.groupby('Age').mean()

age_groups
plt.figure(figsize = (10,4))

plt.bar(age_groups.index.astype(str), 100 * age_groups['Exited'])

plt.xticks(rotation = 45)

plt.xlabel('Age Group (years)')

plt.ylabel('Exited (%)')

plt.title('Exited by Age Group')
ext_data = train_data[['Exited', 'Age', 'Geography_Germany', 'IsActiveMember', 'Balance']]

ext_data_corrs = ext_data.corr()

plt.figure(figsize = (7,7))

sns.heatmap(ext_data_corrs, cmap = plt.cm.RdYlBu_r, vmin = -0.6, annot = True, vmax = 0.6)

plt.title('Correlation Heatmap')
from sklearn.preprocessing import PolynomialFeatures

ptrain_data = train_data[['IsActiveMember','Gender','Geography_France','Balance','Geography_Germany','Age']]

ptarget = train_data['Exited']

poly_transformer = PolynomialFeatures(degree = 3)

poly_transformer.fit(ptrain_data)

ptrain_data = poly_transformer.transform(ptrain_data)

poly_features_names = poly_transformer.get_feature_names(input_features = ['IsActiveMember','Gender','Geography_France','Balance','Geography_Germany','Age'])

poly_features = pd.DataFrame(ptrain_data, columns = poly_features_names) 

poly_features['TARGET'] = target

poly_corrs = poly_features.corr()['TARGET'].sort_values()

plt.figure(figsize = (10, 50))

poly_corrs.plot(kind='barh')
import featuretools as ft

auto_train_data = train_data.copy()

es = ft.EntitySet(id = 'train_data') 

es = es.entity_from_dataframe(entity_id = 'train_data', dataframe = auto_train_data, index = 'SK_ID_CURR')

auto_train_data, features = ft.dfs(entityset = es,target_entity='train_data',verbose=True)

auto_train_data
from sklearn.preprocessing import MinMaxScaler, Imputer

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler
target = train_data['Exited']

train = train_data.drop(['Exited'], axis = 1)

X_std = StandardScaler().fit_transform(train)

X_std_train,X_std_test,y_train, y_test = train_test_split(X_std, target, test_size = 0.2, random_state = 0)
from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier

from xgboost.sklearn import XGBClassifier

from lightgbm.sklearn import LGBMClassifier

from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc,confusion_matrix

def model_metrics(clf, X_train, X_test, y_train, y_test):    

    y_train_pred = clf.predict(X_train)    

    y_test_pred = clf.predict(X_test)        

    y_train_prob = clf.predict_proba(X_train)[:,1]    

    y_test_prob = clf.predict_proba(X_test)[:,1]

    y_scores = clf.predict_proba(X_test)[:,-1]

    print(classification_report(y_train,y_train_pred, target_names=['non-exited','exited']))

    print('Accurancy:')    

    print('Train set: ','%.4f'%accuracy_score(y_train,y_train_pred), end=' ')    

    print('Test set: ','%.4f'%accuracy_score(y_test,y_test_pred),end=' \n\n')

    acu_curve(y_test,y_scores)

def acu_curve(y_test,y_scores):

    fpr,tpr,threshold = roc_curve(y_test,y_scores) 

    roc_auc = auc(fpr,tpr) 

    plt.figure()

    lw = 2

    plt.figure(figsize=(10,10))

    plt.plot(fpr, tpr, color='darkorange',

             lw=lw, label='ROC curve (area = %0.3f)' % roc_auc) 

    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')

    plt.xlim([0.0, 1.0])

    plt.ylim([0.0, 1.05])

    plt.xlabel('False Positive Rate')

    plt.ylabel('True Positive Rate')

    plt.title('Receiver operating characteristic example')

    plt.legend(loc="lower right")

    plt.show()
lr = LogisticRegression()

lr.fit(X_std_train,y_train)

svm = SVC(gamma='scale',probability=True)

svm.fit(X_std_train,y_train)

y_scores = svm.fit(X_std_train,y_train).decision_function(X_std_test)

tree = DecisionTreeClassifier()

tree.fit(X_std_train,y_train)

xgb = XGBClassifier()

xgb.fit(X_std_train,y_train)

lgbm = LGBMClassifier()

lgbm = lgbm.fit(X_std_train,y_train)
print("Logist regression")

model_metrics(lr,X_std_train,X_std_test,y_train,y_test)

print("Decision tree")

model_metrics(tree,X_std_train,X_std_test,y_train,y_test)

print("Svm")

model_metrics(svm,X_std_train,X_std_test,y_train,y_test)

print("Xgb")

model_metrics(xgb,X_std_train,X_std_test,y_train,y_test)

print("Lgbm")

model_metrics(lgbm,X_std_train,X_std_test,y_train,y_test)
from sklearn.model_selection import GridSearchCV

param_test1 = {

 'max_depth':range(3,10,3),

 'min_child_weight':range(1,6,3)

}

gsearch1 = GridSearchCV(estimator = XGBClassifier(  learning_rate =0.1, n_estimators=100, max_depth=5,

min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8, objective= 'binary:logistic', nthread=4,  scale_pos_weight=1, seed=27), 

 param_grid = param_test1,     scoring='roc_auc', n_jobs=4, iid=False, cv=2)

gsearch1.fit(X_std_train,y_train)
xgc = XGBClassifier(  learning_rate =0.1, n_estimators=100, max_depth=4, min_child_weight=7, reg_alpha=0.001, reg_lambda = 0,

                    gamma=0, subsample=0.8, colsample_bytree=0.8, objective= 'binary:logistic', nthread=4,  scale_pos_weight=1, seed=27)



xgc.fit(X_std_train,y_train)

y_scores = xgc.predict_proba(X_std_test)[:,-1]

fpr,tpr,threshold = roc_curve(y_test,y_scores) 

roc_auc = auc(fpr,tpr) 

print('Train set: ','%.4f'%accuracy_score(y_train, xgc.predict(X_std_train)   ), end=' ')    

print('Test set: ','%.4f'%accuracy_score(y_test, xgc.predict(X_std_test) ),end=' \n\n')

print(roc_auc)
import seaborn as sb

confusionMatrix = confusion_matrix(y_test,xgc.predict(X_std_test))

sb.heatmap(confusionMatrix,annot=True,fmt='d')