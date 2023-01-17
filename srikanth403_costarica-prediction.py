import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np



import warnings

warnings.filterwarnings('ignore')



from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import  OneHotEncoder as ohe

from sklearn.preprocessing import StandardScaler as ss

from sklearn.compose import ColumnTransformer as ct

from sklearn.impute import SimpleImputer

from imblearn.over_sampling import SMOTE, ADASYN

from sklearn.decomposition import PCA

from sklearn.pipeline import Pipeline

from sklearn.pipeline import make_pipeline

from sklearn.model_selection import train_test_split



from sklearn.ensemble import RandomForestClassifier as rf

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import GradientBoostingClassifier as gbm

from xgboost.sklearn import XGBClassifier

import lightgbm as lgb





from sklearn.metrics import accuracy_score

from sklearn.metrics import auc, roc_curve

from sklearn.metrics import f1_score

from sklearn.metrics import average_precision_score

import sklearn.metrics as metrics

from xgboost import plot_importance

from sklearn.model_selection import cross_val_score

from sklearn.metrics import confusion_matrix





from bayes_opt import BayesianOptimization

from skopt import BayesSearchCV

from eli5.sklearn import PermutationImportance





%matplotlib inline
pd.options.display.max_columns = 200





train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

print("A glimpse at the columns of training data:")

train.head()
train.info()
target = train['Target']

target.value_counts()
train.isnull().sum().sort_values(ascending=False).head()
train['v18q1'] = train['v18q1'].fillna(0)

test['v18q1'] = test['v18q1'].fillna(0)

train['rez_esc'] = train['rez_esc'].fillna(0)

test['rez_esc'] = test['rez_esc'].fillna(0)

train['SQBmeaned'] = train['SQBmeaned'].fillna(0)

test['SQBmeaned'] = test['SQBmeaned'].fillna(0)

train['meaneduc'] = train['meaneduc'].fillna(0)

test['meaneduc'] = test['meaneduc'].fillna(0)

train['v2a1'] = train['v2a1'].fillna(0)

test['v2a1'] = test['v2a1'].fillna(0)

train.isnull().sum().sort_values(ascending=False).head()
train.head()

def Target(a):

    for i in a:

        if  i==1:

             Class.append('extreme poverty')

        elif i==2:

             Class.append('moderate poverty')

        elif i==3:

             Class.append('vulnerable households')

        else:

             Class.append('non vulnerable households')

    return Class

                          

lst = list(train['Target'])

Class=[]

list2 = Target(lst)

list2

train['Class']=list2
train.tail()
sns.countplot("Class", data=train)

plt.xticks(size=20,rotation=90)
plt.figure(figsize=(10,10))



plt.subplot(2,2,1)

sns.countplot("Class",hue='v18q', data=train)

plt.xticks(size=10,rotation=90)

plt.title('Presence of Tablet in house hold')



plt.subplot(2,2,2)

sns.countplot("Class",hue="refrig", data=train)

plt.xticks(size=10,rotation=90)

plt.title('Presence of Refrigrator in house hold')



plt.subplot(2,2,3)

sns.countplot("Class",hue="computer", data=train)

plt.xticks(size=10,rotation=90)

plt.title('Presence of "Computer" in house hold')



plt.subplot(2,2,4)

sns.countplot("Class",hue="television", data=train)

plt.xticks(size=10,rotation=90)

plt.title('Presence of "Television" in house hold')



sns.countplot("tamhog",hue="Class", data=train)



plt.title('Size of the house hold')
plt.figure(figsize=(10,10))

plt.title('Presence of Tablet in house hold')



plt.subplot(2,2,1)

sns.countplot("hogar_nin",hue='Class', data=train)

plt.xticks(size=10,rotation=90)

plt.xlabel('Children')

plt.title('People distribution in House holds')



plt.subplot(2,2,2)

sns.countplot("hogar_adul",hue='Class', data=train)

plt.xticks(size=10,rotation=90)

plt.xlabel('Adult')

plt.title('People distribution in House holds')



plt.subplot(2,2,3)

sns.countplot("hogar_mayor",hue='Class', data=train)

plt.xticks(size=10,rotation=90)

plt.xlabel('65+')

plt.title('People distribution in House holds')

df_q = train[['Target', 'epared1', 'epared2', 'epared3', 'etecho1', 'etecho2', 'etecho3', 'eviv1', 'eviv2', 'eviv3']]

df_q.loc[df_q['epared1'] == 1, 'wall'] = 'Bad'

df_q.loc[df_q['epared2'] == 1, 'wall'] = 'Regular'

df_q.loc[df_q['epared3'] == 1, 'wall'] = 'Good'



df_q.loc[df_q['etecho1'] == 1, 'roof'] = 'Bad'

df_q.loc[df_q['etecho2'] == 1, 'roof'] = 'Regular'

df_q.loc[df_q['etecho3'] == 1, 'roof'] = 'Good'



df_q.loc[df_q['eviv1'] == 1, 'floor'] = 'Bad'

df_q.loc[df_q['eviv2'] == 1, 'floor'] = 'Regular'

df_q.loc[df_q['eviv3'] == 1, 'floor'] = 'Good'



df_q = df_q[['Target', 'wall', 'roof', 'floor']]
print("Roof quality")

print("==============================================================================================================================")

df_q.loc[df_q['Target'] == 1, 'Target'] = 'Extreme'

df_q.loc[df_q['Target'] == 2,'Target'] = 'Moderate'

df_q.loc[df_q['Target'] == 3,'Target'] = 'Vulnerable'

df_q.loc[df_q['Target'] == 4,'Target'] = 'Non-Vulnerable'

ax = sns.catplot(x = 'roof', col = 'Target', data = df_q, kind="count", col_order=['Extreme', 'Moderate', 'Vulnerable', 'Non-Vulnerable']).set_titles("{col_name}")

ax.fig.set_size_inches(15,4)

ax.set(ylabel = '')

plt.show()



print("Wall quality")

print("==============================================================================================================================")



ax = sns.catplot(x = 'wall', col = 'Target', data = df_q, kind="count" ,col_order=['Extreme', 'Moderate', 'Vulnerable', 'Non-Vulnerable'], order = ['Bad', 'Regular', 'Good']).set_titles("{col_name}")

ax.fig.set_size_inches(15,4)

ax.set(ylabel = '')

plt.show()



print("Floor quality")

print("==============================================================================================================================")



ax = sns.catplot(x = 'floor', col = 'Target', data = df_q, kind="count", col_order=['Extreme', 'Moderate', 'Vulnerable', 'Non-Vulnerable']).set_titles("{col_name}")

ax.fig.set_size_inches(15,4)

ax.set(ylabel = '')

plt.show()
train.select_dtypes('object').head()
yes_no_map = {'no':0,'yes':1}

train['dependency'] = train['dependency'].replace(yes_no_map).astype(np.float32)

train['edjefe'] = train['edjefe'].replace(yes_no_map).astype(np.float32)

train['edjefa'] = train['edjefa'].replace(yes_no_map).astype(np.float32)

    
train.drop(['Id','idhogar',"dependency","edjefe","edjefa"], inplace = True, axis =1)



test.drop(['Id','idhogar',"dependency","edjefe","edjefa"], inplace = True, axis =1)


train.drop(['Class'],inplace=True,axis=1)
y = train.iloc[:,137]

y.unique()
X = train.iloc[:,1:138]

X.shape
my_imputer = SimpleImputer()

X = my_imputer.fit_transform(X)

scale = ss()

X = scale.fit_transform(X)

pca = PCA(0.95)

X = pca.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(

                                                    X,

                                                    y,

                                                    test_size = 0.2)
 

modelrf = rf()
import time

start = time.time()

modelrf = modelrf.fit(X_train, y_train)

end = time.time()

(end-start)/60
classes = modelrf.predict(X_test)
(classes == y_test).sum()/y_test.size 
modelneigh = KNeighborsClassifier(n_neighbors=4)
start = time.time()

modelneigh = modelneigh.fit(X_train, y_train)

end = time.time()

(end-start)/60
classes = modelneigh.predict(X_test)



classes

(classes == y_test).sum()/y_test.size 
modelgbm=gbm()

start = time.time()

modelgbm = modelgbm.fit(X_train, y_train)

end = time.time()

(end-start)/60

classes = modelgbm.predict(X_test)



classes

(classes == y_test).sum()/y_test.size 
modellgb = lgb.LGBMClassifier(max_depth=-1, learning_rate=0.1, objective='multiclass',

                             random_state=None, silent=True, metric='None', 

                             n_jobs=4, n_estimators=5000, class_weight='balanced',

                             colsample_bytree =  0.93, min_child_samples = 95, num_leaves = 14, subsample = 0.96)
start = time.time()

modellgb = modellgb.fit(X_train, y_train)

end = time.time()

(end-start)/60
classes = modellgb.predict(X_test)



classes

(classes == y_test).sum()/y_test.size 