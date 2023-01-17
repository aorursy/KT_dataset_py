import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import datetime

import gc

import numpy as np

import os

import pandas as pd

from tqdm import tqdm

from sklearn.ensemble import RandomForestClassifier as rf

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import GradientBoostingClassifier as gbm

from xgboost.sklearn import XGBClassifier

import lightgbm as lgb





from sklearn.preprocessing import LabelEncoder



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import matplotlib

import matplotlib.pyplot as plt



%matplotlib inline 
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
train.head()
train.info() 
import seaborn as sns

barplot = sns.barplot(x='rooms',y='Target',data=train,palette="Set1") 

barplot.set(xlabel='rooms', ylabel='Target') # to set x and y labels 
sns.heatmap(train.corr(),cmap="coolwarm",annot=True)
sns.boxplot(x='Target', y='escolari', data = train);

plt.title('Years of schooling per household poverty level.')
plt.figure(figsize=(10,10))



plt.subplot(2,2,1)

sns.countplot("Target",hue='v18q', data=train)

plt.xticks(size=10,rotation=90)

plt.title('Presence of Tablet in house hold')



plt.subplot(2,2,2)

sns.countplot("Target",hue="refrig", data=train)

plt.xticks(size=10,rotation=90)

plt.title('Presence of Refrigrator in house hold')



plt.subplot(2,2,3)

sns.countplot("Target",hue="computer", data=train)

plt.xticks(size=10,rotation=90)

plt.title('Presence of "Computer" in house hold')



plt.subplot(2,2,4)

sns.countplot("Target",hue="television", data=train)

plt.xticks(size=10,rotation=90)

plt.title('Presence of "Television" in house hold')
train.dtypes
train.select_dtypes('object').head()
yes_no_map = {'no':0,'yes':1}

train['dependency'] = train['dependency'].replace(yes_no_map).astype(np.float32)

train['edjefe'] = train['edjefe'].replace(yes_no_map).astype(np.float32)

train['edjefa'] = train['edjefa'].replace(yes_no_map).astype(np.float32)
train[["dependency","edjefe","edjefa"]].describe()
 # Number of missing in each column

missing = pd.DataFrame(train.isnull().sum()).rename(columns = {0: 'total'})



# Create a percentage missing

missing['percent'] = missing['total'] / len(train)



missing.sort_values('percent', ascending = False).head(10)
from sklearn import preprocessing

from sklearn.preprocessing import LabelEncoder

for f in train.columns:

    if train[f].dtype=='object':

        print(f)

        lbl=preprocessing.LabelEncoder()

        lbl.fit(list(train[f].values)+list(test[f].values))

        train[f]=lbl.transform(list(train[f].values))

        test[f]=lbl.transform(list(test[f].values))
y = train.iloc[:,137]

y.unique()
X = train.iloc[:,1:138]

X.shape
from sklearn.impute import SimpleImputer

from sklearn.preprocessing import StandardScaler as ss

from sklearn.decomposition import PCA

my_imputer = SimpleImputer()

X = my_imputer.fit_transform(X)

scale = ss()

X = scale.fit_transform(X)

pca = PCA(0.95)

X = pca.fit_transform(X)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(

                                                    X,

                                                    y,

                                                    test_size = 0.2)
import time

from sklearn.ensemble import RandomForestClassifier



start = time.time()



rf = RandomForestClassifier(criterion='gini', 

                             n_estimators=700,

                             min_samples_split=10,

                             min_samples_leaf=1,

                             max_features='auto',

                             oob_score=True,

                             random_state=1,

                             n_jobs=-1)

rf.fit(X_test,y_test)



end = time.time()

print("RandomForestClassifier took {:.2f}s".format(end - start))
rf.score(X_test, y_test)
predictions = rf.predict(X_test)
predictions
from sklearn.metrics import confusion_matrix

f  = confusion_matrix(y_test,predictions)

f
fig, (ax1) = plt.subplots(ncols=1, figsize=(5,5))

sns.heatmap(f, 

            xticklabels=['No', 'Yes'],

            yticklabels=['No', 'Yes'],

            annot=True,ax=ax1,

            linewidths=.2,linecolor="Darkblue", cmap="Blues")

plt.title('Confusion Matrix', fontsize=14)

plt.show()
modelneigh = KNeighborsClassifier(n_neighbors=4)
start = time.time()

modelneigh = modelneigh.fit(X_train, y_train)

end = time.time()

(end-start)/60
predictions = modelneigh.predict(X_test)
predictions
(predictions == y_test).sum()/y_test.size
from sklearn.metrics import confusion_matrix

f  = confusion_matrix(y_test,predictions)

f
fig, (ax1) = plt.subplots(ncols=1, figsize=(5,5))

sns.heatmap(f, 

            xticklabels=['No', 'Yes'],

            yticklabels=['No', 'Yes'],

            annot=True,ax=ax1,

            linewidths=.2,linecolor="Darkblue", cmap="Blues")

plt.title('Confusion Matrix', fontsize=14)

plt.show()
modelgbm=gbm()
import time

start = time.time()

modelgbm = modelgbm.fit(X_train, y_train)

end = time.time()

(end-start)/60
predictions = modelgbm.predict(X_test)



predictions
(predictions == y_test).sum()/y_test.size
from sklearn.metrics import confusion_matrix

f  = confusion_matrix(y_test,predictions)

f
fig, (ax1) = plt.subplots(ncols=1, figsize=(5,5))

sns.heatmap(f, 

            xticklabels=['No', 'Yes'],

            yticklabels=['No', 'Yes'],

            annot=True,ax=ax1,

            linewidths=.2,linecolor="Darkblue", cmap="Blues")

plt.title('Confusion Matrix', fontsize=14)

plt.show()
modelxgb=XGBClassifier()
import time

start = time.time()

modelxgb = modelxgb.fit(X_train, y_train)

end = time.time()

(end-start)/60
prdictions = modelxgb.predict(X_test)



predictions
(predictions == y_test).sum()/y_test.size
from sklearn.metrics import confusion_matrix

f  = confusion_matrix(y_test,predictions)

f
fig, (ax1) = plt.subplots(ncols=1, figsize=(5,5))

sns.heatmap(f, 

            xticklabels=['No', 'Yes'],

            yticklabels=['No', 'Yes'],

            annot=True,ax=ax1,

            linewidths=.2,linecolor="Darkblue", cmap="Blues")

plt.title('Confusion Matrix', fontsize=14)

plt.show()
modellgb = lgb.LGBMClassifier(max_depth=-1, learning_rate=0.1, objective='multiclass',

                             random_state=None, silent=True, metric='None', 

                             n_jobs=4, n_estimators=5000, class_weight='balanced',

                             colsample_bytree =  0.93, min_child_samples = 95, num_leaves = 14, subsample = 0.96)
import time

start = time.time()

modellgb = modellgb.fit(X_train, y_train)

end = time.time()

(end-start)/60
predictions = modellgb.predict(X_test)



predictions
(predictions == y_test).sum()/y_test.size
from sklearn.metrics import confusion_matrix

f  = confusion_matrix(y_test,predictions)

f
fig, (ax1) = plt.subplots(ncols=1, figsize=(5,5))

sns.heatmap(f, 

            xticklabels=['No', 'Yes'],

            yticklabels=['No', 'Yes'],

            annot=True,ax=ax1,

            linewidths=.2,linecolor="Darkblue", cmap="Blues")

plt.title('Confusion Matrix', fontsize=14)

plt.show()
from sklearn.ensemble import ExtraTreesClassifier

extc = ExtraTreesClassifier(n_estimators=580,max_features= 128,criterion= 'entropy',min_samples_split= 3,

                            max_depth= 30, min_samples_leaf= 8)      

start = time.time()

extc = extc.fit(X_train,y_train)

end = time.time()

(end-start)/60
predictions = extc.predict(X_test)



predictions
(predictions == y_test).sum()/y_test.size
from sklearn.metrics import confusion_matrix

f  = confusion_matrix(y_test,predictions)

f
fig, (ax1) = plt.subplots(ncols=1, figsize=(5,5))

sns.heatmap(f, 

            xticklabels=['No', 'Yes'],

            yticklabels=['No', 'Yes'],

            annot=True,ax=ax1,

            linewidths=.2,linecolor="Darkblue", cmap="Blues")

plt.title('Confusion Matrix', fontsize=14)

plt.show()
from sklearn.ensemble  import  RandomForestClassifier as rf

from bayes_opt import BayesianOptimization

from skopt import BayesSearchCV as BayesSCV

bayes_tuner = BayesSCV(RandomForestClassifier(n_jobs = 2),



    #  Estimator parameters to be change/tune

    {

        'n_estimators': (100, 500),           

        'criterion': ['gini', 'entropy'],    

        'max_depth': (4, 100),               

        'max_features' : (10,64),             

        'min_weight_fraction_leaf' : (0,0.5, 'uniform')   

    },



    # 2.13

    n_iter=32,            

    cv = 3               

)
bayes_tuner.fit(X_train, y_train)
opt.best_params_
opt.best_score_
opt.score(X_test, y_test)
opt.cv_results_['params']