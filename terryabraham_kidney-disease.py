# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_val_score, cross_val_predict

from sklearn.svm import SVC

from sklearn.metrics import classification_report, confusion_matrix

from sklearn.metrics import accuracy_score
df = pd.read_csv('/kaggle/input/ckdisease/kidney_disease.csv')

df.head()
# Map text to 1/0 and do some cleaning

df[['htn','dm','cad','pe','ane']] = df[['htn','dm','cad','pe','ane']].replace(to_replace={'yes':1,'no':0})

df[['rbc','pc']] = df[['rbc','pc']].replace(to_replace={'abnormal':1,'normal':0})

df[['pcc','ba']] = df[['pcc','ba']].replace(to_replace={'present':1,'notpresent':0})

df[['appet']] = df[['appet']].replace(to_replace={'good':1,'poor':0,'no':np.nan})

df['classification'] = df['classification'].replace(to_replace={'ckd':1.0,'ckd\t':1.0,'notckd':0.0,'no':0.0})

df.rename(columns={'classification':'class'},inplace=True)
# Further cleaning

df['pe'] = df['pe'].replace(to_replace='good',value=0) # Not having pedal edema is good

df['appet'] = df['appet'].replace(to_replace='no',value=0)

df['cad'] = df['cad'].replace(to_replace='\tno',value=0)

df['dm'] = df['dm'].replace(to_replace={'\tno':0,'\tyes':1,' yes':1, '':np.nan})

df.drop('id',axis=1,inplace=True)
df.head()
df2 = df.dropna(axis=0)

df2['class'].value_counts()
df2.apply(pd.to_numeric)

df2.dtypes
for i in range(0,df2.shape[1]):

     if df2.dtypes[i]=='object':

            print(df2.columns[i],'<--- having object datatype')



            df2['pcv'] = df2.pcv.astype(float) 

            df2['wc'] = df2.wc.astype(float)

            df2['rc'] = df2.rc.astype(float)

            df2['dm'] = df2.dm.astype(float)
df2.dtypes
df2['class']=df2['class'].astype(int)

X = df2.drop('class', axis=1)

X = StandardScaler().fit_transform(X)

y = df2['class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0,stratify= df2['class'])
model = SVC()



parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],

                     'C': [1, 10, 100, 1000]},

                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

grid = GridSearchCV(estimator=model, param_grid=parameters, cv=5)

grid.fit(X_train, y_train)
roc_auc = np.around(np.mean(cross_val_score(grid, X_test, y_test, cv=5, scoring='roc_auc')), decimals=4)

print('Score: {}'.format(roc_auc))
model1= RandomForestClassifier(n_estimators=1000)

tuned_parameters = [{'n_estimators':[7,8,9,10,11,12,13,14,15,16],'max_depth':[2,3,4,5,6,None],

                     'class_weight':[None,{0: 0.33,1:0.67},'balanced'],'random_state':[42]}]

clf = GridSearchCV(model1, tuned_parameters, cv=10,scoring='roc_auc')

clf.fit(X_train, y_train)

score1= np.mean(cross_val_score(model1, X_test, y_test, cv=5, scoring='roc_auc'))

np.around(score1, decimals=4)
df2 = df.dropna(axis=0)

no_na = df2.index.tolist()

some_na = df.drop(no_na).apply(lambda x: pd.to_numeric(x,errors='coerce'))

some_na = some_na.fillna(0) # Fill up all Nan by zero.

clf_best= clf.best_estimator_

X_test = some_na.iloc[:,:-1]

y_test = some_na['class']

y_true = y_test

lr_pred = clf_best.predict(X_test)

print(classification_report(y_true, lr_pred))



confusion = confusion_matrix(y_test, lr_pred)

print('Confusion Matrix:')

print(confusion)

score2=accuracy_score(y_true, lr_pred)

print('Score: %3f' %score2 )
model2=KNeighborsClassifier()

model2.fit(X_train,y_train)

score3= np.around(np.mean(cross_val_score(model2, X_test, y_test, cv=5, scoring='roc_auc')),decimals=4)

print('Score : {}'.format(score3))
model3=LogisticRegression()

parameters={'C':[0.001,0.01,0.1,1,10,100]}

grid = GridSearchCV(estimator=model3, param_grid=parameters, cv=5)

grid.fit(X_train, y_train)
score4= np.around(np.mean(cross_val_score(model3, X_test, y_test, cv=5, scoring='roc_auc')),decimals=4)

print('Score : {}'.format(score4))
names=[]

scores=[]

names.extend(['RF','KNN','LR'])

scores.extend([score2,score3,score4])

alg=pd.DataFrame({'Score':scores},index=names)

print('Most Accurate : \n{}'.format(alg.loc[alg['Score'].idxmax()]))