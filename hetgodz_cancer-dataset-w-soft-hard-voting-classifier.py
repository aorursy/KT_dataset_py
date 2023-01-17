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
import pandas as pd

df = pd.read_csv("../input/cancer.csv")
import numpy as np 

import pandas as pd 

%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.decomposition import PCA

from sklearn.metrics import confusion_matrix

from sklearn.model_selection import cross_val_score

from sklearn.svm import SVC

from sklearn.ensemble import AdaBoostClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import precision_score, recall_score

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score

import warnings

warnings.filterwarnings('ignore')
df.columns
df=df.drop(['id','Unnamed: 32'], axis=1)
df.head()
df.shape


df.columns
df.describe()
df.isnull().any()
M,B=df['diagnosis'].value_counts()

print('No. of malignant cases: ' ,M)

print('No. of benign cases: ' ,B)

sns.catplot(x='diagnosis',kind='count',data=df, palette="husl");
M=df.loc[df['diagnosis']=='M',:]

M.head()
B=df.loc[df['diagnosis']=='B',:]

B.head()
M=M.drop(['diagnosis'],axis=1)

B=B.drop(['diagnosis'],axis=1)
plt.subplots(figsize=(15,45))

sns.set_style('darkgrid')

plt.subplots_adjust (hspace=0.4, wspace=0.2)

i=0

for col in M.columns:

    i+=1

    plt.subplot(10,3,i)

   

    sns.distplot(M[col],color='m',label='Malignant',hist=False, rug=True)

    sns.distplot(B[col],color='b',label='Benign',hist=False, rug=True)

    plt.legend(loc='right-upper')

    plt.title(col)
sns.set(style="white")

fig,ax=plt.subplots(figsize=(16,16))

corr=df.corr()

mask = np.zeros_like(corr)

mask[np.triu_indices_from(mask)] = True

cmap = sns.diverging_palette(220, 10, as_cmap=True)

sns.heatmap(corr,vmin=-1,vmax=1,fmt = ".1f",annot=True,cmap="coolwarm", mask=mask, square=True)
df['diagnosis']=df['diagnosis'].map({'M':1,'B':0})

df=df.drop(['area_mean', 'perimeter_mean', 'radius_worst', 'area_worst', 'perimeter_worst','texture_worst','concavity_mean','perimeter_se', 'area_se'],axis=1)

print(df.shape)
y=df['diagnosis'].values

X=df.drop(['diagnosis'],axis=1).values
X_train,X_test,y_train, y_test=train_test_split(X,y,test_size=0.3, random_state=8)
scaler=StandardScaler()

X_train_std=scaler.fit_transform(X_train)

X_test_std=scaler.transform(X_test)

pca=PCA().fit(X_train_std)

plt.plot(np.cumsum(pca.explained_variance_ratio_))

plt.xlabel('number of components')

plt.ylabel('cumulative explained variance')

plt.xlim(0,22,2)
pca=PCA(n_components=10)

pca.fit(X_train_std)

X_train_pca=pca.transform(X_train_std)

X_test_pca=pca.transform(X_test_std)

print(X_train_pca.shape)

print(X_test_pca.shape)
logreg=LogisticRegression(random_state=1)

score = np.mean(cross_val_score(logreg,  X_train_pca, y_train, scoring='accuracy'))

p_scores = np.mean(cross_val_score(logreg,  X_train_pca, y_train, scoring='precision'))

r_scores = np.mean(cross_val_score(logreg,  X_train_pca, y_train, scoring='recall'))

print("Accuracy: %s" % '{:.2%}'.format(score))

print ('Precision : %s' %'{:.2%}' .format(p_scores))

print ('Recall score: %s' % '{:.2%}'.format(r_scores))
knn=KNeighborsClassifier()

scores = np.mean(cross_val_score(knn,  X_train_pca, y_train, scoring='accuracy'))

p_scores = np.mean(cross_val_score(knn,  X_train_pca, y_train, scoring='precision'))

r_scores = np.mean(cross_val_score(knn,  X_train_pca, y_train, scoring='recall'))

print("Accuracy: %s" % '{:.2%}'.format(score))

print ('Precision : %s' %'{:.2%}' .format(p_scores))

print ('Recall score: %s' % '{:.2%}'.format(r_scores))
X1_train,X1_test,y1_train,y1_test= train_test_split(X_train_pca, y_train,test_size=0.3,random_state=21)

knn.fit(X1_train,y1_train)

y_pred=knn.predict(X1_test)

con=confusion_matrix(y1_test,y_pred)

print('Confusion matrix:')

print(con)
knn=KNeighborsClassifier()

param_grid = {"n_neighbors": np.arange(1,50)}

knn_cv = GridSearchCV(estimator = knn, param_grid = param_grid, 

                          cv = 3, n_jobs = -1, verbose = 0)

knn_cv.fit( X_train_pca, y_train)

print(knn_cv.best_params_)
knn_cv=KNeighborsClassifier(n_neighbors= 9)

score_knn_cv = np.mean(cross_val_score(knn_cv,  X_test_pca, y_test, scoring='accuracy'))

p_score_knn_cv = np.mean(cross_val_score(knn_cv,  X_test_pca, y_test, scoring='precision'))

r_score_knn_cv = np.mean(cross_val_score(knn_cv,  X_test_pca, y_test, scoring='recall'))

print("Accuracy for knn_cv: %s" % '{:.2%}'.format(score_knn_cv))

print ('Precision for knn_cv: %s' %'{:.2%}' .format(p_score_knn_cv))

print ('Recall score for knn_cv: %s' % '{:.2%}'.format(r_score_knn_cv))



score_knn = np.mean(cross_val_score(knn,  X_test_pca, y_test, scoring='accuracy'))

p_score_knn= np.mean(cross_val_score(knn,  X_test_pca, y_test, scoring='precision'))

r_score_knn = np.mean(cross_val_score(knn,  X_test_pca, y_test, scoring='recall'))

print("Accuracy for knn: %s" % '{:.2%}'.format(score_knn))

print ('Precision for knn: %s' %'{:.2%}' .format(p_score_knn))

print ('Recall score for knn: %s' % '{:.2%}'.format(r_score_knn))
svc=SVC(random_state=1)

scores_svc = np.mean(cross_val_score(svc,  X_train_pca, y_train, scoring='accuracy'))

p_score_svc = np.mean(cross_val_score(svc,  X_test_pca, y_test, scoring='precision'))

r_score_svc = np.mean(cross_val_score(svc,  X_test_pca, y_test, scoring='recall'))

print("Accuracy for svc: %s" % '{:.2%}'.format(scores_svc))

print ('Precision for svc: %s' %'{:.2%}' .format(p_score_svc))

print ('Recall score for svc: %s' % '{:.2%}'.format(r_score_svc))
svc=SVC(random_state=1)

param_grid = {"C": [0.001,0.1,1,10], 'degree':[1,3,10]}

svc_cv=GridSearchCV(svc,param_grid=param_grid,cv = 3, n_jobs = -1, verbose = 0)

svc_cv.fit(X_train_pca, y_train)

svc_cv.best_params_
dt=DecisionTreeClassifier(random_state=7)

score_dt = np.mean(cross_val_score(dt,  X_train_pca, y_train, scoring='accuracy'))

p_score_dt = np.mean(cross_val_score(dt,  X_train_pca, y_train, scoring='precision'))

r_score_dt = np.mean(cross_val_score(dt,  X_train_pca, y_train, scoring='recall'))

print("Accuracy for Decision Tree: %s" % '{:.2%}'.format(score_dt))

print ('Precision Decision Tree: %s' %'{:.2%}' .format(p_score_dt))

print ('Recall score Decision Tree: %s' % '{:.2%}'.format(r_score_dt))
rf=RandomForestClassifier(random_state=21)

score_rf = np.mean(cross_val_score(rf,  X_train_pca, y_train, scoring='accuracy'))

p_score_rf = np.mean(cross_val_score(rf,  X_train_pca, y_train, scoring='precision'))

r_score_rf = np.mean(cross_val_score(rf,  X_train_pca, y_train, scoring='recall'))

print("Accuracy for RandomForest: %s" % '{:.2%}'.format(score_rf))

print ('Precision RandomForest:: %s' %'{:.2%}' .format(p_score_rf))

print ('Recall score RandomForest:: %s' % '{:.2%}'.format(r_score_rf))
param_grid = {'max_depth': [80, 90, 100, 110],

              'max_features': [2, 3],

              'min_samples_leaf': [3, 4, 5],

              'min_samples_split': [8, 10, 12],

              'n_estimators': [100, 200, 300, 1000]}



rf = RandomForestClassifier(random_state=21)



rf_cv = GridSearchCV(estimator = rf, param_grid = param_grid, 

                          cv = 3, n_jobs = -1, verbose = 0)

rf_cv.fit(X_train_pca, y_train)

print(rf_cv.best_params_)

score=rf_cv.best_score_

print("Accuracy: %s" % '{:.2%}'.format(score))
rf_cv=RandomForestClassifier(random_state=21,max_depth= 80, max_features= 3,min_samples_leaf= 5, 

                          min_samples_split=8,n_estimators= 100)

score_rf_cv = np.mean(cross_val_score(rf_cv,  X_test_pca, y_test, scoring='accuracy'))

print("Accuracy for rf_cv: %s" % '{:.2%}'.format(score_rf_cv))



score_rf = np.mean(cross_val_score(rf,  X_test_pca, y_test, scoring='accuracy'))

print("Accuracy for rf: %s" % '{:.2%}'.format(score_rf))
# hard
logreg=LogisticRegression(random_state=1)

knn_cv=KNeighborsClassifier(n_neighbors= 9)

svc=SVC(random_state=1)

dt=DecisionTreeClassifier(random_state=7)

rf_cv=RandomForestClassifier(random_state=21,max_depth= 80, max_features= 3,min_samples_leaf= 5, 

                          min_samples_split=8,n_estimators= 100)



from sklearn.ensemble import VotingClassifier

voting_clf = VotingClassifier(estimators=[('logreg',logreg),('knn_cv', knn_cv), ('rf_cv', rf_cv), ('dt',dt), ('svc', svc)], voting='hard')

score = np.mean(cross_val_score(voting_clf,  X_train_pca, y_train, scoring='accuracy'))

print("Accuracy : %s" % '{:.2%}'.format(score))
# soft
svc=SVC(random_state=1,probability=True)



voting_clf = VotingClassifier(estimators=[('logreg',logreg), ('rf_cv', rf_cv),  ('svc', svc)], voting='soft')

score = np.mean(cross_val_score(voting_clf,  X_train_pca, y_train, scoring='accuracy'))

p_score = np.mean(cross_val_score(voting_clf,  X_train_pca, y_train, scoring='precision'))

r_score = np.mean(cross_val_score(voting_clf,  X_train_pca, y_train, scoring='recall'))

print("Accuracy : %s" % '{:.2%}'.format(score))

print ('Precision : %s' %'{:.2%}' .format(p_score))

print ('Recall :: %s' % '{:.2%}'.format(r_score))