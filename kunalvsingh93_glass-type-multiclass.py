import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df=pd.read_csv('../input/glass-multiclass/glass_multiclass.csv')
df.head()
df['Type'].value_counts()
df.shape
df.info()
for i in df.select_dtypes(['int64','float64']):
    sns.boxplot(df[i])
    plt.show()
df.skew()
df.columns
l=['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe']
import scipy.stats as st
for i in l:
    df[i]=st.boxcox(df[i]+1)[0]
df.skew()
for i in df.select_dtypes(['int64','float64']):
    sns.boxplot(y=df[i])
    plt.show()
X=df.drop(['Type'],axis=1)
y=df['Type']
X.shape,y.shape
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
lr=LogisticRegression()
knn=KNeighborsClassifier()
rf=RandomForestClassifier()
svc=SVC()
models=[]
models.append(('MVLC',lr))
models.append(('KNNC',knn))
models.append(('RFC',rf))
models.append(('SVC',svc))
from sklearn.metrics import classification_report

results=[]
names=[]
y_pred=[]
for name,model in models:
    model.fit(X_train,y_train)
    y_pred= model.predict(X_test)
    print(classification_report(y_test,y_pred))
    kfold=KFold(shuffle=True,n_splits=3,random_state=0)
    cv_results=cross_val_score(model,X_train,y_train,cv=kfold,scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    print("%s: %f (%f)"%(name,np.mean(cv_results)*100,np.var(cv_results,ddof=1)))
from imblearn.over_sampling import SMOTE
smote = SMOTE('auto')
X_sm, y_sm = smote.fit_sample(X_train,y_train)
print(X_sm.shape, y_sm.shape)
pd.Series(y_sm).value_counts()
sns.countplot(x='Type',data=df)
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
classifier.fit(X_sm, y_sm)
y_pred= classifier.predict(X_test)
from sklearn import  metrics  
print(metrics.accuracy_score(y_test, y_pred))
print(metrics.f1_score(y_test, y_pred, average='weighted'))
from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(random_state = 42)
modelF = forest.fit(X_sm, y_sm)
y_predF = modelF.predict(X_test)
print(metrics.accuracy_score(y_test, y_predF))
print(metrics.f1_score(y_test, y_predF, average='weighted'))
from sklearn.model_selection import GridSearchCV
n_estimators = [100, 300, 500]
max_depth = [2, 5, 8, 15]
min_samples_split = [2, 5, 10]
min_samples_leaf = [1, 2, 5] 

hyperF = dict(n_estimators = n_estimators, max_depth = max_depth,  
              min_samples_split = min_samples_split, 
             min_samples_leaf = min_samples_leaf)

gridF = GridSearchCV(forest, hyperF, cv = 3, verbose = 1, 
                      n_jobs = -1)
bestF = gridF.fit(X_sm, y_sm)
print (gridF.best_params_)
best_predF = bestF.predict(X_test)
print(metrics.accuracy_score(y_test, best_predF))
print(metrics.f1_score(y_test, best_predF, average='weighted'))