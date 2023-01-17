import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats

% matplotlib inline
healthtrain = pd.read_csv('./data/health-diagnostics-train.csv', parse_dates=True)
healthtest = pd.read_csv('./data/health-diagnostics-test.csv', parse_dates=True)
healthtrain.head(20)
healthtest.head(20)
healthtrain.shape
healthtrain.describe()
healthtest.shape
healthtrain.dtypes
healthtest.dtypes
healthtrain.iloc[3,3]
type(healthtrain.iloc[3,3])
healthtrain.replace('#NULL!', 0, inplace=True)
healthtest.replace('#NULL!', 0, inplace=True)
col_names=healthtrain.columns

col_names
healthtrain[col_names] = healthtrain[col_names].apply(pd.to_numeric)
healthtest[healthtest.columns]=healthtest[healthtest.columns].apply(pd.to_numeric)
healthtest.dtypes
healthtrain.dtypes
healthtrain.describe()
targetdf=healthtrain[healthtrain['target'] > 0]
targetdf.shape
targetdf
healthtest.plot(kind='box', figsize=(12,8))

targetdf.plot(kind='box', figsize=(12,8))
sns.pairplot(healthtest.dropna())
healthtest.dropna(inplace=True)
healthtrain.dropna(inplace=True)
from sklearn.linear_model import LogisticRegression
from sklearn import model_selection
logreg = LogisticRegression()

feature_cols = ['income', 'fam-history', 'mat-illness-past', 'suppl',
        'mat-illness'] #env. lifestyle 'maternal','meds',, 'env'
X = healthtrain[feature_cols]
y = healthtrain.target

Xtest= healthtest[feature_cols]


logreg.fit(X,y)
pred = logreg.predict(Xtest)

seed = 7
kfold = model_selection.KFold(n_splits=10, random_state=seed)
model = LogisticRegression()#(class_weight={0:0.7, 1:0.3})
scoring = 'roc_auc'
results = model_selection.cross_val_score(model, X, y, cv=kfold, scoring=scoring)

print(results.mean())
print(results.std())
np.count_nonzero(pred==1)
df=pd.DataFrame({"target" : pred})
df.head(20)
df.to_csv('kaggle2.csv', encoding='utf-8')
logreg2 = LogisticRegression(class_weight="balanced")

feature_cols = ['income', 'maternal', 'fam-history', 'mat-illness-past', 'suppl',
       'mat-illness', 'meds', 'env', 'lifestyle']
X = healthtrain[feature_cols]
y = healthtrain.target

Xtest= healthtest[feature_cols]


logreg2.fit(X,y)
pred2 = logreg2.predict(Xtest)

seed = 7
kfold = model_selection.KFold(n_splits=10, random_state=seed)
model = LogisticRegression(class_weight="balanced")
scoring = 'roc_auc'
results = model_selection.cross_val_score(model, X, y, cv=kfold, scoring=scoring)

print(results.mean())
print(results.std())
print(np.count_nonzero(pred2==1))
logreg2 = LogisticRegression(class_weight="balanced")

feature_cols = ['income', 'fam-history', 'mat-illness-past', 'suppl',
        'mat-illness']
X = healthtrain[feature_cols]
y = healthtrain.target

Xtest= healthtest[feature_cols]


logreg2.fit(X,y)
pred2 = logreg2.predict(Xtest)

seed = 7
kfold = model_selection.KFold(n_splits=10, random_state=seed)
model = LogisticRegression(class_weight="balanced")
scoring = 'roc_auc'
results = model_selection.cross_val_score(model, X, y, cv=kfold, scoring=scoring)

print(results.mean())
print(results.std())
print(np.count_nonzero(pred2==1))
df2=pd.DataFrame({"target" : pred2})

df2.to_csv('kaggle5.csv', encoding='utf-8')
logreg3 = LogisticRegression(class_weight={0:0.6, 1:0.4})

feature_cols = ['income', 'maternal', 'fam-history', 'mat-illness-past', 'suppl',
       'mat-illness', 'meds', 'env', 'lifestyle']
X = healthtrain[feature_cols]
y = healthtrain.target

Xtest= healthtest[feature_cols]


logreg3.fit(X,y)
pred3 = logreg3.predict(Xtest)

seed = 7
kfold = model_selection.KFold(n_splits=10, random_state=seed)
model = LogisticRegression(class_weight={0:0.6, 1:0.4})
scoring = 'roc_auc'
results = model_selection.cross_val_score(model, X, y, cv=kfold, scoring=scoring)

print(results.mean())
print(results.std())
print(np.count_nonzero(pred3==1))
logreg4 = LogisticRegression(class_weight={0:0.7, 1:0.3})

feature_cols = ['income', 'maternal', 'fam-history', 'mat-illness-past', 'suppl',
       'mat-illness', 'meds', 'env', 'lifestyle']
X = healthtrain[feature_cols]
y = healthtrain.target

Xtest= healthtest[feature_cols]


logreg4.fit(X,y)
pred4 = logreg4.predict(Xtest)
df3=pd.DataFrame({"target" : pred3})
df4=pd.DataFrame({"target" : pred4})
df4.to_csv('kaggle6.csv', encoding='utf-8')
np.count_nonzero(pred2==0)
np.count_nonzero(pred4==0)
np.count_nonzero(pred3==0)
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import confusion_matrix

#roc_auc_score(logreg4.predict(X),y)
#https://stackoverflow.com/questions/477486/how-to-use-a-decimal-range-step-value

weights=np.linspace(0.6,0.7,10, endpoint=False)
weights
for weight in weights:
    logreg5 = LogisticRegression(class_weight={0:weight, 1:(1-weight)})

    #feature_cols = ['income', 'maternal', 'fam-history', 'mat-illness-past', 'suppl',
      # 'mat-illness', 'meds', 'env', 'lifestyle']
    
    feature_cols = ['income', 'fam-history', 'mat-illness-past', 'suppl',
        'mat-illness']
    X = healthtrain[feature_cols]
    y = healthtrain.target

    Xtest= healthtest[feature_cols]


    logreg5.fit(X,y)
    pred5 = logreg5.predict(Xtest)
    print(weight)
    print(roc_auc_score(logreg5.predict(X),y))
logreg6 = LogisticRegression(class_weight={0:0.68, 1:(1-0.68)})

feature_cols = ['income', 'maternal', 'fam-history', 'mat-illness-past', 'suppl',
       'mat-illness', 'meds', 'env', 'lifestyle']
X = healthtrain[feature_cols]
y = healthtrain.target

Xtest= healthtest[feature_cols]


logreg6.fit(X,y)
pred6 = logreg6.predict(Xtest)
np.count_nonzero(pred6==0)
df6=pd.DataFrame({"target" : pred6})
df6.to_csv('kaggle7.csv', encoding='utf-8')
logreg9 = LogisticRegression(class_weight={0:0.6, 1:0.4})

feature_cols = ['income', 'fam-history', 'mat-illness-past', 'suppl',
        'mat-illness'] #env. lifestyle 'maternal','meds',, 'env'
X = healthtrain[feature_cols]
y = healthtrain.target

Xtest= healthtest[feature_cols]


logreg9.fit(X,y)
pred9 = logreg.predict(Xtest)

print(roc_auc_score(logreg9.predict(X),y))
df7=pd.DataFrame({"target" : pred9})
df7.to_csv('kaggle11.csv', encoding='utf-8')
np.count_nonzero(pred9==0)
