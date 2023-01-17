# Data Manipulation 

import numpy as np

import pandas as pd



# Visualization 

import matplotlib.pyplot as plt

import seaborn as sns



# Feature Selection and Encoding

from sklearn.feature_selection import RFE, RFECV

from sklearn.decomposition import PCA

from sklearn.preprocessing import OneHotEncoder, LabelEncoder,StandardScaler



# Machine learning 

from sklearn import model_selection,preprocessing, metrics, linear_model

from sklearn.model_selection import StratifiedKFold

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier,BaggingClassifier,AdaBoostClassifier,GradientBoostingClassifier



# Grid and Random Search

import scipy.stats as st

from scipy.stats import randint as sp_randint

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import RandomizedSearchCV



# Metrics

from sklearn.metrics import precision_recall_fscore_support, roc_curve, auc



# Managing Warnings 

import warnings

warnings.filterwarnings('ignore')



# Plot the Figures Inline

%matplotlib inline
df_train=pd.read_csv('../input/human-activity-recognition-with-smartphones/train.csv')

df_train.head()
shape= df_train.shape

print(shape)
df_train.info()
df_train['Activity'].value_counts().plot(kind='bar')

plt.legend()
df_train=df_train.drop('subject',axis=1)
X=df_train.iloc[:,0:len(df_train.columns)-1]

y=df_train.iloc[:,-1]
le=LabelEncoder()

y=le.fit_transform(y)
le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))

print(le_name_mapping)
sc=StandardScaler()

X=sc.fit_transform(X)
x=pd.DataFrame(X)

from statsmodels.stats.outliers_influence import variance_inflation_factor

vif = [variance_inflation_factor(x.values,i) 

for i in range(x.shape[1])]

M=pd.DataFrame({'vif':vif},index=x.columns)

print(M)
pca =PCA(0.95) #95% variance

X_pca=pca.fit_transform(X) #for training data

print(pca.n_components_)
models=[]

models.append(('LR', LogisticRegression()))

models.append(('KNN', KNeighborsClassifier()))

models.append(('CART', DecisionTreeClassifier()))

models.append(('NB', GaussianNB()))

models.append(("RF",RandomForestClassifier(n_estimators=100)))

models.append(("AB",AdaBoostClassifier(LogisticRegression())))

results = []

names = []

for name,model in models:

    skf  = model_selection.StratifiedKFold(shuffle=True,n_splits=15,random_state=0)

    cv_results =model_selection.cross_val_score(model,X_pca,y,cv=skf,scoring='f1_weighted')

    print(cv_results)

    results.append(cv_results)

    names.append(name)

    #print("%s: %f (%f)" % (name, np.mean(cv_results),np.var(cv_results,ddof=1)))

    print()

    print('F1SCORE WEIGHTED:',name,':',np.mean(cv_results))

    print('BIAS ERROR OF',name,':',1-np.mean(cv_results))

    print('VARIANCE ERROR OF',name,':',np.var(cv_results,ddof=1))

    print('------------------------------------------------------------------------------------------------------------')
from sklearn.model_selection import GridSearchCV 

  

# Creating the hyperparameter grid 

param_grid = {'penalty':['l1','l2']} 

  

# Instantiating logistic regression classifier 

lr = LogisticRegression() 

  

# Instantiating the GridSearchCV object 

lr_cv = GridSearchCV(lr, param_grid, cv = 5) 

  

lr_cv.fit(X_pca,y) 

  

# Print the tuned parameters and score 

print("Tuned Logistic Regression Parameters: {}".format(lr_cv.best_params_))  

print("Best score is {}".format(lr_cv.best_score_)) 
skf=model_selection.StratifiedKFold(shuffle=True,n_splits=15,random_state=0)

LR=LogisticRegression(penalty='l1')

results=model_selection.cross_val_score(LR,X_pca,y,cv=skf,scoring='f1_weighted')

print('F1SCORE WEIGHTED:',':',np.mean(results))

print('BIAS ERROR OF',':',1-np.mean(results))

print('VARIANCE ERROR OF',':',np.var(results,ddof=1))
df_test=pd.read_csv('../input/human-activity-recognition-with-smartphones/test.csv')
df_test.head()
df_test=df_test.drop('subject',axis=1)
X_test=df_test.iloc[:,0:len(df_test.columns)-1]



y_test=df_test.iloc[:,-1]
y_test=le.transform(y_test)
X_test =sc.transform(X_test)
X_test = pca.transform(X_test)
model=LogisticRegression(penalty='l1')

model.fit(X_pca,y)
y_pred=model.predict(X_test)
from sklearn.metrics import classification_report,accuracy_score

print(classification_report(y_test,y_pred))

print("Accuracy:",accuracy_score(y_test, y_pred)*100)