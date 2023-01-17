import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt



import warnings

warnings.filterwarnings("ignore")
df_red = pd.read_csv("../input/wine-quality-red/winequality-red.csv", sep=';') ##Gettint the data
df_red.head()
df_red.describe()
df_red.isnull().sum() ##Checking null values
df_white = pd.read_csv("../input/wine-quality-red/winequality-white.csv",sep=';' )
df_white.head()
df_red.quality.unique()
sns.countplot(x='quality', data=df_red)
df_red.loc[df_red['quality'] <= 6, 'quality'] = 0

df_red.loc[df_red['quality'] > 6, 'quality'] = 1 ##Converting the quality data
sns.barplot('quality','fixed acidity',data=df_red)
df_red[['fixed acidity','quality']].groupby('quality', as_index=False).mean()
sns.barplot('quality','volatile acidity',data=df_red)
df_red[['volatile acidity','quality']].groupby('quality', as_index=False).mean()
sns.barplot('quality','citric acid',data=df_red)
df_red[['citric acid','quality']].groupby('quality', as_index=False).mean()
sns.barplot('quality','residual sugar',data=df_red)
df_red[['residual sugar','quality']].groupby('quality', as_index=False).mean()
sns.barplot('quality','chlorides',data=df_red)
df_red[['chlorides','quality']].groupby('quality', as_index=False).mean()
sns.barplot('quality','free sulfur dioxide',data=df_red)
df_red[['free sulfur dioxide','quality']].groupby('quality', as_index=False).mean()
sns.barplot('quality','total sulfur dioxide',data=df_red)
df_red[['total sulfur dioxide','quality']].groupby('quality', as_index=False).mean()
sns.barplot('quality','density',data=df_red)
df_red[['density','quality']].groupby('quality', as_index=False).mean()
sns.barplot('quality','pH',data=df_red)
df_red[['pH','quality']].groupby('quality', as_index=False).mean()
sns.barplot('quality','sulphates',data=df_red)
df_red[['sulphates','quality']].groupby('quality', as_index=False).mean()
sns.barplot('quality','alcohol',data=df_red)
df_red[['alcohol','quality']].groupby('quality', as_index=False).mean()
plt.figure(figsize=(20,15))

sns.heatmap(df_red.corr(), annot=True)
df_red.columns ##Reading the columns
features = ['volatile acidity', 'citric acid', 'chlorides', 'total sulfur dioxide', 'sulphates', 'alcohol']

targets = ['quality'] ##Choosing features and targets based on the analysis on graphs and tables
from sklearn.model_selection import train_test_split
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(df_red[features],df_red[targets],test_size=0.20, random_state=42)

##Dividing the dataset for train and test
from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier #Import relevant libraries to process the machine learn model
LR = LogisticRegression()

rfc = RandomForestClassifier()
params_log = [{'C':[0.1, 0.5, 1, 10, 100],'max_iter':[250,500,750,1000]}]

params_rfc = [{'n_estimators':[150,200,250,300], 'criterion':['gini', 'entropy'], 'max_depth':[4,5,6,7,8,9]}] ##Parameters for the GridSearchCV
grid_log = GridSearchCV(LR, params_log, n_jobs = 8, cv=10)

grid_rfc = GridSearchCV(rfc, params_rfc, n_jobs = 8, cv=10)
grid_log.fit(df_red[features],df_red[targets].values.ravel())

grid_rfc.fit(df_red[features],df_red[targets].values.ravel())
print(grid_log.best_params_)

print(grid_rfc.best_params_) ##Checking best scores to use in a model
model_log = LogisticRegression(C=10, max_iter=250)

model_rfc = RandomForestClassifier(n_estimators=150, max_depth=6, criterion='entropy') ##Using best parameters discovered above.
model_log.fit(X_train_r, y_train_r)

model_rfc.fit(X_train_r, y_train_r)
y_pred_log = model_log.predict(X_test_r)

y_pred_rfc = model_rfc.predict(X_test_r)
from sklearn.metrics import accuracy_score
print("Accuracy for the Support Vector Classifier Model is: ",round(accuracy_score(y_test_r,y_pred_log)*100,2),"%")
print("Accuracy for the Random Forest Classifier Model is: ",round(accuracy_score(y_test_r,y_pred_rfc)*100,2),"%")
from sklearn.metrics import classification_report
print("For SVC: \n", classification_report(y_test_r, y_pred_log))
print("For RFC: \n", classification_report(y_test_r, y_pred_rfc))
df_white.head()
df_white.describe()
df_white.isnull().sum()
df_white.loc[df_white['quality'] <= 6, 'quality'] = 0

df_white.loc[df_white['quality'] > 6, 'quality'] = 1
sns.barplot('quality','fixed acidity',data=df_white)
df_white[['fixed acidity','quality']].groupby('quality', as_index=False).mean()
sns.barplot('quality','volatile acidity',data=df_white)
df_white[['volatile acidity','quality']].groupby('quality', as_index=False).mean()
sns.barplot('quality','citric acid',data=df_white)
df_white[['citric acid','quality']].groupby('quality', as_index=False).mean()
sns.barplot('quality','residual sugar',data=df_white)
df_white[['residual sugar','quality']].groupby('quality', as_index=False).mean()
sns.barplot('quality','chlorides',data=df_white)
df_white[['chlorides','quality']].groupby('quality', as_index=False).mean()
sns.barplot('quality','free sulfur dioxide',data=df_white)
df_white[['free sulfur dioxide','quality']].groupby('quality', as_index=False).mean()
sns.barplot('quality','total sulfur dioxide',data=df_white)
df_white[['total sulfur dioxide','quality']].groupby('quality', as_index=False).mean()
sns.barplot('quality','density',data=df_white)
df_white[['density','quality']].groupby('quality', as_index=False).mean()
sns.barplot('quality','pH',data=df_white)
df_white[['pH','quality']].groupby('quality', as_index=False).mean()
sns.barplot('quality','sulphates',data=df_white)
df_white[['sulphates','quality']].groupby('quality', as_index=False).mean()
sns.barplot('quality','alcohol',data=df_white)
df_white[['alcohol','quality']].groupby('quality', as_index=False).mean()
df_white.columns
plt.figure(figsize=(20,15))

sns.heatmap(df_white.corr(), annot=True)
df_white.columns
features_white = ['residual sugar', 'chlorides', 'total sulfur dioxide', 'alcohol']

targets = ['quality']
X_train_w, X_test_w, y_train_w, y_test_w = train_test_split(df_white[features_white],df_white[targets],test_size=0.20, random_state=42)
LR1 = LogisticRegression()

rfc1 = RandomForestClassifier()
params_log1 = [{'C':[0.1, 0.5, 1, 10, 100],'max_iter':[250,500,750,1000]}]

params_rfc1 = [{'n_estimators':[150,200,250,300], 'criterion':['gini', 'entropy'], 'max_depth':[5,6,7,8,9]}] ##Parameters for the GridSearchCV
grid_log1 = GridSearchCV(LR1, params_log1, n_jobs = 8, cv=10)

grid_rfc1 = GridSearchCV(rfc1, params_rfc1, n_jobs = 8, cv=10)
grid_log1.fit(df_white[features],df_white[targets].values.ravel())

grid_rfc1.fit(df_white[features],df_white[targets].values.ravel())
print(grid_log1.best_params_)

print(grid_rfc1.best_params_) ##Checking best scores to use in a model
model_log1 = LogisticRegression(C=0.1, max_iter=250)

model_rfc1 = RandomForestClassifier(n_estimators=200, max_depth=8, criterion='gini') ##Using best parameters discovered above.
model_log1.fit(X_train_w, y_train_w)

model_rfc1.fit(X_train_w, y_train_w)
y_pred_log1 = model_log1.predict(X_test_w)

y_pred_rfc1 = model_rfc1.predict(X_test_w)
print("Accuracy for the Support Vector Classifier Model is: ",round(accuracy_score(y_test_w,y_pred_log1)*100,2),"%")
print("Accuracy for the Random Forest Classifier Model is: ",round(accuracy_score(y_test_w,y_pred_rfc1)*100,2),"%")
from sklearn.metrics import classification_report
print("For SVC: \n", classification_report(y_test_w, y_pred_log1))
print("For RFC: \n", classification_report(y_test_w, y_pred_rfc1))
accuracy = pd.DataFrame({'Red Wine':['88%','62%'],'White Wine':['83%','71%']}, index=['Bad Wine','Good Wine'])
accuracy