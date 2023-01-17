%matplotlib inline

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.feature_selection import SelectKBest,chi2

from sklearn.model_selection import train_test_split , StratifiedKFold,GridSearchCV

from scipy.stats import pearsonr

from sklearn.metrics import accuracy_score,recall_score , precision_score,make_scorer,confusion_matrix,precision_recall_curve,mean_squared_error

from sklearn.preprocessing import LabelEncoder,MinMaxScaler

from sklearn.linear_model import LogisticRegression,LinearRegression

from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor

from sklearn.svm import SVR

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df=pd.read_csv('/kaggle/input/breast-cancer-wisconsin-data/data.csv')
df.describe()
df.isnull().sum()  # checking for null values 
df=df.drop(['Unnamed: 32'],axis=1)
# for mean features



data = pd.concat([df.diagnosis,df.iloc[:,2:12]],axis=1)

data = pd.melt(data,id_vars="diagnosis",var_name="features",value_name='value')

plt.figure(figsize=(10,10))

sns.boxplot(x="features", y="value",whis=2.5, data=data)

plt.xticks(rotation=90)

#for std_dev features



data2 = pd.concat([df.diagnosis,df.iloc[:,12:22]],axis=1)

data2 = pd.melt(data2,id_vars="diagnosis",var_name="features",value_name='value')

plt.figure(figsize=(10,10))

sns.boxplot(x="features", y="value", whis=2.5,data=data2)

plt.xticks(rotation=90)
# for worst features



data3 = pd.concat([df.diagnosis,df.iloc[:,22:32]],axis=1)

data3 = pd.melt(data3,id_vars="diagnosis",var_name="features",value_name='value')

plt.figure(figsize=(10,10))

sns.boxplot(x="features", y="value", whis=2.5,data=data3)

plt.xticks(rotation=90)
df=df[df.area_mean<2000]

df=df[df.area_se<300]

df=df[df.area_worst<4000]

df.shape
f,ax = plt.subplots(figsize=(20,20))

sns.heatmap(df.corr(), annot=True,linewidths=.5, fmt= '.2g',ax=ax)
#correlation between all mean and worst features



cols_mean=list(df.columns)[2:12]

cols_worst=list(df.columns)[22:32]

for i in range(0,10):

    corr, _ = pearsonr(df[cols_mean[i]],df[cols_worst[i]])

    print(cols_mean[i],'-',cols_worst[i],'=',corr)
#dropping unwanted features as discussed above 



drop_list = ['perimeter_mean','radius_mean','compactness_mean','concave points_mean','radius_se','perimeter_se','radius_worst','perimeter_worst','compactness_worst','concave points_worst','compactness_se','concave points_se','texture_worst','area_worst']

df=df.drop(drop_list,axis=1)
f1,ax1 = plt.subplots(figsize=(9,9))

sns.heatmap(df.corr(), annot=True,linewidths=.5, fmt= '.2g',ax=ax1)  # after removing correlated features
# splitting the data  



Y=df['diagnosis']      # target variable

en=LabelEncoder()

Y=en.fit_transform(Y)   # encode 'N' to 0 and 'R' to 1

X=df.drop(['diagnosis'],axis=1)

X_train , X_test , y_train , y_test=train_test_split(X,Y,test_size=0.3,random_state=44)

ids1=X_test['id']

X_train=X_train.drop(['id'],axis=1)       # training data

X_test=X_test.drop(['id'],axis=1)         # test data
# we are building custom function which fits the model with best hyperparametrs using grid search , calculates accuracy and recall score

# We optimize our model to maximize recall score

# It also plots precision recall curve and confusion matrix 



def fit_model(model,Xtrain,Xtest,ytrain,ytest,features,param_grid):

  scorers = {'recall_score': make_scorer(recall_score)}

  kf = StratifiedKFold(n_splits=3)

  grids = GridSearchCV(model,param_grid,scoring=scorers, refit='recall_score',cv=kf)

  grids.fit(Xtrain[features],ytrain)

  pred = grids.predict(Xtest)

  print('Best parameters:',grids.best_params_)

  accuracy = accuracy_score(pred,ytest)

  print('Accuracy :',accuracy)

  rscore=recall_score(ytest, pred, average='binary')

  print('Recall-score:',rscore)

  fpr, tpr, thresholds = precision_recall_curve(ytest, pred)

  plt.subplot(1,2,1)

  plt.plot(fpr,tpr, marker='.')

  cm = confusion_matrix(ytest,pred)

  plt.subplot(1,2,2)

  sns.heatmap(cm,annot=True,fmt="d")

  return grids

  
feats = X_train.columns

ss= MinMaxScaler(feature_range=(0,1))  # scaling features for logistic regression

X_train1=pd.DataFrame(ss.fit_transform(X_train),columns=feats)

X_test1=pd.DataFrame(ss.fit_transform(X_test),columns=feats)

param_grid = {'C': [1],'max_iter':[100,200,300]}



lr=LogisticRegression(random_state=8)

fitted=fit_model(lr,X_train1,X_test1,y_train,y_test,feats,param_grid)

feats = X_train.columns

param_grid1 = {'n_estimators': [70,80],'min_samples_split':[4,5],'max_depth':[7,9],'max_features':[9]}

rf = RandomForestClassifier(random_state=8)

grid_search_clf=fit_model(rf,X_train,X_test,y_train,y_test,feats,param_grid1)
#checking class distribution



df2 = pd.read_csv('/kaggle/input/breast-cancer-wisconsin-data/data.csv') 

sns.catplot(x="diagnosis", kind="count", palette="ch:.75", data=df2);

y_probs = grid_search_clf.predict_proba(X_test)[:, 1]

p, r, thresholds = precision_recall_curve(y_test, y_probs)





def threshold_tuner(y_scores, t):    # for setting threshold

    

    return [1 if y >= t else 0 for y in y_scores]



def confusion_m(p, r, thresholds, t):     # plot confusion matrix after setting threshold and return new predictions

    

    y_pred_adj = threshold_tuner(y_probs, t)

    cm1 = confusion_matrix(y_test,y_pred_adj)

    sns.heatmap(cm1,annot=True,fmt="d")    

    predicted=y_pred_adj

    print('Accuracy:',accuracy_score(predicted,y_test))

    return predicted
pr=confusion_m(p, r, thresholds, 0.51)
#Reverse encoding 1 to 'M' and 0 to 'B'



task1_pred=[]             

for x in range(0,len(pr)):

  if pr[x]==0:

    task1_pred.append('B')

  elif pr[x]==1:

    task1_pred.append('M')

# creating csv file of test predictions



sub1 = pd.DataFrame({"ID": ids1, "Outcome": task1_pred})

sub1.to_csv('task_1.csv', index=False)