import pandas as pd 

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn import metrics

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler

from sklearn.compose import ColumnTransformer

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC

from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split

import eli5

from sklearn.model_selection import GridSearchCV

from sklearn.preprocessing import RobustScaler
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
#Loading the dataset

rawdata=pd.read_csv("/kaggle/input/bda-2019-ml-test/Train_Mask.csv")

rawtest=pd.read_csv("/kaggle/input/bda-2019-ml-test/Test_Mask_Dataset.csv")
#initial glimpse of the data

rawdata.head()
rawdata.info()
#checking for null values

rawdata.isnull().sum()
corr = rawdata.drop(columns=['timeindex','flag']).corr()

sns.heatmap(corr)
#removing one of two features that have a correlation higher than 0.9

columns = np.full((corr.shape[0],), True, dtype=bool)

for i in range(corr.shape[0]):

    for j in range(i+1, corr.shape[0]):

        if corr.iloc[i,j] >= 0.9:

            if columns[j]:

                columns[j] = False

selected_columns = rawdata.drop(columns=['timeindex','flag']).columns[columns]

data = rawdata[selected_columns]
#selected features based on corrrelation

selected_columns
#plotting these features against flag

fig = plt.figure(figsize = (20, 25))

j = 0

#iterating over selected columns

for i in selected_columns:

    plt.subplot(6, 4, j+1)

    j += 1

    sns.distplot(rawdata[i][rawdata['flag']==0], color='g', label = 'anomaly')

    sns.distplot(rawdata[i][rawdata['flag']==1], color='r', label = 'normal')

    plt.legend(loc='best')

fig.suptitle('Anomaly Data Analysis')

fig.tight_layout()

fig.subplots_adjust(top=0.95)

plt.show()
#creating pipeline for individual algorithms



#logistic regression

Pipeline_lr = Pipeline(steps=[('scaler',StandardScaler()),

('lr_classifier',LogisticRegression())])



#DecisionTree 

pipeline_dt = Pipeline(steps=[('scaler',StandardScaler()),

('dt_classifier',DecisionTreeClassifier())])



#Randomforest (weighted random forest because of class imbalance)

pipeline_rf = Pipeline(steps=[('scaler',StandardScaler()),

('rf_classifier',RandomForestClassifier(random_state=0, n_jobs=-1, class_weight="balanced"))])



#svm classifier

pipeline_svm = Pipeline(steps=[('scaler',StandardScaler()),

('svm_classifier',SVC())])



#creating a pipeline for XGBoost classifier

pipeline_xgb = Pipeline(steps=[('scaler',StandardScaler()),

('xgb_classifier',XGBClassifier())])



#making a list of pipelines

pipelines=[Pipeline_lr, pipeline_dt, pipeline_rf,pipeline_svm,pipeline_xgb]



#dict of pipelines with name in order for reference

pipe_dict={0:'Logisticregression',1:'Decisiontree',2:'weighted Randomforest',3:'Support Vector machine',4:'XGBoost'}
target=rawdata['flag']
#creating a train test split

X_train,X_test,y_train,y_test=train_test_split(data,target,test_size=0.2,random_state=12)
#fiting the data

for pipe in pipelines:

    pipe.fit(X_train,y_train)
#testing each of the model

for i,model in enumerate(pipelines):

    print("{} Test Accuracy: {}".format(pipe_dict[i],model.score(X_test,y_test)))

#confusion metrics

for i,model in enumerate(pipelines):

    y_pred=model.predict(X_test)

    print(pipe_dict[i])

    print(metrics.confusion_matrix(y_test, y_pred, labels=[1,0]))

    print(metrics.classification_report(y_test, y_pred, labels=[1,0]))
#creating pipeline for individual algorithms



#logistic regression

Pipeline_lr = Pipeline(steps=[('scaler',RobustScaler()),

('lr_classifier',LogisticRegression())])



#DecisionTree 

pipeline_dt = Pipeline(steps=[('scaler',RobustScaler()),

('dt_classifier',DecisionTreeClassifier())])



#Randomforest (weighted random forest because of class imbalance)

pipeline_rf = Pipeline(steps=[('scaler',RobustScaler()),

('rf_classifier',RandomForestClassifier(random_state=0, n_jobs=-1, class_weight="balanced"))])



#svm classifier

pipeline_svm = Pipeline(steps=[('scaler',RobustScaler()),

('svm_classifier',SVC())])



#creating a pipeline for XGBoost classifier

pipeline_xgb = Pipeline(steps=[('scaler',RobustScaler()),

('xgb_classifier',XGBClassifier())])



#making a list of pipelines

pipelines=[Pipeline_lr, pipeline_dt, pipeline_rf,pipeline_svm,pipeline_xgb]



#dict of pipelines with name in order for reference

pipe_dict={0:'Logisticregression',1:'Decisiontree',2:'weighted Randomforest',3:'Support Vector machine',4:'XGBoost'}
#fiting the data

for pipe in pipelines:

    pipe.fit(X_train,y_train)
#testing each of the model

for i,model in enumerate(pipelines):

    print("{} Test Accuracy: {}".format(pipe_dict[i],model.score(X_test,y_test)))

#confusion metrics

for i,model in enumerate(pipelines):

    y_pred=model.predict(X_test)

    print(pipe_dict[i])

    print(metrics.confusion_matrix(y_test, y_pred, labels=[1,0]))

    print(metrics.classification_report(y_test, y_pred, labels=[1,0]))
#even though XGboost perforemed well for train data svm outperformed for the test dataset
# defining parameter range 

param_grid = {'C': [0.1, 1, 10, 100, 1000],  

              'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 

              'kernel': ['rbf']}  

grid = GridSearchCV(pipeline_svm.named_steps['svm_classifier'], param_grid, refit = True, verbose = 3) 

  

# fitting the model for grid search 

grid.fit(X_train, y_train) 
grid.score(X_test,y_test)
# submission1=pd.DataFrame(rawtest['timeindex'])

# submission1['flag']=grid_predictions

# submission1.rename(columns={'timeindex':'Sl.No'},inplace=True)

# submission1.to_csv("submission9.csv",index=False)