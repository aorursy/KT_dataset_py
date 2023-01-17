# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train=pd.read_csv('/kaggle/input/Train.csv')
test=pd.read_csv('/kaggle/input/Test.csv')
train.info()
import seaborn as sns
# to render the graphs
import matplotlib.pyplot as plt
# import module to set some ploting parameters
from matplotlib import rcParams


# This function makes the plot directly on browser
%matplotlib inline

# Seting a universal figure size 
rcParams['figure.figsize'] = 10,8
# let us find the missing values.represented as yellow lines


cardinality_train ={}
for col in train.columns:
    cardinality_train[col] = train[col].nunique()
    
cardinality_test ={}
for col in test.columns:
    cardinality_test[col] = test[col].nunique()


cardinality_train



cardinality_test

data = pd.concat([train, test], axis=0, sort=False)
sns.pairplot(data)
plt.figure(figsize=(12, 7))
sns.boxplot(x='Ever_Married',y='Age',data=data,palette='winter')
plt.show()



sns.countplot("Ever_Married",data=data,hue="Gender", palette="hls")

sns.set(style="ticks")

g = sns.catplot(x="Work_Experience", y="Age", hue="Gender",height=10, data=data)
plt.figure(figsize=(12, 7))
sns.boxplot(x='Work_Experience',y='Age',data=train,palette='winter')
plt.show()
plt.subplot(2,1,1)
sns.countplot("Family_Size",data=data,hue="Spending_Score", palette="hls")
plt.ylabel("count", fontsize=18)
plt.xlabel("size", fontsize=18)
plt.title("size  dist via spend", fontsize=20)
plt.show()
plt.figure(figsize=(12, 7))
sns.boxplot(x='Spending_Score',y='Family_Size',data=data,palette='winter')
plt.show()
tab_info=pd.DataFrame(data.dtypes).T.rename(index={0:'column type'})
tab_info=tab_info.append(pd.DataFrame(data.isnull().sum()).T.rename(index={0:'null values (nb)'}))
tab_info=tab_info.append(pd.DataFrame(data.isnull().sum()/data.shape[0]*100).T.
                         rename(index={0:'null values (%)'}))
tab_info
sns.heatmap(data.isnull(),yticklabels=False,cbar=False,cmap='viridis')

data.info()
# replacing the missing values of different columns
data['Ever_Married'].fillna('unknown',inplace=True)
data['Graduated'].fillna('unknown',inplace=True)
data['Profession'].fillna('unknown',inplace=True)
data['Work_Experience'].fillna(data['Work_Experience'].mode()[0], inplace=True)
data['Family_Size'].fillna(0,inplace=True)
data['Var_1'].fillna('Cat_0',inplace=True)
data.head(5)

sns.heatmap(data.isnull(),yticklabels=False,cbar=False,cmap='viridis')
print( data.shape)
print(train.shape)
print(test.shape)
from sklearn.preprocessing import StandardScaler, LabelEncoder
#label Encoding
le = LabelEncoder()
cat_cols = ['Gender', 'Ever_Married', 'Graduated', 'Profession', 'Spending_Score', 'Var_1']

for col in cat_cols:
    data[col] = data[col].astype(str)
    LE = le.fit(data[col])
    data[col] = LE.transform(data[col])
    

#Encoding Category Variables
#def frequency_encoding(col):
   # fe=combine_set.groupby(col).size()/len(data)
   # data[col]=data[col].apply(lambda x: fe[x])

#for col in list(data.select_dtypes(include=['object']).columns):
    #if col!='Segmentation':
       # frequency_encoding(col)
ForSubmission=data
data=data.drop(['ID'],axis=1)
ForSubmission
data
# all the data cleansing and feature engineering is complete so lets seperate the test and train

train_df=data[data['Segmentation'].isnull()==False]
test_df=data[data['Segmentation'].isnull()==True]
ForSubmission_test_df=ForSubmission[ForSubmission['Segmentation'].isnull()==True]
# Creating Train and Test Data
X=train_df.drop(['Segmentation'],axis=1)
Y=train_df.Segmentation
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.metrics import accuracy_score
#Models
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.linear_model import RidgeClassifier, SGDClassifier, LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, BaggingClassifier, VotingClassifier, RandomTreesEmbedding

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.25, random_state = 22)
clfs = []
seed = 3

clfs.append(("LogReg", 
             Pipeline([("Scaler", StandardScaler()),
                       ("LogReg", LogisticRegression())])))

clfs.append(("XGBClassifier",
             Pipeline([("Scaler", StandardScaler()),
                       ("XGB", XGBClassifier())]))) 
clfs.append(("KNN", 
             Pipeline([("Scaler", StandardScaler()),
                       ("KNN", KNeighborsClassifier())]))) 

clfs.append(("DecisionTreeClassifier", 
             Pipeline([("Scaler", StandardScaler()),
                       ("DecisionTrees", DecisionTreeClassifier())]))) 

clfs.append(("RandomForestClassifier", 
             Pipeline([("Scaler", StandardScaler()),
                       ("RandomForest", RandomForestClassifier())]))) 

clfs.append(("GradientBoostingClassifier", 
             Pipeline([("Scaler", StandardScaler()),
                       ("GradientBoosting", GradientBoostingClassifier(max_features=15, n_estimators=150))]))) 

clfs.append(("RidgeClassifier", 
             Pipeline([("Scaler", StandardScaler()),
                       ("RidgeClassifier", RidgeClassifier())])))

clfs.append(("BaggingRidgeClassifier",
             Pipeline([("Scaler", StandardScaler()),
                       ("BaggingClassifier", BaggingClassifier())])))

clfs.append(("ExtraTreesClassifier",
             Pipeline([("Scaler", StandardScaler()),
                       ("ExtraTrees", ExtraTreesClassifier())])))

#'neg_mean_absolute_error', 'neg_mean_squared_error','r2'
scoring = 'accuracy'
n_folds = 10

results, names  = [], [] 

for name, model  in clfs:
    kfold = KFold(n_splits=n_folds, random_state=seed)
    cv_results = cross_val_score(model, X_train, y_train, cv= 5, scoring=scoring, n_jobs=-1)*100    
    names.append(name)
    results.append(cv_results)    
    #msg = "%s: %f (+/- %f)" % (name, cv_results.mean(),  cv_results.std())
    #print(msg)
    
# boxplot algorithm comparison
fig = plt.figure(figsize=(15,6))
fig.suptitle('Classifier Algorithm Comparison', fontsize=22)
ax = fig.add_subplot(111)
sns.boxplot(x=names, y=results)
ax.set_xticklabels(names)
ax.set_xlabel("Algorithmn", fontsize=20)
ax.set_ylabel("Accuracy of Models", fontsize=18)
ax.set_xticklabels(ax.get_xticklabels(),rotation=45)
plt.show()

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, mean_squared_log_error
from sklearn.preprocessing import *
from sklearn.model_selection import *
from sklearn.metrics import *
#import gc
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, mean_absolute_error,accuracy_score, classification_report
kfold = KFold(n_splits=10, random_state=7)
from lightgbm import LGBMClassifier
from sklearn import metrics
from sklearn.metrics import mean_squared_log_error
lgbc = LGBMClassifier(n_estimators=550,
                     learning_rate=0.03,
                     min_child_samples=40,
                     random_state=1,
                     colsample_bytree=0.5,
                     reg_alpha=2,
                     reg_lambda=2)

resultsLGB = cross_val_score(lgbc,X_train, y_train,cv=kfold)
print("LightGBM",resultsLGB.mean()*100)

#model = LGBMClassifier()
#cv = RepeatedStratifiedKFold(n_splits = 10, n_repeats = 3, random_state = 22)
#n_scores = cross_val_score(model, X_train, y_train, scoring = 'accuracy', cv = cv, n_jobs = -1, error_score = 'raise')
#print("LightGBM",n_scores.mean()*100)
LGB=lgbc.fit(X_train,y_train)
y_predict_LGBM = LGB.predict(X_test)
#print(100*(np.sqrt(mean_squared_log_error(np.exp(y_test), np.exp(y_predict_LGBM)))))
resultsLGB_test = cross_val_score(lgbc,X_test, y_test,cv=kfold)
print("LightGBM",resultsLGB_test.mean()*100)
y_predict_LGBM
sorted(zip(LGB.feature_importances_, X_train), reverse = True)
test_df

test_df=test_df.drop(['Segmentation'],axis=1)
y_predict_LGBM_testdata = LGB.predict(test_df)
y_predict_LGBM_testdata
sorted(zip(LGB.feature_importances_, test_df), reverse = True)
#lets sumbit the solution
pred = pd.DataFrame()
pred['ID'] = ForSubmission_test_df['ID']
#pred['Segmentation'] = pd.Series((model.predict(test_data)).ravel())
pred['Segmentation'] = pd.Series(y_predict_LGBM_testdata)
y_predict_LGBM_testdata
pred.to_csv('LGBM_Customer_Segementation.csv', index = None)
from catboost import  CatBoostClassifier
from sklearn import metrics
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import roc_auc_score

#cb = CatBoostRegressor(
    #n_estimators = 1000,
    #learning_rate = 0.11,
    #iterations=1000,
    #loss_function = 'RMSE',
    #eval_metric = 'RMSE',
    #verbose=0)
    
cb= CatBoostClassifier(
    iterations=100, 
    learning_rate=0.1, 
    #loss_function='CrossEntropy'
)

#rmsle = 0
#for i in ratio:
 # x_train,y_train,x_val,y_val = train_test_split(i)

#CAT=cb.fit(X_train,y_train)
#resultsCAT = cross_val_score(cb,X_train, y_train,cv=kfold)
#print("CAT",resultsCAT.mean()*100)
                        
cb.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=50,early_stopping_rounds = 100)
kfold = KFold(n_splits=10, random_state=7)
resultsCAT = cross_val_score(cb,X_train, y_train,cv=kfold)
print("CAT",resultsCAT.mean()*100)
y_predict_CAT = cb.predict(X_train)
#print(100*(np.sqrt(mean_squared_log_error(np.exp(y_train), np.exp(y_predict_CAT)))))
resultsCAT_train = cross_val_score(cb,X_train, y_train,cv=kfold)
print("CAT",resultsCAT_train.mean()*100)
y_predict_CAT_testdata = cb.predict(test_df)
y_predict_CAT_testdata
sorted(zip(cb.feature_importances_, test_df), reverse = True)
#lets sumbit the solution
pred_CAT = pd.DataFrame()
pred_CAT['ID'] = ForSubmission_test_df['ID']
#pred['Segmentation'] = pd.Series((model.predict(test_data)).ravel())
pred_CAT['Segmentation'] = pd.Series(y_predict_CAT_testdata.ravel())
pred_CAT.to_csv('CAT_Customer_Segementation.csv', index = None)
xg=XGBClassifier(booster='gbtree',verbose=0,learning_rate=0.07,max_depth=8,objective='multi:softmax',
                  n_estimators=1000,seed=294)
xg.fit(X_train,y_train)
resultsXG_train = cross_val_score(xg,X_train, y_train,cv=kfold)
print("XGBOOST",resultsXG_train.mean()*100)
y_predict_XGB_testdata = xg.predict(test_df)
y_predict_XGB_testdata
#lets sumbit the solution
pred_XGB = pd.DataFrame()
pred_XGB['ID'] = ForSubmission_test_df['ID']
#pred['Segmentation'] = pd.Series((model.predict(test_data)).ravel())
pred_XGB['Segmentation'] = pd.Series(y_predict_XGB_testdata)
pred_XGB.to_csv('XGB_Customer_Segementation.csv', index = None)
Final_Submission=pd.DataFrame()
Final_Submission=pd.concat([Final_Submission,pd.DataFrame(y_predict_LGBM_testdata),pd.DataFrame(y_predict_CAT_testdata.ravel()),pd.DataFrame(y_predict_XGB_testdata)],axis=1)
Final_Submission.columns=['LGBM','CAT','XGB']

ForSubmission=Final_Submission.mode(axis=1)[0]
ForSubmission.head(5)
submission_dataframe=pd.DataFrame()
submission_dataframe['ID']=ForSubmission_test_df['ID']
submission_dataframe['Segmentation']=ForSubmission
submission_dataframe.to_csv('Final Ensembelled_LGBM_CAT_XGB__Customer_Segementation.csv', index = None)
