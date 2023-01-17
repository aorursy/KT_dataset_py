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

import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import StandardScaler
SC=StandardScaler()
from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBRFClassifier,XGBClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import SGDClassifier
from sklearn.multiclass import OneVsRestClassifier

from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
train=pd.read_csv('/kaggle/input/cat-in-the-dat/train.csv')
test=pd.read_csv('/kaggle/input/cat-in-the-dat/test.csv')
train.info()
test.info()
train.head(1)
# Preprocessing of Train and test Data together
# Concatenate Train and Test Data with a new column to identify Train (Type=0) VS Test Data(Type=1)  
train['Type']=pd.DataFrame(np.zeros(len(train)).astype(int))
test['Type']=pd.DataFrame(np.ones(len(test)).astype(int))

print('Original Train Data shape:{} and Test Data shape:{}'.format(train.shape,test.shape))
features_Data=pd.concat([train.drop(columns=['target','id']),test.drop(columns=['id'])])
print('features_Data shape after combining Train and Test ',features_Data.shape)
label=train.target

# Lets do some EDA
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
features_Data.head()

# Count plot
plt.figure(figsize=[12,14])
features=list(features_Data.columns)
n=1
for f in features:
    plt.subplot(10,4,n)
    sns.countplot(features_Data[f])
    sns.despine()
    n=n+1
plt.tight_layout()
plt.show()
pd.set_option('display.max_rows', 400)
features_Data.nom_5.value_counts().shape # 222 distinct categories
features_Data.nom_6.value_counts().shape # 522 distinct categories
features_Data.nom_7.value_counts().shape # 1220 distinct categories
features_Data.nom_8.value_counts().shape # 2219 distinct categories
features_Data.nom_9.value_counts().shape # 12068 distinct categories
features_Data.ord_5.value_counts().shape # 192 distinct categories
features_Data.head(1)
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder,LabelBinarizer
# Handling of Binary Features
LB=LabelBinarizer()                           
features_Data['bin_3']=LB.fit_transform(features_Data.bin_3)
features_Data['bin_4']=LB.fit_transform(features_Data.bin_4)

# Handling of Nominal Features
LE=LabelEncoder()
features_nom=features_Data.loc[:,['nom_0','nom_1','nom_2','nom_3','nom_4','nom_5','nom_6','nom_7','nom_8','nom_9']]
features_nom=features_nom.apply(LE.fit_transform)
features_Data[['nom_0','nom_1','nom_2','nom_3','nom_4','nom_5','nom_6','nom_7','nom_8','nom_9']]=features_nom

# Handling of Ordinal Features
OE=OrdinalEncoder()
features_ord=features_Data.loc[:,['ord_1','ord_2','ord_3','ord_4','ord_5']]
features_ord=OE.fit_transform(features_ord)
features_Data[['ord_1','ord_2','ord_3','ord_4','ord_5']]=features_ord
features_Data.info()
# Now that we have lableEncoded both Train and Test Data together , its time to split them apart
train_data=features_Data[features_Data.Type==0].drop(columns=['Type'])
test_ata=features_Data[features_Data.Type==1].drop(columns=['Type'])

# Its better to convert all label encoded data to int64
train_data=train_data.astype('int64')
test_data=test_ata.astype('int64')
train_data['target']=label

#Extrat Features and Label
features=train_data.drop(columns='target').values
label=train_data.loc[:,'target'].values
test_data.shape
#Lets perform Cross validations considering all features and see what could be the best score

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score,classification_report,confusion_matrix
#from sklearn import metrics

def stratified_K_fold_validation(model_obj, model_name, process, n_splits, X, y):
    global df_model_selection
    
    skf = StratifiedKFold(n_splits, random_state=12,shuffle=True)
    
    weighted_f1_score = []
    #print(skf.split(X,y))
    for train_index, test_index in skf.split(X,y):
        X_train, X_test = X[train_index], X[test_index] 
        y_train, y_test = y[train_index], y[test_index]
        
        
        model_obj.fit(X_train, y_train)
        test_ds_predicted = model_obj.predict( X_test )      
        weighted_f1_score.append(round(f1_score(y_true=y_test, y_pred=test_ds_predicted , average='weighted'),2))
        
    sd_weighted_f1_score = np.std(weighted_f1_score, ddof=1)
    range_of_f1_scores = "{}-{}".format(min(weighted_f1_score),max(weighted_f1_score))    
    df_model_selection = pd.concat([df_model_selection,pd.DataFrame([[process,model_name,sorted(weighted_f1_score),range_of_f1_scores,sd_weighted_f1_score]], columns =COLUMN_NAMES) ])
    
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import StandardScaler
SC=StandardScaler()
from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBRFClassifier,XGBClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import SGDClassifier
from sklearn.multiclass import OneVsRestClassifier

from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

COLUMN_NAMES = ["Process","Model Name", "F1 Scores","Range of F1 Scores","Std Deviation of F1 Scores"]
df_model_selection = pd.DataFrame(columns=COLUMN_NAMES)

process='ALl Features'
n_splits = 10
X=SC.fit_transform(features)
y=label

# 1.Naive Bayes
model_NB=BernoulliNB()
model_obj=model_NB
model_name='Naive Bayes'
stratified_K_fold_validation(model_obj, model_name, process, n_splits, X, y)

# # 2.Logistic Regression
model_LR=LogisticRegression()
model_obj=model_LR
model_name='Logistic Regression'
stratified_K_fold_validation(model_obj, model_name, process, n_splits, X, y)

# # 3.Decesion Tree Classifier
model_DTC=DecisionTreeClassifier()
model_obj=model_DTC
model_name='Decesion Tree Classifier'
stratified_K_fold_validation(model_obj, model_name, process, n_splits, X, y)

# 4.Random Forest Classifier
model_RFC=RandomForestClassifier()
model_obj=model_RFC
model_name='Random Forest Classifier'
stratified_K_fold_validation(model_obj, model_name, process, n_splits, X, y)

# 5.XGBoost Classifier
model_XGBC=XGBClassifier()
model_obj=model_XGBC
model_name='XGBoost Classifier'
stratified_K_fold_validation(model_obj, model_name, process, n_splits, X, y)

# 6.Gradient Boosting Classifier
model_GBC=GradientBoostingClassifier()
model_obj=model_GBC
model_name='Gradient Boosting Classifier'
stratified_K_fold_validation(model_obj, model_name, process, n_splits, X, y)

# 7.XGBoost Random Forest Classifier
model_XGBRFC=XGBRFClassifier()
model_obj=model_XGBRFC
model_name='XGBoost Random Forest Classifier'
stratified_K_fold_validation(model_obj, model_name, process, n_splits, X, y)

# 8.Support Vector Machine Classifier
# model_SVC=SVC()
# model_obj=model_SVC
# model_name='Support Vector Machine Classifier'
# stratified_K_fold_validation(model_obj, model_name, process, n_splits, X, y)


# 9.SGD Classifier
# model_sgd = OneVsRestClassifier(SGDClassifier())
# model_obj=model_sgd
# model_name='Stochastic Gradient Descent Classifier'
# stratified_K_fold_validation(model_obj, model_name, process, n_splits, X, y)

#10.Gausian Process Classifier
# model_GPC = GaussianProcessClassifier()
# model_obj=model_GPC
# model_name='Gausian Process Classifier'
# stratified_K_fold_validation(model_obj, model_name, process, n_splits, X, y)

#11.Gausian Process Classifier
# model_KNNC=KNeighborsClassifier()
# model_obj=model_KNNC
# model_name='K Nearst Neighbour Classifier'
# stratified_K_fold_validation(model_obj, model_name, process, n_splits, X, y)

#12 Linear Discriminant Analysis
# model_LDA=LinearDiscriminantAnalysis()
# model_obj=model_LDA
# model_name='Linear Discriminant Analysis'
# stratified_K_fold_validation(model_obj, model_name, process, n_splits, X, y)

#Exporting the results to csv
#df_model_selection.to_csv("Model_statistics.csv",index = False)
df_model_selection
# Lets apply Chisquare test of independence and select K-best features
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest
modelKBest=SelectKBest(score_func=chi2,k='all')
finalFeatures=modelKBest.fit_transform(features,label)
print(modelKBest.scores_)
# From above test we can drop below features from modeling and try  
# ['bin_0','bin_2','bin_3','nom_2','nom_5','nom_8','nom_9','day']

#Features from Chisquare Test
features=train_data.drop(columns=['bin_0','bin_2','bin_3','nom_2','nom_5','nom_8','nom_9','day','target']).values

COLUMN_NAMES = ["Process","Model Name", "F1 Scores","Range of F1 Scores","Std Deviation of F1 Scores"]
df_model_selection = pd.DataFrame(columns=COLUMN_NAMES)

process='ALl Features'
n_splits = 10
X=SC.fit_transform(features)
y=label


# # 5.XGBoost Classifier
model_XGBC=XGBClassifier()
model_obj=model_XGBC
model_name='XGBoost Classifier'
stratified_K_fold_validation(model_obj, model_name, process, n_splits, X, y)

df_model_selection
import scipy.stats as stats
#Lets try to do Chisqueare test of independence one Feature at a time
#if p_value < 0.05: we reject NULL hypothesis and it means two features are dependent on each other
# ch2 , p_value , df, exp_freq = stats.chi2_contingency(pd.crosstab(train_data.bin_0,train_data.target))
# print(p_value)
# if p_value<0.05:
#     print('bin_0 and target are related')
# else:
#     print('bin_0 variable can be dropped')

for col in train_data.columns:
    ch2 , p_value , df, exp_freq = stats.chi2_contingency(pd.crosstab(train_data[col],train_data.target))
    if p_value >=0.05:
        print('p_value of the feature is: {} and feature to be dropped is: {}'.format(p_value,col))
    #else:
         #print('p_value of the feature is: {} and feature to be considered is: {}'.format(p_value,col))
# From above test we can drop only one feature which is : bin_0

features=train_data.drop(columns=['bin_0','target']).values

COLUMN_NAMES = ["Process","Model Name", "F1 Scores","Range of F1 Scores","Std Deviation of F1 Scores"]
df_model_selection = pd.DataFrame(columns=COLUMN_NAMES)

process='ALl Features'
n_splits = 10
X=SC.fit_transform(features)
y=label


# # 5.XGBoost Classifier
model_XGBC=XGBClassifier()
model_obj=model_XGBC
model_name='XGBoost Classifier'
stratified_K_fold_validation(model_obj, model_name, process, n_splits, X, y)

df_model_selection
# Lets apply SelectFromModel Feature Selection technique
from sklearn.feature_selection import SelectFromModel
model_XGBC=XGBClassifier()
selectFeatures=SelectFromModel(estimator=model_XGBC)
selectFeatures.fit(features,label)
print(selectFeatures.get_support())
# SelectFromModel technique says the best features are below
# ['bin_0','bin_4','nom_3','nom_9','ord_0','ord_1','ord_2','ord_3','ord_4']

features=train_data[['bin_0','bin_4','nom_3','nom_9','ord_0','ord_1','ord_2','ord_3','ord_4']].values

COLUMN_NAMES = ["Process","Model Name", "F1 Scores","Range of F1 Scores","Std Deviation of F1 Scores"]
df_model_selection = pd.DataFrame(columns=COLUMN_NAMES)

process='ALl Features'
n_splits = 10
X=SC.fit_transform(features)
y=label


# # 5.XGBoost Classifier
model_XGBC=XGBClassifier()
model_obj=model_XGBC
model_name='XGBoost Classifier'
stratified_K_fold_validation(model_obj, model_name, process, n_splits, X, y)

df_model_selection
# Lets try to see feature importance method from Decesion Tree
COLUMN_NAMES = ["Process","Model Name", "F1 Scores","Range of F1 Scores","Std Deviation of F1 Scores"]
df_model_selection = pd.DataFrame(columns=COLUMN_NAMES)

features=train_data.drop(columns='target').values

process='ALl Features'
n_splits = 10
X=SC.fit_transform(features)
y=label

# # .Decesion Tree Classifier
model_DTC=DecisionTreeClassifier()
model_DTC.fit(X,y)
print('Feature Importance from Decesion Tree Clsssifier',model_DTC.feature_importances_)

# # .Random Forest Classifier
model_RFC=RandomForestClassifier()
model_RFC.fit(X,y)
print('Feature Importance from RandomForest Classifir',model_RFC.feature_importances_)

# Feature Importance from Decesion Tree Clsssifier and RandomForest Classifier. Below are the features
#['nom_5','nom_6','nom_7','nom_8','nom_9','ord_3','ord_4','ord_5']

COLUMN_NAMES = ["Process","Model Name", "F1 Scores","Range of F1 Scores","Std Deviation of F1 Scores"]
df_model_selection = pd.DataFrame(columns=COLUMN_NAMES)

features=train_data[['nom_5','nom_6','nom_7','nom_8','nom_9','ord_3','ord_4','ord_5']].values

process='ALl Features'
n_splits = 10
X=SC.fit_transform(features)
y=label

# 1.Naive Bayes
model_NB=BernoulliNB()
model_obj=model_NB
model_name='Naive Bayes'
stratified_K_fold_validation(model_obj, model_name, process, n_splits, X, y)

# # 2.Logistic Regression
model_LR=LogisticRegression()
model_obj=model_LR
model_name='Logistic Regression'
stratified_K_fold_validation(model_obj, model_name, process, n_splits, X, y)

# # 3.Decesion Tree Classifier
model_DTC=DecisionTreeClassifier()
model_obj=model_DTC
model_name='Decesion Tree Classifier'
stratified_K_fold_validation(model_obj, model_name, process, n_splits, X, y)

# 4.Random Forest Classifier
model_RFC=RandomForestClassifier()
model_obj=model_RFC
model_name='Random Forest Classifier'
stratified_K_fold_validation(model_obj, model_name, process, n_splits, X, y)

# 5.XGBoost Classifier
model_XGBC=XGBClassifier()
model_obj=model_XGBC
model_name='XGBoost Classifier'
stratified_K_fold_validation(model_obj, model_name, process, n_splits, X, y)

df_model_selection
# Conclusion from Feature Engineering Approach SO Far: 
# Based on Chisquare Test of independence, Select From Model & Feature Importance it is obvious that almost all features are dependent on target. 
#Chisquare test of independence and select K-best features will reduce the number of features to some extent without compromising the accuracy too much

#Features from Chisquare Test
features=train_data.drop(columns=['bin_0','bin_2','bin_3','nom_2','nom_5','nom_8','nom_9','day','target']).values

COLUMN_NAMES = ["Process","Model Name", "F1 Scores","Range of F1 Scores","Std Deviation of F1 Scores"]
df_model_selection = pd.DataFrame(columns=COLUMN_NAMES)

process='Chisquare test K-best features'
n_splits = 10
X=SC.fit_transform(features)
y=label


# # 5.XGBoost Classifier
model_XGBC=XGBClassifier()
model_obj=model_XGBC
model_name='XGBoost Classifier'
stratified_K_fold_validation(model_obj, model_name, process, n_splits, X, y)

df_model_selection
features=train_data.drop(columns=['bin_0','bin_2','bin_3','nom_2','nom_5','nom_8','nom_9','day','target']).values

# Now lets try to get the best split score using StratifiedKFold Cross Validation

#Initialize the algo
model=XGBClassifier()

#Initialize StratifiedKFold Method
from sklearn.model_selection import StratifiedKFold
kfold = StratifiedKFold(n_splits=10, 
              random_state=1,
              shuffle=True)

#Initialize For Loop 

i=0
for train,test in kfold.split(features,label):
    i = i+1
    X_train,X_test = features[train],features[test]
    y_train,y_test = label[train],label[test]
    
    model.fit(X_train,y_train)
    test_ds_predicted=model.predict(X_test)
    train_ds_predicted=model.predict(X_train)
    
    test_f1_score=round(f1_score(y_true=y_test, y_pred=test_ds_predicted , average='weighted'),2)
    train_f1_score=round(f1_score(y_true=y_train, y_pred=train_ds_predicted , average='weighted'),2)
    
    #print("Train Score: {}, Test score: {}, for Sample Split: {}".format(model.score(X_train,y_train),model.score(X_test,y_test),i))
    print("Train f1-Score: {}, Test f1-score: {}, for Sample Split: {}".format(train_f1_score,test_f1_score,i))
    

#Lets extract the Train and Test sample for split 1
from sklearn.model_selection import StratifiedKFold
kfold = StratifiedKFold(n_splits=10, #n_splits should be equal to no of cv value in cross_val_score
              random_state=1,
              shuffle=True)
i=0
for train,test in kfold.split(features,label):
    i = i+1
    if i == 1:
        X_train,X_test,y_train,y_test = features[train],features[test],label[train],label[test]

#Final Model
model=XGBClassifier()
model.fit(X_train,y_train)

test_ds_predicted=model.predict(X_test)
train_ds_predicted=model.predict(X_train)

test_f1_score=round(f1_score(y_true=y_test, y_pred=test_ds_predicted , average='weighted'),2)
train_f1_score=round(f1_score(y_true=y_train, y_pred=train_ds_predicted , average='weighted'),2)
print("Train f1-Score: {}, Test f1-score: {}".format(train_f1_score,test_f1_score))


train_score=np.round(model.score(X_train,y_train),2)
test_score=np.round(model.score(X_test,y_test),2)
print('Train Accuracy Score is:{} and  Test Accuracy Score:{}'.format(train_score,test_score))
# Confusion Matrix and Classification Report
from sklearn.metrics import confusion_matrix,classification_report
cm=confusion_matrix(y_true=label, y_pred=model.predict(features))
CR=classification_report(y_true=label, y_pred=model.predict(features))
print('Confusion Matrix:\n',cm)
print('\n Classification Report:\n',CR)
#model.predict()
final_test_data=test_data.drop(columns=['bin_0','bin_2','bin_3','nom_2','nom_5','nom_8','nom_9','day']).values
SC_test_data=SC.fit_transform(final_test_data)
submission=model.predict(SC_test_data)
submission
submission=pd.DataFrame(submission,columns=['target'])
submission.insert(0,'id',test['id'])
submission.to_csv('submission.csv',index=False)

submission.head()


