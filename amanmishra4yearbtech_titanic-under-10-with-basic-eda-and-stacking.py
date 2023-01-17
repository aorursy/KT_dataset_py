# common preprocessing libraries
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter


# model preprocessing libraries
import sklearn
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.preprocessing import  MinMaxScaler, RobustScaler, StandardScaler, LabelEncoder as le
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV, cross_val_score, StratifiedKFold, learning_curve
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA

# model libraries

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier
from xgboost import XGBClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

# load data
train_data = pd.read_csv('../input/titanic/train.csv')
test_data = pd.read_csv('../input/titanic/test.csv')
train_data
# finding is there null values in some columns
train_data.info()
# checking the scales of features
train_data.describe()
# fill NaN values 
train_data.fillna(method='pad',inplace=True)
test_data.fillna(method='pad',inplace=True)

# confirming about null values
train_data.info()
# correlation visualisation using heatmap
sns.set(style='darkgrid')
fig=plt.figure(figsize=(10,10))
sns.heatmap(train_data.corr(),annot=True,linewidths=0.3)
sns.countplot(train_data['Survived'])
y_train = train_data['Survived']

# drop columns which seems irrelevant for x_train and x_test
x_train = train_data.drop(['PassengerId','Name','Cabin','Survived','Ticket'],axis=1)
x_test = test_data.drop(['PassengerId','Name','Cabin','Ticket'],axis=1)
x_train
# making the columns which need to be rescaled
x1_train = pd.DataFrame()
x1_test = pd.DataFrame()
x1_train['Age'] = x_train['Age']
x1_train['Fare'] = x_train['Fare']


x1_test['Age'] = x_test['Age']
x1_test['Fare'] = x_test['Fare']

## Features Scaling

sd1 = MinMaxScaler()
sd2= MinMaxScaler()
norm1 = sd1.fit(x1_train)
norm2 = sd2.fit(x1_test)
x1_train = pd.DataFrame(norm1.transform(x1_train),columns=x1_train.columns)
x1_test = pd.DataFrame(norm2.transform(x1_test),columns=x1_test.columns)
x1_train.describe()

sns.boxplot(y=x1_train['Fare'],x=y_train)
from scipy.stats import norm
fig=plt.figure(figsize=(10,10))
sns.distplot(x1_train['Fare'][y_train==0],color='g',label='notsurvived')
sns.distplot(x1_train['Fare'][y_train==1],color='r',label='survived')
plt.legend(loc='best')


# convert to logarathmic scale to reduce skewness
x1_train['Fare'] = x1_train['Fare'].map(lambda i:np.log(i) if i>0 else 0)
x1_test['Fare'] = x1_test['Fare'].map(lambda i:np.log(i) if i>0 else 0)


from scipy.stats import norm
fig=plt.figure(figsize=(10,10))
sns.distplot(x1_train['Fare'][y_train==0],color='g',label='notsurvived')
sns.distplot(x1_train['Fare'][y_train==1],color='r',label='survived')
plt.legend(loc='best')

fig= plt.figure(figsize=(15,15))
sns.jointplot(x='Fare',y='Age',data=x1_train[y_train==0],color='g')
sns.jointplot(x='Fare',y='Age',data=x1_train[y_train==1],color='r')
sns.jointplot(x='Fare',y='Age',data=x1_test,color='b')
# changing the new columns with existing train and test columns
x_train['Age'] = x1_train['Age']
x_train['Fare'] = x1_train['Fare']
x_test['Age'] = x1_test['Age']
x_test['Fare'] = x1_test['Fare']
## Applying label encoding for all data
encode=le()
x_train['Sex'] = encode.fit_transform(x_train['Sex'])
x_test['Sex'] = encode.fit_transform(x_test['Sex'])
x_train['Embarked'] = encode.fit_transform(x_train['Embarked'])
x_test['Embarked'] = encode.fit_transform(x_test['Embarked'])


x_train
"""
# implementing PCA
#covar_matrix = PCA(n_components = 5)



##covar_matrix.fit(x_train)
#variance = covar_matrix.explained_variance_ratio_ #calculate variance ratios

#var=np.cumsum(np.round(covar_matrix.explained_variance_ratio_, decimals=3)*100)
#var #cumulative sum of variance explained with [n] features

"""
#x_train=covar_matrix.fit_transform(x_train)
#x_train.drop('SibSp',axis=1,inplace=True)
#x_test.drop('SibSp',axis=1,inplace=True)

# Applying p-value to check feature dependence
import statsmodels.api as sm
regressor_OLS = sm.OLS(endog = y_train, exog = x_train).fit()
regressor_OLS.summary()
kfold = StratifiedKFold(n_splits = 5 )

# Modeling step Test differents algorithms 
random_state = 2
classifiers = []
classifiers.append(SVC(kernel = 'rbf',probability = True))
classifiers.append(DecisionTreeClassifier(random_state=random_state))
classifiers.append(AdaBoostClassifier(ExtraTreesClassifier(random_state=2,max_depth = None,min_samples_split= 2,min_samples_leaf = 1,bootstrap = False,n_estimators =320), random_state=random_state,learning_rate=0.1))
classifiers.append(RandomForestClassifier(random_state=random_state))
classifiers.append(ExtraTreesClassifier(random_state=random_state))
classifiers.append(GradientBoostingClassifier(random_state=random_state))
classifiers.append(MLPClassifier(random_state=random_state))
classifiers.append(KNeighborsClassifier())
classifiers.append(LogisticRegression())
classifiers.append(XGBClassifier(random_state=random_state))
classifiers.append(LinearDiscriminantAnalysis())

"""
cv_results = []
for classifier in classifiers :
    cv_results.append(cross_val_score(classifier,x_train, y = y_train, scoring = 'accuracy', cv = kfold , n_jobs =-1))

cv_means = []
cv_std = []
for cv_result in cv_results:
    cv_means.append(cv_result.mean())
    cv_std.append(cv_result.std())

cv_res = pd.DataFrame({"CrossValMeans":cv_means,"CrossValerrors": cv_std,"Algorithm":["SVC","DecisionTree","Adaboost"
"RandomForest","ExtraTrees","GradientBoosting","MLP","KNeighboors","LogisticRegression","xgboost","LDA"]})

g = sns.barplot("CrossValMeans","Algorithm",data = cv_res, palette="Set3",orient = "h",**{'xerr':cv_std})
g.set_xlabel("Mean Accuracy")
g = g.set_title("Cross validation scores")
cv_res

"""
# feeding raw models into stacking ensemble as the metal model will extract tht best out of each one
from vecstack import stacking
from sklearn.metrics import accuracy_score,f1_score

S_train, S_test = stacking(classifiers,                   
                           x_train, y_train, x_test,   
                           regression= False,
                          
     
                           mode='oof_pred_bag', 
       
                           needs_proba=True,
         
                           save_dir=None, 
             
    
                           n_folds=5, 
                 
                           stratified=True,
            
                           shuffle=True,  
            
                           random_state=0,    
         
                           verbose=2)
S_train
S_train.shape
argmax_train = []
argmax_test = []
for i in range(0,S_train.shape[1],2):
    argmax_train.append( np.argmax(S_train[:,i:i+2],axis=1))
    argmax_test.append( np.argmax(S_test[:,i:i+2],axis=1))
argmax_train = np.array(argmax_train,dtype= np.int64).T
argmax_test = np.array(argmax_test,dtype= np.int64).T

argmax_train
# here using overall probabilities for meta model
## from sklearn.metrics import f1_score
modelc = LogisticRegression()
    
model1c = modelc.fit(S_train, y_train)
y_pred1c = model1c.predict_proba(S_train)
y_predc = model1c.predict_proba(S_test)

print('Final test prediction score: [%.8f]' % accuracy_score(y_train, np.argmax(y_pred1c,axis=1)))
print('Final f1-score test prediction: [%.8f]' % f1_score(y_train, np.argmax(y_pred1c,axis=1)))

# here using predictions for metal model
## from sklearn.metrics import f1_score
model = XGBClassifier(random_state=2, objective = 'reg:linear', n_jobs=-1, learning_rate= 0.5, 
                      n_estimators=30, max_depth=20)
    
model1 = model.fit(argmax_train, y_train)
y_pred1 = model1.predict_proba(argmax_train)
y_pred = model1.predict_proba(argmax_test)

print('Final test prediction score: [%.8f]' % accuracy_score(y_train, np.argmax(y_pred1,axis=1)))
print('Final f1-score test prediction: [%.8f]' % f1_score(y_train, np.argmax(y_pred1,axis=1)))

## checking the distribution of prediction
sns.distplot(y_pred)
sns.distplot(y_predc,color = 'r')
# loading sample submission file
submission = pd.read_csv('../input/titanic/gender_submission.csv')
submission
# making submission file
submission1 = pd.DataFrame()
submission1['PassengerId'] = submission['PassengerId']
submission1['Survived'] = np.argmax(y_pred+y_predc,axis=1)
# making submission
submission1.to_csv('submission1.csv',index=False)
submission1
# prediction on train data for classification report
predictions_train=model1c.predict_proba(S_train)
pred_train=np.argmax(predictions_train,axis=1)
pred_train
# confusion matrix
conf = confusion_matrix(y_train,pred_train)
sns.heatmap(conf,annot= True)
conf
# classification report
repo = classification_report(y_train,pred_train)
print(repo)
