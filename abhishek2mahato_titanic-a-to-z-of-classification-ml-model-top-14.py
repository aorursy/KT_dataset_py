import numpy as np
import pandas as pd
import dask.dataframe as dd
import pandas_profiling
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import plotly_express as px
import time
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from pylab import rcParams
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost.sklearn import XGBClassifier
import lightgbm as lgb
from sklearn.ensemble import VotingClassifier
from mlxtend.classifier import StackingClassifier
from vecstack import stacking
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn import metrics
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
#Read data through Dask and compute time taken to read
t_start = time.time()
titanic_train = dd.read_csv('../input/titanic/train.csv')
titanic_test = dd.read_csv('../input/titanic/test.csv')
t_end = time.time()
print('dd.read_csv(): {} s'.format(t_end-t_start)) # time [s]
#Read data through Pandas and compute time taken to read
t_start = time.time()
titanic_train = pd.read_csv('../input/titanic/train.csv')
titanic_test = pd.read_csv('../input/titanic/test.csv')
t_end = time.time()
print('pd.read_csv(): {} s'.format(t_end-t_start)) # time [s]
#Getting the head of the train dataset
titanic_train.tail(5)
#Getting the head of the test dataset
titanic_test.head(5)
#Shape of the train dataset
titanic_train.shape
#Shape of the test dataset
titanic_test.shape
#Data Type of features for train dataset
titanic_train.dtypes
#Data Type of features for test dataset
titanic_test.dtypes
#Info of the train dataset
titanic_train.info()
#Info of the test dataset
titanic_test.info()
#Value counts of target variable
titanic_train['Survived'].value_counts()
#Checking for null values in train dataset
titanic_train.isnull().sum()
#Checking for null values in test dataset
titanic_test.isnull().sum()
#Checking how many males and females survived
pd.crosstab(titanic_train.Sex,titanic_train.Survived)
#Checking survived passengers with respect to region 
pd.crosstab(titanic_train.Survived , titanic_train.Embarked)
#Checking survived passengers with respect to Passenger Class
pd.crosstab(titanic_train.Survived , titanic_train.Pclass)
#Summary statistics for the train dataset
titanic_train.describe()
#Summary statistics for the test dataset
titanic_test.describe()
#Checking fare with histplot
plt.figure(figsize=(4,3))
titanic_train['Fare'].plot.hist()
#Histogram for Age
age = titanic_train['Age']
bins = [0,10,20,30,40,50,60,70,80,90,100]
plt.figure(figsize=(4,3))
plt.hist(age, bins, histtype='bar', rwidth=0.8)
plt.xlabel('Age of the Passengers')
plt.ylabel('Number of Passengers')
plt.title('Histogram')
plt.show()
#Checking passengers from which region had aborded the most
x= titanic_train['Embarked'].value_counts()
labels = {'C' :'Cherbourg', 'Q' : 'Queenstown', 'S' : 'Southampton'}
explode=(0,0,0.2)
colors = ['lightgreen','lightcoral','lightblue']
plt.figure(figsize=(5,4))
plt.pie(x.values,explode=explode,
  labels=labels,
  colors=colors,
  startangle=90,
  shadow= True,counterclock=False,textprops={'fontweight':'bold','fontsize':'large'},
  autopct='%1.1f%%')
plt.title('Pie Plot')
plt.axis('equal')
plt.show()
#Countplot for 'Survived' variable
plt.figure(figsize=(4,3))
sns.countplot(x='Survived', data=titanic_train)
#Checking how many males/females survived
plt.figure(figsize=(4,3))
sns.countplot(x='Survived', hue='Sex', data=titanic_train)
#Checking survival with respect to Passenger Class
plt.figure(figsize=(4,3))
sns.countplot(x='Survived', hue='Pclass', data=titanic_train)
#Checking survival with respect to Sibling and Spouse
plt.figure(figsize=(4,3))
sns.countplot(x='SibSp', hue='Survived', data= titanic_train)
#Checking survival with respect to Parents and Childrain
plt.figure(figsize=(4,3))
sns.countplot(x='Parch', hue='Survived', data= titanic_train)
#Checking survival with respect to regions
plt.figure(figsize=(4,3))
sns.countplot(x='Embarked', hue='Survived', data= titanic_train)
#Chcking which age group passangers were able to survive the most
g = sns.FacetGrid(titanic_train, row = 'Survived')
g = g.map(plt.hist,'Age')
#Checking Fare with box plot
plt.figure(figsize=(4,3))
sns.boxplot(x='Fare',data=titanic_train)
#Analyzing passenger's age with respect to sex and passenger class
plt.figure(figsize=(6,4))
sns.boxplot(x='Age',y='Sex',hue='Pclass',data=titanic_train)
#Using reg plot for analysis
plt.figure(figsize=(6,5))
sns.regplot(x = 'Age', y= 'Fare',data =titanic_train)
#Using joint plot for analysis
plt.figure(figsize=(6,5))
sns.jointplot(x = 'SibSp' , y = 'Age',data= titanic_train,kind='reg')
#Using cat plot for analysis
plt.figure(figsize=(6,5))
sns.catplot(x='Sex',y='Age',data=titanic_train)
#Using swarm plot for analysis
plt.figure(figsize = (6,5))
sns.swarmplot(y = 'Embarked',x='Age',data=titanic_train,hue='Sex')
#Using violin plot for analysis
plt.figure(figsize=(6,5))
sns.violinplot(y = 'Sex',x ='Age',hue ='Pclass',data =titanic_train )
#Checking corelation between the features
corr=titanic_train.corr()
plt.figure(figsize=(6,5))
sns.heatmap(data= corr, annot=True)
#Checking relation between the variables with pairplot
sns.pairplot(titanic_train)
#Adding train and test data for feature engineering
titanic_train['Source']= 'train'
titanic_test['Source']= 'test'
titanic = pd.concat([titanic_train, titanic_test],ignore_index=True)
titanic.head(2)
#Shape of the train, test and combined data
print(titanic_train.shape, titanic_test.shape, titanic.shape)
#Checking missing values in the combined data
titanic.isnull().sum()
#Filling null values with median in age with respect to SibSp, Parch and Pclass
def fill_age_missing_values(titanic):
    Age_Nan_Indices = list(titanic[titanic["Age"].isnull()].index)

    #for loop that iterates over all the missing age indices
    for index in Age_Nan_Indices:
        #temporary variables to hold SibSp, Parch and Pclass values pertaining to the current index
        temp_Pclass = titanic.iloc[index]["Pclass"]
        temp_SibSp = titanic.iloc[index]["SibSp"]
        temp_Parch = titanic.iloc[index]["Parch"]
        age_median = titanic["Age"][((titanic["Pclass"] == temp_Pclass) & (titanic["SibSp"] == temp_SibSp) & (titanic["Parch"] == temp_Parch))].median()
        if titanic.iloc[index]["Age"]:
            titanic["Age"].iloc[index] = age_median
        if np.isnan(age_median):
            titanic["Age"].iloc[index] = titanic["Age"].median()
    return titanic

titanic = fill_age_missing_values(titanic)
#Filling missing value in Embarked column with mode
titanic['Embarked'].mode()
titanic['Embarked']=titanic['Embarked'].fillna('S')
#Filling missing value in Fare column with median
titanic['Fare']=titanic['Fare'].fillna(titanic['Fare'].median())
#Creating a new column according to the age group
titanic["Age"] = titanic["Age"].astype(int)
titanic.loc[(titanic['Age'] <= 2), 'Age Group'] = 'Baby' 
titanic.loc[((titanic["Age"] > 2) & (titanic['Age'] <= 10)), 'Age Group'] = 'Child' 
titanic.loc[((titanic["Age"] > 10) & (titanic['Age'] <= 19)), 'Age Group'] = 'Young Adult'
titanic.loc[((titanic["Age"] > 19) & (titanic['Age'] <= 60)), 'Age Group'] = 'Adult'
titanic.loc[(titanic["Age"] > 60), 'Age Group'] = 'Senior'
titanic["Age Group"] = titanic["Age Group"].map({"Baby": 0, "Child": 1, "Young Adult": 2, "Adult": 3, "Senior": 4})
#By adding SibSp and Parch we can have a new column like total family
titanic["Ftotal"] = 1 + titanic["SibSp"] + titanic["Parch"]
#By extracting title from passengers name we can create a new column Title which can help us in our analysis
title_titanic = [i.split(",")[1].split(".")[0].strip() for i in titanic["Name"]]
titanic["Title"] = pd.Series(title_titanic)
titanic["Title"].unique()
#Creating a new column called Title and assigning to multiple groups
titanic["Title"] = titanic["Title"].replace(['Lady', 'the Countess','Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
titanic["Title"] = titanic["Title"].map({"Master":0, "Miss":1, "Ms" : 1 , "Mme":1, "Mlle":1, "Mrs":1, "Mr":2, "Rare":3})
titanic["Title"] = titanic["Title"].astype(int)
titanic["Title"].unique()
#Applying one hot encoding to the catagorical variables
sex_titanic=pd.get_dummies(titanic['Sex'],prefix="Sex",drop_first=True)
pclass_titanic=pd.get_dummies(titanic['Pclass'],prefix="PClass",drop_first=True)
emb_titanic=pd.get_dummies(titanic['Embarked'],prefix="Emb",drop_first=True)
name_titanic=pd.get_dummies(titanic['Title'],prefix="Title",drop_first=True)
#Adding encoded columns to final data
titanic=pd.concat([titanic,sex_titanic,pclass_titanic,emb_titanic,name_titanic],axis=1)
#Dropping unnecessary columns from dataset
titanic=titanic.drop(['Sex','Pclass','Cabin','Embarked','SibSp','Parch','Age','Name','Title','Ticket','PassengerId'],axis=1)
titanic.head(2)
#Applying Feature Scaling in Fare caloumn to normalize the data
titanic['Fare']= np.log1p(titanic['Fare'])
titanic.head(2)
#Feature engineering is done so will split the data again into train and test so that we can train the model
titanic_train = titanic.loc[titanic['Source']=="train"]
titanic_test = titanic.loc[titanic['Source']=="test"]
#Dropping Source column which we have created while combining the data 
titanic_train.drop(labels=["Source"],axis = 1,inplace=True)
titanic_test.drop(labels=["Source"],axis = 1,inplace=True)
titanic_test.drop(labels=["Survived"],axis = 1,inplace=True)
#Shape of the train & test data
print(titanic_train.shape, titanic_test.shape)
#Assigning X and y variables
X = titanic_train.drop('Survived',1)
y = titanic_train['Survived']
#Splitting the dataset into train and test
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=123)
lr = LogisticRegression()
lr.fit(X_train,y_train)
lr_pred= lr.predict(X_test)
#Coefficients for Logistic Regression
print(lr.coef_)
print(lr.intercept_)
#Accuracy Score
lr_acc = metrics.accuracy_score(lr_pred,y_test)
lr_acc
#Train Score
lr_train=lr.score(X_train,y_train)
lr_train
#Test Score
lr_test=lr.score(X_test,y_test)
lr_test
#Confusion Matrix
lr_conf=metrics.confusion_matrix(lr_pred,y_test)
plt.figure(figsize=(4,3))
cmap = sns.cubehelix_palette(50, hue=0.05, rot=0, light=0.9, dark=0, as_cmap=True)
sns.heatmap(lr_conf,cmap = cmap,xticklabels=['Prediction No','Prediction Yes'],yticklabels=['Actual No','Actual Yes'], annot=True,fmt='d')
#Classification Report
print(metrics.classification_report(lr_pred,y_test))
#Null Accuracy
y_test.value_counts()
y_test.value_counts().head(1) / len(y_test)
#ROC Curve
predict_probabilities = lr.predict_proba(X_test)
fpr, tpr, thresholds = metrics.roc_curve(y_test, predict_probabilities[:,1])

plt.figure(figsize=(4,3))
plt.plot(fpr, tpr)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.title('ROC curve')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.grid(True)
dt= DecisionTreeClassifier()
dt.fit(X_train,y_train)
dt_pred=dt.predict(X_test)
#Accuracy Score
dt_acc = metrics.accuracy_score(dt_pred,y_test)
dt_acc
#Train Score
dt_train=dt.score(X_train,y_train)
dt_train
#Test Score
dt_test=dt.score(X_test,y_test)
dt_test
#Confusion Matrix
dt_conf=metrics.confusion_matrix(dt_pred,y_test)
plt.figure(figsize=(4,3))
cmap = sns.cubehelix_palette(50, hue=0.05, rot=0, light=0.9, dark=0, as_cmap=True)
sns.heatmap(dt_conf,cmap = cmap,xticklabels=['Prediction No','Prediction Yes'],yticklabels=['Actual No','Actual Yes'], annot=True,fmt='d')
#Classification Report
print(metrics.classification_report(dt_pred,y_test))
#ROC Curve
predict_probabilities = dt.predict_proba(X_test)
fpr, tpr, thresholds = metrics.roc_curve(y_test, predict_probabilities[:,1])

plt.figure(figsize=(4,3))
plt.plot(fpr, tpr)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.title('ROC curve')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.grid(True)
rf=RandomForestClassifier()
rf.fit(X_train,y_train)
rf_pred=rf.predict(X_test)
#Accuracy Score
rf_acc = metrics.accuracy_score(rf_pred,y_test)
rf_acc
#Train Score
rf_train=rf.score(X_train,y_train)
rf_train
#Test Score
rf_test=rf.score(X_test,y_test)
rf_test
#Confusion Matrix
rf_conf=metrics.confusion_matrix(rf_pred,y_test)
plt.figure(figsize=(4,3))
cmap = sns.cubehelix_palette(50, hue=0.05, rot=0, light=0.9, dark=0, as_cmap=True)
sns.heatmap(rf_conf,cmap = cmap,xticklabels=['Prediction No','Prediction Yes'],yticklabels=['Actual No','Actual Yes'], annot=True,fmt='d')
#Classification Report
print(metrics.classification_report(rf_pred,y_test))
#ROC Curve
predict_probabilities = rf.predict_proba(X_test)
fpr, tpr, thresholds = metrics.roc_curve(y_test, predict_probabilities[:,1])

plt.figure(figsize=(4,3))
plt.plot(fpr, tpr)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.title('ROC curve')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.grid(True)
knn = KNeighborsClassifier()
knn.fit(X_train,y_train)
knn_pred = knn.predict(X_test)
#Accuracy Score
knn_acc = metrics.accuracy_score(knn_pred,y_test)
knn_acc
#Train Score
knn_train=knn.score(X_train,y_train)
knn_train
#Test Score
knn_test=knn.score(X_test,y_test)
knn_test
#Confusion Matrix
knn_conf=metrics.confusion_matrix(knn_pred,y_test)
plt.figure(figsize=(4,3))
cmap = sns.cubehelix_palette(50, hue=0.05, rot=0, light=0.9, dark=0, as_cmap=True)
sns.heatmap(knn_conf,cmap = cmap,xticklabels=['Prediction No','Prediction Yes'],yticklabels=['Actual No','Actual Yes'], annot=True,fmt='d')
#Classification Report
print(metrics.classification_report(knn_pred,y_test))
#ROC Curve
predict_probabilities = knn.predict_proba(X_test)
fpr, tpr, thresholds = metrics.roc_curve(y_test, predict_probabilities[:,1])

plt.figure(figsize=(4,3))
plt.plot(fpr, tpr)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.title('ROC curve')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.grid(True)
ab = AdaBoostClassifier()
ab.fit(X_train,y_train)
ab_pred = ab.predict(X_test)
#Accuracy Score
ab_acc = metrics.accuracy_score(ab_pred,y_test)
ab_acc
#Train Score
ab_train=ab.score(X_train,y_train)
ab_train
#Test Score
ab_test=ab.score(X_test,y_test)
ab_test
#Confusion Matrix
ab_conf=metrics.confusion_matrix(ab_pred,y_test)
plt.figure(figsize=(4,3))
cmap = sns.cubehelix_palette(50, hue=0.05, rot=0, light=0.9, dark=0, as_cmap=True)
sns.heatmap(ab_conf,cmap = cmap,xticklabels=['Prediction No','Prediction Yes'],yticklabels=['Actual No','Actual Yes'], annot=True,fmt='d')
#Classification Report
print(metrics.classification_report(ab_pred,y_test))
#ROC Curve
predict_probabilities = ab.predict_proba(X_test)
fpr, tpr, thresholds = metrics.roc_curve(y_test, predict_probabilities[:,1])

plt.figure(figsize=(4,3))
plt.plot(fpr, tpr)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.title('ROC curve')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.grid(True)
gb = GradientBoostingClassifier()
gb.fit(X_train,y_train)
gb_pred = gb.predict(X_test)
#Accuracy Score
gb_acc = metrics.accuracy_score(gb_pred,y_test)
gb_acc
#Train Score
gb_train=gb.score(X_train,y_train)
gb_train
#Test Score
gb_test=gb.score(X_test,y_test)
gb_test
#Confusion Matrix
gb_conf=metrics.confusion_matrix(gb_pred,y_test)
plt.figure(figsize=(4,3))
cmap = sns.cubehelix_palette(50, hue=0.05, rot=0, light=0.9, dark=0, as_cmap=True)
sns.heatmap(gb_conf,cmap = cmap,xticklabels=['Prediction No','Prediction Yes'],yticklabels=['Actual No','Actual Yes'], annot=True,fmt='d')
#Classification Report
print(metrics.classification_report(gb_pred,y_test))
#ROC Curve
predict_probabilities = gb.predict_proba(X_test)
fpr, tpr, thresholds = metrics.roc_curve(y_test, predict_probabilities[:,1])

plt.figure(figsize=(4,3))
plt.plot(fpr, tpr)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.title('ROC curve')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.grid(True)
xgb= XGBClassifier()
xgb.fit(X_train,y_train)
xgb_pred=xgb.predict(X_test)
#Accuracy Score
xgb_acc = metrics.accuracy_score(xgb_pred,y_test)
xgb_acc
#Train Score
xgb_train=xgb.score(X_train,y_train)
xgb_train
#Test Score
xgb_test=xgb.score(X_test,y_test)
xgb_test
#Confusion Matrix
xgb_conf=metrics.confusion_matrix(xgb_pred,y_test)
plt.figure(figsize=(4,3))
cmap = sns.cubehelix_palette(50, hue=0.05, rot=0, light=0.9, dark=0, as_cmap=True)
sns.heatmap(xgb_conf,cmap = cmap,xticklabels=['Prediction No','Prediction Yes'],yticklabels=['Actual No','Actual Yes'], annot=True,fmt='d')
#Classification Report
print(metrics.classification_report(xgb_pred,y_test))
#ROC Curve
predict_probabilities = xgb.predict_proba(X_test)
fpr, tpr, thresholds = metrics.roc_curve(y_test, predict_probabilities[:,1])

plt.figure(figsize=(4,3))
plt.plot(fpr, tpr)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.title('ROC curve')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.grid(True)
lgbm = lgb.LGBMClassifier()
lgbm.fit(X_train,y_train)
lgbm_pred = lgbm.predict(X_test)
#Accuracy Score
lgbm_acc = metrics.accuracy_score(lgbm_pred,y_test)
lgbm_acc
#Train Score
lgbm_train=lgbm.score(X_train,y_train)
lgbm_train
#Test Score
lgbm_test=lgbm.score(X_test,y_test)
lgbm_test
#Confusion Matrix
lgbm_conf=metrics.confusion_matrix(lgbm_pred,y_test)
plt.figure(figsize=(4,3))
cmap = sns.cubehelix_palette(50, hue=0.05, rot=0, light=0.9, dark=0, as_cmap=True)
sns.heatmap(lgbm_conf,cmap = cmap,xticklabels=['Prediction No','Prediction Yes'],yticklabels=['Actual No','Actual Yes'], annot=True,fmt='d')
#Classification Report
print(metrics.classification_report(lgbm_pred,y_test))
#ROC Curve
predict_probabilities = lgbm.predict_proba(X_test)
fpr, tpr, thresholds = metrics.roc_curve(y_test, predict_probabilities[:,1])

plt.figure(figsize=(4,3))
plt.plot(fpr, tpr)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.title('ROC curve')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.grid(True)
svm = SVC()
svm.fit(X_train,y_train)
svm_pred= svm.predict(X_test)
#Accuracy Score
svm_acc = metrics.accuracy_score(svm_pred,y_test)
svm_acc
#Train Score
svm_train=svm.score(X_train,y_train)
svm_train
#Test Score
svm_test=svm.score(X_test,y_test)
svm_test
#Confusion Matrix
svm_conf=metrics.confusion_matrix(svm_pred,y_test)
plt.figure(figsize=(4,3))
cmap = sns.cubehelix_palette(50, hue=0.05, rot=0, light=0.9, dark=0, as_cmap=True)
sns.heatmap(svm_conf,cmap = cmap,xticklabels=['Prediction No','Prediction Yes'],yticklabels=['Actual No','Actual Yes'], annot=True,fmt='d')
#Classification Report
print(metrics.classification_report(svm_pred,y_test))
metrics = {'Metrics': ['Train Score','Test Score','Model Accuracy'],'Logistic Regression':[lr_train,lr_test,lr_acc],
          'Decision Tree Classifier':[dt_train,dt_test,dt_acc],'Random Forest Classifier':[rf_train,rf_test,rf_acc],
           'KNeighborsClassifier':[knn_train,knn_test,knn_acc],'AdaBoostClassifier':[ab_train,ab_test,ab_acc],
          'GradientBoostingClassifier':[gb_train,gb_test,gb_acc],'XG Boost Classifier':[xgb_train,xgb_test,xgb_acc],
           'LGBMClassifier':[lgbm_train,lgbm_test,lgbm_acc],'SVMClassifier':[svm_train,svm_test,svm_acc]}
metrics = pd.DataFrame(metrics)
metrics
dt_grid = {'max_features' : ['auto', 'sqrt'],
              'max_depth' : np.arange(1,20),
           'criterion':['gini','entropy'],
           "max_leaf_nodes": [20,30],
              'min_samples_split':[2,5,10],
              'min_samples_leaf':[1,2,4]}

dt = DecisionTreeClassifier()
dt_gs = GridSearchCV(dt, dt_grid, cv = 3, n_jobs=-1, verbose=2)
dt_gs.fit(X_train, y_train)
dt_gs_pred = dt_gs.predict(X_test)
dt_gs.best_estimator_
#Accuracy Score
dt_tune_gs_acc = metrics.accuracy_score(dt_gs_pred,y_test)
dt_tune_gs_acc
#Train Score
dt_tune_gs_train=dt_gs.score(X_train,y_train)
dt_tune_gs_train
#Test Score
dt_tune_gs_test=dt_gs.score(X_test,y_test)
dt_tune_gs_test
rf_grid = {'n_estimators': range(5,20,2),
              'max_features' : ['auto', 'sqrt'],
              'max_depth' : [10,20,30,40],
              'min_samples_split':[2,5,10],
              'min_samples_leaf':[1,2,4]}

rf = RandomForestClassifier()
rf_gs = GridSearchCV(rf, rf_grid, cv = 3, n_jobs=-1, verbose=2)

rf_gs.fit(X_train, y_train)
rf_gs_pred = rf_gs.predict(X_test)
rf_gs.best_estimator_
#Accuracy Score
rf_tune_gs_acc = metrics.accuracy_score(rf_gs_pred,y_test)
rf_tune_gs_acc
#Train Score
rf_tune_gs_train=rf_gs.score(X_train,y_train)
rf_tune_gs_train
#Test Score
rf_tune_gs_test=rf_gs.score(X_test,y_test)
rf_tune_gs_test
knn_grid = {'leaf_size':np.arange(1,50),'n_neighbors':np.arange(1,30),'p':[1,2]}

knn = KNeighborsClassifier()
knn_gs = GridSearchCV(knn, knn_grid, cv = 3, n_jobs=-1, verbose=2)

knn_gs.fit(X_train, y_train)
knn_gs_pred = knn_gs.predict(X_test)
knn_gs.best_estimator_
#Accuracy Score
knn_tune_gs_acc = metrics.accuracy_score(knn_gs_pred,y_test)
knn_tune_gs_acc
#Train Score
knn_tune_gs_train=knn_gs.score(X_train,y_train)
knn_tune_gs_train
#Test Score
knn_tune_gs_test=knn_gs.score(X_test,y_test)
knn_tune_gs_test
ab_grid = {"n_estimators": range(5,20,2) ,  
              "learning_rate": [0.01,0.05,0.1,0.5,1]}

ab = AdaBoostClassifier()
ab_gs = GridSearchCV(ab, ab_grid, cv = 3, n_jobs=-1, verbose=2)

ab_gs.fit(X_train, y_train)
ab_gs_pred = ab_gs.predict(X_test)
ab_gs.best_estimator_
#Accuracy Score
ab_tune_gs_acc = metrics.accuracy_score(ab_gs_pred,y_test)
ab_tune_gs_acc
#Train Score
ab_tune_gs_train=ab_gs.score(X_train,y_train)
ab_tune_gs_train
#Test Score
ab_tune_gs_test=ab_gs.score(X_test,y_test)
ab_tune_gs_test
gb_grid = {"n_estimators": range(5,20,2) ,  
              "learning_rate": [0.01,0.05,0.1,0.5,1]}

gb = GradientBoostingClassifier()
gb_gs = GridSearchCV(gb, gb_grid, cv = 3, n_jobs=-1, verbose=2)

gb_gs.fit(X_train, y_train)
gb_gs_pred = gb_gs.predict(X_test)
gb_gs.best_estimator_
#Accuracy Score
gb_tune_gs_acc = metrics.accuracy_score(gb_gs_pred,y_test)
gb_tune_gs_acc
#Train Score
gb_tune_gs_train=gb_gs.score(X_train,y_train)
gb_tune_gs_train
#Test Score
gb_tune_gs_test=gb_gs.score(X_test,y_test)
gb_tune_gs_test
xgb_grid = {"max_depth": [10,15,20,30],
              "n_estimators": range(5,20,2) , 
              "gamma": [0.03,0.05], 
              "learning_rate": [0.01,0.05]}

xgb = XGBClassifier()
xgb_gs = GridSearchCV(xgb, xgb_grid, cv = 3, n_jobs=-1, verbose=2)

xgb_gs.fit(X_train, y_train)
xgb_gs_pred = xgb_gs.predict(X_test)
xgb_gs.best_estimator_
#Accuracy Score
xgb_tune_gs_acc = metrics.accuracy_score(xgb_gs_pred,y_test)
xgb_tune_gs_acc
#Train Score
xgb_tune_gs_train=xgb_gs.score(X_train,y_train)
xgb_tune_gs_train
#Test Score
xgb_tune_gs_test=xgb_gs.score(X_test,y_test)
xgb_tune_gs_test
lgbm_grid = {"max_depth": [10,15,20,30],
              "n_estimators": range(5,20,2), 
              "learning_rate": [0.01,0.05]}

lgbm = lgb.LGBMClassifier()
lgbm_gs = GridSearchCV(lgbm, lgbm_grid, cv = 3, n_jobs=-1, verbose=2)

lgbm_gs.fit(X_train, y_train)
lgbm_gs_pred = lgbm_gs.predict(X_test)
lgbm_gs.best_estimator_
#Accuracy Score
lgbm_tune_gs_acc = metrics.accuracy_score(lgbm_gs_pred,y_test)
lgbm_tune_gs_acc
#Train Score
lgbm_tune_gs_train=lgbm_gs.score(X_train,y_train)
lgbm_tune_gs_train
#Test Score
lgbm_tune_gs_test=lgbm_gs.score(X_test,y_test)
lgbm_tune_gs_test
svm_grid = {'kernel':['linear','rbf'],'decision_function_shape': ['ovr','ovr'],'class_weight':['balanced', None]}

svm = SVC()
svm_gs = GridSearchCV(svm, svm_grid, cv = 3, n_jobs=-1, verbose=2)

svm_gs.fit(X_train, y_train)
svm_gs_pred = svm_gs.predict(X_test)
svm_gs.best_estimator_
svm_tune_gs_acc = metrics.accuracy_score(svm_gs_pred,y_test)
svm_tune_gs_acc
#Train Score
svm_tune_gs_train=svm_gs.score(X_train,y_train)
svm_tune_gs_train
#Test Score
svm_tune_gs_test=svm_gs.score(X_test,y_test)
svm_tune_gs_test
#Finding optimal parameters using Randomized Search CV
params1 = {'max_depth': np.arange(1,20),'criterion':['gini','entropy'],"max_leaf_nodes": [20,30]}
dt = DecisionTreeClassifier()
tree = RandomizedSearchCV(dt, params1, cv=3 , return_train_score = True)
tree.fit(X,y)
tree.best_params_
dtr = DecisionTreeClassifier(criterion='gini',max_depth=12,max_leaf_nodes=30)
dtr.fit(X_train,y_train)
dtr_pred = dtr.predict(X_test)
#Accuracy Score
dt_tune_rs_acc = metrics.accuracy_score(dtr_pred,y_test)
dt_tune_rs_acc
#Train Score
dt_tune_rs_train=dtr.score(X_train,y_train)
dt_tune_rs_train
#Test Score
dt_tune_rs_test=dtr.score(X_test,y_test)
dt_tune_rs_test
#Finding optimal parameters using Randomized Search CV
params2 = {'n_estimators': np.arange(1,20),'criterion':['entropy','gini'],'max_leaf_nodes':[10,20,30],'max_depth':np.arange(1,20)}
rf = RandomForestClassifier()
forest = RandomizedSearchCV(rf, params2, cv=3 , return_train_score = True)
forest.fit(X,y)
forest.best_params_
rfr = RandomForestClassifier(criterion='gini',max_depth=8,n_estimators=4,max_leaf_nodes=30)
rfr.fit(X_train,y_train)
rfr_pred = rfr.predict(X_test)
#Accuracy Score
rf_tune_rs_acc = metrics.accuracy_score(rfr_pred,y_test)
rf_tune_rs_acc
#Train Score
rf_tune_rs_train=rfr.score(X_train,y_train)
rf_tune_rs_train
#Test Score
rf_tune_rs_test=rfr.score(X_test,y_test)
rf_tune_rs_test
#Finding optimal parameters using Randomized Search CV
params3 = {'leaf_size':np.arange(1,50),'n_neighbors':np.arange(1,30),'p':[1,2]}
knn = KNeighborsClassifier()
neighbor = RandomizedSearchCV(knn, params3, cv=3 , return_train_score = True)
neighbor.fit(X,y)
neighbor.best_params_
knnr = KNeighborsClassifier(n_neighbors=28,leaf_size=28, p=2)
knnr.fit(X_train,y_train)
knnr_pred = knnr.predict(X_test)
#Accuracy Score
knn_tune_rs_acc = metrics.accuracy_score(knnr_pred,y_test)
knn_tune_rs_acc
#Train Score
knn_tune_rs_train=knnr.score(X_train,y_train)
knn_tune_rs_train
#Test Score
knn_tune_rs_test=knnr.score(X_test,y_test)
knn_tune_rs_test
#Finding optimal parameters using Randomized Search CV
params4 = {"n_estimators": range(5,20,2) ,  
              "learning_rate": [0.01,0.05,0.1,0.5,1]}
ab = AdaBoostClassifier()
AB = RandomizedSearchCV(ab,param_distributions=params4,
                           cv = 5,
                           n_jobs=-1,
                           verbose=2)
AB.fit(X,y)
AB.best_params_
abr = AdaBoostClassifier(n_estimators=15,learning_rate=1)
abr.fit(X_train,y_train)
abr_pred = abr.predict(X_test)
#Accuracy Score
ab_tune_rs_acc = metrics.accuracy_score(abr_pred,y_test)
ab_tune_rs_acc
#Train Score
ab_tune_rs_train=abr.score(X_train,y_train)
ab_tune_rs_train
#Test Score
ab_tune_rs_test=abr.score(X_test,y_test)
ab_tune_rs_test
#Finding optimal parameters using Randomized Search CV
params5 = {"n_estimators": range(5,20,2) ,  
              "learning_rate": [0.01,0.05,0.1,0.5,1]}
gb = GradientBoostingClassifier()
GB = RandomizedSearchCV(gb,param_distributions=params5,
                           cv = 5,
                           n_jobs=-1,
                           verbose=2)
GB.fit(X,y)
GB.best_params_
gbr = GradientBoostingClassifier(n_estimators=17,learning_rate=1)
gbr.fit(X_train,y_train)
gbr_pred = gbr.predict(X_test)
#Accuracy Score
gb_tune_rs_acc = metrics.accuracy_score(gbr_pred,y_test)
gb_tune_rs_acc
#Train Score
gb_tune_rs_train=gbr.score(X_train,y_train)
gb_tune_rs_train
#Test Score
gb_tune_rs_test=gbr.score(X_test,y_test)
gb_tune_rs_test
#Finding optimal parameters using Randomized Search CV
params6 = {"max_depth": [10,15,20,30],
              "n_estimators": range(5,20,2) , 
              "gamma": [0.03,0.05], 
              "learning_rate": [0.01,0.05]}
 
xgb = XGBClassifier()
XGB = RandomizedSearchCV(xgb,param_distributions=params6,
                           cv = 5)
XGB.fit(X,y)
XGB.best_params_
xgbr = XGBClassifier(n_estimators=13,max_depth=20,learning_rate=0.01,gamma=0.05)
xgbr.fit(X_train,y_train)
xgbr_pred = xgbr.predict(X_test)
#Accuracy Score
xgb_tune_rs_acc = metrics.accuracy_score(xgbr_pred,y_test)
xgb_tune_rs_acc
#Train Score
xgb_tune_rs_train=xgbr.score(X_train,y_train)
xgb_tune_rs_train
#Test Score
xgb_tune_rs_test=xgbr.score(X_test,y_test)
xgb_tune_rs_test
#Finding optimal parameters using Randomized Search CV
params7 = {"max_depth": [10,15,20,30],
              "n_estimators": range(5,20,2), 
              "learning_rate": [0.01,0.05]} 
lgbm = lgb.LGBMClassifier()
LGBM = RandomizedSearchCV(lgbm,param_distributions=params7,
                           cv = 5)
LGBM.fit(X,y)
LGBM.best_params_
lgbmr = lgb.LGBMClassifier(n_estimators=19,max_depth=30,learning_rate=0.05)
lgbmr.fit(X_train,y_train)
lgbmr_pred = lgbmr.predict(X_test)
#Accuracy Score
lgbm_tune_rs_acc = metrics.accuracy_score(lgbmr_pred,y_test)
lgbm_tune_rs_acc
#Train Score
lgbm_tune_rs_train=lgbmr.score(X_train,y_train)
lgbm_tune_rs_train
#Test Score
lgbm_tune_rs_test=lgbmr.score(X_test,y_test)
lgbm_tune_rs_test
#Finding optimal parameters using Randomized Search CV
params8 = {'kernel':['linear','rbf'],'decision_function_shape': ['ovr','ovr'],'class_weight':['balanced', None]} 
svm = SVC()
SVM = RandomizedSearchCV(svm,param_distributions=params8,
                           cv = 5)
SVM.fit(X,y)
SVM.best_params_
svmr = SVC(kernel ='rbf',decision_function_shape = 'ovr',class_weight=None)
svmr.fit(X_train,y_train)
svmr_pred = svmr.predict(X_test)
#Accuracy Score
svm_tune_rs_acc = metrics.accuracy_score(svmr_pred,y_test)
svm_tune_rs_acc
#Train Score
svm_tune_rs_train=svmr.score(X_train,y_train)
svm_tune_rs_train
#Test Score
svm_tune_rs_test=svmr.score(X_test,y_test)
svm_tune_rs_test
#Creating dictionary for all the metrics and models
metrics = {'Metrics': ['Train Score','Train Score after GridSearchCV','Train Score after RandomizedSearchCV','Test Score','Test Score after GridSearchCV','Test Score after RandomizedSearchCV','Model Accuracy','Model Accuracy after GridSearchCV','Model Accuracy after RandomizedSearchCV'],'Logistic Regression':[lr_train,'NA','NA',lr_test,'NA','NA',lr_acc,'NA','NA'],
          'Decision Tree Classifier':[dt_train,dt_tune_gs_train,dt_tune_rs_train,dt_test,dt_tune_gs_test,dt_tune_rs_test,dt_acc,dt_tune_gs_acc,dt_tune_rs_acc],'Ramdom Forest Classifier':[rf_train,rf_tune_gs_train,rf_tune_rs_train,rf_test,rf_tune_gs_test,rf_tune_rs_test,rf_acc,rf_tune_gs_acc,rf_tune_rs_acc],'KNearestNeighbor Classifier':[knn_train,knn_tune_gs_train,knn_tune_rs_train,knn_test,knn_tune_gs_test,knn_tune_rs_test,knn_acc,knn_tune_gs_acc,knn_tune_rs_acc],'Ada Boost Classifier':[ab_train,ab_tune_gs_train,ab_tune_rs_train,ab_test,ab_tune_gs_test,ab_tune_rs_test,ab_acc,ab_tune_gs_acc,ab_tune_rs_acc],'Gradient Boosting Classifier':[gb_train,gb_tune_gs_train,gb_tune_rs_train,gb_test,gb_tune_gs_test,gb_tune_rs_test,gb_acc,gb_tune_gs_acc,gb_tune_rs_acc],
          'XG Boost Classifier':[xgb_train,xgb_tune_gs_train,xgb_tune_rs_train,xgb_test,xgb_tune_gs_test,xgb_tune_rs_test,xgb_acc,xgb_tune_gs_acc,xgb_tune_rs_acc],'LGBM Classifier':[lgbm_train,lgbm_tune_gs_train,lgbm_tune_rs_train,lgbm_test,lgbm_tune_gs_test,lgbm_tune_rs_test,lgbm_acc,lgbm_tune_gs_acc,lgbm_tune_rs_acc],'Support Vector Classifier':[svm_train,svm_tune_gs_train,svm_tune_rs_train,svm_test,svm_tune_gs_test,svm_tune_rs_test,svm_acc,svm_tune_gs_acc,svm_tune_rs_acc]}
#Converting dictionary to dataframe
metrics = pd.DataFrame(metrics)
metrics
#Assigning estimator models for voting classifier
vote_est = [('rf_gs',rf_gs),('knn_gs',knn_gs),('gbr',gbr)]
vote = VotingClassifier(estimators=vote_est,voting='soft')
vote.fit(X_train,y_train)
vote_pred = vote.predict(X_test)
#Accuracy Score
vote_acc = metrics.accuracy_score(vote_pred,y_test)
vote_acc
#Train Score
vote_train=vote.score(X_train,y_train)
vote_train
#Test Score
vote_test=vote.score(X_test,y_test)
vote_test
#Fitting and training the model
st = StackingClassifier(classifiers=[rf_gs,knn_gs,gbr],meta_classifier=lr)
st.fit(X_train,y_train)
st_pred = st.predict(X_test)
#Accuracy Score
st_acc = metrics.accuracy_score(st_pred,y_test)
st_acc
#Train Score
st_train=st.score(X_train,y_train)
st_train
#Test Score
st_test=st.score(X_test,y_test)
st_test
#Creating dictionary for all the metrics and converting it to dataframe
metrics_stack = {'Models': ['Voting Classifier','Stacking Classifier'],'Train score':[vote_train,st_train],'Test Score':[vote_test,st_test],'Model Accuracy':[vote_acc,st_acc]}
metrics_stack = pd.DataFrame(metrics_stack)
metrics_stack
#1st level model
models = [rf,knn,gbr]
S_train, S_test = stacking(models, X_train, y_train, X_test, 
    regression = False, metric = metrics.accuracy_score, n_folds = 4 , 
    shuffle = True, random_state = 0, verbose = 2)
#2nd level model
models = [dt,xgb,gb,ab]
S_train, S_test = stacking(models, X_train, y_train, X_test, 
    regression = False, metric = metrics.accuracy_score, n_folds = 4 , 
    shuffle = True, random_state = 0, verbose = 2)