#Import the necessary libraries

import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import scipy

import sklearn
#We combine the data from "Test.xlsx" and "Train.xlsx"

#This is because we want the data to undergo same transformation so that it will be easier for our model to make predictions for... 

#.."Test.xlsx"

data_train=pd.read_excel('../input/Train.csv')

print(f'Train data contains {data_train.shape} rows X columns')

data_test=pd.read_excel('../input/Test.csv')

print(f'Test data contains {data_test.shape} rows X columns')

data=pd.concat([data_train,data_test],axis=0,ignore_index=True)

print(f'Combined dataset contains {data.shape} rows X columns')

data.head()
#Check for duplicate entries in the rows and in the ID coulmn

x=data.duplicated(keep='first').sum()

print(f'Number of duplicate rows={x}')

x=data['ID'].duplicated(keep='first').sum()

print(f'Number of IDs={x}')
#check the data types in each column to ensure that they are in line with the column name.. eg. "Age" should contain only numbers

data.dtypes
#Check for any inconsistencies in categorial features. We find that there are no inconsistencies! 

for x in ["Divison","Battalion","Education Stream", 'Gender','Channel of Recruitment','Recommendation from Superior',"Last Year's Performance Rating"]:

    print(data[x].value_counts())
#Check whether there are any missing (NA) values in the data. The 5479 missing values in "Recognition" column can be ignored....

#... as they are the values that we want to predict. There are a large number of missing values in "Education Stream"...

#... and "Last Year's Performance Rating"

data.isnull().sum()
#We replace the missing values by the most frequently appearing values in the columns "Education Stream" & "Last Year's Performance Rating"

data['Education Stream'].fillna('Arts',inplace=True) 

data["Last Year's Performance Rating"].fillna('Good',inplace=True)

data.isnull().sum()
#We check to see whether there are any inconsistencies in continuous features. We see no inconsistencies

data.describe()
#Now we drop the ID label as it is of no use in making predictions.

#We convert the categorical variables into numerical variables using one hot encoding. The features that have no ordinal.....

#...relatoinship are converted to dummy variables

data.drop(labels="ID",inplace=True,axis=1)

data=pd.get_dummies(data=data,columns=["Divison","Battalion","Education Stream","Gender","Channel of Recruitment"]);

data["Last Year's Performance Rating"].replace(['Good', 'Sufficient', 'Very Good', 'Satisfactory', 'Excellent'],[2,0,3,1,4], inplace=True)

data["Recommendation from Superior"].replace(['No','Yes'],[0,1],inplace=True)

from sklearn.preprocessing import MinMaxScaler

scl=MinMaxScaler((0,1),copy=False)

data[["Age","Number of Operations Conducted","Service Length(in years)","Training Score(out of 100)"]]= scl.fit_transform(data[["Age","Number of Operations Conducted","Service Length(in years)","Training Score(out of 100)"]])

data.head()
#Now that the transformation is complete, we segregate the data into test and train

data_train=data.iloc[:(data_train.shape[0]),:]

data_test=data.iloc[:(data_test.shape[0]),:]

print(data_train.shape,data_test.shape)
#We calculate the proportions of 1 i.e. Medal awarded and 0 i.e. Medal not awarded in the "Recognition" column

#The result shows that the dataset is highly imbalanced (as expected), hence we would have to take this into consideration...

#... while developing our model

count_rej=len(data_train[data_train['Recognition']==0])

count_accept = len(data_train[data_train['Recognition']==1])

pct_of_rej = count_rej/(count_rej+count_accept)

print("Percentage of non-awardees", float("{0:.2f}".format(pct_of_rej*100)))

pct_of_accept = count_accept/(count_rej+count_accept)

print("Percentage of awardees", float("{0:.2f}".format(pct_of_accept*100)))

sns.countplot(x='Recognition',data=data_train,palette='hls')

plt.show()
#We plot the histograms for continuous variables to check if there are any outliers present

#We can see from the plot that there are no outliers

data_train.hist(column=["Age","Number of Operations Conducted","Service Length(in years)","Number of Awards Won in Past","Training Score(out of 100)"],figsize=(15,15), bins=40)

plt.show()
#Multi-collinearity in the features can lead to creation of inferior models

#We check the correlation between the features and find that the features are not highly correlated

data1 = data_train[["Age","Number of Operations Conducted","Service Length(in years)","Number of Awards Won in Past","Training Score(out of 100)"]]

corrmat=data1.corr()

fig=plt.figure(figsize=(7,5))

sns.heatmap(corrmat,cmap="YlGnBu",square=True, annot=True)

plt.show()
y=data_train['Recognition']

X=data_train.drop('Recognition',axis=1)

from sklearn.model_selection import train_test_split

from imblearn.over_sampling import SMOTE

os = SMOTE(random_state=0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

columns = X_train.columns

os_data_train_X,os_data_train_y=os.fit_sample(X_train, y_train)

os_data_train_X = pd.DataFrame(data=os_data_train_X,columns=columns)

os_data_train_y= pd.DataFrame(data=os_data_train_y,columns=['y'])

print("length of oversampled data_train is ",len(os_data_train_X))

print("Number of non-awardees in oversampled data_train",len(os_data_train_y[os_data_train_y['y']==0]))

print("Number of awardees",len(os_data_train_y[os_data_train_y['y']==1]))

print("Proportion of awardees in oversampled data_train is ",len(os_data_train_y[os_data_train_y['y']==0])/len(os_data_train_X))

print("Proportion of non-awardees in oversampled data_train is ",len(os_data_train_y[os_data_train_y['y']==1])/len(os_data_train_X))

X_train=os_data_train_X

y_train=os_data_train_y.values.ravel()
#We have to first optimise the hyperparameters of the XGBCLassifier

#We use Random seach for this as it is computationally less expensive and our laptops don't have sufficient processing powers..

#...to optimise parameters using gridsearchcv in less time

from sklearn.model_selection import RandomizedSearchCV

from xgboost import XGBClassifier

XGB=XGBClassifier()

random_grid = {

    'max_depth':[10,11,12],

    'min_child_weight':[1,2],

    'gamma':[i/10.0 for i in range(1,5)],

    'subsample':[i/10.0 for i in range(6,10)],

    'colsample_bytree':[i/10.0 for i in range(6,10)],

    'reg_alpha':[0.01, 0.05, 0.1]

}

xg=RandomizedSearchCV(estimator = XGB, param_distributions = random_grid, n_iter = 100, cv = 2, verbose=2, random_state=42, n_jobs = -1)

xg.fit(X_train, y_train)
#We get the best hyperparameters from the RandomSearchCV which can be used for our model

print(random_grid)

xg.best_params_
#We train the model and test it using performance metrics such as classification report, confusion matrix and accuracy score..

#...though accuracy score is not a relevent metric here as the data is imbalanced. Hence more importance should be given to...

#...the weighted average F1 score from classification report. F! score for our model is 0.93

from xgboost import XGBClassifier

model=XGBClassifier(subsample= 0.8,

 reg_alpha= 0.05,

 min_child_weight= 1,

 max_depth= 12,

 gamma= 0.1,

 colsample_bytree= 0.8,objective= 'binary:logistic',n_jobs = -1, seed=27)

model.fit(X_train,y_train)

y_pred = model.predict(X_test)

print('Accuracy of classifier on test set: {:.2f}'.format(model.score(X_test, y_test)))

from yellowbrick.classifier import ConfusionMatrix

cm = ConfusionMatrix(model, classes=[0,1])

cm.fit(X_train, y_train)

cm.score(X_test, y_test)

cm.poof()

from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))
#We also use ROC AUC as a metric for evaluating classificatio models. It gives us and idea of how good our model is as compared..

#...to a random model that predicts the majority class every time.=

#The ROC AUC is 0.69 for our model while that of a random model is 0.50

from sklearn.metrics import roc_auc_score

from sklearn.metrics import roc_curve

logit_roc_auc = roc_auc_score(y_test, model.predict(X_test))

fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(X_test)[:,1])

plt.figure()

plt.plot(fpr, tpr, label='XGBClassifier (area = %0.2f)' % logit_roc_auc)

plt.plot([0, 1], [0, 1],'r--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Receiver operating characteristic')

plt.legend(loc="lower right")

plt.show()
#Drop the "Recognition" column from the test data to make predictions from the constructed model

data_test=data_test.drop('Recognition',axis=1)

data_test.shape
#Make predictions for the test data

predictions=model.predict(data_test)

predicted_data=pd.read_excel('../input/Test.csv')

predicted_data['Recognition']=predictions

predicted_data.head()
predicted_data["Recognition"].value_counts()