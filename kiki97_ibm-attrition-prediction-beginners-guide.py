# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns
#reading the data 

data = pd.read_csv('../input/ibm-hr-analytics-attrition-dataset/WA_Fn-UseC_-HR-Employee-Attrition.csv')

data.head()
# exploring the data to get a basic idea of the data

data.describe()

print(data.shape)

print(data.dtypes)

print(data.dtypes.value_counts())

# dropping the columns ['Over18','StandardHours','EmployeeCount','EmployeeNumber'] as they all have the same value.

data=data.drop(['Over18','StandardHours','EmployeeCount','EmployeeNumber'],axis=1)

data.shape
#creating dummy values for categorical variable.

char_cols = data.dtypes.pipe(lambda x: x[x == 'object']).index

label_mapping = {}



for c in char_cols:

    data[c], label_mapping[c] = pd.factorize(data[c])

data.head()
#now changing data type of categorical variables into category

data[['BusinessTravel','Department','EducationField','Gender',

      'JobRole','MaritalStatus','OverTime']]=data[['BusinessTravel','Department','EducationField','Gender',

      'JobRole','MaritalStatus','OverTime']].astype('category')

print(data.dtypes)
#obtaining the descriptive statistics of the data.

data.describe().T
# creating boxplot to check for outliers

data.boxplot(column=['Age', 'DailyRate','DistanceFromHome', 'Education','EnvironmentSatisfaction'])
data.boxplot(column=['HourlyRate', 'JobInvolvement','JobLevel', 'JobSatisfaction','MonthlyIncome'])
data.boxplot(column=['TotalWorkingYears', 'TrainingTimesLastYear','WorkLifeBalance', 'YearsAtCompany', 'YearsInCurrentRole',

       'YearsSinceLastPromotion', 'YearsWithCurrManager'])
data.boxplot(column=['NumCompaniesWorked','PercentSalaryHike', 'PerformanceRating', 'RelationshipSatisfaction',

                     'StockOptionLevel'])
#from the above boxplot we can see that Monthly income has more outliers ouliers compared to other varbales

#below is a function that filters out outliers row from our data.

def outlier(df,col_name):

    Q1 = df[col_name].quantile(0.25)

    Q3 = df[col_name].quantile(0.75)

    IQR = Q3 - Q1

    print("IQR: ",IQR)

    print("Q1: ",Q1)

    print("Q3: ",Q3)

    lw=Q1-(1.5*IQR)

    up=Q3+(1.5*IQR)

    print("the limits of outlier is: ",(lw,up))

    print("% of observation above upperlimit is: ",(df[col_name]>up).value_counts(normalize=True)*100)

    print("% of observation below lowerlimit is: ",(df[col_name]<lw).value_counts(normalize=True)*100)

    df=df.loc[(df[col_name]>lw)&(df[col_name]<up)]

    return(df)

data=outlier(data,'MonthlyIncome')

data.shape
#splitting the data into independent var and target variables

y=data['Attrition']

x=data.drop('Attrition',axis=1)

x.head()
# we can apply feature selection method to select the features to be included in our model.

from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import chi2

#apply SelectKBest class to extract top 20 best features using chi-square 

bestfeatures = SelectKBest(score_func=chi2, k=20)

fit = bestfeatures.fit(x,y)

df_scores=pd.DataFrame(fit.scores_)

df_columns=pd.DataFrame(x.columns)

feature_scores=pd.concat([df_columns,df_scores],axis=1)

feature_scores.columns=['variable','score']

final_variables=feature_scores.nlargest(20,'score')

final_variables
#filtering out the important columns alone for building our model

filtered_data=data[['MonthlyIncome','MonthlyRate','DailyRate','TotalWorkingYears','YearsAtCompany','YearsInCurrentRole',

                   'YearsWithCurrManager','Age','DistanceFromHome','StockOptionLevel','OverTime','JobLevel','MaritalStatus',

                   'EducationField','YearsSinceLastPromotion','JobSatisfaction','EnvironmentSatisfaction','NumCompaniesWorked',

                   'JobInvolvement','TrainingTimesLastYear','Attrition']]

filtered_data.shape
# splitting the data into train and test

from sklearn.model_selection import train_test_split

y=filtered_data['Attrition']

x=filtered_data.drop('Attrition',axis=1)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=50)
# buildng a decision tree model using 'entropy' criterion

from sklearn.tree import DecisionTreeClassifier

model=DecisionTreeClassifier(criterion='entropy',max_depth=5,random_state=100)

model.fit(x_train,y_train)
predictions=model.predict(x_test)

from sklearn.metrics import accuracy_score

accuracy=accuracy_score(y_test,predictions)*100

print(accuracy)
# model evaluation using cross validation score to check our accuracy.

from sklearn.model_selection import cross_val_score

dt_score=cross_val_score(model,x_train,y_train,scoring="accuracy",cv=10)

print(dt_score)

print("mean_accuracy:",dt_score.mean())

print("std_deviation of accuracy:",dt_score.std())

#the mean accuracy of our model after running it 10 times is 80.3%
from sklearn.metrics import confusion_matrix

from sklearn.metrics import precision_score,recall_score

print(confusion_matrix(y_test, predictions))

print("precision_score: ",precision_score(y_test,predictions))

print("recall_score: ",recall_score(y_test,predictions))

from sklearn.metrics import classification_report

print(classification_report(y_test,predictions))
# building a random forest classifier for the data

from sklearn.ensemble import RandomForestClassifier

rf=RandomForestClassifier(random_state=500)

rf.fit(x_train,y_train)

predictions_rf=rf.predict(x_test)

print(accuracy_score(y_test,predictions_rf))

# model evaluation

from sklearn.model_selection import cross_val_score

rf_score=cross_val_score(rf,x_train,y_train,scoring="accuracy",cv=10)

print(rf_score)

print("mean_accuracy:",rf_score.mean())

print("std_deviation of accuracy:",rf_score.std())
#confusion matrix

print(confusion_matrix(y_test, predictions))

from sklearn.metrics import classification_report

print(classification_report(y_test,predictions_rf))