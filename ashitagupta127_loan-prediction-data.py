# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# Passing libraries and the functions : 
from matplotlib import pyplot as plt
from matplotlib import style 
style.use("ggplot")
from sklearn.preprocessing import Imputer
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import scale
from collections import Counter
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
#Reading the csv file:
train=pd.read_csv("../input/train_u6lujuX_CVtuZ9i.csv")
test=pd.read_csv("../input/test_Y3wMUE5_7gLdaTN.csv")
train.head(5)


train.shape
test.head(5)
# Data Exploration 
train['source']='train'
test['source']='test'
data = pd.concat([train,test] ,ignore_index=True, sort=False)
print(train.shape, test.shape, data.shape )
data.tail(10)
# printing the database
data.head(10)
# Data structure
data.describe(include='all')
# numeric datastructure
# Inference drwan is: 
# Column CoapplicantIncome, LoanAmount, Loan_amount_Term, Credit_History has missing values in numeric datasets
## Applicants Income and CoApplicantIncome minimum value is zero which is absurd
data.describe()
test['ApplicantIncome'].min()
train['ApplicantIncome'].min()
# Checking Missing Values 
# We observe that we have missing values in Gender, Married, dependents, Self_Employed, LoanAmount,Loan_Amount_Term, Credit_History  
data.apply(lambda x: sum(x.isnull()), axis=0)

## We will input all the missing values afterwards while cleaning the data
# Checking for Unique values 
data.apply(lambda x: len(x.unique()))

## We observe that teher are 4 types of dependents which is being stored as an object than a numeric dtype
## ANd due to presence of missing values the number of nique values being shown is one more than the actual value
# Checking dataframes
data.dtypes
data.dtypes

# Filtering Categorical Columns
categorical_columns = [x for x in data.dtypes.index if data.dtypes[x]=='object']

# excluding Loan_ID and sorce columns
categorical_columns = [x for x in categorical_columns if x not in ['Loan_ID','source']]
data.head(5)



# Chceking frequency of non numeric columns
for col in categorical_columns:
    print('\nFrequency of Categories for varible %s'%col)
    print (data[col].value_counts())
data.dtypes

## Categorical variable Analysis For the Credits History
temp1 = data['Credit_History'].value_counts(ascending=True)
temp2 = data.pivot_table(values='Loan_Status',index=['Credit_History'],aggfunc=lambda x: x.map({'Y':1,'N':0}).mean())
print('Frequency Table for Credit History:') 
print(temp1)

print('\nProbility of getting loan for each Credit History class:')
print(temp2)
## Plotting As a Bar Chart
fig = plt.figure(figsize=(8,4))
ax1 = fig.add_subplot(121)
ax1.set_xlabel('Credit_History')
ax1.set_ylabel('Count of Applicants')
ax1.set_title("Applicants by Credit_History")
temp1.plot(kind='bar', edgecolor='black' , color='skyblue')

## Plotting the probability based on Credits history :

ax2 = fig.add_subplot(122)
temp2.plot(kind = 'bar')
ax2.set_xlabel('Credit_History')
ax2.set_ylabel('Probability of getting loan')
ax2.set_title("Probability of getting loan by credit history")

## This shows that the chances of getting a loan are eight-fold if the applicant has a valid credit history
#plotting A stacked bar

temp3 = pd.crosstab(data['Credit_History'], data['Loan_Status'])
temp3.plot(kind='bar', stacked=True, color=['red','blue'], grid=False)
## Categorical variable Analysis based on Gender 
temp1 = data['Gender'].value_counts(ascending=True)
temp2 = data.pivot_table(values='Loan_Status',index=['Gender'],aggfunc=lambda x: x.map({'Y':1,'N':0}).mean())
print('Frequency Table for Gender:') 
print(temp1)

print('\nProbility of getting loan for each Gender class:')
print(temp2)
## Plotting As a Bar Chart
fig = plt.figure(figsize=(8,4))
ax1 = fig.add_subplot(121)
ax1.set_xlabel('Gender')
ax1.set_ylabel('Count of Applicants')
ax1.set_title("Applicants by Gender")
temp1.plot(kind='bar', edgecolor='black' , color='skyblue')

## Plotting the probability based on Gender:
ax2 = fig.add_subplot(122)
temp2.plot(kind = 'bar')
ax2.set_xlabel('Gender')
ax2.set_ylabel('Probability of getting loan')
ax2.set_title("Probability of getting loan by Gender")

#plotting A stacked bar

temp3 = pd.crosstab(data['Gender'], data['Loan_Status'])
temp3.plot(kind='bar', stacked=True, color=['red','blue'], grid=False)

## We observe that Probability of getting Loans based on gender is almost same for both males and females.
### Now
## Categorical variable Analysis based on Gender 
temp1 = data['Gender'].value_counts(ascending=True)
temp2 = data['Credit_History'].value_counts(ascending=True)

temp3 = data.pivot_table(values='Loan_Status',index=['Gender','Credit_History'],aggfunc=lambda x: x.map({'Y':1,'N':0}).mean())

print('Frequency Table for Gender:') 
print(temp1)

print('Frequency Table for Credit_History:') 
print(temp2)

print('\nProbility of getting loan for each Gender And Credit History class:')
print(temp3)
## Plotting As a Bar Chart
fig = plt.figure(figsize=(8,4))
ax1 = fig.add_subplot(121)
ax1.set_xlabel('Gender')
ax1.set_ylabel('Count of Applicants')
ax1.set_title("Applicants by Gender")
temp1.plot(kind='bar', edgecolor='black' , color='skyblue')


## Plotting As a Bar Chart
fig = plt.figure(figsize=(8,4))
ax1 = fig.add_subplot(121)
ax1.set_xlabel('Credit History')
ax1.set_ylabel('Count of Applicants')
ax1.set_title("Applicants by Credit History")
temp2.plot(kind='bar', edgecolor='black' , color='green')

## Plotting the probability based on mix of Gender And Credi_History:
ax2 = fig.add_subplot(122)
temp3.plot(kind = 'bar' , color='darkorange' , edgecolor='black')
ax2.set_xlabel('Gender')
ax2.set_ylabel('Probability of getting loan')
ax2.set_title("Probability of getting loan by Gender, Credit_History")

#plotting A stacked bar

#temp4 = pd.crosstab(values='Loan_Status',index=['Gender','Credit_History'], columns='Loan_status',aggfunc=np.median())
#emp4.plot(kind='bar', stacked=True, color=['red','blue'], grid=False)
# Categorical variable Analysis based on MaRRIED and Dependents 
temp1 = data['Married'].value_counts(ascending=True)
temp2 = data['Dependents'].value_counts(ascending=True)

temp3 = data.pivot_table(values='Loan_Status',index=['Married'],aggfunc=lambda x: x.map({'Y':1,'N':0}).mean())

temp4 = data.pivot_table(values='Loan_Status',index=['Dependents'],aggfunc=lambda x: x.map({'Y':1,'N':0}).mean())

temp5 = data.pivot_table(values='Loan_Status',index=['Married','Dependents'],aggfunc=lambda x: x.map({'Y':1,'N':0}).mean())

print('Frequency Table for Married:') 
print(temp1)

print('Frequency Table for Dependents:') 
print(temp2)

print('\nProbility of getting loan for each Married class:')
print(temp3)

print('\nProbility of getting loan for each Dependents class:')
print(temp4)

print('\nProbility of getting loan for each Married And Dependents class:')
print(temp5)

## We observed that the probability of loan is almost equal for Married And DEPENDENTS CLASS

# Categorical variable Analysis based on MaRRIED and Dependents 
temp1 = data['Credit_History'].value_counts(ascending=True)
temp2 = data['Dependents'].value_counts(ascending=True)

temp3 = data.pivot_table(values='Loan_Status',index=['Credit_History'],aggfunc=lambda x: x.map({'Y':1,'N':0}).mean())

temp4 = data.pivot_table(values='Loan_Status',index=['Dependents'],aggfunc=lambda x: x.map({'Y':1,'N':0}).mean())

temp5 = data.pivot_table(values='Loan_Status',index=['Credit_History','Dependents'],aggfunc=lambda x: x.map({'Y':1,'N':0}).mean())

print('Frequency Table for Credit_History:') 
print(temp1)

print('Frequency Table for Dependents:') 
print(temp2)

print('\nProbility of getting loan for each Credit_History class:')
print(temp3)

print('\nProbility of getting loan for each Dependents class:')
print(temp4)

print('\nProbility of getting loan for each Credit_History And Dependents class:')
print(temp5)

## Plotting the probability based on Gender:
ax2 = fig.add_subplot(122)
temp5.plot(kind = 'bar' , color='darkblue', edgecolor='black')
ax2.set_xlabel('Dependents, Credit_History')
ax2.set_ylabel('Probability of getting loan')
ax2.set_title("Probability of getting loan by Dependents, Credit_History")

# Categorical variable Analysis based on Property_Area and Credit_History 
temp1 = data['Credit_History'].value_counts(ascending=True)
temp2 = data['Property_Area'].value_counts(ascending=True)

temp3 = data.pivot_table(values='Loan_Status',index=['Credit_History'],aggfunc=lambda x: x.map({'Y':1,'N':0}).mean())

temp4 = data.pivot_table(values='Loan_Status',index=['Property_Area'],aggfunc=lambda x: x.map({'Y':1,'N':0}).mean())

temp5 = data.pivot_table(values='Loan_Status',index=['Credit_History','Property_Area'],aggfunc=lambda x: x.map({'Y':1,'N':0}).mean())

print('Frequency Table for Credit_History:') 
print(temp1)

print('Frequency Table for Property_Area:') 
print(temp2)

print('\nProbility of getting loan for each Credit_History class:')
print(temp3)

print('\nProbility of getting loan for each Property_Area class:')
print(temp4)

print('\nProbility of getting loan for each Credit_History And Property_Area class:')
print(temp5)
data.dtypes
## Categorical variable Analysis based on Loan Amount
## This doesn't give us any kind of relevant information.
temp1 = data['Loan_Amount_Term'].value_counts(ascending=True)
temp2 = data.pivot_table(values='Loan_Status',index=['Loan_Amount_Term'],aggfunc=lambda x: x.map({'Y':1,'N':0}).mean())
print('Frequency Table for Loan_Amount_Term:') 
print(temp1)

print('\nProbility of getting loan for each Loan_Amount_Term class:')
print(temp2) 
# Distribution Analysis

## Plotting histogram of the numeric column ApplicantIncome
data['ApplicantIncome'].hist(bins=50)

data.ApplicantIncome.plot(kind='hist',color='green',edgecolor='black',title='Histogram of ApplicantIncome', bins=50)

## We obsetve here some extreme values which are termed as outliers
#Plotting the boxplot :
data.boxplot(column='ApplicantIncome')
#This confirms the presence of outliers
# Plotting the boxplot w.r.t Educatuion 
data.boxplot(column='ApplicantIncome', by='Education')
## We observed that the income disparity is also due to Education disparity: there is no substantial difference between the mean income for both but the number of educated with higher pay is more than who are neducated
# Plotting the boxplot w.r.t Gender
data.boxplot(column='ApplicantIncome', by='Gender')

## We observed that the income disparity is also due to Gender disparity: there is no substantial difference between the mean income for both the genders but the number of males with higher pay is more than females

## Plotting histogram of the numeric column ApplicantIncome
data.CoapplicantIncome.plot(kind='hist',color='blue',edgecolor='black',title='Histogram of CoApplicantIncome', bins=50)
## We observe here some extreme values which are termed as outliers but are less as compared to Applicantincome

#Plotting the boxplot :
data.boxplot(column='CoapplicantIncome')
#This confirms the presence of outliers
# Plotting the boxplot w.r.t Educatuion 
data.boxplot(column='CoapplicantIncome', by='Education')
## We observed that the income disparity is also due to Education disparity: there is no substantial difference between the mean income for both but the number of educated with higher pay is more than who are uneducated
# Plotting the boxplot w.r.t Gender
data.boxplot(column='CoapplicantIncome', by='Gender')

## We observed that the income disparity is also due to Gender disparity: there is no substantial difference between the mean income for both the genders but the number of males with higher pay is more than females
## Plotting histogram of the numeric column ApplicantIncome
data.LoanAmount.plot(kind='hist',color='darkgreen',edgecolor='black',title='Histogram of LoanAmount', bins=50)

## We could see the presence of outliers here too
#Plotting the boxplot :
data.boxplot(column='LoanAmount')
#This confirms the presence of outliers
# Plotting the boxplot w.r.t Education 
data.boxplot(column='LoanAmount', by='Education')
# Plotting the boxplot w.r.t Gender 
data.boxplot(column='LoanAmount', by='Gender')
# Plotting the boxplot w.r.t 'Married
data.boxplot(column='LoanAmount', by='Married')
# Plotting the boxplot w.r.t Property_Area
data.boxplot(column='LoanAmount', by='Self_Employed')
## Plotting histogram of the numeric column ApplicantIncome
data.Loan_Amount_Term.plot(kind='hist',color='darkred',edgecolor='black',title='Histogram of Loan_Amount_Term', bins=50)
#Plotting the boxplot :
data.boxplot(column='Loan_Amount_Term')

## Filling the missing values 

# Filling the missing values in Gender :
## Since gender here is an object type with two unique values but here since the gender values are very few as compared to the count of the sample thus we add one more category as unknown for NaN values in Gender :
data['Gender'].fillna(value='other', inplace=True)
data.apply(lambda x: sum(x.isnull()), axis=0)

# Filling the missing values in Gender :
## Since gender here is an object type with two unique values but here since the gender values are very few as compared to the count of the sample thus we add one more category as unknown for NaN values in Gender :
data['Married'].fillna(value='Yes', inplace=True)
data.apply(lambda x: sum(x.isnull()), axis=0)
# Filling the missing values in Self_employed :
## Since the value counts for category 'No' is quiet high therefore filling the mode value will serve as the best method.
data['Self_Employed'].fillna(value='No', inplace=True)
data.apply(lambda x: sum(x.isnull()), axis=0)
# Filling the missing values in LoanAmount.
## We would first of all look for a trend in LoanAount As per the applicant is educated Or Self employed by plotting a boxplot of LoanAMount by Educated, Self-Employed

temp = data.pivot_table(values='LoanAmount',index=['Education','Self_Employed'],aggfunc=lambda x: x.map({'Y':1,'N':0}).mean())
temp.boxplot()
data.dtypes
tab=data.pivot_table(values='LoanAmount' , index=['Self_Employed' , 'Education'] , aggfunc=np.median)
print(tab)
print(tab.unstack())
#tab_1=np.log10(tab)
#print(tab_1)


## Checking all vales with NaN as values in LoanAmount column
print(data.loc[data['LoanAmount'].isnull(),['Self_Employed' ,'Education','LoanAmount']])
data.reindex()
#For check get all indexes where there are NaN values 
idx=data.loc[data['LoanAmount'].isnull(), ['Self_Employed','Education','LoanAmount']].index
print(idx)
data.apply(lambda x: sum(x.isnull()), axis=0)
data['LoanAmount'] = data.groupby(['Education','Self_Employed'])['LoanAmount'].apply(lambda x: x.fillna(x.median()))


print (data.loc[data.index.isin(idx), ['Self_Employed','Education', 'LoanAmount']])
data.head()
data.dtypes

data.apply(lambda x: sum(x.isnull()), axis=0)
#tab=data.pivot_table(values='LoanAmount' , index=['Self_Employed' , 'Education'] , aggfunc=np.mean)
#print(tab)

## making dependents column a numeric datatype
data['Dependents']=data['Dependents'].replace({'3+':3})
data.head(10)
data['Dependents']=data['Dependents'].astype(float)
data.dtypes
## Filling the missing value is self employed using the same above approach but with married and gender as key features :
tab=data.pivot_table(values='Dependents', index=['Gender', 'Married'] , aggfunc=np.mean)
print(tab)
##Define a function to return the value of this pivot table

def fill(x):
    if pd.isnull(x['Dependents']):
        return tab.loc[x['Gender'],x['Married']]
    else:
        return x['Dependents']
         

#Final Allocation
data['Dependents'] = data.apply(lambda x : format(fill(x)),axis=1)
#checcking the missing values now ":
data.apply(lambda x: sum(x.isnull()), axis=0)
## filling the missing value in Loan_Amount_Term with the mode value
data['Loan_Amount_Term'].fillna(value=360.0, inplace=True)
data.apply(lambda x: sum(x.isnull()), axis=0)
## Filling Missing Values in the Credit_History with another numeric variable category "10" as it might have so happened that rows with NaN values might not have any credit history and are taking loan for the first time :
data['Credit_History'].fillna(value = 10, inplace=True)
data.apply(lambda x: sum(x.isnull()), axis=0)
data.dtypes

### Dealing with the outliers present in LoanAmount by initially taking the log transformation and then observing again :
data['LoanAmount_log']=np.log10(data['LoanAmount'])
data.head()

data['LoanAmount_log'].hist(bins=50, edgecolor='black')
## This shows that taking the log reduced the number of outliers greatly by bringing the values more neraer
#Dealing with outliers in Applicant's Income by combining both Applicant and Co-Applicant's income and storing it as total Income :
data['TotalIncome']=data['ApplicantIncome'] + data['CoapplicantIncome']
data['TotalIncome_log']=np.log10(data['TotalIncome'])
data.head()

data['TotalIncome_log'].hist(bins=50, edgecolor='black', color='green')
## This shows that taking the log reduced the number of outliers greatly by bringing the values more neraer
data.head()
data.apply(lambda x: sum(x.isnull()), axis=0)
## extracting the data as train and test data sets again

train_new=data[0:614]
print(train_new.tail())

test_new=data[614:]
print(test_new.tail())
## Building A predictive Model in python :
from sklearn.preprocessing import LabelEncoder
var_mod = ['Gender','Married','Dependents','Education','Self_Employed','Property_Area','Loan_Status']
le = LabelEncoder()
for i in var_mod:
    train_new[i] = le.fit_transform(train_new[i].astype(str))
  
train_new.head()
## Building up a training model

### Importing the futher required modules

#Import models from scikit learn module:
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import KFold   #For K-fold cross validation
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import metrics
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
#Generic function for making a classification model and accessing performance:
def classification_model(model, data, predictors, outcome):
  #Fit the model:
  model.fit(data[predictors],data[outcome])
  
  #Make predictions on training set:
  predictions = model.predict(data[predictors])
  
  #Print accuracy
  accuracy = metrics.accuracy_score(predictions,data[outcome])
  print("Accuracy : %s" % "{0:.3%}".format(accuracy))

  #Perform k-fold cross-validation with 5 folds
  kf = KFold(data.shape[0], n_folds=5)
  error = []
  for train, test in kf:
    # Filter training data
    train_predictors = (data[predictors].iloc[train,:])
    
    # The target we're using to train the algorithm.
    train_target = data[outcome].iloc[train]
    
    # Training the algorithm using the predictors and target.
    model.fit(train_predictors, train_target)
    
    #Record error from each cross-validation run
    error.append(model.score(data[predictors].iloc[test,:], data[outcome].iloc[test]))
 
  print("Cross-Validation Score : %s" % "{0:.3%}".format(np.mean(error)))

  #Fit the model again so that it can be refered outside the function:
  model.fit(data[predictors],data[outcome]) 
## Logistic regression

## we caanot use all the variables since this will result in over fitting so let's build up an itutive hypothesis from the data exploration above:
### The chances of getting a loan is high if 
###1.APPLICANT HAS A POSITIVE CREDIT HISTORY
###2.iF THE TOTALINCOME IS HIGH
###3.lOAN AMOUNT
###4.DEPENDENTS AND PROPERTY AREA
###5.EDUCATION
###6.lOAN AMOUNT TERM
## let's beginour model with Credit_History
outcome_var = 'Loan_Status'
model = LogisticRegression()
predictor_var = ['Credit_History']
classification_model(model, train_new,predictor_var,outcome_var)

## We observe that adding onto the less important variables doen't change the accuracy of the model. Now let's look on ffor other types of model
## Decision Tree Model
model = DecisionTreeClassifier()
predictor_var = ['Credit_History','Gender','Married','Education']
classification_model(model, train_new,predictor_var,outcome_var)
## Here we see that just having the categorical variables and including more of them doen't change anything
## including the numeric columns
model = DecisionTreeClassifier()
predictor_var = ['Credit_History','TotalIncome_log','LoanAmount_log','Loan_Amount_Term']
classification_model(model, train_new,predictor_var,outcome_var)

## Here, we observed that though the accuracy reached 100%: the case of over fitting but the cross validation value decreased so let's try on a different more sophisticated algo
### Using random Forest Algo
model = RandomForestClassifier(n_estimators=100)
predictor_var = ['Gender', 'Married', 'Dependents', 'Education',
       'Self_Employed', 'Loan_Amount_Term', 'Credit_History', 'Property_Area',
        'LoanAmount_log','TotalIncome_log']
classification_model(model, train_new,predictor_var,outcome_var)
## Here, we observe that the accuracy reached 100% : an ultimate case of overfitting thus we can now try two methods 1. Reducing the number of predictors or 2.Tuning the model parameters.
#First we see the feature importance matrix from which we’ll take the most important features.
#Create a series with feature importances:
featimp = pd.Series(model.feature_importances_, index=predictor_var).sort_values(ascending=False)
print(featimp)
## Using only the top 5 variables as predictors with further changes in the parameters as well:
model2 = RandomForestClassifier(n_estimators=25, min_samples_split=25, max_depth=7, max_features=1)
predictor_var = ['TotalIncome_log','LoanAmount_log','Credit_History','Dependents','Property_Area']
classification_model(model2, train_new,predictor_var,outcome_var)
model3 = ExtraTreesClassifier(n_estimators=50,max_depth=4)
outcome_var = 'Loan_Status'

predictor_var = ['Education',
       'ApplicantIncome', 'CoapplicantIncome',
       'LoanAmount', 'Loan_Amount_Term', 'Credit_History',
       'TotalIncome']
classification_model(model3, train_new,predictor_var,outcome_var)
test_new.head()
from sklearn.preprocessing import LabelEncoder
var_mod = ['Gender','Married','Dependents','Education','Self_Employed','Property_Area','Loan_Status']
le = LabelEncoder()
for i in var_mod:
    test_new[i] = le.fit_transform(test_new[i].astype(str))
model = ExtraTreesClassifier(n_estimators=50,max_depth=4)
outcome_var = 'Loan_Status'
predictor_var = ['Education',
       'ApplicantIncome', 'CoapplicantIncome',
       'LoanAmount', 'Loan_Amount_Term', 'Credit_History',
       'TotalIncome']
classification_model(model, train_new,predictor_var,outcome_var)
test_new['Loan_Status'] = model.predict(test_new[predictor_var])
test_new.loc[test_new.Loan_Status == 1 , 'Loan_Status' ] = 'Y'
test_new.loc[test_new.Loan_Status == 0 , 'Loan_Status' ] = 'N'
test_new.head(300)
pd.DataFrame(test_new, columns=['Loan_ID' , 'Loan_Status']).to_csv('prediction.csv')
## Testing on the test data:
model3 = RandomForestClassifier(n_estimators=25, min_samples_split=25, max_depth=7, max_features=1)
predictor_var = ['TotalIncome_log','LoanAmount_log','Credit_History','Dependents','Property_Area']
classification_model(model3, test_new,predictor_var,outcome_var)
test_new['Loan_Status'] = model2.predict(test_new[predictor_var])
test_new.loc[test_new.Loan_Status == 1 , 'Loan_Status' ] = 'Y'
test_new.loc[test_new.Loan_Status == 0 , 'Loan_Status' ] = 'N'
test_new.head(300)
pd.DataFrame(test_new, columns=['Loan_ID' , 'Loan_Status']).to_csv('prediction.csv')
