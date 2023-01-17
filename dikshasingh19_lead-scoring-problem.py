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
#import warnings

import warnings

warnings.filterwarnings("ignore")

warnings.filterwarnings("always")
#import all required libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.model_selection import train_test_split

import statsmodels.api as sm



from sklearn.preprocessing import StandardScaler





from sklearn.linear_model import LogisticRegression

from sklearn.feature_selection import RFE

from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn import metrics



from sklearn.metrics import confusion_matrix



from sklearn.metrics import precision_score, recall_score

from sklearn.metrics import precision_recall_curve
# to display all the columns 

pd.set_option('display.max_columns',None)

leads=pd.read_csv("/kaggle/input/leadscore/Leads.csv")

leads.head()
#check shape

leads.shape
#check types

leads.info()
#check all the columns

leads.columns
#checking Lead Number is unique

leads["Lead Number"].is_unique
#check Prospect_ID is unique

leads["Prospect ID"].is_unique
#check for null values

leads.isnull().sum()
#percentage of null values

round(100*(leads.isnull().sum()/len(leads.index)),2)
#checking values for 

leads.Specialization.value_counts()
#converting 'Select' in np.nan because it is equivalent to null values

leads['Specialization']=leads['Specialization'].replace({'Select': np.nan})
#counting values of this categorical column

leads['How did you hear about X Education'].value_counts()
#convert select into np.nan

leads['How did you hear about X Education']=leads['How did you hear about X Education'].replace({'Select': np.nan})
# counting values

leads['What is your current occupation'].value_counts()
# counting values of this categorical column

leads['What matters most to you in choosing a course'].value_counts()
# counting value of this categorical column

leads['Lead Profile'].value_counts()
#convert Lead profile "select" field into np.nan

leads['Lead Profile']=leads['Lead Profile'].replace({'Select': np.nan})
#converting "select" into np.nan.

leads['City']=leads['City'].replace({'Select': np.nan})
# counting values of "Country" column

leads['Country'].value_counts()
# count values of "Last Notable Activity" in percentage

leads['Last Notable Activity'].value_counts(normalize=True)
#Finding null percentage including "select" fields

round(100*(leads.isnull().sum()/len(leads.index)),2)
#Dropping columns

leads.drop(['Asymmetrique Activity Index','Asymmetrique Profile Index','Asymmetrique Activity Score',

                  'Asymmetrique Profile Score','Lead Profile','Lead Quality','Tags',

                  'How did you hear about X Education','Last Notable Activity'],axis=1,inplace=True)

print("Sucessfully Dropped")
#checking the shape again

leads.shape
#dropping rows having all null values

leads=leads.dropna(axis=0,how='all')
#again checking shape

leads.shape
#checking percentage of missing values after dropping many columns

round(100*(leads.isnull().sum()/len(leads.index)),2)
leads['Do Not Email'].value_counts(normalize=True)
leads['Do Not Call'].value_counts(normalize=True)
leads['Search'].value_counts(normalize=True)
leads['Magazine'].value_counts(normalize=True)
leads['Newspaper Article'].value_counts(normalize=True)
leads['X Education Forums'].value_counts(normalize=True)
leads['Newspaper'].value_counts(normalize=True)
leads['Digital Advertisement'].value_counts(normalize=True)
leads['Through Recommendations'].value_counts(normalize=True)
leads['Receive More Updates About Our Courses'].value_counts(normalize=True)
leads['Update me on Supply Chain Content'].value_counts(normalize=True)
leads['Get updates on DM Content'].value_counts(normalize=True)
leads['I agree to pay the amount through cheque'].value_counts(normalize=True)
leads['What matters most to you in choosing a course'].value_counts(normalize=True)
# Dropping all skewed columns

leads.drop(['Prospect ID','Do Not Email','Do Not Call','Search','Magazine','Newspaper Article','X Education Forums','Newspaper',

                  'Digital Advertisement','Through Recommendations','Receive More Updates About Our Courses',

                  'Update me on Supply Chain Content','Get updates on DM Content',

                  'I agree to pay the amount through cheque','Country','What matters most to you in choosing a course'],axis=1,inplace=True)

print("successfully dropped")
#checking shape

leads.shape
#checking counts of "specialization"

leads['Specialization'].value_counts(normalize=True)
# assigning "other" to all the null values

leads.loc[pd.isnull(leads['Specialization']),['Specialization']]='Other'
# checking counts of specialization after adding "other"

leads['Specialization'].value_counts(normalize=True)
#null count check for "specialization"

leads['Specialization'].isnull().sum()
#null percentage of variables

round(100*(leads.isnull().sum()/len(leads.index)),2)
# checking value counts of "What is your current occupation"

leads['What is your current occupation'].value_counts(normalize=True)
# assigning "other" to null values

leads.loc[pd.isnull(leads['What is your current occupation']),['What is your current occupation']]='Other'
#checking null after imputing

leads['What is your current occupation'].isnull().sum()
#checking value counts of "What is your current occupation".

leads['What is your current occupation'].value_counts()
#checking null percentage after imputing

round(100*(leads.isnull().sum()/len(leads.index)),2)
# checking value counts of "Lead Source"

leads['Lead Source'].value_counts(normalize=True)
# here we decide to impute the missing values with mode because it is very less in percentage

leads['Lead Source'].fillna(leads['Lead Source'].mode()[0],inplace=True)
# checking null counts after imputing

leads['Lead Source'].isnull().sum()
#checking null percentge after imputing

round(100*(leads.isnull().sum()/len(leads.index)),2)
# Checking  value counts of "TotalVisits"

leads['TotalVisits'].value_counts(normalize=True)
# imputing missing values with mode

leads['TotalVisits'].fillna(leads['TotalVisits'].mode()[0],inplace=True)
#checking null values after imputing

leads['TotalVisits'].isnull().sum()
# null percentage

round(100*(leads.isnull().sum()/len(leads.index)),2)
# checking value counts "Page Views Per Visit"

leads['Page Views Per Visit'].value_counts(normalize=True)
# number of null values

leads['Page Views Per Visit'].isnull().sum()
# imputing null values with median because we see that it is continous column so it is safe to impute with median

leads['Page Views Per Visit'].fillna(leads['Page Views Per Visit'].median(),inplace=True)
# checking null values after imputing

leads['Page Views Per Visit'].isnull().sum()
# checking null percentage

round(100*(leads.isnull().sum()/len(leads.index)),2)
# checking value counts of "Last Activity"

100*leads['Last Activity'].value_counts(normalize=True)
# imputing missing values with mode

leads['Last Activity'].fillna(leads['Last Activity'].mode()[0],inplace=True)
#checking null values after imputing

leads['Last Activity'].isnull().sum()
# checking null percentage

round(100*(leads.isnull().sum()/len(leads.index)),2)
# checking value counts of "City".

100*leads['City'].value_counts(normalize=True)
# we are putting null values as "Unknown"

leads.loc[pd.isnull(leads['City']),['City']]='Unknown'
#after imputing null values

100*leads['City'].value_counts(normalize=True)
#chcking null percentage again

round(100*(leads.isnull().sum()/len(leads.index)),2)
# After data cleaning 

leads.head()
# making pairplots

sns.pairplot(leads,hue="Converted")

plt.show
ax=sns.heatmap(leads.corr(),annot=True)

bottom,top=ax.get_ylim()

ax.set_ylim(bottom+0.5,top-0.5)

plt.show()
# plotting barplot 

sns.barplot(x="Last Activity",y="Converted",data=leads)

plt.xticks(rotation=90)
# plotting barplot

sns.barplot(x="A free copy of Mastering The Interview",y="Converted",data=leads)

plt.xticks(rotation=90)
# plotting barplot

sns.barplot(x="City",y="Converted",data=leads)

plt.xticks(rotation=90)
sns.barplot(x="What is your current occupation",y="Converted",data=leads)

plt.xticks(rotation=90)
# plotting bar plot

sns.barplot(x="Specialization",y="Converted",data=leads)

plt.xticks(rotation=90)
sns.barplot(x="Lead Source",y="Converted",data=leads)

plt.xticks(rotation=90)
# plotting barplot

sns.barplot(x="Lead Origin",y="Converted",data=leads)

plt.xticks(rotation=90)
# Plotting distplot "Time spent on Website by customer"

sns.distplot(leads['Total Time Spent on Website'],kde=False)

plt.title("Time spent on Website by customer")

plt.show()
# plotting distplot "Page Views Per Visit"

sns.distplot(leads['Page Views Per Visit'],kde=False)

plt.title("Page Views Per Visit")

plt.xlim(0,20)

plt.show()
# plotting distplot "TotalVisits"

sns.distplot(leads['TotalVisits'],kde=False)

plt.title("TotalVisits")

plt.xlim(0,30)

plt.show()
# PLotting "boxplot" for "TotalVisits"

plt.boxplot(leads['TotalVisits'])

plt.show()
# handling the outliers , to handle the upper end with 0.99 quantile.

q1=leads['TotalVisits'].quantile(0.99)

leads['TotalVisits'][leads['TotalVisits']>=q1] = q1
# plotting boxplot for "Page Views Per Visit"

plt.boxplot(leads['Page Views Per Visit'])

plt.show()
q2=leads['Page Views Per Visit'].quantile(0.99)

leads['Page Views Per Visit'][leads['Page Views Per Visit']>=q2] = q2
# plotting boxplot for "Total Time Spent on Website"

plt.boxplot(leads['Total Time Spent on Website'])

plt.show()
# plotting barplot

sns.barplot(x="Lead Origin",y="Page Views Per Visit",hue="Converted",data=leads)

plt.xticks(rotation=90)
#plotting barplot

sns.barplot(x="What is your current occupation",y="TotalVisits",hue="Converted",data=leads)

plt.xticks(rotation=90)
# plotting barpot

sns.barplot(x="What is your current occupation",y="Page Views Per Visit",hue="Converted",data=leads)

plt.xticks(rotation=90)
# putting yes=1,No=0

leads['A free copy of Mastering The Interview']=leads['A free copy of Mastering The Interview'].map({'Yes': 1, "No": 0})
# checking head

leads.head()
# creating dummy variable and concatenating it with out original dataset

dummy1 = pd.get_dummies(leads[['Lead Origin', 'Lead Source', 'Last Activity', 'Specialization',

                               'What is your current occupation','City']], drop_first=True)

leads = pd.concat([leads, dummy1], axis=1)
# checkinge head

leads.head()
#checking shape

leads.shape
# Dropping the original ones after creating dummy variables

leads=leads.drop(['Lead Origin','Lead Source','Last Activity','Specialization','What is your current occupation','City'],1)
# chcking shape again

leads.shape
# checking head

leads.head()
# checking datatypes

leads.info()
# checking statistics

leads.describe
# splitting data into X and Y.

X = leads.drop(['Converted','Lead Number'], axis=1)



X.head()
# getting Y

y = leads['Converted']



y.head()
# Splitting the data into train and test

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=100)
#checking train head

X_train.head()
# Instantiate scaler and perform scaler on Train data

scaler = StandardScaler()

X_train[['TotalVisits','Total Time Spent on Website','Page Views Per Visit']] = scaler.fit_transform(X_train[['TotalVisits','Total Time Spent on Website','Page Views Per Visit']])



X_train.head()
# checking how converted is distributed.

Converted = (sum(leads['Converted'])/len(leads['Converted'].index))*100

Converted
# checking correlation

leads.corr()
# Logistic regression model

logm1 = sm.GLM(y_train,(sm.add_constant(X_train)), family = sm.families.Binomial())

logm1.fit().summary()
# Instantiate object

logreg = LogisticRegression()
rfe = RFE(logreg, 25)             # running RFE with 25 variables as output

rfe = rfe.fit(X_train, y_train)
# RFE chosen variables

rfe.support_
# RFE supported and non supported varibles with rankings

list(zip(X_train.columns, rfe.support_, rfe.ranking_))
# assinging the variables that RFE supports to "col"

col = X_train.columns[rfe.support_]
# variables that RFE doesn't support

X_train.columns[~rfe.support_]
# buliding model

X_train_sm = sm.add_constant(X_train[col])

logm2 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())

res = logm2.fit()

res.summary()
# Getting the predicted values on the train set

y_train_pred = res.predict(X_train_sm)

y_train_pred[:10]
# Reshping 

y_train_pred = y_train_pred.values.reshape(-1)

y_train_pred[:10]
# making dataframe having Convert and "Convert_prob"

y_train_pred_final = pd.DataFrame({'Convert':y_train.values, 'Convert_Prob':y_train_pred})

y_train_pred_final['Lead Number'] = y_train.index

y_train_pred_final.head()
# going with cutoff 0.5

y_train_pred_final['predicted'] = y_train_pred_final.Convert_Prob.map(lambda x: 1 if x > 0.5 else 0)



# Let's see the head

y_train_pred_final.head()
# Confusion matrix 

confusion = metrics.confusion_matrix(y_train_pred_final.Convert, y_train_pred_final.predicted )

print(confusion)
# Accuracy

print(metrics.accuracy_score(y_train_pred_final.Convert, y_train_pred_final.predicted))
vif = pd.DataFrame()

vif['Features'] = X_train[col].columns

vif['VIF'] = [variance_inflation_factor(X_train[col].values, i) for i in range(X_train[col].shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
# Drop

col = col.drop('Lead Source_NC_EDM', 1)

col
# Let's re-run the model using the selected variables

X_train_sm = sm.add_constant(X_train[col])

logm3 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())

res = logm3.fit()

res.summary()
# Getting the predicted values on the train set

y_train_pred = res.predict(X_train_sm)

y_train_pred[:10]
# reshaping

y_train_pred = y_train_pred.values.reshape(-1)

y_train_pred[:10]
# making dataframe having Convert and "Convert_prob"

y_train_pred_final = pd.DataFrame({'Convert':y_train.values, 'Convert_Prob':y_train_pred})

y_train_pred_final['Lead Number'] = y_train.index

y_train_pred_final.head()
# Taking cut off as 0.5

y_train_pred_final['predicted'] = y_train_pred_final.Convert_Prob.map(lambda x: 1 if x > 0.5 else 0)



# Let's see the head

y_train_pred_final.head()
# Confusion matrix 

confusion = metrics.confusion_matrix(y_train_pred_final.Convert, y_train_pred_final.predicted )

print(confusion)
print(metrics.accuracy_score(y_train_pred_final.Convert, y_train_pred_final.predicted))
vif = pd.DataFrame()

vif['Features'] = X_train[col].columns

vif['VIF'] = [variance_inflation_factor(X_train[col].values, i) for i in range(X_train[col].shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
# dropping variable

col = col.drop('Last Activity_Resubscribed to emails', 1)

col
# Building model

X_train_sm = sm.add_constant(X_train[col])

logm4 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())

res = logm4.fit()

res.summary()
# Getting the predicted values on the train set

y_train_pred = res.predict(X_train_sm)

y_train_pred[:10]



y_train_pred = y_train_pred.values.reshape(-1)

y_train_pred[:10]



y_train_pred_final = pd.DataFrame({'Convert':y_train.values, 'Convert_Prob':y_train_pred})

y_train_pred_final['Lead Number'] = y_train.index

y_train_pred_final.head()





y_train_pred_final['predicted'] = y_train_pred_final.Convert_Prob.map(lambda x: 1 if x > 0.5 else 0)



# Let's see the head

y_train_pred_final.head()





# Confusion matrix 

confusion = metrics.confusion_matrix(y_train_pred_final.Convert, y_train_pred_final.predicted )

print(confusion)

# Accuracy

print(metrics.accuracy_score(y_train_pred_final.Convert, y_train_pred_final.predicted))
vif = pd.DataFrame()

vif['Features'] = X_train[col].columns

vif['VIF'] = [variance_inflation_factor(X_train[col].values, i) for i in range(X_train[col].shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
col = col.drop('What is your current occupation_Housewife', 1)

col
X_train_sm = sm.add_constant(X_train[col])

logm5 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())

res = logm5.fit()

res.summary()
# Getting the predicted values on the train set

y_train_pred = res.predict(X_train_sm)

y_train_pred[:10]



y_train_pred = y_train_pred.values.reshape(-1)

y_train_pred[:10]



y_train_pred_final = pd.DataFrame({'Convert':y_train.values, 'Convert_Prob':y_train_pred})

y_train_pred_final['Lead Number'] = y_train.index

y_train_pred_final.head()





y_train_pred_final['predicted'] = y_train_pred_final.Convert_Prob.map(lambda x: 1 if x > 0.5 else 0)



# Let's see the head

y_train_pred_final.head()





# Confusion matrix 

confusion = metrics.confusion_matrix(y_train_pred_final.Convert, y_train_pred_final.predicted )

print(confusion)



print(metrics.accuracy_score(y_train_pred_final.Convert, y_train_pred_final.predicted))
vif = pd.DataFrame()

vif['Features'] = X_train[col].columns

vif['VIF'] = [variance_inflation_factor(X_train[col].values, i) for i in range(X_train[col].shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
col = col.drop('Lead Source_Social Media', 1)

col
X_train_sm = sm.add_constant(X_train[col])

logm6 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())

res = logm6.fit()

res.summary()
# Getting the predicted values on the train set

y_train_pred = res.predict(X_train_sm)

y_train_pred[:10]



y_train_pred = y_train_pred.values.reshape(-1)

y_train_pred[:10]



y_train_pred_final = pd.DataFrame({'Convert':y_train.values, 'Convert_Prob':y_train_pred})

y_train_pred_final['Lead Number'] = y_train.index

y_train_pred_final.head()





y_train_pred_final['predicted'] = y_train_pred_final.Convert_Prob.map(lambda x: 1 if x > 0.5 else 0)



# Let's see the head

y_train_pred_final.head()





# Confusion matrix 

confusion = metrics.confusion_matrix(y_train_pred_final.Convert, y_train_pred_final.predicted )

print(confusion)



print(metrics.accuracy_score(y_train_pred_final.Convert, y_train_pred_final.predicted))
vif = pd.DataFrame()

vif['Features'] = X_train[col].columns

vif['VIF'] = [variance_inflation_factor(X_train[col].values, i) for i in range(X_train[col].shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
col = col.drop('Specialization_Rural and Agribusiness', 1)

col
X_train_sm = sm.add_constant(X_train[col])

logm7 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())

res = logm7.fit()

res.summary()
# Getting the predicted values on the train set

y_train_pred = res.predict(X_train_sm)

y_train_pred[:10]



y_train_pred = y_train_pred.values.reshape(-1)

y_train_pred[:10]



y_train_pred_final = pd.DataFrame({'Convert':y_train.values, 'Convert_Prob':y_train_pred})

y_train_pred_final['Lead Number'] = y_train.index

y_train_pred_final.head()





y_train_pred_final['predicted'] = y_train_pred_final.Convert_Prob.map(lambda x: 1 if x > 0.5 else 0)



# Let's see the head

y_train_pred_final.head()





# Confusion matrix 

confusion = metrics.confusion_matrix(y_train_pred_final.Convert, y_train_pred_final.predicted )

print(confusion)



print(metrics.accuracy_score(y_train_pred_final.Convert, y_train_pred_final.predicted))
vif = pd.DataFrame()

vif['Features'] = X_train[col].columns

vif['VIF'] = [variance_inflation_factor(X_train[col].values, i) for i in range(X_train[col].shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
col = col.drop('Specialization_Retail Management', 1)

col
X_train_sm = sm.add_constant(X_train[col])

logm8 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())

res = logm8.fit()

res.summary()
# Getting the predicted values on the train set

y_train_pred = res.predict(X_train_sm)

y_train_pred[:10]



y_train_pred = y_train_pred.values.reshape(-1)

y_train_pred[:10]



y_train_pred_final = pd.DataFrame({'Convert':y_train.values, 'Convert_Prob':y_train_pred})

y_train_pred_final['Lead Number'] = y_train.index

y_train_pred_final.head()





y_train_pred_final['predicted'] = y_train_pred_final.Convert_Prob.map(lambda x: 1 if x > 0.5 else 0)



# Let's see the head

y_train_pred_final.head()





# Confusion matrix 

confusion = metrics.confusion_matrix(y_train_pred_final.Convert, y_train_pred_final.predicted )

print(confusion)



print(metrics.accuracy_score(y_train_pred_final.Convert, y_train_pred_final.predicted))
vif = pd.DataFrame()

vif['Features'] = X_train[col].columns

vif['VIF'] = [variance_inflation_factor(X_train[col].values, i) for i in range(X_train[col].shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
col = col.drop('Lead Source_Facebook', 1)

col
X_train_sm = sm.add_constant(X_train[col])

logm9 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())

res = logm9.fit()

res.summary()
# Getting the predicted values on the train set

y_train_pred = res.predict(X_train_sm)

y_train_pred[:10]



y_train_pred = y_train_pred.values.reshape(-1)

y_train_pred[:10]



y_train_pred_final = pd.DataFrame({'Convert':y_train.values, 'Convert_Prob':y_train_pred})

y_train_pred_final['Lead Number'] = y_train.index

y_train_pred_final.head()





y_train_pred_final['predicted'] = y_train_pred_final.Convert_Prob.map(lambda x: 1 if x > 0.5 else 0)



# Let's see the head

y_train_pred_final.head()





# Confusion matrix 

confusion = metrics.confusion_matrix(y_train_pred_final.Convert, y_train_pred_final.predicted )

print(confusion)



print(metrics.accuracy_score(y_train_pred_final.Convert, y_train_pred_final.predicted))
vif = pd.DataFrame()

vif['Features'] = X_train[col].columns

vif['VIF'] = [variance_inflation_factor(X_train[col].values, i) for i in range(X_train[col].shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
col = col.drop('Last Activity_Email Link Clicked', 1)

col
X_train_sm = sm.add_constant(X_train[col])

logm10 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())

res = logm10.fit()

res.summary()
# Getting the predicted values on the train set

y_train_pred = res.predict(X_train_sm)

y_train_pred[:10]



y_train_pred = y_train_pred.values.reshape(-1)

y_train_pred[:10]



y_train_pred_final = pd.DataFrame({'Convert':y_train.values, 'Convert_Prob':y_train_pred})

y_train_pred_final['Lead Number'] = y_train.index

y_train_pred_final.head()





y_train_pred_final['predicted'] = y_train_pred_final.Convert_Prob.map(lambda x: 1 if x > 0.5 else 0)



# Let's see the head

y_train_pred_final.head()





# Confusion matrix 

confusion = metrics.confusion_matrix(y_train_pred_final.Convert, y_train_pred_final.predicted )

print(confusion)



print(metrics.accuracy_score(y_train_pred_final.Convert, y_train_pred_final.predicted))
vif = pd.DataFrame()

vif['Features'] = X_train[col].columns

vif['VIF'] = [variance_inflation_factor(X_train[col].values, i) for i in range(X_train[col].shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
col = col.drop('Last Activity_Form Submitted on Website', 1)

col
X_train_sm = sm.add_constant(X_train[col])

logm11 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())

res = logm11.fit()

res.summary()
# Getting the predicted values on the train set

y_train_pred = res.predict(X_train_sm)

y_train_pred[:10]



y_train_pred = y_train_pred.values.reshape(-1)

y_train_pred[:10]



y_train_pred_final = pd.DataFrame({'Convert':y_train.values, 'Convert_Prob':y_train_pred})

y_train_pred_final['Lead Number'] = y_train.index

y_train_pred_final.head()





y_train_pred_final['predicted'] = y_train_pred_final.Convert_Prob.map(lambda x: 1 if x > 0.5 else 0)



# Let's see the head

y_train_pred_final.head()





# Confusion matrix 

confusion = metrics.confusion_matrix(y_train_pred_final.Convert, y_train_pred_final.predicted )

print(confusion)



print(metrics.accuracy_score(y_train_pred_final.Convert, y_train_pred_final.predicted))
vif = pd.DataFrame()

vif['Features'] = X_train[col].columns

vif['VIF'] = [variance_inflation_factor(X_train[col].values, i) for i in range(X_train[col].shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
col = col.drop('City_Unknown', 1)

col
X_train_sm = sm.add_constant(X_train[col])

logm12 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())

res = logm12.fit()

res.summary()
# Getting the predicted values on the train set

y_train_pred = res.predict(X_train_sm)

y_train_pred[:10]



y_train_pred = y_train_pred.values.reshape(-1)

y_train_pred[:10]



y_train_pred_final = pd.DataFrame({'Convert':y_train.values, 'Convert_Prob':y_train_pred})

y_train_pred_final['Lead Number'] = y_train.index

y_train_pred_final.head()





y_train_pred_final['predicted'] = y_train_pred_final.Convert_Prob.map(lambda x: 1 if x > 0.5 else 0)



# Let's see the head

y_train_pred_final.head()





# Confusion matrix 

confusion = metrics.confusion_matrix(y_train_pred_final.Convert, y_train_pred_final.predicted )

print(confusion)



print(metrics.accuracy_score(y_train_pred_final.Convert, y_train_pred_final.predicted))
vif = pd.DataFrame()

vif['Features'] = X_train[col].columns

vif['VIF'] = [variance_inflation_factor(X_train[col].values, i) for i in range(X_train[col].shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
TP = confusion[1,1] # true positive 

TN = confusion[0,0] # true negatives

FP = confusion[0,1] # false positives

FN = confusion[1,0] # false negatives
# Let's see the sensitivity of our logistic regression model

TP / float(TP+FN)
# Let us calculate specificity

TN / float(TN+FP)
# Calculate false postive rate - predicting churn when customer does not have churned

print(FP/ float(TN+FP))
# positive predictive value 

print (TP / float(TP+FP))
# Negative predictive value

print (TN / float(TN+ FN))
def draw_roc( actual, probs ):

    fpr, tpr, thresholds = metrics.roc_curve( actual, probs,

                                              drop_intermediate = False )

    auc_score = metrics.roc_auc_score( actual, probs )

    plt.figure(figsize=(5, 5))

    plt.plot( fpr, tpr, label='ROC curve (area = %0.2f)' % auc_score )

    plt.plot([0, 1], [0, 1], 'k--')

    plt.xlim([0.0, 1.0])

    plt.ylim([0.0, 1.05])

    plt.xlabel('False Positive Rate or [1 - True Negative Rate]')

    plt.ylabel('True Positive Rate')

    plt.title('Receiver operating characteristic example')

    plt.legend(loc="lower right")

    plt.show()



    return None
fpr, tpr, thresholds = metrics.roc_curve( y_train_pred_final.Convert, y_train_pred_final.Convert_Prob, drop_intermediate = False )
# Drawing Roc curve

draw_roc(y_train_pred_final.Convert, y_train_pred_final.Convert_Prob)
# Let's create columns with different probability cutoffs 

numbers = [float(x)/10 for x in range(10)]

for i in numbers:

    y_train_pred_final[i]= y_train_pred_final.Convert_Prob.map(lambda x: 1 if x > i else 0)

y_train_pred_final.head()
# Now let's calculate accuracy sensitivity and specificity for various probability cutoffs.

cutoff_df = pd.DataFrame( columns = ['prob','accuracy','sensi','speci'])



# TP = confusion[1,1] # true positive 

# TN = confusion[0,0] # true negatives

# FP = confusion[0,1] # false positives

# FN = confusion[1,0] # false negatives



num = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]

for i in num:

    cm1 = metrics.confusion_matrix(y_train_pred_final.Convert, y_train_pred_final[i] )

    total1=sum(sum(cm1))

    accuracy = (cm1[0,0]+cm1[1,1])/total1

    

    speci = cm1[0,0]/(cm1[0,0]+cm1[0,1])

    sensi = cm1[1,1]/(cm1[1,0]+cm1[1,1])

    cutoff_df.loc[i] =[ i ,accuracy,sensi,speci]

print(cutoff_df)
# Let's plot accuracy sensitivity and specificity for various probabilities.

cutoff_df.plot.line(x='prob', y=['accuracy','sensi','speci'])

#plt.xlim(0,0.35)

plt.show()
# getting "final_predicted" taking 0.35 cut off.

y_train_pred_final['final_predicted'] = y_train_pred_final.Convert_Prob.map( lambda x: 1 if x > 0.35 else 0)



y_train_pred_final.head()
# Let's check the overall accuracy.

metrics.accuracy_score(y_train_pred_final.Convert, y_train_pred_final.final_predicted)
# confusion matrix for final_predicted.

confusion2 = metrics.confusion_matrix(y_train_pred_final.Convert, y_train_pred_final.final_predicted )

confusion2
TP = confusion2[1,1] # true positive 

TN = confusion2[0,0] # true negatives

FP = confusion2[0,1] # false positives

FN = confusion2[1,0] # false negatives
# Let's see the sensitivity of our logistic regression model

TP / float(TP+FN)
# Let us calculate specificity

TN / float(TN+FP)
# Calculate false postive rate - predicting churn when customer does not have churned

print(FP/ float(TN+FP))
# Positive predictive value 

print (TP / float(TP+FP))
# Negative predictive value

print (TN / float(TN+ FN))
confusion[1,1]/(confusion[0,1]+confusion[1,1])
confusion[1,1]/(confusion[1,0]+confusion[1,1])
# Precision score of model

precision_score(y_train_pred_final.Convert, y_train_pred_final.predicted)
# Recall score of model

recall_score(y_train_pred_final.Convert, y_train_pred_final.predicted)
y_train_pred_final.Convert, y_train_pred_final.predicted
#Ploting the precision, recall and threshold

p, r, thresholds = precision_recall_curve(y_train_pred_final.Convert, y_train_pred_final.Convert_Prob)
#plotting the curve

plt.plot(thresholds, p[:-1], "g-")

plt.plot(thresholds, r[:-1], "r-")

plt.xlim(0,0.8)

plt.show()
#prform scaling on test set

X_test[['TotalVisits','Total Time Spent on Website','Page Views Per Visit']] = scaler.transform(X_test[['TotalVisits','Total Time Spent on Website','Page Views Per Visit']])

X_test = X_test[col]

X_test.head()
# add constant

X_test_sm = sm.add_constant(X_test)
# predict the y values

y_test_pred = res.predict(X_test_sm)
# top 10 value of y_pred on test data

y_test_pred[:10]
# Converting y_pred to a dataframe which is an array

y_pred_1 = pd.DataFrame(y_test_pred)
# Let's see the head

y_pred_1.head()
# Converting y_test to dataframe

y_test_df = pd.DataFrame(y_test)
# Putting Lead Number to index

y_test_df['Lead Number'] = y_test_df.index
# Removing index for both dataframes to append them side by side 

y_pred_1.reset_index(drop=True, inplace=True)

y_test_df.reset_index(drop=True, inplace=True)
# Appending y_test_df and y_pred_1

y_pred_final = pd.concat([y_test_df, y_pred_1],axis=1)
y_pred_final.head()
# Renaming the column 

y_pred_final= y_pred_final.rename(columns={ 0 : 'Convert_Prob'})
# Rearranging the columns

y_pred_final = y_pred_final.reindex(['Lead Number','Convert_Prob','Converted'], axis=1)
# Let's see the head of y_pred_final

y_pred_final.head()
# applying cut off

y_pred_final['final_predicted'] = y_pred_final.Convert_Prob.map(lambda x: 1 if x > 0.35 else 0)
# calculating the score 

y_pred_final['Lead Score']=y_pred_final['Convert_Prob']*100

y_pred_final.head()
# Let's check the overall accuracy.

metrics.accuracy_score(y_pred_final.Converted, y_pred_final.final_predicted)
confusion2 = metrics.confusion_matrix(y_pred_final.Converted, y_pred_final.final_predicted )

confusion2
TP = confusion2[1,1] # true positive 

TN = confusion2[0,0] # true negatives

FP = confusion2[0,1] # false positives

FN = confusion2[1,0] # false negatives
# Let's see the sensitivity of our logistic regression model

TP / float(TP+FN)
# Let us calculate specificity

TN / float(TN+FP)