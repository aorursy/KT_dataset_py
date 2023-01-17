import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
# Suppressing Warnings
import warnings
warnings.filterwarnings('ignore')
#Lets import the basic libraries

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
#Reading the dataframe as df

df = pd.read_csv("/kaggle/input/leads-dataset/Leads.csv")
df.head()
original_len = len(df)
original_len
#Lets check the data types

df.info()
#Checking the statistics of continuous variables

df.describe(percentiles=[0.25,0.5,0.75,0.9,0.95,0.99])
#Replacing all the 'select' as null values

df = df.replace({'Select':np.nan})
df.head()
#Checking the missing values percentage for each column

(df.isnull().sum()/len(df))*100
#Removing Columns with High percentage values (Like 40% and above)

df = df.drop(['How did you hear about X Education','Lead Quality','Lead Profile','City','Asymmetrique Activity Index','Asymmetrique Profile Index','Asymmetrique Activity Score','Asymmetrique Profile Score'],axis=1)
df.columns
(df.isnull().sum()/len(df))*100
#Lets check the unique values of Country column
df['Country'].value_counts(normalize=True)
#Highly Skewed data, lets remove this column!

df = df.drop('Country',axis=1)
df.head()
#We dont need the Prospect ID and Lead Number, lets drop them!

df = df.drop('Prospect ID',axis=1)
df = df.drop('Lead Number',axis=1)
df.head()
#Lets check the unique values of Specialization

df['Specialization'].value_counts(normalize=True)
#Lets club the values with less than 5% into 1 group called Others

df['Specialization'].replace({'Media and Advertising':'Others','Travel and Tourism':'Others','Services Excellence':'Others','E-Business':'Others','Healthcare Management':'Others','Hospitality Management':'Others','Rural and Agribusiness':'Others','Retail Management':'Others','E-COMMERCE':'Others','International Business':'Others'},inplace=True)
df['Specialization'].value_counts()
#This is a sensitive column as it addresses our problem. Lets make the null values as freshers

df['Specialization'] = df['Specialization'].fillna('Freshers')
df['Specialization'].value_counts(normalize=True)
#Lets check the unique values of the course selection criteria column

df['What matters most to you in choosing a course'].value_counts(normalize=True)
#Again highly skewed, lets remove it
df = df.drop('What matters most to you in choosing a course',axis=1)
df.head()
#Lets remove Tag variable since it is a score variable

df = df.drop('Tags',axis = 1)
#Checking the column of occupation

df['What is your current occupation'].value_counts(normalize=True)
#This is also a sensitive column but because of high skewness,we have to remove it. The model wont learn properly

df = df.drop('What is your current occupation',axis=1)
df.columns
#Again checking missing value percentages

(df.isnull().sum()/len(df))*100
#Lets remove the rows with missing values and check the final missing values

df = df.dropna()
(df.isnull().sum()/len(df))*100
#Checking the Newspaper column
df['Newspaper'].value_counts()
#Its a high skewed column, we should drop it
df = df.drop('Newspaper',axis=1)
df.columns
df['Lead Origin'].value_counts()
#Checking Lead Source
df['Lead Source'].value_counts(normalize=True)
#Few categories got repeated here, like google has been created again.Lets add it to the parent category Google

df['Lead Source'] = df['Lead Source'].replace({'google':'Google'})
df['Lead Source'].value_counts(normalize=True)

#Lets group the categories with data less than 10% into 1 category called Others

t = df['Lead Source'].value_counts(normalize=True).reset_index()
list1 = t['index'][t['Lead Source']<0.1]
d={}
for i in list1:
    d[i]='Others'
df['Lead Source'] = df['Lead Source'].replace(d)
df['Lead Source'].value_counts(normalize=True)
df['Do Not Email'].value_counts(normalize=True)
#This is highly skewed. Lets remove it
df = df.drop('Do Not Email',axis=1)
df.columns
#Lets check the outlier effect of total visits on converted by comparing mean and median

df['TotalVisits'].groupby(df['Converted']).median()
df['TotalVisits'].groupby(df['Converted']).mean()
df['Last Activity'].value_counts(normalize=True)
#Lets group the categories with data less than 25% into 1 category called Others

t = df['Last Activity'].value_counts(normalize=True).reset_index()
list1 = t['index'][t['Last Activity']<0.1]
d={}
for i in list1:
    d[i]='Others'
df['Last Activity'] = df['Last Activity'].replace(d)
df['Last Activity'].value_counts(normalize=True)
df.columns
df['Search'].value_counts()
#Highly skewed, lets drop it

df = df.drop('Search',axis=1)
df.columns
df['Magazine'].value_counts()
#Again its a skewed column, acting like a constant column since all the values are No

df = df.drop('Magazine',axis=1)
df.columns
df['Newspaper Article'].value_counts()
#Again its a skewed column

df = df.drop('Newspaper Article',axis=1)
df.columns
df['X Education Forums'].value_counts()
#Again its a skewed column

df = df.drop('X Education Forums',axis=1)
df.columns
df['Digital Advertisement'].value_counts()
#Again its a skewed column

df = df.drop('Digital Advertisement',axis=1)
df.columns
df['Through Recommendations'].value_counts()
#Again its a skewed column

df = df.drop('Through Recommendations',axis=1)
df.columns
df['Receive More Updates About Our Courses'].value_counts()
#Again its a skewed column

df = df.drop('Receive More Updates About Our Courses',axis=1)
df.columns
df['Update me on Supply Chain Content'].value_counts()
#Again its a skewed column

df = df.drop('Update me on Supply Chain Content',axis=1)
df.columns
df['Get updates on DM Content'].value_counts()
#Again its a skewed column

df = df.drop('Get updates on DM Content',axis=1)
df.columns
df['I agree to pay the amount through cheque'].value_counts()
#Again its a skewed column

df = df.drop('I agree to pay the amount through cheque',axis=1)
df.columns
df['A free copy of Mastering The Interview'].value_counts()
df['Last Notable Activity'].value_counts(normalize=True)
#Lets group the categories with data less than 20% into 1 category called Others

t = df['Last Notable Activity'].value_counts(normalize=True).reset_index()
list1 = t['index'][t['Last Notable Activity']<0.1]
d={}
for i in list1:
    d[i]='Others'
df['Last Notable Activity'] = df['Last Notable Activity'].replace(d)
df['Last Notable Activity'].value_counts(normalize=True)
df.describe(percentiles=[0.25,0.5,0.75,0.9,0.95,0.99])
#Removing any row with missing values

df = df.dropna()
df.isnull().sum()
#Calculating rows retained

(len(df)/original_len)*100
#Lets make two dataframes for converted 1 and not converted O
converted = df[df['Converted']==1]
converted
not_converted = df[df['Converted']==0]
not_converted
plt.figure(figsize=(10,7))
sns.countplot(converted['Converted'],hue=df['Specialization'])
plt.show()
plt.figure(figsize=(10,7))
sns.countplot(not_converted['Converted'],hue=df['Specialization'])
plt.show()
plt.figure(figsize=(10,7))
sns.countplot(converted['Converted'],hue=df['Lead Origin'])
plt.show()
plt.figure(figsize=(10,7))
sns.countplot(not_converted['Converted'],hue=df['Lead Origin'])
plt.show()
plt.figure(figsize=(10,7))
sns.countplot(converted['Converted'],hue=df['Lead Source'])
plt.show()
plt.figure(figsize=(10,7))
sns.countplot(not_converted['Converted'],hue=df['Lead Source'])
plt.show()
#We should not priotize those leads whole page views per visit is less than 2

not_converted['Page Views Per Visit'].median()
sns.countplot(x=converted['Converted'],hue=converted['Last Notable Activity'])
sns.countplot(x=not_converted['Converted'],hue=not_converted['Last Notable Activity'])
#lets form a new dataframe df_new

df_new = df
#Lets make a list of all categorical columns

cat_column = df_new.select_dtypes(exclude=['int64','float64']).columns
cat_column
#Lets make a list of all continuous variables with 'Converted' column removed

cont_column = df_new.select_dtypes(include=['int64','float64']).columns.drop('Converted')
cont_column
#Create dummies for categorical variables and dropping the first 

dummies = pd.get_dummies(df_new[cat_column],drop_first=True)
dummies
#Merging the dummies and the original data frame

df_new = pd.concat([df,dummies],axis=1)
df_new.head()
#Lets drop the original columns and only keep their dummied column

df_new = df_new.drop(cat_column,axis=1)
df_new.columns
#Checking the final shape of the dataframe
df_new.shape
#Lets split the data

from sklearn.model_selection import train_test_split
X = df_new.drop(['Converted'],axis=1)
y = df_new['Converted']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
#Lets fit and transform the data. We wont use scaling on categorical variables as they are already with absolute 0 or 1.So lets leave them untouched.We will scale the continuous variables with minmax scaler

X_train[cont_column] = scaler.fit_transform(X_train[cont_column])
X_train.head()
#We transform the test dataset based on training data

X_test[cont_column] = scaler.transform(X_test[cont_column])
X_test.head()
import statsmodels.api as sm
#Lets fit the model by adding a constant to the training and test data set

ls = sm.GLM(y_train,(sm.add_constant(X_train)), family = sm.families.Binomial())
ls.fit().summary()
#Lets use RFE to select the best 20 attributes from the dataset

from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
from sklearn.feature_selection import RFE
rfe = RFE(logreg, 20)             # running RFE with 20 variables as output
rfe = rfe.fit(X_train, y_train)
#Checking the support
rfe.support_
#Lets make a series of the columns selected

col = X_train.columns[rfe.support_]
#Fitting the model with the columns determined from RFE

X_train_sm = sm.add_constant(X_train[col])
logm2 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())
res = logm2.fit()
res.summary()
from statsmodels.stats.outliers_influence import variance_inflation_factor
#Lets check the VIF of all the attributes

vif = pd.DataFrame()
vif['Features'] = X_train[col].columns
vif['VIF'] = [variance_inflation_factor(X_train[col].values, i) for i in range(X_train[col].shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif
#Lets start with removing the variable with high VIF and high p value ie- 'Last Activity_SMS Sent'

remove=['Last Activity_SMS Sent']
logm2 = sm.GLM(y_train,X_train_sm.drop(remove,axis=1), family = sm.families.Binomial())
res = logm2.fit()
res.summary()
vif = pd.DataFrame()
vif['Features'] = X_train[col].drop(remove,axis=1).columns
vif['VIF'] = [variance_inflation_factor(X_train[col].drop(remove,axis=1).values, i) for i in range(X_train[col].drop(remove,axis=1).shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif
remove=['Last Activity_SMS Sent','Lead Source_Others']
logm2 = sm.GLM(y_train,X_train_sm.drop(remove,axis=1), family = sm.families.Binomial())
res = logm2.fit()
res.summary()
vif = pd.DataFrame()
vif['Features'] = X_train[col].drop(remove,axis=1).columns
vif['VIF'] = [variance_inflation_factor(X_train[col].drop(remove,axis=1).values, i) for i in range(X_train[col].drop(remove,axis=1).shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif
remove=['Last Activity_SMS Sent','Do Not Call_Yes','Lead Source_Others']
logm2 = sm.GLM(y_train,X_train_sm.drop(remove,axis=1), family = sm.families.Binomial())
res = logm2.fit()
res.summary()
vif = pd.DataFrame()
vif['Features'] = X_train[col].drop(remove,axis=1).columns
vif['VIF'] = [variance_inflation_factor(X_train[col].drop(remove,axis=1).values, i) for i in range(X_train[col].drop(remove,axis=1).shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif
remove=['Last Activity_SMS Sent','Do Not Call_Yes','Lead Source_Others','Lead Source_Organic Search']
logm2 = sm.GLM(y_train,X_train_sm.drop(remove,axis=1), family = sm.families.Binomial())
res = logm2.fit()
res.summary()
vif = pd.DataFrame()
vif['Features'] = X_train[col].drop(remove,axis=1).columns
vif['VIF'] = [variance_inflation_factor(X_train[col].drop(remove,axis=1).values, i) for i in range(X_train[col].drop(remove,axis=1).shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif
remove=['Last Activity_SMS Sent','Do Not Call_Yes','Lead Source_Others','Lead Source_Organic Search','Lead Origin_Landing Page Submission']
logm2 = sm.GLM(y_train,X_train_sm.drop(remove,axis=1), family = sm.families.Binomial())
res = logm2.fit()
res.summary()
vif = pd.DataFrame()
vif['Features'] = X_train[col].drop(remove,axis=1).columns
vif['VIF'] = [variance_inflation_factor(X_train[col].drop(remove,axis=1).values, i) for i in range(X_train[col].drop(remove,axis=1).shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif
#Lets predict on training dataset

y_train_pred = res.predict(X_train_sm.drop(remove,axis=1))
y_train_pred[:10]
#reshaping the predicted series

y_train_pred = y_train_pred.values.reshape(-1)
y_train_pred[:10]
#Transforming into a dataframe

y_train_pred_final = pd.DataFrame({'Converted':y_train.values, 'Conversion_Rate':y_train_pred})
y_train_pred_final['Conversion_Rate'] = y_train_pred_final['Conversion_Rate']
y_train_pred_final.head()
#Lets calculate the predicted conversion with different cut off points

numbers = [float(x)/10 for x in range(10)]
for i in numbers:
    y_train_pred_final[i]= y_train_pred_final.Conversion_Rate.map(lambda x: 1 if x > i else 0)
y_train_pred_final.head()
# Now let's calculate accuracy sensitivity and specificity for various probability cutoffs.
cutoff_df = pd.DataFrame( columns = ['prob','accuracy','sensi','speci'])
from sklearn.metrics import confusion_matrix

# TP = confusion[1,1] # true positive 
# TN = confusion[0,0] # true negatives
# FP = confusion[0,1] # false positives
# FN = confusion[1,0] # false negatives

num = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
for i in num:
    cm1 = confusion_matrix(y_train_pred_final.Converted, y_train_pred_final[i] )
    total1=sum(sum(cm1))
    accuracy = (cm1[0,0]+cm1[1,1])/total1
    
    speci = cm1[0,0]/(cm1[0,0]+cm1[0,1])
    sensi = cm1[1,1]/(cm1[1,0]+cm1[1,1])
    cutoff_df.loc[i] =[ i ,accuracy,sensi,speci]
print(cutoff_df)
# Let's plot accuracy sensitivity and specificity for various probabilities.
cutoff_df.plot.line(x='prob', y=['accuracy','sensi','speci'])
plt.show()
#Checking the ROC Curve and the AUC score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
def draw_roc( actual, probs ):
    fpr, tpr, thresholds = roc_curve( actual, probs,
                                              drop_intermediate = False )
    auc_score = roc_auc_score( actual, probs )
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
fpr, tpr, thresholds = roc_curve( y_train_pred_final.Converted, y_train_pred_final.Conversion_Rate, drop_intermediate = False )
draw_roc(y_train_pred_final.Converted, y_train_pred_final.Conversion_Rate)
#Creating a new column predicted with cutoff 0.35
y_train_pred_final['Predicted'] = y_train_pred_final.Conversion_Rate.map(lambda x: 1 if x > 0.3 else 0)

# Let's see the head
y_train_pred_final.head()
#Lets plot the confusion matrix

cm = confusion_matrix(y_train_pred_final.Converted, y_train_pred_final.Predicted)
cm
#Accuracy

from sklearn.metrics import accuracy_score
accuracy_score(y_train_pred_final.Converted, y_train_pred_final.Predicted)

TP = cm[1,1] # true positive 
TN = cm[0,0] # true negatives
FP = cm[0,1] # false positives
FN = cm[1,0] # false negatives
#Sensitivity

print(TP/(TP+FN))
#Specificity

print(TN/(TN+FP))
#Precision

print(TP/(TP+FP))
#Recall

print(TP/(TP+FN))
#Adding the lead scores

y_train_pred_final['Lead Score'] = (y_train_pred_final['Conversion_Rate'])*100
y_train_pred_final
#lets filter out X test data by the RFE model
X_test = X_test[col]
#Lets filter out the new X test by the manual selection model
X_test = X_test.drop(remove,axis=1)
X_test.shape
#Lets add the constant
X_test_sm = sm.add_constant(X_test)
#Lets predict the test data

y_test_pred = res.predict(X_test_sm)
y_test_pred[:10]
#Reshaping the series

y_test_pred = y_test_pred.values.reshape(-1)
y_test_pred[:10]
#Forming the dataframe

y_test_pred_final = pd.DataFrame({'Converted':y_test.values, 'Conversion_Rate':y_test_pred})
y_test_pred_final['Conversion_Rate'] = y_test_pred_final['Conversion_Rate']
y_test_pred_final.head()
#Setting the cutoff of 0.3

y_test_pred_final['Predicted'] = y_test_pred_final.Conversion_Rate.map(lambda x: 1 if x > 0.3 else 0)

# Let's see the head
y_test_pred_final.head()
#Lets check the confusion matrix

cm = confusion_matrix(y_test_pred_final.Converted, y_test_pred_final.Predicted)
cm
TP = cm[1,1] # true positive 
TN = cm[0,0] # true negatives
FP = cm[0,1] # false positives
FN = cm[1,0] # false negatives
#Accuracy

from sklearn.metrics import accuracy_score
accuracy_score(y_test_pred_final.Converted, y_test_pred_final.Predicted)
#Sensitivity

print(TP/(TP+FN))
#Specificity

print(TN/(TN+FP))
#Precision

print(TP/(TP+FP))
#Recall

print(TP/(TP+FN))
#Adding the lead Scores
y_test_pred_final['Lead Score'] = y_test_pred_final['Conversion_Rate']*100
y_test_pred_final
#Lets group the lead scores into 3 different groups - cold lead, warm lead and hot lead

y_train_pred_final['Lead group'] = pd.cut(y_train_pred_final['Lead Score'],bins=[0,30,70,100],labels=['Cold Lead','Warm Lead','Hot Lead'])
y_train_pred_final
#Lets see the characteristics of the groups we made

sns.countplot(y_train_pred_final['Converted'],hue=y_train_pred_final['Lead group'])