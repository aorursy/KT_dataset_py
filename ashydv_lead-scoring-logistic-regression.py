# Supress Warnings

import warnings

warnings.filterwarnings('ignore')



# Importing libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



# visulaisation

from matplotlib.pyplot import xticks

%matplotlib inline



# Data display coustomization

pd.set_option('display.max_rows', 100)

pd.set_option('display.max_columns', 100)
data = pd.DataFrame(pd.read_csv('../input/Leads.csv'))

data.head(5) 
#checking duplicates

sum(data.duplicated(subset = 'Prospect ID')) == 0

# No duplicate values
data.shape
data.info()
data.describe()
# As we can observe that there are select values for many column.

#This is because customer did not select any option from the list, hence it shows select.

# Select values are as good as NULL.



# Converting 'Select' values to NaN.

data = data.replace('Select', np.nan)
data.isnull().sum()
round(100*(data.isnull().sum()/len(data.index)), 2)
# we will drop the columns having more than 70% NA values.

data = data.drop(data.loc[:,list(round(100*(data.isnull().sum()/len(data.index)), 2)>70)].columns, 1)
# Now we will take care of null values in each column one by one.
# Lead Quality: Indicates the quality of lead based on the data and intuition the the employee who has been assigned to the lead
data['Lead Quality'].describe()
sns.countplot(data['Lead Quality'])
# As Lead quality is based on the intution of employee, so if left blank we can impute 'Not Sure' in NaN safely.

data['Lead Quality'] = data['Lead Quality'].replace(np.nan, 'Not Sure')
sns.countplot(data['Lead Quality'])
# Asymmetrique Activity Index  |

# Asymmetrique Profile Index   \   An index and score assigned to each customer

# Asymmetrique Activity Score  |    based on their activity and their profile

# Asymmetrique Profile Score   \
fig, axs = plt.subplots(2,2, figsize = (10,7.5))

plt1 = sns.countplot(data['Asymmetrique Activity Index'], ax = axs[0,0])

plt2 = sns.boxplot(data['Asymmetrique Activity Score'], ax = axs[0,1])

plt3 = sns.countplot(data['Asymmetrique Profile Index'], ax = axs[1,0])

plt4 = sns.boxplot(data['Asymmetrique Profile Score'], ax = axs[1,1])

plt.tight_layout()
# There is too much variation in thes parameters so its not reliable to impute any value in it. 

# 45% null values means we need to drop these columns.
data = data.drop(['Asymmetrique Activity Index','Asymmetrique Activity Score','Asymmetrique Profile Index','Asymmetrique Profile Score'],1)
round(100*(data.isnull().sum()/len(data.index)), 2)
# City
data.City.describe()
sns.countplot(data.City)

xticks(rotation = 90)
# Around 60% of the data is Mumbai so we can impute Mumbai in the missing values.
data['City'] = data['City'].replace(np.nan, 'Mumbai')
# Specailization
data.Specialization.describe()
sns.countplot(data.Specialization)

xticks(rotation = 90)
# It maybe the case that lead has not entered any specialization if his/her option is not availabe on the list,

#  may not have any specialization or is a student.

# Hence we can make a category "Others" for missing values. 
data['Specialization'] = data['Specialization'].replace(np.nan, 'Others')
round(100*(data.isnull().sum()/len(data.index)), 2)
# Tags
data.Tags.describe()
fig, axs = plt.subplots(figsize = (15,7.5))

sns.countplot(data.Tags)

xticks(rotation = 90)
# Blanks in the tag column may be imputed by 'Will revert after reading the email'.
data['Tags'] = data['Tags'].replace(np.nan, 'Will revert after reading the email')
# What matters most to you in choosing a course
data['What matters most to you in choosing a course'].describe()
# Blanks in the this column may be imputed by 'Better Career Prospects'.
data['What matters most to you in choosing a course'] = data['What matters most to you in choosing a course'].replace(np.nan, 'Better Career Prospects')
# Occupation
data['What is your current occupation'].describe()
# 86% entries are of Unemployed so we can impute "Unemployed" in it.
data['What is your current occupation'] = data['What is your current occupation'].replace(np.nan, 'Unemployed')
# Country
# Country is India for most values so let's impute the same in missing values.

data['Country'] = data['Country'].replace(np.nan, 'India')
round(100*(data.isnull().sum()/len(data.index)), 2)
# Rest missing values are under 2% so we can drop these rows.

data.dropna(inplace = True)
round(100*(data.isnull().sum()/len(data.index)), 2)
data.to_csv('Leads_cleaned')
# Converted is the target variable, Indicates whether a lead has been successfully converted (1) or not (0).
Converted = (sum(data['Converted'])/len(data['Converted'].index))*100

Converted
sns.countplot(x = "Lead Origin", hue = "Converted", data = data)

xticks(rotation = 90)
fig, axs = plt.subplots(figsize = (15,7.5))

sns.countplot(x = "Lead Source", hue = "Converted", data = data)

xticks(rotation = 90)
data['Lead Source'] = data['Lead Source'].replace(['google'], 'Google')

data['Lead Source'] = data['Lead Source'].replace(['Click2call', 'Live Chat', 'NC_EDM', 'Pay per Click Ads', 'Press_Release',

  'Social Media', 'WeLearn', 'bing', 'blog', 'testone', 'welearnblog_Home', 'youtubechannel'], 'Others')

sns.countplot(x = "Lead Source", hue = "Converted", data = data)

xticks(rotation = 90)
fig, axs = plt.subplots(1,2,figsize = (15,7.5))

sns.countplot(x = "Do Not Email", hue = "Converted", data = data, ax = axs[0])

sns.countplot(x = "Do Not Call", hue = "Converted", data = data, ax = axs[1])
data['TotalVisits'].describe(percentiles=[0.05,.25, .5, .75, .90, .95, .99])
sns.boxplot(data['TotalVisits'])
# As we can see there are a number of outliers in the data.

# We will cap the outliers to 95% value for analysis.
percentiles = data['TotalVisits'].quantile([0.05,0.95]).values

data['TotalVisits'][data['TotalVisits'] <= percentiles[0]] = percentiles[0]

data['TotalVisits'][data['TotalVisits'] >= percentiles[1]] = percentiles[1]
sns.boxplot(data['TotalVisits'])
sns.boxplot(y = 'TotalVisits', x = 'Converted', data = data)
data['Total Time Spent on Website'].describe()
sns.boxplot(data['Total Time Spent on Website'])
sns.boxplot(y = 'Total Time Spent on Website', x = 'Converted', data = data)
data['Page Views Per Visit'].describe()
sns.boxplot(data['Page Views Per Visit'])
# As we can see there are a number of outliers in the data.

# We will cap the outliers to 95% value for analysis.
percentiles = data['Page Views Per Visit'].quantile([0.05,0.95]).values

data['Page Views Per Visit'][data['Page Views Per Visit'] <= percentiles[0]] = percentiles[0]

data['Page Views Per Visit'][data['Page Views Per Visit'] >= percentiles[1]] = percentiles[1]
sns.boxplot(data['Page Views Per Visit'])
sns.boxplot(y = 'Page Views Per Visit', x = 'Converted', data = data)
data['Last Activity'].describe()
fig, axs = plt.subplots(figsize = (15,5))

sns.countplot(x = "Last Activity", hue = "Converted", data = data)

xticks(rotation = 90)
# Let's keep considerable last activities as such and club all others to "Other_Activity"

data['Last Activity'] = data['Last Activity'].replace(['Had a Phone Conversation', 'View in browser link Clicked', 

                                                       'Visited Booth in Tradeshow', 'Approached upfront',

                                                       'Resubscribed to emails','Email Received', 'Email Marked Spam'], 'Other_Activity')
fig, axs = plt.subplots(figsize = (10,5))

sns.countplot(x = "Last Activity", hue = "Converted", data = data)

xticks(rotation = 90)
data.Country.describe()
data.Specialization.describe()
data['Specialization'] = data['Specialization'].replace(['Others'], 'Other_Specialization')
fig, axs = plt.subplots(figsize = (15,5))

sns.countplot(x = "Specialization", hue = "Converted", data = data)

xticks(rotation = 90)
data['What is your current occupation'].describe()
data['What is your current occupation'] = data['What is your current occupation'].replace(['Other'], 'Other_Occupation')
fig, axs = plt.subplots(figsize = (10,5))

sns.countplot(x = "What is your current occupation", hue = "Converted", data = data)

xticks(rotation = 90)
data['What matters most to you in choosing a course'].describe()

data.Search.describe()
data.Magazine.describe()
data['Newspaper Article'].describe()
data['X Education Forums'].describe()
data['Newspaper'].describe()
data['Digital Advertisement'].describe()
data['Through Recommendations'].describe()
data['Receive More Updates About Our Courses'].describe()
data.Tags.describe()
fig, axs = plt.subplots(figsize = (15,5))

sns.countplot(x = "Tags", hue = "Converted", data = data)

xticks(rotation = 90)
# Let's keep considerable last activities as such and club all others to "Other_Activity"

data['Tags'] = data['Tags'].replace(['In confusion whether part time or DLP', 'in touch with EINS','Diploma holder (Not Eligible)',

                                     'Approached upfront','Graduation in progress','number not provided', 'opp hangup','Still Thinking',

                                    'Lost to Others','Shall take in the next coming month','Lateral student','Interested in Next batch',

                                    'Recognition issue (DEC approval)','Want to take admission but has financial problems',

                                    'University not recognized'], 'Other_Tags')
fig, axs = plt.subplots(figsize = (10,5))

sns.countplot(x = "Tags", hue = "Converted", data = data)

xticks(rotation = 90)
data['Lead Quality'].describe()
fig, axs = plt.subplots(figsize = (10,5))

sns.countplot(x = "Lead Quality", hue = "Converted", data = data)

xticks(rotation = 90)
data['Update me on Supply Chain Content'].describe()
data['Get updates on DM Content'].describe()
data['I agree to pay the amount through cheque'].describe()
data['A free copy of Mastering The Interview'].describe()
data.City.describe()
fig, axs = plt.subplots(figsize = (10,5))

sns.countplot(x = "City", hue = "Converted", data = data)

xticks(rotation = 90)
data['Last Notable Activity'].describe()
fig, axs = plt.subplots(figsize = (10,5))

sns.countplot(x = "Last Notable Activity", hue = "Converted", data = data)

xticks(rotation = 90)
data = data.drop(['Lead Number','What matters most to you in choosing a course','Search','Magazine','Newspaper Article','X Education Forums','Newspaper',

           'Digital Advertisement','Through Recommendations','Receive More Updates About Our Courses','Update me on Supply Chain Content',

           'Get updates on DM Content','I agree to pay the amount through cheque','A free copy of Mastering The Interview','Country'],1)
data.shape
data.head()
# List of variables to map



varlist =  ['Do Not Email', 'Do Not Call']



# Defining the map function

def binary_map(x):

    return x.map({'Yes': 1, "No": 0})



# Applying the function to the housing list

data[varlist] = data[varlist].apply(binary_map)
# Creating a dummy variable for some of the categorical variables and dropping the first one.

dummy1 = pd.get_dummies(data[['Lead Origin', 'Lead Source', 'Last Activity', 'Specialization','What is your current occupation',

                              'Tags','Lead Quality','City','Last Notable Activity']], drop_first=True)

dummy1.head()
# Adding the results to the master dataframe

data = pd.concat([data, dummy1], axis=1)

data.head()
data = data.drop(['Lead Origin', 'Lead Source', 'Last Activity', 'Specialization','What is your current occupation','Tags','Lead Quality','City','Last Notable Activity'], axis = 1)
data.head()
from sklearn.model_selection import train_test_split



# Putting feature variable to X

X = data.drop(['Prospect ID','Converted'], axis=1)
X.head()
# Putting response variable to y

y = data['Converted']



y.head()
# Splitting the data into train and test

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=100)
from sklearn.preprocessing import StandardScaler



scaler = StandardScaler()



X_train[['TotalVisits','Total Time Spent on Website','Page Views Per Visit']] = scaler.fit_transform(X_train[['TotalVisits','Total Time Spent on Website','Page Views Per Visit']])



X_train.head()
# Checking the Churn Rate

Converted = (sum(data['Converted'])/len(data['Converted'].index))*100

Converted
import statsmodels.api as sm
# Logistic regression model

logm1 = sm.GLM(y_train,(sm.add_constant(X_train)), family = sm.families.Binomial())

logm1.fit().summary()
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()



from sklearn.feature_selection import RFE

rfe = RFE(logreg, 15)             # running RFE with 15 variables as output

rfe = rfe.fit(X_train, y_train)
rfe.support_
list(zip(X_train.columns, rfe.support_, rfe.ranking_))
col = X_train.columns[rfe.support_]

col
X_train.columns[~rfe.support_]
X_train_sm = sm.add_constant(X_train[col])

logm2 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())

res = logm2.fit()

res.summary()
col1 = col.drop('Tags_invalid number',1)
col1
X_train_sm = sm.add_constant(X_train[col1])

logm2 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())

res = logm2.fit()

res.summary()
col2 = col1.drop('Tags_wrong number given',1)
col2
X_train_sm = sm.add_constant(X_train[col2])

logm2 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())

res = logm2.fit()

res.summary()
# Getting the predicted values on the train set

y_train_pred = res.predict(X_train_sm)

y_train_pred[:10]
y_train_pred = y_train_pred.values.reshape(-1)

y_train_pred[:10]
y_train_pred_final = pd.DataFrame({'Converted':y_train.values, 'Converted_prob':y_train_pred})

y_train_pred_final['Prospect ID'] = y_train.index

y_train_pred_final.head()
y_train_pred_final['predicted'] = y_train_pred_final.Converted_prob.map(lambda x: 1 if x > 0.5 else 0)



# Let's see the head

y_train_pred_final.head()
from sklearn import metrics



# Confusion matrix 

confusion = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final.predicted )

print(confusion)
# Predicted     not_churn    churn

# Actual

# not_churn        3270      365

# churn            579       708  
# Let's check the overall accuracy.

print(metrics.accuracy_score(y_train_pred_final.Converted, y_train_pred_final.predicted))
# Check for the VIF values of the feature variables. 

from statsmodels.stats.outliers_influence import variance_inflation_factor
# Create a dataframe that will contain the names of all the feature variables and their respective VIFs

vif = pd.DataFrame()

vif['Features'] = X_train[col2].columns

vif['VIF'] = [variance_inflation_factor(X_train[col].values, i) for i in range(X_train[col2].shape[1])]

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
fpr, tpr, thresholds = metrics.roc_curve( y_train_pred_final.Converted, y_train_pred_final.Converted_prob, drop_intermediate = False )
draw_roc(y_train_pred_final.Converted, y_train_pred_final.Converted_prob)
# Let's create columns with different probability cutoffs 

numbers = [float(x)/10 for x in range(10)]

for i in numbers:

    y_train_pred_final[i]= y_train_pred_final.Converted_prob.map(lambda x: 1 if x > i else 0)

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

    cm1 = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final[i] )

    total1=sum(sum(cm1))

    accuracy = (cm1[0,0]+cm1[1,1])/total1

    

    speci = cm1[0,0]/(cm1[0,0]+cm1[0,1])

    sensi = cm1[1,1]/(cm1[1,0]+cm1[1,1])

    cutoff_df.loc[i] =[ i ,accuracy,sensi,speci]

print(cutoff_df)
# Let's plot accuracy sensitivity and specificity for various probabilities.

cutoff_df.plot.line(x='prob', y=['accuracy','sensi','speci'])

plt.show()
#### From the curve above, 0.2 is the optimum point to take it as a cutoff probability.



y_train_pred_final['final_predicted'] = y_train_pred_final.Converted_prob.map( lambda x: 1 if x > 0.2 else 0)



y_train_pred_final.head()
y_train_pred_final['Lead_Score'] = y_train_pred_final.Converted_prob.map( lambda x: round(x*100))



y_train_pred_final.head()
# Let's check the overall accuracy.

metrics.accuracy_score(y_train_pred_final.Converted, y_train_pred_final.final_predicted)



confusion2 = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final.final_predicted )

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
#Looking at the confusion matrix again



confusion = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final.predicted )

confusion
##### Precision

TP / TP + FP



confusion[1,1]/(confusion[0,1]+confusion[1,1])
##### Recall

TP / TP + FN



confusion[1,1]/(confusion[1,0]+confusion[1,1])
from sklearn.metrics import precision_score, recall_score
precision_score(y_train_pred_final.Converted , y_train_pred_final.predicted)
recall_score(y_train_pred_final.Converted, y_train_pred_final.predicted)
from sklearn.metrics import precision_recall_curve
y_train_pred_final.Converted, y_train_pred_final.predicted
p, r, thresholds = precision_recall_curve(y_train_pred_final.Converted, y_train_pred_final.Converted_prob)
plt.plot(thresholds, p[:-1], "g-")

plt.plot(thresholds, r[:-1], "r-")

plt.show()
X_test[['TotalVisits','Total Time Spent on Website','Page Views Per Visit']] = scaler.fit_transform(X_test[['TotalVisits','Total Time Spent on Website','Page Views Per Visit']])



X_train.head()
X_test = X_test[col2]

X_test.head()
X_test_sm = sm.add_constant(X_test)
y_test_pred = res.predict(X_test_sm)
y_test_pred[:10]
# Converting y_pred to a dataframe which is an array

y_pred_1 = pd.DataFrame(y_test_pred)
# Let's see the head

y_pred_1.head()
# Converting y_test to dataframe

y_test_df = pd.DataFrame(y_test)
# Putting CustID to index

y_test_df['Prospect ID'] = y_test_df.index
# Removing index for both dataframes to append them side by side 

y_pred_1.reset_index(drop=True, inplace=True)

y_test_df.reset_index(drop=True, inplace=True)
# Appending y_test_df and y_pred_1

y_pred_final = pd.concat([y_test_df, y_pred_1],axis=1)
y_pred_final.head()
# Renaming the column 

y_pred_final= y_pred_final.rename(columns={ 0 : 'Converted_prob'})
# Rearranging the columns

y_pred_final = y_pred_final.reindex_axis(['Prospect ID','Converted','Converted_prob'], axis=1)
# Let's see the head of y_pred_final

y_pred_final.head()
y_pred_final['final_predicted'] = y_pred_final.Converted_prob.map(lambda x: 1 if x > 0.2 else 0)
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