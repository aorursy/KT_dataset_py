# *********************************************************************************************************************

# * @Assignment: Machine Learning 1 : Lead Scoring Case study

# *********************************************************************************************************************

# *

# * @author    : Poonam Yadav and Hitesh Yevale

# * @version   : v0.1.1

# * @StartDate : 09-Mar-2020

# 

# *********************************************************************************************************************
# *********************************************************************************************************************

# * Suppress Warnings

# *********************************************************************************************************************

import warnings

warnings.filterwarnings('ignore')
# *********************************************************************************************************************

# * Import Libraries 

# *********************************************************************************************************************

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import datetime
# *********************************************************************************************************************

# To Scale our data

from sklearn.preprocessing import scale

from sklearn.preprocessing import StandardScaler



#Model building

from sklearn.datasets import fetch_openml

from sklearn.decomposition import PCA

from sklearn.preprocessing import StandardScaler

from sklearn import metrics

from sklearn.model_selection import train_test_split

import statsmodels.api as sm

from sklearn.metrics import precision_score, recall_score

from sklearn.metrics import precision_recall_curve

from random import sample

from numpy.random import uniform

import numpy as np

from math import isnan





#display setting

pd.set_option('display.max_rows', 100)

pd.set_option('display.max_columns', 200)



 

# *********************************************************************************************************************
# *********************************************************************************************************************

# * Loading Data frames

# *********************************************************************************************************************

Lead_score= pd.read_csv("../input/lead-scoring-x-online-education/Leads X Education.csv")

Lead_score_backup= Lead_score
# ********************************************************************************************************************* 

# * 1. Importing required libraries

# * 2. Data Reading

# * 3. Data understanding 

# * 4. Data cleansing 

# * 5. Data Preparation

# * 6. Model building

# * 7. Final observations
Lead_score.head()
Lead_score.shape
Lead_score.info()
Lead_score.describe()
# Checking for duplicate rows in the dataset

Lead_score.loc[Lead_score.duplicated()]
#Conclusion: no duplicates found
Lead_score = Lead_score.replace('Select', np.nan)
# % of null values

round(100*(Lead_score.isnull().sum()/len(Lead_score.index)), 2)

Lead_score = Lead_score.drop(Lead_score.loc[:,list(round(100*(Lead_score.isnull().sum()/len(Lead_score.index)), 2)>70)].columns, 1)
round(100*(Lead_score.isnull().sum()/len(Lead_score.index)), 2)
Lead_score['Country'] = Lead_score['Country'].apply(lambda x: 'India' if x=='India' else 'Outside India')

Lead_score['Country'].value_counts()
Lead_score['Country'].isnull().sum()
Lead_score.loc[pd.isnull(Lead_score['Country']), ['Country']] = 'India'
Lead_score['Country'].isnull().sum()
Lead_score['Specialization'].value_counts()
# Lets check the number of null values in this column

Lead_score['Specialization'].isnull().sum()
# Since the amount of null values is high , Lets compute the null values and replace them with text 'Unknown'

Lead_score['Specialization'].fillna("Unknown", inplace = True)

Lead_score['Specialization'].value_counts()
Lead_score['What is your current occupation'].isnull().sum()
Lead_score['What is your current occupation'].value_counts()
Lead_score['What is your current occupation'].fillna("Unknown", inplace = True)

Lead_score['What is your current occupation'].value_counts()
Lead_score['What matters most to you in choosing a course'].isnull().sum()
Lead_score['What matters most to you in choosing a course'].unique()
# Since this column does not give away a lot of information and mostly are null or 'other' we can drop the column

Lead_score = Lead_score.drop('What matters most to you in choosing a course' , axis= 1)
Lead_score['Tags'].value_counts()
# This column does not add much value to the analysis , hence dropping it. 

Lead_score = Lead_score.drop('Tags',axis =1)
Lead_score['Lead Quality'].unique()
sns.countplot(Lead_score['Lead Quality'])
# Here all the null values are as good as not sure.



Lead_score['Lead Quality'] = Lead_score['Lead Quality'].replace(np.nan, 'Not Sure')
sns.countplot(Lead_score['Lead Quality'])

    
Lead_score['City'].fillna("unknown",inplace = True)

Lead_score['City'].value_counts()
Lead_score = Lead_score.drop(['Asymmetrique Activity Score', 'Asymmetrique Profile Score'], axis=1)
Lead_score['Asymmetrique Activity Index'].fillna("Unknown", inplace = True)

Lead_score['Asymmetrique Activity Index'].value_counts()

Lead_score['Asymmetrique Profile Index'].fillna("Unknown", inplace = True)

Lead_score['Asymmetrique Profile Index'].value_counts()
print(Lead_score.columns)
print(Lead_score['Magazine'].value_counts())

print(Lead_score['Receive More Updates About Our Courses'].value_counts())

print(Lead_score['Update me on Supply Chain Content'].value_counts())

print(Lead_score['I agree to pay the amount through cheque'].value_counts())

# Since the above columns have only 1 unique value they do not add any value to the data , let's drop them and all similar

# columns from the data 
Lead_score= Lead_score.loc[:,Lead_score.nunique()!=1]
# Prospect ID gives out the same information as Lead Number , therefore we can drop it from  the dataset 

Lead_score = Lead_score.drop('Prospect ID' , axis=1)
print(Lead_score.columns)
round(100*(Lead_score.isnull().sum()/len(Lead_score.index)), 2)
print(Lead_score['Do Not Email'].value_counts())

print(Lead_score['Do Not Call'].value_counts())

print(Lead_score['Search'].value_counts())

print(Lead_score['Through Recommendations'].value_counts())

print(Lead_score['A free copy of Mastering The Interview'].value_counts())

print(Lead_score['Newspaper Article'].value_counts())

print(Lead_score['X Education Forums'].value_counts())

print(Lead_score['Newspaper'].value_counts())

print(Lead_score['Digital Advertisement'].value_counts())
# Since these columns do not give away any specific information we can drop these columns .

Lead_score = Lead_score.drop(columns =['Do Not Email' , 'Do Not Call' , 'Search' ,'Through Recommendations' ,'A free copy of Mastering The Interview','Newspaper Article', 'X Education Forums', 'Newspaper', 

            'Digital Advertisement'],axis=1)
Lead_score.columns
Lead_score.shape
# Lets drop rows with null values

Lead_score= Lead_score.dropna()
round(100*(Lead_score.isnull().sum()/len(Lead_score.index)), 2)
# Creating dummy variables for categorial variables



categorical_columns = ['Lead Origin' ,'Country','Lead Quality' ,'Lead Source','Last Activity' , 'Specialization' , 'What is your current occupation'

                                       ,'City' ,'Last Notable Activity' ,'Asymmetrique Activity Index' , 'Asymmetrique Profile Index']



for x in categorical_columns:

    cont = pd.get_dummies(Lead_score[x],prefix=x,drop_first=True)

    Lead_score = pd.concat([Lead_score,cont],axis=1)

Lead_score.shape
print(Lead_score.columns)
Lead_score.head(1)
#dropping the original columns we created dummies for 

Lead_score = Lead_score.drop(columns = ['Lead Origin' ,'Country','Lead Quality' ,'Lead Source','Last Activity' , 'Specialization' , 'What is your current occupation'

                                       ,'City' ,'Last Notable Activity' ,'Asymmetrique Activity Index' , 'Asymmetrique Profile Index'] , axis =1)

Lead_score.shape
Lead_score.head()
#checking for outliers 

outlier_check = Lead_score[['TotalVisits','Total Time Spent on Website','Page Views Per Visit']]

outlier_check.describe(percentiles=[.01,.1,.2,.25, .5, .75, .90, .95, .99])
Q1 = Lead_score['Page Views Per Visit'].quantile(0.25)

Q3 = Lead_score['Page Views Per Visit'].quantile(0.75)

IQR = Q3 - Q1

Lead_score=Lead_score.loc[(Lead_score['Page Views Per Visit'] >= Q1 - 1.5*IQR) & (Lead_score['Page Views Per Visit'] <= Q3 + 1.5*IQR)]
Q1 = Lead_score['TotalVisits'].quantile(0.25)

Q3 = Lead_score['TotalVisits'].quantile(0.75)

IQR = Q3 - Q1

Lead_score=Lead_score.loc[(Lead_score['TotalVisits'] >= Q1 - 1.5*IQR) & (Lead_score['TotalVisits'] <= Q3 + 1.5*IQR)]
Q1 = Lead_score['Total Time Spent on Website'].quantile(0.25)

Q3 = Lead_score['Total Time Spent on Website'].quantile(0.75)

IQR = Q3 - Q1

Lead_score=Lead_score.loc[(Lead_score['Total Time Spent on Website'] >= Q1 - 1.5*IQR) & (Lead_score['Total Time Spent on Website'] <= Q3 + 1.5*IQR)]
X = Lead_score.drop(['Lead Number','Converted'], axis=1)



X.head()

y = Lead_score['Converted']



y.head()
# Splitting the data into train and test

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=100)
#scaling continuous variables

scaler = StandardScaler()



X_train[['TotalVisits','Total Time Spent on Website','Page Views Per Visit']] = scaler.fit_transform(X_train[['TotalVisits','Total Time Spent on Website','Page Views Per Visit']])



X_train.head()
converted = (sum(Lead_score['Converted'])/len(Lead_score['Converted'].index))*100

converted
# Almost 37% conversion rate 
#Logistic regression model

logistics = sm.GLM(y_train,(sm.add_constant(X_train)), family = sm.families.Binomial())

logistics.fit().summary()

from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()
  # running RFE with 20 variables as output

from sklearn.feature_selection import RFE

rfe = RFE(logreg, 20)           

rfe = rfe.fit(X_train, y_train)
rfe.support_
list(zip(X_train.columns, rfe.support_, rfe.ranking_))
X_train.columns[~rfe.support_]
cols = X_train.columns[rfe.support_]
X_train_sm = sm.add_constant(X_train[cols])

logm1 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())

res = logm1.fit()

res.summary()
from statsmodels.stats.outliers_influence import variance_inflation_factor
vif = pd.DataFrame()

vif['Features'] = X_train[cols].columns

vif['VIF'] = [variance_inflation_factor(X_train[cols].values, i) for i in range(X_train[cols].shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
plt.figure(figsize=(20,15), dpi=80, facecolor='w', edgecolor='k', frameon='True')



cor = X_train[cols].corr()

sns.heatmap(cor, annot=True, cmap="YlGnBu")



plt.tight_layout()

plt.show()
#lets drop the variables with high VIF
cols = cols.drop('Lead Quality_Not Sure' , 1)
X_train_sm = sm.add_constant(X_train[cols])

logm2 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())

res = logm2.fit()

res.summary()
vif = pd.DataFrame()

vif['Features'] = X_train[cols].columns

vif['VIF'] = [variance_inflation_factor(X_train[cols].values, i) for i in range(X_train[cols].shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
cols = cols.drop('Lead Source_Olark Chat' , 1)
X_train_sm = sm.add_constant(X_train[cols])

logm3 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())

res = logm3.fit()

res.summary()
vif = pd.DataFrame()

vif['Features'] = X_train[cols].columns

vif['VIF'] = [variance_inflation_factor(X_train[cols].values, i) for i in range(X_train[cols].shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
cols = cols.drop('Last Notable Activity_Modified' , 1)
X_train_sm = sm.add_constant(X_train[cols])

logm4 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())

res = logm4.fit()

res.summary()
vif = pd.DataFrame()

vif['Features'] = X_train[cols].columns

vif['VIF'] = [variance_inflation_factor(X_train[cols].values, i) for i in range(X_train[cols].shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
cols = cols.drop('Lead Quality_Might be',1)
X_train_sm = sm.add_constant(X_train[cols])

logm5 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())

res = logm5.fit()

res.summary()
vif = pd.DataFrame()

vif['Features'] = X_train[cols].columns

vif['VIF'] = [variance_inflation_factor(X_train[cols].values, i) for i in range(X_train[cols].shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
plt.figure(figsize=(20,15), dpi=80, facecolor='w', edgecolor='k', frameon='True')



cor = X_train[cols].corr()

sns.heatmap(cor, annot=True, cmap="YlGnBu")



plt.tight_layout()

plt.show()
# This shows now we have very less multi-collinearity compared to earlier Heatmap
y_train_pred = res.predict(X_train_sm)

y_train_pred[:10]
y_train_pred = y_train_pred.values.reshape(-1)

y_train_pred[:10]
y_train_pred_final = pd.DataFrame({'Converted':y_train.values, 'Conversion_Prob':y_train_pred})

y_train_pred_final['LeadID'] = y_train.index

y_train_pred_final.head()
y_train_pred_final['predicted'] = y_train_pred_final.Conversion_Prob.map(lambda x: 1 if x > 0.5 else 0)

y_train_pred_final.head()
# Confusion matrix 

confusion = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final.predicted )

print(confusion)

# Accuracy 

print(metrics.accuracy_score(y_train_pred_final.Converted, y_train_pred_final.predicted))
TP = confusion[1,1] # true positive 

TN = confusion[0,0] # true negatives

FP = confusion[0,1] # false positives

FN = confusion[1,0] # false negatives
# Sensitivity

TP / float(TP+FN)
# Specificity

TN / float(TN+FP)

# Calculate false postive rate - predicting churn when customer does not have churned

print(FP/ float(TN+FP))



# positive predictive value 

print (TP / float(TP+FP))



# Negative predictive value

print (TN / float(TN+ FN))
def plot_roc( actual, probs ):

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



    return fpr,tpr, thresholds
fpr, tpr, thresholds = metrics.roc_curve( y_train_pred_final.Converted, y_train_pred_final.Conversion_Prob, drop_intermediate = False )
plot_roc(y_train_pred_final.Converted, y_train_pred_final.Conversion_Prob)
def roc_auc(fpr,tpr):

    AreaUnderCurve = 0.

    for i in range(len(fpr)-1):

        AreaUnderCurve += (fpr[i+1]-fpr[i]) * (tpr[i+1]+tpr[i])

    AreaUnderCurve *= 0.5

    return AreaUnderCurve
auc = roc_auc(fpr,tpr)

auc
# Let's try and create columns with different probability cutoffs and figure the optimal cutoff 

numbers = [float(x)/10 for x in range(10)]

for i in numbers: 

    y_train_pred_final[i]= y_train_pred_final.Conversion_Prob.map(lambda x: 1 if x > i else 0)



y_train_pred_final.head()
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

fig = plt.figure(figsize = (12,8))

cutoff_df.plot.line(x='prob', y=['accuracy','sensi','speci'],figsize=(10,6))

plt.xticks(np.arange(0, 1, step=0.05), size = 12)

plt.show()

fig.savefig('threshold.png')
y_train_pred_final['final_predicted'] = y_train_pred_final.Conversion_Prob.map( lambda x: 1 if x > 0.425 else 0)



y_train_pred_final.head()
# Accuracy

metrics.accuracy_score(y_train_pred_final.Converted, y_train_pred_final.final_predicted)
confusion1 = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final.final_predicted)

confusion1
TP = confusion1[1,1] # true positive 

TN = confusion1[0,0] # true negatives

FP = confusion1[0,1] # false positives

FN = confusion1[1,0] # false negatives
# Sensitivity

TP / float(TP+FN)
# Specificity

TN / float(TN+FP)
# Calculating false postive rate - predicting churn when customer does not have churned

print(FP/ float(TN+FP))
# Positive predictive value 

print (TP / float(TP+FP))

# Negative predictive value

print (TN / float(TN+ FN))
precision = confusion1[1,1]/(confusion1[0,1]+confusion1[1,1])

precision
recall = confusion1[1,1]/(confusion1[1,0]+confusion1[1,1])

recall
precision_score(y_train_pred_final.Converted, y_train_pred_final.final_predicted)
recall_score(y_train_pred_final.Converted, y_train_pred_final.final_predicted)
y_train_pred_final.Converted, y_train_pred_final.final_predicted
p, r, thresholds = precision_recall_curve(y_train_pred_final.Converted, y_train_pred_final.Conversion_Prob)
plt.figure(figsize=(8, 4), dpi=100, facecolor='w', edgecolor='k', frameon='True')

plt.plot(thresholds, p[:-1], "g-")

plt.plot(thresholds, r[:-1], "r-")

plt.xticks(np.arange(0, 1, step=0.05))

plt.show()
F1 = 2*(precision*recall)/(precision+recall)

F1
X_test[['TotalVisits','Total Time Spent on Website','Page Views Per Visit']] = scaler.transform(X_test[['TotalVisits','Total Time Spent on Website','Page Views Per Visit']])

X_test.head()
X_test = X_test[cols]

X_test.head()
#Adding the constant 

X_test_sm = sm.add_constant(X_test)
#making predictions on the test data set

y_test_pred = res.predict(X_test_sm)
y_test_pred[:10]
#converting into an array

y_pred_1 = pd.DataFrame(y_test_pred)

y_pred_1.head()
y_test_df = pd.DataFrame(y_test)
y_test_df['LeadID'] = y_test_df.index
y_pred_1.reset_index(drop=True, inplace=True)

y_test_df.reset_index(drop=True, inplace=True)
y_pred_final = pd.concat([y_test_df, y_pred_1],axis=1)

y_pred_final.head()
y_pred_final.shape
y_pred_final= y_pred_final.rename(columns={ 0 : 'Conversion_Prob'})
y_pred_final = y_pred_final.reindex(['LeadID','Converted','Conversion_Prob'], axis=1)
y_pred_final.head()
y_pred_final['final_predicted'] = y_pred_final.Conversion_Prob.map(lambda x: 1 if x > 0.42 else 0)
y_pred_final.head()
#check the overall accuracy.

accuracy_score=metrics.accuracy_score(y_pred_final.Converted, y_pred_final.final_predicted)

accuracy_score
confusion_test = metrics.confusion_matrix(y_pred_final.Converted, y_pred_final.final_predicted )

print(confusion_test)
TP = confusion_test[1,1] # true positive 

TN = confusion_test[0,0] # true negatives

FP = confusion_test[0,1] # false positives

FN = confusion_test[1,0] # false negatives
# sensitivity of our logistic regression model

TP / float(TP+FN)

# specificity

TN / float(TN+FP)

# Calculate false postive rate - predicting converion when customer does not have converted

print(FP/ float(TN+FP))

# Positive predictive value 

print (TP / float(TP+FP))

# Negative predictive value

print (TN / float(TN+ FN))

# Precision

confusion_test[1,1]/(confusion_test[0,1]+confusion_test[1,1])
# Recall

confusion_test[1,1]/(confusion_test[1,0]+confusion_test[1,1])
y_pred_final.Converted, y_pred_final.final_predicted
p, r, thresholds = precision_recall_curve(y_pred_final.Converted, y_pred_final.Conversion_Prob)
plt.plot(thresholds, p[:-1], "g-")

plt.plot(thresholds, r[:-1], "r-")

plt.show()
fpr, tpr, thresholds = metrics.roc_curve( y_pred_final.Converted, y_pred_final.Conversion_Prob, drop_intermediate = False )
plot_roc(y_pred_final.Converted, y_pred_final.Conversion_Prob)
y_test_pred = y_test_pred * 100

y_test_pred[:10]
pd.options.display.float_format = '{:.2f}'.format

new_params = res.params[1:]

new_params
# Getting a relative coeffient value for all the features wrt the feature with the highest coefficient

feature_importance = new_params

feature_importance = 100.0 * (feature_importance / feature_importance.max())

feature_importance
# Sorting the feature variables based on their relative coefficient values

sorted_idx = np.argsort(feature_importance,kind='quicksort',order='list of str')

sorted_idx
# Plot showing the feature variables based on their relative coefficient values

# Plotting the scree plot

%matplotlib inline

fig = plt.figure(figsize = (12,8))

pos = np.arange(sorted_idx.shape[0]) + .5



featfig = plt.figure(figsize=(10,6))

featax = featfig.add_subplot(1, 1, 1)

featax.barh(pos, feature_importance[sorted_idx], align='center', color = 'tab:green',alpha=0.8)

featax.set_yticks(pos)

featax.set_yticklabels(np.array(X_train[cols].columns)[sorted_idx], fontsize=12)

featax.set_xlabel('Relative Feature Importance', fontsize=14)



plt.tight_layout()   

plt.show()

fig.savefig('Question1.png')

# selecting top 4 features

pd.DataFrame(feature_importance).reset_index().sort_values(by=0,ascending=False).head(4)