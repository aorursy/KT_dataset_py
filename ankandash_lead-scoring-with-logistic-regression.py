import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



import warnings

warnings.filterwarnings('ignore')
leads = pd.read_csv('/kaggle/input/lead-scoring-x-online-education/Leads X Education.csv')
leads.head()
leads.columns
leads.info()
leads.shape
# target variable

leads['Converted'].value_counts()
leads.describe()
# checking for the number of null values in each column 



leads.isnull().sum(axis = 0)
# checking for the percentage of null values in each column 



round((leads.isnull().sum(axis = 0)/ len(leads.index))*100 , 2)
leads.columns
# Dropping the columns 'Asymmetrique Activity Index' and 'Asymmetrique Profile Index' as there is score column for both



leads = leads.drop(['Asymmetrique Activity Index','Asymmetrique Profile Index'], axis = 1)
leads['Asymmetrique Activity Score'].value_counts()
sum(leads['Asymmetrique Activity Score'].isnull())
leads['Asymmetrique Activity Score'] = leads['Asymmetrique Activity Score'].fillna('Unknown')

print(leads['Asymmetrique Activity Score'].value_counts())

print('\n')

print('Number of null values = ',sum(leads['Asymmetrique Activity Score'].isnull()))
leads['Asymmetrique Profile Score'] = leads['Asymmetrique Profile Score'].fillna('Unknown')

leads['Asymmetrique Profile Score'].value_counts()
leads['Lead Quality'].value_counts()
sum(leads['Lead Quality'].isnull())
leads['Lead Quality'].fillna("Unknown", inplace = True)

leads['Lead Quality'].value_counts()
# Tags column

leads['Tags'].value_counts()
sum(leads['Tags'].isnull())
leads['Tags'] = leads['Tags'].fillna('Unknown')

leads['Tags'].value_counts()
sum(leads['Tags'].isnull())
# Country column 

sum(leads['Country']=='India')/len(leads.index)
leads['Country'] = leads['Country'].apply(lambda x: 'India' if x=='India' else 'Foreign Country')

leads['Country'].value_counts()
# Total visits column

leads['TotalVisits'].value_counts() 
leads['TotalVisits'].median() #Since the above column has lot of outliers we will impute with the median value
leads['TotalVisits'].replace(np.NaN, leads['TotalVisits'].median(), inplace =True)
# Page Views Per Visit column null values are similarly imputed using the median values



leads['Page Views Per Visit'].replace(np.NaN, leads['Page Views Per Visit'].median(), inplace =True)
leads['Last Activity'].value_counts()
sum(leads['Last Activity'].isnull())
leads['Last Activity'].fillna("Unknown", inplace = True)

leads['Last Activity'].value_counts()
leads['Specialization'].value_counts()
sum(leads['Specialization'].isnull())
leads['Specialization'].replace('Select', 'Unknown', inplace =True)

leads['Specialization'].value_counts()
leads['Specialization'].fillna("Unknown", inplace = True)

leads['Specialization'].value_counts()
leads['How did you hear about X Education'].value_counts()
leads = leads.drop('How did you hear about X Education', axis=1)
leads['What is your current occupation'].value_counts()
sum(leads['What is your current occupation'].isnull())
leads['What is your current occupation'].fillna("Unknown", inplace = True)

leads['What is your current occupation'].value_counts()
leads['What matters most to you in choosing a course'].value_counts()
sum(leads['What matters most to you in choosing a course'].isnull())
leads = leads.drop('What matters most to you in choosing a course', axis = 1)
leads['Lead Profile'].value_counts()
sum(leads['Lead Profile'].isnull())
leads['Lead Profile'].replace('Select', 'Unknown', inplace =True)

leads['Lead Profile'].value_counts()
leads['Lead Profile'].fillna("Unknown", inplace = True)

leads['Lead Profile'].value_counts()
# City column

leads['City'].value_counts()
sum(leads['City'].isnull())
leads['City'].fillna("Unknown", inplace = True) # Replacing null values with 'NotSpecified' 

leads['City'].value_counts()
leads['City'].replace('Select', 'Unknown', inplace =True)

leads['City'].value_counts()
# re-checking for the percentage of null values in each column 



round((leads.isnull().sum(axis = 0)/ len(leads.index))*100 , 2)
leads.shape
# removing all the rows with null values



leads = leads.dropna()
leads.shape
# checking again for missing values in the dataframe 



round((leads.isnull().sum(axis = 0)/ len(leads.index))*100 , 2)
leads.head()
leads.columns
for col in leads.columns:

    print(col, ':', leads[col].nunique())

    print('\n')
# Prospect ID and Lead Number are the same thing so having both the columsn is redundant so we will drop the Prospect ID column



leads = leads.drop('Prospect ID',axis=1)



# Also a lot of the columns have just one unique value so they are of no use as they do not provide any information so dropping them as well

leads = leads.drop(['Magazine','Receive More Updates About Our Courses',

                    'Update me on Supply Chain Content','Get updates on DM Content',

                    'I agree to pay the amount through cheque'], axis=1)
leads.head()
print(leads.shape)
leads.columns
def mapping(x):

    return x.map({'Yes':1, 'No':0})
col_list = ['Search',

            'Do Not Email',

            'Do Not Call',

            'Newspaper Article',

            'X Education Forums',

            'Newspaper',

            'Digital Advertisement',

            'Through Recommendations',

            'A free copy of Mastering The Interview']
leads[col_list] = leads[col_list].apply(mapping)
leads.head()
leads.columns
leads.info()
# creating dummy variables for some of the other categorical columns 

leads = pd.get_dummies(leads, columns=['Lead Origin', 'Lead Source', 'Country', 'Last Notable Activity'], drop_first=True)
# Creating dummmy variables for the rest of the columns and dropping the level called 'Unknown'





# Creating dummy variables for the variable 'City'

dummy = pd.get_dummies(leads['Asymmetrique Activity Score'], prefix='Asymmetrique Activity Score')

final_dummy = dummy.drop(['Asymmetrique Activity Score_Unknown'], 1)

leads = pd.concat([leads,final_dummy], axis=1)



# Creating dummy variables for the variable 'City'

dummy = pd.get_dummies(leads['Asymmetrique Profile Score'], prefix='Asymmetrique Profile Score')

final_dummy = dummy.drop(['Asymmetrique Profile Score_Unknown'], 1)

leads = pd.concat([leads,final_dummy], axis=1)



# Creating dummy variables for the variable 'Last Activity'

dummy = pd.get_dummies(leads['Last Activity'], prefix='Last Activity')

final_dummy = dummy.drop(['Last Activity_Unknown'], 1)

leads = pd.concat([leads,final_dummy], axis=1)



# Creating dummy variables for the variable 'What is your current occupation'

dummy = pd.get_dummies(leads['What is your current occupation'], prefix='What is your current occupation')

final_dummy = dummy.drop(['What is your current occupation_Unknown'], 1)

leads = pd.concat([leads,final_dummy], axis=1)



# Creating dummy variables for the variable 'Lead Profile'

dummy = pd.get_dummies(leads['Lead Profile'], prefix='Lead Profile')

final_dummy = dummy.drop(['Lead Profile_Unknown'], 1)

leads = pd.concat([leads,final_dummy], axis=1)



# Creating dummy variables for the variable 'Specialization'

dummy = pd.get_dummies(leads['Specialization'], prefix='Specialization')

final_dummy = dummy.drop(['Specialization_Unknown'], 1)

leads = pd.concat([leads,final_dummy], axis=1)



# Creating dummy variables for the variable 'City'

dummy = pd.get_dummies(leads['City'], prefix='City')

final_dummy = dummy.drop(['City_Unknown'], 1)

leads = pd.concat([leads,final_dummy], axis=1)



# Creating dummy variables for the variable 'City'

dummy = pd.get_dummies(leads['Lead Quality'], prefix='Lead Quality')

final_dummy = dummy.drop(['Lead Quality_Unknown'], 1)

leads = pd.concat([leads,final_dummy], axis=1)



# Creating dummy variables for the variable 'City'

dummy = pd.get_dummies(leads['Tags'], prefix='Tags')

final_dummy = dummy.drop(['Tags_Unknown'], 1)

leads = pd.concat([leads,final_dummy], axis=1)
leads.shape
leads = leads.drop(['Lead Quality','Asymmetrique Profile Score','Asymmetrique Activity Score','Last Activity', 

                    'What is your current occupation', 'Lead Profile','Specialization','City','Tags'],axis=1)
leads.shape
leads.head()
leads.info()
# checking for outliers in the continuous variables



numerical = leads[['TotalVisits','Total Time Spent on Website', 'Page Views Per Visit']]
numerical.describe()
plt.figure(figsize=(20,10))



plt.subplot(2,2,1)

sns.boxplot(numerical['TotalVisits'])



plt.subplot(2,2,2)

sns.boxplot(numerical['Total Time Spent on Website'])



plt.subplot(2,2,3)

sns.boxplot(numerical['Page Views Per Visit'])
# removing outliers using the IQR



Q1 = leads['TotalVisits'].quantile(0.25)

Q3 = leads['TotalVisits'].quantile(0.75)

IQR = Q3 - Q1

leads = leads.loc[(leads['TotalVisits'] >= Q1 - 1.5*IQR) & (leads['TotalVisits'] <= Q3 + 1.5*IQR)]



Q1 = leads['Page Views Per Visit'].quantile(0.25)

Q3 = leads['Page Views Per Visit'].quantile(0.75)

IQR = Q3 - Q1

leads=leads.loc[(leads['Page Views Per Visit'] >= Q1 - 1.5*IQR) & (leads['Page Views Per Visit'] <= Q3 + 1.5*IQR)]
plt.figure(figsize=(20,10))



plt.subplot(2,2,1)

sns.boxplot(leads['TotalVisits'])



plt.subplot(2,2,2)

sns.boxplot(leads['Total Time Spent on Website'])



plt.subplot(2,2,3)

sns.boxplot(leads['Page Views Per Visit'])
leads.shape
# Lets look at the head of the dataframe again

leads.head()
# Lets look at the info of the dataframe again

leads.info()
from sklearn.model_selection import train_test_split
X = leads.drop(['Lead Number', 'Converted'], axis = 1)

y = leads['Converted']
X.head()
y.head()
# Splitting the data into train and test sets

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=100)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train[['TotalVisits','Total Time Spent on Website','Page Views Per Visit']] = scaler.fit_transform(X_train[['TotalVisits','Total Time Spent on Website','Page Views Per Visit']])
X_train.head()
y.head()
round((y.sum()/len(y))*100,2) 
import statsmodels.api as sm
# logistic regression model



logm1 = sm.GLM(y_train,(sm.add_constant(X_train)), family=sm.families.Binomial())

logm1.fit().summary()
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
from sklearn.feature_selection import RFE

rfe = RFE(logreg, 20) # running RFE with 20 variables

rfe = rfe.fit(X_train,y_train)
rfe.support_
list(zip(X_train.columns, rfe.support_, rfe.ranking_))
col = X_train.columns[rfe.support_]
X_train.columns[~rfe.support_] # rfe.support_ = false 
X_train_sm = sm.add_constant(X_train[col])

logm2 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())

res = logm2.fit()

res.summary()
# Getting the predicted values on the train set

y_train_pred = res.predict(X_train_sm)

y_train_pred[:10]
y_train_pred = y_train_pred.values.reshape(-1)

y_train_pred[:10]
y_train_pred_final = pd.DataFrame({'Converted':y_train.values, 'Conversion_Prob':y_train_pred})

y_train_pred_final['LeadID'] = y_train.index

y_train_pred_final.head()
y_train_pred_final['Predicted'] = y_train_pred_final.Conversion_Prob.map(lambda x: 1 if x > 0.5 else 0)

y_train_pred_final.head()
from sklearn import metrics
# Confusion matrix 

confusion = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final.Predicted)

print(confusion)
# Let's check the overall accuracy.

print(metrics.accuracy_score(y_train_pred_final.Converted, y_train_pred_final.Predicted))
from statsmodels.stats.outliers_influence import variance_inflation_factor
# Create a dataframe that will contain the names of all the feature variables and their respective VIFs

vif = pd.DataFrame()

vif['Features'] = X_train[col].columns

vif['VIF'] = [variance_inflation_factor(X_train[col].values, i) for i in range(X_train[col].shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
col = col.drop('Tags_Diploma holder (Not Eligible)', 1)

col
# Let's re-run the model using the selected variables

X_train_sm = sm.add_constant(X_train[col])

logm3 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())

res = logm3.fit()

res.summary()
# Getting the predicted values on the train set

y_train_pred = res.predict(X_train_sm)

y_train_pred[:10]
y_train_pred = y_train_pred.values.reshape(-1)

y_train_pred[:10]
y_train_pred_final = pd.DataFrame({'Converted':y_train.values, 'Conversion_Prob':y_train_pred})

y_train_pred_final['LeadID'] = y_train.index

y_train_pred_final.head()
y_train_pred_final['Predicted'] = y_train_pred_final.Conversion_Prob.map(lambda x: 1 if x > 0.5 else 0)

y_train_pred_final.head()
# Confusion matrix 

confusion = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final.Predicted)

print(confusion)
# Let's check the overall accuracy.

print(metrics.accuracy_score(y_train_pred_final.Converted, y_train_pred_final.Predicted))
#### Checking VIFs again

vif = pd.DataFrame()

vif['Features'] = X_train[col].columns

vif['VIF'] = [variance_inflation_factor(X_train[col].values, i) for i in range(X_train[col].shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
col = col.drop('Tags_wrong number given', 1)



# Let's re-run the model using the selected variables

X_train_sm = sm.add_constant(X_train[col])

logm4 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())

res = logm4.fit()

res.summary()
y_train_pred = res.predict(X_train_sm)

y_train_pred[:10]
y_train_pred = y_train_pred.values.reshape(-1)

y_train_pred[:10]
y_train_pred_final = pd.DataFrame({'Converted':y_train.values, 'Conversion_Prob':y_train_pred})

y_train_pred_final['LeadID'] = y_train.index

y_train_pred_final.head()
y_train_pred_final['Predicted'] = y_train_pred_final.Conversion_Prob.map(lambda x: 1 if x > 0.5 else 0)

y_train_pred_final.head()
# Confusion matrix 

confusion = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final.Predicted)

print(confusion)
# Let's check the overall accuracy.

print(metrics.accuracy_score(y_train_pred_final.Converted, y_train_pred_final.Predicted))
#### Checking VIFs again

vif = pd.DataFrame()

vif['Features'] = X_train[col].columns

vif['VIF'] = [variance_inflation_factor(X_train[col].values, i) for i in range(X_train[col].shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
col = col.drop('Tags_number not provided', 1)



# Let's re-run the model using the selected variables

X_train_sm = sm.add_constant(X_train[col])

logm5 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())

res = logm5.fit()

res.summary()
y_train_pred = res.predict(X_train_sm)

y_train_pred[:10]
y_train_pred = y_train_pred.values.reshape(-1)

y_train_pred[:10]
y_train_pred_final = pd.DataFrame({'Converted':y_train.values, 'Conversion_Prob':y_train_pred})

y_train_pred_final['LeadID'] = y_train.index

y_train_pred_final.head()
y_train_pred_final['Predicted'] = y_train_pred_final.Conversion_Prob.map(lambda x: 1 if x > 0.5 else 0)

y_train_pred_final.head()
# Confusion matrix 

confusion = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final.Predicted)

print(confusion)
# Let's check the overall accuracy.

print(metrics.accuracy_score(y_train_pred_final.Converted, y_train_pred_final.Predicted))
#### Checking VIFs again

vif = pd.DataFrame()

vif['Features'] = X_train[col].columns

vif['VIF'] = [variance_inflation_factor(X_train[col].values, i) for i in range(X_train[col].shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
# correlation matrix 

plt.figure(figsize = (20,10),dpi=200)  

sns.heatmap(X_train[col].corr(),annot = True)

plt.show()



plt.savefig('corr.png')
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

    plt.title('Receiver operating characteristic (RoC) curve')

    plt.legend(loc="lower right")

    plt.show()



    return None
fpr, tpr, thresholds = metrics.roc_curve( y_train_pred_final.Converted, y_train_pred_final.Conversion_Prob, 

                                         drop_intermediate = False )
draw_roc(y_train_pred_final.Converted, y_train_pred_final.Conversion_Prob)
# Let's create columns with different probability cutoffs 

numbers = [float(x)/10 for x in range(10)]

for i in numbers:

    y_train_pred_final[i]= y_train_pred_final.Conversion_Prob.map(lambda x: 1 if x > i else 0)

y_train_pred_final.head(10)
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

plt.vlines(x=0.34, ymax=1, ymin=0, colors="r", linestyles="--")

plt.show()
y_train_pred_final['final_predicted'] = y_train_pred_final.Conversion_Prob.map( lambda x: 1 if x > 0.34 else 0)

y_train_pred_final.head()
# Let's check the overall accuracy.

metrics.accuracy_score(y_train_pred_final.Converted, y_train_pred_final.final_predicted)
# Confusion matrix

confusion2 = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final.final_predicted)

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
confusion2[1,1]/(confusion2[0,1]+confusion2[1,1])
confusion2[1,1]/(confusion2[1,0]+confusion2[1,1])
from sklearn.metrics import precision_score, recall_score
precision_score(y_train_pred_final.Converted, y_train_pred_final.Predicted)
recall_score(y_train_pred_final.Converted, y_train_pred_final.Predicted)
from sklearn.metrics import precision_recall_curve
p, r, thresholds = precision_recall_curve(y_train_pred_final.Converted, y_train_pred_final.Conversion_Prob)
plt.plot(thresholds, p[:-1], "g-")

plt.plot(thresholds, r[:-1], "r-")

plt.show()
from sklearn.metrics import classification_report
print(classification_report(y_train_pred_final.Converted, y_train_pred_final.final_predicted))
X_test[['TotalVisits','Total Time Spent on Website','Page Views Per Visit']] = scaler.transform(X_test[['TotalVisits','Total Time Spent on Website','Page Views Per Visit']])

X_test.head()
X_test = X_test[col]

X_test.head()
# adding constant for statsmodel

X_test_sm = sm.add_constant(X_test)
# making prediction on the test set

y_test_pred = res.predict(X_test_sm)
y_test_pred[:10]
# Converting y_pred to a dataframe which is an array

y_pred = pd.DataFrame(y_test_pred)
y_pred.head()
# Converting y_test to dataframe

y_test_df = pd.DataFrame(y_test)
y_test_df.head()
# Putting LeadID to index

y_test_df['LeadID'] = y_test_df.index

y_test_df.head()
# concatenating both the prediction and the orginal labels

y_pred_final = pd.concat([y_test_df, y_pred],axis=1)
y_pred_final.head()
# Renaming the column 

y_pred_final= y_pred_final.rename(columns={ 0 : 'Conversion_Prob'})
# Rearranging the columns

y_pred_final = y_pred_final[['LeadID','Converted','Conversion_Prob']]
y_pred_final.head()
y_pred_final['Predicted'] = y_pred_final.Conversion_Prob.map(lambda x: 1 if x > 0.34 else 0)
y_pred_final.head()
# Let's check the overall accuracy.

accuracy_score=metrics.accuracy_score(y_pred_final.Converted, y_pred_final.Predicted)

accuracy_score
confusion_test_set = metrics.confusion_matrix(y_pred_final.Converted, y_pred_final.Predicted)

print(confusion_test_set)
TP = confusion_test_set[1,1] # true positive 

TN = confusion_test_set[0,0] # true negatives

FP = confusion_test_set[0,1] # false positives

FN = confusion_test_set[1,0] # false negatives
# Let's see the sensitivity of our logistic regression model

TP / float(TP+FN)
# Let us calculate specificity

TN / float(TN+FP)
# Calculate false postive rate - predicting converion when customer does not have converted

print(FP/ float(TN+FP))
# Positive predictive value 

print (TP / float(TP+FP))
# Negative predictive value

print (TN / float(TN+ FN))
#precision

confusion_test_set[1,1]/(confusion_test_set[0,1]+confusion_test_set[1,1])
#recall

confusion_test_set[1,1]/(confusion_test_set[1,0]+confusion_test_set[1,1])
from sklearn.metrics import classification_report
print(classification_report(y_pred_final.Converted, y_pred_final.Predicted))
from sklearn.metrics import precision_recall_curve
p, r, thresholds = precision_recall_curve(y_pred_final.Converted, y_pred_final.Conversion_Prob)
plt.plot(thresholds, p[:-1], "g-")

plt.plot(thresholds, r[:-1], "r-")

plt.show()
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

    plt.title('Receiver operating characteristic (ROC) curve')

    plt.legend(loc="lower right")

    plt.show()



    return fpr,tpr, thresholds

fpr, tpr, thresholds = metrics.roc_curve(y_pred_final.Converted, y_pred_final.Conversion_Prob, drop_intermediate = False)
draw_roc(y_pred_final.Converted, y_pred_final.Conversion_Prob)
y_pred_final.head()
y_pred_final['Lead Score'] = y_pred_final['Conversion_Prob']*100

y_pred_final.head()
y_pred_final = pd.merge(leads[['Lead Number']], y_pred_final,how='inner',left_index=True, right_index=True)
y_pred_final.head()  # test dataset with all the Lead Score values
y_train_pred_df = y_train_pred_final[['Converted', 'Conversion_Prob', 'LeadID','Predicted']]

y_train_pred_df.head()
y_train_pred_df = pd.merge(leads[['Lead Number']], y_train_pred_df,how='inner',left_index=True, right_index=True)

y_train_pred_df.head()
y_train_pred_df['Lead Score'] = y_train_pred_df['Conversion_Prob']*100
y_train_pred_df.head()     # train dataset with all the Lead Score values
final_df_lead_score = pd.concat([y_train_pred_df,y_pred_final],axis=0)

final_df_lead_score.head()
final_df_lead_score = final_df_lead_score.set_index('LeadID')



final_df_lead_score = final_df_lead_score[['Lead Number','Converted','Conversion_Prob','Predicted','Lead Score']]
final_df_lead_score.head()  # final dataframe with all the Lead Scores
final_df_lead_score.shape
# coefficients of our final model 



pd.options.display.float_format = '{:.2f}'.format

new_params = res.params[1:]

new_params
# Getting a relative coeffient value for all the features wrt the feature with the highest coefficient



feature_importance = new_params

feature_importance = 100.0 * (feature_importance / feature_importance.max())

feature_importance
# Sorting the feature variables based on their relative coefficient values



sorted_idx = np.argsort(feature_importance,kind='quicksort',order='list of str')
feature_importance_df = pd.DataFrame(feature_importance).reset_index().sort_values(by=0,ascending=False)

feature_importance_df = feature_importance_df.rename(columns={'index':'Variables', 0:'Relative coeffient value'})

feature_importance_df = feature_importance_df.reset_index(drop=True)

feature_importance_df.head(3)