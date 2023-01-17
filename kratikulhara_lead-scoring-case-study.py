# Suppressing Warnings

import warnings

warnings.filterwarnings('ignore')
# Importing Pandas and NumPy

import pandas as pd, numpy as np



import matplotlib.pyplot as plt

import seaborn as sns
# Importing lead dataset

lead_data = pd.read_csv("../input/leads-data/Leads.csv")

lead_data.head()
# checking the shape of the data 

lead_data.shape
# checking non null count and datatype of the variables

lead_data.info()
# Describing data

lead_data.describe()
# Converting 'Select' values to NaN.

lead_data = lead_data.replace('Select', np.nan)
# checking the columns for null values

lead_data.isnull().sum()
# Finding the null percentages across columns

round(lead_data.isnull().sum()/len(lead_data.index),2)*100
# dropping the columns with missing values greater than or equal to 40% .

lead_data=lead_data.drop(columns=['How did you hear about X Education','Lead Quality','Lead Profile',

                                  'Asymmetrique Activity Index','Asymmetrique Profile Index','Asymmetrique Activity Score',

                                 'Asymmetrique Profile Score'])
# Finding the null percentages across columns after removing the above columns

round(lead_data.isnull().sum()/len(lead_data.index),2)*100
plt.figure(figsize=(17,5))

sns.countplot(lead_data['Specialization'])

plt.xticks(rotation=90)
# Creating a separate category called 'Others' for this 

lead_data['Specialization'] = lead_data['Specialization'].replace(np.nan, 'Others')
# Visualizing Tags column

plt.figure(figsize=(10,7))

sns.countplot(lead_data['Tags'])

plt.xticks(rotation=90)
# Imputing the missing data in the tags column with 'Will revert after reading the email'

lead_data['Tags']=lead_data['Tags'].replace(np.nan,'Will revert after reading the email')
# Visualizing this column

sns.countplot(lead_data['What matters most to you in choosing a course'])

plt.xticks(rotation=45)
# Finding the percentage of the different categories of this column:

round(lead_data['What matters most to you in choosing a course'].value_counts(normalize=True),2)*100
# Dropping this column 

lead_data=lead_data.drop('What matters most to you in choosing a course',axis=1)
sns.countplot(lead_data['What is your current occupation'])

plt.xticks(rotation=45)
# Finding the percentage of the different categories of this column:

round(lead_data['What is your current occupation'].value_counts(normalize=True),2)*100
# Imputing the missing data in the 'What is your current occupation' column with 'Unemployed'

lead_data['What is your current occupation']=lead_data['What is your current occupation'].replace(np.nan,'Unemployed')
plt.figure(figsize=(17,5))

sns.countplot(lead_data['Country'])

plt.xticks(rotation=90)
# Imputing the missing data in the 'Country' column with 'India'

lead_data['Country']=lead_data['Country'].replace(np.nan,'India')
plt.figure(figsize=(10,5))

sns.countplot(lead_data['City'])

plt.xticks(rotation=90)
# Finding the percentage of the different categories of this column:

round(lead_data['City'].value_counts(normalize=True),2)*100
# Imputing the missing data in the 'City' column with 'Mumbai'

lead_data['City']=lead_data['City'].replace(np.nan,'Mumbai')
# Finding the null percentages across columns after removing the above columns

round(lead_data.isnull().sum()/len(lead_data.index),2)*100
# Dropping the rows with null values

lead_data.dropna(inplace = True)
# Finding the null percentages across columns after removing the above columns

round(lead_data.isnull().sum()/len(lead_data.index),2)*100
# Percentage of rows retained 

(len(lead_data.index)/9240)*100
lead_data[lead_data.duplicated()]
Converted = (sum(lead_data['Converted'])/len(lead_data['Converted'].index))*100

Converted
plt.figure(figsize=(10,5))

sns.countplot(x = "Lead Origin", hue = "Converted", data = lead_data,palette='Set1')

plt.xticks(rotation = 45)
plt.figure(figsize=(13,5))

sns.countplot(x = "Lead Source", hue = "Converted", data = lead_data, palette='Set1')

plt.xticks(rotation = 90)
# Need to replace 'google' with 'Google'

lead_data['Lead Source'] = lead_data['Lead Source'].replace(['google'], 'Google')
# Creating a new category 'Others' for some of the Lead Sources which do not have much values.

lead_data['Lead Source'] = lead_data['Lead Source'].replace(['Click2call', 'Live Chat', 'NC_EDM', 'Pay per Click Ads', 'Press_Release',

  'Social Media', 'WeLearn', 'bing', 'blog', 'testone', 'welearnblog_Home', 'youtubechannel'], 'Others')

# Visualizing again

plt.figure(figsize=(10,5))

sns.countplot(x = "Lead Source", hue = "Converted", data = lead_data,palette='Set1')

plt.xticks(rotation = 90)
sns.countplot(x = "Do Not Email", hue = "Converted", data = lead_data,palette='Set1')

plt.xticks(rotation = 90)
sns.countplot(x = "Do Not Call", hue = "Converted", data = lead_data,palette='Set1')

plt.xticks(rotation = 90)
lead_data['TotalVisits'].describe(percentiles=[0.05,.25, .5, .75, .90, .95, .99])
sns.boxplot(lead_data['TotalVisits'],orient='vert',palette='Set1')
percentiles = lead_data['TotalVisits'].quantile([0.05,0.95]).values

lead_data['TotalVisits'][lead_data['TotalVisits'] <= percentiles[0]] = percentiles[0]

lead_data['TotalVisits'][lead_data['TotalVisits'] >= percentiles[1]] = percentiles[1]
# Visualizing again

sns.boxplot(lead_data['TotalVisits'],orient='vert',palette='Set1')
sns.boxplot(y = 'TotalVisits', x = 'Converted', data = lead_data,palette='Set1')
lead_data['Total Time Spent on Website'].describe()
sns.boxplot(lead_data['Total Time Spent on Website'],orient='vert',palette='Set1')
sns.boxplot(y = 'Total Time Spent on Website', x = 'Converted', data = lead_data,palette='Set1')
lead_data['Page Views Per Visit'].describe()
sns.boxplot(lead_data['Page Views Per Visit'],orient='vert',palette='Set1')
percentiles = lead_data['Page Views Per Visit'].quantile([0.05,0.95]).values

lead_data['Page Views Per Visit'][lead_data['Page Views Per Visit'] <= percentiles[0]] = percentiles[0]

lead_data['Page Views Per Visit'][lead_data['Page Views Per Visit'] >= percentiles[1]] = percentiles[1]

# Visualizing again

sns.boxplot(lead_data['Page Views Per Visit'],palette='Set1',orient='vert')
sns.boxplot(y = 'Page Views Per Visit', x = 'Converted', data =lead_data,palette='Set1')
lead_data['Last Activity'].describe()
plt.figure(figsize=(15,6))

sns.countplot(x = "Last Activity", hue = "Converted", data = lead_data,palette='Set1')

plt.xticks(rotation = 90)
# We can club the last activities to "Other_Activity" which are having less data.

lead_data['Last Activity'] = lead_data['Last Activity'].replace(['Had a Phone Conversation', 'View in browser link Clicked', 

                                                       'Visited Booth in Tradeshow', 'Approached upfront',

                                                       'Resubscribed to emails','Email Received', 'Email Marked Spam'], 'Other_Activity')
# Visualizing again

plt.figure(figsize=(15,6))

sns.countplot(x = "Last Activity", hue = "Converted", data = lead_data,palette='Set1')

plt.xticks(rotation = 90)
plt.figure(figsize=(15,6))

sns.countplot(x = "Country", hue = "Converted", data = lead_data,palette='Set1')

plt.xticks(rotation = 90)
plt.figure(figsize=(15,6))

sns.countplot(x = "Specialization", hue = "Converted", data = lead_data,palette='Set1')

plt.xticks(rotation = 90)
plt.figure(figsize=(15,6))

sns.countplot(x = "What is your current occupation", hue = "Converted", data = lead_data,palette='Set1')

plt.xticks(rotation = 90)
sns.countplot(x = "Search", hue = "Converted", data = lead_data,palette='Set1')

plt.xticks(rotation = 90)
sns.countplot(x = "Magazine", hue = "Converted", data = lead_data,palette='Set1')

plt.xticks(rotation = 90)
sns.countplot(x = "Newspaper Article", hue = "Converted", data = lead_data,palette='Set1')

plt.xticks(rotation = 90)
sns.countplot(x = "X Education Forums", hue = "Converted", data = lead_data,palette='Set1')

plt.xticks(rotation = 90)
sns.countplot(x = "Newspaper", hue = "Converted", data = lead_data,palette='Set1')

plt.xticks(rotation = 90)
sns.countplot(x = "Digital Advertisement", hue = "Converted", data = lead_data,palette='Set1')

plt.xticks(rotation = 90)
sns.countplot(x = "Through Recommendations", hue = "Converted", data = lead_data,palette='Set1')

plt.xticks(rotation = 90)
sns.countplot(x = "Receive More Updates About Our Courses", hue = "Converted", data = lead_data,palette='Set1')

plt.xticks(rotation = 90)
plt.figure(figsize=(15,6))

sns.countplot(x = "Tags", hue = "Converted", data = lead_data,palette='Set1')

plt.xticks(rotation = 90)
sns.countplot(x = "Update me on Supply Chain Content", hue = "Converted", data = lead_data,palette='Set1')

plt.xticks(rotation = 90)
sns.countplot(x = "Get updates on DM Content", hue = "Converted", data = lead_data,palette='Set1')

plt.xticks(rotation = 90)
plt.figure(figsize=(15,5))

sns.countplot(x = "City", hue = "Converted", data = lead_data,palette='Set1')

plt.xticks(rotation = 90)
sns.countplot(x = "I agree to pay the amount through cheque", hue = "Converted", data = lead_data,palette='Set1')

plt.xticks(rotation = 90)
sns.countplot(x = "A free copy of Mastering The Interview", hue = "Converted", data = lead_data,palette='Set1')

plt.xticks(rotation = 90)
plt.figure(figsize=(15,5))

sns.countplot(x = "Last Notable Activity", hue = "Converted", data = lead_data,palette='Set1')

plt.xticks(rotation = 90)
lead_data = lead_data.drop(['Lead Number','Tags','Country','Search','Magazine','Newspaper Article','X Education Forums',

                            'Newspaper','Digital Advertisement','Through Recommendations','Receive More Updates About Our Courses',

                            'Update me on Supply Chain Content','Get updates on DM Content','I agree to pay the amount through cheque',

                            'A free copy of Mastering The Interview'],1)
lead_data.shape
lead_data.info()
vars =  ['Do Not Email', 'Do Not Call']



def binary_map(x):

    return x.map({'Yes': 1, "No": 0})



lead_data[vars] = lead_data[vars].apply(binary_map)
# Creating a dummy variable for the categorical variables and dropping the first one.

dummy_data = pd.get_dummies(lead_data[['Lead Origin', 'Lead Source', 'Last Activity', 'Specialization','What is your current occupation',

                             'City','Last Notable Activity']], drop_first=True)

dummy_data.head()
# Concatenating the dummy_data to the lead_data dataframe

lead_data = pd.concat([lead_data, dummy_data], axis=1)

lead_data.head()
lead_data = lead_data.drop(['Lead Origin', 'Lead Source', 'Last Activity', 'Specialization','What is your current occupation',

                             'City','Last Notable Activity'], axis = 1)
lead_data.head()
from sklearn.model_selection import train_test_split



# Putting feature variable to X

X = lead_data.drop(['Prospect ID','Converted'], axis=1)

X.head()
# Putting target variable to y

y = lead_data['Converted']



y.head()
# Splitting the data into train and test

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=100)
from sklearn.preprocessing import StandardScaler



scaler = StandardScaler()



X_train[['TotalVisits','Total Time Spent on Website','Page Views Per Visit']] = scaler.fit_transform(X_train[['TotalVisits','Total Time Spent on Website','Page Views Per Visit']])



X_train.head()
# Checking the Lead Conversion rate

Converted = (sum(lead_data['Converted'])/len(lead_data['Converted'].index))*100

Converted
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()



from sklearn.feature_selection import RFE

rfe = RFE(logreg, 20)             # running RFE with 20 variables as output

rfe = rfe.fit(X_train, y_train)
rfe.support_
list(zip(X_train.columns, rfe.support_, rfe.ranking_))
# Viewing columns selected by RFE

cols = X_train.columns[rfe.support_]

cols
import statsmodels.api as sm
X_train_sm = sm.add_constant(X_train[cols])

logm1 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())

result = logm1.fit()

result.summary()
# Dropping the column 'What is your current occupation_Housewife'

col1 = cols.drop('What is your current occupation_Housewife')
X_train_sm = sm.add_constant(X_train[col1])

logm2 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())

res = logm2.fit()

res.summary()
col1 = col1.drop('Last Notable Activity_Had a Phone Conversation')
X_train_sm = sm.add_constant(X_train[col1])

logm3 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())

res = logm3.fit()

res.summary()
col1 = col1.drop('What is your current occupation_Student')
X_train_sm = sm.add_constant(X_train[col1])

logm4 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())

res = logm4.fit()

res.summary()


col1 = col1.drop('Lead Origin_Lead Add Form')
X_train_sm = sm.add_constant(X_train[col1])

logm5 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())

res = logm5.fit()

res.summary()
# Check for the VIF values of the feature variables. 

from statsmodels.stats.outliers_influence import variance_inflation_factor



# Create a dataframe that will contain the names of all the feature variables and their respective VIFs

vif = pd.DataFrame()

vif['Features'] = X_train[col1].columns

vif['VIF'] = [variance_inflation_factor(X_train[col1].values, i) for i in range(X_train[col1].shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
# Dropping the column  'What is your current occupation_Unemployed' because it has high VIF

col1 = col1.drop('What is your current occupation_Unemployed')
X_train_sm = sm.add_constant(X_train[col1])

logm5 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())

res = logm5.fit()

res.summary()
# Dropping the column  'Lead Origin_Lead Import' because it has high Pvalue

col1 = col1.drop('Lead Origin_Lead Import')
X_train_sm = sm.add_constant(X_train[col1])

logm5 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())

res = logm5.fit()

res.summary()
# Check for the VIF values of the feature variables. 

from statsmodels.stats.outliers_influence import variance_inflation_factor



# Create a dataframe that will contain the names of all the feature variables and their respective VIFs

vif = pd.DataFrame()

vif['Features'] = X_train[col1].columns

vif['VIF'] = [variance_inflation_factor(X_train[col1].values, i) for i in range(X_train[col1].shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
# Dropping the column  'Last Activity_Unsubscribed' to reduce the variables

col1 = col1.drop('Last Activity_Unsubscribed')
X_train_sm = sm.add_constant(X_train[col1])

logm5 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())

res = logm5.fit()

res.summary()
# Check for the VIF values of the feature variables. 

from statsmodels.stats.outliers_influence import variance_inflation_factor



# Create a dataframe that will contain the names of all the feature variables and their respective VIFs

vif = pd.DataFrame()

vif['Features'] = X_train[col1].columns

vif['VIF'] = [variance_inflation_factor(X_train[col1].values, i) for i in range(X_train[col1].shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
# Dropping the column  'Last Notable Activity_Unreachable' to reduce the variables

col1 = col1.drop('Last Notable Activity_Unreachable')
X_train_sm = sm.add_constant(X_train[col1])

logm5 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())

res = logm5.fit()

res.summary()
# Check for the VIF values of the feature variables. 

from statsmodels.stats.outliers_influence import variance_inflation_factor



# Create a dataframe that will contain the names of all the feature variables and their respective VIFs

vif = pd.DataFrame()

vif['Features'] = X_train[col1].columns

vif['VIF'] = [variance_inflation_factor(X_train[col1].values, i) for i in range(X_train[col1].shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
# Getting the predicted values on the train set

y_train_pred = res.predict(X_train_sm)

y_train_pred[:10]
# Reshaping into an array

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
# The confusion matrix indicates as below

# Predicted     not_converted    converted

# Actual

# not_converted        3461      444

# converted            719       1727  
# Let's check the overall accuracy.

print('Accuracy :',metrics.accuracy_score(y_train_pred_final.Converted, y_train_pred_final.predicted))
TP = confusion[1,1] # true positive 

TN = confusion[0,0] # true negatives

FP = confusion[0,1] # false positives

FN = confusion[1,0] # false negatives
# Sensitivity of our logistic regression model

print("Sensitivity : ",TP / float(TP+FN))
# Let us calculate specificity

print("Specificity : ",TN / float(TN+FP))
# Calculate false postive rate - predicting converted lead when the lead actually was not converted

print("False Positive Rate :",FP/ float(TN+FP))
# positive predictive value 

print("Positive Predictive Value :",TP / float(TP+FP))
# Negative predictive value

print ("Negative predictive value :",TN / float(TN+ FN))
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
y_train_pred_final['final_predicted'] = y_train_pred_final.Converted_prob.map( lambda x: 1 if x > 0.34 else 0)



y_train_pred_final.head()
y_train_pred_final['Lead_Score'] = y_train_pred_final.Converted_prob.map( lambda x: round(x*100))



y_train_pred_final.head()
# Let's check the overall accuracy.

print("Accuracy :",metrics.accuracy_score(y_train_pred_final.Converted, y_train_pred_final.final_predicted))
# Confusion matrix

confusion2 = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final.final_predicted )

confusion2
TP = confusion2[1,1] # true positive 

TN = confusion2[0,0] # true negatives

FP = confusion2[0,1] # false positives

FN = confusion2[1,0] # false negatives
# Let's see the sensitivity of our logistic regression model

print("Sensitivity : ",TP / float(TP+FN))
# Let us calculate specificity

print("Specificity :",TN / float(TN+FP))
# Calculate false postive rate - predicting converted lead when the lead was actually not have converted

print("False Positive rate : ",FP/ float(TN+FP))
# Positive predictive value 

print("Positive Predictive Value :",TP / float(TP+FP))
# Negative predictive value

print("Negative Predictive Value : ",TN / float(TN+ FN))
#Looking at the confusion matrix again



confusion = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final.predicted )

confusion
# Precision

TP / TP + FP



print("Precision : ",confusion[1,1]/(confusion[0,1]+confusion[1,1]))
# Recall

TP / TP + FN



print("Recall :",confusion[1,1]/(confusion[1,0]+confusion[1,1]))
from sklearn.metrics import precision_score, recall_score
print("Precision :",precision_score(y_train_pred_final.Converted , y_train_pred_final.predicted))
print("Recall :",recall_score(y_train_pred_final.Converted, y_train_pred_final.predicted))
from sklearn.metrics import precision_recall_curve



y_train_pred_final.Converted, y_train_pred_final.predicted
p, r, thresholds = precision_recall_curve(y_train_pred_final.Converted, y_train_pred_final.Converted_prob)
# plotting a trade-off curve between precision and recall

plt.plot(thresholds, p[:-1], "g-")

plt.plot(thresholds, r[:-1], "r-")

plt.show()
X_test[['TotalVisits','Total Time Spent on Website','Page Views Per Visit']] = scaler.transform(X_test[['TotalVisits',

                                                                                                        'Total Time Spent on Website',

                                                                                                        'Page Views Per Visit']])
# Assigning the columns selected by the final model to the X_test 

X_test = X_test[col1]

X_test.head()
# Adding a const

X_test_sm = sm.add_constant(X_test)



# Making predictions on the test set

y_test_pred = res.predict(X_test_sm)

y_test_pred[:10]
# Converting y_test_pred to a dataframe which is an array

y_pred_1 = pd.DataFrame(y_test_pred)
# Let's see the head

y_pred_1.head()
# Converting y_test to dataframe

y_test_df = pd.DataFrame(y_test)
# Putting Prospect ID to index

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

y_pred_final = y_pred_final.reindex(columns=['Prospect ID','Converted','Converted_prob'])
# Let's see the head of y_pred_final

y_pred_final.head()

y_pred_final['final_predicted'] = y_pred_final.Converted_prob.map(lambda x: 1 if x > 0.34 else 0)
y_pred_final.head()
# Let's check the overall accuracy.

print("Accuracy :",metrics.accuracy_score(y_pred_final.Converted, y_pred_final.final_predicted))
# Making the confusion matrix

confusion2 = metrics.confusion_matrix(y_pred_final.Converted, y_pred_final.final_predicted )

confusion2
TP = confusion2[1,1] # true positive 

TN = confusion2[0,0] # true negatives

FP = confusion2[0,1] # false positives

FN = confusion2[1,0] # false negatives
# Let's see the sensitivity of our logistic regression model

print("Sensitivity :",TP / float(TP+FN))
# Let us calculate specificity

print("Specificity :",TN / float(TN+FP))
y_pred_final['Lead_Score'] = y_pred_final.Converted_prob.map( lambda x: round(x*100))



y_pred_final.head()
hot_leads=y_pred_final.loc[y_pred_final["Lead_Score"]>=85]

hot_leads
print("The Prospect ID of the customers which should be contacted are :")



hot_leads_ids = hot_leads["Prospect ID"].values.reshape(-1)

hot_leads_ids
res.params.sort_values(ascending=False)