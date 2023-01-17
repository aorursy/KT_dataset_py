#importing libraries for numpy and dataframe

import pandas as pd

import numpy as np



#importing libraries for data visualization

import matplotlib.pyplot as plt

from matplotlib.pyplot import xticks

import seaborn as sns

%matplotlib inline



#importing library for data scaling

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import scale



#importing library to suppress warnings

import warnings

warnings.filterwarnings('ignore')



#importing libraries for Logistic Regression

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.feature_selection import RFE

import statsmodels.api as sm



#Checking VIF values for the feature variables

from statsmodels.stats.outliers_influence import variance_inflation_factor



#Creation of confusion matrix

from sklearn import metrics
#Reading the file from the local

leads = pd.read_csv("../input/Leads.csv")
#Checking the first 5 records of the dataframe

leads.head()
#To determine the number of rows and columns present in the dataset.

leads.shape
#to ensure complete rows and columns are shown for a dataframe

pd.set_option('display.max_columns',150)

pd.set_option('display.max_rows',9500)
#to check if all the columns are visible in the dataframe

leads.head()
#Checking the statistical values of the numerical columns of the dataset

leads.describe()
#Checking the datatype of the dataset

leads.info()
#To check if there exists any duplicate records in the dataframe specially for Prospect ID

print('The number of duplicate records in the column , Prospect ID are :',leads['Prospect ID'].duplicated().sum())

print('The number of duplicate records in the dataset are :',leads.duplicated().sum())
#To check the percentage of missing values in each column

total = pd.DataFrame(leads.isna().sum().sort_values(ascending=False),columns=['Total Missing'])

percentage = pd.DataFrame(round(100*(leads.isna().sum()/len(leads)),2).sort_values(ascending=False),columns=['Percentage'])

pd.concat([total,percentage],axis = 1)
#Converting Select to NaN values

leads = leads.replace('Select',np.nan)
#Checking the missing values and their respective percentage values again after replacing Select as NaN.

total = pd.DataFrame(leads.isna().sum().sort_values(ascending=False),columns=['Total Missing'])

percentage = pd.DataFrame(round(100*(leads.isna().sum()/len(leads)),2).sort_values(ascending=False),columns=['Percentage'])

pd.concat([total,percentage],axis = 1)
#Dropping 'How did you hear about X Education' & 'Lead Profile' from the dataframe leads

leads = leads.drop(['How did you hear about X Education','Lead Profile'],axis=1)
leads.head()
#Analysing Lead Quality

sns.countplot(leads['Lead Quality'])
#Replacing nan value with Unknown

leads['Lead Quality'] = leads['Lead Quality'].replace(np.nan,'Not Sure')
#Plotting and reanalyzing Lead Quality

plt.figure(figsize=(10,5))

sns.countplot(leads['Lead Quality'])

plt.show()
#Analayzing the Asymmetric Score and Index for Profile and Activity

plt.figure(figsize=(8,8))

plt.subplot(2,2,1)

sns.countplot(leads['Asymmetrique Activity Index'])

plt.subplot(2,2,2)

sns.boxplot(leads['Asymmetrique Activity Score'])

plt.subplot(2,2,3)

sns.countplot(leads['Asymmetrique Profile Index'])

plt.subplot(2,2,4)

sns.boxplot(leads['Asymmetrique Profile Score'])
#Dropping the columns Asymmetrique Activity Index,Asymmetrique Profile Index,Asymmetrique Activity Score,Asymmetrique Profile Score

leads = leads.drop(['Asymmetrique Activity Index','Asymmetrique Profile Index','Asymmetrique Activity Score','Asymmetrique Profile Score'],axis=1)
#Rechecking the null values and percentage values

total = pd.DataFrame(leads.isna().sum().sort_values(ascending=False),columns=['Total Missing'])

percentage = pd.DataFrame(round(100*(leads.isna().sum()/len(leads)),2).sort_values(ascending=False),columns=['Percentage'])

pd.concat([total,percentage],axis = 1)
#Analysing City

plt.figure(figsize=(15,5))

sns.countplot(leads['City'])

plt.show()
#Replacing nan value to Mumbai

leads['City']=leads['City'].replace(np.nan,'Mumbai')
#Analysing the columns again based on the replacing

plt.figure(figsize=(15,5))

sns.countplot(leads['City'])

plt.show()
#Analyzing Specialization

plt.figure(figsize=(10,5))

sns.countplot(leads['Specialization'])

xticks(rotation = 90)

plt.show()
#Replacing nan with Others in Specilization

leads['Specialization'] = leads['Specialization'].replace(np.nan,'Others')
#Plotting countplot again to check the plot of the different values in Specialization

#Analyzing Specialization

plt.figure(figsize=(10,5))

sns.countplot(leads['Specialization'])

xticks(rotation = 90)

plt.show()
#Checking the count and percentage of missing values

total = pd.DataFrame(leads.isna().sum().sort_values(ascending=False),columns=['Total Missing'])

percentage = pd.DataFrame(round(100*(leads.isna().sum()/len(leads)),2).sort_values(ascending=False),columns=['Percentage'])

pd.concat([total,percentage],axis = 1)
#Analyzing Tags

plt.figure(figsize=(10,5))

sns.countplot(leads['Tags'])

xticks(rotation = 90)

plt.show()
#Replacing nan with Will revert after reading the email

leads['Tags'] = leads['Tags'].replace(np.nan,'Will revert after reading the email')
#Analyzing Tags

plt.figure(figsize=(10,5))

sns.countplot(leads['Tags'])

xticks(rotation = 90)

plt.show()
#Analyzing What matters most to you in choosing a course

sns.countplot(leads['What matters most to you in choosing a course'])

xticks(rotation = 90)
#Replacing nan with Better Career Prospects

leads['What matters most to you in choosing a course'] = leads['What matters most to you in choosing a course'].replace(np.nan,'Better Career Prospects')
#Analyzing What matters most to you in choosing a course

sns.countplot(leads['What matters most to you in choosing a course'])

xticks(rotation = 90)
#Analyzing What is your current occupation

sns.countplot(leads['What is your current occupation'])

xticks(rotation = 90)
#Replacing nan with Unemployed

leads['What is your current occupation'] = leads['What is your current occupation'].replace(np.nan,'Unemployed')
#Analyzing What is your current occupation

sns.countplot(leads['What is your current occupation'])

xticks(rotation = 90)
#Analyzing Country

plt.figure(figsize=(10,5))

sns.countplot(leads['Country'])

xticks(rotation = 90)

plt.show()
#Replacing nan values with India

leads['Country'] = leads['Country'].replace(np.nan,'India')
#Analyzing Country

plt.figure(figsize=(10,5))

sns.countplot(leads['Country'])

xticks(rotation = 90)

plt.show()
#Checking the count and percentage of missing values

total = pd.DataFrame(leads.isna().sum().sort_values(ascending=False),columns=['Total Missing'])

percentage = pd.DataFrame(round(100*(leads.isna().sum()/len(leads)),2).sort_values(ascending=False),columns=['Percentage'])

pd.concat([total,percentage],axis = 1)
#Replacing the NA values with mean of the columns

leads.fillna(leads.mean(), inplace=True)
##Checking the count and percentage of missing values

total = pd.DataFrame(leads.isna().sum().sort_values(ascending=False),columns=['Total Missing'])

percentage = pd.DataFrame(round(100*(leads.isna().sum()/len(leads)),2).sort_values(ascending=False),columns=['Percentage'])

pd.concat([total,percentage],axis = 1)
#Analyzing Lead Activity

plt.figure(figsize=(10,5))

sns.countplot(leads['Last Activity'])

xticks(rotation = 90)

plt.show()
#Analyzing Lead Source

plt.figure(figsize=(10,5))

sns.countplot(leads['Lead Source'])

xticks(rotation = 90)

plt.show()
#Dropping NA values

leads.dropna(axis=1,inplace=True)
##Checking the count and percentage of missing values

total = pd.DataFrame(leads.isna().sum().sort_values(ascending=False),columns=['Total Missing'])

percentage = pd.DataFrame(round(100*(leads.isna().sum()/len(leads)),2).sort_values(ascending=False),columns=['Percentage'])

pd.concat([total,percentage],axis = 1)
#Dropping Prospect ID from the dataframe as it has no role to play in the modelling

leads = leads.drop(['Prospect ID'],axis=1)
#Let us check how many columns and rows we have now in the dataframe

leads.shape
#To check which all columns are present, looking into the first 5 records of the dataframe

leads.head()
#To analyze the statistics of the numerical columns

leads.describe()
#let us analyze the columns with respect to Converted field

#Analyzing Lead Number with Converted 

sns.boxplot(x='Converted',y='Lead Number',data=leads)
#Dropping Lead Number

leads = leads.drop(['Lead Number'],axis=1)
#Analyzing Total Visits with Converted 

sns.boxplot(x='Converted',y='TotalVisits',data=leads,showfliers=False)
#Analyzing Country with Converted columns

sns.countplot(leads['Country'],hue=leads['Converted'])

xticks(rotation=90)
#Dropping Country

leads = leads.drop(['Country'],axis=1)
#Analyzing Total time spent on website with Converted

sns.boxplot(y = leads['Total Time Spent on Website'],x = leads['Converted'],showfliers=False)
plt.figure(figsize=(20,20))

plt.subplot(4,3,1)

sns.countplot(leads['What matters most to you in choosing a course'],hue=leads['Converted'])

xticks(rotation=90)

plt.subplot(4,3,2)

sns.countplot(leads['Search'],hue=leads['Converted'])

xticks(rotation=90)

plt.subplot(4,3,3)

sns.countplot(leads['Magazine'],hue=leads['Converted'])

xticks(rotation=90)

plt.subplot(4,3,4)

sns.countplot(leads['Newspaper Article'],hue=leads['Converted'])

xticks(rotation=90)

plt.subplot(4,3,5)

sns.countplot(leads['X Education Forums'],hue=leads['Converted'])

xticks(rotation=90)

plt.subplot(4,3,6)

sns.countplot(leads['Newspaper'],hue=leads['Converted'])

xticks(rotation=90)

plt.subplot(4,3,7)

sns.countplot(leads['Digital Advertisement'],hue=leads['Converted'])

xticks(rotation=90)

plt.subplot(4,3,8)

sns.countplot(leads['Through Recommendations'],hue=leads['Converted'])

xticks(rotation=90)

plt.subplot(4,3,9)

sns.countplot(leads['Receive More Updates About Our Courses'],hue=leads['Converted'])

xticks(rotation=90)

plt.subplot(4,3,10)

sns.countplot(leads['Update me on Supply Chain Content'],hue=leads['Converted'])

xticks(rotation=90)

plt.subplot(4,3,11)

sns.countplot(leads['Get updates on DM Content'],hue=leads['Converted'])

xticks(rotation=90)

plt.subplot(4,3,12)

sns.countplot(leads['I agree to pay the amount through cheque'],hue=leads['Converted'])

xticks(rotation=90)
#Dropping columns from the dataframe

leads = leads.drop(['What matters most to you in choosing a course','Search','Magazine','Newspaper Article',\

                    'X Education Forums','Newspaper','Digital Advertisement',\

                    'Through Recommendations','Receive More Updates About Our Courses','Update me on Supply Chain Content',\

                   'Get updates on DM Content','I agree to pay the amount through cheque'],axis=1)
#To assess the number of rows and columns present in the dataframe

leads.shape
#To check the first 5 records of the dataframe

leads.head()
#Converting Yes/No in the form of 1 & 0

leads['Do Not Email'] = leads['Do Not Email'].map({'Yes':1,'No':0})

leads['Do Not Call'] = leads['Do Not Call'].map({'Yes':1,'No':0})

leads['A free copy of Mastering The Interview'] = leads['A free copy of Mastering The Interview'].map({'Yes':1,'No':0})
#Checking the first 5 rows to see Yes/No converted to 1/0

leads.head()
#Creating dummy variables for the categorical variables

leads_dummy = pd.get_dummies(leads[['Lead Origin','Specialization','What is your current occupation','Tags','Lead Quality','City','Last Notable Activity']])
#Checking the creation of dummy variables for the first 5 entries of the leads_dummy dataframe

leads_dummy.head()
#Concatenating leads with leads_dummy dataframes

leads_final = pd.concat([leads,leads_dummy],axis=1)
#Lets check the dataframe after concatenation

leads_final.head()
#Dropping those columns for which dummy variables were created

leads_final = leads_final.drop(['Lead Origin','Specialization','What is your current occupation','Tags','Lead Quality','City','Last Notable Activity'],axis=1)
#Checking to see there are only numerical columns present in the dataframe

leads_final.head()
#Splitting dataframe into feature variable and response variables

leads_final_Y=leads['Converted']

leads_final_X=leads_final.drop(['Converted'],axis=1)
#Checking both the dataframes

leads_final_X.head()
leads_final_Y.head()
#Number of rows present in leads_final_X dataframe

leads_final_X.shape
#Splitting dataset into train & test model

X_train , X_test , y_train , y_test = train_test_split(leads_final_X,leads_final_Y,train_size=0.7,random_state=100)
#Scaling the train dataset to normalize the values

scale = StandardScaler()

X_train[['TotalVisits','Total Time Spent on Website','Page Views Per Visit']]= scale.fit_transform(X_train[['TotalVisits','Total Time Spent on Website','Page Views Per Visit']])
#Checking if the X_train data is normalized

X_train.head()
#Creating Logistic Regression Model

leads_Model1= sm.GLM(y_train,(sm.add_constant(X_train)),family = sm.families.Binomial())

leads_Model1.fit().summary()
logreg = LogisticRegression()
#Building model with Logistic Regression

#Selecting 15 most relevant features

from sklearn.feature_selection import RFE

rfe = RFE(logreg,15)

rfe = rfe.fit(X_train,y_train)
rfe.support_
list(zip(X_train.columns,rfe.support_,rfe.ranking_))
col = X_train.columns[rfe.support_]
X_train.columns[~rfe.support_]
#Stats Model for RFE selected model 

X_train_sm = sm.add_constant(X_train[col])

logm2 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())

res = logm2.fit()

res.summary()
col = col.drop('Tags_number not provided',1)
#Stats Model after dropping Tags_number not provided

X_train_sm = sm.add_constant(X_train[col])

logm2 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())

res = logm2.fit()

res.summary()
#Dropping Tags_wrong number given columns

col = col.drop('Tags_wrong number given',1)
##Stats Model after dropping Tags_wrong number given

X_train_sm = sm.add_constant(X_train[col])

logm3 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())

res = logm3.fit()

res.summary()
#Dropping Tags_Not doing further education from the col

col = col.drop('Tags_Already a student',1)
#Stats Model after dropping Tags_Not doing further education

X_train_sm = sm.add_constant(X_train[col])

logm4 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())

res = logm4.fit()

res.summary()
#Dropping Tags_invalid number from the col

col = col.drop('Tags_invalid number',1)
#Stats Model after dropping Tags_invalid number

X_train_sm = sm.add_constant(X_train[col])

logm5 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())

res = logm5.fit()

res.summary()
#Creating a dataframe that will contain VIF values of all the features

vif = pd.DataFrame()

vif['Features'] = X_train[col].columns

vif['VIF']=[variance_inflation_factor(X_train[col].values, i) for i in range(X_train[col].shape[1])]

vif['VIF']=round(vif['VIF'],2)

vif = vif.sort_values(by='VIF',ascending = False)

vif
#Getting predicted values on the train set

y_train_pred = res.predict(X_train_sm)

y_train_pred[:10]
y_train_pred = y_train_pred.values.reshape(-1)

y_train_pred[:10]
#Creating a dataframe with the actual converted lead

y_train_pred_final = pd.DataFrame({'Converted':y_train.values,'Predicted_Conversion_Probability':y_train_pred})

y_train_pred_final['Prospect ID'] = y_train.index

y_train_pred_final.head(10)
#Creating a new column Final_Prediction and its value with 1 if Predicted_Conversion_Probability is greater or equal to 0.5 else 0

y_train_pred_final['Prediction'] = y_train_pred_final.Predicted_Conversion_Probability.map(lambda x: 1 if x>0.5 else 0)

y_train_pred_final.head(10)
#Creating a confusion matrix to check the correctness of prediction

ConfusionMatrix = metrics.confusion_matrix(y_train_pred_final.Converted,y_train_pred_final.Prediction)

print(ConfusionMatrix)
#Calculating the overall accuracy

print(metrics.accuracy_score(y_train_pred_final.Converted,y_train_pred_final.Prediction))
TP = ConfusionMatrix[1,1] # true positive 

TN = ConfusionMatrix[0,0] # true negatives

FP = ConfusionMatrix[0,1] # false positives

FN = ConfusionMatrix[1,0] # false negatives
# Let's see the sensitivity of our logistic regression model

TP / float(TP+FN)
# Let us calculate specificity

TN / float(TN+FP)
# Calculate false predictive rate - predicting converted when lead is not converted

print(FP/ float(TN+FP))
# Calculate postive predictive rate - predicting converted when lead is actually converted

print (TP / float(TP+FP))
# Calculate negative predictive rate - predicting not converted when lead is actually not converted 

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
fpr, tpr, thresholds = metrics.roc_curve(y_train_pred_final.Converted,y_train_pred_final.Prediction, drop_intermediate = False )
#Plotting the ROC Curve

draw_roc(y_train_pred_final.Converted,y_train_pred_final.Prediction)
# Let's create columns with different probability cutoffs 

numbers = [float(x)/10 for x in range(10)]

for i in numbers:

    y_train_pred_final[i]= y_train_pred_final.Predicted_Conversion_Probability.map(lambda x: 1 if x > i else 0)

y_train_pred_final.head()
# Now let's calculate accuracy sensitivity and specificity for various probability cutoffs.

cutoff_df = pd.DataFrame( columns = ['Probability','Accuracy','Sensitivity','Specificity'])

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

cutoff_df.plot.line(x='Probability', y=['Accuracy','Sensitivity','Specificity'])

plt.show()
#Creating a new column Final_Prediction and its value with 1 if Predicted_Conversion_Probability is greater or equal to 0.2 else 0

y_train_pred_final['Final_Prediction'] = y_train_pred_final.Predicted_Conversion_Probability.map(lambda x: 1 if x > 0.2 else 0)

y_train_pred_final.head(10)
# Let's check the overall accuracy.

metrics.accuracy_score(y_train_pred_final.Converted, y_train_pred_final.Final_Prediction)
confusion2 = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final.Final_Prediction)

confusion2
TP = confusion2[1,1] # true positive 

TN = confusion2[0,0] # true negatives

FP = confusion2[0,1] # false positives

FN = confusion2[1,0] # false negatives
# Let's see the sensitivity of our logistic regression model

TP / float(TP+FN)
# Let us calculate specificity

TN / float(TN+FP)
# Calculate false postive rate

print(FP/ float(TN+FP))
# Positive predictive value 

print (TP / float(TP+FP))
# Negative predictive value

print (TN / float(TN+ FN))
from sklearn.metrics import precision_recall_curve
y_train_pred_final.Converted, y_train_pred_final.Prediction
p, r, thresholds = precision_recall_curve(y_train_pred_final.Converted, y_train_pred_final.Predicted_Conversion_Probability)
#Plotting the precision recall curve

plt.plot(thresholds, p[:-1], "g-")

plt.plot(thresholds, r[:-1], "r-")

plt.show()
#Scaling the test dataset to normalize the values

scale = StandardScaler()

X_test[['TotalVisits','Total Time Spent on Website','Page Views Per Visit']]= scale.fit_transform(X_test[['TotalVisits','Total Time Spent on Website','Page Views Per Visit']])
X_test = X_test[col]

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
y_test_df['Prospect ID'] = y_test_df.index
# Removing index for both dataframes to append them side by side 

y_pred_1.reset_index(drop=True, inplace=True)

y_test_df.reset_index(drop=True, inplace=True)
# Appending y_test_df and y_pred_1

y_pred_final = pd.concat([y_test_df, y_pred_1],axis=1)
y_pred_final.head()
# Renaming the column 

y_pred_final= y_pred_final.rename(columns={ 0 : 'Prediction Probability'})
# Let's see the head of y_pred_final

y_pred_final.head()
y_pred_final['Final Prediction'] = y_pred_final['Prediction Probability'].map(lambda x:1 if x>0.25 else 0)
y_pred_final['Lead Score'] = round(y_pred_final['Prediction Probability']*100,2)
y_pred_final.head()
# Let's check the overall accuracy.

metrics.accuracy_score(y_pred_final.Converted, y_pred_final['Final Prediction'])
confusion3 = metrics.confusion_matrix(y_pred_final.Converted, y_pred_final['Final Prediction'])

confusion3 = pd.DataFrame(confusion3)

confusion3.columns = ['Predicted Non Converted','Predicted Converted']

confusion3.index=['Actual Non Converted','Actual Converted']

confusion3
def scores(actuals,predictions):

    confusion=metrics.confusion_matrix(actuals,predictions)

    TP = confusion[1,1] # true positives 

    TN = confusion[0,0] # true negatives

    FP = confusion[0,1] #False Positives

    FN = confusion[1,0] #False negatives

    accuracy_score = metrics.accuracy_score(actuals,predictions)

    sensitivity= TP / float(TP+FN)

    specificity= TN/ float(TN+FP)

    precision_score = metrics.precision_score(actuals,predictions)

    recall_score = metrics.recall_score(actuals,predictions)

    final_scores=pd.Series({'Accuracy':accuracy_score,'Sensitivity':sensitivity,'Specificity':specificity,'Precision':precision_score,'Recall':recall_score})

    return(final_scores)
test_scores= scores(y_pred_final['Converted'], y_pred_final['Final Prediction'])

test_scores