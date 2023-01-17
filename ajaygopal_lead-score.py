#importing all the necessary libraries





import pandas as pd

import numpy as np



# For Visualisation

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



# To ignore warnings:

import warnings

warnings.filterwarnings('ignore')



#import pandas_profiling package for a quick overview of the dataset (Please install this package)

import pandas_profiling as pp



# To Scale our data

from sklearn.preprocessing import scale





# To display all the rows and columns:

pd.set_option('display.max_rows', 500)

pd.set_option('display.max_columns', 500)

pd.set_option('display.width', 1000)
#importing the data



lead_df=pd.read_csv('../input/leads-dataset/Leads.csv')
lead_df.head()
lead_df.describe()
lead_df.info()
lead_df.shape

#A detailed report of the leads dataset:



#pp.ProfileReport(lead_df)
# dropping Asymmetrique Activity Index(45.6%)



lead_df.drop(['Asymmetrique Activity Index'],axis=1,inplace=True)
# dropping Asymmetrique Activity Score(45.6%)



lead_df.drop(['Asymmetrique Activity Score'],axis=1,inplace=True)
# dropping Asymmetrique Profile Index(45.6%)



lead_df.drop(['Asymmetrique Profile Index'],axis=1,inplace=True)
# dropping Asymmetrique Profile Score(45.6%)



lead_df.drop(['Asymmetrique Profile Score'],axis=1,inplace=True)
#checking % of missing values again after dropping :



round(100*(lead_df.isnull().sum()/len(lead_df.index)), 2)
# 1.To see how to impute values in the 'Lead Quality' column:



lead_df['Lead Quality'].describe()
sns.countplot(lead_df['Lead Quality'])
lead_df['Lead Quality'] = lead_df['Lead Quality'].replace(np.nan, 'Not Sure')
# 2.To see how to impute values in the 'Tags' column:



lead_df['Tags'].describe()
plt.figure(num=None, figsize=(40, 40))

sns.countplot(lead_df['Tags'])
# Replacing missing values with 'Will revert after reading the email':



lead_df['Tags'] = lead_df['Tags'].replace(np.nan, 'Will revert after reading the email')
# 3.To see how to impute values in the 'Lead Profile' column:



lead_df['Lead Profile'].describe()
sns.countplot(lead_df['Lead Profile'])
#Replacing all Select values with Null values:



lead_df['Lead Profile'] = lead_df['Lead Profile'].replace('Select',np.nan)
#checking % of missing values in the column Lead Profile:



round(100*(lead_df['Lead Profile'].isnull().sum()/len(lead_df['Lead Profile'].index)), 2)
#Dropping the Lead Profile column:



lead_df.drop(['Lead Profile'],axis=1,inplace=True)
# 4.To see how to impute values in the 'What matters most to you in choosing a course' column:



lead_df['What matters most to you in choosing a course'].describe()
sns.countplot(lead_df['What matters most to you in choosing a course'])
lead_df['What matters most to you in choosing a course'] = lead_df['What matters most to you in choosing a course'].replace(np.nan,'Better Career Prospects')
# 5.To see how to impute values in the 'What is your current occupation' column:



lead_df['What is your current occupation'].describe()
sns.countplot(lead_df['What is your current occupation'])
lead_df['What is your current occupation'] = lead_df['What is your current occupation'].replace(np.nan,'Unemployed')
#checking % of missing values again:



round(100*(lead_df.isnull().sum()/len(lead_df.index)), 2)
# 6.To see how to impute values in the 'Country' column:



lead_df['Country'].describe()
plt.figure(num=None, figsize=(40, 40))

sns.countplot(lead_df['Country'])
lead_df['Country'] = lead_df['Country'].replace(np.nan,'India')
# 7.To see how to impute values in the 'How did you hear about X Education' column:



lead_df['How did you hear about X Education'].describe()
sns.countplot(lead_df['How did you hear about X Education'])
#Replacing Select with NaN:



lead_df['How did you hear about X Education'] = lead_df['How did you hear about X Education'].replace('Select',np.nan)
#Finding the % of missing values now:



round(100*(lead_df['How did you hear about X Education'].isnull().sum()/len(lead_df['How did you hear about X Education'].index)), 2)
# Dropping the How did you hear about X Education column.



lead_df.drop(['How did you hear about X Education'],axis=1,inplace=True)
# 8.To see how to impute values in the 'Specialization' column:



lead_df['Specialization'].describe()
# Replacing Select with NaN:



lead_df['Specialization'] = lead_df['Specialization'].replace('Select',np.nan)
#Finding the % of missing values now:



round(100*(lead_df['Specialization'].isnull().sum()/len(lead_df['Specialization'].index)), 2)
plt.figure(figsize=(40, 40))

sns.countplot(lead_df['Specialization'])
lead_df['Specialization'] = lead_df['Specialization'].replace(np.nan, 'Others')
# 9.To see how to impute values in the 'City' column:



lead_df['Specialization'].describe()
plt.figure(figsize=(30, 30))

sns.countplot(lead_df['City'])
#Replacing Select with NaN:



lead_df['City'] = lead_df['City'].replace('Select',np.nan)
#Finding the % of missing values now:



round(100*(lead_df['City'].isnull().sum()/len(lead_df['City'].index)), 2)
lead_df['City'] = lead_df['City'].replace(np.nan,'Mumbai')
# 10.To see how to impute values in the 'Page Views Per Visit' column:



lead_df['Page Views Per Visit'].describe()
#Replacing the missing values with the mean of the column

lead_df['Page Views Per Visit'] = lead_df['Page Views Per Visit'].replace(np.nan,np.mean)
# 11.To see how to impute values in the 'TotalVisits' column:



lead_df['TotalVisits'].describe()
#Replacing the missing values with the mean of the column

lead_df['TotalVisits'] = lead_df['TotalVisits'].replace(np.nan,np.mean)
# 12.To see how to impute values in the 'Last Activity' column:



lead_df['Last Activity'].describe()
plt.figure(figsize=(30, 30))

sns.countplot(lead_df['Last Activity'])
#Replacing the missing values

lead_df['Last Activity'] = lead_df['Last Activity'].replace(np.nan,'Email Marked Spam')
# 13.To see how to impute values in the 'Lead Source' column:



lead_df['Lead Source'].describe()
plt.figure(figsize=(30, 30))

sns.countplot(lead_df['Lead Source'])
#Replacing the missing values

lead_df['Lead Source'] = lead_df['Lead Source'].replace(np.nan,'Google')
#To check how many columns are present now:



len(lead_df.columns)
#checking % of missing values again:



round(100*(lead_df.isnull().sum()/len(lead_df.index)), 2)

lead_df.columns.values
lead_df
# Checking the Converted Rate :



conv_rate= (sum(lead_df['Converted'])/len(lead_df['Converted'].index))*100

conv_rate
#1.Column- 'Lead Origin'



sns.countplot(x = "Lead Origin", hue = "Converted", data = lead_df)
#2.Column- 'Do Not Email'



sns.countplot(x = "Do Not Email", hue = "Converted", data = lead_df)
#3.Column- 'Do Not Call'



sns.countplot(x = "Do Not Call", hue = "Converted", data = lead_df)
#4.Column- 'Total Time Spent on Website'. Plotting a boxplot because it is a numerical variable.



#Checking outliers first:



sns.boxplot(lead_df['Total Time Spent on Website'])
#Plotting the boxplot:



sns.boxplot(y = "Total Time Spent on Website", x = "Converted", data = lead_df)
#5.Column- 'Country'



plt.figure(figsize=(30, 30))

sns.countplot(x = "Country", hue = "Converted", data = lead_df)

#6.Column- 'Specialization'



plt.figure(figsize=(60,20))

sns.countplot(x = "Specialization", hue = "Converted", data = lead_df)
#7. Column- 'What is your current occupation'



sns.countplot(x= 'What is your current occupation', hue='Converted',data=lead_df)
#8. Column-'What matters most to you in choosing a course'



sns.countplot(x='What matters most to you in choosing a course',hue='Converted',data=lead_df)
#9 column: 'Search'

#10 column: 'Magazine'

#11 column: 'Newspaper Article'

#12 column: 'X Education Forums'

#13 column: 'Newspaper'

#14 column: 'Digital Advertising'



plt.figure(figsize = (20,15))



plt.subplot(3, 3, 1)

plt1=sns.countplot(x='Search',hue='Converted',data=lead_df)



plt.subplot(3,3,2)

plt2=sns.countplot(x='Magazine',hue='Converted',data=lead_df)



plt.subplot(3,3,3)

plt3=sns.countplot(x='Newspaper Article',hue='Converted',data=lead_df)



plt.subplot(3,3,4)

plt4=sns.countplot(x='X Education Forums',hue='Converted',data=lead_df)



plt.subplot(3,3,5)

plt5=sns.countplot(x='Newspaper',hue='Converted',data=lead_df)



plt.subplot(3,3,6)

plt6=sns.countplot(x='Digital Advertisement',hue='Converted',data=lead_df)



plt.show()
#15 column: 'Through Recommendations'



sns.countplot(x='Through Recommendations', hue='Converted', data=lead_df)
#16 column: 'Receive More Updates About Our Courses'



sns.countplot(x='Receive More Updates About Our Courses',hue='Converted',data=lead_df)




#17 column: 'Tags'





sns.countplot(x='Tags',hue='Converted',data=lead_df)
#18 column: 'Lead Quality'



sns.countplot(x='Lead Quality',hue='Converted',data=lead_df)
#19 column: 'Update me on Supply Chain Content'



sns.countplot(x='Update me on Supply Chain Content',hue='Converted',data=lead_df)
#20 column:'Get updates on DM Content



sns.countplot(x='Get updates on DM Content',hue='Converted',data=lead_df)
#21 column:'City'



sns.countplot(x='City',hue='Converted',data=lead_df)
#22 column:'I agree to pay the amount through cheque'



sns.countplot(x='I agree to pay the amount through cheque',hue='Converted',data=lead_df)
#23 column:'A free copy of Mastering The Interview'



sns.countplot(x='A free copy of Mastering The Interview',hue='Converted',data=lead_df)
#24 column:'Last Notable Activity'



plt.figure(figsize=(60,20))

sns.countplot(x='Last Notable Activity',hue='Converted',data=lead_df)
#25 column:'Lead Source'



plt.figure(figsize=(60,20))

sns.countplot(x='Lead Source',hue='Converted',data=lead_df)
lead_df = lead_df.drop(['What matters most to you in choosing a course','Search','Magazine','Newspaper Article','X Education Forums','Newspaper',

           'Digital Advertisement','Through Recommendations','Page Views Per Visit','TotalVisits','Last Activity','Receive More Updates About Our Courses','Update me on Supply Chain Content',

           'Get updates on DM Content','I agree to pay the amount through cheque','Lead Number','A free copy of Mastering The Interview','Country'],1)
lead_df.columns.values
dummy1 = pd.get_dummies(lead_df[['Lead Origin','Specialization','What is your current occupation',

                              'Tags','Lead Quality','City','Lead Source','Last Notable Activity']], drop_first=True)

dummy1.head()
# Adding the results back to the master dataframe

lead_df = pd.concat([lead_df, dummy1], axis=1)

lead_df.head()
# We have created dummies for the above categorical variables, so now we can drop them:



lead_df=lead_df.drop(['Lead Origin','Specialization','What is your current occupation','Tags','Lead Quality','City','Lead Source','Last Notable Activity'], axis = 1)

lead_df.head()
# List of variables to map



varlist =  ['Do Not Email', 'Do Not Call']



# Defining the map function

def binary_map(x):

    return x.map({'Yes': 1, "No": 0})



# Applying the function:

lead_df[varlist] = lead_df[varlist].apply(binary_map)
lead_df.head()
from sklearn.model_selection import train_test_split



# Putting feature variable to X

X = lead_df.drop(['Prospect ID','Converted'], axis=1)

X.head()
# Putting response variable to y

y = lead_df['Converted']



y.head()
# Splitting the data into train and test:

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=100)
from sklearn.preprocessing import StandardScaler



scaler = StandardScaler()



X_train[['Total Time Spent on Website']] = scaler.fit_transform(X_train[['Total Time Spent on Website']])



X_train.head()
# Plotting the the correlation matrix 

plt.figure(figsize = (40,40))        # Size of the figure

sns.heatmap(lead_df.corr(),annot = True)

plt.show()
import statsmodels.api as sm
# Logistic regression model

logm1 = sm.GLM(y_train,(sm.add_constant(X_train)), family = sm.families.Binomial())

logm1.fit().summary()
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()



#Using RFE to select optimum variables:



from sklearn.feature_selection import RFE

rfe = RFE(logreg, 15)             # running RFE with 15 variables as output

rfe = rfe.fit(X_train, y_train)
rfe.support_
list(zip(X_train.columns, rfe.support_, rfe.ranking_))
col = X_train.columns[rfe.support_]
X_train.columns[~rfe.support_]
X_train_sm = sm.add_constant(X_train[col])

logm2 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())

res = logm2.fit()

res.summary()
col1 = col.drop('Tags_Lateral student',1)
col1
X_train_sm = sm.add_constant(X_train[col1])

logm2 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())

res = logm2.fit()

res.summary()
col2 = col1.drop('Tags_number not provided',1)
col2
X_train_sm = sm.add_constant(X_train[col2])

logm2 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())

res = logm2.fit()

res.summary()
col3=col2.drop('Tags_invalid number',1)
X_train_sm = sm.add_constant(X_train[col3])

logm2 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())

res = logm2.fit()

res.summary()
# Getting the predicted values on the train set

y_train_pred = res.predict(X_train_sm)

y_train_pred[:10]
y_train_pred = y_train_pred.values.reshape(-1)

y_train_pred[:10]
y_train_pred_final = pd.DataFrame({'Converted':y_train.values, 'Conv_Prob':y_train_pred})

y_train_pred_final['ProspectID'] = y_train.index

y_train_pred_final.head()
y_train_pred_final['predicted'] = y_train_pred_final.Conv_Prob.map(lambda x: 1 if x > 0.5 else 0)



# Let's see the head

y_train_pred_final.head()
from sklearn import metrics
# Confusion matrix 

confusion = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final.predicted )

print(confusion)
# Predicted     not_converted    converted

# Actual

# not_converted        3843            159

# converted           367           2099 
# Let's check the overall accuracy.

print(metrics.accuracy_score(y_train_pred_final.Converted, y_train_pred_final.predicted))
from statsmodels.stats.outliers_influence import variance_inflation_factor
# Create a dataframe that will contain the names of all the feature variables and their respective VIFs

vif = pd.DataFrame()

vif['Features'] = X_train[col3].columns

vif['VIF'] = [variance_inflation_factor(X_train[col3].values, i) for i in range(X_train[col3].shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
TP = confusion[1,1] # true positive 

TN = confusion[0,0] # true negatives

FP = confusion[0,1] # false positives

FN = confusion[1,0] # false negatives
# Sensitivity

TP / float(TP+FN)
#Specificity

TN / float(TN+FP)
# Calculate false postive rate - predicting converted when customer did not convert

print(FP/ float(TN+FP))
# Positive predictive value 

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
fpr, tpr, thresholds = metrics.roc_curve( y_train_pred_final.Converted, y_train_pred_final.Conv_Prob, drop_intermediate = False )
draw_roc(y_train_pred_final.Converted, y_train_pred_final.Conv_Prob)
# Let's create columns with different probability cutoffs 

numbers = [float(x)/10 for x in range(10)]

for i in numbers:

    y_train_pred_final[i]= y_train_pred_final.Conv_Prob.map(lambda x: 1 if x > i else 0)

y_train_pred_final.head()
#Calculating accuracy,sensitivity and specificity for various probability cutoffs.

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
#### From the curve above, 0.18 is the optimum point to take it as a cutoff probability.



y_train_pred_final['final_predicted'] = y_train_pred_final.Conv_Prob.map( lambda x: 1 if x > 0.18 else 0)



y_train_pred_final.head()
y_train_pred_final['Lead_Score'] = y_train_pred_final.Conv_Prob.map( lambda x: round(x*100))



y_train_pred_final.head()
# Accuracy-

metrics.accuracy_score(y_train_pred_final.Converted, y_train_pred_final.final_predicted)
confusion2 = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final.final_predicted )

confusion2
TP = confusion2[1,1] # true positive 

TN = confusion2[0,0] # true negatives

FP = confusion2[0,1] # false positives

FN = confusion2[1,0] # false negative
# Sensitivity 

TP / float(TP+FN)
#Specificity

TN / float(TN+FP)
#False postive rate

print(FP/ float(TN+FP))
#Positive predictive value 

print (TP / float(TP+FP))

# Negative predictive value

print (TN / float(TN+ FN))

#Confusion matrix:



confusion = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final.predicted )

confusion
#Precision:

confusion[1,1]/(confusion[0,1]+confusion[1,1])
#Recall:

confusion[1,1]/(confusion[1,0]+confusion[1,1])
from sklearn.metrics import precision_recall_curve
y_train_pred_final.Converted, y_train_pred_final.predicted
p, r, thresholds = precision_recall_curve(y_train_pred_final.Converted, y_train_pred_final.Conv_Prob)
plt.plot(thresholds, p[:-1], "g-")

plt.plot(thresholds, r[:-1], "r-")

plt.show()
X_test[['Total Time Spent on Website']] = scaler.transform(X_test[['Total Time Spent on Website']])
X_test = X_test[col3]

X_test.head()
X_test_sm = sm.add_constant(X_test)
y_test_pred = res.predict(X_test_sm)
y_test_pred[:10]
# Converting y_pred to a dataframe which is an array

y_pred_1 = pd.DataFrame(y_test_pred)
y_pred_1.head()
y_test_df = pd.DataFrame(y_test)
# Putting ProspectID to index

y_test_df['ProspectID'] = y_test_df.index
# Removing index for both dataframes to append them side by side 

y_pred_1.reset_index(drop=True, inplace=True)

y_test_df.reset_index(drop=True, inplace=True)
# Appending y_test_df and y_pred_1

y_pred_final = pd.concat([y_test_df, y_pred_1],axis=1)
y_pred_final.head()
# Renaming the column :

y_pred_final= y_pred_final.rename(columns={ 0 : 'Conv_Prob'})
# Rearranging the columns:

#y_pred_final = y_pred_final.reindex_axis(['ProspectID','Converted','Conv_Prob'], axis=1)
#Head of y_pred_final

y_pred_final.head()
y_pred_final['final_predicted'] = y_pred_final.Conv_Prob.map(lambda x: 1 if x > 0.18 else 0)
y_pred_final.head()
#Overall accuracy.

metrics.accuracy_score(y_pred_final.Converted, y_pred_final.final_predicted)
#Adding a 'Lead Score' column :



y_pred_final['Lead_Score'] = y_pred_final.Conv_Prob.map( lambda x: round(x*100))
confusion2 = metrics.confusion_matrix(y_pred_final.Converted, y_pred_final.final_predicted )

confusion2
TP = confusion2[1,1] # True positive.

TN = confusion2[0,0] # True negative.

FP = confusion2[0,1] # False positive.

FN = confusion2[1,0] # Talse negative.
#Sensitivity:

TP / float(TP+FN)
# Specificity:

TN / float(TN+FP)
# False postive rate - predicting converted when customer did not convert:

print(FP/ float(TN+FP))
# Positive predictive value 

print (TP / float(TP+FP))
# Negative predictive value

print (TN / float(TN+ FN))
#Precision

Precision = confusion2[1,1]/(confusion2[0,1]+confusion2[1,1])

Precision
#Recall



Recall = confusion2[1,1]/(confusion2[1,0]+confusion2[1,1])

Recall
y_pred_final.head()
y_pred_final