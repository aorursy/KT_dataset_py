# importing all the required libraries

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings('ignore')

import seaborn as sns 

from matplotlib.pyplot import xticks

%matplotlib inline

from sklearn.model_selection import train_test_split

import statsmodels.api as sm

from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.metrics import precision_score, recall_score

from sklearn.metrics import precision_recall_curve
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
path = '/kaggle/input/leads-dataset/Leads.csv' 
leadScoreDf = pd.read_csv( path )
leadScoreDf.head( )
leadScoreDf.shape
leadScoreDf.describe( )
leadScoreDf.info( )
leadScoreDf = leadScoreDf.replace('Select', np.nan)
# Checking for null values for all the columns

leadScoreDf.isnull( ).sum( axis = 0 )
# Looking at the percentages would give a better insight - 

round( 100*( leadScoreDf.isnull( ).sum( )/len( leadScoreDf.index ) ), 2 )
#Dropping columns with more than 70% missing values - 



leadScoreDf = leadScoreDf.drop( leadScoreDf.loc[:,list( round( 100*( leadScoreDf.isnull().sum()/len( leadScoreDf.index ) ), 2 ) > 70 ) ].columns, 1 )
leadScoreDf[ 'Lead Quality' ].describe( )
leadScoreDf[ 'Lead Quality' ].value_counts( )
leadScoreDf[ 'Lead Quality' ] = leadScoreDf[ 'Lead Quality' ].replace( np.nan, 'Not sure' )
fig, axs = plt.subplots(2,2, figsize = (10,8))

plt1 = sns.countplot(leadScoreDf['Asymmetrique Activity Index'], ax = axs[0,0])

plt2 = sns.boxplot(leadScoreDf['Asymmetrique Activity Score'], ax = axs[0,1])

plt3 = sns.countplot(leadScoreDf['Asymmetrique Profile Index'], ax = axs[1,0])

plt4 = sns.boxplot(leadScoreDf['Asymmetrique Profile Score'], ax = axs[1,1])

plt.tight_layout()
leadScoreDf[ 'Asymmetrique Activity Index' ].describe( )
leadScoreDf[ 'Asymmetrique Activity Index' ].value_counts( )
leadScoreDf[ 'Asymmetrique Profile Index' ].value_counts( )
leadScoreDf[ 'Asymmetrique Activity Score' ].value_counts( )
leadScoreDf[ 'Asymmetrique Profile Score' ].value_counts( )
leadScoreDf = leadScoreDf.drop( [ 'Asymmetrique Activity Index','Asymmetrique Activity Score',

                                 'Asymmetrique Profile Index','Asymmetrique Profile Score' ], 1 )
leadScoreDf[ 'City' ].value_counts( )
leadScoreDf['City'] = leadScoreDf['City'].replace( np.nan, 'Mumbai' )
leadScoreDf[ 'Tags' ].value_counts( )
leadScoreDf[ 'Tags' ] = leadScoreDf[ 'Tags' ].replace( np.nan, 'Will revert after reading the email' )
leadScoreDf[ 'Specialization' ].value_counts( )
leadScoreDf[ 'Specialization' ] = leadScoreDf[ 'Specialization' ].replace( np.nan, 'Others' )
leadScoreDf[ 'What is your current occupation'].value_counts( )
leadScoreDf[ 'What is your current occupation'] = leadScoreDf[ 'What is your current occupation'].replace( np.nan, 'Unemployed' )
leadScoreDf[ 'What matters most to you in choosing a course' ].value_counts( )
leadScoreDf['What matters most to you in choosing a course'] = leadScoreDf['What matters most to you in choosing a course'].replace( np.nan, 'Better Career Prospects' )
leadScoreDf[ 'Country' ].value_counts( )
leadScoreDf[ 'Country' ] = leadScoreDf[ 'Country' ].replace( np.nan, 'India' )
leadScoreDf.dropna( inplace = True )
# Checking the % of missing values after the imputation - 

round( 100*( leadScoreDf.isnull( ).sum( )/len( leadScoreDf.index ) ), 2 )
# We can drop the last five columns as they have only 1 unique value

leadScoreDf.drop(['Receive More Updates About Our Courses', 'Update me on Supply Chain Content', 'Get updates on DM Content', 'Magazine',  

       'I agree to pay the amount through cheque'], axis = 1, inplace = True)
# Dropping Prospect ID & Lead Number as they are only IDs

leadScoreDf.drop(['Prospect ID','Lead Number'], axis = 1, inplace = True)
for var in leadScoreDf.select_dtypes(exclude = ['int64', 'float64']).columns:

    print(leadScoreDf[var].value_counts(), '\n')
sns.countplot(x = "Lead Origin", hue = "Converted", data = leadScoreDf )

xticks(rotation = 90)
sns.countplot(x = "Lead Source", hue = "Converted", data = leadScoreDf )

xticks( rotation = 90 )
leadScoreDf['Lead Source'].value_counts( )
leadScoreDf['Lead Source'] = leadScoreDf[ 'Lead Source'].replace(['Click2call', 'Live Chat', 'NC_EDM', 'Pay per Click Ads', 'Press_Release',

  'Social Media', 'WeLearn', 'bing', 'blog', 'testone', 'welearnblog_Home', 'youtubechannel'], 'Others')



leadScoreDf['Lead Source']  = leadScoreDf[ 'Lead Source' ].replace( 'google', 'Google' )
sns.countplot(x = "Lead Source", hue = "Converted", data = leadScoreDf )

xticks( rotation = 90 )
fig, axs = plt.subplots(1,2, figsize = (10,8))

plt1 = sns.countplot(x = "Do Not Email", hue = "Converted", data = leadScoreDf, ax = axs[ 0]  )

xticks( rotation = 90 )



plt2 = sns.countplot(x = "Do Not Call", hue = "Converted", data = leadScoreDf, ax = axs[ 1 ] )

xticks( rotation = 90 )
sns.countplot(x = "Last Activity", hue = "Converted", data = leadScoreDf )

xticks( rotation = 90 )
leadScoreDf['Last Activity'] = leadScoreDf['Last Activity'].replace(['Had a Phone Conversation', 'View in browser link Clicked', 

                                                       'Visited Booth in Tradeshow', 'Approached upfront',

                                                       'Resubscribed to emails','Email Received', 'Email Marked Spam'], 'Other Activity')
sns.countplot(x = "Last Activity", hue = "Converted", data = leadScoreDf )

xticks( rotation = 90 )
fig, axs = plt.subplots(figsize = (15,5))

sns.countplot(x = "Country", hue = "Converted", data = leadScoreDf )

xticks( rotation = 90 )
leadScoreDf.drop( 'Country', inplace = True, axis = 1 )
fig, axs = plt.subplots(figsize = (15,5))

sns.countplot(x = "Specialization", hue = "Converted", data = leadScoreDf )

xticks( rotation = 90 )
fig, axs = plt.subplots(2,1, figsize = (15,10))

plt1 = sns.countplot(y = "What is your current occupation", hue = "Converted", data = leadScoreDf, ax = axs[ 0]  )

xticks( rotation = 90 )



plt2 = sns.countplot(y = "What matters most to you in choosing a course", hue = "Converted", data = leadScoreDf, ax = axs[ 1 ] )

xticks( rotation = 90 )
leadScoreDf['What matters most to you in choosing a course'].value_counts( )
leadScoreDf.drop( 'What matters most to you in choosing a course', inplace = True, axis = 1 )
# 'Others' category is already present in Specialization, it would be better to rename this category to 'Other Occupation'

leadScoreDf['What is your current occupation'] = leadScoreDf['What is your current occupation'].replace( 'Others', 'Other Occupation' )
sns.countplot( x = 'Search', hue = 'Converted', data = leadScoreDf )
leadScoreDf['Search'].value_counts( )
leadScoreDf.drop( 'Search', axis = 1, inplace = True )
fig, axs = plt.subplots(1,3, figsize = (15,5))

plt1 = sns.countplot(x = "Newspaper Article", hue = "Converted", data = leadScoreDf, ax = axs[ 0]  )

xticks( rotation = 90 )



plt2 = sns.countplot(x = "Newspaper", hue = "Converted", data = leadScoreDf, ax = axs[ 1 ] )

xticks( rotation = 90 )



plt2 = sns.countplot( x = 'Digital Advertisement', hue = 'Converted', data = leadScoreDf, ax = axs[ 2 ] )

xticks( rotation = 90 )
leadScoreDf['Newspaper Article'].value_counts( )
leadScoreDf['Newspaper'].value_counts( )
leadScoreDf['Digital Advertisement'].value_counts( )
leadScoreDf.drop(['Newspaper', 'Newspaper Article', 'Digital Advertisement' ], axis = 1, inplace = True )
sns.countplot( x = 'X Education Forums', hue = 'Converted', data = leadScoreDf )

xticks( rotation = 90 )
leadScoreDf[ 'X Education Forums' ].value_counts( )
leadScoreDf.drop( 'X Education Forums', axis = 1, inplace = True )
sns.countplot( x = 'Through Recommendations', hue = 'Converted', data = leadScoreDf )
leadScoreDf['Through Recommendations'].value_counts( )
leadScoreDf.drop( 'Through Recommendations', axis = 1, inplace = True )
plt.subplots(figsize = (15,5))

sns.countplot( x = 'Tags', hue = 'Converted', data = leadScoreDf )

xticks( rotation = 90 )
leadScoreDf['Tags'].value_counts( )
leadScoreDf['Tags'] = leadScoreDf['Tags'].replace( ['in touch with EINS', 'Lost to Others', 'Still Thinking', 

                              'Want to take admission but has financial problems', 'Interested in Next batch',

                              'Shall take in the next coming month', 'University not recognized'

                              'In confusion whether part time or DLP', 'Lateral student', 

                              'University not recognized', 'Recognition issue (DEC approval)'], 'Other Tags' )
plt.subplots(figsize = (15,5))

sns.countplot( x = 'Tags', hue = 'Converted', data = leadScoreDf )

xticks( rotation = 90 )
sns.countplot( x = 'Lead Quality', hue = 'Converted', data = leadScoreDf )

xticks( rotation = 90 )
leadScoreDf['Lead Quality'] = leadScoreDf['Lead Quality'].replace('Not sure', 'Not Sure')
plt.subplots(figsize = (15,5))

sns.countplot( x = 'City', hue = 'Converted', data = leadScoreDf )

xticks( rotation = 90 )
leadScoreDf.shape
leadScoreDf.head( )
# columns to be mapped -

colList =  ['Do Not Email', 'Do Not Call', 'A free copy of Mastering The Interview']



# Defining the map function

def binary_map(x):

    return x.map({'Yes': 1, "No": 0})



# Applying the function to the columns

leadScoreDf[colList] = leadScoreDf[colList].apply(binary_map)
dummyCols = [ 'Lead Origin', 'Lead Source', 'Last Activity', 'Specialization','What is your current occupation',

                              'Tags','Lead Quality','City','Last Notable Activity' ]
# Creating a dummy variable for some of the categorical variables and dropping the first one.

dummyDf = pd.get_dummies( leadScoreDf [ dummyCols ], drop_first=True )

dummyDf.head()
# Joining the dummy dataframe and the leadScore dataframe -



leadScoreDf = pd.concat( [ leadScoreDf, dummyDf ], axis=1 )

leadScoreDf.head()
leadScoreDf.drop( dummyCols, axis = 1, inplace = True )
leadScoreDf.head()
# Putting feature variable to X

X = leadScoreDf.drop( 'Converted', axis=1 )

y = leadScoreDf[ 'Converted' ]
y.head( )
X.head( )
# Splitting the data into train and test

X_train, X_test, y_train, y_test = train_test_split( X, y, train_size=0.7, test_size=0.3, random_state=100 )
from sklearn.preprocessing import StandardScaler



scaler = StandardScaler()



X_train[['TotalVisits','Total Time Spent on Website','Page Views Per Visit']] = scaler.fit_transform(X_train[['TotalVisits','Total Time Spent on Website','Page Views Per Visit']])



X_train.head( )
Converted = ( sum( leadScoreDf[ 'Converted' ] ) / len( leadScoreDf[ 'Converted' ] ) ) * 100

Converted
# Logistic regression model

logm1 = sm.GLM( y_train,( sm.add_constant( X_train ) ), family = sm.families.Binomial( ) )

logm1.fit( ).summary() 
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()



from sklearn.feature_selection import RFE

rfe = RFE(logreg, 15)             # running RFE with 15 variables as output

rfe = rfe.fit(X_train, y_train)
rfe.support_
rfeCols = X_train.columns[rfe.support_]

rfeCols
X_train_sm = sm.add_constant( X_train[ rfeCols ] )

logm2 = sm.GLM( y_train,X_train_sm, family = sm.families.Binomial( ) )

res = logm2.fit( )

res.summary( )
rfeCol1 = rfeCols.drop( [ 'Tags_invalid number', 'Tags_number not provided' ], 1 )
# Rebuilding the model - 

X_train_sm = sm.add_constant( X_train[ rfeCol1 ] )

logm2 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())

res = logm2.fit()

res.summary()
y_train_pred = res.predict(X_train_sm)

y_train_pred[ : 5 ]
y_train_pred = y_train_pred.values.reshape(-1)

y_train_pred[ : 5 ]
y_train_pred_final = pd.DataFrame( { 'Converted':y_train.values, 'Converted_prob':y_train_pred } )

y_train_pred_final['Prospect ID'] = y_train.index

y_train_pred_final.head()
y_train_pred_final['predicted'] = y_train_pred_final.Converted_prob.map(lambda x: 1 if x > 0.5 else 0)



y_train_pred_final.head()
from sklearn import metrics



# Confusion matrix -

confusion = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final.predicted )

print(confusion)
# Overall accuracy -

print(metrics.accuracy_score(y_train_pred_final.Converted, y_train_pred_final.predicted))
#feature variables and their respective VIFs - 

vif = pd.DataFrame()

vif[ 'Features' ] = X_train[ rfeCol1 ].columns

vif[ 'VIF' ] = [ variance_inflation_factor(X_train[ rfeCols ].values, i ) for i in range( X_train[ rfeCol1 ].shape[1] ) ]

vif[ 'VIF' ] = round( vif[ 'VIF' ], 2 )

vif = vif.sort_values( by = "VIF", ascending = False )

vif
TP = confusion[1,1] # true positive 

TN = confusion[0,0] # true negatives

FP = confusion[0,1] # false positives

FN = confusion[1,0] # false negatives
TP / float(TP+FN)
TN / float(TN+FP)
print (TP / float(TP+FP))
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
draw_roc( y_train_pred_final.Converted, y_train_pred_final.Converted_prob )
# create columns with different probability cutoffs 

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
# Plotting accuracy sensitivity and specificity for various probabilities.

cutoff_df.plot.line(x='prob', y=['accuracy','sensi','speci'])

plt.show()
y_train_pred_final['final_predicted'] = y_train_pred_final.Converted_prob.map( lambda x: 1 if x > 0.2 else 0)



y_train_pred_final.head()
y_train_pred_final[ 'Lead_Score' ] = y_train_pred_final.Converted_prob.map( lambda x: round(x*100))



y_train_pred_final.head()


metrics.accuracy_score(y_train_pred_final.Converted, y_train_pred_final.final_predicted)



confusion2 = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final.final_predicted )

confusion2



TP = confusion2[1,1] # true positive 

TN = confusion2[0,0] # true negatives

FP = confusion2[0,1] # false positives

FN = confusion2[1,0] # false negatives
TP / float(TP+FN)
TN / float(TN+FP)


TP / TP + FP



confusion[1,1]/(confusion[0,1]+confusion[1,1])
precision_score(y_train_pred_final.Converted , y_train_pred_final.predicted)
TP / TP + FN



confusion[1,1]/(confusion[1,0]+confusion[1,1])
recall_score(y_train_pred_final.Converted, y_train_pred_final.predicted)
X_test[['TotalVisits','Total Time Spent on Website','Page Views Per Visit']] = scaler.fit_transform(X_test[['TotalVisits','Total Time Spent on Website','Page Views Per Visit']])



X_train.head()
X_test = X_test[ rfeCol1 ]

X_test.head()
X_test_sm = sm.add_constant(X_test)
y_test_pred = res.predict( X_test_sm )
# Converting y_pred and y_test to a dataframe which is an array

y_pred_1 = pd.DataFrame(y_test_pred)

y_test_df = pd.DataFrame(y_test)
y_test_df['Prospect ID'] = y_test_df.index
y_pred_1.reset_index(drop=True, inplace=True)

y_test_df.reset_index(drop=True, inplace=True)

y_pred_final = pd.concat([y_test_df, y_pred_1],axis=1)
y_pred_final.head()
# Renaming the column 

y_pred_final= y_pred_final.rename(columns={ 0 : 'Converted_prob'})



y_pred_final['final_predicted'] = y_pred_final.Converted_prob.map(lambda x: 1 if x > 0.2 else 0)
y_pred_final.head()
metrics.accuracy_score(y_pred_final.Converted, y_pred_final.final_predicted)
confusion3 = metrics.confusion_matrix(y_pred_final.Converted, y_pred_final.final_predicted )

confusion3
TP = confusion2[1,1] # true positive 

TN = confusion2[0,0] # true negatives

FP = confusion2[0,1] # false positives

FN = confusion2[1,0] # false negatives
TP / float(TP+FN)
TN / float(TN+FP)
res.params