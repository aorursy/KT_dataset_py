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



import statsmodels.api as sm

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression

from sklearn.feature_selection import RFE

from sklearn import metrics

from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.metrics import precision_score, recall_score

from sklearn.metrics import precision_recall_curve
df_Leads = pd.DataFrame(pd.read_csv('../input/Leads.csv'))

df_Leads.head() 
df_Leads.shape
df_Leads.describe()
df_Leads.info()
df_Leads.isna().sum()
df_Leads = df_Leads.replace('Select', np.nan)
round(100*(df_Leads.isnull().sum()/len(df_Leads.index)), 2)
df_Leads = df_Leads.drop(df_Leads.loc[:,list(round(100*(df_Leads.isnull().sum()/len(df_Leads.index)), 2)>70)].columns, 1)
df_Leads['Lead Quality'].describe()
sns.countplot(df_Leads['Lead Quality'])
df_Leads['Lead Quality'] = df_Leads['Lead Quality'].replace(np.nan, 'Not Sure')
sns.countplot(df_Leads['Lead Quality'])
df_Leads['City'].describe()
# lets visualise the countplots of each value in the column City.

sns.countplot(df_Leads['City'])

xticks(rotation = 45)
df_Leads['City'] = df_Leads['City'].replace(np.nan, 'Mumbai')
sns.countplot(df_Leads['City'])

xticks(rotation = 90)
fig, axs = plt.subplots(2,2, figsize = (10,7.5))

plt1 = sns.countplot(df_Leads['Asymmetrique Activity Index'], ax = axs[0,0])

plt2 = sns.boxplot(df_Leads['Asymmetrique Activity Score'], ax = axs[0,1])

plt3 = sns.countplot(df_Leads['Asymmetrique Profile Index'], ax = axs[1,0])

plt4 = sns.boxplot(df_Leads['Asymmetrique Profile Score'], ax = axs[1,1])

plt.tight_layout()
df_Leads = df_Leads.drop(['Asymmetrique Activity Index','Asymmetrique Activity Score','Asymmetrique Profile Index','Asymmetrique Profile Score'],1)
# Lets look at  .

df_Leads['Specialization'].describe()
plt.figure(figsize=(16, 5))

plt.xticks(rotation=45,fontsize=12,horizontalalignment='right')

ax1 = sns.countplot(df_Leads['Specialization'],hue=df_Leads.Converted)

plt.show()
df_Leads['Specialization'] = df_Leads['Specialization'].replace(np.nan, 'Others')
# Lets look at Tags.

df_Leads['Tags'].describe()
plt.figure(figsize=(16, 5))

plt.xticks(rotation=45,fontsize=12,horizontalalignment='right')

sns.countplot(df_Leads['Tags'])

plt.show()
df_Leads['Tags'] = df_Leads['Tags'].replace(np.nan, 'Will revert after reading the email')
# lets see the occupation coloumn.

df_Leads['What is your current occupation'].describe()
plt.figure(figsize=(16, 5))

ax1 = sns.countplot(df_Leads['What is your current occupation'])

plt.show()
df_Leads['What is your current occupation'] = df_Leads['What is your current occupation'].replace(np.nan, 'Unemployed')
plt.figure(figsize=(16, 5))

plt.xticks(rotation=45,fontsize=12,horizontalalignment='right')

ax1 = sns.countplot(df_Leads['Country'])

plt.show()
df_Leads['Country'].describe()
df_Leads['Country'] = df_Leads['Country'].replace(np.nan, 'India')
df_Leads['What matters most to you in choosing a course'].describe()
plt.figure(figsize=(16, 5))

plt.xticks(rotation=45,fontsize=12,horizontalalignment='right')

ax1 = sns.countplot(df_Leads['What matters most to you in choosing a course'])

plt.show()
df_Leads['What matters most to you in choosing a course'] = df_Leads['What matters most to you in choosing a course'].replace(np.nan,'Better Career Prospects')
df_Leads.dropna(inplace = True)
round(100*(df_Leads.isnull().sum()/len(df_Leads.index)), 2)
df_Leads['What matters most to you in choosing a course'].describe()
df_Leads.Search.describe()
df_Leads.Magazine.describe()
df_Leads['Newspaper Article'].describe()
df_Leads['X Education Forums'].describe()
df_Leads['Newspaper'].describe()
df_Leads['Digital Advertisement'].describe()
df_Leads['Through Recommendations'].describe()
df_Leads['Receive More Updates About Our Courses'].describe()
df_Leads['Update me on Supply Chain Content'].describe()
df_Leads['Get updates on DM Content'].describe()
df_Leads['I agree to pay the amount through cheque'].describe()
df_Leads['A free copy of Mastering The Interview'].describe()
Converted = round((sum(df_Leads['Converted'])/len(df_Leads['Converted'].index))*100,1)

Converted
sns.boxplot(df_Leads['TotalVisits'])
percentiles = df_Leads['TotalVisits'].quantile([0.05,0.95]).values

df_Leads['TotalVisits'][df_Leads['TotalVisits'] <= percentiles[0]] = percentiles[0]

df_Leads['TotalVisits'][df_Leads['TotalVisits'] >= percentiles[1]] = percentiles[1]



sns.boxplot(y = 'TotalVisits', x = df_Leads.Converted , data = df_Leads)
sns.boxplot(df_Leads['Total Time Spent on Website'])
sns.boxplot(y = 'Total Time Spent on Website', x = df_Leads.Converted, data = df_Leads)
sns.boxplot(df_Leads['Page Views Per Visit'])
percentiles = df_Leads['Page Views Per Visit'].quantile([0.05,0.95]).values

df_Leads['Page Views Per Visit'][df_Leads['Page Views Per Visit'] <= percentiles[0]] = percentiles[0]

df_Leads['Page Views Per Visit'][df_Leads['Page Views Per Visit'] >= percentiles[1]] = percentiles[1]



sns.boxplot(y = 'Page Views Per Visit', x = df_Leads.Converted, data = df_Leads)
plt.figure(figsize=(8,5))

plt.xticks(rotation=45,fontsize=12,horizontalalignment='right')

sns.countplot(df_Leads['Lead Origin'], hue=df_Leads.Converted)

plt.show()
fig, axs = plt.subplots(1,2,figsize = (15,7))

sns.countplot(x = "Do Not Email", hue = df_Leads.Converted , data = df_Leads, ax = axs[0])

sns.countplot(x = "Do Not Call", hue = df_Leads.Converted, data = df_Leads, ax = axs[1])
plt.figure(figsize=(8,5))

plt.xticks(rotation=45,fontsize=12,horizontalalignment='right')

sns.countplot(df_Leads['Lead Source'], hue=df_Leads.Converted)

plt.show()
# correcting the spelling of google

df_Leads['Lead Source'] = df_Leads['Lead Source'].replace(['google'], 'Google')
df_Leads['Lead Source'] = df_Leads['Lead Source'].replace(['Click2call', 'Live Chat', 'NC_EDM', 'Pay per Click Ads', 'Press_Release',

  'Social Media', 'WeLearn', 'bing', 'blog', 'testone', 'welearnblog_Home', 'youtubechannel'], 'Others')
plt.figure(figsize=(8,5))

plt.xticks(rotation=45,fontsize=12,horizontalalignment='right')

sns.countplot(x = "Lead Source", hue = df_Leads.Converted , data = df_Leads)

plt.show()
plt.figure(figsize=(16,5))

plt.xticks(rotation=45,fontsize=12,horizontalalignment='right')

sns.countplot(x = "Last Activity", hue = df_Leads.Converted, data = df_Leads)

plt.show()
df_Leads['Last Activity'] = df_Leads['Last Activity'].replace(['Had a Phone Conversation', 'View in browser link Clicked', 

                                                         'Visited Booth in Tradeshow', 'Approached upfront',

                                                         'Resubscribed to emails','Email Received', 'Email Marked Spam'], 

                                                         'Other_Activity')
plt.figure(figsize=(16,5))

plt.xticks(rotation=45,fontsize=12,horizontalalignment='right')

sns.countplot(x = "Last Activity", hue = df_Leads.Converted, data = df_Leads)

plt.show()
plt.figure(figsize=(16,5))

plt.xticks(rotation=45,fontsize=12,horizontalalignment='right')

sns.countplot(x = 'Specialization', hue = df_Leads.Converted, data = df_Leads)

plt.show()
df_Leads['Specialization'] = df_Leads['Specialization'].replace(['Others'], 'Other_Specialization')
plt.figure(figsize=(16,5))

plt.xticks(rotation=45,fontsize=12,horizontalalignment='right')

sns.countplot(x = 'Specialization', hue = df_Leads.Converted, data = df_Leads)

plt.show()
df_Leads['What is your current occupation'] = df_Leads['What is your current occupation'].replace(['Other'], 

                                                                                                  'Other_Occupation')
plt.figure(figsize=(16,5))

plt.xticks(rotation=45,fontsize=12,horizontalalignment='right')

sns.countplot(x = 'What is your current occupation', hue = df_Leads.Converted, data = df_Leads)

plt.show()
plt.figure(figsize=(16,5))

plt.xticks(rotation=45,fontsize=12,horizontalalignment='right')

sns.countplot(x = 'Tags', hue = df_Leads.Converted, data = df_Leads)

plt.show()
df_Leads['Tags'] = df_Leads['Tags'].replace(['In confusion whether part time or DLP', 'in touch with EINS','Diploma holder (Not Eligible)',

                                     'Approached upfront','Graduation in progress','number not provided', 'opp hangup','Still Thinking',

                                    'Lost to Others','Shall take in the next coming month','Lateral student','Interested in Next batch',

                                    'Recognition issue (DEC approval)','Want to take admission but has financial problems',

                                    'University not recognized'], 'Other_Tags')
plt.figure(figsize=(16,5))

plt.xticks(rotation=45,fontsize=12,horizontalalignment='right')

sns.countplot(x = 'Tags', hue = df_Leads.Converted, data = df_Leads)

plt.show()
plt.figure(figsize=(16,5))

plt.xticks(rotation=45,fontsize=12,horizontalalignment='right')

sns.countplot(x = 'Lead Quality', hue = df_Leads.Converted, data = df_Leads)

plt.show()
plt.figure(figsize=(16,5))

plt.xticks(rotation=45,fontsize=12,horizontalalignment='right')

sns.countplot(x = 'City', hue = df_Leads.Converted, data = df_Leads)

plt.show()
plt.figure(figsize=(16,5))

plt.xticks(rotation=45,fontsize=12,horizontalalignment='right')

sns.countplot(x = 'Last Notable Activity', hue = df_Leads.Converted, data = df_Leads)

plt.show()
df_Leads = df_Leads.drop(['Lead Number','What matters most to you in choosing a course','Search','Magazine','Newspaper Article','X Education Forums','Newspaper',

           'Digital Advertisement','Through Recommendations','Receive More Updates About Our Courses','Update me on Supply Chain Content',

           'Get updates on DM Content','I agree to pay the amount through cheque','A free copy of Mastering The Interview','Country'],axis=1)
df_Leads.shape
df_Leads.head()
# List of binary variables to map

varlist =  ['Do Not Email', 'Do Not Call']



# standardize values by converting all to lower case

df_Leads[varlist] = df_Leads[varlist].applymap(lambda s : s.lower() if type(s) == str else s)



# Defining the map function

def binary_map(x):

    return x.map({'yes': 1, "no": 0})



# Applying the function to the housing list

df_Leads[varlist] = df_Leads[varlist].apply(binary_map)
dummy = pd.get_dummies(df_Leads[['Lead Origin', 'Lead Source', 'Last Activity', 'Specialization','What is your current occupation',

                              'Tags','Lead Quality','City','Last Notable Activity']], drop_first=True)

dummy.head()
df_Leads = pd.concat([df_Leads, dummy], axis=1)

df_Leads.head()
df_Leads = df_Leads.drop(['Lead Origin', 'Lead Source', 'Last Activity', 'Specialization',

                          'What is your current occupation','Tags','Lead Quality','City',

                          'Last Notable Activity'], axis = 1)
df_Leads.shape
df_Leads.head()
X = df_Leads.drop(['Prospect ID','Converted'], axis=1)

X.head()
y = df_Leads['Converted']

y.head()
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=100)
# instantiating the standard scaler

scaler = StandardScaler()



num_cols = X_train[['TotalVisits','Total Time Spent on Website','Page Views Per Visit']]



# Scaling the numerical columns..

X_train[['TotalVisits','Total Time Spent on Website','Page Views Per Visit']] = scaler.fit_transform(num_cols)



X_train.head()
logm1 = sm.GLM(y_train,(sm.add_constant(X_train)), family = sm.families.Binomial())

logm1.fit().summary()
logreg = LogisticRegression()

rfe = RFE(logreg, 15)

rfe = rfe.fit(X_train, y_train)
col = X_train.columns[rfe.support_]

col
X_train.columns[~rfe.support_]
X_train_sm = sm.add_constant(X_train[col])

logm2 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())

res = logm2.fit()

res.summary()
col1 = col.drop('Tags_invalid number',1)
X_train_sm = sm.add_constant(X_train[col1])

logm2 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())

res = logm2.fit()

res.summary()
col2 = col1.drop('Tags_wrong number given',1)
X_train_sm = sm.add_constant(X_train[col2])

logm2 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())

res = logm2.fit()

res.summary()
y_train_pred = res.predict(X_train_sm)

y_train_pred.head()
y_train_pred = y_train_pred.values.reshape(-1)

y_train_pred[:15]
y_train_pred_final = pd.DataFrame({'Converted':y_train.values, 'Converted_prob':y_train_pred})

y_train_pred_final['Prospect ID'] = y_train.index

y_train_pred_final.head()
y_train_pred_final['predicted'] = y_train_pred_final['Converted_prob'].map(lambda x: 1 if x > 0.5 else 0)

y_train_pred_final.head()
confusion_1 = metrics.confusion_matrix(y_train_pred_final['Converted'], y_train_pred_final['predicted'] )

print(confusion_1)
print(metrics.accuracy_score(y_train_pred_final['Converted'], y_train_pred_final['predicted']))
vif = pd.DataFrame()

vif['Features'] = X_train[col2].columns

vif['VIF'] = [variance_inflation_factor(X_train[col].values, i) for i in range(X_train[col2].shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
TP = confusion_1[1,1] # true positive 

TN = confusion_1[0,0] # true negatives

FP = confusion_1[0,1] # false positives

FN = confusion_1[1,0] # false negatives
print(TP)

print(TN)

print(FP)

print(FN)
TP / float(TP+FN)
TN / float(TN+FP)
print(FP/ float(TN+FP))
print(TP / float(TP+FP))
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

    plt.xlabel('False Positive Rate or [1 - Specificity]')

    plt.ylabel('True Positive Rate / Sensitivity')

    plt.title('Receiver operating characteristic Curve')

    plt.legend(loc="lower right")

    plt.show()



    return None
fpr, tpr, thresholds = metrics.roc_curve( y_train_pred_final['Converted'], y_train_pred_final['Converted_prob'], drop_intermediate = False )
draw_roc(y_train_pred_final['Converted'], y_train_pred_final['Converted_prob'])
prob_numbers = [float(x)/10 for x in range(10)]

for i in prob_numbers:

    y_train_pred_final[i]= y_train_pred_final.Converted_prob.map(lambda m: 1 if m > i else 0)

y_train_pred_final.head()
prob_cutoff_df = pd.DataFrame( columns = ['prob','accuracy','sensi','speci'])



num = [float(x)/10 for x in range(10)]

for i in num:

    cm1 = metrics.confusion_matrix(y_train_pred_final['Converted'], y_train_pred_final[i] )

    

    # accuracy.

    total1=sum(sum(cm1))

    accuracy = (cm1[0,0]+cm1[1,1])/total1

    

    # sensitivity and specificity.

    speci = cm1[0,0]/(cm1[0,0]+cm1[0,1])

    sensi = cm1[1,1]/(cm1[1,0]+cm1[1,1])

    prob_cutoff_df.loc[i] =[ i ,accuracy,sensi,speci]

    

print(prob_cutoff_df)
prob_cutoff_df.plot.line(x='prob', y=['accuracy','sensi','speci'])

plt.show()
y_train_pred_final['final_predicted'] = y_train_pred_final.Converted_prob.map( lambda c: 1 if c > 0.2 else 0)

y_train_pred_final.head()
y_train_pred_final['Lead_Score'] = y_train_pred_final['Converted_prob'].map( lambda z: round(z*100))

y_train_pred_final.head()
metrics.accuracy_score(y_train_pred_final.Converted, y_train_pred_final.final_predicted)
confusion_2 = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final.final_predicted )

confusion_2
TP = confusion_2[1,1] # true positive 

TN = confusion_2[0,0] # true negatives

FP = confusion_2[0,1] # false positives

FN = confusion_2[1,0] # false negatives
TP / float(TP+FN)
TN / float(TN+FP)
print(FP/ float(TN+FP))
print (TP / float(TP+FP))
print (TN / float(TN+ FN))
precision_score(y_train_pred_final['Converted'] , y_train_pred_final['predicted'])
recall_score(y_train_pred_final['Converted'], y_train_pred_final['predicted'])
p, r, thresholds = precision_recall_curve(y_train_pred_final['Converted'], y_train_pred_final['Converted_prob'])

plt.plot(thresholds, p[:-1], "g-")

plt.plot(thresholds, r[:-1], "r-")

plt.show()
num_cols = X_test[['TotalVisits','Total Time Spent on Website','Page Views Per Visit']]

X_test[['TotalVisits','Total Time Spent on Website','Page Views Per Visit']] = scaler.fit_transform(num_cols)

X_test.head()
X_test = X_test[col2]

X_test.head()
X_test_sm = sm.add_constant(X_test)
y_test_pred = res.predict(X_test_sm)

y_test_pred[:15]
# Converting y_pred to a dataframe from array.

y_pred_1 = pd.DataFrame(y_test_pred)

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

y_pred_final = y_pred_final[['Prospect ID','Converted','Converted_prob']]

y_pred_final.head()
# adding our prediction for threshold value of 0.2

y_pred_final['final_predicted'] = y_pred_final['Converted_prob'].map(lambda x: 1 if x > 0.2 else 0)

y_pred_final.head()
metrics.accuracy_score(y_pred_final['Converted'], y_pred_final['final_predicted'])
confusion_3 = metrics.confusion_matrix(y_pred_final.Converted, y_pred_final.final_predicted )

confusion_3
TP = confusion_3[1,1] # true positive 

TN = confusion_3[0,0] # true negatives

FP = confusion_3[0,1] # false positives

FN = confusion_3[1,0] # false negatives
TP / float(TP+FN)
TN / float(TN+FP)
precision_score(y_pred_final['Converted'] , y_pred_final['final_predicted'])
recall_score(y_pred_final['Converted'], y_pred_final['final_predicted'])