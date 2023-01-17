# Importing libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

import warnings

warnings.filterwarnings('ignore')

sns.set(style="whitegrid", color_codes=True)

pd.set_option('display.max_columns', 500)
# Read dataset with "Select" values as "NA"

df = pd.read_csv('../input/Leads.csv', na_values= 'Select')
copy = pd.read_csv('../input/Leads.csv', na_values= 'Select')
df.head()
df.info()
df.duplicated().sum()
(df.select_dtypes(include = ['int64', 'float64']).isnull().mean()*100).sort_values(ascending = False)
(df.select_dtypes(exclude = ['int64', 'float64']).isnull().mean()*100).sort_values(ascending = False)
#Checking for Outliers

df.select_dtypes(include = ['int64', 'float64']).describe(percentiles=[.25, .5, .75, .90, .95, .99])
# Need to remove outliers (Pending)
plt.figure(figsize = (15,6))

sns.heatmap(df.isna())
# Given high NA values for "Asymmetriuq" features, checking whether index features are sufficient to give info about scores

sns.boxplot(x = 'Asymmetrique Activity Index', y = 'Asymmetrique Activity Score', data = df)
sns.boxplot(x = 'Asymmetrique Profile Index' , y = 'Asymmetrique Profile Score', data = df)
df.drop(['Asymmetrique Profile Score', 'Asymmetrique Activity Score'], axis = 1, inplace = True)
# Dropping rows in "Page Views Per Visit" and "TotalVisits" with NA values -> ~1.5% of data

df.dropna(subset = ["Page Views Per Visit", "TotalVisits"], inplace = True)
(df.select_dtypes(include = ['int64', 'float64']).isnull().mean()*100).sort_values(ascending = False)
df.describe()
# Dropping 'Prospect ID' & 'Lead Number' since these are just ID features

df.drop(['Prospect ID', 'Lead Number'], axis = 1, inplace = True)
df.select_dtypes(exclude = ['int64', 'float64']).nunique().sort_values(ascending = False)
# Dropping features with only 1 value (i.e. no variance/ info)

df.drop(['Update me on Supply Chain Content', 'Magazine', 'Get updates on DM Content', 

       'I agree to pay the amount through cheque', 'Receive More Updates About Our Courses'], axis = 1, inplace = True)
for var in df.select_dtypes(exclude = ['int64', 'float64']).columns:

    print(df[var].value_counts(), '\n')
# Dropping features with almost no variance (only 1 value for all samples)

drop_col = ['Do Not Call', 'What matters most to you in choosing a course', 'Newspaper Article', 'X Education Forums',

           'Newspaper']
df.drop(drop_col, axis = 1, inplace = True)
pd.crosstab(df['Last Activity'], df['Last Notable Activity'])
# Dropping 'Last Notable Activity': Since there is high correlation with 'Last activity' - so dropping the repeated feature

df.drop('Last Notable Activity', axis = 1, inplace = True)
df.select_dtypes(include = ['int64', 'float64']).info()
df.select_dtypes(exclude = ['int64', 'float64']).nunique().sort_values(ascending = False)
# categorize_label = lambda x: x.astype('category')

# LABELS = df.select_dtypes(exclude=['int64', 'float64']).columns

# df[LABELS] = df[LABELS].apply(categorize_label, axis=0)
plt.figure(figsize = (15,25))

for i, var in enumerate(df.select_dtypes(exclude = ['int64', 'float64']).columns):

    plt.subplot(6,3,i+1)

    sns.barplot(x = var, y = 'Converted', data = df)

    #plt.xticks(rotation=45)

plt.tight_layout()
plt.figure(figsize = (15,10))

sns.countplot(x = 'Tags',  data = df, palette="Set3")

plt.xlim(xmin = 0)

plt.xticks(rotation = 90)

ax2 = plt.twinx()

sns.lineplot(x = 'Tags', y = 'Converted', data = df, ax = ax2)
"""Need to drop rows with following tags since decision has already been taken (converted/ not converted) i.e. 

target variable is known and there is no need to predict for these samples""" 

tags_to_drop = ['Already a student', 'Closed by Horizzon', 'Lateral student', 'Lost to EINS', 'Lost to Others']
df = df[~df['Tags'].isin(tags_to_drop)]
plt.figure(figsize = (15,10))

sns.countplot(x = 'Last Activity',  data = df, palette="Set3")

#plt.xlim(xmin = 0)

plt.xticks(rotation = 90)

ax2 = plt.twinx()

sns.lineplot(x = 'Last Activity', y = 'Converted', data = df, ax = ax2)
sns.pairplot(df, hue = 'Converted')
sns.heatmap(df.corr(), annot = True, cmap="YlGnBu")

plt.xticks(rotation = '30')
df.select_dtypes(exclude = ['int64', 'float64']).nunique().sort_values(ascending = False)
# First, converting the categorical features with relatively less unique values (<10)

cat_col = ['How did you hear about X Education', 'City', 'What is your current occupation', 'Lead Profile', 'Lead Quality',

           'Lead Origin', 'Asymmetrique Activity Index', 'Asymmetrique Profile Index' , 'Do Not Email' , 

           'A free copy of Mastering The Interview', 'Digital Advertisement' , 'Through Recommendations', 'Search']
df = pd.get_dummies(df, columns = cat_col)
df.head()
# Dropping 1 column each for the features with no NaN values.

# No need to drop 1 column for features which had NaN values initially, since NaN column has already been dropped during encoding

drop_col = ['Do Not Email_No', 'A free copy of Mastering The Interview_No', 'Digital Advertisement_No', 

            'Through Recommendations_No', 'Search_No', 'Lead Origin_Lead Import']
df.columns
df.drop(drop_col, axis = 1, inplace = True)
df.head()
round(df['Country'].value_counts()/len(df['Country'])*100,1)
Top_countries = ['India', 'United States', 'United Arab Emirates', 'Singapore']
df['Country'] = df['Country'].apply(lambda x: x if x in Top_countries else (x if x is np.nan else 'Other'))
round(df['Lead Source'].value_counts()/len(df['Lead Source'])*100,1)
Top_source = ['Google' , 'Direct Traffic', 'Olark Chat', 'Organic Search', 'Reference']
df['Lead Source'] = df['Lead Source'].apply(lambda x: x if x in Top_source else (x if x is np.nan else 'Other'))
round(df['Last Activity'].value_counts()/len(df['Last Activity'])*100,1)
Top_activity = ['Email Opened', 'SMS Sent', 'Olark Chat Conversation', 'Page Visited on Website', 'Converted to Lead']
df['Last Activity'] = df['Last Activity'].apply(lambda x: x if x in Top_activity else (x if x is np.nan else 'Other'))
round(df['Specialization'].value_counts()/len(df['Specialization'])*100,1)
Top_specialization = ['Finance Management', 'Human Resource Management', 'Marketing Management', 'Operations Management',

                     'Business Administration', 'IT Projects Management', 'Supply Chain Management', 

                     'Banking, Investment And Insurance', 'Media and Advertising', 'Travel and Tourism',

                     'International Business', 'Healthcare Management']
df['Specialization'] = df['Specialization'].apply(lambda x: x if x in Top_specialization else (x if x is np.nan else 'Other'))
round(df['Tags'].value_counts()/len(df['Tags'])*100,1)
Top_tags = ['Will revert after reading the email', 'Ringing', 'Interested in other courses', 'switched off', 'Busy', 

            'Not doing further education', 'Interested  in full time MBA', 'Graduation in progress']
df['Tags'] = df['Tags'].apply(lambda x: x if x in Top_tags else (x if x is np.nan else 'Other'))
df = pd.get_dummies(df, columns = ['Country', 'Tags', 'Lead Source', 'Specialization', 'Last Activity'])
df.head()
# No need to drop 1 column for these 5 categorical features, since:

# each of them had NaN values initially, and NaN column for each has already been dropped during encoding
df.columns
plt.figure(figsize = (15,6))

sns.heatmap(df.isna())
from sklearn.model_selection import train_test_split
X = df.drop('Converted', axis = 1)

y = df['Converted']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify = y)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
Num_col = ['Page Views Per Visit', 'TotalVisits', 'Total Time Spent on Website']
X_train[Num_col] = scaler.fit_transform(X_train[Num_col])
X_train[Num_col].describe()
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred = logreg.predict_proba(X_train)
from sklearn.metrics import roc_auc_score

from sklearn import metrics
round(roc_auc_score(y_train, y_pred[:, 1]),2)
from sklearn.feature_selection import RFE
import time

start_time = time.time()



AUC_feat = pd.DataFrame()

for i in range(X_train.shape[1]):

    rfe = RFE(logreg, X_train.shape[1]-i)           # running RFE

    rfe = rfe.fit(X_train, y_train)

    col = X_train.columns[rfe.support_]

    logreg.fit(X_train[col], y_train)

    y_pred = logreg.predict_proba(X_train[col])

    AUC_feat = AUC_feat.append({'# of features': X_train.shape[1]-i,

                              'AUC score': round(roc_auc_score(y_train, y_pred[:, 1]),2)}, ignore_index= True)



print("- %s seconds - " % (time.time() - start_time))
plt.figure(figsize = (14,4))

ax = sns.barplot(x = '# of features', y = 'AUC score', data = AUC_feat, palette="GnBu_d")

ax.set_xticklabels(X_train.shape[1]+1-AUC_feat['# of features'].astype(int)) # This line is just to remove decimal points in X_ticks

plt.tight_layout()
AUC_feat.tail(20)
rfe = RFE(logreg, 12)           # running RFE

rfe = rfe.fit(X_train, y_train)
pd.DataFrame(list(zip(X_train.columns, rfe.support_, rfe.ranking_))).sort_values(by = 2)
import statsmodels.api as sm
rfe_col = X_train.columns[rfe.support_]
rfe_col
X_train_rfe = X_train[rfe_col]
logm1 = sm.GLM(y_train,(sm.add_constant(X_train_rfe)), family = sm.families.Binomial())

res = logm1.fit()

res.summary()
from statsmodels.stats.outliers_influence import variance_inflation_factor
# Create a dataframe that will contain the names of all the feature variables and their respective VIFs

vif = pd.DataFrame()

vif['Features'] = X_train[rfe_col].columns

vif['VIF'] = [variance_inflation_factor(X_train[rfe_col].values, i) for i in range(X_train[rfe_col].shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
# If VIFs greater than 5, drop columns 1 by 1

# rfe_col_new = rfe_col.drop('feature_x')
logreg.fit(X_train[rfe_col], y_train)

y_train_pred = logreg.predict_proba(X_train[rfe_col])
round(roc_auc_score(y_train, y_train_pred[:, 1]),2)
y_train_pred_final = pd.DataFrame({'Actual_convert':y_train.values, 'Convert_prob':y_train_pred[:, 1]})

y_train_pred_final['Index'] = y_train.index

y_train_pred_final.head()
y_train_pred_final['Predicted'] = y_train_pred_final['Convert_prob'].apply(lambda x: 1 if x > 0.5 else 0)
confusion = metrics.confusion_matrix(y_train_pred_final['Actual_convert'], y_train_pred_final['Predicted'])

print(confusion)
# Predicted         Not_converted    Converted

# Actual

# Not_converted        3540             76

# Converted            475             1616  
print(round(metrics.accuracy_score(y_train_pred_final['Actual_convert'], y_train_pred_final['Predicted']),3))
plt.figure(figsize = (12,8))

for i, col in enumerate(rfe_col):

    plt.subplot(4,3,i+1)

    sns.barplot(x = col, y = 'Converted', data = df)

plt.tight_layout()
pd.DataFrame(rfe_col)
def draw_roc(actual, probs):

    fpr, tpr, thresholds = metrics.roc_curve(actual, probs,

                                              drop_intermediate = False )

    auc_score = metrics.roc_auc_score(actual, probs)

    plt.figure(figsize=(5, 5))

    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % auc_score )

    plt.plot([0, 1], [0, 1], 'k--')

    plt.xlim([0.0, 1.0])

    plt.ylim([0.0, 1.05])

    plt.xlabel('False Positive Rate or [1 - True Negative Rate]')

    plt.ylabel('True Positive Rate')

    plt.title('Receiver operating characteristic curve')

    plt.legend(loc="lower right")

    plt.show()



    return None
draw_roc(y_train_pred_final.Actual_convert, y_train_pred_final.Convert_prob)
y_train_pred_final.head()
numbers = [float(x)/10 for x in range(10)]

for i in numbers:

    y_train_pred_final[i]= y_train_pred_final.Convert_prob.apply(lambda x: 1 if x > i else 0)

y_train_pred_final.head()
cutoff_df = pd.DataFrame(columns = ['prob','accuracy','sensi','speci'])



# TP = confusion[1,1] # true positive 

# TN = confusion[0,0] # true negatives

# FP = confusion[0,1] # false positives

# FN = confusion[1,0] # false negatives



num = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]

for i in num:

    cm1 = metrics.confusion_matrix(y_train_pred_final.Actual_convert, y_train_pred_final[i] )

    total1=sum(sum(cm1))

    accuracy = (cm1[0,0]+cm1[1,1])/total1    

    speci = cm1[0,0]/(cm1[0,0]+cm1[0,1])

    sensi = cm1[1,1]/(cm1[1,0]+cm1[1,1])

    cutoff_df.loc[i] =[i ,accuracy,sensi,speci]

print(cutoff_df)
cutoff_df.plot.line(x='prob', y=['accuracy','sensi','speci'])

plt.show()
y_train_pred_final['final_predicted'] = y_train_pred_final.Convert_prob.apply(lambda x: 1 if x > 0.4 else 0)

y_train_pred_final.head()
print(metrics.accuracy_score(y_train_pred_final['Actual_convert'], y_train_pred_final['final_predicted']))
confusion = metrics.confusion_matrix(y_train_pred_final['Actual_convert'], y_train_pred_final['final_predicted'])

print(confusion)
# Predicted         Not_converted    Converted

# Actual

# Not_converted        3379             237

# Converted            288             1803  
from sklearn.metrics import precision_score, recall_score
round(precision_score(y_train_pred_final.Actual_convert, y_train_pred_final.final_predicted),2)
round(recall_score(y_train_pred_final.Actual_convert, y_train_pred_final.final_predicted),2)
X_test[Num_col] = scaler.transform(X_test[Num_col])
X_test = X_test[rfe_col]
y_test_pred = logreg.predict_proba(X_test)
round(roc_auc_score(y_test, y_test_pred[:, 1]),2)
y_test_final = pd.DataFrame()
y_test_final[['index', 'Converted']] = y_test.reset_index()[['index', 'Converted']]
y_test_final['Probability'] = y_test_pred[:,1]
y_test_final.head()
y_test_final['final_pred'] = y_test_final['Probability'].apply(lambda x: 1 if x>0.4 else 0)
y_test_final.head()
metrics.accuracy_score(y_test_final.Converted, y_test_final.final_pred)
confusion_test = metrics.confusion_matrix(y_test_final.Converted, y_test_final.final_pred)

confusion_test
TP = confusion_test[1,1] # true positive 

TN = confusion_test[0,0] # true negatives

FP = confusion_test[0,1] # false positives

FN = confusion_test[1,0] # false negatives
# Let's see the sensitivity of our logistic regression model

TP / float(TP+FN)
# Let us calculate specificity

TN / float(TN+FP)