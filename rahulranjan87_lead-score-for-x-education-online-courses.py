# Importing Pandas and NumPy

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



from matplotlib.pyplot import xticks

%matplotlib inline



import warnings

warnings.filterwarnings('ignore')
# Importing  datasets

leads_orig_data = pd.read_csv('../input/leads-dataset/Leads.csv')
# Let's see the head of our dataset

pd.set_option("display.max_columns", 500)

leads_orig_data.head()
leads_orig_data.shape
leads_orig_data.info()
leads_data = leads_orig_data
# Converting Yes to 1 and NO to 0

leads_data['Do Not Email'] = leads_data['Do Not Email'].map({'Yes': 1, 'No': 0})

leads_data['Do Not Call'] = leads_data['Do Not Call'].map({'Yes': 1, 'No': 0})



leads_data['Search'] = leads_data['Search'].map({'Yes': 1, 'No': 0})

leads_data['Magazine'] = leads_data['Magazine'].map({'Yes': 1, 'No': 0})

leads_data['Newspaper Article'] = leads_data['Newspaper Article'].map({'Yes': 1, 'No': 0})

leads_data['X Education Forums'] = leads_data['X Education Forums'].map({'Yes': 1, 'No': 0})

leads_data['Newspaper'] = leads_data['Newspaper'].map({'Yes': 1, 'No': 0})

leads_data['Digital Advertisement'] = leads_data['Digital Advertisement'].map({'Yes': 1, 'No': 0})

leads_data['Through Recommendations'] = leads_data['Through Recommendations'].map({'Yes': 1, 'No': 0})

leads_data['Receive More Updates About Our Courses'] = leads_data['Receive More Updates About Our Courses'].map({'Yes': 1, 'No': 0})



leads_data['Update me on Supply Chain Content'] = leads_data['Update me on Supply Chain Content'].map({'Yes': 1, 'No': 0})

leads_data['Get updates on DM Content'] = leads_data['Get updates on DM Content'].map({'Yes': 1, 'No': 0})

leads_data['I agree to pay the amount through cheque'] = leads_data['I agree to pay the amount through cheque'].map({'Yes': 1, 'No': 0})

leads_data['A free copy of Mastering The Interview'] = leads_data['A free copy of Mastering The Interview'].map({'Yes': 1, 'No': 0})

# Deriving Asymmetrique Activity Index to numerical



leads_data['Asymmetrique Activity Index']=leads_data['Asymmetrique Activity Index'].str.split('.',n = 1, expand = True)[0].astype(float)

leads_data['Asymmetrique Profile Index']=leads_data['Asymmetrique Profile Index'].str.split('.',n = 1, expand = True)[0].astype(float)
# changing the case of all column values to lower case

for col in leads_data.columns:

    leads_data[col] = leads_data[col].apply(lambda s: s.lower() if type(s)==str else s)
# divide all features to numerical and categorical for creating dummies

leads_data_col=list(leads_data.columns)



#'Lead Number'and 'Prospect ID' can be dropped from the list of features

leads_data_col.remove('Lead Number')

leads_data_col.remove('Prospect ID')



# divide all features to numerical and categorical

leads_data_col_num= [x for x in leads_data_col if leads_data[x].dtype in ['float64','int64']]

leads_data_col_cat= [x for x in leads_data_col if leads_data[x].dtype=='object']



#'Lead Number'and 'Prospect ID' can be dropped from the list of features

print(leads_data_col_num)

print(leads_data_col_cat)
# Checking the missing values in Numerical Columns

leads_data_num_nulls=leads_data[leads_data_col_num].isnull().any()

leads_data_num_nulls_cols = list(leads_data_num_nulls[leads_data_num_nulls.values==True].index)

leads_data_num_nulls_cols
leads_data[leads_data_num_nulls_cols]=leads_data[leads_data_num_nulls_cols].fillna(0)

leads_data.shape
# Checking the missing values in Categorical Columns

leads_data_cat_nulls=leads_data[leads_data_col_cat].isnull().any()

leads_data_cat_nulls_cols = list(leads_data_cat_nulls[leads_data_cat_nulls.values==True].index)

leads_data_cat_nulls_cols
# 'Lead Source' impute 'unknown'

leads_data['Lead Source']=leads_data['Lead Source'].fillna('unknown')



fig, axs = plt.subplots(figsize = (15,5))

sns.countplot(x = 'Lead Source', hue = 'Converted', data = leads_data)

xticks(rotation = 90)

# 'Last Activity' impute 'unknown'

leads_data['Last Activity']=leads_data['Last Activity'].fillna('unknown')



fig, axs = plt.subplots(figsize = (15,5))

sns.countplot(x = 'Last Activity', hue = 'Converted', data = leads_data)

xticks(rotation = 90)
# 'Country' impute mode 'india'

leads_data['Country']=leads_data['Country'].fillna('india')



fig, axs = plt.subplots(figsize = (15,5))

sns.countplot(x = 'Country', hue = 'Converted', data = leads_data)

xticks(rotation = 90)
# 'Specialization' impute 'unknown' as if leads has not selected means they don't have one

leads_data['Specialization']=leads_data['Specialization'].fillna('unknown')



# 'Specialization' also impute 'select' with unknown' as they are as good as null

leads_data['Specialization']=leads_data['Specialization'].apply(lambda x : 'unknown' if x=='select' else x)



fig, axs = plt.subplots(figsize = (15,5))

sns.countplot(x = 'Specialization', hue = 'Converted', data = leads_data)

xticks(rotation = 90)
# 'How did you hear about X Education' impute 'other' as if leads has not selected means it is not in options

leads_data['How did you hear about X Education']=leads_data['How did you hear about X Education'].fillna('other')



# 'How did you hear about X Education' also impute 'select' with unknown' as they are as good as null

leads_data['How did you hear about X Education']=leads_data['How did you hear about X Education'].apply(lambda x : 'other' if x=='select' else x)



fig, axs = plt.subplots(figsize = (15,5))

sns.countplot(x = 'How did you hear about X Education', hue = 'Converted', data = leads_data)

xticks(rotation = 90)
# 'What is your current occupation' impute 'other' as if leads has not selected means it is not in options

leads_data['What is your current occupation']=leads_data['What is your current occupation'].fillna('other')



fig, axs = plt.subplots(figsize = (15,5))

sns.countplot(x = 'What is your current occupation', hue = 'Converted', data = leads_data)

xticks(rotation = 90)
# 'What matters most to you in choosing a course' impute 'other' as if leads has not selected means it is not in options

leads_data['What matters most to you in choosing a course']=leads_data['What matters most to you in choosing a course'].fillna('other')



fig, axs = plt.subplots(figsize = (15,5))

sns.countplot(x = 'What matters most to you in choosing a course', hue = 'Converted', data = leads_data)

xticks(rotation = 90)
# 'Tags' impute 'unknown' as if nothing is tagged to leads

leads_data['Tags']=leads_data['Tags'].fillna('unknown')



fig, axs = plt.subplots(figsize = (15,5))

sns.countplot(x = 'Tags', hue = 'Converted', data = leads_data)

xticks(rotation = 90)
#Lead Quality impute 'not sure' as if nothing is selected for leads 

leads_data['Lead Quality']=leads_data['Lead Quality'].fillna('not sure')



fig, axs = plt.subplots(figsize = (15,5))

sns.countplot(x = 'Lead Quality', hue = 'Converted', data = leads_data)

xticks(rotation = 90)
# 'Lead Profile' impute 'other leads' as if nothing is selected for leads 

leads_data['Lead Profile']=leads_data['Lead Profile'].fillna('other leads')



# 'Lead Profile' also impute 'select' with 'other leads' as they are as good as null

leads_data['Lead Profile']=leads_data['Lead Profile'].apply(lambda x : 'other leads' if x=='select' else x)



fig, axs = plt.subplots(figsize = (15,5))

sns.countplot(x = 'Lead Profile', hue = 'Converted', data = leads_data)

xticks(rotation = 90)
# 'City' impute 'other cities' as leads does not select city available in the list

leads_data['City']=leads_data['City'].fillna('other cities')



# 'City' also impute 'select' with 'other cities' as they are as good as null

leads_data['City']=leads_data['City'].apply(lambda x : 'other cities' if x=='select' else x)



fig, axs = plt.subplots(figsize = (15,5))

sns.countplot(x = 'City', hue = 'Converted', data = leads_data)

xticks(rotation = 90)
# Checking missing values if any after removing the missing values

leads_data.isnull().any().sum()
# Checking outliers at 25%,50%,75%,90%,95% and 99%

leads_data[leads_data_col_num].describe(percentiles=[.25,.5,.75,.90,.95,.99])
# box plot above variable

plt.figure(figsize=(20, 50))

j=0

for i in ['TotalVisits' , 'Total Time Spent on Website', 'Page Views Per Visit']:

    j=j+1

    plt.subplot(5,3,j)

    sns.boxplot(data=leads_data, y=i, x='Converted')

    

plt.show()
leads_data=leads_data[leads_data['TotalVisits'] < 20]

leads_data=leads_data[leads_data['Page Views Per Visit'] < 200]
# Creating a dummy variable for 'Lead Origin' and dropping the first one.

cont = pd.get_dummies(leads_data['Lead Origin'],prefix='Lead Origin',drop_first=True)

#Adding the results to the master dataframe

leads_data = pd.concat([leads_data,cont],axis=1)



# Creating a dummy variable for 'Lead Source' and dropping the first one.

cont = pd.get_dummies(leads_data['Lead Source'],prefix='Lead Source',drop_first=True)

#Adding the results to the master dataframe

leads_data = pd.concat([leads_data,cont],axis=1)



# Creating a dummy variable for 'Last Activity' and dropping the first one.

cont = pd.get_dummies(leads_data['Last Activity'],prefix='Last Activity',drop_first=True)

#Adding the results to the master dataframe

leads_data = pd.concat([leads_data,cont],axis=1)



# Creating a dummy variable for 'Country' and dropping the first one.

cont = pd.get_dummies(leads_data['Country'],prefix='Country',drop_first=True)

#Adding the results to the master dataframe

leads_data = pd.concat([leads_data,cont],axis=1)



# Creating a dummy variable for 'Specialization' and dropping the first one.

cont = pd.get_dummies(leads_data['Specialization'],prefix='Specialization',drop_first=True)

#Adding the results to the master dataframe

leads_data = pd.concat([leads_data,cont],axis=1)



# Creating a dummy variable for 'How did you hear about X Education and dropping the first one.

cont = pd.get_dummies(leads_data['How did you hear about X Education'],prefix='How did you hear about X Education',drop_first=True)

#Adding the results to the master dataframe

leads_data = pd.concat([leads_data,cont],axis=1)



# Creating a dummy variable for 'What is your current occupation' and dropping the first one.

cont = pd.get_dummies(leads_data['What is your current occupation'],prefix='What is your current occupation',drop_first=True)

#Adding the results to the master dataframe

leads_data = pd.concat([leads_data,cont],axis=1)



# Creating a dummy variable for 'What matters most to you in choosing a course' and dropping the first one.

cont = pd.get_dummies(leads_data['What matters most to you in choosing a course'],prefix='What matters most to you in choosing a course',drop_first=True)

#Adding the results to the master dataframe

leads_data = pd.concat([leads_data,cont],axis=1)



# Creating a dummy variable for 'Tags' and dropping the first one.

cont = pd.get_dummies(leads_data['Tags'],prefix='Tags',drop_first=True)

#Adding the results to the master dataframe

leads_data = pd.concat([leads_data,cont],axis=1)



# Creating a dummy variable for 'Lead Quality' and dropping the first one.

cont = pd.get_dummies(leads_data['Lead Quality'],prefix='Lead Quality',drop_first=True)

#Adding the results to the master dataframe

leads_data = pd.concat([leads_data,cont],axis=1)



# Creating a dummy variable for 'Lead Profile' and dropping the first one.

cont = pd.get_dummies(leads_data['Lead Profile'],prefix='Lead Profile',drop_first=True)

#Adding the results to the master dataframe

leads_data = pd.concat([leads_data,cont],axis=1)



# Creating a dummy variable for 'City' and dropping the first one.

cont = pd.get_dummies(leads_data['City'],prefix='City',drop_first=True)

#Adding the results to the master dataframe

leads_data = pd.concat([leads_data,cont],axis=1)



# Creating a dummy variable for 'Last Notable Activity' and dropping the first one.

cont = pd.get_dummies(leads_data['Last Notable Activity'],prefix='Last Notable Activity',drop_first=True)

#Adding the results to the master dataframe

leads_data = pd.concat([leads_data,cont],axis=1)



# We have created dummies for the below variables, so we can drop them

leads_data = leads_data.drop(leads_data_col_cat, 1)



leads_data .head()
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()



numvars = ['TotalVisits' , 'Total Time Spent on Website', 'Page Views Per Visit','Asymmetrique Activity Index','Asymmetrique Profile Index','Asymmetrique Activity Score','Asymmetrique Profile Score']



leads_data[numvars] = scaler.fit_transform(leads_data[numvars])

leads_data.head()
score = (sum(leads_data['Converted'])/len(leads_data['Converted'].index))*100

score
# make Lead Number as index before split

leads_data= leads_data.set_index('Lead Number')



# test train split

from sklearn.model_selection import train_test_split



# Putting feature variable to X

X = leads_data.drop(['Converted','Prospect ID'],axis=1)



# Putting response variable to y

y = leads_data['Converted']



y.head()
# Splitting the data into train and test

X_train, X_test, y_train, y_test = train_test_split(X,y, train_size=0.7,test_size=0.3,random_state=100)

X_train.shape
# Checking if all values is same for dummy columns after split

uniques= X_train.loc[:,X_train.nunique()==1]



# remove those columns

col_drop = uniques.columns

for col in col_drop:

    X_train = X_train.drop([col], axis = 1)

    X_test = X_test.drop([col], axis = 1)

    

print(X_train.shape)

print(X_test.shape)
X_train.columns
import statsmodels.api as sm

logm1 = sm.GLM(y_train,(sm.add_constant(X_train)), family = sm.families.Binomial())

logm1.fit().summary()
# As there are 170 features manual selection of features is not possible. Hence running RFE with 15 variables as output

from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()

from sklearn.feature_selection import RFE

rfe = RFE(logreg, 15)   

rfe = rfe.fit(X_train,y_train)
rfe_col = X_train.columns[rfe.support_]

rfe_result = pd.DataFrame(list(zip(X_train.columns,rfe.support_,rfe.ranking_)))

rfe_result = rfe_result.rename(columns={ 0 : 'Feature',1 : 'Selection', 2 : 'Ranking'})

rfe_20 = rfe_result[rfe_result['Selection']==True]

rfe_20.sort_values(by='Ranking')

rfe_20
# Let's see the correlation matrix 

plt.figure(figsize = (20,10))        # Size of the figure

sns.heatmap(X_train[rfe_col].corr(),annot = True)
rfe_col = rfe_col.drop(['What matters most to you in choosing a course_other', 'Asymmetrique Activity Score'])

X_test2 = X_test[rfe_col]

X_train2 = X_train[rfe_col]
logm2 = sm.GLM(y_train,(sm.add_constant(X_train2)), family = sm.families.Binomial())

logm2.fit().summary()
# Drop 'Tags_wrong number given' due to high p value

rfe_col = rfe_col.drop(['Tags_wrong number given'])

X_test3 = X_test[rfe_col]

X_train3 = X_train[rfe_col]
logm3 = sm.GLM(y_train,(sm.add_constant(X_train3)), family = sm.families.Binomial())

model =logm3.fit()

model.summary()
# see the VIF 

from statsmodels.stats.outliers_influence import variance_inflation_factor



vif = pd.DataFrame()

vif['Features'] = X_train3.columns

vif['VIF'] = [variance_inflation_factor(X_train3.values, i) for i in range(X_train3.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
# get p values and columns selected in data frame and sort by p values to get top 3 variables

pvalues_df= pd.DataFrame(model.pvalues)

pvalues_df = pvalues_df.rename(columns={ 0 : 'pvalues'}) 

pvalues_df = pvalues_df.sort_values(by='pvalues')

pvalues_df
# Let's run the model using the selected variables

from sklearn.linear_model import LogisticRegression

from sklearn import metrics

logsk = LogisticRegression(C=1e9)

logsk.fit(X_train3, y_train)
y_pred = logsk.predict_proba(X_train3)



# Converting pred_probs_test to a dataframe which is an array

y_pred_df = pd.DataFrame(y_pred)

y_pred_1 = y_pred_df.iloc[:,[1]]

y_pred_1.head()
# Converting y_test to dataframe

y_train_df = pd.DataFrame(y_train)

y_train_df['Lead Number']=y_train_df.index

y_train_df.reset_index(drop=True, inplace=True)

y_train_df.head()
# Appending y_test_df and y_pred_1

y_pred_final = pd.concat([y_train_df,y_pred_1],axis=1)

# Renaming the column 

y_pred_final= y_pred_final.rename(columns={ 1 : 'Lead_Score'})

# Rearranging the columns

y_pred_final = y_pred_final.reindex(['Lead Number','Converted','Lead_Score'], axis=1)

# Let's see the head of y_pred_final

y_pred_final.head()
# Creating new column 'predicted' with 1 if lead_score>0.8 else 0

y_pred_final['predicted'] = y_pred_final.Lead_Score.map( lambda x: 1 if x > 0.8 else 0)

# Let's see the head

y_pred_final.head()
from sklearn import metrics
#create columns with different probability cutoffs

numbers = [float(x)/10 for x in range(10)]

for i in numbers:

    y_pred_final[i]= y_pred_final.Lead_Score.map(lambda x: 1 if x > i else 0)

y_pred_final.head()
# calculate accuracy sensitivity and specificity for various probability cutoffs.

cutoff_df = pd.DataFrame( columns = ['prob','accuracy','sensi','speci'])

from sklearn.metrics import confusion_matrix



# TP = confusion[1,1] # true positive 

# TN = confusion[0,0] # true negatives

# FP = confusion[0,1] # false positives

# FN = confusion[1,0] # false negatives



num = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]

for i in num:

    cm1 = metrics.confusion_matrix(y_pred_final.Converted, y_pred_final[i] )

    total1=sum(sum(cm1))

    accuracy = (cm1[0,0]+cm1[1,1])/total1

    

    speci = cm1[0,0]/(cm1[0,0]+cm1[0,1])

    sensi = cm1[1,1]/(cm1[1,0]+cm1[1,1])

    cutoff_df.loc[i] =[ i ,accuracy,sensi,speci]

print(cutoff_df)
#plot accuracy sensitivity and specificity for various probabilities.

ax = cutoff_df.plot.line(x='prob', y=['accuracy','sensi','speci'])

ax.vlines(x=0.32, ymax=1, ymin=0, colors="r", linestyles="--")
# overwrite the previous prediction using new cut-off

y_pred_final['predicted'] = y_pred_final.Lead_Score.map( lambda x: 1 if x > 0.32 else 0)

# Let's see the head

y_pred_final.head()
# Confusion matrix 

confusion = metrics.confusion_matrix( y_pred_final.Converted, y_pred_final.predicted )

confusion
# overall accuracy.

metrics.accuracy_score(y_pred_final.Converted, y_pred_final.predicted)
TP = confusion[1,1] # true positive 

TN = confusion[0,0] # true negatives

FP = confusion[0,1] # false positives

FN = confusion[1,0] # false negatives
# sensitivity of our logistic regression model

TP / float(TP+FN)
# specificity

TN / float(TN+FP)
#false postive rate - predicting Conversion when customer does not have Converted

print(FP/ float(TN+FP))
# positive predictive value 

print (TP / float(TP+FP))
# Negative predictive value

print (TN / float(TN+ FN))
def draw_roc( actual, probs ):

    fpr, tpr, thresholds = metrics.roc_curve( actual, probs,

                                              drop_intermediate = False )

    auc_score = metrics.roc_auc_score( actual, probs )

    plt.figure(figsize=(6, 6))

    plt.plot( fpr, tpr, label='ROC curve (area = %0.2f)' % auc_score )

    plt.plot([0, 1], [0, 1], 'k--')

    plt.xlim([0.0, 1.0])

    plt.ylim([0.0, 1.05])

    plt.xlabel('False Positive Rate or [1 - True Negative Rate]')

    plt.ylabel('True Positive Rate')

    plt.title('Receiver operating characteristic example')

    plt.legend(loc="lower right")

    plt.show()



    return fpr, tpr, thresholds
draw_roc(y_pred_final.Converted, y_pred_final.predicted)
#draw_roc(y_pred_final.Converted, y_pred_final.predicted)

"{:2.2f}".format(metrics.roc_auc_score(y_pred_final.Converted, y_pred_final.Lead_Score))
from sklearn.metrics import precision_score, recall_score





print(precision_score(y_pred_final.Converted , y_pred_final.predicted))

print(recall_score(y_pred_final.Converted, y_pred_final.predicted))
from sklearn.metrics import precision_recall_curve

p, r, thresholds = precision_recall_curve(y_pred_final.Converted, y_pred_final.Lead_Score)



plt.plot(thresholds, p[:-1], "g-")

plt.plot(thresholds, r[:-1], "r-")

plt.vlines(x=0.37, ymax=1, ymin=0, colors="b", linestyles="--")

plt.show()
y_pred = logsk.predict_proba(X_test3)



# Converting pred_probs_test to a dataframe which is an array

y_pred_df = pd.DataFrame(y_pred)

y_pred_1 = y_pred_df.iloc[:,[1]]

y_pred_1.head()
# Converting y_test to dataframe

y_test_df = pd.DataFrame(y_test)

y_test_df['Lead Number']=y_test_df.index

y_test_df.reset_index(drop=True, inplace=True)

y_test_df.head()
# Appending y_test_df and y_pred_1

y_pred_test_final = pd.concat([y_test_df,y_pred_1],axis=1)

# Renaming the column 

y_pred_test_final= y_pred_test_final.rename(columns={ 1 : 'Lead_Score'})

# Rearranging the columns

y_pred_test_final = y_pred_test_final.reindex(['Lead Number','Converted','Lead_Score'], axis=1)

# Let's see the head of y_pred_final

y_pred_test_final.head()
# Creating new column 'predicted' with 1 if lead_score>0.8 else 0

y_pred_test_final['predicted'] = y_pred_test_final.Lead_Score.map( lambda x: 1 if x > 0.37 else 0)

# Let's see the head

y_pred_test_final.head()
from sklearn import metrics
# Confusion matrix 

confusion = metrics.confusion_matrix( y_pred_test_final.Converted, y_pred_test_final.predicted )

confusion
# overall accuracy.

metrics.accuracy_score(y_pred_test_final.Converted, y_pred_test_final.predicted)
draw_roc(y_pred_test_final.Converted, y_pred_test_final.predicted)
#draw_roc(y_pred_final.Converted, y_pred_final.predicted)

"{:2.2f}".format(metrics.roc_auc_score(y_pred_test_final.Converted, y_pred_test_final.Lead_Score))
a = y_pred_test_final[['Lead Number','Lead_Score', 'predicted']]

b = y_pred_final[['Lead Number','Lead_Score', 'predicted']]

y_predicted = pd.concat([a,b], axis=0)

y_predicted['Lead_Score']= y_predicted['Lead_Score'].apply(lambda x : round(x*100,2))

y_predicted.head()
leads_scored_data=leads_orig_data.merge(y_predicted, how='inner', on='Lead Number')

leads_scored_data = leads_scored_data.sort_values(by='Lead_Score', ascending=False)

leads_scored_data.head()
hot_leads = leads_scored_data[leads_scored_data['Lead_Score']>=18]

hot_leads.shape
score = (sum(hot_leads['Converted'])/len(hot_leads['Converted'].index))*100

score
# columns used for prediction

rfe_col
#  hot_leads against columns 'Lead Source', 'Last Activity', 'Tags', 'Lead_Quality', 'Last Notable Activity'



hot_leads_ordered0 = hot_leads[(hot_leads['Tags'] =='will revert after reading the email')

                              &(hot_leads['Last Activity']== 'sms sent')]



hot_leads_ordered1 = hot_leads[(hot_leads['Last Notable Activity']!='modified') |

                               (hot_leads['Lead Quality']!='worst') |

                               ((hot_leads['Tags']!='invalid number') |

                                (hot_leads['Tags']!='ringing') |

                                (hot_leads['Tags']!='switched off'))]



print(hot_leads_ordered0.shape)

print(hot_leads_ordered1.shape)