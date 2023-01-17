# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
from __future__ import print_function

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import matplotlib.ticker as ticker

from matplotlib.pyplot import xticks

%matplotlib inline

sns.set(style="whitegrid")

sns.set(rc={'figure.figsize':(12,8)})

pd.set_option('display.max_rows', 50)

pd.set_option('display.max_columns', 40)

pd.set_option('display.max_colwidth', -1)



from ipywidgets import interact, interactive, fixed, interact_manual

import ipywidgets as widgets
#function for missing values in columns

def missing_coldata(df):

    missin_col = pd.DataFrame(round(df.isnull().sum().sort_values(ascending=False)/len(df.index)*100,1), columns=['% of missing value'])

    missin_col['Count of Missing Values'] = df.isnull().sum()

    return missin_col



#function for missing values in rows

def missing_rowdata(df):

    missin_row = pd.DataFrame(round(df.isnull().sum(axis=1).sort_values(ascending=False)/len(df.columns)*100), columns=['% of missing value'])

    missin_row['Count of Missing Values'] = df.isnull().sum(axis=1)

    return missin_row
leadsdata = pd.read_csv('../input/leads-dataset/Leads.csv')

leadsdata.head(5) 
leadsdata.shape
leadsdata.info()
leadsdata.isnull().sum()
dupcheck=leadsdata[leadsdata.duplicated(["Prospect ID"])]

dupcheck
sum(leadsdata.duplicated('Prospect ID')) == 0
sum(leadsdata.duplicated('Lead Number')) == 0
leadsdata.nunique()
Conversion_rate = (sum(leadsdata['Converted'])/len(leadsdata['Converted'].index))*100

print("The conversion rate of leads is: ",Conversion_rate)
# Divide the data into Numeric and categorical data  

numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

# NUMERIC

numdata=leadsdata[list(leadsdata.select_dtypes(numerics).columns)]

# CATEGORICAL 

catdata=leadsdata[list(leadsdata.select_dtypes(exclude=numerics).columns)]

catdata.columns
# Conversion rate for each categorical feature

@interact

def counts(col =catdata.iloc[:,1:].columns):

    sns.countplot(x=col,data=leadsdata,hue="Converted",palette="husl",hue_order=[0,1])

    plt.xlabel(col)

    plt.ylabel('Total count')

    plt.legend(loc='upper center', bbox_to_anchor=(1, 0.8), ncol=1)

    plt.xticks(rotation=65, horizontalalignment='right',fontweight='light')

    convertcount=leadsdata.pivot_table(values='Lead Number',index=col,columns='Converted', aggfunc='count').fillna(0)

    convertcount["Conversion(%)"] =round(convertcount[1]/(convertcount[0]+convertcount[1]),2)*100

    return print(convertcount.sort_values(ascending=False,by=1),plt.show())
@interact

def described(col=leadsdata.iloc[:,2:].columns):

    return leadsdata[col].describe()
#Choosing to drop the columns that have only 1 unique value

leadsdata=leadsdata.drop(["Receive More Updates About Our Courses","Magazine","Update me on Supply Chain Content","Get updates on DM Content","I agree to pay the amount through cheque"],axis=1)
leadsdata=leadsdata.drop(["Newspaper","X Education Forums","Newspaper Article",

                          "Through Recommendations","Digital Advertisement",

                          "What matters most to you in choosing a course","Search","Do Not Call"],axis=1)
missing_coldata(leadsdata)
leadsdata["Lead Source"]=leadsdata["Lead Source"].fillna("Google")
# What is your current occupation

leadsdata["What is your current occupation"]=leadsdata["What is your current occupation"].fillna("Unemployed")
# Also the missing values can be imputed with Any_Other

leadsdata["Specialization"]=leadsdata["Specialization"].replace("Select","Any_Other")

leadsdata["Specialization"]=leadsdata["Specialization"].fillna("Any_Other")
leadsdata["How did you hear about X Education"]=leadsdata["How did you hear about X Education"].replace("Select","Not_Mentioned")

leadsdata["How did you hear about X Education"]=leadsdata["How did you hear about X Education"].fillna("Not_Mentioned")
leadsdata["Lead Profile"]=leadsdata["Lead Profile"].replace("Select","Any_Other")

leadsdata["Lead Profile"]=leadsdata["Lead Profile"].fillna("Any_other")
leadsdata["Lead Quality"]=leadsdata["Lead Quality"].replace("Select","Might be")

leadsdata["Lead Quality"]=leadsdata["Lead Quality"].fillna("Might be")
# Tags

leadsdata["Tags"]=leadsdata["Tags"].fillna("Will revert after reading the email")
leadsdata["Country"]=leadsdata["Country"].fillna("India")
leadsdata["City"]=leadsdata["City"].fillna("Mumbai")
leadsdata.shape
missing_coldata(leadsdata)
numdata.columns
@interact

def density( y=numdata.iloc[:,2:].columns,tick_spacing = [100,50,25,10,5]):

    ax=leadsdata[y].plot(kind="hist",title=y,bins=50, rot=30)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))

    return
@interact

def outliers_check( y=numdata.iloc[:,2:].columns):

    return leadsdata.plot(kind='box',y=y,figsize=[6,5]) 
leadsdata.drop(['Asymmetrique Activity Score', 'Asymmetrique Profile Score','Asymmetrique Activity Index','Asymmetrique Profile Index'],axis=1,inplace=True)
missing_coldata(leadsdata)
leadsdata.dropna(inplace = True)
missing_coldata(leadsdata)
leadsdata.shape
leadsdata['Tags'] = leadsdata['Tags'].replace(['In confusion whether part time or DLP', 'in touch with EINS','Diploma holder (Not Eligible)',

                                     'Approached upfront','Graduation in progress','number not provided', 'opp hangup','Still Thinking',

                                    'Lost to Others','Shall take in the next coming month','Lateral student','Interested in Next batch',

                                    'Recognition issue (DEC approval)','Want to take admission but has financial problems',

                                    'University not recognized'], 'Misc_Tags')
leadsdata['Tags'] = leadsdata['Tags'].replace(["Ringing","Misc_Tags","Interested in other courses","switched off","Already a student",

                                               "Interested  in full time MBA","Not doing further education","invalid number","wrong number given"], 'Misc_Tags')                                  

leadsdata['Tags'] = leadsdata['Tags'].replace(["Ringing","Misc_Tags","Interested in other courses","switched off","Already a student",

                                               "Interested  in full time MBA","Not doing further education","invalid number","wrong number given"], 'Misc_Tags')                                  

leadsdata["Last Notable Activity"] = leadsdata["Last Notable Activity"].replace(['Approached upfront',

       'Resubscribed to emails', 'View in browser link Clicked',

       'Form Submitted on Website', 'Email Received', 'Email Marked Spam'], 'Misc_Notable_Activity')
leadsdata['Lead Source'] = leadsdata['Lead Source'].replace(['Click2call', 'Live Chat', 'NC_EDM', 'Pay per Click Ads', 'Press_Release',

  'Social Media', 'WeLearn', 'bing', 'blog', 'testone', 'welearnblog_Home', 'youtubechannel'], 'Miscellaneous')
#There are two google in lead source which should be corrected to one

leadsdata['Lead Source'] = leadsdata['Lead Source'].replace('google',"Google")

# As we can see that There are various categories in Last Activity which have very few records, thus combining all those to one category Miscellaneous

leadsdata['Last Activity'] = leadsdata['Last Activity'].replace(['Had a Phone Conversation', 'View in browser link Clicked', 

                                                       'Visited Booth in Tradeshow', 'Approached upfront',

                                                       'Resubscribed to emails','Email Received', 'Email Marked Spam'], 'Miscellaneous')
# Total Visit

(leadsdata["TotalVisits"]>=30).sum()
# removed outliers with values greater than 30



leadsdata=leadsdata[leadsdata["TotalVisits"] < 30]
# Page view per visit

# can easily remove these two outliers

(leadsdata["Page Views Per Visit"]>=15).sum()
#### removed outliers with values greater than 15

leadsdata=leadsdata[leadsdata["Page Views Per Visit"]<15]
# dataframe is sliced for more than one hour time spent on website

leads1hrplus=leadsdata[leadsdata['Total Time Spent on Website']>=60]

leads1hrplus["hours spent"]=round(leads1hrplus["Total Time Spent on Website"]/60).astype(int)
time_spent_abv1hr=leads1hrplus.pivot_table(values='Lead Number',index=['hours spent'],columns='Converted', aggfunc='count').fillna(0)

time_spent_abv1hr["Conversion(%)"] =round(time_spent_abv1hr[1]/(time_spent_abv1hr[0]+time_spent_abv1hr[1]),2)*100

time_spent_abv1hr.sort_values(ascending=False,by=1)
time_spent_abv1hr.iloc[:,:-1].plot(kind='bar',title= "Conversion count for the leads that spend at least 1 hour on website",stacked=True,figsize=[8,6])
leadslessthan1hr=leadsdata[leadsdata['Total Time Spent on Website']<60]

leadslessthan1hr["mins_spent"]=leadslessthan1hr["Total Time Spent on Website"].astype(int)
time_spent_upto1hr=leadslessthan1hr.pivot_table(values='Lead Number',index=['mins_spent'],columns='Converted', aggfunc='count').fillna(0)

time_spent_upto1hr["Conversion(%)"] =round(time_spent_upto1hr[1]/(time_spent_upto1hr[0]+time_spent_upto1hr[1]),2)*100

time_spent_upto1hr.sort_values(ascending=False,by="Conversion(%)")
time_spent_upto1hr.iloc[:,:-1].plot(kind='bar',title="Conversion count for the leads that spend atmost 1 hour on website",stacked=True,figsize=[10,8],log=True)
@interact

def numcount(cols=['TotalVisits','Asymmetrique Activity Score', 'Asymmetrique Profile Score']):

    numdfcount=round(leadsdata.pivot_table(values='Lead Number',index=cols,columns='Converted', aggfunc='count')).fillna(0)

    numdfcount["Conversion(%)"]=round((numdfcount[1]/(numdfcount[0]+numdfcount[1]))*100)

    cnplot=numdfcount.iloc[:,:-1].plot(kind="bar",stacked=True, legend="upper right", title=cols,figsize=[8,6])

    return print(numdfcount, "\n", cnplot)
pageview=leadsdata.pivot_table(values='Lead Number',index=['Page Views Per Visit'],columns='Converted', aggfunc='count')

pageview.reset_index(inplace=True)
pageview.fillna(0,inplace=True)
pageviews=pageview.round().groupby("Page Views Per Visit").sum()

pageviews["Conversion(%)"]=round((pageviews[1]/(pageviews[0]+pageviews[1]))*100)
pageviews.iloc[:,:-1].plot(kind="bar",legend="upper right",stacked=True,figsize=[7,5])

pageviews
# there are two unique keys for the data, hence dropping Prospect ID for now n keeping Lead Number.

leadsdata=leadsdata.drop("Prospect ID",axis=1)
leadsdata.nunique()
leadsdata.drop(['Country'],axis=1,inplace=True)
leadsdata.columns
catdata=leadsdata[list(leadsdata.select_dtypes(exclude=numerics).columns)]

catdata.columns
@interact

def counts(col =catdata.columns):

    sns.countplot(x=col,data=leadsdata,hue="Converted",palette="husl",hue_order=[0,1])

    plt.xlabel(col)

    plt.ylabel('Total count')

    plt.legend(loc='upper center', bbox_to_anchor=(1, 0.8), ncol=1)

    plt.xticks(rotation=65, horizontalalignment='right',fontweight='light')

    convertcount=leadsdata.pivot_table(values='Lead Number',index=col,columns='Converted', aggfunc='count').fillna(0)

    convertcount["Conversion(%)"] =round(convertcount[1]/(convertcount[0]+convertcount[1]),2)*100

    return print(convertcount.sort_values(ascending=False,by=1),plt.show())
catdata.nunique()
df = pd.get_dummies(leadsdata[catdata.columns], drop_first=True)

df.head()
#Create a copy of leads data to add these dummies to the whole data

leads_copy = leadsdata.copy(deep=True)
leads = leadsdata.drop(catdata.columns, axis = 1)
leads.columns
leads = pd.concat([leads, df], axis=1)
leads.shape
%matplotlib inline

plt.figure(figsize = (10,6))

sns.heatmap(leadsdata.corr(),annot = True)
# Total Visits and Page Views Per Visit are significantly correlated, hence we drop one of those

leads = leads.drop("Page Views Per Visit", axis = 1)
leads.columns
from sklearn.preprocessing import MinMaxScaler

scaler =  MinMaxScaler()

leads[['TotalVisits','Total Time Spent on Website']] = scaler.fit_transform(leads[['TotalVisits','Total Time Spent on Website']])

leads.head()
from sklearn.model_selection import train_test_split

# Creating target variable as y and remaining as X

X = leads.drop(["Lead Number",'Converted'], axis=1)

y = leads['Converted']

display(y.head(),X.head())
# Splitting the data into train and test

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=100)
numdata=X_train[list(X_train.select_dtypes(numerics).columns)]

numdata.columns
import statsmodels.api as sm

logm1 = sm.GLM(y_train,(sm.add_constant(X_train)), family = sm.families.Binomial())

logm1.fit().summary()
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()



from sklearn.feature_selection import RFE

rfe = RFE(logreg, 24)             # running RFE with 15 variables as output

rfe = rfe.fit(X_train, y_train)

rfe.support_

list(zip(X_train.columns, rfe.support_, rfe.ranking_))
vars=X_train.columns[rfe.support_]
X_train_sm = sm.add_constant(X_train[vars])

logm2 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())

res = logm2.fit()

res.summary()
# VIF

X_train_sm = X_train_sm.drop(['const'], axis=1)

# Checking the  VIF of all the  features

from statsmodels.stats.outliers_influence import variance_inflation_factor



vif = pd.DataFrame()

X = X_train_sm

vif['Features'] = X.columns

vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
vars=vars.drop(['How did you hear about X Education_SMS'],1)



X_train_sm = sm.add_constant(X_train[vars])

logm2 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())

res = logm2.fit()

res.summary()
vars=vars.drop(['Specialization_Travel and Tourism'],1)



X_train_sm = sm.add_constant(X_train[vars])

logm2 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())

res = logm2.fit()

res.summary()
X_train_sm = X_train_sm.drop(['const'], axis=1)

# Checking the  VIF of all the  features



vif = pd.DataFrame()

X = X_train_sm

vif['Features'] = X.columns

vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif


vars=vars.drop(['Last Activity_Miscellaneous'],1)



X_train_sm = sm.add_constant(X_train[vars])

logm2 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())

res = logm2.fit()

res.summary()
vars=vars.drop(['Last Activity_SMS Sent'],1)



X_train_sm = sm.add_constant(X_train[vars])

logm2 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())

res = logm2.fit()

res.summary()
vars=vars.drop(['Lead Quality_Might be'],1)



X_train_sm = sm.add_constant(X_train[vars])

logm2 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())

res = logm2.fit()

res.summary()
X_train_sm = X_train_sm.drop(['const'], axis=1)

# Checking the  VIF of all the  features



vif = pd.DataFrame()

X = X_train_sm

vif['Features'] = X.columns

vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
vars=vars.drop(['Tags_Misc_Tags'],1)



X_train_sm = sm.add_constant(X_train[vars])

logm2 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())

res = logm2.fit()

res.summary()
vars=vars.drop(['What is your current occupation_Unemployed'],1)



X_train_sm = sm.add_constant(X_train[vars])

logm2 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())

res = logm2.fit()

res.summary()
X_train_sm = X_train_sm.drop(['const'], axis=1)

# Checking the  VIF of all the  features



vif = pd.DataFrame()

X = X_train_sm

vif['Features'] = X.columns

vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
vars=vars.drop(['Lead Profile_Any_other'],1)



X_train_sm = sm.add_constant(X_train[vars])

logm2 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())

res = logm2.fit()

res.summary()
vars=vars.drop(['Lead Quality_Worst'],1)



X_train_sm = sm.add_constant(X_train[vars])

logm2 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())

res = logm2.fit()

res.summary()
X_train_sm = X_train_sm.drop(['const'], axis=1)

# Checking the  VIF of all the  features



vif = pd.DataFrame()

X = X_train_sm

vif['Features'] = X.columns

vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
X_train_sm = sm.add_constant(X_train[vars])

logm2 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())

res = logm2.fit()

res.summary()
print("The final variables selected by the logsitic regression model are ","\n",vars)
# Let's run the model using the selected variables

from sklearn.linear_model import LogisticRegression

from sklearn import metrics

logsk = LogisticRegression(C=1e9)

logsk.fit(X_train[vars], y_train)
# Predicted probabilities

y_pred = logsk.predict_proba(X_train[vars])

# Converting y_pred to a dataframe which is an array

y_pred_df = pd.DataFrame(y_pred)

# Converting to column dataframe

y_pred_1 = y_pred_df.iloc[:,[1]]

# Let's see the head

y_pred_1.head()
# Converting y_train to dataframe

y_train_df = pd.DataFrame(y_train)

y_train_df.head()
# Putting index to LeadID

y_train_df['LeadID'] = y_train_df.index

y_train_df.head()


# Removing index for both dataframes to append them side by side 

y_pred_1.reset_index(drop=True, inplace=True)

y_train_df.reset_index(drop=True, inplace=True)

# Appending y_test_df and y_pred_1

y_pred_final = pd.concat([y_train_df,y_pred_1],axis=1)

# Renaming the column 

y_pred_final= y_pred_final.rename(columns={ 1 : 'Conv_Prob'})

# Rearranging the columns

y_pred_final = y_pred_final.reindex(['LeadID','Converted','Conv_Prob'], axis=1)

# Let's see the head of y_pred_final

y_pred_final.head()
# Creating new column 'predicted' with 1 if Conversion_Rate>0.5 else 0

y_pred_final['predicted'] = y_pred_final.Conv_Prob.map( lambda x: 1 if x > 0.5 else 0)

# Let's see the head

y_pred_final.head()
# Creating new column "Lead Score" with 1to100 using conversion rates

y_pred_final['Lead Score'] = y_pred_final.Conv_Prob.map( lambda x: round(x*100))

# Let's see the head

y_pred_final.head()
from sklearn import metrics

# Confusion matrix 

confusion = metrics.confusion_matrix( y_pred_final.Converted, y_pred_final.predicted )

confusion
#Let's check the overall accuracy.

metrics.accuracy_score(y_pred_final.Converted, y_pred_final.predicted)
metrics.precision_score(y_pred_final.Converted, y_pred_final.predicted)
metrics.recall_score(y_pred_final.Converted, y_pred_final.predicted)
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

"{:2.2f}".format(metrics.roc_auc_score(y_pred_final.Converted, y_pred_final.Conv_Prob))
from sklearn import metrics



# Confusion matrix 

confusion = metrics.confusion_matrix(y_pred_final.Converted, y_pred_final.predicted )

print(confusion)
# Let's create columns with different probability cutoffs 

numbers = [float(x)/10 for x in range(10)]

for i in numbers:

    y_pred_final[i]= y_pred_final.Conv_Prob.map(lambda x: 1 if x > i else 0)

y_pred_final.head()
# Calculating accuracy sensitivity and specificity for various probability cutoffs.

cutoff_df = pd.DataFrame( columns = ['probability','accuracy','sensitivity','specificity'])

from sklearn.metrics import confusion_matrix



num = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]

for i in num:

    cm1 = metrics.confusion_matrix(y_pred_final.Converted, y_pred_final[i] )

    total1=sum(sum(cm1))

    accuracy = (cm1[0,0]+cm1[1,1])/total1

    

    specificity = cm1[0,0]/(cm1[0,0]+cm1[0,1])

    sensitivity = cm1[1,1]/(cm1[1,0]+cm1[1,1])

    cutoff_df.loc[i] =[ i ,accuracy,sensitivity,specificity]

print(cutoff_df)
# Let's plot accuracy sensitivity and specificity for various probabilities.

cutoff_df.plot.line(x='probability', y=['accuracy','sensitivity','specificity'])

plt.show()
#  0.3 is the optimum point to take it as a cutoff probability tp predict the final probability



y_pred_final['final_pred'] = y_pred_final.Conv_Prob.map( lambda x: 1 if x > 0.3 else 0)



y_pred_final.head(10)
# Let's check the overall accuracy.

metrics.accuracy_score(y_pred_final.Converted, y_pred_final.final_pred)



cm2 = metrics.confusion_matrix(y_pred_final.Converted, y_pred_final.final_pred )

cm2

TP = cm2[1,1] 

TN = cm2[0,0] 

FP = cm2[0,1] 

FN = cm2[1,0] 

print("SENSITIVITY of the logistic regression model is  ",TP / float(TP+FN))

print("True negatives are ",TN / float(TN+FP))

print("False Positives are  ",FP/ float(TN+FP))

print ("True Positives are  ",TP / float(TP+FP))

print (TN / float(TN+ FN))
from sklearn.metrics import precision_recall_curve

precision, recall, thresholds = precision_recall_curve(y_pred_final.Converted, y_pred_final.Conv_Prob)

plt.plot(thresholds, precision[:-1], "b")

plt.plot(thresholds, recall[:-1], "g")

plt.show()
X_test[['TotalVisits','Total Time Spent on Website']] = scaler.fit_transform(X_test[['TotalVisits','Total Time Spent on Website']])

X_test=X_test[vars]

X_test.head()
X_test_sm = sm.add_constant(X_test)

y_test_pred = res.predict(X_test_sm)

y_test_pred.head()
y_pred_1 = pd.DataFrame(y_test_pred)

y_test_df = pd.DataFrame(y_test)

# Putting CustID to index

y_test_df['LeadID'] = y_test_df.index

# Removing index for both dataframes to append them side by side 

y_pred_1.reset_index(drop=True, inplace=True)

y_test_df.reset_index(drop=True, inplace=True)

# Appending y_test_df and y_pred_1

y_pred_final = pd.concat([y_test_df, y_pred_1],axis=1)

y_pred_final= y_pred_final.rename(columns={ 0 : 'Conv_Prob'})

y_pred_final = y_pred_final.reindex(['LeadID','Converted','Conv_Prob'], axis=1)

y_pred_final.head()
# Creating new column "Lead Score" with 1to100 using conversion rates

y_pred_final['Lead_Score'] = y_pred_final.Conv_Prob.map( lambda x: round(x*100))

# Let's see the head

y_pred_final.head()
y_pred_final['final_pred'] = y_pred_final.Conv_Prob.map(lambda x: 1 if x > 0.38 else 0)

y_pred_final.head(10)
print("Model Accuracy on Test data is ",metrics.accuracy_score(y_pred_final.Converted, y_pred_final.final_pred))
confusion2 = metrics.confusion_matrix(y_pred_final.Converted, y_pred_final.final_pred )

confusion2

TP = confusion2[1,1]  

TN = confusion2[0,0] 

FP = confusion2[0,1] 

FN = confusion2[1,0] 



print("Sensitivity of the model on test data is ",round(TP / float(TP+FN),2))
print("Specificity of the model on test data is ",TN / float(TN+FP))
ydf=y_train_df.set_index("LeadID")
Xy_Traindf=pd.concat([ydf,X_train_sm.iloc[:,1:]],axis=1)
Xy_Traindf.corr()["Converted"].sort_values()
Xy_Traindf.reset_index(inplace=True)
Xy_Traindf=Xy_Traindf.rename(columns={"index":"LeadID"})
@interact

def counts(col =['Lead Origin_Lead Add Form', 'Lead Source_Olark Chat',

       'Lead Source_Welingak Website', 'Do Not Email_Yes',

       'Tags_Closed by Horizzon', 'Tags_Lost to EINS',

       'Tags_Will revert after reading the email', 'Lead Quality_Not Sure',

       'Lead Profile_Other Leads', 'Lead Profile_Potential Lead',

       'Last Notable Activity_Modified',

       'Last Notable Activity_Olark Chat Conversation',

       'Last Notable Activity_SMS Sent']):

    sns.countplot(x=col,data=Xy_Traindf,hue="Converted",palette="husl")

    plt.xlabel(col)

    plt.ylabel('Total count')

    plt.legend(loc='upper center', bbox_to_anchor=(1, 0.8), ncol=1)

    plt.xticks(rotation=65, horizontalalignment='right',fontweight='light')

    convertcount=Xy_Traindf.pivot_table(values='LeadID',index=col,columns='Converted', aggfunc='count').fillna(0)

    convertcount["Conversion(%)"] =round(convertcount[1]/(convertcount[0]+convertcount[1]),2)*100

    return print(convertcount.sort_values(ascending=False,by=1),plt.show())