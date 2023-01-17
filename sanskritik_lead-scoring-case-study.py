# Supress Warnings

import warnings

warnings.filterwarnings('ignore')
# Importing libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

import missingno as msno



# visulaisation

from matplotlib.pyplot import xticks

%matplotlib inline

import itertools

import copy



import statsmodels.api as sm

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.feature_selection import RFE

from sklearn import metrics

from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.metrics import confusion_matrix

from sklearn.metrics import precision_score, recall_score

from sklearn.metrics import precision_recall_curve
# Data display customization

pd.set_option('display.max_rows', None)

pd.set_option('display.max_columns', None)

pd.set_option('display.width', None)

pd.set_option('display.max_colwidth', None)
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

# Importing Leads.csv

Leaddata = pd.read_csv('/kaggle/input/lead-scoring/Leads.csv')

Leaddata.head()
Data_dict = pd.read_excel("/kaggle/input/lead-scoring/Leads Data Dictionary.xlsx",sheet_name= 'Sheet1', header= [1] ,skiprows = [0],  usecols=lambda x: 'Unnamed' not in x)

Data_dict
# To check the rows and columns of the data

#Shape of dataset

print("There are {} countries and {} features: ".format(Leaddata.shape[0],Leaddata.shape[1]))
# to check data type

Leaddata.info()
# to get the basic statistics of the numerical column

# Checking outliers at 1%, 5%, 10%, 25%,50%,75%,90%,95% and 99%

Leaddata.describe(percentiles=[.01,.05,.1,.25,.5,.75,.90,.95,.99])
#checking duplicates

sum(Leaddata.duplicated(subset = 'Prospect ID')) == 0
#check for duplicates

sum(Leaddata.duplicated(subset = 'Lead Number')) == 0
#dropping Lead Number since they have all unique values

Leaddata.drop(['Lead Number'], 1, inplace = True)
# Calculating the Missing Values % contribution in DF

display(round(100*(Leaddata.isnull().sum()/len(Leaddata.index)), 2))
# Plotting Null Count

msno.bar(Leaddata)

plt.show()
# Percentage of Missing values in Lead data

sns.set(style="whitegrid")

fig = plt.figure(figsize=(12,5))

missing = pd.DataFrame((Leaddata.isnull().sum())*100/Leaddata.shape[0]).reset_index()

missing["type"] = "Lead Data"

ax = sns.pointplot("index",0,data=missing,hue="type", palette="Set2")

plt.xticks(rotation =90,fontsize =12)

plt.title("Percentage of Missing values in Lead data",fontsize =14)

plt.ylabel("Percentage of Missing values",fontsize =14)

plt.xlabel("Column Name",fontsize =14)

plt.show()
# Converting 'Select' values to NaN.

Leaddata = Leaddata.replace('Select', np.nan)
#check null values again

display(round(100*(Leaddata.isnull().sum()/len(Leaddata.index)), 2))
#Percentage of Missing values in application data

fig = plt.figure(figsize=(12,5))

missing = pd.DataFrame((Leaddata.isnull().sum())*100/Leaddata.shape[0]).reset_index()

missing["type"] = "Lead Data"

ax = sns.pointplot("index",0,data=missing,hue="type", palette="Set2")

plt.xticks(rotation =90,fontsize =12)

plt.title("Percentage of Missing values in Lead data",fontsize =14)

plt.ylabel("Percentage of Missing values",fontsize =14)

plt.xlabel("Column Name", fontsize =14)

plt.show()
# columns having more than 70 % of null values are actually useless , so for such columns , we will get rid of the column itself

Leaddata = Leaddata.drop(Leaddata.loc[:,list(round(100*(Leaddata.isnull().sum()/len(Leaddata.index)), 2)>50)].columns, 1)
#check null values again

display(round(100*(Leaddata.isnull().sum()/len(Leaddata.index)), 2))
def countPlot(col_name, x = None):

    ap = sns.countplot(x= col_name, data = Leaddata, palette="Set2", hue = x )

    for p in ap.patches:

        ap.annotate('{:1.2f}%'.format((p.get_height()*100)/float(len(Leaddata[col_name]))), 

                (p.get_x()+0.05, p.get_height()+20), size =12) 
def valueCount(df):

    for i in df:

        print('Column \"' +i+ '\" value counts\n')

        print(df[i].value_counts(ascending=False,normalize=True), '\n\n')
# Count Plot before Imputing the values

#sns.set(style="white")

plt.figure(figsize =(15,4))

plt.subplot(121)

countPlot('What matters most to you in choosing a course')

xticks(rotation = 45)

plt.subplot(122)

countPlot('What is your current occupation')

xticks(rotation = 45)

plt.show()
null_col = Leaddata[['City', 'Specialization']]

valueCount(null_col) 
#impute 'Mumbai' in column 'City'

Leaddata['City'] = Leaddata['City'].replace(np.nan, 'Mumbai')
# So Let's make a category "Others" for missing values.

Leaddata['Specialization'] = Leaddata['Specialization'].replace(np.nan, 'Not Specified')
#combining Management Specializations 

Leaddata['Specialization'] = Leaddata['Specialization'].replace(['Finance Management','Human Resource Management',

                                                           'Marketing Management','Operations Management',

                                                           'IT Projects Management','Supply Chain Management',

                                                    'Healthcare Management','Hospitality Management',

                                                           'Retail Management'] ,'Management_Specializations')
Leaddata['Specialization'] = Leaddata['Specialization'].replace(['Services Excellence','E-Business',

                                                                'Rural and Agribusiness','E-COMMERCE',] ,

                                                                'Other Specilization')
# Count Plot After Imputing the values

#sns.set(style="white")

plt.figure(figsize =(15,4))

plt.subplot(121)

countPlot('Specialization')

xticks(rotation = 90)

plt.subplot(122)

countPlot('City')

xticks(rotation = 90)

plt.show()
null_col = Leaddata[['What matters most to you in choosing a course','What is your current occupation']]

valueCount(null_col) 
# Impute Missing Values

Leaddata['What is your current occupation'] = Leaddata['What is your current occupation'].replace(np.nan, 'Unemployed')
plt.figure(figsize =(15,4))

sns.countplot(Leaddata['Country'],palette="Set2")

xticks(rotation = 90)

plt.show()
#Leaddata["Country"].value_counts(normalize=True)

null_col = Leaddata[["Country"]]

valueCount(null_col)
Leaddata.drop([ 'Country','What matters most to you in choosing a course','Tags',

                'Asymmetrique Activity Index','Asymmetrique Activity Score',

                'Asymmetrique Profile Index','Asymmetrique Profile Score'],

                axis =1, inplace= True)
# drop rows containing missing values

Leaddata.dropna(inplace = True)
#check null values again

display(round(100*(Leaddata.isnull().sum()/len(Leaddata.index)), 2))
# Plotting Null Count

msno.bar(Leaddata)

plt.show()
# Code to check outliers for numerical columns based on boxplot

numerical = ['TotalVisits', 'Total Time Spent on Website', 'Page Views Per Visit']

plt.figure(figsize =(15,5))

for i in enumerate(numerical):

    plt.subplot(1,3, i[0]+1)

    sns.boxplot(x= i[1], data = Leaddata,palette="Set2",orient = 'v')
# Check the distribution of Numerical columns

numerical = ['TotalVisits', 'Total Time Spent on Website', 'Page Views Per Visit']

plt.figure(figsize =(15,5))

for i in enumerate(numerical):

    plt.subplot(1,3, i[0]+1)

    sns.violinplot(x= i[1], data = Leaddata,palette="Set2",scale="count", inner="quartile", orient = 'v')
#Dist Plot for each Numerical Variables

a = ['TotalVisits', 'Total Time Spent on Website', 'Page Views Per Visit']

fig = plt.figure(figsize =(15,5))

for i,j in itertools.zip_longest(a, range(len(a))):

    plt.subplot(1,3,j+1)

    ax = sns.distplot(Leaddata[i])
Leaddata.info()
# Converting datatype of categorical columns to object type from int/float

#Leaddata['Lead Number'] = Leaddata['Lead Number'].astype(object)

Leaddata['Converted'] = Leaddata['Converted'].astype(object)
# to get the basic statistics of the numerical column

# Checking outliers at 1%, 5%, 10%, 25%,50%,75%,90%,95% and 99%

Leaddata.describe(percentiles=[.01,.05,.1,.25,.5,.75,.90,.95,.99])
TotalVisits_q4 = Leaddata['TotalVisits'].quantile(0.99)

TimeSpent_q4 = Leaddata['Total Time Spent on Website'].quantile(0.99)

PageViews_q4 = Leaddata['Page Views Per Visit'].quantile(0.99)



Leaddata['TotalVisits'][Leaddata['TotalVisits']>= TotalVisits_q4] = TotalVisits_q4

Leaddata['Total Time Spent on Website'][Leaddata['Total Time Spent on Website']>= TimeSpent_q4] = TimeSpent_q4

Leaddata['Page Views Per Visit'][Leaddata['Page Views Per Visit']>= PageViews_q4] = PageViews_q4
Converted = (sum(Leaddata['Converted'])/len(Leaddata['Converted'].index))*100

Converted
def pie_plot(column_name):

    Leaddata[column_name].value_counts().plot.pie(autopct = "%1.2f%%",colors = sns.color_palette("Set2",7),

                            startangle = 60,labels=["No","Yes"],

                            wedgeprops={"linewidth":2,"edgecolor":"k"},explode=[.1,0],shadow =True)

    plt.title("Distribution of "+column_name+" " , size = 12)

    
# Plot Distribution of Target variable

plt.figure(figsize=(15,5))

plt.subplot(121)

ap = sns.countplot(x= 'Converted', data = Leaddata,palette="Set2")

for p in ap.patches:

    ap.annotate('{:1.1f}%'.format((p.get_height()*100)/float(len(Leaddata["Converted"]))), (p.get_x()+0.05, p.get_height()+20))  

plt.xticks(size = 16)

plt.ylabel('converted' , size =16)

plt.yticks(size = 16)

plt.xlabel('count' , size =16)

plt.title("How many Leads are Converted", size =16)

plt.subplot(122)

pie_plot("Converted")

plt.title("Distribution of Converted Leads", size =16)

plt.show()
null_col = Leaddata[["Newspaper Article","X Education Forums","Newspaper","Search",

"Digital Advertisement","Through Recommendations","Do Not Email","Do Not Call"]]

valueCount(null_col) 
columns = ["Newspaper Article","X Education Forums","Newspaper", "Search",

"Digital Advertisement","Through Recommendations","Do Not Email","Do Not Call"] 

plt.figure(figsize=(15,8))

i = 1

for A in columns:

    plt.subplot(2,4,i)

    pie_plot(A)

    i= i+1
def pie_plot_2(column_name):

    Leaddata[column_name].value_counts().plot.pie(autopct = "%1.2f%%",colors = sns.color_palette("Set2",7),

                            startangle = 60,labels=["No"],

                            wedgeprops={"linewidth":2,"edgecolor":"k"},shadow =True)

    #plt.title("Distribution of "+column_name+" " , size = 12)

    
null_col = Leaddata[["I agree to pay the amount through cheque",

                     "Get updates on DM Content",

                     "Update me on Supply Chain Content",

                     "Receive More Updates About Our Courses"]]

valueCount(null_col) 
columns = [ "I agree to pay the amount through cheque",

            "Get updates on DM Content","Magazine",

            "Update me on Supply Chain Content",

            "Receive More Updates About Our Courses"] 

plt.figure(figsize=(15,5))

i = 1

for A in columns:

    plt.subplot(1,5,i)

    pie_plot_2(A)

    i= i+1
null_col = Leaddata[["Lead Origin","What is your current occupation","City",

                     "A free copy of Mastering The Interview"]]

valueCount(null_col) 
categorical1 = ["Lead Origin","What is your current occupation"]

plt.figure(figsize=(15,4))

i = 1

for A in categorical1:

    plt.subplot(1,2,i)

    countPlot(A)

    i= i+1

    xticks(rotation = 90)
categorical2 = ["City","A free copy of Mastering The Interview"]

plt.figure(figsize=(15,4))

i = 1

for A in categorical2:

    plt.subplot(1,2,i)

    countPlot(A)

    i= i+1

    xticks(rotation = 45)
Leaddata["Lead Source"].value_counts(normalize=True)

# This variable looks useful
#replacing 'google' with 'Google' and combining low frequency values

Leaddata['Lead Source'] = Leaddata['Lead Source'].replace('google','Google')

Leaddata['Lead Source'] = Leaddata['Lead Source'].replace(['Welingak Website', 'Referral Sites', 'Facebook',

                                                           'bing','Click2call','Social Media','Press_Release',

                                                           'Live Chat','WeLearn','testone','NC_EDM','welearnblog_Home',

                                                           'blog','youtubechannel','Pay per Click Ads'] ,'Others')                                                   
Leaddata["Last Notable Activity"].value_counts(normalize=True)

# This column looks useful
Leaddata['Last Notable Activity'] = Leaddata['Last Notable Activity'].replace(['View in browser link Clicked', 

                                                                               'Approached upfront', 'Resubscribed to emails',

                                                                               'Email Received','Form Submitted on Website',

                                                                              'Email Marked Spam','Had a Phone Conversation',

                                                                              'Unreachable','Unsubscribed',

                                                                              'Email Bounced','Email Link Clicked'] ,'Other Notable Activity')                                                   
plt.figure(figsize=(15,4))

plt.subplot(121)

countPlot("Lead Source")

xticks(rotation = 45)

plt.subplot(122)

countPlot("Last Notable Activity")

xticks(rotation = 45)

plt.show()
Leaddata["Last Activity"].value_counts(normalize=True)
Leaddata.drop([ "Newspaper Article","X Education Forums","Newspaper", "Search",

                "Digital Advertisement","Through Recommendations","Do Not Email",

                "Do Not Call","I agree to pay the amount through cheque",

                "Get updates on DM Content","Magazine","Last Activity",

                "Update me on Supply Chain Content",

                "Receive More Updates About Our Courses"],

                axis =1, inplace= True)
Leaddata.shape
Leaddata.head()
def countPlot2(col_name, x = None):

    ax = sns.countplot(x= col_name, data = Leaddata, palette="Set2", hue = x )
Leaddata.columns
plt.figure(figsize=(15,4))

plt.subplot(131)

sns.boxplot(x='Converted', y='TotalVisits', data=Leaddata, palette="Set2")

plt.subplot(132)

sns.boxplot(x='Converted', y='Total Time Spent on Website', data=Leaddata, palette="Set2")

plt.subplot(133)

sns.boxplot(x='Converted', y='Page Views Per Visit', data=Leaddata, palette="Set2")

plt.show()
sns.set(style = 'white')

fig = plt.figure(figsize =(20,20))

sns.pairplot(data=Leaddata,diag_kind='kde',corner = True,

             vars=['TotalVisits', 'Total Time Spent on Website', 'Page Views Per Visit'],hue='Converted',palette="husl")

plt.show()
# 'Lead Source', 'Converted',

plt.figure(figsize=(12,5))

countPlot2("Lead Origin","Converted" )

plt.show()
plt.figure(figsize=(12,5))

countPlot2("Lead Source","Converted" )

xticks(rotation = 45)

plt.show()
plt.figure(figsize=(12,5))

countPlot2("Specialization","Converted" )

xticks(rotation = 45)

plt.show()
plt.figure(figsize=(12,5))

countPlot2("What is your current occupation","Converted" )

xticks(rotation = 45)

plt.show()
plt.figure(figsize=(12,5))

countPlot2("City","Converted" )

xticks(rotation = 45)

plt.show()
plt.figure(figsize=(12,5))

countPlot2("Last Notable Activity","Converted" )

xticks(rotation = 45)

plt.show()
plt.figure(figsize=(12,5))

countPlot2("A free copy of Mastering The Interview","Converted" )

xticks(rotation = 45)

plt.show()
var = ["A free copy of Mastering The Interview"]

def binary_map(x):

    return x.map({'Yes': 1, "No": 0})



# Applying the function to the housing list

Leaddata[var] = Leaddata[var].apply(binary_map)
Leaddata.head()
Leaddata.columns
# Creating a dummy variable for some of the categorical variables and dropping the first one.

data = pd.get_dummies(data = Leaddata , columns=['Lead Origin', 'Lead Source', 

       'Specialization', 'What is your current occupation', 'A free copy of Mastering The Interview',

       'City', 'Last Notable Activity'], drop_first=True)

data.head()
data.shape
data = data.set_index('Prospect ID')
# Putting feature variable to X

X = data.drop(['Converted'], axis=1)
X.head()
# Putting response variable to y

y = data['Converted']



y.head()
# Splitting the data into train and test

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=333)
X_train.shape
X_test.shape
y_train.shape
y_train.dtype
y_train = y_train.astype(float)
y_train.dtype
X_train.info()
X_test.info()
y_test.dtype
y_test = y_test.astype(float)
y_test.dtype
y_test.shape
scaler = StandardScaler()

X_train[['TotalVisits','Total Time Spent on Website','Page Views Per Visit']] = scaler.fit_transform(X_train[['TotalVisits',

                                                                                                              'Total Time Spent on Website','Page Views Per Visit']])

X_train.head()
# Checking the Churn Rate

Converted = (sum(data['Converted'])/len(data['Converted'].index))*100

Converted
logreg = LogisticRegression()

rfe = RFE(logreg, 18)             # running RFE with 18 variables as output

rfe = rfe.fit(X_train, y_train)
rfe.support_
list(zip(X_train.columns, rfe.support_, rfe.ranking_))
# List of variables to be considered for model building

col1 = X_train.columns[rfe.support_]

col1
# List of variables to be removed from data set

X_train.columns[~rfe.support_]
X_train_sm = sm.add_constant(X_train[col1])

logm2 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())

res = logm2.fit()

res.summary()
# Create a dataframe that will contain the names of all the feature variables and their respective VIFs

def vif_show(X_vif):

    vif = pd.DataFrame()

    vif['Features'] = X_vif.columns

    vif['VIF'] = [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])]

    vif['VIF'] = round(vif['VIF'], 2)

    vif = vif.sort_values(by = "VIF", ascending = False)

    print(vif)
vif_show(X_train[col1])
col2 = col1.drop('What is your current occupation_Housewife',1)
col2
X_train_sm = sm.add_constant(X_train[col2])

logm2 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())

res = logm2.fit()

res.summary()
vif_show(X_train[col2])
col3 = col2.drop('Lead Source_Others',1)
col3
X_train_sm = sm.add_constant(X_train[col3])

logm2 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())

res = logm2.fit()

res.summary()
vif_show(X_train[col3])
col4 = col3.drop('What is your current occupation_Unemployed',1)
col4
X_train_sm = sm.add_constant(X_train[col4])

logm2 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())

res = logm2.fit()

res.summary()
vif_show(X_train[col4])
col5 = col4.drop('What is your current occupation_Student',1)
col5
X_train_sm = sm.add_constant(X_train[col5])

logm2 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())

res = logm2.fit()

res.summary()
vif_show(X_train[col5])
col6 = col5.drop('Specialization_Media and Advertising',1)
col6
X_train_sm = sm.add_constant(X_train[col6])

logm2 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())

res = logm2.fit()

res.summary()
vif_show(X_train[col6])
col7 = col6.drop('Lead Origin_Lead Import',1)
col7
X_train_sm = sm.add_constant(X_train[col7])

logm2 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())

res = logm2.fit()

res.summary()
vif_show(X_train[col7])
col8 = col7.drop('Last Notable Activity_Page Visited on Website',1)
col8
X_train_sm = sm.add_constant(X_train[col8])

logm2 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())

res = logm2.fit()

res.summary()
vif_show(X_train[col8])
# Getting the predicted values on the train set

y_train_pred = res.predict(X_train_sm)

y_train_pred[:10]
y_train_pred_final = pd.DataFrame({'Converted':y_train.values, 'Converted_prob':y_train_pred})

y_train_pred_final.head()
y_train_pred_final['predicted'] = y_train_pred_final.Converted_prob.map(lambda x: 1 if x > 0.5 else 0)

# Let's see the head

y_train_pred_final.head()
# Confusion matrix 

confusion = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final.predicted )

print(confusion)
# Let's check the overall accuracy.

accuracy_train = metrics.accuracy_score(y_train_pred_final.Converted, y_train_pred_final.predicted)

print("Train Data Accuracy: " +str(round(accuracy_train,4)))
TP = confusion[1,1] # true positive 

TN = confusion[0,0] # true negatives

FP = confusion[0,1] # false positives

FN = confusion[1,0] # false negatives
# Calculate sensitivity of our logistic regression model

sensitivity_train = TP / float(TP+FN)

# Let us calculate specificity

specificity_train = TN / float(TN+FP)

# Calculate false postive rate - predicting churn when customer does not have churned

fpRate_train =  FP/ float(TN+FP)

# positive predictive value 

positive_predictive_train = TP / float(TP+FP)

# Negative predictive value

ngRate_train = TN / float(TN+ FN)
print("Printing all values before optimal cut-off calculation\n")

print("Train Data Accuracy:            {} ".format(round(accuracy_train,4)))

print("Train Data Specificity:         {} ".format(round(sensitivity_train,4)))

print("Train Data Sensitivity:         {} ".format(round(specificity_train,4)))

print("Train Data False postive rate:  {} ".format(round(fpRate_train,4)))

print("Train Data Positive predictive: {} ".format(round(positive_predictive_train,4)))

print("Train Data Negative predictive: {} ".format(round(ngRate_train,4)))
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

sns.set(style='whitegrid')

cutoff_df.plot.line(x='prob', y=['accuracy','sensi','speci'])

plt.show()
y_train_pred_final['final_predicted'] = y_train_pred_final.Converted_prob.map( lambda x: 1 if x > 0.35 else 0)

y_train_pred_final.head()
y_train_pred_final['Lead_Score'] = y_train_pred_final.Converted_prob.map( lambda x: round((x*100),2))

y_train_pred_final.head()
# Let's check the overall accuracy.

accuracy_train = metrics.accuracy_score(y_train_pred_final.Converted, y_train_pred_final.final_predicted)
confusion2 = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final.final_predicted )

confusion2
TP = confusion2[1,1] # true positive 

TN = confusion2[0,0] # true negatives

FP = confusion2[0,1] # false positives

FN = confusion2[1,0] # false negatives
# Calculate sensitivity of our logistic regression model

sensitivity_train = TP / float(TP+FN)

#calculate specificity

specificity_train = TN / float(TN+FP)

# Calculate false postive rate - predicting churn when customer does not have churned

fpRate_train = FP/ float(TN+FP)

# Positive predictive value 

positive_predictive_train = TP / float(TP+FP)

# Negative predictive value

ngRate_train = TN / float(TN+ FN)
print("Printing all values post optimal cut-off calculation\n")

print("Train Data Accuracy:            {} ".format(round(accuracy_train,4)))

print("Train Data Specificity:         {} ".format(round(sensitivity_train,4)))

print("Train Data Sensitivity:         {} ".format(round(specificity_train,4)))

print("Train Data False postive rate:  {} ".format(round(fpRate_train,4)))

print("Train Data Positive predictive: {} ".format(round(positive_predictive_train,4)))

print("Train Data Negative predictive: {} ".format(round(ngRate_train,4)))
confusion = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final.predicted)

confusion
# Precision

confusion[1,1]/(confusion[0,1]+confusion[1,1])
# Recall

confusion[1,1]/(confusion[1,0]+confusion[1,1])
# Calculate Recall

recall_train = recall_score(y_train_pred_final.Converted, y_train_pred_final.predicted)

# Calculate Precision

precision_train = precision_score(y_train_pred_final.Converted , y_train_pred_final.predicted)

# Calculate F1 Score

F1_score_train =  (2* precision_train * recall_train)/(precision_train + recall_train)
print("Printing all values post optimal cut-off calculation\n")

print("Train Data Precision:    {} ".format(round(precision_train,4)))

print("Train Data Recall:       {} ".format(round(recall_train,4)))

print("Train Data F1 Score:     {} ".format(round(F1_score_train,4)))
p, r, thresholds = precision_recall_curve(y_train_pred_final.Converted, y_train_pred_final.Converted_prob)
plt.plot(thresholds, p[:-1], "g-")

plt.plot(thresholds, r[:-1], "r-")

plt.show()
#Preparing Test Dataset

X_test[['TotalVisits','Total Time Spent on Website','Page Views Per Visit']] = scaler.fit_transform(X_test[['TotalVisits','Total Time Spent on Website','Page Views Per Visit']])



X_train.head()
X_test = X_test[col8]

X_test.head()
X_test_sm = sm.add_constant(X_test)
X_test_sm.shape
y_test_pred = res.predict(X_test_sm)
y_test_pred[:10]
y_test.shape
y_test_pred.shape
# Converting y_test to dataframe

y_test_df = pd.DataFrame(y_test)

y_pred_final_temp = pd.DataFrame(y_test_pred)

y_pred_final = pd.merge(y_test_df,y_pred_final_temp,on='Prospect ID')

# Appending y_test_df and y_pred_1

y_pred_final.head()
# Renaming the column 

y_pred_final= y_pred_final.rename(columns={ 0 : 'Converted_prob'})
y_pred_final.head()
y_pred_final['final_predicted'] = y_pred_final.Converted_prob.map(lambda x: 1 if x > 0.35 else 0)

y_pred_final.head()
# Let's check the overall accuracy.

accuracy_test = metrics.accuracy_score(y_pred_final.Converted, y_pred_final.final_predicted)
confusion2 = metrics.confusion_matrix(y_pred_final.Converted, y_pred_final.final_predicted )

confusion2
TP = confusion2[1,1] # true positive 

TN = confusion2[0,0] # true negatives

FP = confusion2[0,1] # false positives

FN = confusion2[1,0] # false negatives
# Let's see the sensitivity of our logistic regression model

sensitivity_test = TP / float(TP+FN)



# Let us calculate specificity

specificity_test = TN / float(TN+FP)
print("Printing Accuracy, Specificity and Sensitivity of test data \n")

print("Test Data Accuracy:            {} ".format(round(accuracy_test,4)))

print("Test Data Specificity:         {} ".format(round(sensitivity_test,4)))

print("Test Data Sensitivity:         {} ".format(round(specificity_test,4)))
# Assigning Lead Score

y_pred_final['Lead_Score'] = y_pred_final.Converted_prob.map( lambda x: round((x*100),2))

y_pred_final.head(10)
confusion = metrics.confusion_matrix(y_pred_final.Converted, y_pred_final.final_predicted )

confusion
# Precision

confusion[1,1]/(confusion[0,1]+confusion[1,1])
# Recall

confusion[1,1]/(confusion[1,0]+confusion[1,1])
recall_test = recall_score(y_pred_final.Converted, y_pred_final.final_predicted )

precision_test = precision_score(y_pred_final.Converted, y_pred_final.final_predicted )

F1_score_test =  (2* precision_test * recall_test)/(precision_test + recall_test)
print("Printing Precision and Recall of test data\n")

print("Test Data Precision:    {} ".format(round(precision_test,4)))

print("Test Data Recall:       {} ".format(round(recall_test,4)))

print("Test Data F1 Score:     {} ".format(round(F1_score_test,4)))
print("Printing Results of Train Data\n")

print("Train Data Accuracy:     {} ".format(round(accuracy_train,4)))

print("Train Data Specificity:  {} ".format(round(sensitivity_train,4)))

print("Train Data Sensitivity:  {} ".format(round(specificity_train,4)))

print("Train Data Precision:    {} ".format(round(precision_train,4)))

print("Train Data Recall:       {} ".format(round(recall_train,4)))

print("Train Data F1 Score:     {} ".format(round(F1_score_train,4)))
Lead_train = y_train_pred_final.copy()

Lead_train.drop(["Converted","Converted_prob","predicted",0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 

                 "final_predicted"], 1, inplace = True)

Lead_train.reset_index(inplace=True)

Lead_train.sort_values(by=["Lead_Score"], inplace=True, ascending=False)

Lead_train.head()
print("Printing Results of Test Data \n")

print("Test Data Accuracy:     {} ".format(round(accuracy_test,4)))

print("Test Data Specificity:  {} ".format(round(sensitivity_test,4)))

print("Test Data Sensitivity:  {} ".format(round(specificity_test,4)))

print("Test Data Precision:    {} ".format(round(precision_test,4)))

print("Test Data Recall:       {} ".format(round(recall_test,4)))

print("Test Data F1 Score:     {} ".format(round(F1_score_test,4)))
Lead_test = y_pred_final.copy()

Lead_test.drop(['Converted', 'Converted_prob', 'final_predicted'], 1, inplace = True)

Lead_test.reset_index(inplace=True)

Lead_test.sort_values(by=['Lead_Score'], inplace=True, ascending=False)

Lead_test.head()