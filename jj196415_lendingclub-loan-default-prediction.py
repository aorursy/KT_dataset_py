# Import libraries 

import os

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

%matplotlib inline

warnings.simplefilter('ignore') #Hide warning messages

sns.set_style('whitegrid') #Set the plots to whitegrid



# Increase the default display to 1000 for methods like head()

pd.options.display.max_rows = 1000
# Input data files are available in the "../input/" directory.

os.chdir('/kaggle/input') # Change working dirtory to the input folder



all_files = []

for dirname, _, filenames in os.walk('/kaggle/input'):

    all_files = filenames
# Import all data files and combine into one dataframe

lst = []



for filename in all_files:

    df = pd.read_csv(filename, skiprows=[0])

    lst.append(df)



data_consol = pd.concat(lst, join="inner", ignore_index=True)
data_consol.shape
# Show the first few records with column names

data_consol.head(3).T # .T is to transpose the matrix.
data_consol['int_rate'].describe # Provide key info of the column.
# Use lambda+appy method to remove the ""%" from each row of data

# As some rows were already in float, need to first convert the data to string to use strip

data_consol['int_rate'] = data_consol["int_rate"].apply(lambda x: str(x).strip('%'))
# Confirm there's no % in the column any more

len([rate for rate in data_consol['int_rate'] if '%' in str(rate)])
# Use astype to cast the data frame column to float 

data_consol['int_rate'] = data_consol['int_rate'].astype(float)
#Confirm

data_consol["int_rate"].dtypes
data_consol['issue_d']
# As null are treated as float here, convert the data to string first

# use split() to retrieve the year, which is the last element of output list

# use[-1] to grab the last element, this is because the null rows will only have one element

data_consol['issue_yr'] = data_consol['issue_d'].apply(lambda y: str(y).split('-')[-1])
data_consol['issue_yr'].unique()
data_consol[data_consol['loan_amnt'].isnull()==True]["id"].count() # Number of records with NULL loan amount
na_rows = data_consol[data_consol['loan_amnt'].isnull()==True].index

data_consol.drop(na_rows,axis=0,inplace=True)
# Confirm all invalid rows have been removed

data_consol[data_consol['loan_amnt'].isnull()==True].index
data_consol["loan_status"].unique()
# Define remove_str() function to reclassify categories

def remove_str(x):

    if ":Fully Paid" in x:

        return "Fully Paid"

    elif ":Charged Off" in x:

        return "Charged Off"

    else:

        return x
data_consol["loan_status"] = data_consol["loan_status"].apply(remove_str)
data_consol["loan_status"].unique() # Confirm the categories have been updated
[col for col in data_consol.columns if 'fico' in col.lower()]
data_consol['grade'].unique() # Column showing loan grade
data_consol['sub_grade'].unique() # Column showing loan sub-grade
data_consol['purpose'].unique() # Column showing loan purpose
data_consol['home_ownership'].unique() # Column showing home owenership status of the applicant
sns.distplot(data_consol['int_rate'].dropna(),

             kde=False, color='darkblue', bins=40)
sns.countplot(x='grade', data=data_consol, order='A B C D E F G'.split())
plt.figure(figsize=(12,6))

ax = sns.countplot(x='issue_yr',data=data_consol)

#Show the value count of each category

for p in ax.patches:

    height = p.get_height()

    ax.text(p.get_x()+p.get_width()/2., height+0.1,height ,ha="center")
plt.figure(figsize=(10,6))

ax = sns.countplot(x='loan_status',data=data_consol)

ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha='right') #show the x lable with an angel to avoid overlapping

plt.tight_layout()



#Show the value count of each category

for p in ax.patches:

    height = p.get_height()

    ax.text(p.get_x()+p.get_width()/2., height+0.1,height ,ha="center")
plt.figure(figsize=(12,6))

ax=sns.countplot(x='loan_status',data=data_consol,hue='grade', hue_order='A B C D E F G'.split())

ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha='right') #show the x lable with an angel to avoid overlapping

plt.legend(loc='upper right')
X = data_consol[data_consol['loan_status'].isin(['Default','Charged Off']) ]

plt.figure(figsize=(12,10))

sns.countplot(x='issue_yr',data=X)
#Check which columns can be used for FICO

data_consol.iloc[0:3][['fico_range_low', 'fico_range_high', 'last_fico_range_high',

 'last_fico_range_low',

 'sec_app_fico_range_low',

 'sec_app_fico_range_high']]
#Check if there're nulls in fico_range_low and fico_range_high

sns.heatmap(data_consol[['fico_range_low', 'fico_range_high']].isnull(),yticklabels=False, cbar=False,cmap='viridis')
# Number of NULLs in these two columns:

print("fico_range_low :{}".format(data_consol[data_consol['fico_range_low'].isnull()==True]["id"].count()))

print("fico_range_high :{}".format(data_consol[data_consol['fico_range_high'].isnull()==True]["id"].count()))
#Create a new column for avg fico:

data_consol['fico_range_avg']=data_consol[['fico_range_low','fico_range_high']].mean(axis=1)
sns.distplot(data_consol['fico_range_avg'],kde=False, bins=50)
plt.figure(figsize=(10,6))

ax=sns.boxplot(x='loan_status', y='fico_range_avg',data=data_consol)

ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha='right') #show the x lable with an angel to avoid overlapping

X = data_consol[["annual_inc","annual_inc_joint","loan_status"]]
sns.boxplot(x="annual_inc",data=X)
#Looks like there're few outliners over $1MM:

X[X["annual_inc"]>1000000]["annual_inc"].count()
X[X["annual_inc"]<50000000]["annual_inc"]
X1 = X[X["annual_inc"]<500000]

X2 = X[X["annual_inc_joint"]<500000]

sns.distplot(X1["annual_inc"], kde=True)

sns.distplot(X2["annual_inc_joint"], kde=True)
plt.figure(figsize=(10,6))

sns.boxplot(y="annual_inc",data=X1,x="loan_status")
plt.figure(figsize=(10,6))

sns.boxplot(y="annual_inc_joint",data=X2,x="loan_status")
data_consol[data_consol["annual_inc_joint"].isnull()==True]["id"].count()
data_consol[data_consol["annual_inc"].isnull()==True]["id"].count()
data_consol["home_ownership"].unique()
plt.figure(figsize=(10,6))

sns.countplot(x="loan_status",data=data_consol,hue='home_ownership')
data_pivot=data_consol.pivot_table(values='id',aggfunc='count',columns='loan_status',index='home_ownership')

data_pivot
num_of_rows = data_pivot.sum()

num_of_rows
num_of_rows["Current"]
#What's the % distribution of home ownership in each loan status

for x in data_pivot.columns:

    data_pivot[x] = data_pivot[x].apply(lambda y:y/num_of_rows[x])
data_pivot
#How about the other way around?

#Loan default % in each homeownership bucket

data_pivot2=data_consol.pivot_table(values='id',aggfunc='count',index='loan_status',columns='home_ownership')

data_pivot2
num_of_rows = data_pivot2.sum()

num_of_rows
for x in data_pivot2.columns:

    data_pivot2[x] = data_pivot2[x].apply(lambda y:y/num_of_rows[x])

data_pivot2
plt.style.use('default')

plt.figure(figsize=(10,6))

data_pivot2.T.plot.bar(stacked=True)

plt.legend(bbox_to_anchor=(1, 1), loc="upper left", ncol=1)
sns.distplot(data_consol["int_rate"])
sns.boxplot(y='int_rate',data=data_consol, x='grade',order='A B C D E F G'.split())
plt.figure(figsize=(10,6))

sns.boxplot(y='int_rate',data=data_consol, x='loan_status',

            order=['Current','Fully Paid','Charged Off','Default','In Grace Period', 'Late (16-30 days)','Late (31-120 days)'])
def labeling(x):

    if x in ["Fully Paid","Current"]:

        return 1

    elif x in ["Charged Off","Default"]:

        return 0

    else:

        return 2 #Include: 'In Grace Period', 'Late (16-30 days)', 'Late (31-120 days)'
#Create a new column to assign the labels

data_consol["loan_label"] = data_consol["loan_status"].apply(labeling)
data_consol["loan_label"].value_counts()
#Define the list of columns to be included in the model input

column_list = ["fico_range_avg","annual_inc","home_ownership","int_rate","loan_label"]
#Take only the featuring columns, and filter for loans with status "1" or "0"

X = data_consol[column_list]

data_model = X[(X["loan_label"]==1) | (X["loan_label"]==0)]

del X
#Quick check on the loan_label

data_model["loan_label"].unique()
data_model.shape
#Create dummy columns for "home_owneship" column

home = pd.get_dummies(data_model["home_ownership"],drop_first=True)
data_model = pd.concat([data_model, home], axis=1)
data_model.drop(["home_ownership"],axis=1, inplace=True)
data_model
#Final data cleaning - remove any rows with NULL

data_model = data_model.dropna()
data_model.shape
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(data_model.drop('loan_label',axis=1), 

                                                    data_model['loan_label'], test_size=0.3)
y_train.value_counts()
y_test.value_counts()
#Training dataset:

from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()

logmodel.fit(X_train, y_train)
#Predictions

predit_logistic = logmodel.predict(X_test)
#Evaluation:

from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_test, predit_logistic))

print("\n")

print(confusion_matrix(y_test,predit_logistic))
from sklearn.tree import DecisionTreeClassifier

dtree = DecisionTreeClassifier()
dtree.fit(X_train, y_train)
predit_dtree = dtree.predict(X_test)
print(classification_report(y_test, predit_dtree))

print("\n")

print(confusion_matrix(y_test,predit_dtree))
from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators=100)

rfc.fit(X_train, y_train)
predit_rfc = rfc.predict(X_test)
print(classification_report(y_test, predit_rfc))

print("\n")

print(confusion_matrix(y_test,predit_rfc))