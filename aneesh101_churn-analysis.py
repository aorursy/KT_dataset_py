# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as  sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
org=pd.read_csv("/kaggle/input/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv")

org.head()
#making a copy of the original dataset

df=org.copy()



#Dropping the customerID column 

df.drop('customerID', axis=1, inplace= True)
assert 'customerID' not in df.columns
df.info()
df.describe()
#Converting all possible values in the TotalCharges Column into numeric and converting rest of them into missing values

df['TotalCharges']=df['TotalCharges'].apply(pd.to_numeric, errors='coerce')



#Checking if there are any null values now

df.isnull().sum().sum()
df[df.isnull().any(axis=1)]
assert df['TotalCharges'].dtype=='float64'
df.loc[(pd.isnull(df.TotalCharges)), 'TotalCharges'] = df.MonthlyCharges
assert len(df.loc[(pd.isnull(df.TotalCharges))])==0
#extracting all categorical columns

categorical_columns=df.select_dtypes(include='object').columns.tolist()

print(categorical_columns)
#Made a list of number of unique values allowed in each column according to the details provided to us

x=[2,2,2,2,3,3,3,3,3,3,3,3,3,2,4,2]

y=[]



#Number of unique elements present in each column

for i in categorical_columns:

    y.append(len(df[i].unique()))



#Comparing Values

for i,j in zip(x,y):

    print(i,j)
# Summarize our dataset 

print ("Rows     : " ,df.shape[0])

print ("Columns  : " ,df.shape[1])

print ("\nFeatures : \n" ,df.columns.tolist())

print ("\nMissing values :  ", df.isnull().sum().values.sum())

print ("\nUnique values :  \n",df.nunique())
df['Churn'].value_counts()/df.shape[0]
labels = df['Churn'].value_counts(sort = True).index

sizes = df['Churn'].value_counts(sort = True)



colors = ["green","red"]

explode = (0.05,0)  # explode 1st slice

 

plt.figure(figsize=(7,7))

# Plot

plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=90,)



plt.title('Customer Churn Breakdown')

plt.show()
fig, axes = plt.subplots(1, 3, figsize=(20, 5))

sns.distplot( df["tenure"] , color="skyblue", ax=axes[0])

sns.distplot(df['MonthlyCharges'],color='orange',ax=axes[1])

sns.distplot(df['TotalCharges'],color='green',ax=axes[2])

fig.suptitle('Histogram of Numerical Columns')
fig, axes = plt.subplots(3, 2, figsize=(20, 20))

fig1=sns.countplot( df["gender"] ,ax=axes[0,0])

fig2=sns.countplot( df["SeniorCitizen"] , ax=axes[0,1])

fig3=sns.countplot( df["Contract"] , ax=axes[2,0])

fig4=sns.countplot( df["PaymentMethod"] , ax=axes[2,1])

fig5=sns.countplot( df["Partner"] , ax=axes[1,0])

fig6=sns.countplot( df["Dependents"] , ax=axes[1,1])



figures=[fig1,fig2,fig3,fig4,fig5,fig6]



for graph in figures:

    graph.set_xticklabels(graph.get_xticklabels(),rotation=90)

    

    for p in graph.patches:

        height = p.get_height()

        graph.text(p.get_x()+p.get_width()/2., height,height ,ha="center")





fig.suptitle('')
fig, axes = plt.subplots(1, 2, figsize=(12, 7))

g = sns.violinplot(x="Churn", y = "MonthlyCharges",data = df, palette = "Pastel1",ax=axes[0])

g = sns.violinplot(x="Churn", y = "tenure",data = df, palette = "Pastel1",ax=axes[1])

fig, axes = plt.subplots(3, 2, figsize=(20, 20))

fig1=sns.countplot( x=df["Churn"],hue=df["gender"] ,ax=axes[0,0])

fig2=sns.countplot( x=df["Churn"],hue=df["SeniorCitizen"] , ax=axes[0,1])

fig3=sns.countplot( x=df["Churn"],hue=df["Contract"] , ax=axes[2,0])

fig4=sns.countplot( x=df["Churn"],hue=df["PaymentMethod"] , ax=axes[2,1])

fig5=sns.countplot( x=df["Churn"],hue=df["Partner"] , ax=axes[1,0])

fig6=sns.countplot( x=df["Churn"],hue=df["Dependents"] , ax=axes[1,1])



figures=[fig1,fig2,fig3,fig4,fig5,fig6]



for graph in figures:

    graph.set_xticklabels(graph.get_xticklabels(),rotation=90)

    

    for p in graph.patches:

        height = p.get_height()

        graph.text(p.get_x()+p.get_width()/2., height,round((height/7043)*100,2) ,ha="center")





fig.suptitle('')
#First Create a dataset which only has categorical columns for the cross-tab



#Method 1: Dropping All Numerical Columns or adding All Categorical Columns MANUALLY

df_cat=df.drop(['MonthlyCharges', 'TotalCharges', 'tenure'], axis=1)

print(df_cat.shape)



#Method 2: Create a Method to automatically parse through all columns and recognise categorical columns

cat_cols = df.nunique()[df.nunique() < 6].keys().tolist()

cat_cols = [x for x in cat_cols]

df_cat=df[cat_cols]

print(df_cat.shape)

summary = pd.concat([pd.crosstab(df_cat[x], df_cat.Churn) for x in df_cat.columns[:-1]], keys=df_cat.columns[:-1])

summary['Churn_Percentage'] = summary['Yes'] / (summary['No'] + summary['Yes'])
#Lets check cases where more than 1/3rd of the customers have left

summary[summary['Churn_Percentage']>0.33]
#Creating a Predicted Values Column

df['PCharges']=df['MonthlyCharges']*df['tenure']

#Creating a Column to calculate Absolute Percentage Difference between predicted and actual values

df['PDifference']=(((df['PCharges']-df['TotalCharges'])/df['TotalCharges'])*100)



fig, axes = plt.subplots(1, 2, figsize=(10, 5))

sns.distplot( df[df['Churn']=="No"]["PDifference"] , color="green",ax=axes[0])

sns.distplot( df[df['Churn']=="Yes"]["PDifference"] , color="red",ax=axes[1])

df['PDifference'].describe()
df.drop(["PCharges"],axis=1,inplace=True)
from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import StandardScaler



target_col = ["Churn"]



#numerical columns

num_cols = [x for x in df.columns if x not in cat_cols + target_col]



#Binary columns with 2 values

bin_cols = df.nunique()[df.nunique() == 2].keys().tolist()



#Columns more than 2 values

multi_cols = [i for i in cat_cols if i not in bin_cols]



#Label encoding Binary columns

le = LabelEncoder()

for i in bin_cols :

    df[i] = le.fit_transform(df[i])

    

#Duplicating columns for multi value columns

df = pd.get_dummies(data = df, columns = multi_cols )

df.head()
#Scaling Numerical columns

std = StandardScaler()



# Scale data

scaled = std.fit_transform(df[num_cols])

scaled = pd.DataFrame(scaled,columns=num_cols)



#dropping original values merging scaled values for numerical columns

df_telcom_og = df.copy()

df = df.drop(columns = num_cols,axis = 1)

df = df.merge(scaled, left_index=True, right_index=True, how = "left")



#churn_df.info()

df.head()
from sklearn.model_selection import train_test_split



# We remove the label values from our training data

X = df.drop(['Churn'], axis=1).values



# We assigned those label values to our Y dataset

y = df['Churn'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=109)
from sklearn.naive_bayes import GaussianNB

from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report

from sklearn import metrics



print("Gaussian Naive Bayes Classifier Results")

#Create a Gaussian Classifier

gnb = GaussianNB()



#Train the model using the training sets

gnb.fit(X_train, y_train)



#Predict the response for test dataset

y_pred = gnb.predict(X_test)

print(confusion_matrix(y_test, y_pred))

print(classification_report(y_test, y_pred))

print(metrics.accuracy_score(y_test,y_pred))
from sklearn.tree import DecisionTreeClassifier



parameters = {'max_depth':[1, 5, 10, 50],'min_samples_split':[5, 10, 100, 500]}

dec = DecisionTreeClassifier()

clf = GridSearchCV(dec, parameters, cv=3, scoring='accuracy',return_train_score=True)

clf.fit(X_train, y_train)

results = pd.DataFrame.from_dict(clf.cv_results_)

results_sort = results.sort_values(['mean_test_score'])

results_sort.tail()
print("Decision Tree Classifier Results")

#Create a Gaussian Classifier

dec = DecisionTreeClassifier(max_depth=5,min_samples_split=500)



#Train the model using the training sets

dec.fit(X_train, y_train)



#Predict the response for test dataset

y_pred = dec.predict(X_test)

print(confusion_matrix(y_test, y_pred))

print(classification_report(y_test, y_pred))

print(metrics.accuracy_score(y_test,y_pred))
from xgboost.sklearn import XGBClassifier



parameters = {'max_depth':[1, 5, 10, 50],'min_child_weight':range(1,6,2)}

dec = XGBClassifier()

clf = GridSearchCV(dec, parameters, cv=3, scoring='accuracy',return_train_score=True)

clf.fit(X_train, y_train)

results = pd.DataFrame.from_dict(clf.cv_results_)

results_sort = results.sort_values(['mean_test_score'])

results_sort.tail()
print("XGBoost Classifier Results")

#Create a Gaussian Classifier

dec = XGBClassifier(max_depth=1,min_child_weight=3)



#Train the model using the training sets

dec.fit(X_train, y_train)



#Predict the response for test dataset

y_pred = dec.predict(X_test)

print(confusion_matrix(y_test, y_pred))

print(classification_report(y_test, y_pred))

print(metrics.accuracy_score(y_test,y_pred))
from sklearn.calibration import CalibratedClassifierCV



gnb=CalibratedClassifierCV(gnb,method='isotonic')

gnb.fit(X_train,y_train)



final_pred=[]

for dp in X_test:

    dp=dp.reshape(1, -1)

    gnb_prob=gnb.predict(dp)

    xgb_prob=dec.predict(dp)

    if gnb_prob[0]!=xgb_prob[0]:

        if gnb_prob[0]==0:

            final_pred.append(1)

        else:

            prob_1=gnb.predict_proba(dp)[0][1]

            prob_0=dec.predict_proba(dp)[0][0]

            if prob_1>=prob_0:

                final_pred.append(1)

            else:

                final_pred.append(0)

    else:

        final_pred.append(gnb_prob[0])

    
y_pred=np.asarray(final_pred)

print(confusion_matrix(y_test, y_pred))

print(classification_report(y_test, y_pred))

print(metrics.accuracy_score(y_test,y_pred))
