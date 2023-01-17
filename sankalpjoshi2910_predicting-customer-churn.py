import pandas as pd

import numpy as np

import matplotlib as mpl

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

import warnings

warnings.filterwarnings("ignore")
df = pd.read_csv("../input/Churn_Modelling.csv")

df.head()
# Column Headings



print(list(df.columns))
df.set_index('RowNumber')

df.RowNumber.count()
# so we have 1000 rows of data with 14 column headings
# Further Information about data

df.info()
# 10,000 rows with no NaN value
#  Different Column Exploration

print(df["Geography"].unique())

print(df["Tenure"].unique())

print(df["NumOfProducts"].unique())
#Looking at the results we can say that we have 3 main geographical region in our dataset,i.e.,France,Spain and Germany.

#Following: [ 2  1  8  7  4  6  3 10  5  9  0] tenure

#And 4 different Number of Products
# Exited Count



pd.value_counts(df['Exited'].values, sort=False)
# IsActive Count

df["IsActiveMember"].value_counts()
#So we have approximately 8000 costumer who does not Exited and aprrox. 2000 count that Exited

#And Aprroximately 5000 are still active membert and approx. 4.8k are not an active member
# Stastical Analysis Of Dataset

df.describe()
# Costumer Left



left=df.groupby('Exited').count()

plt.figure(figsize=(5,5))

plt.bar(left.index.values, left['RowNumber'])

plt.xlabel('Left')

plt.ylabel('Number of Costumer')

plt.show()
#  Number of costumer from each Region



region=df.groupby('Geography').count()

plt.bar(region.index.values, region['RowNumber'])

plt.xlabel('Region')

plt.ylabel('Number of Costumer')

plt.show()
# Most of the costumer are from France and number of costumer from Germany and Spain are equal.
# Total count of costumer based on Gender



gender=df.groupby('Gender').count()

plt.bar(gender.index.values, gender['RowNumber'])

plt.xlabel('Age')

plt.ylabel('Number of Costumer')

plt.show()
# we have more male costumer than the female costumer
# Total count of costumer based on Different AgeGroup



age=df.groupby('Age').count()

plt.figure(figsize=(12,6))

plt.bar(age.index.values, age['RowNumber'])

plt.xlabel('Age')

plt.ylabel('Number of Costumer')

plt.show()
# most of the Costumer lies between age group of 25-50
# Total count of costumer based on Credit Score



cred_sco=df.groupby('CreditScore').count()

plt.figure(figsize=(12,6))

plt.bar(cred_sco.index.values, cred_sco['RowNumber'])

plt.xlabel('Credit Score')

plt.ylabel('Number of Employees')

plt.show()

print("Max of Credit Score:", df["CreditScore"].max())

count3=0

for i in df["CreditScore"]:

    if i==850:

        count3=count3+1

print("Costumers with max Credit Score:", count3)
# Subplots For various parameters 



features=[ 'Geography', 'Gender','Tenure', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'Exited']

fig=plt.subplots(figsize=(15,15))

for i, j in enumerate(features):

    plt.subplot(3, 3, i+1)

    plt.subplots_adjust(hspace = 1.0)

    sns.countplot(x=j,data = df)

    plt.title("No. of costumers")
# plot based on Number of costumer who exited based on differnt parameters

fig=plt.subplots(figsize=(15,15))

for i, j in enumerate(features):

    plt.subplot(4, 2, i+1)

    plt.subplots_adjust(hspace = 1.0)

    sns.countplot(x=j,data = df, hue='Exited')

    plt.title("No. of costumer")
# In the plots shown above the Orange line represent the count of the costumer who "Exited"  

# title shows the Parameter on which it is counted 
# Data Preprocessing for developing model
# Keeping only those column which are useful for prediction 

#like RowNumber and CostumerID wont affect the prediction.

column_to_keep=['CreditScore', 'Geography', 'Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary', 'Exited']

df=df[column_to_keep]

df.head()
# Representing Gender and Age Value in Numeric 

df["Geography"][df["Geography"]=="France"]=1

df["Geography"][df["Geography"]=="Spain"]=2

df["Geography"][df["Geography"]=="Germany"]=3



df["Gender"][df["Gender"]=="Female"]=1

df["Gender"][df["Gender"]=="Male"]=2



df.head()
#finding Correlation between parametrs 

corr=df.corr(method ='pearson') 

corr
# Correlation Heatmap Using seaborn library

sns.heatmap(corr, xticklabels=corr.columns,yticklabels=corr.columns)
# Excluding "HasCrCard"  



column_to_keep2=['CreditScore', 'Geography', 'Gender', 'Age',"HasCrCard", 'Tenure', 'Balance', 'NumOfProducts', 'IsActiveMember', 'EstimatedSalary', 'Exited']

df=df[column_to_keep2]
# Preparing Data For Splitting And Testing

X=df.iloc[:,0:10]

Y=df.iloc[:,10]
# Splitting Data In Training Set and Test Set 



from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y,test_size=0.05,random_state = 0)

Y_train=Y_train.astype('int')
# Feature Scaling



from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)
# Logistic Regression Algorithm



from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression(random_state = 0)

classifier.fit(X_train,Y_train)
# Prediction

Y_pred = classifier.predict(X_test)
# Confusion Matrix For Evaluation

from sklearn.metrics import confusion_matrix

confusion_matrix = confusion_matrix(Y_test, Y_pred)

print(confusion_matrix)
#Import scikit-learn metrics module for accuracy calculation

from sklearn import metrics

# Model Accuracy, how often is the classifier correct?

print("Accuracy:",metrics.accuracy_score(Y_test, Y_pred))

# Model Precision

print("Precision:",metrics.precision_score(Y_test, Y_pred))

# Model Recall

print("Recall:",metrics.recall_score(Y_test, Y_pred))
from sklearn.neighbors import KNeighborsClassifier

classifier2 = KNeighborsClassifier(n_neighbors = 15, metric = 'minkowski', p = 2)

classifier2.fit(X_train, Y_train)
# Prediction

Y_pred2 = classifier2.predict(X_test)

# Confusion Matrix for evaluation

from sklearn.metrics import confusion_matrix

confusion_matrix = confusion_matrix(Y_test, Y_pred2)

print(confusion_matrix)
#Import scikit-learn metrics module for accuracy calculation

from sklearn import metrics

# Model Accuracy, how often is the classifier correct?

print("Accuracy:",metrics.accuracy_score(Y_test, Y_pred2))

# Model Precision

print("Precision:",metrics.precision_score(Y_test, Y_pred2))

# Model Recall

print("Recall:",metrics.recall_score(Y_test, Y_pred2))