#Import libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
#Reading csv file

df=pd.read_csv("../input/creditcard.csv")
#Checking the variables

df.head()
#Check null values

df.isnull().sum()
#Checking the available columns

df.columns
#Dimensions of it

df.shape
#Counting all unique values of column "sentiment" from dataset 

df["Class"].value_counts()
# Check Class variables that has 0 value for Genuine transactions and 1 for Fraud

print("Class as pie chart:")

fig, ax = plt.subplots(1, 1)

ax.pie(df.Class.value_counts(),autopct='%1.1f%%', labels=['Genuine','Fraud'], colors=['Black','Blue'])

plt.axis('equal')

plt.ylabel('')
import seaborn as sns

sns.countplot('Class', data=df)

print('Frauds: ', round(df['Class'].value_counts()[1] / len(df) * 100, 2), '%')
#I dnt want time column for this prediction.So,i am dropping it

df = df.drop('Time', 1)
#Checking dataset after droping "time column".

df.head()
df.describe()
df.shape

df.info()
#separating dependent and independent variable.

X = df.iloc[:, :29].values

y = df.iloc[:,-1].values

print(X)

print(y)
X.shape
y.shape


#Splitiing dataset into training_set and testing_set.

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2 ,random_state = 0)

#Scaling the features

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)

# Fitting Logistic Regression to the Training set

from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression(random_state = 0)

classifier.fit(X_train, y_train)



# Predicting the Test set results

y_pred = classifier.predict(X_test)
# Making the Confusion Matrix

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

print(cm)
Total =56852+9+37+64
#accuracy=(TP+TN)/total

Accuracy = (56852+64)/Total



print(Accuracy)
#Error_rate=1-accuracy

Error_rate = 1-Accuracy

print(Error_rate)
#Recall=TP/FN+TP

Recall = 64/(64+37)

print(Recall)
#Precision=TP/FP+TP



Precision = 64/(9+64)

print(Precision)
#Visualize the confusion matrix.....

plt.figure(figsize=(20,10))

plt.subplot(2,4,3)

plt.title("LogisticRegression_cm")

sns.heatmap(cm,annot=True,cmap="Wistia",fmt="d",cbar=False)