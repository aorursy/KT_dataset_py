import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split
heart_data = pd.read_csv("../input/heart-disease-detection-dataset/datasets_33180_43520_heart.csv")

heart_data_df = pd.DataFrame(heart_data)

heart_data_df
heart_data.target.value_counts()
heart_data_df.info()
heart_data_df.isnull().sum()
plt.figure(figsize=(6,3))

sns.countplot('target', data=heart_data_df, palette='terrain')

plt.xlabel("Sex (0 = No Disease, 1= Have Disease)")
sns.countplot('sex', data=heart_data_df, palette="magma")

plt.xlabel("Sex (0 = female, 1= male)")

plt.show()
FemaleCount = len(heart_data_df[heart_data_df.sex == 0])

MaleCount = len(heart_data_df[heart_data_df.sex == 1])

print("Percentage of Female Patients: {:.2f}%".format((FemaleCount / (len(heart_data_df.sex))*100)))

print("Percentage of Male Patients: {:.2f}%".format((MaleCount / (len(heart_data_df.sex))*100)))
#grouping Dataset by the desired o/p and other columns respective of their mean

heart_data_df.groupby('target').mean()
#Compute a simple cross tabulation of two (or more) factors. By default computes a frequency table of the factors unless

#an array of values and an aggregation function are passed.

pd.crosstab(heart_data_df.age, heart_data_df.target).plot(kind="bar",figsize=(20,6))

plt.title('Heart Disease Frequency for Age & Sex')

plt.xlabel('Age')

plt.ylabel('Frequency')

plt.show()
plt.scatter(heart_data_df.age[heart_data_df.target==1], heart_data_df.thalach[heart_data_df.target==1], c='red')

plt.scatter(heart_data_df.age[heart_data_df.target==0],heart_data_df.thalach[heart_data_df.target==0])

plt.legend(["Disease", "Not Disease"])

plt.xlabel("Age")

plt.ylabel("Maximum Heart Rate")

plt.show()
pd.crosstab(heart_data_df.sex,heart_data_df.target).plot(kind="bar",figsize=(15,6),color=['#800080','#FFFF00' ])

plt.title('Heart Disease Frequency for Sex')

plt.xlabel('Sex (0 = Female, 1 = Male)')

plt.legend(["Haven't Disease", "Have Disease"])

plt.ylabel('Frequency')

plt.show()
pd.crosstab(heart_data_df.cp,heart_data_df.target).plot(kind="bar",figsize=(15,6),color=['#7D0552','#05517D' ])

plt.title('Heart Disease Frequency According To Chest Pain Type')

plt.xlabel('Chest Pain Type')

plt.ylabel('Frequency of Disease or Not')

plt.show()
sns.heatmap(heart_data_df.corr())
plt.figure(figsize=(12,10))

sns.catplot('sex', 'age', hue='cp', data = heart_data_df, kind='bar')
pd.crosstab(heart_data_df.chol,heart_data_df.target).plot(kind="area",figsize=(15,6),color=['#8C001A','#347C17' ])

plt.title('Heart Disease Frequency According To Cholestrol')

plt.xlabel('Cholestrol')

plt.ylabel('Frequency of Disease or Not')

plt.show()

pd.crosstab(heart_data_df.restecg,heart_data_df.sex).plot(kind="barh",figsize=(15,6),color=['#50EBEC','#ED5185' ])

plt.title('Heart Disease Frequency based on Rest ECG')

plt.xlabel('Rest_ECG')

plt.ylabel('Frequency of Disease or Not')

plt.show()
heart_data_df.drop(['slope', 'thal'], axis=1, inplace=True)

heart_data_df.head(10)
x = heart_data_df.drop(['target'], axis=1)

y = heart_data_df['target'].values
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=32)
from sklearn.linear_model import LogisticRegression

log = LogisticRegression()

log.fit(X_train,y_train)
accuracies = {}
acc = log.score(X_test,y_test)*100

accuracies['Logistic Regression'] = acc

print("Test Accuracy {:.2f}%".format(acc))
from sklearn.tree import DecisionTreeClassifier

dc = DecisionTreeClassifier()

dc.fit(X_train, y_train)



acc = dc.score(X_test, y_test)*100

accuracies['Decision Tree'] = acc

print("Decision Tree Test Accuracy {:.2f}%".format(acc))
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators = 1000, random_state = 1)

rf.fit(X_train, y_train)



acc = rf.score(X_test,y_test)*100

accuracies['Random Forest'] = acc

print("Random Forest Algorithm Accuracy Score : {:.2f}%".format(acc))
colors = ["purple", "magenta", "cyan"]



sns.set_style("whitegrid")

plt.figure(figsize=(16,5))

plt.yticks(np.arange(0,100,10))

plt.ylabel("Accuracy %")

plt.xlabel("Algorithms")

sns.barplot(x=list(accuracies.keys()), y=list(accuracies.values()), palette=colors)

plt.show()

log_pred = log.predict(X_test)

dt_pred = dc.predict(X_test)

rand_for_pred = rf.predict(X_test)

print(log_pred)

print(dt_pred)

print(rand_for_pred)
from sklearn.metrics import confusion_matrix



cm_log = confusion_matrix(y_test, log_pred )

cm_dt = confusion_matrix(y_test, dt_pred)

cm_rf = confusion_matrix(y_test,rand_for_pred)
plt.figure(figsize=(24,12))



plt.suptitle("Confusion Matrixes",fontsize=24)

plt.subplots_adjust(wspace = 0.4, hspace= 0.4)



plt.subplot(2,3,1)

plt.title("Logistic Regression Confusion Matrix")

sns.heatmap(cm_log,annot=True,cmap="YlOrBr",fmt="d",cbar=False, annot_kws={"size": 24})



plt.subplot(2,3,2)

plt.title("Decision Tree Classifier Confusion Matrix")

sns.heatmap(cm_dt,annot=True,cmap="YlOrRd",fmt="d",cbar=False, annot_kws={"size": 24})



plt.subplot(2,3,3)

plt.title("Random Forest Confusion Matrix")

sns.heatmap(cm_rf,annot=True,cmap="YlGnBu",fmt="d",cbar=False, annot_kws={"size": 24})



plt.show()