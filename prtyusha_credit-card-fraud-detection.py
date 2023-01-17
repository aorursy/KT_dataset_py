# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input/creditcardfraud'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#Importing and seeing data

dataset = pd.read_csv('../input/creditcardfraud/creditcard.csv')

print(dataset)
# Understanding data

dataset.describe()

dataset.info()



import matplotlib.pyplot as plt

plt.scatter(dataset["Time"], dataset["Amount"], color = 'red') 



#Barplot of frauds and valids

import seaborn as sns

sns.countplot(x = 'Class', data = dataset)



#%Frauds

counts = dataset.Class.value_counts()

frauds = counts[1]

valids = counts[0]

perc_frauds = frauds*100/(frauds+valids)

print("% of total frauds is {:.3f}",format(perc_frauds))

print(frauds)
#Correlation of variables heat map

correlation_matrix = dataset.corr()

sns.heatmap(correlation_matrix, cmap = 'coolwarm', center = 0)

#DataPreprocessing_Scaling time, amount

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

sc2 = StandardScaler()

dataset[['Time']] = sc.fit_transform(dataset[['Time']])

dataset[['Amount']] = sc2.fit_transform(dataset[['Amount']])

#Making a new dataset with 1:1 fraud and non_frauds

# we know that no.of frauds = 492



fraud_data = dataset.loc[dataset['Class'] == 1]

non_fraud_data = dataset.loc[dataset['Class'] == 0]



#Selecting 492 rows from non_fraud_data

selected_non_fraud_data = non_fraud_data.sample(492)



#Combining to form new dataset

new_data = pd.concat([fraud_data, selected_non_fraud_data])

sns.countplot(x = 'Class', data = new_data)
#Checking correlation of different variables to choose what to retain



correl = new_data.corr()

class_correl = correl[['Class']]

negative = class_correl[class_correl.Class< -0.5]

positive = class_correl[class_correl.Class> 0.5]

print("negative")

print(negative)

print("positive")

print(positive)
#visualizing the features with high negative correlation

f, axes = plt.subplots(nrows=2, ncols=4, figsize=(26,16))



f.suptitle('Features With High Negative Correlation', size=35)

sns.boxplot(x="Class", y="V3", data=new_data, ax=axes[0,0])

sns.boxplot(x="Class", y="V9", data=new_data, ax=axes[0,1])

sns.boxplot(x="Class", y="V10", data=new_data, ax=axes[0,2])

sns.boxplot(x="Class", y="V12", data=new_data, ax=axes[0,3])

sns.boxplot(x="Class", y="V14", data=new_data, ax=axes[1,0])

sns.boxplot(x="Class", y="V16", data=new_data, ax=axes[1,1])

sns.boxplot(x="Class", y="V17", data=new_data, ax=axes[1,2])

f.delaxes(axes[1,3])
#visualizing the features w high positive correlation

f, axes = plt.subplots(nrows=1, ncols=2, figsize=(18,9))



f.suptitle('Features With High Positive Correlation', size=20)

sns.boxplot(x="Class", y="V4", data=new_data, ax=axes[0])

sns.boxplot(x="Class", y="V11", data=new_data, ax=axes[1])
#Removing Extreme Outliers

Q25 = new_data.quantile(0.25)

Q75 = new_data.quantile(0.75)

IQR = Q75-Q25

threshold = 2.5*IQR

print(threshold)

final_data = new_data[~((new_data < (Q25 - threshold)) |(new_data > (Q75 + threshold))).any(axis=1)]

print("Length of data before:",len(new_data))

print("Length of data after:", len(final_data))

print("Extreme outliers:", len(new_data)-len(final_data))

# train test split

X = final_data.iloc[:, :30].values

y = final_data.iloc[:, -1].values



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train.shape)

print(X_test.shape)

print(y_train.shape)

print(y_test.shape)
#importing 

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score



from sklearn.metrics import roc_auc_score

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix



from sklearn.linear_model import LogisticRegression

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.svm import SVC

from xgboost import XGBClassifier

from sklearn.ensemble import RandomForestClassifier
#fitting models

models = []



models.append(('LR', LogisticRegression()))

models.append(('LDA', LinearDiscriminantAnalysis()))

models.append(('KNN', KNeighborsClassifier()))

models.append(('CART', DecisionTreeClassifier()))

models.append(('SVM', SVC()))

models.append(('XGB', XGBClassifier()))

models.append(('RF', RandomForestClassifier()))



#testing models



results = []

names = []



for name, model in models:

    kfold = KFold(n_splits=10)

    cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring='roc_auc')

    results.append(cv_results)

    names.append(name)

    summary = '%s: %f (%f)' % (name, cv_results.mean(), cv_results.std())

    print(summary)

#Compare Algorithms



fig = plt.figure(figsize=(12,10))

plt.title('Comparison of Classification Algorithms')

plt.xlabel('Algorithm')

plt.ylabel('ROC-AUC Score')

plt.boxplot(results)

ax = fig.add_subplot(111)

ax.set_xticklabels(names)

plt.show()
#Predicting Results and confusion matrix using RandomForestClassifier

classifier = RandomForestClassifier()

classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

cm = confusion_matrix(y_pred, y_test)

cm

    

              

              

    


