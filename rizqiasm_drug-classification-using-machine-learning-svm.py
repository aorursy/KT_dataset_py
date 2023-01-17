import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
data = pd.read_csv('../input/drug-classification/drug200.csv')

data
data.describe()
data.dtypes
data1 = data.Sex.value_counts()

print(data1)

data2 = data.BP.value_counts()

print(data2)
data3 = data.Cholesterol.value_counts()

print(data3)

data4 = data.Drug.value_counts()

print(data4)
data.replace("drugX", "DrugX", inplace = True)

data.replace("drugA", "DrugA", inplace = True)

data.replace("drugC", "DrugC", inplace = True)

data.replace("drugB", "DrugB", inplace = True)

data
data.isnull().sum()
ax = sns.barplot(data1.index, data1.values, alpha = 0.8)

ax.set_xticklabels(ax.get_xticklabels(), ha ='right')

plt.title('Type of Sex')

plt.xlabel('Sex')

plt.ylabel('Total Type of Sex')
ax = sns.barplot(data2.index, data2.values, alpha = 0.8)

ax.set_xticklabels(ax.get_xticklabels(), ha ='right')

plt.title('Blood Preassure')

plt.xlabel('Blood Preassure')

plt.ylabel('Total of Blood Preassure')
ax = sns.barplot(data3.index, data3.values, alpha = 0.8)

ax.set_xticklabels(ax.get_xticklabels(), ha ='right')

plt.title('Type of Cholesterol')

plt.xlabel('Cholesterol')

plt.ylabel('Total of Cholesterol')
ax = sns.barplot(data4.index, data4.values, alpha = 0.8)

ax.set_xticklabels(ax.get_xticklabels(), ha ='right')

plt.title('Type of Drug')

plt.xlabel('Drug')

plt.ylabel('Total of Drug')
ax = sns.boxplot(data=data, orient="h", palette="Set2")

plt.title('Box Plot for each Variable')
n = 0 

for x in ["Age", "Na_to_K"]:

    n = 1

    plt.subplot(2 , 2 , n)

    plt.subplots_adjust(hspace =0.5, wspace = 0.5)

    sns.distplot(data[x] , kde=True, bins = 20)

    plt.title(x)

    plt.show()
ax = sns.set(style="ticks", color_codes='pallete')

ax = sns.pairplot(data, hue="Drug")
Name= ['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']



X = data[Name]

Y = data['Drug']
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=0)
X_train = pd.get_dummies(X_train,columns=Name,drop_first=True)

X_test = pd.get_dummies(X_test,columns=Name,drop_first=True)
from sklearn.svm import SVC



svm = SVC()

svm.fit(X_train, Y_train)

print('Accuracy of SVM classifier on this training is {:.2f}'

     .format(svm.score(X_train, Y_train)))