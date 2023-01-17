from typing import Any, Union
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import warnings
import re
from statistics import mode
from pandas import DataFrame, Series
from pandas.io.parsers import TextFileReader
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_selector, make_column_transformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.impute import SimpleImputer
import seaborn as sns
warnings.filterwarnings('ignore')
plt.style.use('fivethirtyeight')

# X data
X_train = pd.read_csv('../input/titanic/train.csv')
X_test = pd.read_csv('../input/titanic/test.csv')

# y data
y_train = X_train['Survived']
# y_test = pd.read_csv("../data/gender_submission.csv")
def display_heatmap_na(df, mode=1):
    if mode == 1:
        plt.style.use('seaborn')
        plt.figure()
        sns.heatmap(df.isnull(), yticklabels = False, cmap='plasma')
        plt.title('Null Values in Training Set')
    else:
        print(X_train.isnull().sum())
X_train.head()
print(f'Unique Values in Pclass :{X_train.Pclass.unique()}')
print(f'Unique Values in SibSp :{X_train.SibSp.unique()}')
print(f'Unique Values in Embarked :{X_train.Embarked.unique()}')
plt.figure(figsize=(15,5))
plt.subplot(1,2,1)
sns.countplot(X_train.Survived)
plt.title('Number of passenger Survived');

plt.subplot(1,2,2)
sns.countplot(x="Survived", hue="Sex", data=X_train)
plt.title('Number of passenger Survived');
display_heatmap_na(X_train)
display_heatmap_na(X_test)
plt.figure(figsize=(15,5))
plt.style.use('fivethirtyeight')

plt.subplot(1,2,1)
sns.countplot(X_train['Pclass'])
plt.title('Count Plot for PClass');

plt.subplot(1,2,2)
sns.countplot(x="Survived", hue="Pclass", data=X_train)
plt.title('Number of passenger Survived');
pclass1 = X_train[X_train.Pclass == 1]['Survived'].value_counts(normalize=True).values[0]*100
pclass2 = X_train[X_train.Pclass == 2]['Survived'].value_counts(normalize=True).values[1]*100
pclass3 = X_train[X_train.Pclass == 3]['Survived'].value_counts(normalize=True).values[1]*100

print("Lets look at some satistical data!\n")
print("Pclaas-1: {:.1f}% People Survived".format(pclass1))
print("Pclaas-2: {:.1f}% People Survived".format(pclass2))
print("Pclaas-3: {:.1f}% People Survived".format(pclass3))
X_train['Age'].plot(kind='hist')
# outliers detected

X_train['Age'].hist(bins=40)
plt.title('Age Distribution');
# set plot size
plt.figure(figsize=(15, 3))

# plot a univariate distribution of Age observations
sns.distplot(X_train[(X_train["Age"] > 0)].Age, kde_kws={"lw": 3}, bins = 50)

# set titles and labels
plt.title('Distribution of passengers age',fontsize= 14)
plt.xlabel('Age')
plt.ylabel('Frequency')
# clean layout
# plt.tight_layout()
plt.figure(figsize=(15, 3))

# Draw a box plot to show Age distributions with respect to survival status.
sns.boxplot(y = 'Survived', x = 'Age', data = X_train,
     palette=["#3f3e6fd1", "#85c6a9"], fliersize = 0, orient = 'h')

# Add a scatterplot for each category.
sns.stripplot(y = 'Survived', x = 'Age', data = X_train,
     linewidth = 0.6, palette=["#3f3e6fd1", "#85c6a9"], orient = 'h')

plt.yticks( np.arange(2), ['drowned', 'survived'])
plt.title('Age distribution grouped by surviving status (train data)',fontsize= 14)
plt.ylabel('Passenger status after the tragedy')
plt.tight_layout()
plt.figure(figsize=(15,5))
plt.subplot(1,2,1)
sns.countplot(X_train['SibSp'])
plt.title('Number of siblings/spouses aboard');

plt.subplot(1,2,2)
sns.countplot(x="Survived", hue="SibSp", data=X_train)
plt.legend(loc='right')
plt.title('Number of passenger Survived');
plt.figure(figsize=(15,5))
plt.subplot(1,2,1)
sns.countplot(X_train['Embarked'])
plt.title('Number of Port of embarkation');

plt.subplot(1,2,2)
sns.countplot(x="Survived", hue="Embarked", data=X_train)
plt.legend(loc='right')
plt.title('Number of passenger Survived');
sns.heatmap(X_train.corr(), annot=True)
plt.title('Corelation Matrix');
corr = X_train.corr()
sns.heatmap(corr[((corr >= 0.3) | (corr <= -0.3)) & (corr != 1)], annot=True, linewidths=.5, fmt= '.2f')
plt.title('Configured Corelation Matrix');
sns.catplot(x="Embarked", y="Fare", kind="violin", inner=None,
            data=X_train, height = 6, order = ['C', 'Q', 'S'])
plt.title('Distribution of Fare by Embarked')
plt.tight_layout()
sns.catplot(x="Pclass", y="Fare", kind="swarm", data=X_train, height = 6)

plt.tight_layout()
sns.catplot(x="Pclass", y="Fare",  hue = "Survived", kind="swarm", data=X_train,
                                    palette=["#3f3e6fd1", "#85c6a9"], height = 6)
plt.tight_layout()
X_train['Fare'].nlargest(10).plot(kind='bar', title = '10 largest Fare', color = ['#C62D42', '#FE6F5E']);
plt.xlabel('Index')
plt.ylabel('Fare')
X_train['Age'].nlargest(10).plot(kind='bar', color = ['#5946B2','#9C51B6']);
plt.title('10 largest Ages')
plt.xlabel('Index')
plt.ylabel('Ages');
X_train['Age'].nsmallest(10).plot(kind='bar', color = ['#A83731','#AF6E4D'])
plt.title('10 smallest Ages')
plt.xlabel('Index')
plt.ylabel('Ages');