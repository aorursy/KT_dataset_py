import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import pandas as pd

import numpy as np

import missingno as msno

import matplotlib.pyplot as plt

import seaborn as sns



from scipy.stats import norm

from sklearn.preprocessing import StandardScaler

from scipy import stats

import warnings

warnings.filterwarnings('ignore')

%matplotlib inline
df_train = pd.read_csv("/kaggle/input/train.csv")

df_test = pd.read_csv("/kaggle/input/test.csv")

df_sample_submission = pd.read_csv("/kaggle/input/submission.csv")
df_sample_submission
display("Train File",df_train.head(10))

display("Test File",df_test.head(10))
display("Train File",df_train.columns)

display("Test File",df_test.columns)
display("Train File",df_train.dtypes)

display("Test File",df_test.dtypes)
print("Train File Shape:",df_train.shape)

print("Test File Shape:",df_test.shape)
display("Train File",df_train.isnull().sum())

display("Test File",df_test.isnull().sum())
display("Train File",msno.bar(df_train))
display("Test File",msno.bar(df_test))
def missing_data_percentage(a):

    total = a.isnull().sum().sort_values(ascending=False)

    percent = (a.isnull().sum()/a.isnull().count()).sort_values(ascending=False)

    percent = percent*100

    b = pd.concat([total, percent], axis=1, keys=['Total NULL values', 'Percentage'])

    return b



print("TRAIN FILE\n")

display(missing_data_percentage(df_train))

print("\n TEST FILE \n")

display(missing_data_percentage(df_test))
display("Train File")

a = missing_data_percentage(df_train)

f, ax = plt.subplots(figsize=(10, 8))

plt.xticks(rotation='90')

sns.barplot(x=a.index, y=a['Percentage']*100)

plt.xlabel('Features', fontsize=1)

plt.ylabel('Percent of missing values', fontsize=15)

plt.title('Percent missing data by feature', fontsize=15)
display("Test File")

a = missing_data_percentage(df_test)

f, ax = plt.subplots(figsize=(10, 8))

plt.xticks(rotation='90')

sns.barplot(x=a.index, y=a['Percentage']*100)

plt.xlabel('Features', fontsize=1)

plt.ylabel('Percent of missing values', fontsize=15)

plt.title('Percent missing data by feature', fontsize=15)
display("Train File",sns.countplot(df_train['Segmentation']))
for i in df_train.columns:

    print("\n",i,"\n")

    print(df_train[i].unique())
df_train.drop(["ID"], axis = 1)



plt.figure(figsize=(10,6))

plt.title("Ages Frequency")

sns.axes_style("dark")

sns.violinplot(y=df_train["Age"])

plt.show()
genders = df_train.Gender.value_counts()

sns.set_style("darkgrid")

plt.figure(figsize=(10,4))

sns.barplot(x=genders.index, y=genders.values)

plt.show()
age18_25 = df_train.Age[(df_train.Age <= 25) & (df_train.Age >= 18)]

age26_35 = df_train.Age[(df_train.Age <= 35) & (df_train.Age >= 26)]

age36_45 = df_train.Age[(df_train.Age <= 45) & (df_train.Age >= 36)]

age46_55 = df_train.Age[(df_train.Age <= 55) & (df_train.Age >= 46)]

age55above = df_train.Age[df_train.Age >= 56]



x = ["18-25","26-35","36-45","46-55","55+"]

y = [len(age18_25.values),len(age26_35.values),len(age36_45.values),len(age46_55.values),len(age55above.values)]



plt.figure(figsize=(15,6))

sns.barplot(x=x, y=y, palette="rocket")

plt.title("Number of Customer and Ages")

plt.xlabel("Age")

plt.ylabel("Number of Customer")

plt.show()
sns.countplot(df_train['Spending_Score'])
for i in df_train.columns:

    a = df_train[i].value_counts().idxmax()

    df_train[i] = df_train[i].replace({np.nan : a})
for i in df_test.columns:

    a = df_test[i].value_counts().idxmax()

    df_test[i] = df_test[i].replace({np.nan : a})
#label encoding for object type columns

object_type_columns_train = df_train.select_dtypes(include=['object']).columns

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

for i in object_type_columns_train:

    df_train[i] = le.fit_transform(df_train[i])
corrmat = df_train.corr()

k = 10

cols = corrmat.nlargest(k, 'Segmentation')['Segmentation'].index

cm = np.corrcoef(df_train[cols].values.T)

sns.set(font_scale=1.25)

hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)

plt.show()
object_type_columns_test = df_test.select_dtypes(include=['object']).columns

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

for i in object_type_columns_test:

    df_test[i] = le.fit_transform(df_test[i])
y_train = df_train['Segmentation']

X_train = df_train.drop(['ID','Segmentation','Var_1','Graduated','Ever_Married'], axis=1)

X_test = df_test.drop(['ID','Var_1','Graduated','Ever_Married'],axis=1)

X_train.shape, y_train.shape, X_test.shape
from sklearn.neural_network import MLPClassifier
clf = MLPClassifier(solver='lbfgs', alpha=1e-5)
clf.fit(X_train, y_train)
clf.predict(X_test)
y_pred = clf.predict(X_test)
clf.score(X_train,y_train)
y_pred = le.inverse_transform(y_pred)
submission = pd.DataFrame(data = { 'ID' : df_test['ID'], 'Segmentation' : y_pred})
submission.to_csv('Submission.csv', index=False)