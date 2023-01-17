import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline
raw_data = pd.read_csv('../input/diabetes.csv')

raw_data.head()
raw_data.describe()
raw_data.info()
fig, axes = plt.subplots(2, 4, figsize=(20,10))



sns.boxplot(  y='Pregnancies', data=raw_data, orient='v', ax=axes[0, 0])

sns.boxplot(  y='Glucose', data=raw_data,  orient='v', ax=axes[0, 1])

sns.boxplot(  y='BloodPressure', data=raw_data, orient='v', ax=axes[0,2])

sns.boxplot(  y='SkinThickness', data=raw_data, orient='v', ax=axes[0,3])

sns.boxplot(  y='Insulin', data=raw_data, orient='v', ax=axes[1,0])

sns.boxplot(  y='BMI', data=raw_data, orient='v', ax=axes[1,1])

sns.boxplot(  y='DiabetesPedigreeFunction', data=raw_data, orient='v', ax=axes[1,2])

sns.boxplot(  y='Age', data=raw_data, orient='v', ax=axes[1,3])
raw_data.hist(figsize=(18, 9))
# Finding the number of Zeros per columns

for col in raw_data:

    print('{:>25}:{:>5}'.format(col, raw_data[col].loc[raw_data[col] == 0].count()))
# dataset with non zero values of critical attributes

# ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin','BMI', 'Age']



non_zero_data = raw_data.loc[(raw_data['Glucose'] != 0) & (raw_data['BloodPressure'] != 0) 

                             & (raw_data['SkinThickness'] != 0) & (raw_data['BMI'] != 0)]

corr = non_zero_data.corr()



# Generate a mask for the upper triangle

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



plt.figure(figsize=(12,6))

sns.heatmap(corr, mask=mask, annot=True, cmap='plasma',vmin=-1,vmax=1)
raw_data.groupby('Outcome')['Outcome'].count()
plt.figure(figsize=(8,4))

ax = sns.countplot(raw_data['Outcome'])

plt.title('Distribution of OutCome')

plt.xlabel('Outcomes')

plt.ylabel('Frequency')
fig, axes = plt.subplots(1, 2, figsize=(18, 6))

sns.boxplot(x='Outcome', y='Pregnancies', data=raw_data, ax=axes[0])

sns.countplot(raw_data['Pregnancies'], hue = raw_data['Outcome'], ax=axes[1])
raw_data['age_group'] = pd.cut(raw_data['Age'], range(0, 100, 10))

g = sns.catplot(x="age_group", y="Pregnancies", hue="Outcome",

               data=raw_data, kind="box"

              )

g.fig.set_figheight(4)

g.fig.set_figwidth(20)
raw_data['age_group'] = pd.cut(raw_data['Age'], range(0, 100, 10))

g = sns.catplot(x="age_group", y="BMI", hue="Outcome",

               data=raw_data, kind="box"

              )

g.fig.set_figheight(4)

g.fig.set_figwidth(20)
raw_data['age_group'] = pd.cut(raw_data['Age'], range(0, 100, 10))

g = sns.catplot(x="age_group", y="SkinThickness", hue="Outcome",

               data=raw_data, kind="box"

              )

g.fig.set_figheight(4)

g.fig.set_figwidth(20)
raw_data['age_group'] = pd.cut(raw_data['Age'], range(0, 100, 10))

g = sns.catplot(x="age_group", y="Glucose", hue="Outcome",

               data=raw_data, kind="box"

              )

g.fig.set_figheight(4)

g.fig.set_figwidth(20)
raw_data.groupby('Outcome')['Glucose'].plot(kind='density', legend=True)
raw_data.groupby('Outcome')['SkinThickness'].plot(kind='density', legend=True)
raw_data.groupby('Outcome')['Pregnancies'].plot(kind='density', legend=True)
raw_data.groupby('Outcome')['BloodPressure'].plot(kind='density', legend=True)
raw_data.groupby('Outcome')['Insulin'].plot(kind='density', legend=True)
raw_data.groupby('Outcome')['DiabetesPedigreeFunction'].plot(kind='density', legend=True)
raw_data.groupby('Outcome')['Age'].plot(kind='density', legend=True)
#import the model and datsplit and crossvalidation utilities

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score

from sklearn.metrics import accuracy_score



feature_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

X = raw_data[feature_names]

Y = raw_data.Outcome



lr = LogisticRegression(solver='liblinear')

X_train, X_test, y_train, y_test = train_test_split(X, Y, stratify = raw_data.Outcome, random_state=0)
lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)
accuracy_score(y_test, y_pred)
from sklearn.model_selection import KFold

kfold = KFold(n_splits=5, random_state=10)

recall = cross_val_score(lr, X, Y, cv=kfold, scoring='recall').mean()

accuracy = cross_val_score(lr, X, Y, cv=kfold, scoring='accuracy').mean()

print('With {:0.2f}% Accuracy and {:0.2f}% true positive rate, the model is able to predict that a given patent have diabetes or not'.format(accuracy, recall))
cross_val_score(lr, X, Y, cv=kfold, scoring='precision').mean()