import warnings
warnings.filterwarnings('ignore')

import numpy as np  # I may not be using it

# For EDA and cleaning the data
import pandas as pd

# For visualizations
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

# For building a model
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
income_df = pd.read_csv('../input/adult.csv', names=['age', 'workclass', 'fnlwgt', 'education', 'education-num',
                                           'marital-status', 'occupation', 'relationship', 'race',
                                           'sex', 'capital-gain', 'capital-loss', 'hours-per-week',
                                           'native-country', 'salary'])
income_df.head()
income_df.shape
income_df.info()
income_df.describe()
income_df.columns
income_df.head()
sns.distplot(income_df.age)
sns.countplot(income_df.salary)
sns.countplot(income_df.salary, hue=income_df.sex, palette='rainbow')
sns.barplot(income_df.salary, income_df['capital-gain'])
income_df.occupation.unique()
plt.xticks(rotation=90)
sns.countplot(income_df.occupation, hue=income_df.salary, palette='Blues_r')
income_df.relationship.unique()
plt.xticks(rotation=90)

sns.countplot(income_df.relationship, hue=income_df.salary, palette='Accent')
income_df.workclass.value_counts()
plt.xticks(rotation=90)
sns.countplot(income_df.workclass, hue=income_df.salary)
income_df.race.unique()
plt.xticks(rotation=90)
sns.barplot(income_df.workclass, income_df['hours-per-week'], hue=income_df.salary, palette='cool')
sns.set()
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.xticks(rotation=90)
sns.countplot(income_df.race, hue=income_df.salary)
income_df.head()
income_df.drop(['fnlwgt', 'education-num'], axis=1, inplace=True)
income_df.head()
dummies = pd.get_dummies(income_df.drop(['salary', 'age', 'capital-gain', 'capital-loss',
                                        'hours-per-week'], axis=1))
dummies.shape
dummies.head()
merged = pd.concat([income_df, dummies], axis=1)
merged.shape
merged.head()
merged.columns[:100]
final_df = merged.drop(['workclass', 'education', 'marital-status', 'occupation', 'relationship',
                       'race', 'sex', 'native-country'], axis=1)
final_df.head()
final_df.shape
X_train, X_test, y_train, y_test = train_test_split(final_df.drop('salary', axis=1), final_df.salary, 
                                                   test_size=0.30, random_state=4)
X_train.shape
X_test.shape
gbm = GradientBoostingClassifier()
gbm.fit(X_train, y_train)
predictions = gbm.predict(X_test)
predictions
print(metrics.classification_report(y_test, predictions))
metrics.accuracy_score(y_test, predictions)