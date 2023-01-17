import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt
%matplotlib inline
df = pd.read_csv('../input/HR_comma_sep.csv')
df.head()
df.left.value_counts()
df.groupby('number_project')['left'].mean()
sns.barplot(df.number_project, df.left)
plt.hist(df.loc[df.left==1, 'number_project'], bins=8)

plt.title('Fig 2')

plt.figure()

plt.hist(df.loc[df.left==0, 'number_project'], bins=7)

plt.show()
df.number_project.value_counts()
sns.barplot(df.number_project, df.satisfaction_level)
sns.barplot(df.number_project, df.average_montly_hours)
plt.scatter(df.average_montly_hours, df.left)
sns.barplot(df.time_spend_company, df.left)
df.average_montly_hours.hist()
df.loc[df.time_spend_company==5, 'number_project'].hist()
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import KFold
for col in df.columns:

    if df[col].dtype=='object':

        le = LabelEncoder()

        df[col] = le.fit_transform(df[col])
predictors = df.columns.drop(['left'])
clf = xgb.XGBClassifier(learning_rate=0.03, max_depth=7)
kf = KFold(n = df.shape[0], n_folds=3, random_state=1)
for train_index, test_index in kf:

    train = df.loc[train_index]

    test = df.loc[test_index]

    clf.fit(train[predictors], train['left'])

    preds = clf.predict(test[predictors])

    print('accuracy =', (preds==test['left']).mean())