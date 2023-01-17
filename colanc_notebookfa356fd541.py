import pandas as pd

import numpy as np

from sklearn import cross_validation

from sklearn import preprocessing

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier



df = pd.read_csv("../input/train.csv", error_bad_lines=True)

df2 = pd.read_csv('../input/test.csv')

df = df[np.isfinite(df['Age'])]

df = df[np.isfinite(df['Pclass'])]

df = df[np.isfinite(df['Fare'])]

df = df[np.isfinite(df['Survived'])]









for index, row in df.iterrows():

      df.loc[index, 'Sex'] = 0 if row['Sex'] == 'female' else 1

 

for index, row in df2.iterrows():

      df2.loc[index, 'Sex'] = 0 if row['Sex'] == 'female' else 1





x = []

y = []



x2 = []



for index, row in df.iterrows():

    x.append([float(row['Sex']), row['Parch'], float(row['Age']), float(row['Fare']), float(row['Pclass'])])

    y.append(row['Survived'])

    

for index, row in df.iterrows():

    x2.append([float(row['Sex']), row['Parch'], float(row['Age']), float(row['Fare']), float(row['Pclass'])])

    

x = np.array(x)

y = np.array(y)



x2 = np.array(x2)







clf = RandomForestClassifier(n_estimators=25, min_samples_split=2)

clf.fit(x, y)



result = clf.predict(x2)



res = pd.DataFrame()

for index, row in df2.iterrows():

    res = res.append(

        {

            'PassengerId': str(row['PassengerId']),

            'Survived': str(result[index]) 

        }, 

        ignore_index=True

    )

res.head()

res.to_csv('submission.csv', index=False)