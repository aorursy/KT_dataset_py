# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train_data = pd.read_csv('/kaggle/input/titanic/train.csv')

train_data.head()
train_data.shape
test_data = pd.read_csv("/kaggle/input/titanic/test.csv")

test_data.head()
women = train_data.loc[train_data.Sex == 'female']["Survived"]

rate_women = sum(women)/len(women)



print("% of women who survived:", rate_women)
men = train_data.loc[train_data.Sex == 'male']["Survived"]

rate_men = sum(men)/len(men)



print("% of men who survived:", rate_men)
missing_values_amount = train_data.isna().sum()

missing_values_percent = (missing_values_amount / train_data.shape[0]).round(6) * 100

missing_df = pd.DataFrame({

    "attributes": train_data.columns,

    "missing_values": missing_values_amount,

    "percent": missing_values_percent

})

top_missing_values = missing_df.loc[

    missing_df['missing_values']>0

].sort_values(

    by='missing_values', ascending=False

)

top_missing_values.reset_index().drop('index', axis=1)
figure = plt.figure(figsize=(10, 5))

chart = sns.barplot(x="attributes", y="percent", data=top_missing_values, palette=None)

for p in chart.patches:

        width, height = p.get_width(), p.get_height()

        x, y = p.get_xy() 

        chart.annotate('{:.2f}'.format(height), (x + 0.2, y + height + 0.5))

plt.title("Missing Values by Attributes")

plt.tight_layout()
train_data = train_data.dropna(subset=['Embarked'], how='any')
from sklearn.ensemble import RandomForestClassifier



y = train_data["Survived"]



features = ["Pclass", "Sex", "SibSp", "Parch", "Embarked"]

X = pd.get_dummies(train_data[features])

X_test = pd.get_dummies(test_data[features])



model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)

model.fit(X, y)

predictions = model.predict(X_test)



output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})

output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")