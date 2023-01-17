import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier
df = pd.read_csv("../input/diabetes.csv")
print(df.shape)

df.head()
df.info()
df.describe()
for feature in df.columns[:-1]:

    print(df[[feature, 'Outcome']].groupby(feature, as_index=False).mean().sort_values(by='Outcome', ascending=False))

    print('-' * 100)
sns.distplot(df['Pregnancies'], rug=True)

plt.show()
sns.pairplot(df)

plt.title("Pima Indians Diabetes Data의 Pair Plot")

plt.show()
columns = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']



for feature in columns:

    print('Count of zero: {0}, {1:0.2}%: '.format(df[feature].value_counts()[0], df[feature].value_counts()[0] / df.shape[0]))
mean_value = df[df['Insulin'] != 0]['Insulin'].mean()

df[df['Insulin'] == 0] = mean_value



mean_value = df[df['SkinThickness'] != 0]['SkinThickness'].mean()

df[df['SkinThickness'] == 0] = mean_value
X = df.drop('Outcome', axis=1).copy()

y = df['Outcome'].copy()



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
clf = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=43)

clf.fit(X_train, y_train)

pred = clf.predict(X_test)



(pred == y_test).sum() / y_test.shape[0]