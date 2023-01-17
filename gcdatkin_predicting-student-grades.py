import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split



from sklearn.linear_model import LinearRegression
data = pd.read_csv("../input/student-alcohol-consumption/student-mat.csv")
data
plt.figure(figsize=(14, 12))

sns.heatmap(data.corr(), annot=True)

plt.show()
data.isnull().sum()
data.dtypes
nonnumeric_columns = [data.columns[index] for index, dtype in enumerate(data.dtypes) if dtype == 'object']

nonnumeric_columns
for column in nonnumeric_columns:

    print(f"{column}: {data[column].unique()}")
data['Mjob'] = data['Mjob'].apply(lambda x: "m_" + x)

data['Fjob'] = data['Fjob'].apply(lambda x: "f_" + x)

data['reason'] = data['reason'].apply(lambda x: "r_" + x)

data['guardian'] = data['guardian'].apply(lambda x: "g_" + x)
data
dummies = pd.concat([pd.get_dummies(data['Mjob']),

                     pd.get_dummies(data['Fjob']),

                     pd.get_dummies(data['reason']),

                     pd.get_dummies(data['guardian'])],

                     axis=1)
dummies
data = pd.concat([data, dummies], axis=1)



data.drop(['Mjob', 'Fjob', 'reason', 'guardian'], axis=1, inplace=True)
data
nonnumeric_columns = [data.columns[index] for index, dtype in enumerate(data.dtypes) if dtype == 'object']



for column in nonnumeric_columns:

    print(f"{column}: {data[column].unique()}")
encoder = LabelEncoder()



for column in nonnumeric_columns:

    data[column] = encoder.fit_transform(data[column])
for dtype in data.dtypes:

    print(dtype)
y = data['G3']

X = data.drop('G3', axis=1)
X
scaler = StandardScaler()



X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
X
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7)
model = LinearRegression()

model.fit(X_train, y_train)
print(f"Model R2: {model.score(X_test, y_test)}")