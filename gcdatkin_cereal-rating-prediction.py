import numpy as np

import pandas as pd



from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split



from sklearn.linear_model import LinearRegression, Ridge, Lasso
data = pd.read_csv('../input/80-cereals/cereal.csv')
data
data.info()
data = data.drop('name', axis=1)
(data == -1).sum()
data = data.replace(-1, np.NaN)
data.isna().sum()
for column in ['carbo', 'sugars', 'potass']:

    data[column] = data[column].fillna(data[column].mean())
data.isna().sum().sum()
data
{column: list(data[column].unique()) for column in ['mfr', 'type']}
data['type'] = data['type'].apply(lambda x: 1 if x == 'H' else 0)
# One-Hot Encode the "mfr" column



dummies = pd.get_dummies(data['mfr'])

data = pd.concat([data, dummies], axis=1)

data = data.drop('mfr', axis=1)
data
y = data.loc[:, 'rating']

X = data.drop('rating', axis=1)
scaler = StandardScaler()



X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
X
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=42)
model = LinearRegression()

l2_model = Ridge(alpha=1.5)

l1_model = Lasso(alpha=0.001)
model.fit(X_train, y_train)

l2_model.fit(X_train, y_train)

l1_model.fit(X_train, y_train)



print("Models trained.")
model_r2 = model.score(X_test, y_test)

l2_model_r2 = l2_model.score(X_test, y_test)

l1_model_r2 = l1_model.score(X_test, y_test)
print("R^2 Scores\n" + "*" * 10)

print("        Without Regularization:", model_r2)

print("With L2 (Ridge) Regularization:", l2_model_r2)

print("With L1 (Lasso) Regularization:", l1_model_r2)