import numpy as np

import pandas as pd



from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split



from sklearn.linear_model import LogisticRegression
data = pd.read_csv('../input/bank-direct-marketing/bank-full.csv', delimiter=';')
data
data.info()
y = data['y']

X = data.drop('y', axis=1)
def get_categorical_features(df):

    return [feature for feature in df.columns if df[feature].dtype == 'object']
get_categorical_features(X)
def get_uniques(df, columns):

    return {column: list(df[column].unique()) for column in columns}
get_uniques(X, get_categorical_features(X))
X = X.replace('unknown', np.NaN)
X.isna().sum()
X.drop('poutcome', axis=1, inplace=True)
get_uniques(X, get_categorical_features(X))
binary_features = ['default', 'housing', 'loan']



ordinal_features = ['education', 'month']



nominal_features = ['job', 'marital', 'contact']
def binary_encode(df, columns, positive_label):

    df = df.copy()

    for column in columns:

        df[column] = df[column].apply(lambda x: 1 if x == positive_label else 0)

    return df
X = binary_encode(X, binary_features, 'yes')
def ordinal_encode(df, columns, orderings):

    df = df.copy()

    for column, ordering in zip(columns, orderings):

        df[column] = df[column].apply(lambda x: ordering.index(x) if str(x) != 'nan' else x)

    return df
education_ordering = ['primary', 'secondary', 'tertiary']



month_ordering = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']



orderings = [education_ordering, month_ordering]





X = ordinal_encode(X, ordinal_features, orderings)
def onehot_encode(df, columns):

    df = df.copy()

    for column in columns:

        dummies = pd.get_dummies(df[column])

        df = pd.concat([df, dummies], axis=1)

        df.drop(column, axis=1, inplace=True)

    return df
X = onehot_encode(X, nominal_features)
X
X.isna().sum()
X['education'] = X['education'].fillna(X['education'].median())
scaler = StandardScaler()



X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
label_encoder = LabelEncoder()



y = label_encoder.fit_transform(y)
X
y
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7)
model = LogisticRegression()



model.fit(X_train, y_train)
model_acc = model.score(X_test, y_test)

print("Model Accuracy:", model_acc)