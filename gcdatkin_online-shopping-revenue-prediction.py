import numpy as np

import pandas as pd



from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split



from sklearn.linear_model import LogisticRegression
data = pd.read_csv('../input/online-shoppers-intention/online_shoppers_intention.csv')
data
data.info()
data.isna().sum()
data[data.isna().sum(axis=1).astype(bool)]
data = data.dropna(axis=0).reset_index(drop=True)
print("Total missing values:", data.isna().sum().sum())
data
{column: list(data[column].unique()) for column in data.columns if data.dtypes[column] == 'object'}
def ordinal_encode(df, column, ordering):

    df = df.copy()

    df[column] = df[column].apply(lambda x: ordering.index(x))

    return df



def onehot_encode(df, column, prefix):

    df = df.copy()

    dummies = pd.get_dummies(df[column], prefix=prefix)

    df = pd.concat([df, dummies], axis=1)

    df = df.drop(column, axis=1)

    return df
month_ordering = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'June', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']



visitor_prefix = 'V'
data = ordinal_encode(

    data,

    'Month',

    month_ordering

)



data = onehot_encode(

    data,

    'VisitorType',

    visitor_prefix

)



data['Weekend'] = data['Weekend'].astype(np.int)

data['Revenue'] = data['Revenue'].astype(np.int)
data
y = data['Revenue'].copy()

X = data.drop('Revenue', axis=1)
scaler = StandardScaler()



X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=20)
models = []

Cs = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]



for i in range(len(Cs)):

    model = LogisticRegression(C=Cs[i])

    model.fit(X_train, y_train)

    models.append(model)
model_acc = [model.score(X_test, y_test) for model in models]



print(f"   Model Accuracy (C={Cs[0]}):", model_acc[0])

print(f"    Model Accuracy (C={Cs[1]}):", model_acc[1])

print(f"    Model Accuracy (C={Cs[2]}):", model_acc[2])

print(f"   Model Accuracy (C={Cs[3]}):", model_acc[3])

print(f"  Model Accuracy (C={Cs[4]}):", model_acc[4])

print(f" Model Accuracy (C={Cs[5]}):", model_acc[5])

print(f"Model Accuracy (C={Cs[6]}):", model_acc[6])