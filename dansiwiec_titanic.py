import pandas as pd



train = pd.read_csv('../input/titanic/train.csv')

test = pd.read_csv('../input/titanic/test.csv')
train.head()
test.head()
train['Age'].hist()
train['Fare'].plot.density()
train['Fare'].describe()
train['Fare'].hist(bins=30)
pd.cut(train['Fare'], bins=10, labels=range(10)).hist()
train['Parch'].hist()
corr = train.corr()['Survived'].abs().sort_values(ascending=False)

corr
from pandas.plotting import scatter_matrix



scatter_matrix(train[corr.keys()[:4].tolist()], figsize=(12, 8))
X_train = train.drop(['Survived'], axis='columns')

y_train = train['Survived'].copy()



X_test = test
X_train.info()
X_train['Sex'].describe()
X_train['Age'].describe()
X_train['Age'].hist()
X_train.isnull().sum()
X_train['Cabin'].isnull().values.sum()
y_train.describe()
pd.qcut(train['Fare'], 4)
pd.cut(train['Fare'], bins=[0, 35, 100, 500, 1000], labels=[1, 2, 3, 4])
from sklearn.impute import SimpleImputer

from sklearn.preprocessing import FunctionTransformer

from sklearn.preprocessing import OrdinalEncoder

from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import make_pipeline

from sklearn.ensemble import RandomForestClassifier





from sklearn.compose import ColumnTransformer



def encode_fare(X):

    X['FareType'] = pd.cut(X['Fare'], bins=[0, 35, 100, 500, 1000], labels=[1, 2, 3, 4])

    return X



def encode_age(X):

    X['AgeBin'] = pd.cut(X['Age'], bins=[0, 15, 30, 45, 60, 75], labels=[1, 2, 3, 4, 5])

    return X



def encode_family_size(X):

    X['FamilySize'] = X['Parch'] + X['SibSp']

    return X



prepare_pipeline = ColumnTransformer([

    ('identity', FunctionTransformer(lambda X: X), ['Pclass']),

    ('family_size', FunctionTransformer(encode_family_size), ['Parch', 'SibSp']),

    ('fare_encoder', FunctionTransformer(encode_fare), ['Fare']),

    ('age_encoder', FunctionTransformer(encode_age), ['Age']),

    ('ordinal_encoder', OrdinalEncoder(), ['Sex']),

    ('embarked_encoder', make_pipeline(SimpleImputer(strategy='most_frequent'), OrdinalEncoder()), ['Embarked'])

])



pipeline = make_pipeline(

    prepare_pipeline,

    SimpleImputer(strategy='median'),

    StandardScaler(),

    RandomForestClassifier(n_estimators=100, random_state=42)

)



pipeline.fit(X_train, y_train)

list(pipeline.predict(X_train.iloc[:10]))
list(y_train.iloc[:10])
from sklearn.model_selection import cross_val_score



scores = cross_val_score(pipeline, X_train, y_train, cv=10)

scores.mean()
from sklearn.model_selection import cross_val_predict

from sklearn.metrics import accuracy_score



y_train_predict = cross_val_predict(pipeline, X_train, y_train, cv=10)

accuracy_score(y_train, y_train_predict)
y_test_predict = pipeline.predict(X_test)
output = X_test[['PassengerId']].copy()

output['Survived'] = y_test_predict
output.head()
output.to_csv('output.csv', index=False)