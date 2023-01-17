import pandas as pd
train_data_CSV = '../input/titanic/train.csv'

test_data_CSV = '../input/titanic/test.csv'

example_data = '../input/titanic/gender_submission.csv'
train_data = pd.read_csv(train_data_CSV, index_col='PassengerId')

test_data = pd.read_csv(test_data_CSV, index_col='PassengerId')
train_data = train_data.drop(['Cabin', 'Name', 'Ticket'], axis="columns")

test_data = test_data.drop(['Cabin', 'Name', 'Ticket'], axis="columns")

train_y = train_data['Survived']

train_data = train_data.drop(['Survived'], axis=1)
train_data.describe()
train_data.head()
from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder

from sklearn.pipeline import Pipeline

from sklearn.compose import ColumnTransformer
categorical_cols = [cname for cname in train_data.columns if train_data[cname].dtype == "object"]

numerical_cols = [cname for cname in train_data.columns if train_data[cname].dtype != "object"]
# Preprocessing for numerical data

numerical_transformer = SimpleImputer(strategy='constant')



# Preprocessing for categorical data

categorical_transformer = Pipeline(steps=[

    ('imputer', SimpleImputer(strategy='most_frequent')),

    ('onehot', OneHotEncoder(handle_unknown='ignore'))

])
preprocessor = ColumnTransformer(

    transformers=[

        ('num', numerical_transformer, numerical_cols),

        ('cat', categorical_transformer, categorical_cols)

    ])
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier

from sklearn.neural_network import MLPClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import StratifiedKFold
models = [

    XGBClassifier(n_jobs=4, n_estimators=500),

]
for model in models:

    my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),

                              ('model', model)

                             ])

    kfold = StratifiedKFold(n_splits=4, shuffle=True)

    acc = cross_val_score(my_pipeline, train_data, train_y, cv=kfold, scoring='accuracy')

    print(model.__class__.__name__, acc.mean())
import tqdm

n_est = []

acc_s = []

for i in tqdm.tqdm(range(180, 210, 2)):

    my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),

                              ('model', XGBClassifier(n_jobs=4, n_estimators=i, learning_rate=0.02),)

                             ])

    kfold = StratifiedKFold(n_splits=4, shuffle=True, random_state=0)

    acc = cross_val_score(my_pipeline, train_data, train_y, cv=kfold, scoring='accuracy').mean()

    n_est.append(i)

    acc_s.append(acc)
from matplotlib import pyplot as plt

plt.plot(n_est, acc_s)

plt.show()
model = XGBClassifier(n_jobs=4, n_estimators=500)

my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),

                              ('model', model)])
pipe = my_pipeline.fit(train_data, train_y)
pred = pipe.predict(test_data)

ids = test_data.index
pd.read_csv(example_data)
finalData = pd.DataFrame({

    'PassengerId':ids,

    'Survived': pred

})
finalData.to_csv('submit.csv', index=False)