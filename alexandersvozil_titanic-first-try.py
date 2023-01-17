import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_absolute_error

from sklearn.impute import SimpleImputer



X = pd.read_csv("/kaggle/input/titanic/train.csv");

X_test = pd.read_csv("/kaggle/input/titanic/test.csv");

y = X["Survived"]



X['travel_alone'] = X['SibSp'].apply(lambda x: min(x,1))

X_test['travel_alone'] = X_test['SibSp'].apply(lambda x: min(x,1))

X.head()
[print(cname + " " + str(X[cname].dtype)) for cname in X.columns]

[print(str(cname) + " " + str(X[cname].nunique())) for cname in X.columns if X[cname].dtype == "object" ]

print(X['Name'])
import itertools



X.drop(['Survived','Name'], axis=1, inplace=True)

X_test.drop(['Name'], axis=1, inplace=True)



categorical_cols = [cname for cname in X.columns if

                   X[cname].dtype == "object"]



numerical_cols = [cname for cname in X.columns if

                 X[cname].dtype in ['int64', 'float64']]



print(categorical_cols)

print(numerical_cols)







import itertools

from sklearn import preprocessing, metrics

interactions = pd.DataFrame(index=X.index)

interactions_test = pd.DataFrame(index=X_test.index)



for cat1, cat2 in itertools.combinations(categorical_cols,2):

    print(cat1, cat2)

    new_col = cat1 + '_'+cat2;

    new_values = X[cat1].map(str)+"_"+X[cat2].map(str)

    new_values_test = X_test[cat1].map(str)+"_"+X_test[cat2].map(str)

    encoder=preprocessing.LabelEncoder()

    interactions[new_col] = encoder.fit_transform(new_values)

    encoder_test=preprocessing.LabelEncoder()

    interactions_test[new_col] = encoder_test.fit_transform(new_values_test)
interactions.head()

X = X.join(interactions)

X_test = X_test.join(interactions_test)
X.head()
from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error

from sklearn.preprocessing import OrdinalEncoder



from xgboost import XGBClassifier





X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=1)



numerical_transformer = SimpleImputer(strategy='mean')



categorical_transformer = Pipeline(steps=[

    ('imputer',  SimpleImputer(strategy='most_frequent')),

    ('onehot', OneHotEncoder(handle_unknown='ignore'))

])



preprocessor = ColumnTransformer(

    transformers=[

        ('num', numerical_transformer, numerical_cols),

        ('cat', categorical_transformer, categorical_cols),

    ]

)



#model = (n_estimators=2000, learning_rate=0.01, verbose=True)

# Bundle preprocessing and modeling code in a pipeline



X_train.head()
from sklearn.model_selection import cross_val_score

from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import RandomForestClassifier



clf = Pipeline(steps=[('preprocessor', preprocessor),

                     ('rand_forest', RandomForestClassifier(random_state=0, n_estimators=100))

                     ])

clf.fit(X_train,y_train)

pred = clf.predict(X_valid)

print("MAE:" + str(mean_absolute_error(pred,y_valid)));

final_pipeline = Pipeline(steps=[('preprocessor', preprocessor),

                      ('rand_forest', RandomForestClassifier(n_estimators=100))

                     ])



final_pipeline.fit(X,y)

prediction = final_pipeline.predict(X_test)

output = pd.DataFrame({'PassengerId': X_test.PassengerId, 'Survived': prediction})

output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")