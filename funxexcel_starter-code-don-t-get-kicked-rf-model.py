import numpy as np 

import pandas as pd 



from sklearn.pipeline import make_pipeline

from sklearn.compose import make_column_transformer



from sklearn.impute import SimpleImputer

from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder, StandardScaler



from sklearn.linear_model import LogisticRegression



from sklearn.model_selection import cross_val_score
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train = pd.read_csv('/kaggle/input/DontGetKicked/training.csv')

test = pd.read_csv('/kaggle/input/DontGetKicked/test.csv')
train.head()
train.info()
y = train['IsBadBuy']

X = train.drop(['IsBadBuy','RefId'], axis = 1)
numerical_features = [c for c, dtype in zip(X.columns, X.dtypes)

                     if dtype.kind in ['i','f']]

categorical_features = [c for c, dtype in zip(X.columns, X.dtypes)

                     if dtype.kind not in ['i','f']]
print('Numerical : ' + str(numerical_features))

print('Categorical : ' + str(categorical_features))
#import train_test_split library

from sklearn.model_selection import train_test_split



# create train test split

X_train, X_test, y_train, y_test = train_test_split( X,  y, test_size=0.3, random_state=0, stratify = y)
preprocessor = make_column_transformer(

    

    (make_pipeline(

    SimpleImputer(strategy = 'median'),

        StandardScaler(),

    KBinsDiscretizer(n_bins=3)), numerical_features),

    

    (make_pipeline(

    SimpleImputer(strategy = 'constant', fill_value = 'missing'),

    OneHotEncoder(categories = 'auto', handle_unknown = 'ignore')), categorical_features),

    

)
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import RandomizedSearchCV
grid_param =   {"randomforestclassifier__n_estimators": [10, 100, 200, 300, 400, 500, 600, 700],

                 "randomforestclassifier__max_depth":[5,8,15,25,30,None],

                 "randomforestclassifier__min_samples_leaf":[1,2,5,10,15,100],

                 "randomforestclassifier__max_leaf_nodes": [2, 5,10]}
RF_Model = make_pipeline(preprocessor, RandomForestClassifier())

randonSearch = RandomizedSearchCV(RF_Model, grid_param, cv=5, verbose=0,n_jobs=-1, scoring="accuracy") # Fit grid search

best_model = randonSearch.fit(X_train,y_train)
best_model
print(f'Train : {best_model.score(X_train, y_train):.3f}')

print(f'Test : {best_model.score(X_test, y_test):.3f}')
sub_test = test.drop(['RefId'], axis = 1)
sub_test_pred = best_model.predict(sub_test).astype(int)
AllSub = pd.DataFrame({ 'RefId': test['RefId'],

                       'IsBadBuy' : sub_test_pred

    

})



AllSub.to_csv("DGK_Pipeline_RF_RGS.csv", index = False)