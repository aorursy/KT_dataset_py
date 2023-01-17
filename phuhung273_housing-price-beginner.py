# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import warnings

warnings.filterwarnings("ignore")



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from xgboost import XGBRegressor



from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV

from sklearn.pipeline import Pipeline, make_pipeline

from sklearn.compose import ColumnTransformer

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder, LabelEncoder, OrdinalEncoder

from sklearn.metrics import mean_absolute_error, make_scorer



raw_data = pd.read_csv('../input/train.csv')

X_test = pd.read_csv('../input/test.csv')



y = raw_data.SalePrice

X = raw_data.drop(columns=['SalePrice'])



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
cat_col = [col for col in X.columns if X[col].dtype == 'object']

num_cat_col = ['MSSubClass', 'MoSold', 'YrSold']    #Some numerical col can be categorical data

cat_col.extend(num_cat_col)



high_cardinal_col = [col for col in cat_col if X[col].nunique() > 10]

low_cardinal_col = list(set(cat_col) - set(high_cardinal_col))



X_drop_cardinal = X.drop(columns=high_cardinal_col)



quant_col = list(set(X.columns) - set(cat_col))
quant_trans = SimpleImputer(strategy='median')

#quant_trans = SimpleImputer(strategy='mean')



low_cat_trans = Pipeline(steps=[

        ('imputer', SimpleImputer(strategy='constant', fill_value='NaN')),

        ('onehot', OneHotEncoder(handle_unknown='ignore'))

        ])



#high_cat_trans = Pipeline(steps=[

#        ('impute', SimpleImputer(strategy='most_frequent')),

#        ('label', OrdinalEncoder())

#        ])

    

preprocessor = ColumnTransformer(transformers=[

        ('quant', quant_trans, quant_col),

        ('low_cat', low_cat_trans, low_cardinal_col)

#        ('high_cat', high_cat_trans, high_cardinal_col)

        ])



model = XGBRegressor(random_state=0,

                     tree_method='gpu_hist'

                     )
n_estimators_candidate = [100*i for i in range(1,11)]  

learn_rate_candidate = [0.01*i for i in range(1,11)]

#xgb_param = {'model__n_estimators': n_estimators_candidate,

#              'model__learning_rate': learn_rate_candidate}



xgb_param = {'n_estimators': n_estimators_candidate,

              'learning_rate': learn_rate_candidate}



xgb_grid = GridSearchCV(estimator=model,

                        param_grid=xgb_param,

                        cv=3,

                        scoring=make_scorer(mean_absolute_error, greater_is_better=True),

                        verbose=2,

                        refit=True)





pipe = make_pipeline(preprocessor, xgb_grid) #  Avoid repeating preprocessing



pipe.fit(X_drop_cardinal, y)



best_model = xgb_grid.best_estimator_



print("Best Model Parameter: ",xgb_grid.best_params_)
preds = pipe.predict(X_test)



submissions=pd.DataFrame({"Id": list(range(1461,len(preds)+1461)),

                         "SalePrice": preds})

submissions.to_csv("house_price_3rd.csv", index=False, header=True)