import numpy as np

import pandas as pd



from sklearn import preprocessing

from sklearn import model_selection # train_test_split

from sklearn import pipeline

from sklearn import ensemble   # RandomForestClassifier

from sklearn import impute

from sklearn import compose

from sklearn import metrics    # accuracy_score, balanced_accuracy_score, plot_confusion_matrix

from sklearn import inspection # permutation_importance, plot_partial_dependence
train = pd.read_csv("../input/murcia-car-challenge/train.csv", index_col="Id")

test  = pd.read_csv("../input/murcia-car-challenge/test.csv",  index_col="Id")

sub   = pd.read_csv("../input/murcia-car-challenge/sampleSubmission.csv", index_col="Id")
train.head(1)
# 'Modelo',  'Localidad' 'Puertas',

cat_vars = ['Marca',  'Provincia', 'Cambio', 'Combust',  'Vendedor']

num_vars = ['Año', 'Kms', 'Cv']

target_var = 'Precio'



x = train[cat_vars + num_vars]

y = train[target_var]
x.info()
# train = train.dropna(axis='rows')

# test  = test.dropna(axis='rows')
num_preprocessing = pipeline.Pipeline(steps=[

    ('imputer', impute.SimpleImputer(strategy='median')),

    ('encoder', preprocessing.StandardScaler())

])



cat_preporcessing = pipeline.Pipeline(steps=[

    ('imputer', impute.SimpleImputer(strategy='constant', fill_value='missing')),

    ('encoder', preprocessing.OrdinalEncoder())

])



preprocessor = compose.ColumnTransformer(transformers=[

    ('num', num_preprocessing, num_vars),

    ('cat', cat_preporcessing, cat_vars)

])
# cat_encoder = ce.OrdinalEncoder    or     preprocessing.OrdinalEncoder()
x_train, x_valid, y_train, y_valid = model_selection.train_test_split(x, y,

                                                      test_size=0.2,

                                                      random_state=0)
model = ensemble.RandomForestRegressor(n_jobs=-1)



prep_model = pipeline.Pipeline(steps=[

    ('preprocessor', preprocessor),

    ('classifier', ensemble.RandomForestRegressor(

                              n_estimators=100,

                              n_jobs=-1

    ))

])
prep_model.fit(x_train, y_train);
preds = prep_model.predict(x_valid)

preds
metrics.mean_squared_log_error(y_valid, preds)
test_preds = prep_model.predict(test)

test_preds
sub['Precio'] = test_preds # test_preds.astype(int)

sub.head()
sub.to_csv('sub.csv', header=True, index=True)