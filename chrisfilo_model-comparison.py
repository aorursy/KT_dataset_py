import sys

!cp ../input/rapids/rapids.0.13.0 /opt/conda/envs/rapids.tar.gz

!cd /opt/conda/envs/ && tar -xzvf rapids.tar.gz > /dev/null

sys.path = ["/opt/conda/envs/rapids/lib/python3.6/site-packages"] + sys.path

sys.path = ["/opt/conda/envs/rapids/lib/python3.6"] + sys.path

sys.path = ["/opt/conda/envs/rapids/lib"] + sys.path

!cp /opt/conda/envs/rapids/lib/libxgboost.so /opt/conda/lib/
import pandas as pd

import numpy as np

import cudf
train_scores_df = pd.read_csv("/kaggle/input/trends-assessment-prediction/train_scores.csv", index_col='Id')

train_scores_df['is_train'] = True

train_scores_df
loading_df = pd.read_csv("/kaggle/input/trends-assessment-prediction/loading.csv", index_col='Id')

loading_features = list(loading_df.columns.values)

fnc_df = pd.read_csv("/kaggle/input/trends-assessment-prediction/fnc.csv", index_col='Id')

fnc_features = list(fnc_df.columns.values)

features_df = loading_df.join(fnc_df)

#dropping IC_20 since it shows large site effects

all_features = list(set(features_df.columns.values) - set(['IC_20']))

features_df.shape
combined_df = train_scores_df.merge(features_df, how='outer', left_index=True, right_index=True)

print(combined_df.shape)

combined_df
combined_df.is_train.fillna(False, inplace=True)

combined_df.is_train.mean()
from sklearn.metrics import make_scorer

def MAPE(true, predicted, **kwargs):

    absolute_error = np.abs(predicted-true).sum()

    normalized_error = absolute_error/true.sum()

    return normalized_error



MAPE_scorer = make_scorer(MAPE, greater_is_better=False)



def weighted_absolute_error(true, predicted, weight):

    return weight*MAPE(true, predicted)
from sklearn.dummy import DummyRegressor

from sklearn.linear_model import LinearRegression

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.svm import SVR

from cuml import SVR as cuSVR

from sklearn.pipeline import Pipeline

from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import RobustScaler

from sklearn.model_selection import KFold



kf = KFold(n_splits=5, shuffle=True, random_state=0)

label_weights = {'age': .3, 

                 'domain1_var1': .175, 

                 'domain1_var2': .175, 

                 'domain2_var1': .175, 

                 'domain2_var2': .175}



train_and_val_df = combined_df[combined_df.is_train]

test_df = combined_df[~combined_df.is_train]



def evaluate_model(model_fn, model_name):

    fold_scores = []

    for fold_i, (train_index, val_index) in enumerate(kf.split(train_and_val_df)):

        weighted_absolute_errors = []

        for label in label_weights.keys():

            train_df = train_and_val_df.iloc[train_index]

            train_df = train_df[~train_df[label].isna()]



            val_df = train_and_val_df.iloc[val_index]

            val_df = val_df[~val_df[label].isna()]



            preprocess, model = model_fn()

            preprocess.fit(train_df, train_df[label])

            

            model.fit(np.asfortranarray(preprocess.transform(train_df)), 

                      np.array(train_df[label].values, order='F'))

            

            predicted = np.array(model.predict(np.asfortranarray(preprocess.transform(val_df))))



            weighted_absolute_errors.append(weighted_absolute_error(true=val_df[label], 

                                                                    predicted=predicted,

                                                                    weight=label_weights[label]))

        score = sum(weighted_absolute_errors)

        print("[%s] Fold %d, score: %f"%(model_name, fold_i, score))

        fold_scores.append(score)



    print("[%s] Average score: %f"%(model_name, np.mean(fold_scores)))



def create_dummy_model():

    model = DummyRegressor('median')

    preprocess = Pipeline([

        ('union', ColumnTransformer([('scale', RobustScaler(), loading_features)]))

    ])

    return preprocess, model

          

evaluate_model(create_dummy_model, 'median')



def create_linear_regression_model():

    regressor = LinearRegression()

    pipeline = Pipeline([

        ('union', ColumnTransformer([('scale', RobustScaler(), loading_features)])),

        ("dummy_regressor", regressor)

    ])

    return pipeline

        

#evaluate_model(create_linear_regression_model, 'linear')



def create_gb_regression_model():

    regressor = GradientBoostingRegressor()

    pipeline = Pipeline([

        ('union', ColumnTransformer([('scale', RobustScaler(), loading_features)])),

        ("dummy_regressor", regressor)

    ])

    return pipeline



#evaluate_model(create_gb_regression_model, 'gb')

          

def create_sv_regression_model():

    regressor = SVR()

    pipeline = Pipeline([

        ('union', ColumnTransformer([('scale', RobustScaler(), loading_features)])),

        ("dummy_regressor", regressor)

    ])

    return pipeline



#evaluate_model(create_sv_regression_model, 'sv')



def create_cusvr_model():

    model = cuSVR(cache_size=3000.0)

    preprocess = Pipeline([

        ('union', ColumnTransformer([('scale', RobustScaler(), loading_features)]))

    ])

    return preprocess, model

          

evaluate_model(create_cusvr_model, 'cuSVR')
from sklearn.linear_model import Ridge

from sklearn.preprocessing import FunctionTransformer

from sklearn.model_selection import GridSearchCV

import warnings

warnings.simplefilter("ignore")



def model_factory(C, factor):

    def create_svr_model_v2():

        def down_scale(x):

            return x/factor



        PREPROCESSING_STAGE_NAME = 'preprocessing'

        REGRESSION_STAGE_NAME = 'regression'

        preprocess = Pipeline([

            (PREPROCESSING_STAGE_NAME, 

             ColumnTransformer([('down_scale_fnc', FunctionTransformer(down_scale), fnc_features),

                                ('scale_others', 'passthrough', loading_features)])),

        ])



        model = cuSVR(C=C, cache_size=3000.0)

        return preprocess, model

    

    return create_svr_model_v2



for C in [1, 10, 100]:

    for factor in [200.0, 250.0, 500.0]:

        evaluate_model(model_factory(C, factor), "C=%f, factor=%f"%(C,factor))
from sklearn.linear_model import Ridge

from sklearn.preprocessing import FunctionTransformer

from sklearn.model_selection import GridSearchCV

from sklearn.feature_selection import SelectKBest, f_regression

from sklearn.preprocessing import StandardScaler





import warnings

warnings.simplefilter("ignore")



def model_factory_v3(C, k):



    def create_svr_model_v3():

        PREPROCESSING_STAGE_NAME = 'preprocessing'

        preprocess = Pipeline([

            (PREPROCESSING_STAGE_NAME, 

             ColumnTransformer([('top_k', SelectKBest(f_regression, k=k), fnc_features),

                                ('others', 'passthrough', loading_features)])),

            ('scaling', StandardScaler())

        ])



        model = cuSVR(C=C, cache_size=3000.0)

        return preprocess, model

    

    return create_svr_model_v3



for C in [1, 10, 100]:

    for k in [10, 100, 200]:

        evaluate_model(model_factory_v3(C, k), "C=%f, k=%f"%(C,k))
from sklearn.linear_model import Ridge

from sklearn.preprocessing import FunctionTransformer

from sklearn.model_selection import GridSearchCV

from sklearn.feature_selection import SelectKBest, f_regression

from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA



import warnings

warnings.simplefilter("ignore")



def model_factory_v4(C, k):



    def create_svr_model_v4():

        PREPROCESSING_STAGE_NAME = 'preprocessing'

        preprocess = Pipeline([

            (PREPROCESSING_STAGE_NAME, 

             ColumnTransformer([('top_k', PCA(n_components=k), fnc_features),

                                ('others', 'passthrough', loading_features)])),

            ('scaling', StandardScaler())

        ])



        model = cuSVR(C=C, cache_size=3000.0)

        return preprocess, model

    

    return create_svr_model_v4



for C in [1, 10, 100]:

    for k in [10, 100, 200]:

        evaluate_model(model_factory_v4(C, k), "C=%f, k=%f"%(C,k))
from sklearn.linear_model import Ridge

from sklearn.preprocessing import FunctionTransformer

from sklearn.model_selection import GridSearchCV

from sklearn.feature_selection import SelectKBest, f_regression

from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA



import warnings

warnings.simplefilter("ignore")



def model_factory_v5(C, k, factor):



    def create_svr_model_v5():

        def down_scale(x):

            return x/factor



        PREPROCESSING_STAGE_NAME = 'preprocessing'

        preprocess = Pipeline([

            (PREPROCESSING_STAGE_NAME, 

             ColumnTransformer([('top_k', Pipeline([('PCA', PCA(n_components=k)), 

                                                    ('scale', FunctionTransformer(down_scale))]), fnc_features),

                                ('others', 'passthrough', loading_features)]))

        ])



        model = cuSVR(C=C, cache_size=3000.0)

        return preprocess, model

    

    return create_svr_model_v5



for C in [10, 20]:

    for k in [200, 300, 400]:

        for factor in [250]:

            evaluate_model(model_factory_v5(C, k, factor), "C=%f, k=%f, factor=%f"%(C,k, factor))
for C in [5, 10, 15]:

    for k in [500, 600, 700]:

        for factor in [250, 350]:

            evaluate_model(model_factory_v5(C, k, factor), "C=%f, k=%f, factor=%f"%(C,k, factor))
for C in [15]:

    for k in [800, 900]:

        for factor in [450, 550]:

            evaluate_model(model_factory_v5(C, k, factor), "C=%f, k=%f, factor=%f"%(C,k, factor))
output_dfs = []

for label in label_weights.keys():

    train_df = train_and_val_df

    train_df = train_df[~train_df[label].isna()]



    preprocess, model = model_factory_v5(C=10.000000, k=700, factor=250.0)()

    preprocess.fit(train_df, train_df[label])



    model.fit(np.asfortranarray(preprocess.transform(train_df)), 

              np.array(train_df[label].values, order='F'))



    predicted = np.array(model.predict(np.asfortranarray(preprocess.transform(test_df))))



    output_dfs.append(pd.DataFrame(predicted, index=test_df.index, columns=[label]))



output_df = pd.concat(output_dfs, axis=1)
melt_df = output_df.reset_index().melt(id_vars=['Id'], 

                                         value_vars=['age', 'domain1_var1', 'domain1_var2', 'domain2_var1', 'domain2_var2'],

                                         value_name='Predicted')

melt_df['Id'] = melt_df['Id'].astype('str') + '_' + melt_df['variable']

melt_df.to_csv("submission.csv", index=False, columns=['Id', 'Predicted'])
!head submission.csv