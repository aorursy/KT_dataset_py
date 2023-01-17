import pandas as pd

import numpy as np



df_train = pd.read_csv("../input/train.csv")

df_test = pd.read_csv("../input/test.csv")



y_train = df_train.set_index("Id")["SalePrice"]

X_train = df_train.set_index("Id").iloc[:,:-1]

X_test = df_test.set_index("Id")
from sklearn.base import TransformerMixin, BaseEstimator, clone



class SelectColumnsTransfomer(BaseEstimator, TransformerMixin):

    """ A DataFrame transformer that provides column selection

    

    Allows to select columns by name from pandas dataframes in scikit-learn

    pipelines.

    

    Parameters

    ----------

    columns : list of str, names of the dataframe columns to select

        Default: [] 

    

    """

    def __init__(self, columns=[]):

        self.columns = columns



    def transform(self, X, **transform_params):

        """ Selects columns of a DataFrame

        

        Parameters

        ----------

        X : pandas DataFrame

            

        Returns

        ----------

        

        trans : pandas DataFrame

            contains selected columns of X      

        """

        trans = X[self.columns].copy() 

        return trans



    def fit(self, X, y=None, **fit_params):

        """ Do nothing function

        

        Parameters

        ----------

        X : pandas DataFrame

        y : default None

                

        

        Returns

        ----------

        self  

        """

        return self

    



class DataFrameFunctionTransformer(BaseEstimator, TransformerMixin):

    """ A DataFrame transformer providing imputation or function application

    

    Parameters

    ----------

    impute : Boolean, default False

        

    func : function that acts on an array of the form [n_elements, 1]

        if impute is True, functions must return a float number, otherwise 

        an array of the form [n_elements, 1]

    

    """

    

    def __init__(self, func, impute = False):

        self.func = func

        self.impute = impute

        self.series = pd.Series() 



    def transform(self, X, **transformparams):

        """ Transforms a DataFrame

        

        Parameters

        ----------

        X : DataFrame

            

        Returns

        ----------

        trans : pandas DataFrame

            Transformation of X 

        """

        

        if self.impute:

            trans = pd.DataFrame(X).fillna(self.series).copy()

        else:

            trans = pd.DataFrame(X).apply(self.func).copy()

        return trans



    def fit(self, X, y=None, **fitparams):

        """ Fixes the values to impute or does nothing

        

        Parameters

        ----------

        X : pandas DataFrame

        y : not used, API requirement

                

        Returns

        ----------

        self  

        """

        

        if self.impute:

            self.series = pd.DataFrame(X).apply(self.func).copy()

        return self

    

    

class DataFrameFeatureUnion(BaseEstimator, TransformerMixin):

    """ A DataFrame transformer that unites several DataFrame transformers

    

    Fit several DataFrame transformers and provides a concatenated

    Data Frame

    

    Parameters

    ----------

    list_of_transformers : list of DataFrameTransformers

        

    """ 

    def __init__(self, list_of_transformers):

        self.list_of_transformers = list_of_transformers

        

    def transform(self, X, **transformparamn):

        """ Applies the fitted transformers on a DataFrame

        

        Parameters

        ----------

        X : pandas DataFrame

        

        Returns

        ----------

        concatted :  pandas DataFrame

        

        """

        

        concatted = pd.concat([transformer.transform(X)

                            for transformer in

                            self.fitted_transformers_], axis=1).copy()

        return concatted





    def fit(self, X, y=None, **fitparams):

        """ Fits several DataFrame Transformers

        

        Parameters

        ----------

        X : pandas DataFrame

        y : not used, API requirement

        

        Returns

        ----------

        self : object

        """

        

        self.fitted_transformers_ = []

        for transformer in self.list_of_transformers:

            fitted_trans = clone(transformer).fit(X, y=None, **fitparams)

            self.fitted_transformers_.append(fitted_trans)

        return self

    



class ToDummiesTransformer(BaseEstimator, TransformerMixin):

    """ A Dataframe transformer that provide dummy variable encoding

    """

    

    def transform(self, X, **transformparams):

        """ Returns a dummy variable encoded version of a DataFrame

        

        Parameters

        ----------

        X : pandas DataFrame

        

        Returns

        ----------

        trans : pandas DataFrame

        

        """

    

        trans = pd.get_dummies(X).copy()

        return trans



    def fit(self, X, y=None, **fitparams):

        """ Do nothing operation

        

        Returns

        ----------

        self : object

        """

        return self





class DropAllZeroTrainColumnsTransformer(BaseEstimator, TransformerMixin):

    """ A DataFrame transformer that provides dropping all-zero columns

    """



    def transform(self, X, **transformparams):

        """ Drops certain all-zero columns of X

        

        Parameters

        ----------

        X : DataFrame

        

        Returns

        ----------

        trans : DataFrame

        """

        

        trans = X.drop(self.cols_, axis=1).copy()

        return trans



    def fit(self, X, y=None, **fitparams):

        """ Determines the all-zero columns of X

        

        Parameters

        ----------

        X : DataFrame

        y : not used

        

        Returns

        ----------

        self : object

        """

        

        self.cols_ = X.columns[(X==0).all()]

        return self
from sklearn.pipeline import Pipeline, make_pipeline
area_cols = X_train.columns[X_train.columns.str.contains('(?i)area|(?i)porch|(?i)sf')].tolist()



area_cols_pipeline = make_pipeline(  

        SelectColumnsTransfomer(area_cols),

        DataFrameFunctionTransformer(func = lambda x: x.astype(np.float64)),

        DataFrameFunctionTransformer(func = np.mean, impute=True),

        DataFrameFunctionTransformer(func = np.log1p) 

    )
object_columns = X_train.columns[X_train.dtypes == object].tolist()

object_levels = np.union1d(X_train[object_columns].fillna('NAN'), X_test[object_columns].fillna('NAN'))



categorical_cols_pipeline = make_pipeline(

        SelectColumnsTransfomer(object_columns),

        DataFrameFunctionTransformer(lambda x:'NAN', impute=True),

        DataFrameFunctionTransformer(lambda x:x.astype('category', categories=object_levels)),

        ToDummiesTransformer(),

        DropAllZeroTrainColumnsTransformer()

    )
remaining_cols = [x for x in X_train.columns.tolist() if x not in object_columns and x not in area_cols]



remaining_cols_pipeline = make_pipeline(

        SelectColumnsTransfomer(remaining_cols),

        DataFrameFunctionTransformer(func = lambda x: x.astype(np.float64)),

        DataFrameFunctionTransformer(func = np.mean, impute=True)

    )
preprocessing_features = DataFrameFeatureUnion([area_cols_pipeline, categorical_cols_pipeline, remaining_cols_pipeline])

preprocessing_features.fit_transform(X_train).head()
from sklearn.model_selection import GridSearchCV, cross_val_score

from sklearn.linear_model import Ridge

pipe_ridge = make_pipeline(preprocessing_features, Ridge())

param_grid = {'ridge__alpha' : [0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 30, 50, 75]}

pipe_ridge_gs = GridSearchCV(pipe_ridge, param_grid=param_grid, scoring = 'neg_mean_squared_error', cv=3)

result = np.sqrt(-cross_val_score(pipe_ridge_gs, X_train, np.log(y_train), scoring = 'neg_mean_squared_error', cv = 5))

np.mean(result)
pipe_ridge_gs.fit(X_train, np.log(y_train))

predicted = np.exp(pipe_ridge_gs.predict(X_test))

X_test["SalePrice"] = predicted

X_test["SalePrice"].reset_index().to_csv('pipe_ridge_gs.csv', index=False)

pipe_ridge_gs.best_params_