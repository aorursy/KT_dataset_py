import warnings
warnings.filterwarnings('ignore')
# Print files in input directory from Kaggle
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
# Basic Imports
import numpy as np 
import pandas as pd
data = pd.read_csv("/kaggle/input/housesalesprediction/kc_house_data.csv")
# Other sklearn imports
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
# sklearn pipeline imports
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion, Pipeline
# sklearn imports choosing the model type
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
# Metric Imports
from sklearn import metrics
from sklearn.model_selection import cross_val_score
# Feature Selection transformer class
class FeatureSelector( BaseEstimator, TransformerMixin ):
    """Custom Transformer that extracts columns passed as argument to its constructor """
    
    #Class Constructor 
    def __init__( self, feature_names ):
        self._feature_names = feature_names 
    
    #Return self nothing else to do here    
    def fit( self, X, y = None ):
        return self 
    
    #Method that describes what we need this transformer to do
    def transform( self, X, y = None ):
        return X.loc[:, self._feature_names ]
# Categorical Transformer
class CategoricalTransformer( BaseEstimator, TransformerMixin ):
    """ Custom transformer that breaks dates column into year,
        month and day into separate columns and
        converts certain features to binary. """
    
    #Class constructor method that takes in a list of values as its argument
    def __init__(self, use_dates = ['year', 'month', 'day'], cols = []):
        self._use_dates = use_dates
        self._cols = cols
        
    #Return self nothing else to do here
    def fit( self, X, y = None  ):
        return self

    #Helper function to extract year from column 'dates' 
    def get_year( self, obj ):
        return str(obj)[:4]
    
    #Helper function to extract month from column 'dates'
    def get_month( self, obj ):
        return str(obj)[4:6]
    
    #Helper function to extract day from column 'dates'
    def get_day(self, obj):
        return str(obj)[6:8]
    
    #Helper function that converts values to Binary depending on input 
    def create_binary(self, obj):
        if obj == 0:
            return 'No'
        else:
            return 'Yes'
    
    #Transformer method we wrote for this transformer 
    def transform(self, X , y = None ):
        #Depending on constructor argument break dates column into specified units
        #using the helper functions written above 
        for spec in self._use_dates:
            exec( "X.loc[:,'{}'] = X['date'].apply(self.get_{})".format( spec, spec ) )
            
        if (len(self._use_dates) != 0):
            #Drop unusable column 
            X = X.drop('date', axis = 1)
        
        for col in self._cols:
            #Convert these columns to binary for one-hot-encoding later
            if col in X.columns:
                X.loc[:, col ] = X[col].apply( self.create_binary )
        
        #returns numpy array
        return X.values 
# Numerical Transformer
class NumericalTransformer(BaseEstimator, TransformerMixin):
    """ Custom transformer to engineer features (bathrooms per
        bedroom and/or how old the house is in 2019) passed
        as boolen arguements to its constructor """
    #Class Constructor
    def __init__( self, bath_per_bed = True, years_old = True, cols = []):
        self._bath_per_bed = bath_per_bed
        self._years_old = years_old
        self._cols = cols
        
    #Return self, nothing else to do here
    def fit( self, X, y = None ):
        return self 
    
    #Custom transform method we wrote that creates aformentioned features and drops redundant ones 
    def transform(self, X, y = None):
        
        #Check if needed 
        if self._bath_per_bed and "bathrooms" in self._cols and "bedrooms" in self._cols:
            #create new column
            X.loc[:,'bath_per_bed'] = X['bathrooms'] / X['bedrooms']
            #drop redundant column
            X.drop('bathrooms', axis = 1 )
            
        #Check if needed     
        if self._years_old and "yr_built" in self._cols:
            #create new column
            X.loc[:,'years_old'] =  2019 - X['yr_built']
            #drop redundant column 
            X.drop('yr_built', axis = 1)
            
        #Converting any infinity values in the dataset to Nan
        X = X.replace( [ np.inf, -np.inf ], np.nan )
        
        #returns a numpy array
        return X.values
def build_pipeline(params, model_name):
    """Defining the steps in the categorical pipeline"""
    
    categorical_pipeline = Pipeline( steps = [ ( 'cat_selector', FeatureSelector(params[model_name]['cat_feat']) ),
                                  
                                              ( 'cat_transformer', CategoricalTransformer(use_dates = params[model_name]["use_dates"], cols=params[model_name]['cat_feat']) ),
                                  
                                              ( 'one_hot_encoder', OneHotEncoder( sparse = False ) ) ] )
    
    """Defining the steps in the numerical pipeline """
    numerical_pipeline = Pipeline( steps = [ ( 'num_selector', FeatureSelector(params[model_name]['num_feat']) ),
                                  
                                              ( 'num_transformer', NumericalTransformer(cols=params[model_name]['num_feat']) ),
                                  
                                              ('imputer', SimpleImputer(strategy = params[model_name]['simple_impute_strategy']) ),
                                  
                                              ( 'std_scaler', StandardScaler() ) ] )
    
    """ Combining numerical and categorical piepline into one
        full big pipeline horizontally using FeatureUnion """
    full_pipeline = FeatureUnion( transformer_list = [ ( 'categorical_pipeline', categorical_pipeline ), 
                                                  
                                                       ( 'numerical_pipeline', numerical_pipeline ) ] )
    
    return full_pipeline
""" Leave it as a dataframe becuase our pipeline is called on a 
    pandas dataframe to extract the appropriate columns, remember? """

X = data.drop('price', axis = 1)
#You can convert the target variable to numpy 
y = data['price'].values 

X_train, X_test, y_train, y_test = train_test_split( X, y , test_size = 0.2 , random_state = 42 )
# adjusted R squared function
def adjustedR2(r2,n,k):
    return r2-(k-1)/(n-k)*(1-r2)
evaluation = pd.DataFrame({'Model': [],
                           'Details':[],
                           'Root Mean Squared Error (RMSE)':[],
                           'R-squared (training)':[],
                           'Adjusted R-squared (training)':[],
                           'R-squared (test)':[],
                           'Adjusted R-squared (test)':[],
                           '5-Fold Cross Validation':[]})
models = {}
params = {}
# Linear Regression with only sqft_living
models['LinReg1'] = LinearRegression()

params["LinReg1"] = {"cat_feat": ['sqft_living'],
                     "num_feat": ["yr_built"],
                     "simple_impute_strategy":"median",
                     "use_dates": []}

def fit_now(params):
    y_pred = {}
    model_dict = {}
    for model_name, model in models.items():
        
        full_pipeline = build_pipeline(params, model_name)

        model_dict[model_name] = Pipeline( steps = [ ('full_pipeline', full_pipeline), ('model', model) ] )

        #Can call fit on it just like any other pipeline
        model_dict[model_name].fit( X_train, y_train )

        #Can predict with it like any other pipeline
        y_pred[model_name] = model_dict[model_name].predict( X_test )
    
    return model_dict, y_pred
model_dict, y_pred = fit_now(params)
y_pred
