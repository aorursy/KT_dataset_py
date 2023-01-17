# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np



from xgboost import XGBClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.svm import SVC



from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import OneHotEncoder

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import MinMaxScaler

from sklearn.base import BaseEstimator, TransformerMixin



from sklearn.metrics import make_scorer, accuracy_score

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import train_test_split



import sklearn

import sys



from sklearn.feature_selection import VarianceThreshold

from sklearn.feature_selection import RFE

from sklearn.feature_selection import SelectFromModel

from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import chi2

from sklearn.feature_selection import mutual_info_classif

from sklearn.feature_selection import f_classif



from sklearn.preprocessing import PolynomialFeatures



from scipy.stats import norm

from scipy import stats



from sklearn.pipeline import FeatureUnion

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer



import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline

sns.set_style("darkgrid")



print ("Import finished...")
# Load and Split input Data

df_train = pd.read_csv ("../input/train.csv", index_col=0)

X_test = pd.read_csv ("../input/test.csv", index_col=0)



X_full = df_train.dropna (axis=0, subset=['Survived'])



y = X_full.pop ('Survived')



X_train, X_valid, y_train, y_valid = train_test_split (X_full, y, train_size=0.5, test_size=0.5, random_state=0)
class FetureProcessTransform (BaseEstimator, TransformerMixin):

    ''' Feture Processing and Transformation '''

    

    def __init__ (self):

        pass

    

    def fit (self, X, y=None):

        return self

    

    def transform (self, X):

        

        X.loc[:, 'EmbarkedNum'] = X['Embarked'].map({'S' : 0, 'C' : 1, 'Q' : 2, np.nan : 0})

        

        X.loc[:, 'FamilySize'] = X['SibSp'] + X['Parch']



        FamilySizeDict = {0 : 2, 1 : 3, 2 : 3, 3 : 3, 4 : 1, 5 : 2, 6 : 1, 7 : 1, 10 : 1}

        X.loc[:,'FamilySize'] = X['FamilySize'].map(lambda x: FamilySizeDict[x])

        

        X.loc[:, 'SexBinary'] = X['Sex'].map({'female' : 0, 'male' : 1})

        

        DeckDict = {'A' : 1, 'B' : 2, 'C' : 2, 'D' : 2, 'E' : 2, 'F' : 2, 'G' : 1, 'T' : 1, 'M' : 0}

        

        X.loc[:, 'CabinMissing'] = X['Cabin'].isna()

        

        X.loc[:,'Cabin'] = X['Cabin'].fillna('Missing')

        X.loc[:,'Deck'] = X['Cabin'].map(lambda x: DeckDict[x[0]])



        ####

        title_trans_dict = {'Mrs': 0, 'Mme': 0,'Countess': 0, 'Lady': 0, 'Dona': 0,

                            'Mr': 1, 'Don': 1, 'Major': 1, 'Capt': 1, 'Jonkheer': 1, 

                            'Rev': 1, 'Col': 1, 'Sir': 1, 'Dr': 1, 

                            'Master': 2, 'Miss': 3, 'Mlle': 3, 'Ms': 3}

    

        X.loc[:,'Title'] = X['Name'].str.extract('([A-Za-z]+)\.', expand=False)

        X.loc[:,'Title'] = X['Title'].map (title_trans_dict)



        X.loc[(X['Title'] == 1) & (X['Sex'] == 0), 'Title'] = 0

        

        ### Bin Fare column

        X.loc[:, 'FareBinned'] = X['Fare'].fillna(X['Fare'].mean())

        

        X.loc[:, 'FareBinned'] = pd.cut(X['FareBinned'], bins=[0, 25, 50, 100, X['FareBinned'].max()], 

                                        right=True, labels=[0, 1, 2, 3], include_lowest=True)

        

        X.loc[:, 'AgeImputed'] = 0

        X.loc[:, 'AgeImputed'] = X['Age'].isna()

        

        X.loc[:, 'AgeBinned'] = X['Age'].fillna(X['Age'].mean())

        

        X.loc[:, 'AgeBinned'] = pd.cut(X['AgeBinned'], bins=[0, 20, 40, 60, X['AgeBinned'].max()], 

                                        right=True, labels=[0, 1, 2, 3], include_lowest=True)       

        

        # Return columns

#         cols = ['Pclass', 'SexBinary', 'FamilySize', 'EmbarkedNum','Deck', 'Title', 'FareBinned', 'AgeBinned']

        cols = ['Pclass', 'SexBinary', 'FamilySize', 'EmbarkedNum', 'Deck', 'Title', 

                'FareBinned', 'AgeBinned', 'AgeImputed', 'CabinMissing']

        

        return X[cols].values
cont_pl = Pipeline (steps = [ ( 'ageFare', SimpleImputer (strategy='mean') ),

                              ( 'scaler', MinMaxScaler () ), 

#                               ( 'scaler', StandardScaler () ), 

                              ( 'poly_c', PolynomialFeatures (interaction_only=True, include_bias=False) )

                            ] )
onehot_pl = Pipeline (steps = [ ( 'fetset', FetureProcessTransform () ),

                                ( 'onehot', OneHotEncoder (handle_unknown='ignore') ),

#                                 ( 'poly_o', PolynomialFeatures (degree=2, interaction_only=True, 

#                                                                 include_bias=False, order='F') ),

                                ( 'var_thresh', VarianceThreshold (0.8 * (1 - 0.8)))

                              ] )
cont_pl_cols = ['Age', 'Fare']

cont_col_trans = ColumnTransformer ( transformers = [ ( 'cont_pl', cont_pl, cont_pl_cols ) ] )
onehot_pl_cols = X_train.columns

onehot_col_trans = ColumnTransformer ( transformers = [ ( 'onehot_pl', onehot_pl, onehot_pl_cols ) ] )
all_features = FeatureUnion ( [ ( 'cont_col_trans', cont_col_trans ), 

                                ( 'onehot_col_trans', onehot_col_trans ) ] )
# svc0 = SVC (kernel="linear", C=0.1, random_state = 1)



# select_pl = Pipeline ( steps = [ ( 'all_features', all_features ),

#                                  ( 'rfe', RFE (estimator=svc0, n_features_to_select=11, step=1) ) ] )
# etc0 = ExtraTreesClassifier (n_estimators=50, criterion='entropy', max_depth = 3, 

#                             max_features='log2', random_state=0)



# select_pl = Pipeline ( steps = [ ( 'all_features', all_features ),

#                                  ( 'sfm', SelectFromModel (etc0, max_features = 11) ) ] )
select_pl = Pipeline ( steps = [ ( 'all_features', all_features ),

                                 ( 'selectKBest', SelectKBest ( f_classif, k=15 ) ) ] )
# Extra Trees Classifier

etc = ExtraTreesClassifier (n_estimators = 50, criterion='entropy', max_depth = 3, 

                            max_features='log2', random_state=0)
# SVC

svc = SVC (kernel="linear", C=0.1, random_state = 1)
# Gradient Boosting Classifier

gbc = GradientBoostingClassifier (max_depth = 3, learning_rate = 0.14, n_estimators = 50,

                                  max_features = 'log2', random_state = 0)
# Logistic Regression

lr = LogisticRegression (solver='lbfgs', C=0.1, random_state=0, multi_class='ovr')
# Random Forest Classifier

rfc = RandomForestClassifier (max_depth = 3, n_estimators = 50, criterion='entropy', 

                              random_state = 0, max_features = 'log2')
# XGB Classifier

xgb = XGBClassifier (max_depth=3, eta=0.14, silent=1, objective='binary:hinge')
model_pl = Pipeline (steps = [ ('select_pl', select_pl), ('model', xgb) ])
# param_grid = {

#     'select_pl__selectKBest__k' : [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

# }



# search = GridSearchCV (model_pl, param_grid, iid=False, cv=5)



# search.fit (X_full, y);



# print("Best parameter (CV score=%0.3f):" % search.best_score_)



# print("Best parameters :", search.best_params_)
# results = pd.DataFrame(search.cv_results_)

# results.head(28)
# Fit the model

model_pl.fit (X_train, y_train);
# Validate and fine the score

print ('Score : %.3f' % model_pl.score (X_valid, y_valid))
# Predict 

pred = model_pl.predict (X_test)
# Save prediction data to output

output = pd.DataFrame ({'PassengerId': X_test.index, 'Survived': pred})

output.to_csv ('submission.csv', index=False)
# output.head(10)