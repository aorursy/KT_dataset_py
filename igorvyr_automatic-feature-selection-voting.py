import numpy as np
import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import Imputer, LabelBinarizer, StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline, FeatureUnion
rawdata=pd.read_csv("../input/train.csv") #reading data
# Definition of the CategoricalEncoder class, copied from PR #9151.
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array
from sklearn.preprocessing import LabelEncoder
from scipy import sparse

class CategoricalEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, encoding='onehot', categories='auto', dtype=np.float64,
                 handle_unknown='error'):
        self.encoding = encoding
        self.categories = categories
        self.dtype = dtype
        self.handle_unknown = handle_unknown

    def fit(self, X, y=None):
        if self.encoding not in ['onehot', 'onehot-dense', 'ordinal']:
            template = ("encoding should be either 'onehot', 'onehot-dense' "
                        "or 'ordinal', got %s")
            raise ValueError(template % self.handle_unknown)

        if self.handle_unknown not in ['error', 'ignore']:
            template = ("handle_unknown should be either 'error' or "
                        "'ignore', got %s")
            raise ValueError(template % self.handle_unknown)

        if self.encoding == 'ordinal' and self.handle_unknown == 'ignore':
            raise ValueError("handle_unknown='ignore' is not supported for"
                             " encoding='ordinal'")

        X = check_array(X, dtype=np.object, accept_sparse='csc', copy=True)
        n_samples, n_features = X.shape
        self._label_encoders_ = [LabelEncoder() for _ in range(n_features)]
        for i in range(n_features):
            le = self._label_encoders_[i]
            Xi = X[:, i]
            if self.categories == 'auto':
                le.fit(Xi)
            else:
                valid_mask = np.in1d(Xi, self.categories[i])
                if not np.all(valid_mask):
                    if self.handle_unknown == 'error':
                        diff = np.unique(Xi[~valid_mask])
                        msg = ("Found unknown categories {0} in column {1}"
                               " during fit".format(diff, i))
                        raise ValueError(msg)
                le.classes_ = np.array(np.sort(self.categories[i]))
        self.categories_ = [le.classes_ for le in self._label_encoders_]
        return self

    def transform(self, X):
        X = check_array(X, accept_sparse='csc', dtype=np.object, copy=True)
        n_samples, n_features = X.shape
        X_int = np.zeros_like(X, dtype=np.int)
        X_mask = np.ones_like(X, dtype=np.bool)

        for i in range(n_features):
            valid_mask = np.in1d(X[:, i], self.categories_[i])

            if not np.all(valid_mask):
                if self.handle_unknown == 'error':
                    diff = np.unique(X[~valid_mask, i])
                    msg = ("Found unknown categories {0} in column {1}"
                           " during transform".format(diff, i))
                    raise ValueError(msg)
                else:
                    X_mask[:, i] = valid_mask
                    X[:, i][~valid_mask] = self.categories_[i][0]
            X_int[:, i] = self._label_encoders_[i].transform(X[:, i])

        if self.encoding == 'ordinal':
            return X_int.astype(self.dtype, copy=False)

        mask = X_mask.ravel()
        n_values = [cats.shape[0] for cats in self.categories_]
        n_values = np.array([0] + n_values)
        indices = np.cumsum(n_values)

        column_indices = (X_int + indices[:-1]).ravel()[mask]
        row_indices = np.repeat(np.arange(n_samples, dtype=np.int32),
                                n_features)[mask]
        data = np.ones(n_samples * n_features)[mask]

        out = sparse.csc_matrix((data, (row_indices, column_indices)),
                                shape=(n_samples, indices[-1]),
                                dtype=self.dtype).tocsr()
        if self.encoding == 'onehot-dense':
            return out.toarray()
        else:
            return out
def first_digit(x):
    if str.isdigit(x):
        return x[0]
    else:
        return x
    
def first_letter(x):
    if isinstance(x,str):
        return x[0]
    else:
        return '0'
# class for droping data with some incomplete features:
class NaDroper(BaseEstimator, TransformerMixin):  
    def fit(self,X,y=None):
        return self
    def transform(self, X):
        features_to_check=["Pclass", "Sex", "Embarked", "SibSp", "Parch"]
        return X.dropna(subset=features_to_check)
    
# class for selecting numerical features:
class NumSelector(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.features=["Age", "Fare", "SibSp", "Parch"]  
    def fit(self,X,y=None):
        return self
    def transform(self, X):
        XX=X.copy()
        return XX[self.features]
    
# class for selecting/creating categorical features:    
class CatSelector(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.features=["Pclass", "Sex", "Embarked", "CabinCat",
                       "Title", "TicketCat", "SibSpBin", "ParchBin", "AgeBin"]       
    def fit(self,X,y=None):
        return self
    def transform(self, X):
        XX=X.copy()
        #extracting Title:
        XX['Title']=XX['Name'].str.split(',').map(lambda x: x[1]).str.split('.').map(lambda x: x[0])
        #extracting the string in front of Ticket number OR first digit of Ticket number
        XX['TicketCat']=XX['Ticket'].str.split(' ').map(lambda x: x[0]).str.split('/').map(lambda x: x[0])\
        .str.replace('.','').map(lambda x: first_digit(x))
        XX["SibSpBin"]=X["SibSp"]>0
        XX["ParchBin"]=X["Parch"]>0
        #extracting first letter of Cabin
        XX["CabinCat"]=X["Cabin"].map(lambda x: first_letter(x))
        XX["AgeBin"]=np.isnan(X["Age"])
        return XX[self.features]  

#class for selecting important features
class LogL1Selector(BaseEstimator, TransformerMixin):
    def __init__(self, C=1):
        self.C=C
    def fit(self,X,y):
        self.lgr=LogisticRegression(C=self.C, penalty="l1", fit_intercept=True, tol=1e-6)
        self.lgr.fit(X,y)
        self.features=np.where(self.lgr.coef_!=0)[1]
        return self
    def transform(self, X):
        return X[:,self.features]
# instance of NaDroper class for droping the data that are incomplete in categorical features.
nadroper=NaDroper()

# pipeline to preprocess numerical data
num_pipeline=Pipeline([('numselector',NumSelector()),
                      ('imputer',Imputer(strategy='median')),
                      ('standardscaler',StandardScaler())
                      ])
# pipeline to preprocess categorical data
cat_pipeline=Pipeline([('catselector',CatSelector()),
                       ('categoricalencoder',CategoricalEncoder(encoding='onehot-dense', 
                                                               handle_unknown='ignore')),
                      ])
# joint pipeline for preprocessing both numerical and categorical data:
preprocess_pipeline0=FeatureUnion(transformer_list=[('numpip',num_pipeline),
                                                    ('catpip',cat_pipeline)])

# pipeline for possible inclusion of interactions and for feature selction
preprocess_pipeline1=Pipeline([('ppip0', preprocess_pipeline0),
                               ('polyf', PolynomialFeatures(degree=1)),
                               ('logl1selector',LogL1Selector(C=1))])
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

full_pipeline=[]
full_pipeline.append(Pipeline([('ppip1', preprocess_pipeline1),
                        ('algo', SVC(kernel='rbf',probability=True))]))

full_pipeline.append(Pipeline([('ppip1', preprocess_pipeline1),
                        ('algo', RandomForestClassifier(n_estimators=1000, max_depth=3))]))

full_pipeline.append(Pipeline([('ppip1', preprocess_pipeline1),
                        ('algo', KNeighborsClassifier())]))

full_pipeline.append(Pipeline([('ppip1', preprocess_pipeline1),
                        ('algo', LogisticRegression(tol=1e-6))]))

# parametric grid we will search in
param_grid ={'ppip1__polyf__degree':[ 1, 2],
         'ppip1__polyf__interaction_only':[True],
         'ppip1__logl1selector__C':[0.1,0.25,0.5,0.75,1,1.25,1.5,1.75,2,2.5,5,7.5,10,15],
        }
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import cross_validate, cross_val_score, cross_val_predict

data_cleaned=nadroper.fit_transform(rawdata)
y=data_cleaned['Survived']

gscv=[None] * len(full_pipeline)
top_estim=[None] * len(full_pipeline)
for k in range(len(full_pipeline)):
    gscv[k]=GridSearchCV(full_pipeline[k], param_grid, cv=3, scoring='accuracy',n_jobs=-1,
                                   verbose=1)
    gscv[k].fit(data_cleaned,data_cleaned["Survived"])
    print(gscv[k].best_params_)
    top_estim[k]=gscv[k].best_estimator_
    print(cross_val_score(top_estim[k], data_cleaned,y))
estimators=[('est'+str(k),top_estim[k]) for k in range(len(full_pipeline))]
from sklearn.ensemble import VotingClassifier
votc=VotingClassifier(estimators=estimators, n_jobs=-1, voting="soft")
cross_val_score(votc, data_cleaned,data_cleaned["Survived"])
votc.fit(data_cleaned,data_cleaned["Survived"])
testdata=pd.read_csv("../input/test.csv") 
y_test_prediction=votc.predict(testdata)
submission = pd.DataFrame({
        "PassengerId": testdata["PassengerId"],
        "Survived": y_test_prediction
    })
submission.to_csv('submission.csv', index=False)
