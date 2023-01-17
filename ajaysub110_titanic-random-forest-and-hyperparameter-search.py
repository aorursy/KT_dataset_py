import numpy as np
import pandas as pd
%matplotlib inline
import matplotlib.pyplot as plt
data_path = '../input'

def fetch_data(data_path):
    train_df = pd.read_csv(data_path + '/train.csv')
    test_df = pd.read_csv(data_path + '/test.csv')
    samp_df = pd.read_csv(data_path + '/gender_submission.csv')
    return train_df, test_df, samp_df
train_df, test_df, samp_df = fetch_data(data_path)
train_df.head()
train_df.info()
train_df.describe()
train_df.Survived.value_counts()
train_df.Pclass.value_counts()
train_df.Sex.value_counts()
train_df.Embarked.value_counts()
from collections import Counter

def detect_outliers(df,n,features):
    """
    Takes a dataframe df of features and returns a list of the indices
    corresponding to the observations containing more than n outliers according
    to the Tukey method.
    """
    outlier_indices = []
    
    # iterate over features(columns)
    for col in features:
        # 1st quartile (25%)
        Q1 = np.percentile(df[col], 25)
        # 3rd quartile (75%)
        Q3 = np.percentile(df[col],75)
        # Interquartile range (IQR)
        IQR = Q3 - Q1
        
        # outlier step
        outlier_step = 1.5 * IQR
        
        # Determine a list of indices of outliers for feature col
        outlier_list_col = df[(df[col] < Q1 - outlier_step) | (df[col] > Q3 + outlier_step )].index
        
        # append the found outlier indices for col to the list of outlier indices 
        outlier_indices.extend(outlier_list_col)
        
    # select observations containing more than 2 outliers
    outlier_indices = Counter(outlier_indices)        
    multiple_outliers = list( k for k, v in outlier_indices.items() if v > n )
    
    return multiple_outliers   

# detect outliers from Age, SibSp , Parch and Fare
Outliers_to_drop = detect_outliers(train_df,2,["Age","SibSp","Parch","Fare"])
train_df.loc[Outliers_to_drop] # Show the outliers rows
corr_matrix = train_df.corr()
corr_matrix['Survived'].sort_values(ascending=False)
train_df['Family_number'] = train_df.SibSp + train_df.Parch # number of family members on board
train_df['Age_cats'] = np.ceil(train_df.Age / 18) # Creating Age ranges to split into categories
train_df.head()
# Drop outliers
train_drop = train_df.drop(Outliers_to_drop, axis = 0).reset_index(drop=True)
from sklearn.base import BaseEstimator, TransformerMixin

class DataFrameSelector(BaseEstimator,TransformerMixin):
    def __init__(self,attrib_names):
        self.attrib_names = attrib_names
    def fit(self,X,y=None):
        return self
    def transform(self,X):
        return X[self.attrib_names]
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Imputer

imputer = Imputer(strategy="median")

num_pipeline = Pipeline([
        ("select_numeric", DataFrameSelector(["Age", "SibSp", "Parch", "Fare", "Family_number"])),
        ("imputer", Imputer(strategy="median")),
    ])
num_pipeline.fit_transform(train_drop)
from sklearn.preprocessing import OneHotEncoder
from sklearn_pandas import CategoricalImputer
class MostFrequentImputer(BaseEstimator,TransformerMixin):
    def fit(self,X,y=None):
        self.imputer = CategoricalImputer()
        return self
    def transform(self,X):
        age_cats_imputed = pd.Series(self.imputer.fit_transform(X.Age_cats.copy())).astype('category')
        sex_imputed = pd.Series(self.imputer.fit_transform(X.Sex.copy())).astype('category')
        embarked_imputed = pd.Series(self.imputer.fit_transform(X.Embarked.copy())).astype('category')
        X.Sex = sex_imputed.cat.codes
        X.Embarked = embarked_imputed.cat.codes
        X.Age_cats = age_cats_imputed
        return X
cat_pipeline = Pipeline([
        ("select_cat", DataFrameSelector(["Pclass", "Sex", "Embarked", "Age_cats"])),
        ("imputer", MostFrequentImputer()),
        ("cat_encoder", OneHotEncoder(sparse=False)),
    ])
cat_pipeline.fit_transform(train_drop)
from sklearn.pipeline import FeatureUnion
preprocess_pipeline = FeatureUnion(transformer_list=[
        ("num_pipeline", num_pipeline),
        ("cat_pipeline", cat_pipeline),
    ])
x_train = preprocess_pipeline.fit_transform(train_drop)
x_train
y_train = train_drop["Survived"]
test_df['Family_number'] = test_df.SibSp + test_df.Parch # number of family members on board
test_df['Age_cats'] = np.ceil(test_df.Age / 18) # Creating Age ranges to split into categories
# SVM
from sklearn.svm import SVC

svm_clf = SVC()
svm_clf.fit(x_train,y_train)
X_test = preprocess_pipeline.transform(test_df)
y_pred = svm_clf.predict(X_test)
from sklearn.model_selection import cross_val_score

svm_scores = cross_val_score(svm_clf, x_train, y_train, cv=10)
svm_scores.mean()
# Random forest
from sklearn.ensemble import RandomForestClassifier

forest_clf = RandomForestClassifier(random_state=42)
forest_clf.fit(x_train,y_train)
forest_scores = cross_val_score(forest_clf, x_train, y_train, cv=10)
forest_scores.mean()
plt.figure(figsize=(8, 4))
plt.plot([1]*10, svm_scores, ".")
plt.plot([2]*10, forest_scores, ".")
plt.boxplot([svm_scores, forest_scores], labels=("SVM","Random Forest"))
plt.ylabel("Accuracy", fontsize=14)
plt.show()
from sklearn.model_selection import GridSearchCV

param_grid = [
    {'n_estimators': [3, 10, 20, 30], 'max_features': [2, 4, 6, 8, 10, 12]},
    # then try 6 (2×3) combinations with bootstrap set as False
    {'bootstrap': [False], 'n_estimators': [3, 10, 20], 'max_features': [2, 3, 4]},
]

forest_clf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(forest_clf, param_grid, cv=5,
                           scoring='accuracy', return_train_score=True)
grid_search.fit(x_train,y_train)
grid_search.best_params_
grid_search.best_estimator_
cvres = grid_search.cv_results_
for acc_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(acc_score, params)
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

param_distribs = {
        'n_estimators': randint(low=1, high=200),
        'max_features': randint(low=1, high=8),
    }

forest_clf = RandomForestClassifier(random_state=42)
rnd_search = RandomizedSearchCV(forest_clf, param_distributions=param_distribs,
                                n_iter=30, cv=5, scoring='accuracy', random_state=42)
rnd_search.fit(x_train, y_train)
cvres = rnd_search.cv_results_
for acc_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(acc_score, params)
rnd_search.best_score_
rnd_search.best_estimator_
preds = rnd_search.best_estimator_.predict(X_test)
preds
preds.shape
samp_df.head()
subm_df = samp_df.copy()
subm_df.Survived = preds
subm_df.head()
subm_df.to_csv('submission.csv', index=False)