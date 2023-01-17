# Python ≥3.5 is required
import sys
assert sys.version_info >= (3, 5)

# Scikit-Learn ≥0.20 is required
import sklearn
assert sklearn.__version__ >= "0.20"

# Common imports
import numpy as np
import os

# To plot pretty figures
%matplotlib inline
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

# Where to save the figures
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "end_to_end_project"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)
os.makedirs(IMAGES_PATH, exist_ok=True)

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)

# Ignore useless warnings (see SciPy issue #5998)
import warnings
warnings.filterwarnings(action="ignore", message="^internal gelsd")
import pandas as pd
import numpy as np 
%matplotlib inline
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
# read the data
train_data = pd.read_csv('../input/home-data-for-ml-course/train.csv')
test_data  = pd.read_csv('../input/home-data-for-ml-course/test.csv')
train_data.head()
test_data.head()
train_data.shape
train_data.info()
train_data.describe().T
index_num = list(train_data.describe().T.index)
train_data_num = train_data[index_num]
train_data_num.head()
import matplotlib.pyplot as plt
train_data.hist(bins=50, figsize=(25,20))
plt.show()
corr_matrix = train_data_num.corr()
corr_matrix["SalePrice"].sort_values(ascending=False)
plt.figure(figsize=(20,20))
sns.heatmap(corr_matrix,cmap='coolwarm',annot=True,linewidths=1)
plt.figure(figsize=(10,6))
train_data_num['SalePrice'].hist(bins=50)
#MSSubClass=The building class
train_data['MSSubClass'] = train_data['MSSubClass'].apply(str)


#Changing OverallCond into a categorical variable
train_data['OverallCond'] = train_data['OverallCond'].astype(str)


#Year and month sold are transformed into categorical features.
train_data['YrSold'] = train_data['YrSold'].astype(str)
train_data['MoSold'] = train_data['MoSold'].astype(str)


train_data.info()
index_num = list(train_data.describe().T.index)
train_data_num = train_data[index_num]
housing = train_data_num.drop(["SalePrice",'Id'], axis=1) # drop labels for training set
housing_labels = train_data_num["SalePrice"].copy()
housing.head()

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy="median")
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('std_scaler', StandardScaler()),
                        ])

housing_num_tr = num_pipeline.fit_transform(housing)
housing_num_tr = pd.DataFrame(housing_num_tr,columns= housing.columns,
                              index=housing.index)
housing_num_tr.info()
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
X = housing_num_tr #independent columns
y = housing_labels  #target column i.e price range
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
model = ExtraTreesClassifier()
model.fit(X,y)
print(model.feature_importances_) #use inbuilt class feature_importances of tree based classifiers
#plot graph of feature importances for better visualization
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
plt.figure(figsize=(20,20))
feat_importances.nlargest(36).plot(kind='barh')
plt.show()
most_30_feat= (feat_importances.nlargest(30)).index
most_30_feat





train_data_cat = train_data.drop(index_num,axis=1)
train_data_cat.info()
train_data_cat.head()
train_data_cat.head()
train_data_cat.mode()
train_data.shape
train_data_cat.isna().sum()
count_20 = (1460/100)*20
count_20
x = (train_data_cat.isna().sum()) > count_20
a = x[x!=True]
a
index_20 = list(a.index)
train_cat_20 = train_data_cat[index_20]
train_cat_20.head()
def clean_20 (data):
    count__20 = (len(data)/100)*20
    x = (data.isna().sum()) > count__20
    a = x[x!=True]
    index_20 = list(a.index)
    clean_data = data[index_20]
    return clean_data

from sklearn.preprocessing import LabelEncoder
cols = ( 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 
        'ExterQual', 'ExterCond','HeatingQC', 'KitchenQual', 'BsmtFinType1', 
        'BsmtFinType2', 'Functional',  'BsmtExposure', 'GarageFinish', 'LandSlope',
        'LotShape', 'PavedDrive', 'Street', 'CentralAir', 'MSSubClass', 'OverallCond', 
        'YrSold', 'MoSold')
# process columns, apply LabelEncoder to categorical features
for c in cols:
    lbl = LabelEncoder() 
    lbl.fit(list(train_cat_20[c].values)) 
    train_cat_20[c] = lbl.transform(list(train_cat_20[c].values))

# shape        
print('Shape train_cat_20: {}'.format(train_cat_20.shape))



train_cat_20 = pd.get_dummies(train_cat_20)
print(train_cat_20.shape)
train_cat_20.head()
housing_num_tr.head()
train_data_model = housing_num_tr.join(train_cat_20)
train_data_model.head()
test_data.info()
test_data_clean_20 = clean_20(test_data)
#MSSubClass=The building class
test_data_clean_20['MSSubClass'] = test_data_clean_20['MSSubClass'].apply(str)


#Changing OverallCond into a categorical variable
test_data_clean_20['OverallCond'] = test_data_clean_20['OverallCond'].astype(str)


#Year and month sold are transformed into categorical features.
test_data_clean_20['YrSold'] = test_data_clean_20['YrSold'].astype(str)
test_data_clean_20['MoSold'] = test_data_clean_20['MoSold'].astype(str)


index_num = list(test_data_clean_20.describe().T.index)
test_data_num = test_data_clean_20[index_num]

test_data_num.drop('Id',axis=1,inplace=True)
test_data_cat = test_data_clean_20.drop(index_num,axis=1)

test_data_cat.head()
test_data_tr = num_pipeline.transform(test_data_num)
test_data_num_tr = pd.DataFrame(test_data_tr,columns=test_data_num.columns,
                                            index = test_data_num.index )
test_data_num_tr.head()
# cat data
from sklearn.preprocessing import LabelEncoder
cols = ( 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 
        'ExterQual', 'ExterCond','HeatingQC', 'KitchenQual', 'BsmtFinType1', 
        'BsmtFinType2', 'Functional',  'BsmtExposure', 'GarageFinish', 'LandSlope',
        'LotShape', 'PavedDrive', 'Street', 'CentralAir', 'MSSubClass', 'OverallCond', 
        'YrSold', 'MoSold')
# process columns, apply LabelEncoder to categorical features
for c in cols:
    lbl = LabelEncoder() 
    lbl.fit(list(test_data_cat[c].values)) 
    test_data_cat[c] = lbl.transform(list(test_data_cat[c].values))

# shape        
print('Shape test_data_cat: {}'.format(test_data_cat.shape))



test_data_cat = pd.get_dummies(test_data_cat)
print(test_data_cat.shape)
test_data_cat.head()
s=list(test_data_model.columns)
d=list(train_data_model.columns)
tr = train_data_model.drop(s,axis=1)
miss_cl=(tr.columns)
test_data_model = test_data_num_tr.join(test_data_cat)
test_data_model.head()
train_data_model=train_data_model.drop(miss_cl,axis=1)
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(train_data_model, housing_labels)
from sklearn.metrics import mean_squared_error

housing_predictions = lin_reg.predict(train_data_model)
lin_mse = mean_squared_error(housing_labels, housing_predictions)
lin_rmse = np.sqrt(lin_mse)
lin_rmse
from sklearn.metrics import mean_absolute_error

lin_mae = mean_absolute_error(housing_labels, housing_predictions)
lin_mae
from sklearn.tree import DecisionTreeRegressor

tree_reg = DecisionTreeRegressor(random_state=42)
tree_reg.fit(train_data_model, housing_labels)
housing_predictions = tree_reg.predict(train_data_model)
tree_mse = mean_squared_error(housing_labels, housing_predictions)
tree_rmse = np.sqrt(tree_mse)
tree_rmse
from sklearn.model_selection import cross_val_score

scores = cross_val_score(tree_reg, train_data_model, housing_labels,
                         scoring="neg_mean_squared_error", cv=10)
tree_rmse_scores = np.sqrt(-scores)
def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())

display_scores(tree_rmse_scores)
lin_scores = cross_val_score(lin_reg, train_data_model, housing_labels,
                             scoring="neg_mean_squared_error", cv=10)
lin_rmse_scores = np.sqrt(-lin_scores)
display_scores(lin_rmse_scores)
from sklearn.ensemble import RandomForestRegressor

forest_reg = RandomForestRegressor(n_estimators=100, random_state=42)
forest_reg.fit(train_data_model, housing_labels)
housing_predictions = forest_reg.predict(train_data_model)
forest_mse = mean_squared_error(housing_labels, housing_predictions)
forest_rmse = np.sqrt(forest_mse)
forest_rmse
from sklearn.model_selection import cross_val_score

forest_scores = cross_val_score(forest_reg, train_data_model, housing_labels,
                                scoring="neg_mean_squared_error", cv=10)
forest_rmse_scores = np.sqrt(-forest_scores)
display_scores(forest_rmse_scores)
scores = cross_val_score(lin_reg, train_data_model, housing_labels, scoring="neg_mean_squared_error", cv=10)
pd.Series(np.sqrt(-scores)).describe()
from sklearn.model_selection import GridSearchCV

param_grid = [
    # try 12 (3×4) combinations of hyperparameters
    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
    # then try 6 (2×3) combinations with bootstrap set as False
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
  ]

forest_reg = RandomForestRegressor(random_state=42)
# train across 5 folds, that's a total of (12+6)*5=90 rounds of training 
grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
                           scoring='neg_mean_squared_error',
                           return_train_score=True)
grid_search.fit(train_data_model, housing_labels)
grid_search.best_params_
grid_search.best_estimator_
cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

param_distribs = {
        'n_estimators': randint(low=1, high=200),
        'max_features': randint(low=1, high=8),
    }

forest_reg = RandomForestRegressor(random_state=42)
rnd_search = RandomizedSearchCV(forest_reg, param_distributions=param_distribs,
                                n_iter=10, cv=5, scoring='neg_mean_squared_error', random_state=42)
rnd_search.fit(train_data_model, housing_labels)
cvres = rnd_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)
feature_importances = grid_search.best_estimator_.feature_importances_
feature_importances
final_model = grid_search.best_estimator_

final_predictions = final_model.predict(test_data_model)

final_predictions
submit=pd.read_csv('../input/home-data-for-ml-course/sample_submission.csv')
submit['SalePrice']= final_predictions
submit.head()
sumbit.to_csv('house1.csv')
