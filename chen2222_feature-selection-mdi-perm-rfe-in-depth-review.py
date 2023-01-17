import numpy as np

import os

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import category_encoders as ce 

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.linear_model import Ridge, Lasso

from sklearn.model_selection import train_test_split, GridSearchCV

import xgboost as xgb

from sklearn.model_selection import cross_val_score

from sklearn import metrics

from sklearn.feature_selection import SelectFromModel

from collections import Counter

import pickle

from sklearn.ensemble import RandomForestRegressor

from sklearn.pipeline import FeatureUnion

from sklearn.impute import MissingIndicator

from sklearn.model_selection import KFold

from sklearn.feature_selection import RFECV

from sklearn.inspection import permutation_importance

from category_encoders import TargetEncoder, LeaveOneOutEncoder

import random
df_train =  pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")

df_test = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")

def seed_everything(seed):

    random.seed(seed)

    np.random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

seed_everything(0)
y_train = np.log(df_train['SalePrice'])

X_train = df_train.drop(['SalePrice','Id'],axis=1)

X_test = df_test.drop('Id',axis=1)

X_train_noise = X_train.copy()

print('Train data: ',X_train.shape)

print('Test data: ',X_test.shape)
cat_nominal_features = pickle.load(open('../input/features-housing/cat_nominal_features.p', "rb" ))

cat_ordinal_features = pickle.load(open('../input/features-housing/cat_ordinal_features.p', "rb" ))

num_features = pickle.load(open('../input/features-housing/num_features.p', "rb" ))



# Besides these features, we are also going to add a random noise feature

cat_features = cat_nominal_features + cat_ordinal_features

num_features_noise = num_features+['noise']

print('Number of numeric features: ',len(num_features))

print('Number of numeric features including noise: ',len(num_features_noise))

print('Number of ordinal features: ',len(cat_ordinal_features))

print('Number of nominal featuures: ',len(cat_nominal_features))
# Fill missing categorical entries with string "Missing"

# Fill missing numeric entries with np.nan

# We are going to let Xgboost to automatically handle missing features for us

X_train_noise['noise'] = np.random.normal(size=(1460))

X_train[cat_ordinal_features] = X_train[cat_ordinal_features].fillna(np.nan)

X_train[cat_nominal_features] = X_train[cat_nominal_features].fillna('Missing')

X_train_noise[cat_ordinal_features] = X_train_noise[cat_ordinal_features].fillna(np.nan)

X_train_noise[cat_nominal_features] = X_train_noise[cat_nominal_features].fillna('Missing')
model = xgb.XGBRegressor(random_state=0)

scoring = 'neg_root_mean_squared_error'
# These features have order information embeded in them, that's why we use ordinal encoder.

ce_ord = ce.OrdinalEncoder(cols=cat_ordinal_features, mapping=[{'col': 'Street', 'mapping': {'Grvl': 1, 'Pave': 2}},

                                            {'col': 'Alley', 'mapping': {np.nan:0,'Grvl': 1, 'Pave': 2}}, 

                                            {'col': 'Utilities', 'mapping': {'NoSeWa': 1, 'AllPub':2}},

                                            {'col': 'ExterQual', 'mapping': {'Po':1,'Fa':2,'TA':3,

                                                                            'Gd':4,'Ex':5}},

                                            {'col': 'ExterCond', 'mapping': {'Po':1,'Fa':2,'TA':3,

                                                                            'Gd':4,'Ex':5}},

                                            {'col': 'BsmtCond', 'mapping': {np.nan:0,'Po':1,'Fa':2,'TA':3,

                                                                            'Gd':4,'Ex':5}},

                                            {'col': 'BsmtQual', 'mapping': {np.nan:0,'Po':1,'Fa':2,'TA':3,

                                                                            'Gd':4,'Ex':5}},

                                            {'col': 'BsmtExposure', 'mapping': {np.nan:0,'No':1,'Mn':2,'Av':3,

                                                                            'Gd':4}},

                                            {'col': 'BsmtFinType1', 'mapping': {np.nan:0,'Unf':1,'LwQ':2,'Rec':3,

                                                                            'BLQ':4,'ALQ':5,'GLQ':6}},

                                            {'col': 'BsmtFinType2', 'mapping': {np.nan:0,'Unf':1,'LwQ':2,'Rec':3,

                                                                            'BLQ':4,'ALQ':5,'GLQ':6}},     

                                            {'col': 'HeatingQC', 'mapping': {'Po':1,'Fa':2,'TA':3,

                                                                            'Gd':4,'Ex':5}},

                                            {'col': 'CentralAir', 'mapping': {'Y':1,'N':0}},

                                            {'col': 'KitchenQual', 'mapping': {'Po':1,'Fa':2,'TA':3,

                                                                            'Gd':4,'Ex':5}}, 

                                            {'col': 'Functional', 'mapping': {'Typ':8,'Min1':7,'Min2':6,

                                                                            'Mod':5,'Maj1':4,'Maj2':3,

                                                                             'Sev':2,"Sal":1}},                                    

                                            {'col': 'FireplaceQu', 'mapping': {np.nan:0,'Po':1,'Fa':2,'TA':3,

                                                                            'Gd':4,'Ex':5}},

                                            {'col': 'GarageFinish', 'mapping': {np.nan:0,'Unf':1,'RFn':2,'Fin':3}},

                                            {'col': 'GarageQual', 'mapping': {np.nan:0,'Po':1,'Fa':2,'TA':3,

                                                                            'Gd':4,'Ex':5}},

                                            {'col': 'GarageCond', 'mapping': {np.nan:0,'Po':1,'Fa':2,'TA':3,

                                                                            'Gd':4,'Ex':5}},

                                            {'col': 'PavedDrive', 'mapping': {'N':1,'P':2,'Y':3}},

                                            {'col': 'PoolQC', 'mapping': {np.nan:0,'Fa':1,'TA':2,

                                                                            'Gd':3,'Ex':4}},

                                            {'col': 'Fence', 'mapping': {np.nan:0,'MnWw':1,'GdWo':2,'MnPrv':3,

                                                                            'GdPrv':4}}],

                                            handle_unknown='value',

                                            handle_missing='value')



ce_nom = ce.OneHotEncoder(cols=cat_nominal_features,handle_unknown='value',handle_missing='value')
# This function is used to define the processing pipeline

def get_tree(seed):

    numeric_transformer = Pipeline(steps=[

            ('imputer', SimpleImputer(strategy='constant',fill_value=-1)),

            ])



    ct1 = ColumnTransformer(

            transformers=[

                ('nominal',ce_nom,cat_nominal_features),

                ('ordinal',ce_ord,cat_ordinal_features),

                ('num',numeric_transformer,num_features_noise)

                ],remainder = 'passthrough')

    clf_tree = Pipeline(steps=[('preprocessor', ct1),

                          ('regressor', xgb.XGBRegressor(random_state=seed))])

    return clf_tree
# We are going to do a 5-fold cross-validation for MDI-based features

def MDI_imp(train_data,y_data):

    all_feature_imp = []

    kFold = KFold(n_splits=5, random_state=0, shuffle=True)

    clf_tree = get_tree(0)

    for fold, (trn_idx, val_idx) in enumerate(kFold.split(train_data)):

        X_train_temp = train_data[trn_idx]

        X_val_temp = train_data[val_idx]

        y_train_temp = y_data.loc[trn_idx]

        y_val_temp = y_data.loc[val_idx] 

        clf_tree.named_steps['regressor'].fit(X_train_temp,y_train_temp)

        feature_imp_temp = np.abs(clf_tree.named_steps['regressor'].feature_importances_)

        all_feature_imp.append(feature_imp_temp)

    all_feature_imp = np.array(all_feature_imp)

    feature_imp_mean = np.mean(all_feature_imp,axis=0)

    feature_imp_std = np.std(all_feature_imp,axis=0)

    return feature_imp_mean, feature_imp_std
# Similarly, we are going to do 5-fold for permutation feature importance. 

# Both training and testing datasets will be evaluated. 

# Their results will be combined.

# train_imp_mean shape - [n_features, n_repeats]

def perm_imp(train_data,y_data, num_repeats,seed):

    all_perm_train_imp = np.zeros((train_data.shape[1],num_repeats))

    all_perm_test_imp = np.zeros((train_data.shape[1],num_repeats))

    kFold = KFold(n_splits=5, random_state=0, shuffle=True)

    clf_tree = get_tree(0)

    for fold, (trn_idx, val_idx) in enumerate(kFold.split(train_data.index)):

        X_train_temp = train_data.loc[trn_idx]

        X_test_temp = train_data.loc[val_idx]

        y_train_temp = y_data.loc[trn_idx]

        y_test_temp = y_data.loc[val_idx]

        

        clf_tree.fit(X_train_temp, y_train_temp)

        feature_imp_temp = permutation_importance(clf_tree, X_train_temp, y_train_temp, n_repeats=num_repeats,

                                         random_state=seed, n_jobs=-1)

        feature_imp_temp = np.abs(feature_imp_temp['importances'])

        all_perm_train_imp += feature_imp_temp

        

        feature_imp_temp = permutation_importance(clf_tree, X_test_temp, y_test_temp, n_repeats=num_repeats,

                                         random_state=seed, n_jobs=-1)

        feature_imp_temp = np.abs(feature_imp_temp['importances'])

        all_perm_test_imp += feature_imp_temp

        print(fold)

        

    all_perm_train_imp = all_perm_train_imp / 5

    all_perm_test_imp = all_perm_test_imp / 5

    train_imp_mean = np.mean(all_perm_train_imp,axis=1)

    test_imp_mean = np.mean(all_perm_test_imp,axis=1)

    train_imp_std = np.std(all_perm_train_imp,axis=1)

    test_imp_std = np.std(all_perm_test_imp,axis=1)

    return train_imp_mean, test_imp_mean, train_imp_std, test_imp_std
# RFECV is going to do a kFold evaluation on all the features and rank them

# Eliminate features at a step 0.05*n_featurees

def feature_RFE(train_data,y_data,seed):

    support = []

    n_features = []

    scores = []

    rfecv = RFECV(estimator=model, step=0.05, cv=KFold(5,random_state=seed,shuffle=True),

                  scoring=scoring)

    rfecv.fit(train_data, y_train)

    return rfecv
# Helper function for geting unique feature 

# names as some features are one-hot encoded

def filter_features(features):

    features_unique = []

    for c in features:

        features_unique.append(c.split("_")[0])

        feat_unique = list(Counter(features_unique).keys())

        feat_counts = list(Counter(features_unique).values())

    return feat_unique, feat_counts
# Helper function for finding features ranked higher than noise feature

def find_useful_features(feat_imp,noise_loc,columns):

    idx = feat_imp.argsort()

    idx_noise = np.where(idx == noise_loc)[0][0]

    idx = idx[idx_noise+1:]

    features = np.array(columns)[idx]

    return features
# We need to get column names and check our noise feature

clf_tree = get_tree(0)

train_data = clf_tree.named_steps['preprocessor'].fit_transform(X_train_noise,y_train)

nominal_columns =  list(ce_nom.fit_transform(X_train_noise[cat_nominal_features]).columns)

ordinal_columns = list(ce_ord.fit_transform(X_train_noise[cat_ordinal_features]).columns)

column_names = nominal_columns + ordinal_columns + num_features_noise

print('Training data shape: ',train_data.shape)

print('Number of columns: ',len(column_names))



# We make a plot about this noise feature

plt.hist(train_data[:,228])

plt.title('Noise Feature')
#Now apply MDI_imp, and find features ranked higher than noise features

mdi_imp_mean, mdi_imp_std = MDI_imp(train_data,y_train)

features_mdi = find_useful_features(mdi_imp_mean,228,column_names)

print("Number of MDI features ranked above noisy feature: %d" % len(features_mdi))
# We make a plot for visualization purpose for top 20 features

top_idx = mdi_imp_mean.argsort()[-20:]

y_ticks = np.arange(0, 20)

fig, ax = plt.subplots()

ax.barh(y_ticks, mdi_imp_mean[top_idx])

ax.set_yticklabels(np.array(column_names)[top_idx])

ax.set_yticks(y_ticks)

ax.set_title("Xgboost MDI")

fig.tight_layout()

plt.show()
# Some of these features are one-hot-encoded as you can see based on the figure above 

# we can get their unique feature names by applying the filter function

features_mdi_filter, features_mdi_count = filter_features(features_mdi)

print('Num mdi features after filter: ',len(features_mdi_filter))

print(features_mdi_filter)
# Now we do permutation feature importance

perm_imp_mean_train1,perm_imp_mean_test1,perm_imp_std_train1,perm_imp_std_test1 = perm_imp(X_train_noise,y_train,5,1)
# The last feature in this list is noise feature, i.e. position 79

perm_scores_train = dict(zip(list(X_train_noise.columns),list(perm_imp_mean_train1)))

perm_scores_test = dict(zip(list(X_train_noise.columns),list(perm_imp_mean_test1)))

print(len(perm_scores_train))

print(len(perm_scores_test))
## We make some visualization

top_idx = perm_imp_mean_test1.argsort()[-40:]

y_ticks = np.arange(0, 40)

fig, ax = plt.subplots(figsize=(7,7))

ax.barh(y_ticks, perm_imp_mean_test1[top_idx])

ax.set_yticklabels(np.array(X_test.columns)[top_idx])

ax.set_yticks(y_ticks)

ax.set_title("Xgboost Perm")

fig.tight_layout()

plt.show()
# We are going to combine features found based on

# both training and testing datasets

perm_features_train = list(find_useful_features(perm_imp_mean_train1,79,X_train_noise.columns))

perm_features_test = list(find_useful_features(perm_imp_mean_test1,79,X_train_noise.columns))

perm_features = list(set(perm_features_train) | set(perm_features_test))

perm_features_scores_train = dict((k, perm_scores_train[k]) for k in perm_features)

perm_features_scores_test = dict((k, perm_scores_test[k]) for k in perm_features)

perm_features_scores = {}

for key in perm_features_scores_train.keys():

    perm_features_scores[key] = (perm_features_scores_train[key] + perm_features_scores_test[key]) / 2 

perm_features = np.array(perm_features)[np.array(list(perm_features_scores.values())).argsort()]

perm_scores = np.array(list(perm_features_scores.values()))[np.array(list(perm_features_scores.values())).argsort()]
print(perm_features)

print(perm_scores)
# Now we run RFE

rfecv = feature_RFE(train_data,y_train,0)
print("Optimal RFE number of features : %d" % rfecv.n_features_)

print("Feature Ranking: ")

print(rfecv.ranking_)

features_rfe = np.array(column_names)[rfecv.support_]

# Check whether noise feature is in optimal features or not 

print('Is noise in optimal features: ','noise' in features_rfe)

features_rfe = [c for c in features_rfe if c not in ['noise']]

# We filter these rfe features to get their unique names

features_rfe, features_rfe_count = filter_features(features_rfe)

print('Number of RFE features after filter: ',len(features_rfe))
# We make some plot for visiualization

plt.figure()

plt.xlabel("Steps")

plt.ylabel("CV score")

plt.scatter(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)

plt.show()