# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train_values = pd.read_csv('/kaggle/input/predicting-poverty/train_values_wJZrCmI.csv')

train_labels = pd.read_csv('/kaggle/input/predicting-poverty/train_labels.csv')

test = pd.read_csv('/kaggle/input/predicting-poverty/test_values.csv')
print(train_values.shape)

print(train_labels.shape)
train_values.head().transpose()
train_labels.head()
train_values.dtypes
train_values.describe().transpose()
df = train_values.merge(train_labels, on='row_id')
df.shape
print(df.isnull().sum())
df.drop('bank_interest_rate', axis = 1, inplace = True)

df.drop('mm_interest_rate', axis = 1, inplace = True)

df.drop('mfi_interest_rate', axis = 1, inplace = True)

df.drop('other_fsp_interest_rate', axis = 1, inplace = True)
df['education_level'].fillna(4, inplace=True)

df['share_hh_income_provided'].fillna(0, inplace=True)
# def replace_boolean(data):

#     for col in data:

#         data[col].replace(True, 1, inplace=True)

#         data[col].replace(False, 0, inplace=True)

        

# replace_boolean(df)

# df.dtypes
df.describe().transpose()
list(df.columns.values)
def create_age_group(data):

    age_conditions = [

    (data['age'] < 30 ),

    (data['age'] >= 30) & (data['age'] < 45),

    (data['age'] >= 45) & (data['age'] < 60),

    (data['age'] >= 60)

    ]

    age_choices = ['Under 30', '30 to 44', '45 to 59', '60 or Over']

    data['age_group'] = np.select(age_conditions, age_choices)

    #return data['age_group']



create_age_group(df)
def count_unique(df, cols):

    for col in cols:

        print('\n' + 'For column ' + col)

        print(df[col].value_counts())



cat_cols = ['age_group','country','is_urban','female','married','religion','relationship_to_hh_head',

 'education_level','literacy','can_add','can_divide','can_calc_percents','can_calc_compounding',

 'employed_last_year','employment_category_last_year','employment_type_last_year',

 'income_ag_livestock_last_year','income_friends_family_last_year','income_government_last_year',

 'income_own_business_last_year','income_private_sector_last_year','income_public_sector_last_year',

 'borrowing_recency','formal_savings','informal_savings','cash_property_savings',

 'has_insurance','has_investment','borrowed_for_emergency_last_year','borrowed_for_daily_expenses_last_year',

 'borrowed_for_home_or_biz_last_year','phone_technology','can_call','can_text','can_use_internet',

 'can_make_transaction','phone_ownership','advanced_phone_use','reg_bank_acct',

 'reg_mm_acct','reg_formal_nbfi_account','financially_included','active_bank_user',

 'active_mm_user','active_formal_nbfi_user','active_informal_nbfi_user','nonreg_active_mm_user', 'share_hh_income_provided', 

'num_times_borrowed_last_year','num_shocks_last_year','num_formal_institutions_last_year',

            'num_informal_institutions_last_year']



count_unique(df, cat_cols)

df.shape
import matplotlib.pyplot as plt

import seaborn as sns
def plot_violin(df, cols, col_y, title):

    for col in cols:

        sns.set(style="whitegrid")

        sns.set_palette("Set1", n_colors=7, desat=.7)

        sns.violinplot(col, col_y, data=df)

        plt.xlabel(col) # Set text for the x axis

        plt.ylabel(col_y)# Set text for y axis

        plt.title(title + ' by ' + col)

        plt.show()

        

plot_violin(df, cat_cols, 'poverty_probability', 'PPI')
num_cols = ['age', 'avg_shock_strength_last_year', 

            'num_financial_activities_last_year', 

            'poverty_probability'] 



def plot_density_hist(df, cols, bins = 10, hist = False):

    for col in cols:

        sns.set(style="whitegrid", palette='Blues_r')

        sns.distplot(df[col], bins = bins, rug=True, hist = hist)

        plt.title('Histogram of ' + col) # Give the plot a main title

        plt.xlabel(col) # Set text for the x axis

        plt.ylabel('Frequency')# Set text for y axis

        plt.show()

        

plot_density_hist(df, num_cols, bins = 20, hist = True)
num_cols = ['age', 'avg_shock_strength_last_year', 

            'num_financial_activities_last_year', 'poverty_probability'] 



sns.set(style="whitegrid", palette='Blues_r')

sns.pairplot(df[num_cols])
num_corrs = df[num_cols].corr()

f,ax = plt.subplots(figsize=(5, 5))

sns.heatmap(num_corrs, annot=True, square=True, linewidths=.1, fmt= '.2f',ax=ax, 

           cmap="RdBu")

plt.show()
#Feature engineering

#Aggregate categorical features



#The aggregation of categorical features was performed to reduce the number of categories. 

#For discrete variables, rare values are combined to form a range of values. 

#For categorical variables, rare values are combined with common values that share a more similar distribution in PPI.

religion_categories = {'N':'N_Q', 'O':'O_P',

                       'P':'O_P', 'Q':'N_Q','X':'X'}

df['religion'] = [religion_categories[x] for x in df['religion']]

print(df['religion'].value_counts())



#num_shocks_last_year 4_5

num_shocks_last_year_categories = {0:'0', 1:'1', 2:'2',

                       3:'3', 4:'4_5', 5:'4_5'}

df['num_shocks_last_year'] = [num_shocks_last_year_categories[x] for x in df['num_shocks_last_year']]

print(df['num_shocks_last_year'].value_counts())



#num_formal_institutions_last_year 3_or_over

num_formal_institutions_last_year_categories = {0:'0', 1:'1', 2:'2',

                       3:'3_4_5_6', 4:'3_4_5_6', 5:'3_4_5_6', 6:'3_4_5_6'}

df['num_formal_institutions_last_year'] = [num_formal_institutions_last_year_categories[x] for x in df['num_formal_institutions_last_year']]

print(df['num_formal_institutions_last_year'].value_counts())



#num_informal_institutions_last_year 2_or_over

num_informal_institutions_last_year_categories = {0:'0', 1:'1', 2:'2_3_4',

                       3:'2_3_4', 4:'2_3_4'}

df['num_informal_institutions_last_year'] = [num_informal_institutions_last_year_categories[x] for x in df['num_informal_institutions_last_year']]

print(df['num_informal_institutions_last_year'].value_counts())



relationship_to_hh_head_categories = {'Other':'Other', 'Spouse':'Spouse',

                                      'Head':'Head',

                                      'Son/Daughter':'Son/Daughter',

                                      'Sister/Brother':'Sister/Brother',

                                      'Father/Mother': 'Father/Mother',

                                      'Unknown':'Other'}

df['relationship_to_hh_head'] = [relationship_to_hh_head_categories[x] for x in df['relationship_to_hh_head']]

print(df['relationship_to_hh_head'].value_counts())
Labels = np.array(df['poverty_probability'])
from sklearn import preprocessing
def encode_string(cat_features):

    ## Now, apply one hot encoding

    ohe = preprocessing.OneHotEncoder(categories='auto')

    encoded = ohe.fit_transform(cat_features.values.reshape(-1,1)).toarray()

    pdfn = ohe.get_feature_names()

    print(pdfn)

    return encoded



features_cat_cols = ['country','is_urban','female','married','religion','relationship_to_hh_head',

 'education_level','literacy','can_add','can_divide','can_calc_percents','can_calc_compounding',

 'employed_last_year','employment_category_last_year','employment_type_last_year',

 'income_ag_livestock_last_year','income_friends_family_last_year','income_government_last_year',

 'income_own_business_last_year','income_private_sector_last_year','income_public_sector_last_year',

 'borrowing_recency','formal_savings','informal_savings','cash_property_savings',

 'has_insurance','has_investment','borrowed_for_emergency_last_year','borrowed_for_daily_expenses_last_year',

 'borrowed_for_home_or_biz_last_year','phone_technology','can_call','can_text','can_use_internet',

 'can_make_transaction','phone_ownership','advanced_phone_use','reg_bank_acct',

 'reg_mm_acct','reg_formal_nbfi_account','financially_included','active_bank_user',

 'active_mm_user','active_formal_nbfi_user','active_informal_nbfi_user','nonreg_active_mm_user', 'share_hh_income_provided', 

'num_times_borrowed_last_year','num_shocks_last_year','num_formal_institutions_last_year',

            'num_informal_institutions_last_year']



Features = encode_string(df['age_group'])

for col in features_cat_cols:

    temp = encode_string(df[col])

    Features = np.concatenate([Features, temp], axis = 1)

    

print(Features.shape)

print(Features[:2, :])
features_num_cols = ['avg_shock_strength_last_year',

                     'num_financial_activities_last_year']

Features = np.concatenate([Features, np.array(df[features_num_cols])], axis = 1)

print(Features.shape)

print(Features[:2, :])
from sklearn import feature_selection as fs
# Features selection

print(Features.shape)



## Define the variance threhold and fit the threshold to the feature array.

sel = fs.VarianceThreshold(threshold=(.95 * (1 - .95)))

Features_reduced = sel.fit_transform(Features)

print(sel.get_support())



## Print the support and shape for the transformed features

print(Features_reduced.shape)
#https://www.kaggle.com/johnnyyiu/poverty-prediction-from-visualization-to-stacking
Labels = Labels.reshape(Labels.shape[0],)
import numpy.random as nr

import sklearn.model_selection as ms
nr.seed(562)

feature_folds = ms.KFold(n_splits=10, shuffle = True)
from sklearn import linear_model
lin_mod_l2 = linear_model.Ridge()

nr.seed(265)

selector = fs.RFECV(estimator = lin_mod_l2, cv = feature_folds,scoring = 'r2')
selector = selector.fit(Features_reduced, Labels)

print(selector.support_)

print(selector.ranking_)
Features_reduced = selector.transform(Features_reduced)

print(Features_reduced.shape)
nr.seed(265)

indx = range(Features.shape[0])

indx = ms.train_test_split(indx, test_size = 0.2)

x_train = Features_reduced[indx[0],:]

y_train = np.ravel(Labels[indx[0]])

x_test = Features_reduced[indx[1],:]

y_test = np.ravel(Labels[indx[1]])
scaler = preprocessing.StandardScaler().fit(x_train[:,104:])

x_train[:,104:] = scaler.transform(x_train[:,104:])

x_test[:,104:] = scaler.transform(x_test[:,104:])

print(x_train[:2,])
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
import lightgbm as lgb

import xgboost as xgb
GBoost = GradientBoostingRegressor()

model_xgb = xgb.XGBRegressor()

model_lgb = lgb.LGBMRegressor(objective='regression', num_leaves = 32,learning_rate=0.01)
nr.seed(265)

inside = ms.KFold(n_splits=5, shuffle = True)

nr.seed(562)

outside = ms.KFold(n_splits=5, shuffle = True)
nr.seed(2652)

param_grid = {'n_estimators': [2000, 3000]}
gsearch = ms.GridSearchCV(estimator = model_lgb, param_grid = param_grid, 

                      cv = inside, # Use the inside folds

                      scoring = 'r2',

                      return_train_score = True)
gsearch.fit(Features_reduced, Labels)

gsearch.best_params_, gsearch.best_score_
GBoost = GradientBoostingRegressor(n_estimators= 2000, learning_rate=0.01,

                                   max_depth=5, max_features='sqrt',

                                   min_samples_leaf=7, min_samples_split=15, 

                                   loss='ls', random_state = 1)



model_xgb = xgb.XGBRegressor(max_depth = 5, min_child_weight = 0, gamma = 0, 

                           subsample = 0.8, colsample_bytree = 0.8, 

                           scale_pos_weight = 1, reg_lambda = 1,

                           learning_rate =0.01, n_estimators=2000, 

                           objective = 'reg:squarederror', seed = 14)



model_lgb = lgb.LGBMRegressor(objective='regression', num_leaves = 32,

                              learning_rate=0.01, n_estimators=2100, 

                              bagging_fraction = 0.8,

                              bagging_freq = 5, feature_fraction = 0.4,

                              min_data_in_leaf = 5,  

                              feature_fraction_seed=3, bagging_seed=2)
n_folds = 5



def r2_cv(model):

    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(x_train)

    r2 = cross_val_score(model, x_train, y_train, scoring="r2", cv = kf)

    return(r2)
from sklearn.model_selection import KFold, cross_val_score, train_test_split

from sklearn.metrics import r2_score
score = r2_cv(GBoost)

print("Gradient Boosting score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
score = r2_cv(model_xgb)

print("Xgboost score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
score = r2_cv(model_lgb)

print("LGBM score: {:.4f} ({:.4f})\n" .format(score.mean(), score.std()))