! pip install impyute
import numpy as np
import pandas as pd
import os
from impyute.imputation.cs import mice
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import RandomForestRegressor
from mlxtend.feature_selection import SequentialFeatureSelector
from sklearn.model_selection import GridSearchCV
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error

import warnings
warnings.filterwarnings("ignore")

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
train = pd.read_csv('../input/prohack-hackathon/train.csv')
test = pd.read_csv('../input/prohack-hackathon/test.csv')
submission = pd.read_csv('../input/prohack-hackathon/sample_submit.csv')
train.head()
train.info()
# Imputation runs for a very long time, so we comment out the code below and upload the final result

# train_to_mice = train.drop('galaxy', axis=1)

# imputed_training = mice(train_to_mice.values)
# train_mice = pd.DataFrame(data=imputed_training, columns=train_to_mice.columns.tolist())
train_mice = pd.read_csv('../input/prohack-mice-imputed/df_mice.csv')
# Return the column with galaxies

train_mice['galaxy'] = train['galaxy']
# Let's make the order of the columns as in the original frame

new_list = ['galactic year',
 'galaxy',
 'existence expectancy index',
 'existence expectancy at birth',
 'Gross income per capita',
 'Income Index',
 'Expected years of education (galactic years)',
 'Mean years of education (galactic years)',
 'Intergalactic Development Index (IDI)',
 'Education Index',
 'Intergalactic Development Index (IDI), Rank',
 'Population using at least basic drinking-water services (%)',
 'Population using at least basic sanitation services (%)',
 'Gross capital formation (% of GGP)',
 'Population, total (millions)',
 'Population, urban (%)',
 'Mortality rate, under-five (per 1,000 live births)',
 'Mortality rate, infant (per 1,000 live births)',
 'Old age dependency ratio (old age (65 and older) per 100 creatures (ages 15-64))',
 'Population, ages 15–64 (millions)',
 'Population, ages 65 and older (millions)',
 'Life expectancy at birth, male (galactic years)',
 'Life expectancy at birth, female (galactic years)',
 'Population, under age 5 (millions)',
 'Young age (0-14) dependency ratio (per 100 creatures ages 15-64)',
 'Adolescent birth rate (births per 1,000 female creatures ages 15-19)',
 'Total unemployment rate (female to male ratio)',
 'Vulnerable employment (% of total employment)',
 'Unemployment, total (% of labour force)',
 'Employment in agriculture (% of total employment)',
 'Labour force participation rate (% ages 15 and older)',
 'Labour force participation rate (% ages 15 and older), female',
 'Employment in services (% of total employment)',
 'Labour force participation rate (% ages 15 and older), male',
 'Employment to population ratio (% ages 15 and older)',
 'Jungle area (% of total land area)',
 'Share of employment in nonagriculture, female (% of total employment in nonagriculture)',
 'Youth unemployment rate (female to male ratio)',
 'Unemployment, youth (% ages 15–24)',
 'Mortality rate, female grown up (per 1,000 people)',
 'Mortality rate, male grown up (per 1,000 people)',
 'Infants lacking immunization, red hot disease (% of one-galactic year-olds)',
 'Infants lacking immunization, Combination Vaccine (% of one-galactic year-olds)',
 'Gross galactic product (GGP) per capita',
 'Gross galactic product (GGP), total',
 'Outer Galaxies direct investment, net inflows (% of GGP)',
 'Exports and imports (% of GGP)',
 'Share of seats in senate (% held by female)',
 'Natural resource depletion',
 'Mean years of education, female (galactic years)',
 'Mean years of education, male (galactic years)',
 'Expected years of education, female (galactic years)',
 'Expected years of education, male (galactic years)',
 'Maternal mortality ratio (deaths per 100,000 live births)',
 'Renewable energy consumption (% of total final energy consumption)',
 'Estimated gross galactic income per capita, male',
 'Estimated gross galactic income per capita, female',
 'Rural population with access to electricity (%)',
 'Domestic credit provided by financial sector (% of GGP)',
 'Population with at least some secondary education, female (% ages 25 and older)',
 'Population with at least some secondary education, male (% ages 25 and older)',
 'Gross fixed capital formation (% of GGP)',
 'Remittances, inflows (% of GGP)',
 'Population with at least some secondary education (% ages 25 and older)',
 'Intergalactic inbound tourists (thousands)',
 'Gross enrolment ratio, primary (% of primary under-age population)',
 'Respiratory disease incidence (per 100,000 people)',
 'Interstellar phone subscriptions (per 100 people)',
 'Interstellar Data Net users, total (% of population)',
 'Current health expenditure (% of GGP)',
 'Intergalactic Development Index (IDI), female',
 'Intergalactic Development Index (IDI), male',
 'Gender Development Index (GDI)',
 'Intergalactic Development Index (IDI), female, Rank',
 'Intergalactic Development Index (IDI), male, Rank',
 'Adjusted net savings ',
 'Creature Immunodeficiency Disease prevalence, adult (% ages 15-49), total',
 'Private galaxy capital flows (% of GGP)',
 'Gender Inequality Index (GII)',
 'y']
train_mice = train_mice[new_list]
train_mice.head()
train['galaxy'].value_counts()
test['galaxy'].value_counts()
train_mice['galaxy'].nunique()
test['galaxy'].nunique()
# The names of some galaxies are not in the test dataset

set(train_mice['galaxy'].tolist()) ^ set(test['galaxy'].tolist())
# Delete galaxies that are not in the test dataset

df_train = train_mice.loc[~train_mice['galaxy'].isin(['Andromeda XII',
 'Andromeda XIX[60]',
 'Andromeda XVIII[60]',
 'Andromeda XXII[57]',
 'Andromeda XXIV',
 'Hercules Dwarf',
 'NGC 5253',
 'Triangulum Galaxy (M33)',
 'Tucana Dwarf'])]
# Let's check

set(df_train['galaxy'].tolist()) ^ set(test['galaxy'].tolist())
# Create datasets with one-hot encoded galaxy names

train_dummies = pd.get_dummies(df_train['galaxy'])
test_dummies = pd.get_dummies(test['galaxy'])
# Let's see how much data is missing in the training dataset

total = train.isnull().sum().sort_values(ascending=False)
percent = (train.isnull().sum()/train.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head()
missing_data.loc[missing_data['Percent'] > 0.7]
list_of_missing_data = missing_data.loc[missing_data['Percent'] > 0.7].index.tolist()
# delete columns with missing data more than 70%
# delete from the MICE IMPUTED set!

df_train = df_train.drop(list_of_missing_data, axis=1)
df_train.shape
impute_data = df_train.drop(['galaxy'], axis=1)
num_colums = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
numerical_columns = list(impute_data.select_dtypes(include=num_colums).columns)
impute_data = impute_data[numerical_columns]
impute_data.shape

train_features, test_features, train_labels, test_labels = train_test_split(
    impute_data.drop(labels=['y'], axis=1),
    impute_data['y'],
    test_size=0.2,
    random_state=41)

correlated_features = set()
correlation_matrix = impute_data.corr()
for i in range(len(correlation_matrix .columns)):
    for j in range(i):
        if abs(correlation_matrix.iloc[i, j]) > 0.9:
            colname = correlation_matrix.columns[i]
            correlated_features.add(colname)


train_features.drop(labels=correlated_features, axis=1, inplace=True)
test_features.drop(labels=correlated_features, axis=1, inplace=True)

train_features.shape, test_features.shape
# We will use the Random Forest regressor to find the most optimal parameters. Set the number of features to 5 (5 gave the best result)

# from sklearn.ensemble import RandomForestRegressor
# from mlxtend.feature_selection import SequentialFeatureSelector

feature_selector = SequentialFeatureSelector(RandomForestRegressor(n_jobs=-1),
           k_features=5,
           forward=True,
           verbose=2,
           scoring='neg_mean_squared_error',
           cv=4)
features = feature_selector.fit(np.array(train_features.fillna(0)), train_labels)
filtered_features= train_features.columns[list(features.k_feature_idx_)]
filtered_features
best_features = filtered_features.tolist()
best_features
best_features.append('y')
# We get the final data set for the predictive model

df = impute_data[best_features]
X = df.drop(['y'], axis=1)
y = df['y']
# Similar to the training dataset, I used MICE imputation. Download the finished data set

test_mice = pd.read_csv('../input/prohack-mice-imputed/df_mice_test.csv')
test_mice.head()
# list of columns left for prediction:

data_columns = X.columns.tolist()
df_test = test_mice[data_columns]
df_test.head()
# Join datasets with encoded galaxy names

X_joined_dummies = X.join(train_dummies)
df_test_joined_dummies = df_test.join(test_dummies)
X = X_joined_dummies

df_test_pred = df_test_joined_dummies
# rename columns with galaxies from alphabetic names to numbers

galaxy_rename_list = train_dummies.columns.tolist()

i = 1
for name in galaxy_rename_list:
    X.rename(columns={name: i}, inplace=True)
    df_test_pred.rename(columns={name: i}, inplace=True)
    i = i + 1
# I used the code below to pick up the parameters


# import warnings
# warnings.filterwarnings("ignore")

# from sklearn.model_selection import train_test_split
# from sklearn.model_selection import GridSearchCV
# from catboost import CatBoostRegressor

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# model = CatBoostRegressor()
# parameters = {'depth'         : [3, 4, 5],
#               'learning_rate' : [0.05, 0.1, 0.2],
#               'iterations'    : [8000, 12000],
#               'subsample'     : [0.3, 0.5, 1]
#             }
# grid = GridSearchCV(estimator=model, param_grid = parameters, cv = 2, n_jobs=-1)
# grid.fit(X_train, y_train)    

# # Results from Grid Search
# print("\n========================================================")
# print(" Results from Grid Search " )
# print("========================================================")    
    
# print("\n The best estimator across ALL searched params:\n",
#     grid.best_estimator_)
    
# print("\n The best score across ALL searched params:\n",
#     grid.best_score_)
    
# print("\n The best parameters across ALL searched params:\n",
#     grid.best_params_)
    
# print("\n ========================================================")
# ========================================================
#  Results from Grid Search 
# ========================================================

#  The best estimator across ALL searched params:
#  <catboost.core.CatBoostRegressor object at 0x000000000C956348>

#  The best score across ALL searched params:
#  0.9482559923204557

#  The best parameters across ALL searched params:
#  {'depth': 3, 'iterations': 12000, 'learning_rate': 0.1, 'subsample': 0.3}

#  ========================================================
model = CatBoostRegressor(iterations=12000,
                          learning_rate=0.1,
                          subsample=0.3,
                          depth=3)
# Fit model

model.fit(X_train,y_train)
# Get predictions

preds = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, preds))
print("RMSE: %f" % (rmse))
# Predict 'y' on test case

y_pred_test = model.predict(df_test_pred)
