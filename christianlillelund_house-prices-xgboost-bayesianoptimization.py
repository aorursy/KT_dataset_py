# Install GPyOpt

!pip install GPyOpt
import pandas as pd

import os

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

import warnings



warnings.filterwarnings('ignore')

pd.reset_option('^display.', silent=True)



X_train = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")

X_test = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")



num_train = len(X_train)



y_train = X_train.SalePrice

X_train.drop(['SalePrice'], axis=1, inplace=True)



print("Total training samples:", len(X_train), "\n")



# Merge train and test to simplify pre-processing

df = pd.concat([X_train, X_test], ignore_index=True)



X_train.head()
# Show descriptive statistics of training set

X_train.describe()
# Show how many values are non-null for each feature

X_train.info()
# Print a random house as a sample

sample_index = 25

print(X_train.iloc[sample_index])
# Show the column types we are dealing with

df.dtypes.value_counts()

cat_columns = df.select_dtypes('object').columns

num_columns = [i for i in list(df.columns) if i not in list(df.select_dtypes('object').columns)]

print(len(df.columns)-len(df.select_dtypes('object').columns),'numerical columns:')

print(num_columns, '\n')

print(len(df.select_dtypes('object').columns),'categorical columns:')

print(list(cat_columns))
# Plot numerical feature variables

fig = plt.figure(figsize=(17,22))



num_data = df[num_columns]

num_data = num_data.drop(['Id'], axis=1)



for i in range(len(num_data.columns)):

    fig.add_subplot(9,4,i+1)

    sns.distplot(num_data.iloc[:,i].dropna(), hist=False, kde_kws={'bw':0.1}, color='mediumslateblue')

    plt.xlabel(num_data.columns[i])

plt.tight_layout()

plt.show()
# Plot categorial feature variables

cat_data = df[cat_columns]

cat_data_cols = cat_data.columns

cat_data_cols_length = (len(cat_data_cols)/5)+1



fg, ax = plt.subplots(figsize=(25, 35))

fg.subplots_adjust(hspace=0.5)

for i, col in enumerate(cat_data):

    fg.add_subplot(cat_data_cols_length, 5, i+1)

    sns.countplot(cat_data[col], palette='rocket')

    plt.xlabel(col)

    plt.xticks(rotation=90)



plt.show()
# Show what the SalePrice variable looks like

df_train = pd.concat([df[:num_train], y_train], axis=1)



sns.set_style("white")

sns.set_color_codes(palette='deep')

f, ax = plt.subplots(figsize=(8, 7))

sns.distplot(df_train.SalePrice, color="b");

ax.xaxis.grid(False)

ax.set(ylabel="Frequency")

ax.set(xlabel="SalePrice")

ax.set(title="SalePrice distribution")

sns.despine(trim=True, left=True)

plt.show()
# Show how SalePrice and OverallQual are related

data_seg1 = pd.concat([df_train['SalePrice'], df_train['OverallQual']], axis=1)



plt.subplots(figsize=(9,5))

plt.figure(1); plt.title("SalePrice vs Overall Quality")

sns.boxplot(x='OverallQual', y='SalePrice', data=data_seg1, color="mediumslateblue")



plt.subplots(figsize=(9,5))

plt.figure(2); plt.title("SalePrice vs Overall Quality")

sns.lineplot(x='OverallQual', y='SalePrice', data=data_seg1, color="mediumslateblue")
# Rename odd-named columns

df = df.rename(columns={"1stFlrSF": "FirstFlrSF",

                        "2ndFlrSF": "SecondFlrSF",

                       "3SsnPorch": "ThirdSsnPorch"})



# Remove whitespace in MSZoning

df.MSZoning[~df.MSZoning.isnull()] = df.MSZoning[~df.MSZoning.isnull()].map(lambda x: x[:2])



# Remove dots from BldgType

df.BldgType[~df.BldgType.isnull()] = df.BldgType[~df.BldgType.isnull()].map(lambda x: x.replace('.', ''))



# Remove ampersand from RoofStyle

df.RoofStyle[~df.RoofStyle.isnull()] = df.RoofStyle[~df.RoofStyle.isnull()].map(lambda x: x.replace('&', ''))



# Remove whitespace in Exterior1st, Exterior2nd

df.Exterior1st[~df.Exterior1st.isnull()] = df.Exterior1st[~df.Exterior1st.isnull()].map(lambda x: x.replace(' ', ''))

df.Exterior2nd[~df.Exterior2nd.isnull()] = df.Exterior2nd[~df.Exterior2nd.isnull()].map(lambda x: x.replace(' ', ''))
# Visualize missing values

sns.set_style("white")

f, ax = plt.subplots(figsize=(8, 7))

sns.set_color_codes(palette='deep')

missing = round(df.isnull().mean()*100,2)

missing = missing[missing > 0]

missing.sort_values(inplace=True)

missing.plot.bar(color="b")



# Tweak the visual presentation

ax.xaxis.grid(False)

ax.set(ylabel="Percent of missing values")

ax.set(xlabel="Features")

ax.set(title="Percent missing data by feature")

sns.despine(trim=True, left=True)
# Find columns with more than 1000 NaN's and drop them (see above)

columns = [col for col in df.columns if df[col].isnull().sum() > 1000]

df = df.drop(columns, axis=1)



# Manually encode LotFrontage using median of neighborhood

df.LotFrontage = df.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))



# No garage values means no year, area or cars

for col in ['GarageYrBlt', 'GarageArea', 'GarageCars']:

    df[col] = df[col].fillna(0)

    

# No garage info means you don't have one

for col in ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']:

    df[col] = df[col].fillna('None')



# Fill no basement

for col in ['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']:

    df[col] = df[col].fillna('None')



# Fill remaining cat and num cols with None and 0

cat_columns = df.select_dtypes('object').columns

num_columns = [i for i in list(df.columns) if i not in cat_columns]

df.update(df[cat_columns].fillna('None'))

df.update(df[num_columns].fillna(0))



# Make year variables relative to year sold

year_cols = ['YearBuilt','YearRemodAdd','GarageYrBlt']

for col in year_cols:

    df[col] = df['YrSold'] - df[col]
# Check for missing values 

print(df.isnull().values.any())
# Plot a heatmap of the all the features and their correlation to SalePrice

# Thanks to https://www.kaggle.com/shubhamksingh/top-3-stacking-blending-in-depth-eda



df_train = pd.concat([df[:num_train], y_train], axis=1)

corrmat = df_train.corr()

plt.subplots(figsize=(17,17))

plt.title("Correlation Matrix")

sns.heatmap(corrmat, vmax=0.9, square=True, cmap="Oranges", annot=True, fmt='.1f', linewidth='.1')
# Plot the most correlated variables as a matrix

imp_ftr = corrmat['SalePrice'].sort_values(ascending=False).head(11).to_frame()

plt.subplots(figsize=(5,8))

plt.title('SalePrice Correlation Matrix')

sns.heatmap(imp_ftr, vmax=0.9, annot=True, fmt='.2f', cmap="Oranges", linewidth='.1')
plt.subplots(figsize=(15, 15))

sns.heatmap(corrmat>0.8, annot=True, square=True, cmap="Oranges", linewidth='.1')
# Drop correlated columns and plot the result



df = df.drop(['FirstFlrSF', 'GarageCars', 'TotRmsAbvGrd'], axis=1)



df_train = pd.concat([df[:num_train], y_train], axis=1)

corrmat = df_train.corr()

plt.subplots(figsize=(15, 15))

sns.heatmap(corrmat>0.8, annot=True, square=True, cmap="Oranges", linewidth='.1')
# Find outliers using IsolationForest

from sklearn.ensemble import IsolationForest



num_outliers = 20

anomaly_dict = {}

num_columns = [i for i in list(df.columns) if i not in list(df.select_dtypes('object').columns)]

for feature in num_columns:

    model = IsolationForest(n_estimators=50, max_samples='auto', contamination=float(0.1), max_features=1.0)

    model.fit(df[[feature]])

    df['score'] = model.decision_function(df[[feature]])

    df['anomaly'] = model.predict(df[[feature]])

    anomaly = df.loc[df['anomaly']==-1]

    feature_anomaly = anomaly[[feature] + ['score']][:num_outliers]

    feature_anomaly['min'] = df[feature].min()

    feature_anomaly['max'] = df[feature].max()

    feature_anomaly['mean'] = df[feature].mean()

    anomaly_dict[f'{feature}'] = feature_anomaly
print(anomaly_dict['GarageArea'].sort_values(by='score'), '\n')
# Clip data to fix outliers

df.MSSubClass = df.MSSubClass.clip(20, 180)

df.LotFrontage = df.LotFrontage.clip(20, 200)

df.LotArea = df.LotArea.clip(1300,50000)

df.OverallQual = df.OverallQual.clip(2,10)

df.OverallCond = df.OverallCond.clip(2,9)

df.YearBuilt = df.YearBuilt.clip(1880,2010)

df.MasVnrArea = df.MasVnrArea.clip(0,1000)

df.BsmtFinSF1 = df.BsmtFinSF1.clip(0,2300)

df.BsmtFinSF2 = df.BsmtFinSF2.clip(0,1100)

df.BsmtUnfSF = df.BsmtUnfSF.clip(0,2000)

df.TotalBsmtSF = df.TotalBsmtSF.clip(0,3000)

df.SecondFlrSF = df.SecondFlrSF.clip(0,1800)

df.LowQualFinSF = df.LowQualFinSF.clip(0,700)

df.GrLivArea = df.GrLivArea.clip(334,4000)

df.BsmtFullBath = df.BsmtFullBath.clip(0,2)

df.BsmtHalfBath = df.BsmtHalfBath.clip(0,2)

df.FullBath = df.FullBath.clip(0,3)

df.HalfBath = df.HalfBath.clip(0,2)

df.BedroomAbvGr = df.BedroomAbvGr.clip(0,6)

df.KitchenAbvGr = df.KitchenAbvGr.clip(0,2)

df.Fireplaces = df.Fireplaces.clip(0,3)

df.GarageYrBlt = df.GarageYrBlt.clip(1900,2207)

df.GarageArea = df.GarageArea.clip(0,1400)

df.WoodDeckSF = df.WoodDeckSF.clip(0,700)

df.OpenPorchSF = df.OpenPorchSF.clip(0,550)

df.EnclosedPorch = df.EnclosedPorch.clip(0,560)

df.ThirdSsnPorch = df.ThirdSsnPorch.clip(0,300)

df.ScreenPorch = df.ScreenPorch.clip(0,400)

df.PoolArea = df.PoolArea.clip(0,400)

df.MiscVal = df.MiscVal.clip(0,3000)

# SalePrice is skewed, so apply the log function to its value to correct it



plt.figure(figsize=[11,4])

plt.subplot(1,2,1)

plt.title('Sale Price')

plt.hist(y_train,bins=40)



plt.subplot(1,2,2)

plt.title('Log of Sale Price')

log = y_train.apply(lambda x: np.log(x))

plt.hist(log,bins=40)

y_train = log
# Visualize numerical columns to identified skewed variables

df_train = pd.concat([df[:num_train], y_train], axis=1)



columns = list(df_train.select_dtypes(exclude='object').columns)

plt.figure(figsize=[16,30])

for i in range(len(columns)):

    try:

        ax = plt.subplot(10,4,i+1)

        plt.scatter(df_train[columns[i]],df_train.SalePrice,alpha=0.15)

        plt.title(columns[i])

        box = ax.get_position()

        box.y1 = box.y1 - 0.01 

        ax.set_position(box)

    except:

        pass

plt.show()
# Transform numerical variables to be more symmetrical based on above plots

df_train = pd.concat([df[:num_train], y_train], axis=1)

plt.figure(figsize=[11,4])

plt.subplot(1,2,1)

plt.title('Before')

plt.scatter(df_train.LotArea, df_train.SalePrice,alpha=0.25)

plt.xlabel('LotArea')



columns_to_log = ['LotFrontage','LotArea','BsmtFinSF1','BsmtFinSF2','MasVnrArea', 

               'BsmtUnfSF','TotalBsmtSF','GrLivArea','WoodDeckSF','OpenPorchSF']



for col in columns_to_log:

    df[col] = df[col].apply(lambda x: np.log(x) if x !=0 else x)

    

# Show transformation example with LotArea

df_train = pd.concat([df[:num_train], y_train], axis=1)

plt.figure(figsize=[16,4])

plt.subplot(1,2,2)

plt.title('After')

plt.scatter(df_train.LotArea, df_train.SalePrice,alpha=0.25)

plt.xlabel('LotArea')
df.head()
# Apply one-hot encoding on categorial variables

from sklearn.preprocessing import OneHotEncoder



def encode_df(df, object_cols):

    ohe = OneHotEncoder(handle_unknown='ignore', sparse=False)

    df_enc = pd.DataFrame(ohe.fit_transform(df[object_cols]))

    df_enc.columns = ohe.get_feature_names(object_cols)

    df_enc.index = df.index

    return df_enc



# Use OH encoder to encode cat cols

df_enc = encode_df(df, cat_columns)

num_df = df.drop(cat_columns, axis=1)

df = pd.concat([num_df, df_enc], axis=1)



# Split train and test set

X_train = df.iloc[:num_train,:]

X_test = df.iloc[num_train:,:]



# Apply RobustScaler scaling

from sklearn.preprocessing import RobustScaler

rs = RobustScaler()

X_train = rs.fit_transform(X_train)

X_test = rs.transform(X_test)
# Train an XGBRegressor using BayesianOptimization to find best params

# Credit: http://krasserm.github.io/2018/03/21/bayesian-optimization/

import GPy

import GPyOpt

from xgboost import XGBRegressor

from GPyOpt.methods import BayesianOptimization

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score



xgb = XGBRegressor()

baseline = cross_val_score(xgb, X_train, y_train, scoring='neg_mean_squared_error').mean()



search_space = [{'name': 'learning_rate', 'type': 'continuous', 'domain': (0, 1)},

                {'name': 'gamma', 'type': 'continuous', 'domain': (0, 5)},

                {'name': 'max_depth', 'type': 'discrete', 'domain': (1, 2, 3, 4, 5, 10)},

                {'name': 'n_estimators', 'type': 'discrete', 'domain': (5000, 6000, 7000)},

                {'name': 'min_child_weight', 'type': 'discrete', 'domain': (0, 10)}]



def cv_score(parameters):

    parameters = parameters[0]

    score = cross_val_score(

                XGBRegressor(learning_rate=parameters[0],

                              gamma=int(parameters[1]),

                              max_depth=int(parameters[2]),

                              n_estimators=int(parameters[3]),

                              min_child_weight = parameters[4]), 

                X_train, y_train, scoring='neg_mean_squared_error').mean()

    score = np.array(score)

    return score



optimizer = BayesianOptimization(f=cv_score, 

                                 domain=search_space,

                                 model_type='GP',

                                 acquisition_type ='EI',

                                 acquisition_jitter = 0.05,

                                 exact_feval=True, 

                                 maximize=True,

                                 verbosity=True,

                                 verbosity_model=True)



optimizer.run_optimization(max_iter=3, verbosity=True)
# Plot the convergence as a function of iterations

optimizer.plot_convergence()
# Plot the accumulated score of the optimizer for Y compared to baseline

y_bo = np.maximum.accumulate(-optimizer.Y).ravel()



print(f'Baseline neg. MSE = {baseline:.3f}')

print(f'Bayesian optimization neg. MSE = {y_bo[-1]:.3f}')



plt.plot(baseline, 'ro-', label='Baseline')

plt.plot(y_bo, 'bo-', label='Bayesian optimization')

plt.xlabel('Iteration')

plt.ylabel('Neg. MSE')

plt.title('Value of the best sampled CV score')

plt.legend()
# Put our results in a dataframe and show them

header_params = []

for param in search_space:

    header_params.append(param['name'])



df_results = pd.DataFrame(data=optimizer.X, columns=header_params)

df_results['error'] = optimizer.Y

df_results = df_results.sort_values(by=['error'])

df_results
# Train a model with val set using best found parameters

from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.3, random_state=0)

xgb = XGBRegressor(n_estimators=6000,

                   min_child_weight=10,

                   max_depth=1,

                   gamma=0.06,

                   learning_rate=0.01,

                   random_state=0)

xgb.fit(X_train, y_train, early_stopping_rounds=10, eval_metric="rmse",

       eval_set=[(X_train, y_train), (X_val, y_val)], verbose=False)

predictions = xgb.predict(X_test)
# Load sample submission

submission = pd.read_csv("../input/house-prices-advanced-regression-techniques/sample_submission.csv")



# Append XGBoost predictions

submission.iloc[:,1] = np.floor(np.expm1(predictions))



# Set quantile to leave out eventual outliers

q1 = submission['SalePrice'].quantile(0.0045)

q2 = submission['SalePrice'].quantile(0.99)



# Calibrate for eventual outliers by quantile

submission['SalePrice'] = submission['SalePrice'].apply(lambda x: x if x > q1 else x*0.77)

submission['SalePrice'] = submission['SalePrice'].apply(lambda x: x if x < q2 else x*1.1)



submission.to_csv("submission_regression.csv", index=False)
