import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from sklearn.model_selection import train_test_split
# keep only relevant imports based on the regresssion or classification goals

from sklearn.metrics import mean_squared_error

from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV
# common classifiers

#from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

#from sklearn.linear_model import LogisticRegression, SGDClassifier

from sklearn.svm import SVC, LinearSVC
import xgboost as xgb

import lightgbm as lgbm
# common regresssors

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, SGDRegressor, BayesianRidge

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor

from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
# skip future warnings and display enough columns for wide data sets

import warnings

warnings.simplefilter(action='ignore') #, category=FutureWarning)

pd.set_option('display.max_columns', 100)
df = pd.read_csv('../input/train.csv', index_col='Id' )

df.head()
df.shape
df.info()
df.dtypes.value_counts()
df.select_dtypes('object').apply(pd.Series.nunique, axis = 0)
def missing_values_table(df):

        """Function to calculate missing values by column# Funct // credits Will Koehrsen"""

    

        # Total missing values

        mis_val = df.isnull().sum()

        

        # Percentage of missing values

        mis_val_percent = 100 * df.isnull().sum() / len(df)

        

        # Make a table with the results

        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)

        

        # Rename the columns

        mis_val_table_ren_columns = mis_val_table.rename(

        columns = {0 : 'Missing Values', 1 : '% of Total Values'})

        

        # Sort the table by percentage of missing descending

        mis_val_table_ren_columns = mis_val_table_ren_columns[

            mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(

        '% of Total Values', ascending=False).round(1)

        

        # Print some summary information

        print ("Le jeu de données a " + str(df.shape[1]) + " colonnes.\n"      

            "Il y a " + str(mis_val_table_ren_columns.shape[0]) +

              " colonnes avec des valeurs manquantes.")

        

        # Return the dataframe with missing information

        return mis_val_table_ren_columns
missing_values = missing_values_table(df)

missing_values.head(10)
cat_feat = list(df.select_dtypes('object').columns)

num_feat = list(df.select_dtypes(exclude='object').columns)
plt.figure(figsize=(8, 4))

sns.kdeplot(df.SalePrice, shade=True)

plt.show()
plt.figure(figsize=(10, 6))

for zone in list(df.MSZoning.unique()):

    sns.distplot(df[df.MSZoning==zone].SalePrice, label=zone, hist=False)

plt.show()
plt.figure(figsize=(10, 6))

for ms_sub_class in list(df.MSSubClass.unique()):

    sns.distplot(df[df.MSSubClass==ms_sub_class].SalePrice, label=ms_sub_class, hist=False)

plt.show()

plt.figure(figsize=(10, 6))

for qual in list(df.OverallQual.unique()):

    sns.distplot(df[df.OverallQual==qual].SalePrice, label=qual, hist=False)

plt.show()
df.SalePrice.describe()
corr = df.corr()

corr



# Generate a mask for the upper triangle

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(14, 12))



# Generate a custom diverging colormap

cmap = sns.diverging_palette(220, 10, as_cmap=True)



# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5}) #annot=True
top_feature = corr.index[abs(corr['SalePrice']>0.5)]

plt.subplots(figsize=(12, 8))

top_corr = df[top_feature].corr()

sns.heatmap(top_corr, annot=True)

plt.show()
sns.barplot(df.OverallQual, df.SalePrice)
plt.figure(figsize=(18, 8))

sns.boxplot(x=df.OverallQual, y=df.SalePrice)
col = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'TotRmsAbvGrd', 'YearBuilt']

sns.pairplot(df[col], height=3, kind='reg')
print("Most postively correlated features with the target")

corr = df.corr()

corr.sort_values(['SalePrice'], ascending=False, inplace=True)

corr.SalePrice
df.duplicated().sum()
from sklearn.preprocessing import LabelEncoder
def prepare_data(dataframe):



    dataframe = dataframe.drop(columns=['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu'])



    cat_feat = list(dataframe.select_dtypes('object').columns)

    num_feat = list(dataframe.select_dtypes(exclude='object').columns)



    dataframe[num_feat] = dataframe[num_feat].fillna(dataframe[num_feat].median())

    dataframe[cat_feat] = dataframe[cat_feat].fillna("Not communicated")

    

    for c in cat_feat:

        lbl = LabelEncoder() 

        lbl.fit(list(dataframe[c].values)) 

        dataframe[c] = lbl.transform(list(dataframe[c].values))

    

    return dataframe
df = prepare_data(df)
#df[num_feat] = MinMaxScaler().fit_transform(df[num_feat])
y = df['SalePrice']

X = df.drop(columns=['SalePrice'])

X.shape, y.shape
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

X_train.shape, X_test.shape, y_train.shape, y_test.shape
rnd_reg = RandomForestRegressor(n_estimators=500, n_jobs=-1)

rnd_reg.fit(X, y)



feature_importances = pd.DataFrame(rnd_reg.feature_importances_, index = X.columns,

                                    columns=['importance']).sort_values('importance', ascending=False)
feature_importances[:10]
plt.figure(figsize=(10, 14))

sns.barplot(x="importance", y=feature_importances.index, data=feature_importances)

plt.show()
# f1_score binary by default

def get_rmse(reg, model_name):

    """Print the score for the model passed in argument and retrun scores for the train/test sets"""

    

    y_train_pred, y_pred = reg.predict(X_train), reg.predict(X_test)

    rmse_train, rmse_test = np.sqrt(mean_squared_error(y_train, y_train_pred)), np.sqrt(mean_squared_error(y_test, y_pred))

    print(model_name, f'\t - RMSE on Training  = {rmse_train:.0f} / RMSE on Test = {rmse_test:.0f}')

    

    return rmse_train, rmse_test
model_list = [

    LinearRegression(), Lasso(), SVR(),

    RandomForestRegressor(), GradientBoostingRegressor(), Ridge(), ElasticNet(), LinearSVC(),

    BayesianRidge(), ExtraTreesRegressor()

             ]
model_names = [str(m)[:str(m).index('(')] for m in model_list]

rmse_train, rmse_test = [], []
model_names
for model, name in zip(model_list, model_names):

    model.fit(X_train, y_train)

    sc_train, sc_test = get_rmse(model, name)

    rmse_train.append(sc_train)

    rmse_test.append(sc_test)
df_score = pd.DataFrame({'model_names' : model_names,

                         'rmse_train' : rmse_train,

                         'rmse_test' : rmse_test})

ax = df_score.plot.barh(y=['rmse_test', 'rmse_train'], x='model_names')
svm_reg = Pipeline([

    ("scaler", StandardScaler()),

    ("svm_regresssor", LinearSVC())

])

svm_reg.fit(X_train, y_train)

_, _ = get_rmse(svm_reg, "svr_rbf")
svr_rbf = SVR(kernel = 'rbf')

svr_rbf.fit(X_train, y_train)

_, _ = get_rmse(svr_rbf, "svr_rbf")
svm_reg = Pipeline([

    ("scaler", StandardScaler()),

    ("svm_regresssor", SVR())

])

svm_reg.fit(X_train, y_train)

_, _ = get_rmse(svm_reg, "svr_rbf")



svm_reg = Pipeline([

    ("scaler", StandardScaler()),

    ("svm_regresssor", SVR(kernel="poly"))

])

svm_reg.fit(X_train, y_train)

_, _ = get_rmse(svm_reg, "svr_poly")



sgd_reg = Pipeline([

    ("scaler", StandardScaler()),

    ("sgd_regresssor", SGDRegressor())

])

sgd_reg.fit(X_train, y_train)

_, _ = get_rmse(sgd_reg, "sgd_reg") 
xgb_reg = xgb.XGBRegressor()

xgb_reg.fit(X_train, y_train)

_, _ = get_rmse(xgb_reg, "xgb_reg")
from sklearn.model_selection import GridSearchCV





rf = RandomForestRegressor()

param_grid = { 

    'n_estimators': [80, 100, 120],

    'max_features': [14, 15, 16, 17],

    'max_depth' : [14, 16, 18]

}





rfc_cv = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1)

rfc_cv.fit(X_train, y_train)

print(rfc_cv.best_params_)

_, _ = get_rmse(rfc_cv, "rfc_reg")
gb = GradientBoostingRegressor()

param_grid = { 

    'n_estimators': [100, 400],

    'max_features': [14, 15, 16, 17],

    'max_depth' : [1, 2, 8, 14, 18]

}





gb_cv = GridSearchCV(estimator=gb, param_grid=param_grid, cv=5, n_jobs=-1)

gb_cv.fit(X_train, y_train)

print(gb_cv.best_params_)

_, _ = get_rmse(gb_cv, "gb_cv")
xg = xgb.XGBRegressor()

param_grid = { 

    'n_estimators': [100, 400],

    'max_features': [10, 14, 16],

    'max_depth' : [1, 2, 8, 18]

}





xg_cv = GridSearchCV(estimator=xg, param_grid=param_grid, cv=5, n_jobs=-1)

xg_cv.fit(X_train, y_train)

print(xg_cv.best_params_)

_, _ = get_rmse(xg_cv, "xg_cv")
df_test = pd.read_csv('../input/test.csv', index_col='Id' )

df_test.head()
df_test = prepare_data(df_test)

df_test.shape
rfc_sub, gb_sub, xg_sub = rfc_cv.predict(df_test), gb_cv.predict(df_test), xg_cv.predict(df_test)
sub = pd.DataFrame()

sub['Id'] = df_test.index

sub['SalePrice'] = np.mean([rfc_sub, gb_sub, xg_sub], axis=0) / 3

sub.to_csv('submission.csv',index=False)