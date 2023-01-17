
import numpy as np # linear algebra
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt

# Comment this if the data visualisations doesn't work on your side
%matplotlib inline

plt.style.use('bmh')
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge, Lasso
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
#data
test=pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
train=pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')
sample=pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/sample_submission.csv')
train.head()
train.info()
print(train['SalePrice'].describe())
plt.figure(figsize=(9, 8))
sns.distplot(train['SalePrice'], color='g', bins=100, hist_kws={'alpha': 0.4});
df_num = train.select_dtypes(include = ['float64', 'int64'])

df_num_corr = df_num.corr()['SalePrice'][:-1] # -1 because the latest row is SalePrice
golden_features_list = df_num_corr[abs(df_num_corr) > 0.5].sort_values(ascending=False)
print("There are  {} strong correlated values with SalePrice:\n{}".format(len(golden_features_list), golden_features_list))
for i in range(0, len(df_num.columns), 5):
    sns.pairplot(data=df_num,
                x_vars=df_num.columns[i:i+5],
                y_vars=['SalePrice'])
corr = df_num.drop('SalePrice', axis=1).corr() # 
plt.figure(figsize=(12, 10))

sns.heatmap(corr[(corr >= 0.5) | (corr <= -0.4)], 
            cmap='jet', vmax=1.0, vmin=-1.0, linewidths=0.1,
            annot=True, annot_kws={"size": 8}, square=True);
quantitative_features_list = ['LotFrontage', 'LotArea', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'TotalBsmtSF', '1stFlrSF',
    '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath',
    'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 
    'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal', 'SalePrice']
df_quantitative_values = train[quantitative_features_list]
df_quantitative_values.head()
import operator

individual_features_df = []
for i in range(0, len(df_num.columns) - 1): # -1 because the last column is SalePrice
    tmpDf = df_num[[df_num.columns[i], 'SalePrice']]
    tmpDf = tmpDf[tmpDf[df_num.columns[i]] != 0]
    individual_features_df.append(tmpDf)

all_correlations = {feature.columns[0]: feature.corr()['SalePrice'][0] for feature in individual_features_df}
all_correlations = sorted(all_correlations.items(), key=operator.itemgetter(1))

golden_features_list = [key for key, value in all_correlations if abs(value) >= 0.5]
print("There are {} strongly correlated values with SalePrice:\n{}".format(len(golden_features_list), golden_features_list))
features_to_analyse = [x for x in quantitative_features_list if x in golden_features_list]
features_to_analyse.append('SalePrice')
categorical_features = [a for a in quantitative_features_list[:-1] + train.columns.tolist() if (a not in quantitative_features_list[:-1]) or (a not in train.columns.tolist())]
df_categ = train[categorical_features]
#fetching categorical variables
df_categ
df_not_num = df_categ.select_dtypes(include = ['O'])
print('There is {} non numerical features including:\n{}'.format(len(df_not_num.columns), df_not_num.columns.tolist()))
X_train=train.drop(columns=['SalePrice'])
Y_train=train[['SalePrice']]
num_feat=X_train.select_dtypes(include='number').columns.to_list()
cat_feat=X_train.select_dtypes(exclude='number').columns.to_list()
!pip install impyute
from impyute.imputation.cs import mice

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
num_pipe=Pipeline([
    ('imputer',SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

cat_pipe=Pipeline([
    ('imputer',SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])
ct=ColumnTransformer(remainder='drop',
                    transformers=[
                        ('numeric', num_pipe, num_feat),
                        ('categorical', cat_pipe, cat_feat)
                    ])
model=Pipeline([
    ('transformer',ct),
    ('poly',PolynomialFeatures(2)),
    ('predictor', Lasso())
])
model.fit(X_train, Y_train)

print(model.score(X_train, Y_train))

def submission(test, model):
    y_pred=model.predict(test)
    sample['SalePrice']=y_pred
    date=pd.datetime.now().strftime(format='%d_%m_%Y_%H-%M_')
    sample.to_csv(f'/kaggle/working/{date}result.csv',index=False)
submission(test,model)
