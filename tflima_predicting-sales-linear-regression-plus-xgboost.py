import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import missingno as msno
from sklearn.model_selection import train_test_split, cross_val_predict, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn import linear_model
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, make_scorer
import shap
import warnings
warnings.filterwarnings('ignore')
!pip install xgboost==1.0.0
import xgboost
class CreateFeatures(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    def fit(self, X, y=None):
        return self
        
    def transform(self, X, y=None):
        X['week_of_year'] = pd.to_datetime(X.Date).dt.weekofyear
        X['year'] = pd.to_datetime(X.Date).dt.year

        X = X.merge(df_4holidays[['year', 'current_week', 'week_holiday']]
                                       , how='left', left_on=['year', 'week_of_year'],right_on=['year', 'current_week'])

        X = X.merge(df_4holidays[['year', 'last_week', 'last_week_holiday']]
                                       , how='left', left_on=['year', 'week_of_year'],right_on=['year', 'last_week'])

        X = X.merge(df_4holidays[['year', 'next_week', 'next_week_holiday']]
                                       , how='left', left_on=['year', 'week_of_year'],right_on=['year', 'next_week'])
        
        X['prop_to_buy'] =  ((X.Temperature * (100 - X.Unemployment) ) / (X.CPI * X.Fuel_Price ))
        X['move_cost'] = X.CPI / X.Fuel_Price
        X['revenue_potential'] = (100 * X.Unemployment) * X.Size
        return X

class FeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        X.Store = X.Store.astype(str)
        X.Dept = X.Dept.astype(str)
        X.Type = X.Type.astype(str)
        X.week_holiday = X.week_holiday.astype(str)
        X.last_week_holiday = X.last_week_holiday.astype(str)
        X.next_week_holiday = X.next_week_holiday.astype(str)
        X = pd.get_dummies(X)
        return X[FEATURES_TO_MODEL]
    
class FeatureSelector1(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        X.Store = X.Store.astype(str)
        X.Dept = X.Dept.astype(str)
        X.Type = X.Type.astype(str)
        X.week_holiday = X.week_holiday.astype(str)
        X.last_week_holiday = X.last_week_holiday.astype(str)
        X.next_week_holiday = X.next_week_holiday.astype(str)
        X = pd.get_dummies(X)
        return X


class FillNaValues(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        X.MarkDown1 = X.MarkDown1.fillna(X.MarkDown1.dropna().median())
        X.next_week_holiday.fillna('None', inplace=True)
        X.last_week_holiday.fillna('None', inplace=True)
        X.week_holiday.fillna('None', inplace=True)
        X.Unemployment.fillna(X.Unemployment.dropna().median(), inplace=True)
        X.IsHoliday.fillna(0, inplace=True)
        X.prop_to_buy.fillna(X.prop_to_buy.dropna().median(), inplace=True)
        X.move_cost.fillna(X.move_cost.dropna().median(), inplace=True)
        X.revenue_potential.fillna(X.revenue_potential.dropna().median(), inplace=True)
        return X

    
def train_linear_regression(X_train, y_train, X_val, y_val):
    lr = linear_model.ElasticNet(random_state=42)
    lr.fit(X_train, y_train)
    print('R^2 = {}'.format(r2_score(y_val, lr.predict(X_val))))
    print('MAE = {}'.format(mean_absolute_error(y_val, lr.predict(X_val)) ))
    print('RMSE = {}'.format(mean_squared_error(y_val, lr.predict(X_val), squared=False) ))
    # cross_val_predict returns an array of the same size as `y` where each entry
    # is a prediction obtained by cross validation:
    predicted = cross_val_predict(lr, X_train, y_train, cv=5)
    fig, ax = plt.subplots()
    ax.scatter(y_train, predicted, edgecolors=(0, 0, 0))
    ax.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'k--', lw=4)
    ax.set_xlabel('Measured')
    ax.set_ylabel('Predicted')
    plt.show()
    return lr  

def train_xgboost_regressor(X_train, y_train, X_val, y_val):
    xgb_model = xgboost.XGBRegressor(random_state=42, n_jobs=-1)
    xgb_model.fit(X_train, y_train)
    print('R^2 = {}'.format(r2_score(y_val, xgb_model.predict(X_val))))
    print('MAE = {}'.format(mean_absolute_error(y_val, xgb_model.predict(X_val)) ))
    print('RMSE = {}'.format(mean_squared_error(y_val, xgb_model.predict(X_val), squared=False) ))
    predicted = cross_val_predict(xgb_model, X_train, y_train, cv=5)
    fig, ax = plt.subplots()
    ax.scatter(y_train, predicted, edgecolors=(0, 0, 0))
    ax.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'k--', lw=4)
    ax.set_xlabel('Measured')
    ax.set_ylabel('Predicted')
    plt.show()
    return xgb_model

def ensemble_xgb_elastic_net(model_1, model_2, X_train_1, X_train_2, y_train, X_val_1, X_val_2, y_val):
    train_pred1 = model_1.predict(X_train_1)
    train_pred2 = model_2.predict(X_train_2)
    val_pred1 = model_1.predict(X_val_1)
    val_pred2 = model_2.predict(X_val_2)
    df_train = pd.DataFrame({'feat_model_1': train_pred1, 'feat_model_2': train_pred2})
    df_val = pd.DataFrame({'feat_model_1': val_pred1, 'feat_model_2': val_pred2})
    model_lr = linear_model.LinearRegression()
    model_lr.fit(df_train, y_train)
    print('R^2 = {}'.format(r2_score(y_val, model_lr.predict(df_val))))
    print('MAE = {}'.format(mean_absolute_error(y_val, model_lr.predict(df_val)) ))
    print('RMSE = {}'.format(mean_squared_error(y_val, model_lr.predict(df_val), squared=False) ))
    return model_lr

df_stores = pd.read_csv('/kaggle/input/walmart-recruiting-store-sales-forecasting/stores.csv', sep=',')
df_features = pd.read_csv('/kaggle/input/walmart-recruiting-store-sales-forecasting/features.csv.zip', sep=',')
df_train = pd.read_csv('/kaggle/input/walmart-recruiting-store-sales-forecasting/train.csv.zip', sep=',')
df_teste = pd.read_csv('/kaggle/input/walmart-recruiting-store-sales-forecasting/test.csv.zip', sep=',')
df_stores.shape
df_stores.info()
df_stores.head()
df_stores.Store.nunique()
df_stores['Type'].value_counts()
ax = sns.barplot(x=sorted(df_stores.Type.unique()),y=df_stores['Type'].value_counts(),
                 palette="Blues_d")
plt.xlabel('Store types')
plt.ylabel("Quantity")
plt.title('Quantity analysis of store types')
sns.despine()
plt.show();

sns.distplot(df_stores['Size']);
sns.despine();
df_stores['Size'].plot.hist(density=True);
plt.xlabel('Store Size');
plt.title('Analysing the Size distribution ');
plt.show();
for t in df_stores['Type'].unique():
    df_stores.loc[df_stores.Type==t, 'Size'].plot.hist(density=False, label=t, alpha=0.8);
    plt.xlabel('Store Size');
plt.legend(title='Type of store');
plt.title('Analysing the Size distribution by Type of Store');
plt.show();
for t in df_stores['Type'].unique():
    print('Analysing the Size distribution for Store Type {}'.format(t));
    display(df_stores.loc[df_stores.Type==t, 'Size'].describe())
    df_stores.loc[df_stores.Type==t, 'Size'].plot.hist(density=True, label=t, alpha=0.8);
    plt.xlabel('Store Size');
    plt.legend(title='Type of store');
    plt.title('');
    plt.show();
df_features.shape
df_features.info()
df_features.head()
df_features.groupby(['Store']).agg({'Date':['count', 'min','max']})
df_features.Temperature.plot.hist(density=True, alpha=0.85);
plt.xlabel('Temperature');
plt.title('Temperature Distribution for the whole dataset');
plt.show()
df_features.groupby(['Store']).agg({'Temperature': ['min','mean','max', 'std']})
df_features.loc[df_features.Store==7, 'Temperature'].plot.hist(label='Store 7', density=True, alpha=.6);
df_features.loc[df_features.Store==33, 'Temperature'].plot.hist(label='Store 33', density=True, alpha=.6);
plt.xlabel('Temperature');
plt.legend();
plt.title('Comparing the temperature of the hottest store to the coldest store');
plt.show();
df_features.Fuel_Price.plot.hist();
plt.xlabel('Fuel Price');
plt.title('Fuel Price Distribution for the whole dataset');
plt.show()
for i in df_features.Store.unique():
    df_features.loc[df_features.Store==i, 'Fuel_Price'].plot();
    plt.title('Fuel Price through time for Store {}'.format(i));
    plt.ylabel('Fuel Price');
    plt.xticks([]);
    plt.xlabel('Time {} to {}'.format(min(df_features.Date),max(df_features.Date) ));
    plt.show();
df_features.sample()
for mark in [i for i in df_features.columns if 'Mark' in i]:
    df_features[mark].plot.hist(density=False);
    plt.title('Distribuition of {}'.format(mark));
    plt.show();  
df_features.groupby(['Store']).agg({'MarkDown1':['min','mean', 'max']})
for st in df_features.Store.unique():
    df_features.loc[df_features.Store==st, 'MarkDown1'].plot.hist(density=True, alpha=0.8, label=st);
    plt.xlabel('MarkDown1');
    plt.title('MarkDown1 distribution for Store {}'.format(st));
    plt.show();
df_features.CPI.plot.hist();
for st in df_features.Store.unique():
    df_features.loc[df_features.Store==st, 'CPI'].plot.hist(density=True, alpha=0.8, label=st);
    plt.xlabel('CPI value');
    plt.title('CPI distribution for Store {}'.format(st));
    plt.show();
df_features.groupby('Store').agg({'CPI': ['min', 'mean', 'max']})
df_features.sample()
df_features.Unemployment.plot.hist();
df_features.groupby('Store').agg({'Unemployment':['min', 'mean', 'max']})
df_features.IsHoliday.dtype
df_features.IsHoliday = df_features.IsHoliday.astype(int)
df_features.groupby(['Store']).agg({'IsHoliday':sum})
df_features.sample()
ax = sns.scatterplot(x="CPI", y="Fuel_Price",hue='Store', data=df_features)
ax = sns.scatterplot(x="CPI", y="Unemployment",hue='Store', data=df_features)
df_train.head()
df_teste.head()
df_stores.head()
df_features.head()
(df_train.Store.dtype == df_stores.Store.dtype, df_teste.Store.dtype == df_stores.Store.dtype )
df_temp_train = df_train.merge(df_stores, how='left', on='Store')
df_temp_test = df_teste.merge(df_stores, how='left', on='Store')
df_temp_test.sample()
df_temp_train.sample()
(df_temp_train.Store.dtype == df_features.Store.dtype, df_temp_train.Date.dtype == df_features.Date.dtype)
(df_temp_test.Store.dtype == df_features.Store.dtype, df_temp_test.Date.dtype == df_features.Date.dtype)
df_train_full = df_temp_train.merge(df_features, how='left', on=['Store', 'Date'])
df_test_full = df_temp_test.merge(df_features, how='left', on=['Store', 'Date'])
df_train_full.shape
df_test_full.shape
df_train_full.head()
df_train_full.IsHoliday_x.astype(int).sum() == df_train_full.IsHoliday_y.sum()
df_train_full.drop('IsHoliday_x', axis=1,inplace=True)
df_train_full.rename(columns={'IsHoliday_y':'IsHoliday'}, inplace=True)
df_test_full.head()
df_test_full.IsHoliday_x.astype(int).sum() == df_test_full.IsHoliday_y.sum()
df_test_full.drop('IsHoliday_x', axis=1,inplace=True)
df_test_full.rename(columns={'IsHoliday_y':'IsHoliday'}, inplace=True)
dict_lgt_hlds ={'Super_Bowl': ['12-Feb-10', '11-Feb-11', '10-Feb-12', '8-Feb-13']
               ,'Labor_Day': ['10-Sep-10', '9-Sep-11', '7-Sep-12', '6-Sep-13']
                ,'Thanksgiving': ['26-Nov-10', '25-Nov-11', '23-Nov-12', '29-Nov-13']
                ,'Christmas': ['31-Dec-10', '30-Dec-11', '28-Dec-12', '27-Dec-13']
               }
lista = []
for hol in dict_lgt_hlds.keys():
    for dt in dict_lgt_hlds[hol]:
        lista.append([hol, pd.to_datetime(dt).year, pd.to_datetime(dt).week])
df_4holidays = pd.DataFrame(lista, columns=['week_holiday','year', 'current_week'])
df_4holidays['last_week'] =df_4holidays['current_week'] +1
df_4holidays['next_week'] =df_4holidays['current_week'] - 1
df_4holidays['last_week_holiday'] = df_4holidays['week_holiday']
df_4holidays['next_week_holiday'] = df_4holidays['week_holiday']
df_4holidays
df_train_full['week_of_year'] = pd.to_datetime(df_train_full.Date).dt.weekofyear
df_train_full['year'] = pd.to_datetime(df_train_full.Date).dt.year
sns.countplot(data=df_train_full, x='week_of_year');
plt.xticks(rotation=45);
sns.countplot(data=df_train_full, x='year');
df_train_full_4h = df_train_full.merge(df_4holidays[['year', 'current_week', 'week_holiday']]
                                       , how='left', left_on=['year', 'week_of_year'],right_on=['year', 'current_week'])
    
df_train_full_4h = df_train_full_4h.merge(df_4holidays[['year', 'last_week', 'last_week_holiday']]
                                       , how='left', left_on=['year', 'week_of_year'],right_on=['year', 'last_week'])
df_train_full_4h = df_train_full_4h.merge(df_4holidays[['year', 'next_week', 'next_week_holiday']]
                                       , how='left', left_on=['year', 'week_of_year'],right_on=['year', 'next_week'])
pd.options.display.max_columns=None
df_train_full_4h.head()
data = df_train_full_4h.groupby('Store').agg({'Weekly_Sales':'mean', 'Type':'max', 'Size':'mean'}).reset_index()
data.groupby('Type').agg({'Weekly_Sales':'mean'})
ax = sns.scatterplot(x="Store", y="Weekly_Sales",hue='Type', data=data)
data['sales_per_size'] = data['Weekly_Sales'] /data['Size']
data.groupby('Type').agg({'sales_per_size':'mean'})
ax = sns.scatterplot(x="Store", y="sales_per_size",hue='Type', data=data)
data = df_train_full_4h.groupby(['Type','Dept']).agg({'Weekly_Sales':'mean'}).reset_index()
ax = sns.scatterplot(x="Dept", y="Weekly_Sales",hue='Type' ,data=data)
data = df_train_full_4h.groupby(['Temperature']).agg({'Weekly_Sales':'mean'}).reset_index()
ax = sns.scatterplot(x="Temperature", y="Weekly_Sales" ,data=data)
data['temp_bins'] = pd.cut(data.Temperature, bins=10).astype(str)
ax = sns.lineplot(x="temp_bins", y="Weekly_Sales" ,data=data.groupby('temp_bins').agg({'Weekly_Sales':'mean'}).reset_index())
plt.xticks(rotation=45);
ax = sns.scatterplot(x="Temperature", y="Weekly_Sales",hue='Type' ,data=df_train_full_4h)
ax = sns.scatterplot(x="Fuel_Price", y="Weekly_Sales",hue='Type' ,data=df_train_full_4h)
ax = sns.scatterplot(x="MarkDown1", y="Weekly_Sales",hue='Type' ,data=df_train_full_4h)
ax = sns.scatterplot(x="CPI", y="Weekly_Sales",hue='Type' ,data=df_train_full_4h)
ax = sns.scatterplot(x="Unemployment", y="Weekly_Sales",hue='Type' ,data=df_train_full_4h)
df_train_full_4h.groupby('IsHoliday').agg({'Weekly_Sales':'mean'})
ax = sns.lineplot(x="IsHoliday", y="Weekly_Sales", markers=True ,data=df_train_full_4h.groupby('IsHoliday').agg({'Weekly_Sales':'mean'}).reset_index())
plt.xticks(rotation=45);
df_train_full_4h.groupby('week_holiday').agg({'Weekly_Sales':'mean'})
ax = sns.lineplot(x="week_holiday", y="Weekly_Sales", markers=True ,data=df_train_full_4h.groupby('week_holiday').agg({'Weekly_Sales':'mean'}).reset_index())
plt.xticks(rotation=45);
df_train_full_4h.groupby('last_week_holiday').agg({'Weekly_Sales':'mean'})
ax = sns.lineplot(x="last_week_holiday", y="Weekly_Sales", markers=True ,data=df_train_full_4h.groupby('last_week_holiday').agg({'Weekly_Sales':'mean'}).reset_index())
plt.xticks(rotation=45);
df_train_full_4h.groupby('next_week_holiday').agg({'Weekly_Sales':'mean'})
ax = sns.lineplot(x="next_week_holiday", y="Weekly_Sales", markers=True ,data=df_train_full_4h.groupby('next_week_holiday').agg({'Weekly_Sales':'mean'}).reset_index())
plt.xticks(rotation=45);
df_train_full_4h.describe()
df_train_full_4h['prop_to_buy'] =  ((df_train_full_4h.Temperature * (100 - df_train_full_4h.Unemployment) ) / (df_train_full_4h.CPI * df_train_full_4h.Fuel_Price ))
ax = sns.scatterplot(x="prop_to_buy", y="Weekly_Sales", hue='Type' ,data=df_train_full_4h)
#plt.xticks(rotation=45);
g = sns.jointplot(x="prop_to_buy", y="Weekly_Sales" ,data=df_train_full_4h,
                  kind="reg", truncate=False,
                  #xlim=(0, 60), ylim=(0, 12),
                  color="m"
                  #, height=7
                 )
df_train_full_4h['move_cost'] = df_train_full_4h.CPI / df_train_full_4h.Fuel_Price
ax = sns.scatterplot(x="move_cost", y="Weekly_Sales", hue='Type' ,data=df_train_full_4h)
#plt.xticks(rotation=45);
g = sns.jointplot(x="move_cost", y="Weekly_Sales" ,data=df_train_full_4h,
                  kind="reg", truncate=False,
                  #xlim=(0, 60), ylim=(0, 12),
                  color="m"
                  #, height=7
                 )
df_train_full_4h['revenue_potential'] = (100 * df_train_full_4h.Unemployment) * df_train_full_4h.Size
ax = sns.scatterplot(x="revenue_potential", y="Weekly_Sales", hue='Type' ,data=df_train_full_4h)
g = sns.jointplot(x="revenue_potential", y="Weekly_Sales" ,data=df_train_full_4h,
                  kind="reg", truncate=False,
                  #xlim=(0, 60), ylim=(0, 12),
                  color="m"
                  #, height=7
                 )
msno.matrix(df_train_full_4h);
df_train_sel = df_train_full_4h.drop(['MarkDown2'
                                     ,'MarkDown3'
                                     ,'MarkDown4'
                                     ,'MarkDown5'
                                     ,'year'
                                     ,'Date'
                                     ,'current_week'
                                      ,'last_week'
                                      ,'next_week'
                                      ,'week_of_year'
                                     ], axis=1)
df_train_sel.MarkDown1 = df_train_sel.MarkDown1.fillna(df_train_sel.MarkDown1.dropna().median())
df_train_sel.next_week_holiday.fillna('None', inplace=True)
df_train_sel.last_week_holiday.fillna('None', inplace=True)
df_train_sel.week_holiday.fillna('None', inplace=True)
FET_TO_SCALER = [
    'Size'
    ,'Temperature'
    ,'Fuel_Price'
    ,'MarkDown1'
    ,'CPI'
    ,'Unemployment'
    ,'prop_to_buy'
    ,'move_cost'
    ,'revenue_potential'
]
scaler = StandardScaler()
df_train_sel[FET_TO_SCALER] = scaler.fit_transform(df_train_sel[FET_TO_SCALER])
df_train_sel.describe()
msno.matrix(df_train_sel);
df_train_sel.info()
df_train_sel.Store = df_train_sel.Store.astype(str)
df_train_sel.Dept = df_train_sel.Dept.astype(str)
df_train_sel.Type = df_train_sel.Type.astype(str)
df_train_sel.week_holiday = df_train_sel.week_holiday.astype(str)
df_train_sel.last_week_holiday = df_train_sel.last_week_holiday.astype(str)
df_train_sel.next_week_holiday = df_train_sel.next_week_holiday.astype(str)
df_train_dummies = pd.get_dummies(df_train_sel)
df_train_dummies.shape
df_train_dummies.sample(3)
list(df_train_dummies.columns)
GP1 = ['Size'
       ,'Temperature'
       ,'Fuel_Price'
       ,'MarkDown1'
       ,'CPI'
       ,'Unemployment'
       ,'IsHoliday'
       ,'prop_to_buy'
       ,'move_cost'
       ,'revenue_potential'
    
]
GP2 = [
    'week_holiday_Christmas'
    ,'week_holiday_Labor_Day'
    ,'week_holiday_None'
    ,'week_holiday_Super_Bowl'
    ,'week_holiday_Thanksgiving'
    ,'last_week_holiday_Labor_Day'
    ,'last_week_holiday_None'
    ,'last_week_holiday_Super_Bowl'
    ,'last_week_holiday_Thanksgiving'
    ,'next_week_holiday_Christmas'
    ,'next_week_holiday_Labor_Day'
    ,'next_week_holiday_None'
    ,'next_week_holiday_Super_Bowl'
    ,'next_week_holiday_Thanksgiving'
]

GP3 = [
    'Type_A'
    ,'Type_B'
    ,'Type_C'
]
f, ax = plt.subplots(figsize=(11, 6))
sns.heatmap(df_train_dummies[GP1].corr(), annot=True, linewidths=.5, ax=ax);
f, ax = plt.subplots(figsize=(15, 8))
sns.heatmap(df_train_dummies[GP2].corr(), annot=True, linewidths=.5, ax=ax);
f, ax = plt.subplots(figsize=(9, 6))
sns.heatmap(df_train_dummies[GP3].corr(), annot=True, linewidths=.5, ax=ax);
FEATURES_TO_MODEL = [

 'MarkDown1',
 'Unemployment',
 'IsHoliday',
 'prop_to_buy',
 'move_cost',
 'revenue_potential',
 
 'Store_1',
 'Store_10',
 'Store_11',
 'Store_12',
 'Store_13',
 'Store_14',
 'Store_15',
 'Store_16',
 'Store_17',
 'Store_18',
 'Store_19',
 'Store_2',
 'Store_20',
 'Store_21',
 'Store_22',
 'Store_23',
 'Store_24',
 'Store_25',
 'Store_26',
 'Store_27',
 'Store_28',
 'Store_29',
 'Store_3',
 'Store_30',
 'Store_31',
 'Store_32',
 'Store_33',
 'Store_34',
 'Store_35',
 'Store_36',
 'Store_37',
 'Store_38',
 'Store_39',
 'Store_4',
 'Store_40',
 'Store_41',
 'Store_42',
 'Store_43',
 'Store_44',
 'Store_45',
 'Store_5',
 'Store_6',
 'Store_7',
 'Store_8',
 'Store_9',
 
 'Dept_1',
 'Dept_10',
 'Dept_11',
 'Dept_12',
 'Dept_13',
 'Dept_14',
 'Dept_16',
 'Dept_17',
 'Dept_18',
 'Dept_19',
 'Dept_2',
 'Dept_20',
 'Dept_21',
 'Dept_22',
 'Dept_23',
 'Dept_24',
 'Dept_25',
 'Dept_26',
 'Dept_27',
 'Dept_28',
 'Dept_29',
 'Dept_3',
 'Dept_30',
 'Dept_31',
 'Dept_32',
 'Dept_33',
 'Dept_34',
 'Dept_35',
 'Dept_36',
 'Dept_37',
 'Dept_38',
 'Dept_39',
 'Dept_4',
 'Dept_40',
 'Dept_41',
 'Dept_42',
 'Dept_43',
 'Dept_44',
 'Dept_45',
 'Dept_46',
 'Dept_47',
 'Dept_48',
 'Dept_49',
 'Dept_5',
 'Dept_50',
 'Dept_51',
 'Dept_52',
 'Dept_54',
 'Dept_55',
 'Dept_56',
 'Dept_58',
 'Dept_59',
 'Dept_6',
 'Dept_60',
 'Dept_65',
 'Dept_67',
 'Dept_7',
 'Dept_71',
 'Dept_72',
 'Dept_74',
 'Dept_77',
 'Dept_78',
 'Dept_79',
 'Dept_8',
 'Dept_80',
 'Dept_81',
 'Dept_82',
 'Dept_83',
 'Dept_85',
 'Dept_87',
 'Dept_9',
 'Dept_90',
 'Dept_91',
 'Dept_92',
 'Dept_93',
 'Dept_94',
 'Dept_95',
 'Dept_96',
 'Dept_97',
 'Dept_98',
 'Dept_99',
 
 'Type_A',
 
 'Type_C',
 
 'week_holiday_Christmas',
 'week_holiday_Labor_Day',
 'week_holiday_None',
 'week_holiday_Super_Bowl',
 'week_holiday_Thanksgiving',
 'last_week_holiday_Labor_Day',
 'last_week_holiday_None',
 'last_week_holiday_Super_Bowl',
 'last_week_holiday_Thanksgiving',
 'next_week_holiday_Christmas',
 'next_week_holiday_Labor_Day',
 'next_week_holiday_None',
 'next_week_holiday_Super_Bowl',
 'next_week_holiday_Thanksgiving'
]
pipeline_regression = Pipeline([
                                ('createFeatures', CreateFeatures())                         
                                ,('fillNaValues', FillNaValues()) 
                                ,('featureSelector', FeatureSelector())
                                ,('scaler', StandardScaler())
                                
                               ])

pipeline_xgboost = Pipeline([
                                ('createFeatures', CreateFeatures())
                                ,('featureSelector', FeatureSelector1())
                                
                               ])
X = df_train_full.drop('Weekly_Sales', axis=1)
y = df_train_full['Weekly_Sales']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=.2, random_state=42)
X_train.shape
X_val.shape
X_train_LR = pipeline_regression.fit_transform(X_train) 
X_val_LR = pipeline_regression.transform(X_val)
lr_model = train_linear_regression(X_train_LR, y_train, X_val_LR, y_val)
hyperparameters = {"max_iter": [1, 5, 10],
                      "alpha": [0.0001, 0.001, 0.01, 0.1, 1, 10, 100],
                      "l1_ratio": np.arange(0.0, 1.0, 0.1)}
    
    
scoring = {'MAE': make_scorer(mean_absolute_error), 'r2': make_scorer(r2_score)}

# Create randomized search 5-fold cross validation 
rand_ser = RandomizedSearchCV(lr_model
                         , hyperparameters
                         , random_state=42
                        , n_iter=100
                         , cv=5
                         , verbose=0
                         , n_jobs=-1
                         , scoring=scoring
                         ,refit='r2'
                         )

# Fit randomized search
best_model_lr = rand_ser.fit(X_train_LR, y_train)

print('R^2 = {}'.format(r2_score(y_val, best_model_lr.predict(X_val_LR))))
print('MAE = {}'.format(mean_absolute_error(y_val, best_model_lr.predict(X_val_LR)) ))
print('RMSE = {}'.format(mean_squared_error(y_val, best_model_lr.predict(X_val_LR), squared=False) ))
X_train_XG = pipeline_xgboost.fit_transform(X_train)
X_val_XG = pipeline_xgboost.transform(X_val)
xgb = train_xgboost_regressor(X_train_XG, y_train, X_val_XG, y_val)
explainer = shap.TreeExplainer(xgb)
shap.initjs()
shap_values = explainer.shap_values(X_train_XG, check_additivity=False)
shap.summary_plot(shap_values, X_train_XG)
model_xgb_elatcnet = ensemble_xgb_elastic_net(model_1=lr_model
                         , model_2=xgb
                         , X_train_1=X_train_LR
                         , X_train_2=X_train_XG
                         , y_train = y_train
                         , X_val_1 = X_val_LR
                         , X_val_2 = X_val_XG
                         , y_val = y_val
                        )