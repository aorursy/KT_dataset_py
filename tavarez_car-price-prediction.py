import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cbook import boxplot_stats
from sklearn.preprocessing import KBinsDiscretizer
import seaborn as sns
import datetime
df = pd.read_csv('../input/vehicle-dataset-from-cardekho/car data.csv')
df.head()
df.describe()
df.isnull().any()
df.Car_Name.value_counts()
ax = sns.boxplot(x=df.Year, orient='v')
print(boxplot_stats(df.Year)[0])
df = df[df.Year >= boxplot_stats(df.Year)[0]['whislo']]
ax = sns.boxplot(x=df.Year, orient='v')
ax = sns.distplot(df.Year)
ax = sns.distplot(df.Present_Price)
ax = sns.boxplot(x=df.Present_Price, orient='v')
ax = sns.pairplot(df, y_vars='Selling_Price', x_vars=['Present_Price'], kind='reg', height=5)
ax.fig.suptitle('Selling_Price and Present_Price', fontsize=20, y=1.05)
ax
boxplot_stats(df.Present_Price)[0]
df = df[df.Present_Price <=  boxplot_stats(df.Present_Price)[0]['whishi']]
ax = sns.boxplot(x=df.Present_Price, orient='v')
ax = sns.distplot(df.Present_Price)
k = 4
discr = KBinsDiscretizer(n_bins=k, encode='ordinal', strategy='uniform')
df = df.assign(Present_Price_Disc = discr.fit_transform(df.Present_Price.values.reshape(-1,1)) )
[ df[df.Present_Price_Disc == i].Present_Price.plot.density() for i in range(k)]
ax = sns.boxplot(x=df.Kms_Driven, orient='v')
ax = sns.distplot(df.Kms_Driven)
ax = sns.pairplot(df, y_vars='Selling_Price', x_vars=['Kms_Driven'], kind='reg', height=5)
ax.fig.suptitle('Selling_Price and Kms_Driven', fontsize=20, y=1.05)
ax
boxplot_stats(df.Kms_Driven)[0]
df = df[df.Kms_Driven <= boxplot_stats(df.Kms_Driven)[0]['whishi']]

ax = sns.pairplot(df, y_vars='Selling_Price', x_vars=['Kms_Driven'], kind='reg', height=5)
ax.fig.suptitle('Selling_Price and Kms_Driven', fontsize=20, y=1.05)
ax
ax = sns.distplot(df.Kms_Driven)
k = 4
discr = KBinsDiscretizer(n_bins=k, encode='ordinal', strategy='uniform')
df = df.assign(Kms_Driven_Disc = discr.fit_transform(df.Kms_Driven.values.reshape(-1,1)) )
[df[df.Kms_Driven_Disc == i].Kms_Driven.plot.density() for i in range(k) ]
df.Fuel_Type.value_counts().plot.bar()
df.Fuel_Type.value_counts()
df.Seller_Type.value_counts().plot.bar()
df.Seller_Type.value_counts()
df.Transmission.value_counts().plot.bar()
df.Transmission.value_counts()
df.Owner.value_counts().plot.bar()
df.Owner.value_counts()
def Owner_type(x: int) -> str:
    if x == 0:
        return 'one_owner'
    else:
        return 'more_than_one'

df = df.assign(Owner_type = df.Owner.apply(Owner_type))
df.head()
df.Owner_type.value_counts().plot.bar()
def car_type(c:int, y:int, k:int, o:str) -> str:
    if ((c-y) <= 3) and (k <= 30000) and (o == 'one_owner'):
        return 'newest'
    else:
        return 'oldest'

now = datetime.datetime.now()
df = df.assign(Car_Type = df.apply(lambda x: car_type(now.year, x.Year, x.Kms_Driven, x.Owner_type), axis=1))
df.head()  
df = df.assign(Kms_Year = df.apply(lambda x: (x.Kms_Driven/x.Year), axis=1))
df.head() 
now = datetime.datetime.now()
df = df.assign(Years_Use = df.apply(lambda x: (now.year-x.Year), axis=1))
df.head() 
ax = sns.distplot(df.Kms_Year)
k = 3
discr = KBinsDiscretizer(n_bins=k, encode='ordinal', strategy='uniform')
df = df.assign(Kms_Year_Disc = discr.fit_transform(df.Kms_Year.values.reshape(-1,1)) )
[df[df.Kms_Year_Disc == i].Kms_Year.plot.density() for i in range(k)]
ax = sns.boxplot(data=df.Selling_Price, orient='v', width=0.2)
ax.figure.set_size_inches(12,6)
ax.set_title('Selling_Price', fontsize=20)
ax
boxplot_stats(df.Selling_Price)[0]
df = df[df.Selling_Price <= boxplot_stats(df.Selling_Price)[0]['whishi']]
ax = sns.boxplot(y='Selling_Price', x='Car_Type', data=df, orient='v', width=0.5)
ax.figure.set_size_inches(12,6)
ax.set_ylabel('Selling_Price', fontsize=16)
ax.set_xlabel('Car_Type', fontsize=16)
ax
ax = sns.boxplot(y='Selling_Price', x='Seller_Type', data=df, orient='v', width=0.5)
ax.figure.set_size_inches(12,6)
ax.set_ylabel('Selling_Price', fontsize=16)
ax.set_xlabel('Seller_Type', fontsize=16)
ax
ax = sns.boxplot(y='Selling_Price', x='Transmission', data=df, orient='v', width=0.5)
ax.figure.set_size_inches(12,6)
ax.set_ylabel('Selling_Price', fontsize=16)
ax.set_xlabel('Transmission', fontsize=16)
ax
ax = sns.boxplot(y='Selling_Price', x='Year', data=df, orient='v', width=0.5)
ax.figure.set_size_inches(12,6)
ax.set_ylabel('Selling_Price', fontsize=16)
ax.set_xlabel('Year', fontsize=16)
ax
ax = sns.pairplot(df, y_vars='Selling_Price', x_vars=['Year'], kind='reg', height=5)
ax.fig.suptitle('Selling_Price and Year', fontsize=20, y=1.05)
ax
ax = sns.distplot(df.Selling_Price)
varX = ['Year', 'Years_Use','Present_Price', 'Present_Price_Disc', 'Kms_Driven', 'Kms_Driven_Disc', 'Kms_Year', 'Kms_Year_Disc', 'Owner']
ax = sns.pairplot(df, y_vars='Selling_Price', x_vars=varX, kind='reg')
ax.fig.suptitle('Selling_Price and independent variables', fontsize=20, y=1.05)
ax
df.corr()
df['Fuel_Type_Transmission'] = df['Fuel_Type'].astype(str) + '_' + df['Transmission'].astype(str)        
df['Seller_Type_Owner_type'] = df['Seller_Type'].astype(str) + '_' + df['Owner_type'].astype(str)
df['Kms_Year_Disc_Kms_Driven_Disc'] = df['Kms_Year_Disc'].astype(str) + '_' + df['Kms_Driven_Disc'].astype(str)
df['Transmission_Car_Type'] = df['Transmission'].astype(str) + '_' + df['Car_Type'].astype(str) 
df.head()
[df[df.Kms_Driven_Disc == i].Selling_Price.plot.density() for i in range(4)]
df[df.Kms_Driven_Disc == 0].Selling_Price.plot.density()
df[df.Kms_Driven_Disc != 0].Selling_Price.plot.density()
[df[df.Kms_Year_Disc == i].Selling_Price.plot.density() for i in range(3)]
df[df.Fuel_Type == 'Petrol'].Selling_Price.plot.density()
df[df.Fuel_Type == 'Diesel'].Selling_Price.plot.density()
df[df.Fuel_Type == 'CNG'].Selling_Price.plot.density()
df[df.Seller_Type == 'Dealer'].Selling_Price.plot.density()
df[df.Seller_Type != 'Dealer'].Selling_Price.plot.density()
df.Seller_Type.unique()
df[df.Fuel_Type_Transmission == 'Petrol_Manual'].Selling_Price.plot.density()
df[df.Fuel_Type_Transmission != 'Petrol_Manual'].Selling_Price.plot.density()
df[df.Transmission == 'Manual'].Selling_Price.plot.density()
df[df.Transmission != 'Manual'].Selling_Price.plot.density()
[df[df.Present_Price_Disc == i].Selling_Price.plot.density() for i in range(4)]
df[df.Present_Price_Disc == 0].Selling_Price.plot.density()
df[df.Present_Price_Disc != 0].Selling_Price.plot.density()
df = df.assign(Present_Price_Disc_type = df.Present_Price_Disc.apply(lambda x: 'yes' if x == 0 else 'no'))
df[df.Kms_Driven_Disc.isin([0,1])].Selling_Price.plot.density()
df[~df.Kms_Driven_Disc.isin([0,1])].Selling_Price.plot.density()
df[df.Owner_type == 'one_owner'].Selling_Price.plot.density()
df[df.Owner_type != 'one_owner'].Selling_Price.plot.density()
df[df.Car_Type == 'oldest'].Selling_Price.plot.density()
df[df.Car_Type != 'oldest'].Selling_Price.plot.density()
df[df.Kms_Year_Disc.isin([4,3,0])].Selling_Price.plot.density()
df[~df.Kms_Year_Disc.isin([4,3,0])].Selling_Price.plot.density()
df[df.Years_Use <= 5].Selling_Price.plot.density()
df[df.Years_Use > 5].Selling_Price.plot.density()
df[df.Seller_Type_Owner_type == 'Dealer_one_owner'].Selling_Price.plot.density()
df[df.Seller_Type_Owner_type != 'Dealer_one_owner'].Selling_Price.plot.density()
df = df.assign(Dealer_unico_dono = df.Seller_Type_Owner_type.apply(lambda x: 'yes' if x == 'Dealer_one_owner' else 'no'))
df[df.Kms_Year_Disc_Kms_Driven_Disc == '0.0_1.0'].Selling_Price.plot.density()
df[df.Kms_Year_Disc_Kms_Driven_Disc == '1.0_1.0'].Selling_Price.plot.density()
df[df.Kms_Year_Disc_Kms_Driven_Disc == '0.0_0.0'].Selling_Price.plot.density()
df[df.Kms_Year_Disc_Kms_Driven_Disc == '1.0_2.0'].Selling_Price.plot.density()
df[df.Kms_Year_Disc_Kms_Driven_Disc == '2.0_3.0'].Selling_Price.plot.density()
df[df.Kms_Year_Disc_Kms_Driven_Disc == '2.0_2.0'].Selling_Price.plot.density()
df[df.Transmission_Car_Type == 'Automatic_newest'].Selling_Price.plot.density()
df[df.Transmission_Car_Type != 'Automatic_newest'].Selling_Price.plot.density()
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LassoCV
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from math import sqrt
import warnings
warnings.filterwarnings('ignore')

def yes_no_mapping(x: str) -> int:
    if x == 'yes':
        return 1
    else:
        return 0
    
def return_model(alg: str):
    model = None
    if alg == 'lr':
        model = LinearRegression()
    elif alg == 'lasso':
        model = LassoCV(alphas=np.linspace(0, 5, num=100), cv=KFold(n_splits=k, shuffle=True))
    elif alg == 'ridge':
        model = RidgeCV(alphas=np.linspace(0, 5, num=100), cv=KFold(n_splits=k, shuffle=True))
    
    return model 
def build_features(df_train, df_test, mode):
    mm = MinMaxScaler()
    ohe = OneHotEncoder(sparse=False)                     
          
    if mode == 'full':
        x_train_mm = mm.fit_transform(df_train[['Year','Present_Price','Kms_Driven','Kms_Year','Years_Use']])
        x_test_mm = mm.transform(df_test[['Year','Present_Price','Kms_Driven','Kms_Year','Years_Use']])
        x_train_ohe = ohe.fit_transform(df_train[['Fuel_Type','Seller_Type','Transmission','Owner_type','Car_Type','Present_Price_Disc','Kms_Driven_Disc','Kms_Year_Disc','Fuel_Type_Transmission','Seller_Type_Owner_type','Kms_Year_Disc_Kms_Driven_Disc','Transmission_Car_Type','Present_Price_Disc_type','Dealer_unico_dono']])
        x_test_ohe = ohe.transform(df_test[['Fuel_Type','Seller_Type','Transmission','Owner_type','Car_Type','Present_Price_Disc','Kms_Driven_Disc','Kms_Year_Disc','Fuel_Type_Transmission','Seller_Type_Owner_type','Kms_Year_Disc_Kms_Driven_Disc','Transmission_Car_Type','Present_Price_Disc_type','Dealer_unico_dono']])
    elif mode == 'm1':
        x_train_mm = mm.fit_transform(df_train[['Kms_Year','Present_Price','Years_Use','Kms_Driven']])
        x_test_mm = mm.transform(df_test[['Kms_Year','Present_Price','Years_Use','Kms_Driven']])
        x_train_ohe = ohe.fit_transform(df_train[['Fuel_Type_Transmission','Present_Price_Disc','Seller_Type_Owner_type','Kms_Year_Disc_Kms_Driven_Disc', 'Present_Price_Disc_type', 'Transmission_Car_Type']])
        x_test_ohe = ohe.transform(df_test[['Fuel_Type_Transmission','Present_Price_Disc','Seller_Type_Owner_type','Kms_Year_Disc_Kms_Driven_Disc', 'Present_Price_Disc_type', 'Transmission_Car_Type']])
    else:
        x_train_mm = mm.fit_transform(df_train[[ 'Year','Present_Price','Kms_Driven','Kms_Year'  ]])
        x_test_mm = mm.transform(df_test[[ 'Year','Present_Price','Kms_Driven','Kms_Year'  ]])
        x_train_ohe = ohe.fit_transform(df_train[['Fuel_Type','Seller_Type','Transmission','Present_Price_Disc','Kms_Driven_Disc','Kms_Year_Disc','Present_Price_Disc_type']])
        x_test_ohe = ohe.transform(df_test[['Fuel_Type','Seller_Type','Transmission','Present_Price_Disc','Kms_Driven_Disc','Kms_Year_Disc','Present_Price_Disc_type']])
        
    x_train = np.hstack([x_train_ohe, x_train_mm])
    x_test = np.hstack([x_test_ohe, x_test_mm])
    y_train = df_train.Selling_Price.values.reshape(-1, 1)
    y_test = df_test.Selling_Price.values.reshape(-1, 1)
    
    return x_train, y_train, x_test, y_test
for alg in ['lr', 'lasso', 'ridge']:
    k = 5
    mean_r2 = np.zeros(shape=(k, 2))
    mean_rmse = np.zeros(shape=(k, 2))
    cv = KFold(n_splits=k, shuffle=True)
    
    for fold, (train, test) in enumerate(cv.split(df)):
        df_train = df.iloc[train]
        df_test = df.iloc[test]
        x_train, y_train, x_test, y_test = build_features(df_train, df_test, mode='full')
        
        model = return_model(alg)
        model.fit(x_train, y_train)
        
        y_predict_train = model.predict(x_train)
        rmse_train = sqrt(mean_squared_error(y_train, y_predict_train))
        r2_train = r2_score(y_train, y_predict_train)

        y_predict_test = model.predict(x_test)
        rmse_test = sqrt(mean_squared_error(y_test, y_predict_test))  
        r2_test = r2_score(y_test, y_predict_test)

        mean_rmse[fold, 0] = rmse_train
        mean_rmse[fold, 1] = rmse_test
        
        mean_r2[fold, 0] = r2_train
        mean_r2[fold, 1] = r2_test
    
    print('-' * 50)
    print('Algoritmo:', alg)
    print('-' * 50)
    print('Treino - Média RMSE:', mean_rmse[:,0].mean())
    print('Teste - Média RMSE:', mean_rmse[:,1].mean())
    print('-' * 50)
    print('Treino - Média R2:', mean_r2[:,0].mean())
    print('Teste - Média R2:', mean_r2[:,1].mean())
    print('-' * 50)
    print()  
for alg in ['lr', 'lasso', 'ridge']:
    k = 5
    mean_r2 = np.zeros(shape=(k, 2))
    mean_rmse = np.zeros(shape=(k, 2))
    cv = KFold(n_splits=k, shuffle=True)
    
    for fold, (train, test) in enumerate(cv.split(df)):
        df_train = df.iloc[train]
        df_test = df.iloc[test]
        x_train, y_train, x_test, y_test = build_features(df_train, df_test, mode='m1')
        
        model = return_model(alg)
        model.fit(x_train, y_train)
        
        y_predict_train = model.predict(x_train)
        rmse_train = sqrt(mean_squared_error(y_train, y_predict_train))
        r2_train = r2_score(y_train, y_predict_train)

        y_predict_test = model.predict(x_test)
        rmse_test = sqrt(mean_squared_error(y_test, y_predict_test))  
        r2_test = r2_score(y_test, y_predict_test)

        mean_rmse[fold, 0] = rmse_train
        mean_rmse[fold, 1] = rmse_test
        
        mean_r2[fold, 0] = r2_train
        mean_r2[fold, 1] = r2_test
    
    print('-' * 50)
    print('Algoritmo:', alg)
    print('-' * 50)
    print('Treino - Média RMSE:', mean_rmse[:,0].mean())
    print('Teste - Média RMSE:', mean_rmse[:,1].mean())
    print('-' * 50)
    print('Treino - Média R2:', mean_r2[:,0].mean())
    print('Teste - Média R2:', mean_r2[:,1].mean())
    print('-' * 50)
    print()  
for alg in ['lr', 'lasso', 'ridge']:
    k = 5
    mean_r2 = np.zeros(shape=(k, 2))
    mean_rmse = np.zeros(shape=(k, 2))
    cv = KFold(n_splits=k, shuffle=True)
    
    for fold, (train, test) in enumerate(cv.split(df)):
        df_train = df.iloc[train]
        df_test = df.iloc[test]
        x_train, y_train, x_test, y_test = build_features(df_train, df_test, mode='m2')
        
        model = return_model(alg)
        model.fit(x_train, y_train)
        
        y_predict_train = model.predict(x_train)
        rmse_train = sqrt(mean_squared_error(y_train, y_predict_train))
        r2_train = r2_score(y_train, y_predict_train)

        y_predict_test = model.predict(x_test)
        rmse_test = sqrt(mean_squared_error(y_test, y_predict_test))  
        r2_test = r2_score(y_test, y_predict_test)

        mean_rmse[fold, 0] = rmse_train
        mean_rmse[fold, 1] = rmse_test
        
        mean_r2[fold, 0] = r2_train
        mean_r2[fold, 1] = r2_test
    
    print('-' * 50)
    print('Algoritmo:', alg)
    print('-' * 50)
    print('Treino - Média RMSE:', mean_rmse[:,0].mean())
    print('Teste - Média RMSE:', mean_rmse[:,1].mean())
    print('-' * 50)
    print('Treino - Média R2:', mean_r2[:,0].mean())
    print('Teste - Média R2:', mean_r2[:,1].mean())
    print('-' * 50)
    print()  
cv = KFold(n_splits=k, shuffle=True)

predicted = list()
ground_truth = list()
for fold, (train, test) in enumerate(cv.split(df)):
    df_train = df.iloc[train]
    df_test = df.iloc[test]

    x_train, y_train, x_test, y_test = build_features(df_train, df_test, mode='m1')
        
    model = return_model('ridge')
    model.fit(x_train, y_train)

    predicted.extend(model.predict(x_test))
    ground_truth.extend(y_test)
y_true = pd.DataFrame(ground_truth, columns=["Selling_Price"]) 
y_true=y_true.reset_index()["Selling_Price"]
y_pred = pd.DataFrame(predicted, columns=["predicted"]) 
y_pred=y_pred.reset_index()["predicted"]
residuals = y_true - y_pred
ax = sns.distplot(residuals)
ax = sns.scatterplot(x=y_pred, y=residuals, s=100)
ax.figure.set_size_inches(20,8)
ax.set_title('Residual X Predict', fontsize=18, y=1.05)
ax.set_xlabel('Selling Price (y_pred)', fontsize=14)
ax.set_ylabel('Residuals', fontsize=14)
ax
ax = sns.scatterplot(x=y_pred, y=y_true)
ax.figure.set_size_inches(12,6)
ax.set_title('y_pred X y_true', fontsize=18, y=1.05)
ax.set_xlabel('Selling Price - Predict', fontsize=14)
ax.set_ylabel('Selling Price - Real', fontsize=14)
ax
ax=y_true.plot(label="y_true",color="b")
ax.figure.set_size_inches(12,6)
ax=y_pred.plot(label = "y_pred",color="g")
ax.figure.set_size_inches(12,6)
plt.legend()
plt.title("y_true x y_pred")
plt.xlabel("index")
plt.ylabel("value")
plt.show()
