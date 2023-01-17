# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd

import numpy as np

import seaborn as sns 

import matplotlib.pyplot as plt

%matplotlib inline 

import warnings

warnings.filterwarnings('ignore')
car = pd.read_csv('../input/car-data/CarPrice_Assignment.csv')
car.head()
car.shape
car.describe()
car.info()
car.nunique()
plt.figure(figsize=(10,6))

sns.heatmap(data=car.isna(),yticklabels=False)

#as you can see there is no null value
#data cleaning and prepration

car['CarName'] = car['CarName'].apply(lambda x: x.split(' ')[0])
car.head()

#now we can see that we have properly data in CarNmae column
car['CarName'].unique()
#as you can see here is some spelling mistakes we need to correct it 

def replace_name(a,b):

    car['CarName'].replace(a,b,inplace=True)



replace_name('maxda','mazda')

replace_name('toyouta','toyota')

replace_name('porcshce','porsche')

replace_name('vokswagen','volkswagen')

replace_name('vw','volkswagen')
#now checking after replcement

car['CarName'].nunique()
#checking the duplicated entry

duplicate_row = car[car.duplicated()]

print(duplicate_row)

#there is no duplicate row in the dataset
car.columns
car.head(2)
#checking the distribution of the car_price

plt.figure(figsize=(20,8))

plt.subplot(1,2,1)

plt.title('Price Distribution of the Car')

sns.distplot(car['price'])

#plt.show()

plt.subplot(1,2,2)

plt.title('Price  of the Car')

sns.boxplot(car['price'],orient='v')
print(car['price'].describe(percentiles=[0.25,0.50,0.65,0.75,0.85,0.90,1]))
sns.distplot(car['wheelbase'])

plt.title('Wheelbase Info')
sns.distplot(car['horsepower'])

plt.title('horse power details')
plt.figure(figsize=(20,6))

plt.subplot(1,4,1)

plt1 = car['CarName'].value_counts().plot(kind='bar')

print("Frequecney of car type \n",car['CarName'].value_counts(normalize=True).head()*100)

print("*********************************************")

plt.title('Frequency of cars')



plt1.set(xlabel='Car_Names',ylabel='Frequecy')

plt.subplot(1,4,2)

plt1 = car['fueltype'].value_counts().plot(kind='bar')

plt1.set(xlabel='Fuel_Type',ylabel='Frequecy')

print("fuel Type counts \n",car['fueltype'].value_counts(normalize=True)*100)

print("*********************************************")

plt.title('Fueltype of cars')



plt.subplot(1,4,3)

plt1 = car['drivewheel'].value_counts().plot(kind='bar')

plt1.set(xlabel='Wheel_Type',ylabel='Frequecy')

print("type of drive wheel count \n",car['drivewheel'].value_counts(normalize=True)*100)

print("*********************************************")

plt.title('Driver Wheel of Car')



plt.subplot(1,4,4)

plt1 = car['carbody'].value_counts().plot(kind='bar')

plt1.set(xlabel='Car_Type',ylabel='Frequecy')

print("car type count\n",car['carbody'].value_counts(normalize=True)*100)

print("*********************************************")

plt.title('Car Type')

plt.show()
plt.figure(figsize=(20,8))

plt.subplot(1,2,1)

sns.countplot(x='symboling',data=car)

plt.title('Spread of symboiling')

print("Now plotting symboiling vs price")

plt.subplot(1,2,2)

sns.boxplot(x='symboling',y='price',data=car)

plt.title('Symboiling VS price')
car.head(2)
car['enginetype'].unique()
car['enginetype'].hist(bins=20)
plt.figure(figsize=(16,6))

plt.subplot(1,2,1)

sns.countplot(x='enginetype',data=car)

plt.subplot(1,2,2)

sns.boxplot(x='enginetype',y='price',data=car)
plt.figure(figsize=(14,6))

plt.subplot(1,2,1)

sns.countplot(x='doornumber',data=car)

plt.title("Distrbution of data no of doorwise")

plt.subplot(1,2,2)

sns.boxplot(x='doornumber',y='price',data=car)

plt.show()
car.head(2)
plt.figure(figsize=(20,8))

plt.subplot(1,2,1)

c = sns.countplot(x='CarName',data=car)

c.set_xticklabels(c.get_xticklabels(), rotation=90)

plt.title("Distrbution of data no of car wise")

plt.subplot(1,2,2)

b = sns.boxplot(x='CarName',y='price',data=car)

b.set_xticklabels(b.get_xticklabels(),rotation=90)

plt.show()
car.groupby('enginetype').mean()['price'].plot(kind='bar')
plt.figure(figsize=(16,10))

df=pd.DataFrame(car.groupby('CarName')['price'].mean().sort_values(ascending=False))

df.plot(kind='bar')

df=pd.DataFrame(car.groupby('fueltype')['price'].mean().sort_values(ascending=False))

df.plot(kind='bar')

df=pd.DataFrame(car.groupby('carbody')['price'].mean().sort_values(ascending=False))

df.plot(kind='bar')

plt.show()
plt.figure(figsize=(14,6))

plt.subplot(1,2,1)

sns.countplot(x='aspiration',data=car,palette=('plasma'))

plt.subplot(1,2,2)

sns.boxplot(x='aspiration',y='price',data=car)
#now building a function to plot reaming categorical variables

def plot_cats(var,fig):

    plt.subplot(4,2,fig)

    sns.countplot(car[var])

    plt.title(var+'_'+'Histogram')

    plt.subplot(4,2,(fig+1))

    sns.boxplot(x=car[var],y=car['price'])

    plt.title(var+'_'+'VS'+'_'+'Price')

    #plt.show()

plt.figure(figsize=(15,20))

plot_cats('enginelocation', 1)

plot_cats('cylindernumber', 3)

plot_cats('fuelsystem', 5)

plot_cats('drivewheel', 7)

def scatter_plot(var,fig):

    plt.subplot(4,2,fig)

    sns.scatterplot(x=var,y='price',data=car)

    plt.title(var+'_'+'VS'+'_'+'price')

plt.figure(figsize=(15,20))

scatter_plot('carlength', 1)

scatter_plot('carwidth', 2)

scatter_plot('carheight', 3)

scatter_plot('curbweight', 4)



plt.tight_layout()
def scatter_plot(var,fig,h):

    plt.subplot(4,2,fig)

    sns.scatterplot(x=var,y='price',data=car,hue=h)

    plt.title(var+'_'+'VS'+'_'+'price')

plt.figure(figsize=(15,20))

scatter_plot('carlength', 1,'aspiration')

scatter_plot('carwidth', 2,'aspiration')

scatter_plot('carheight', 3,'aspiration')

scatter_plot('curbweight', 4,'aspiration')
def pp(x,y,z):

    sns.pairplot(car, x_vars=[x,y,z], y_vars='price',size=4, aspect=1,kind='scatter')

    plt.show()



pp('enginesize', 'boreratio', 'stroke')

pp('compressionratio', 'horsepower', 'peakrpm')

pp('wheelbase', 'citympg', 'highwaympg')
###driving the new feature based on perivous

car['fueleconomey'] = (car['citympg']*0.55) + (car['highwaympg']*0.45)
car['fueleconomey'].head()
#binning the companies based on the average price of each company

car['price'] = car['price'].astype(int)

temp = car.copy()

table = temp.groupby(['CarName']).mean()['price'].sort_values(ascending=False)

table

temp = temp.merge(table.reset_index(),how='left',on='CarName')

cars_bin=['Budget','Medium','Highend']

bins = [0,10000,20000,40000]

car['carsrange'] = pd.cut(temp['price_y'],bins,right=False,labels=cars_bin)

car.head()
#now perforrming bivariate analysis

plt.figure(figsize=(8,6))



plt.title('Fuel economy vs Price')

sns.scatterplot(x=car['fueleconomey'],y=car['price'],hue=car['drivewheel'])

plt.xlabel('Fuel Economy')

plt.ylabel('Price')



plt.show()

plt.tight_layout()
plt.figure(figsize=(25, 6))



df = pd.DataFrame(car.groupby(['fuelsystem','drivewheel','carsrange'])['price'].mean().unstack(fill_value=0))

df

df.plot.bar()

plt.title('Car Range vs Average Price')

plt.show()
cars_lr = car[['price', 'fueltype', 'aspiration','carbody', 'drivewheel','wheelbase',

                  'curbweight', 'enginetype', 'cylindernumber', 'enginesize', 'boreratio','horsepower', 

                    'fueleconomey', 'carlength','carwidth', 'carsrange']]

cars_lr.head()
def dummies(x,df):

    temp = pd.get_dummies(df[x],drop_first=True)

    df = pd.concat([df,temp],axis=1)

    df.drop([x],axis=1,inplace=True)

    return df



cars_lr = dummies('fueltype',cars_lr)

cars_lr = dummies('aspiration',cars_lr)

cars_lr = dummies('carbody',cars_lr)

cars_lr = dummies('drivewheel',cars_lr)

cars_lr = dummies('enginetype',cars_lr)

cars_lr = dummies('cylindernumber',cars_lr)

cars_lr = dummies('carsrange',cars_lr)
cars_lr.head()
cars_lr.shape
from sklearn.model_selection import train_test_split

df_train,df_test = train_test_split(cars_lr,random_state=100,train_size=0.7,test_size=0.3)
df_train.head()
from sklearn.preprocessing import MinMaxScaler



scaler = MinMaxScaler()

num_var = ['price','wheelbase','curbweight','enginesize','boreratio','horsepower','fueleconomey',

                    'carlength','carwidth']

df_train[num_var] = scaler.fit_transform(df_train[num_var])
df_train.head()
df_train.describe()
plt.figure(figsize=(30,25))

sns.heatmap(data=df_train.corr(),annot=True,cmap="YlGnBu")
y_train = df_train.pop('price')

x_train = df_train

y_train.shape
from statsmodels.api import add_constant

import statsmodels.api as sm

from sklearn.feature_selection import RFE

from sklearn.linear_model import LinearRegression
lm = LinearRegression()

lm.fit(x_train,y_train)

rfe = RFE(lm, 10)

rfe = rfe.fit(x_train, y_train)

#del x_train['const']
list(zip(x_train.columns,rfe.support_,rfe.ranking_))
rfe.support_
x_train.columns[rfe.support_]
x_train.columns
x_train_rfe = x_train[x_train.columns[rfe.support_]]

x_train_rfe.head()

y_train.shape
def build_model(x_train_rfe,y_train):

    x_train_rfe = add_constant(x_train_rfe)

    model = sm.OLS(y_train,x_train_rfe)

    result = model.fit()

    print(result.summary())

    return x_train_rfe

    
model1 = build_model(x_train_rfe,y_train)

#x_train_rfe.shape

#y_train.shape
model1 = x_train_rfe.drop(["twelve"], axis = 1)
model2 = build_model(model1,y_train)
model2 = model1.drop('fueleconomey',axis = 1)
model3 = build_model(model2,y_train)
from statsmodels.stats.outliers_influence import variance_inflation_factor
def vif_checker(x_train_rfe):

    vif = pd.DataFrame()

    vif['VIF'] = [variance_inflation_factor(x_train_rfe.values,i) for i in range (x_train_rfe.shape[1])]

    vif['features']= x_train_rfe.columns

    return vif
vif_checker(model3)
model4 = model3.drop(['curbweight'],axis=1)
model4 = build_model(model4,y_train)
vif_checker(model4)
model5 = model4.drop(['sedan'],axis=1)
model5 = build_model(model5,y_train)
vif_checker(model5)
#dropping wegon because of the highest p_value

model6 = model5.drop(['wagon'],axis=1)
build_model(model6,y_train)
vif_checker(model6)
##predeciting



model6 = model6.drop(['const'],axis=1)
model6.head()
#now predecting value and checking error

def mape(y_true,y_pred):

    return np.mean(np.abs((y_true-y_pred)/y_true)*100)
df_test.head()
#y_test = df_test.pop('price')

x_test = df_test
model6 = add_constant(model6)

model7 = sm.OLS(y_train,model6)

result_f = model7.fit()

print(result_f.summary())
result_f.fittedvalues
from sklearn.preprocessing import MinMaxScaler



scaler = MinMaxScaler()

num_var = ['price','wheelbase','curbweight','enginesize','boreratio','horsepower','fueleconomey',

                    'carlength','carwidth']

df_test[num_var] = scaler.fit_transform(df_test[num_var])
df_test.head()
#y_test = df_test.pop('price')

x_test = df_test

#x_test
#model6 = model6.drop('const',axis=1)

x_test = x_test[model6.columns]

x_test = add_constant(x_test)
y_pred = result_f.predict(x_test)
y_pred.head()
from sklearn.metrics import mean_absolute_error

print('MAE on Train Set is :',mean_absolute_error(y_train, result_f.fittedvalues))



print ('********************************')



print('MAE on Test Set is :',mean_absolute_error(y_test,y_pred))
from sklearn.metrics import mean_squared_error

print('MAE on Train Set is :',mean_squared_error(y_train, result_f.fittedvalues))



print ('********************************')



print('MAE on Test Set is :',mean_squared_error(y_test,y_pred))
from sklearn.metrics import r2_score 

r2_score(y_test, y_pred)
#now plotting the data 

sns.scatterplot(y_test,y_pred)

plt.title('y_test VS y_pred')

plt.xlabel('y_test')

plt.ylabel('y_pred')

plt.show()
#now checking it follow the linear assumption or not

#fitted_values

model_fit_y = result_f.fittedvalues

#residuals

model_resd = result_f.resid

#normalizing the resdials

model_norm_res = result_f.get_influence().resid_studentized_internal

#taking tha abosulate square root

model_norm_res_abs_sqrt = np.sqrt(np.abs(model_norm_res))

#absulte resuidals

model_resd_abs = np.abs(model_resd)

##levarage

model_levarge = result_f.get_influence().hat_matrix_diag

##cooks distance from model intera

models_cooks = result_f.get_influence().cooks_distance[0]

#np.mean(model_levarge)---0.04195804195804197

from statsmodels.graphics.gofplots import ProbPlot

QQ = ProbPlot(model_norm_res)

plot_lm_2 = QQ.qqplot(line='45', alpha=0.5, color='#4C72B0', lw=1,xlabel='Theoretical Quantiles',ylabel='Standardized Residuals')



plot_lm_2.set_figheight(8)

plot_lm_2.set_figwidth(12)
infl=result_f.get_influence()

influential_df=infl.summary_frame().filter(["hat_diag","cooks_d"])
influential_df[influential_df['hat_diag']> 3* np.mean(influential_df['hat_diag'])]
influential_df.sort_values('cooks_d',ascending=False).head()