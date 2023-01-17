## Competição DSA - Kaggle - Março 2019



## Prevendo as vendas



# Desenvolvido por: Silvio Lima  

     

######################################################

# Imports



import numpy as np

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

from IPython.core.pylabtools import figsize

import seaborn as sns



warnings.filterwarnings('ignore')

%matplotlib inline

plt.style.use('ggplot')

pd.options.mode.chained_assignment = None

pd.set_option('display.max_columns', 60)



from sklearn.ensemble import GradientBoostingRegressor

from sklearn.model_selection import KFold, StratifiedKFold, RandomizedSearchCV



# Imputing missing values and scaling values

from sklearn.preprocessing import Imputer, MinMaxScaler
######################################################

# Datasets

data_train = pd.read_csv('../input/dataset_treino.csv')

data_test = pd.read_csv('../input/dataset_teste.csv')

#

loja = pd.read_csv('../input/lojas.csv')



# Colunas Train

colunas_train=data_train.columns

colunas_train
# Colunas teste

colunas_test=data_test.columns

colunas_test
# Colunas lojas

colunas_loja=loja.columns

colunas_loja
# check out the size of the data

print("Train data shape:", data_train.shape)

print("Test data shape:", data_test.shape)

print("Loja data shape:", loja.shape)
# First lines

data_train.head()
# Describe like count, mean, std, min, max etc

data_train.Sales.describe()

# Histogram of Sales

#plt.hist(data_train.Sales, color='blue')

#plt.show()
# Subset of columns

numeric_features = data_train.select_dtypes(include=[np.number])

numeric_features.dtypes
# Correlation between the columns and examine the correlations between the features and the target.

corr = numeric_features.corr()

corr
# Features correlated with Sales.

print (corr['Sales'].sort_values(ascending=False)[-9:])
# Scatter plots to visualize the relationship between Sales and others 

#plt.scatter(x=data_train['DayOfWeek'], y=data_train['Sales'])

#plt.ylabel('Sales')

#plt.xlabel('DayOfWeek')

#plt.show()

#

# Domingo é dia 1. As vendas estão bem distribuidas ao longo da semana.
# Scatter plots to visualize the relationship between Sales and others 

#plt.scatter(x=data_train['Promo'], y=data_train['Sales'])

#plt.ylabel('Sales')

#plt.xlabel('Promo')

#plt.show()

#

# As vendas continuam se há promoção ou não.
# Scatter plots to visualize the relationship between Sales and others 

#plt.scatter(x=data_train['Customers'], y=data_train['Sales'])

#plt.ylabel('Sales')

#plt.xlabel('Customers')

#plt.show()

#

# As vendas com certeza dependem do cliente.
# Scatter plots to visualize the relationship between Sales and others 

#plt.scatter(x=data_train['SchoolHoliday'], y=data_train['Sales'])

#plt.ylabel('Sales')

#plt.xlabel('SchoolHoliday')

#plt.show()

#

# Apesar das escolas fecharem nos feriados escolares, as vendas continuam.
# Scatter plots to visualize the relationship between Sales and others 

#plt.scatter(x=data_train['Open'], y=data_train['Sales'])

#plt.ylabel('Sales')

#plt.xlabel('Open')

#plt.show()

# 

# Só há venda se lojas esta aberta, faz sentido.

# Talvez analisar apenas as lojas abertas, pois se esta fechada a previsao de vendas é zero.
# Extracting Year, Month and Day from Date from data_train

DATE = pd.to_datetime(data_train.Date) 

data_train['Year'] = DATE.dt.year

data_train['Month'] = DATE.dt.month

data_train['Day'] = DATE.dt.day

# Extracting Year, Month and Day from Date from data_teste de envio

DATE = pd.to_datetime(data_test.Date) 

data_test['Year'] = DATE.dt.year

data_test['Month'] = DATE.dt.month

data_test['Day'] = DATE.dt.day

# Drop column Date - Train

data_train=data_train.drop(['Date'],axis=1)
# Drop column Date - Test

data_test=data_test.drop(['Date'],axis=1)

# StateHoliday has 1 number and char mixed. Changing "0" to "d" -- dataset train and test

#temp=data_train['StateHoliday']

#temp=temp.replace('0','d')

#data_train['StateHoliday']=temp

#data_train.describe

#

# Test

#temp=data_test['StateHoliday']

#temp=temp.replace('0','d')

#data_test['StateHoliday']=temp

#data_test.describe
# Put data_train and lojas together - inner join by Store

data_full=data_train.set_index('Store').join(loja.set_index('Store'))

data_full.shape



# Put data_test and lojas together - inner join by Store

data_full_test=data_test.set_index('Store').join(loja.set_index('Store'))

data_full_test.shape





# Let´s try to remove Open and use only rows where Store was Open = 1

data_full_open_1 = data_full[data_full['Open'] == 1]

data_full_open_1.shape
# Train and lojas

data_full.columns
# Reorder dataset to put target first position

data_full=data_full[['Sales','DayOfWeek', 'Customers', 'Open', 'Promo', 'StateHoliday',

       'SchoolHoliday', 'Year', 'Month', 'Day', 'StoreType', 'Assortment',

       'CompetitionDistance', 'CompetitionOpenSinceMonth',

       'CompetitionOpenSinceYear', 'Promo2', 'Promo2SinceWeek',

       'Promo2SinceYear', 'PromoInterval']]
# Train and lojas

data_full.columns
# Train numeric subset

train_numeric_subset = data_full.select_dtypes('number')

# DataFrame test numeric - Total nulls

nulls = pd.DataFrame(train_numeric_subset.isnull().sum().sort_values(ascending=False)[:25])

nulls.columns = ['Null Count']

nulls.index.name = 'Feature'



nulls
# Fill NA with mean() in train_numeric_subset

train_numeric_subset.fillna(train_numeric_subset.mean(),inplace=True)
# Check if OK - Total nulls

nulls = pd.DataFrame(train_numeric_subset.isnull().sum().sort_values(ascending=False)[:25])

nulls.columns = ['Null Count']

nulls.index.name = 'Feature'

nulls
# Test Numeric subset

test_numeric_subset = data_full_test.select_dtypes('number')



# DataFrame test numeric - Total nulls

nulls = pd.DataFrame(test_numeric_subset.isnull().sum().sort_values(ascending=False)[:25])

nulls.columns = ['Null Count']

nulls.index.name = 'Feature'



nulls
# Fill NA with mean() in train_numeric_subset

test_numeric_subset.fillna(test_numeric_subset.mean(),inplace=True)
# Check if OK - Total nulls

nulls = pd.DataFrame(test_numeric_subset.isnull().sum().sort_values(ascending=False)[:25])

nulls.columns = ['Null Count']

nulls.index.name = 'Feature'

nulls
# One hot encode Train dataset

#df=train_categorical_subset

#df = df.fillna(df.mode().iloc[0])

train_categorical_subset = data_full[['StateHoliday','StoreType', 'Assortment','PromoInterval']]

train_categorical_subset = pd.get_dummies(train_categorical_subset)

train_categorical_subset
# One hot encode Test dataset

test_categorical_subset = data_full_test[['StateHoliday', 'StoreType', 'Assortment','PromoInterval']]

test_categorical_subset = pd.get_dummies(test_categorical_subset)

test_categorical_subset
# Train 

# Join the two dataframes using concat

# Make sure to use axis = 1 to perform a column bind

# Train

data_train = pd.concat([train_numeric_subset, train_categorical_subset], axis = 1)

# Final Test

# Backing to data_full

# Join the two dataframes using concat

# Make sure to use axis = 1 to perform a column bind

# Test

data_test = pd.concat([test_numeric_subset, test_categorical_subset], axis = 1)





data_test.head(5)
data_test.columns
# Correlatin after adjustes

corr=data_train.corr()
correlations = data_train.corr()['Sales'].dropna().sort_values()
# Five Lowest correlations

correlations.head(5)

# Five Biggest correlation

correlations.tail(7)

# Plot correlation matrix



corr = data_train.corr()

_ , ax = plt.subplots( figsize =( 30 , 30 ) )

cmap = sns.diverging_palette( 220 , 10 , as_cmap = True )

_ = sns.heatmap(corr, cmap = cmap, square=True, cbar_kws={ 'shrink' : .9 }, ax=ax, annot = True, annot_kws = {'fontsize' : 10 })
# Split X e Y from train



#data_train_final=data_train[['Sales','DayOfWeek', 'Open', 'Promo', 'SchoolHoliday', 'Year', 'Month','Day', 'CompetitionDistance', 'CompetitionOpenSinceMonth','CompetitionOpenSinceYear', 'Promo2', 'Promo2SinceWeek','Promo2SinceYear', 'StateHoliday_0', 'StateHoliday_a', 'StoreType_a','StoreType_b', 'StoreType_c', 'StoreType_d', 'Assortment_a','Assortment_b', 'Assortment_c', 'PromoInterval_Feb,May,Aug,Nov','PromoInterval_Jan,Apr,Jul,Oct', 'PromoInterval_Mar,Jun,Sept,Dec']]

        

data_train_final=data_train[['Sales','DayOfWeek','Open', 'Promo','StoreType_b']]



previsores = data_train_final.iloc[:, 1:5].values

target = data_train.iloc[:,0].values



# Create the scaler object with a range of 0-1

# Previsores (X)

#scaler = MinMaxScaler(feature_range=(0, 1))

#X=previsores

# Fit on the training data

#scaler.fit(X)

#previsores = scaler.transform(X)

# Target (y)

#scaler_y = MinMaxScaler(feature_range=(0, 1))

#y=target

#target= scaler_y.fit_transform(y.reshape(-1, 1))

target
# Split dataset in train and test

from sklearn.model_selection import train_test_split

previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores, target, test_size=0.3, random_state=0)



X=previsores_treinamento

y=classe_treinamento

############################################################################

## Modelo Final



model = GradientBoostingRegressor(

    n_estimators = 500

    ,min_samples_split = 6

    ,min_samples_leaf = 6

    ,max_features = None

    ,max_depth = 5

    ,loss = 'lad'

    ,random_state=42)



model.fit(X, y)

pred = np.round(model.predict(previsores_teste)).astype(int)

pred
from sklearn.metrics import r2_score

r2_score(classe_teste,pred)

# Resultado

X_test=d=data_test[['DayOfWeek','Open', 'Promo','StoreType_b']]



X_test = X_test.iloc[:,:].values



pred = np.round(model.predict(X_test)).astype(int)





submission = pd.DataFrame({'Id': data_test['Id'], 'Sales': pred })

submission.to_csv('submission.csv',index=False)








