import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sbn

from sklearn.preprocessing import OneHotEncoder 

from sklearn.compose import ColumnTransformer

from sklearn import preprocessing

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_absolute_error, r2_score

from sklearn.ensemble import RandomForestRegressor

import warnings

warnings.filterwarnings('ignore')
dataset = pd.read_csv('../input/fish-market/Fish.csv')
dataset
dataset.rename(columns={'Length1' :'Body Height', 'Length2' : 'Total Length', 'Length3' : 'Diagonal Length'} , inplace= True)
dataset.head(10)
dataset.tail(10)
for empty in dataset.columns:

    

    print('Am Column {} with {} missing values and {} datatype'.format(empty,dataset[empty].isnull().sum(),

                                                                       dataset[empty].dtype))
dataset.describe()
dataset.Species.value_counts()




plt.hist(dataset.Species.value_counts())


dataset.hist(bins = 20, grid = False, xlabelsize= 10, ylabelsize= 10, linewidth = 3.0)

plt.tight_layout(rect=(0, 0, 1.5, 1.5))  
 
corr = dataset.corr()

corr
f, ax = plt.subplots(figsize=(10,6))

sbn.heatmap(corr, annot=True, ax = ax, linewidths=1.9, fmt='1.2f')

f.subplots_adjust(top = 1)
# dividing the dataset into X and Y category

features = ['Species','Body Height', 'Total Length', 'Diagonal Length', 'Height', 'Width']

x = dataset[features]

y = dataset['Weight']

#y = dataset.Weight   another way of getting the column value of weight

plt.scatter(x['Height'],y, color = 'red')

plt.title('Height & Weight')

plt.xlabel('Height')

plt.ylabel('Weight')

plt.grid(True)
plt.scatter(x['Width'],y , color = ['blue'])

plt.title('Width $ Weight')

plt.xlabel('Width')

plt.ylabel('Weight')

plt.grid(True)
encode = preprocessing.LabelEncoder()

x['Species']= encode.fit_transform(x['Species'])



transformer = ColumnTransformer([('transfor', OneHotEncoder(), ['Species'] )],  remainder = 'passthrough')

x=  np.array(transformer.fit_transform(x), dtype = np.float)
x.shape
x = pd.DataFrame(x)

pd.set_option('display.max_rows', None)

pd.set_option('display.max_columns', None)

pd.set_option('display.width', None)

pd.set_option('display.max_colwidth', -1)
x = x.drop(columns = [0])



x.columns=['Perkki' ,'Perch','Piki' ,'Roach', 'Smelt', 'Whitefish', 'Body Height',  'Total Length',  'Diagonal Length', 

           'Height', 'Width']
x.shape
X_train, X_test, Y_train, Y_test = train_test_split(x,y, test_size = 0.2, random_state = 0)

ran_reg = RandomForestRegressor(random_state= 0)

lin_reg = LinearRegression()

lin_reg.fit(X_train, Y_train)

ran_reg.fit(X_train,Y_train)
y_pred = lin_reg.predict(X_test)

y_ran_pred = ran_reg.predict(X_test)

y_pred_error = mean_absolute_error(Y_test , y_pred)

y_ran_error = mean_absolute_error(Y_test, y_ran_pred)

y_pred_r2_score = r2_score(Y_test, y_pred)

y_ran_r2_score = r2_score(Y_test, y_ran_pred)

print('Result of using a Linear Regression Model\n',

      y_pred,'\n..............................................\n')

print('Result of using a Random Forest Regression Model\n',y_ran_pred)


print(' mean_absolute_error for the linear regression', y_pred_error)

print(' mean_absolute_error for the linear regression', y_ran_error)

print(' R square for the linear regression', y_pred_r2_score)

print(' R square for the linear regression', y_ran_r2_score)