# some imports



from IPython.core.display import display, HTML

display(HTML("<style>.container { width:100% !important; }</style>"))



# Python ≥3.5 is required

import sys

assert sys.version_info >= (3, 5)

 

# Scikit-Learn ≥0.20 is required

import sklearn

assert sklearn.__version__ >= "0.20"



# Common imports

import numpy as np

import os



# to make this notebook's output stable across runs

np.random.seed(42)



# To plot pretty figures

%matplotlib inline

import matplotlib as mpl

import matplotlib.pyplot as plt

mpl.rc('axes', labelsize=14)

mpl.rc('xtick', labelsize=12)

mpl.rc('ytick', labelsize=12)

plt.rc('font', size=12) 

plt.rc('figure', figsize = (12, 5))



# Settings for the visualizations

import seaborn as sns

sns.set_style("whitegrid")

sns.set_context("notebook", font_scale=1, rc={"lines.linewidth": 2,'font.family': [u'times']})



import pandas as pd

pd.set_option('display.max_rows', 25)

pd.set_option('display.max_columns', 500)

pd.set_option('display.max_colwidth', 50)



# Ignore useless warnings (see SciPy issue #5998)

import warnings

warnings.filterwarnings(action="ignore", message="^internal gelsd")



# create output folder

if not os.path.exists('output'):

    os.makedirs('output')

if not os.path.exists('output/session1'):

    os.makedirs('output/session1')
## load data



train_set = pd.read_csv('/kaggle/input/mlub-housing-house-prediction/train_set.csv',index_col=0)  

test_set = pd.read_csv('/kaggle/input/mlub-housing-house-prediction/test_set.csv',index_col=0) 
# print the dataset size

print("There is", train_set.shape[0], "samples")

print("Each sample has", train_set.shape[1], "features")
# print the top elements from the dataset

train_set.head(10)
# As it can be seen the database contains several features, some of them numerical and some of them are categorical.

# It is important to check each of the to understand it.
# we can see the type of each features as follows

train_set.dtypes
# print those categorical features

train_set.select_dtypes(include=['object']).head()
# We can check how many different type there is in the dataset using the folliwing line

train_set["Type"].value_counts()
sns.countplot(y="Type", data=train_set, color="c")
sns.distplot(train_set["Price"])

plt.show()
## the features



features = ['Rooms','Landsize', 'BuildingArea', 'YearBuilt']

## DEFINE YOUR FEATURES

X = train_set[features].fillna(0)

y = train_set[['Price']]



## the model

# KNeighborsRegressor

from sklearn import neighbors

n_neighbors = 3 # you can modify this paramenter (ONLY THIS ONE!!!)

model = neighbors.KNeighborsRegressor(n_neighbors)



## fit the model

model.fit(X, y)



## predict training set

y_pred = model.predict(X)



## Evaluate the model and plot it

from sklearn.metrics import mean_squared_error, r2_score

print("----- EVALUATION ON TRAIN SET ------")

print("RMSE",np.sqrt(mean_squared_error(y, y_pred)))

print("R^2: ",r2_score(y, y_pred))





plt.scatter(y, y_pred)

plt.xlabel('Price')

plt.ylabel('Predicted price');

plt.show()



## predict the test set and generate the submission file

X_test = test_set[features].fillna(0)

y_pred = model.predict(X_test)



df_output = pd.DataFrame(y_pred)

df_output = df_output.reset_index()

df_output.columns = ['index','Price']



df_output.to_csv('output/session1/baseline.csv',index=False)
#Modify the given data

train_set_c = train_set.copy()

test_set_c = test_set.copy()



## Define methods



#Detects data different to the 99% of the data with the std and returns its index 

def outlier_detecter(df):

    out=[]

    for i in np.array(df.index): 

        z = (df.loc[i]-np.mean(df))/np.std(df)

        if np.abs(z) > 10: #99% of the passed data 

            out.append(i)

    return out



#'normalizes' outliers, assigns the value to the first value that is not an outlier (if the outlier is the maximum and above the mean, it will select the max value that is not an outlier, and viceversa with min values)

def outlier_deleter(indexes,data_set,feature=''):

    max_outlier = []

    min_outlier = []

    data_set_mean = data_set[feature].mean()

    

    for i in indexes: 

        if data_set.loc[i,[feature]].item() > data_set_mean: 

            max_outlier.append(i)

        else:

            min_outlier.append(i)

                

    ordered_max_indexes = np.argpartition(-np.array(data_set[feature]),len(max_outlier)+1)[:len(indexes)+1]

    ordered_min_indexes = np.argpartition(np.array(data_set[feature]),len(min_outlier)+1)[:len(indexes)+1]

    

    for i in indexes: 

        if i in max_outlier: 

            data_set.loc[i,[feature]] = data_set.loc[data_set.index[ordered_max_indexes[-1]],[feature]].item()

        else:

            data_set.loc[i,[feature]] = data_set.loc[data_set.index[ordered_min_indexes[-1]],[feature]].item()

    return data_set

#-----------------------------------------------------Drop nan values----------------------------------------------------



train_set_c = train_set_c.dropna()



#-------------------------------------------Create new columns and dummies-----------------------------------------------



train_set_c['Building_age'] = pd.DatetimeIndex(train_set_c['Date']).year - train_set_c['YearBuilt']

test_set_c['Building_age'] = pd.DatetimeIndex(test_set_c['Date']).year - test_set_c['YearBuilt']



train_set_c = pd.get_dummies(train_set_c, columns=['Type'], prefix = ['Type'])

test_set_c = pd.get_dummies(test_set_c, columns=['Type'], prefix = ['Type'])



train_set_c = pd.get_dummies(train_set_c, columns=['Regionname'], prefix = ['Regionname'])

test_set_c = pd.get_dummies(test_set_c, columns=['Regionname'], prefix = ['Regionname'])







#-----------------------------------Filter possible outliers from numeric data-------------------------------------------



train_set_c = outlier_deleter(outlier_detecter(train_set_c['Price']),train_set_c,'Price')

train_set_c = outlier_deleter(outlier_detecter(train_set_c['Rooms']),train_set_c,'Rooms')

train_set_c = outlier_deleter(outlier_detecter(train_set_c['Landsize']),train_set_c,'Landsize')

train_set_c = outlier_deleter(outlier_detecter(train_set_c['BuildingArea']),train_set_c,'BuildingArea')

train_set_c = outlier_deleter(outlier_detecter(train_set_c['Car']),train_set_c,'Car')

train_set_c = outlier_deleter(outlier_detecter(train_set_c['Bathroom']),train_set_c,'Bathroom')

train_set_c = outlier_deleter(outlier_detecter(train_set_c['Distance']),train_set_c,'Distance')

train_set_c = outlier_deleter(outlier_detecter(train_set_c['Building_age']),train_set_c,'Building_age')

train_set_c = outlier_deleter(outlier_detecter(train_set_c['Distance']),train_set_c,'Distance')





#-----------------------------------------------Select features---------------------------------------------------------



features = ['Rooms','Landsize', 'BuildingArea','Car','Bathroom','Distance','Building_age','Lattitude','Longtitude','Postcode','Propertycount',

            'Type_h','Type_t','Type_u',

           'Regionname_Eastern Metropolitan','Regionname_Eastern Victoria','Regionname_Northern Metropolitan','Regionname_Northern Victoria','Regionname_South-Eastern Metropolitan','Regionname_Southern Metropolitan','Regionname_Western Metropolitan','Regionname_Western Victoria']



            

features_norm = features

            

#---------------------------------------------Normalize features--------------------------------------------------------



train_set_c[features_norm] = train_set_c[features_norm].apply(lambda x: (x - x.min()) / (x.max() - x.min()))

test_set_c[features_norm] = test_set_c[features_norm].apply(lambda x: (x - x.min()) / (x.max() - x.min()))



## DEFINE YOUR FEATURES

#-----------------------------------------------------------------------------------------------------------------------

X = train_set_c[features].fillna(0)

y = train_set_c[['Price']]

#-----------------------------------------------------------------------------------------------------------------------



## the model

# KNeighborsRegressor





from sklearn import neighbors

n_neighbors = 13# you can modify this paramenter (ONLY THIS ONE!!!) #6



model = neighbors.KNeighborsRegressor(n_neighbors)





## fit the model

model.fit(X, y)

## predict training set

y_pred = model.predict(X)



## Evaluate the model and plot it

from sklearn.metrics import mean_squared_error, r2_score

print("----- EVALUATION ON TRAIN SET ------")

print("RMSE",np.sqrt(mean_squared_error(y, y_pred)))

print("R^2: ",r2_score(y, y_pred))





plt.scatter(y, y_pred)

plt.xlabel('Price')

plt.ylabel('Predicted price');

plt.show()



## predict the test set and generate the submission file

X_test = test_set_c[features].fillna(0)

y_pred = model.predict(X_test)



df_output = pd.DataFrame(y_pred)

df_output = df_output.reset_index()

df_output.columns = ['index','Price']





df_output.to_csv('output/session1/baseline.csv',index=False)