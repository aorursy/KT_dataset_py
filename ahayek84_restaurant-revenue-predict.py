# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import seaborn as sns; sns.set(style="ticks", color_codes=True)

import matplotlib.pyplot as plt

%matplotlib inline
#load data

train = pd.read_csv('/kaggle/input/restaurant-revenue-prediction/train.csv')

test = pd.read_csv('/kaggle/input/restaurant-revenue-prediction/test.csv')
train.columns

#Drop sequences 

#train = train.drop('Id', axis=1)

#test = test.drop('Id', axis=1)
print("Train :",train.shape)

print("Test:",test.shape)
train['Open Date'] = pd.to_datetime(train['Open Date'])

test['Open Date'] = pd.to_datetime(test['Open Date'])
# get column with null values  

train.columns[train.isna().any()].tolist()
test.columns[test.isna().any()].tolist()
#Seperate categorical from numberical variables for analysis 

numerical_features = train.select_dtypes([np.number]).columns.tolist()

categorical_features = train.select_dtypes(exclude = [np.number,np.datetime64]).columns.tolist()
categorical_features
train[numerical_features].head()
train[categorical_features].head()
train.describe()
print(train['revenue'].describe())

sns.set(style='whitegrid', palette="deep", font_scale=1.1, rc={"figure.figsize": [8, 5]})

sns.distplot(

    train['revenue'], norm_hist=False, kde=True

).set(xlabel='revenue', ylabel='P(revenue)');
train[train['revenue'] > 10000000 ]
# Drop outlayers

train = train[train['revenue'] < 10000000 ]

train.reset_index(drop=True).head()
train.shape
#train[numerical_features].hist(figsize=(30, 35), layout=(12, 4));



k = len(train[numerical_features].columns)

n = 3

m = (k - 1) // n + 1 ## Floor Division (also called Integer Division)

fig, axes = plt.subplots(m, n, figsize=(n * 5, m * 3))

for i, (name, col) in enumerate(train[numerical_features].iteritems()):

    r, c = i // n, i % n

    ax = axes[r, c]

    col.hist(ax=ax)

    ax2 = col.plot.kde(ax=ax, secondary_y=True, title=name)

    ax2.set_ylim(0)



fig.tight_layout()
train[train['P37']  == 0].shape
fig, ax = plt.subplots(3, 1, figsize=(40, 30))

for variable, subplot in zip(categorical_features, ax.flatten()):

    df_2 = train[[variable,'revenue']].groupby(variable).revenue.sum().reset_index()

    df_2.columns = [variable,'total_revenue']

    sns.barplot(x=variable, y='total_revenue', data=df_2 , ax=subplot)

    for label in subplot.get_xticklabels():

        label.set_rotation(90)
fig, ax = plt.subplots(10, 4, figsize=(30, 35))

for variable, subplot in zip(numerical_features, ax.flatten()):

    sns.regplot(x=train[variable], y=train['revenue'], ax=subplot)
plt.figure(figsize=(45,25))

sns.heatmap(train.corr(),annot=True)
fig, ax = plt.subplots(3, 1, figsize=(40, 30))

for var, subplot in zip(categorical_features, ax.flatten()):

    sns.boxplot(x=var, y='revenue', data=train, ax=subplot)
# Not Applicable 

## STudy relation between P2 and revenue in light of Categorical Variables

#cond_plot = sns.FacetGrid(data=train, col='City', hue='Type', col_wrap=4)

#cond_plot.map(sns.scatterplot, 'P2', 'revenue').add_legend();

#cond_plot = sns.FacetGrid(data=train, col='City Group', hue='Type', col_wrap=4)

#cond_plot.map(sns.scatterplot, 'P2', 'revenue').add_legend();
cats = train["City"].unique()

cats
tem = train.copy()

#cats = train["City"].unique().tolist() 

#fig, ax = plt.subplots(34, 1, figsize=(25, 400))

cats = ['İstanbul','İzmir','Ankara','Bursa','Samsun']

fig, ax = plt.subplots(5, 1, figsize=(25, 45))

for variable, subplot in zip(cats, ax.flatten()):

    #x = tem.where(train["City"]==variable, inplace = False)

    x = tem[train["City"]==variable]

    x = x.sort_values(by=['Open Date'])

    if len(x) <= 4:        

        g = sns.barplot(x="Open Date", y="revenue",hue="Type", data=x, ax=subplot)

        g.title.set_text(variable)

    else:

        g = sns.lineplot(x="Open Date", y="revenue", style = "Type",label=variable, linestyle="-", data=x, ax=subplot)

        g.title.set_text(variable)

        for label in subplot.get_xticklabels():

            label.set_rotation(90)
from sklearn import preprocessing

##from sklearn.model_selection import train_test_split
## as can be seen that test has more rows than the training dataset 

## as the viuslization clearly express no clear relation in the train

num_train = train.shape[0]

num_test = test.shape[0]

print(num_train, num_test)



# For feature engineering, combine train and test data

data = pd.concat((train.loc[:, "Id" : "P37"],

                  test.loc[:, "Id" : "P37"]), ignore_index=True)
# Plotting mean of P-variables over each city helps us see which P-variables are highly related to City

# since we are given that one class of P-variables is geographical attributes.

distinct_cities = train.loc[:, "City"].unique()



# Get the mean of each p-variable for each city

means = []

for col in train.columns[5:42]:

    temp = []

    for city in distinct_cities:

        temp.append(train.loc[train.City == city, col].mean())     

    means.append(temp)

    

# Construct data frame for plotting

city_pvars = pd.DataFrame(columns=["city_var", "means"])

for i in range(37):

    for j in range(len(distinct_cities)):

        city_pvars.loc[i+37*j] = ["P"+str(i+1), means[i][j]]

#print(city_pvars)        

# Plot boxplot

plt.rcParams['figure.figsize'] = (18.0, 6.0)

sns.boxplot(x="city_var", y="means", data=city_pvars)
from sklearn import cluster

# K Means treatment for city (mentioned in the paper)

def adjust_cities(data, train, k):

    

    # As found by box plot of each city's mean over each p-var

    relevant_pvars =  ["P1", "P2", "P11", "P19", "P20", "P23", "P30"]

    train = train.loc[:, relevant_pvars]

    

    # Optimal k is 20 as found by DB-Index plot    

    kmeans = cluster.KMeans(n_clusters=k)

    kmeans.fit(train)

    

    # Get the cluster centers and classify city of each data instance to one of the centers

    data['City Cluster'] = kmeans.predict(data.loc[:, relevant_pvars])

    del data["City"]

    

    return data



def one_hot_ecoding(data,col,pref):

    # One hot encode City Group

    data = data.join(pd.get_dummies(data[col], prefix=pref))

    # Since only n-1 columns are needed to binarize n categories, drop one of the new columns.  

    # And drop the original columns.

    data = data.drop([col], axis=1)

    return data 
# Convert unknown cities in test data to clusters based on known cities using KMeans

data = adjust_cities(data, train, 20)

#data = data.drop(['City'], axis=1)

data = one_hot_ecoding(data,'City Group',"CG")

data = one_hot_ecoding(data,'Type',"T")
data.dtypes
# Split into train and test datasets

train_processed = data[:num_train]

test_processed = data[num_train:]

# check the shapes 

print("Train :",train.shape)

print("Test:",test.shape)
#sav_train = pd.DataFrame()

#sav_train = train["revenue"].copy()
train["revenue"] = [np.log(num) for num in train["revenue"]]

len(train["revenue"])
train_processed["revenue"] = train["revenue"].values ## if you do not put this values , then you are in complete danger

#train_processed["Id"] = train["Id"]

#test_processed["Id"] = test["Id"]

train = train_processed

test = test_processed
# check the shapes 

print("Train :",train.shape)

print("Test:",test.shape)
import time

from datetime import datetime as dt



## prepartion function 

def prepare_data_frame(dataframe, target):

    df = dataframe.copy()

    ## Splittin the date column into three columns 

    df['Open Date Year']  = df['Open Date'].dt.year

    df['Open Date Month']  = df['Open Date'].dt.month

    df['Open Date Day']  = df['Open Date'].dt.day

    ##---------------------------------------

    ## feature Engineering Begining 

    ##--------------------------------------

    ## feature Engineering a diff column 

    ## since when this resturant was opened

    all_diff = []

    for date in df["Open Date"]:

        diff = dt.now() - date

        all_diff.append(int(diff.days/1000))

    df['Days_from_open'] = pd.Series(all_diff)

    ##---------------------------------------

    ## feature Engineering End 

    ##--------------------------------------

    # drop Open Date column s

    df = df.drop(['Open Date'], axis=1)

    # drop target column s

    if target in df.columns:

        tar = df[target]

        df = df.drop([target], axis=1)

    else:

        tar = None

    # get numberical variables 

    num_vars = df.select_dtypes([np.number]).columns.tolist()

    # encoding categrical variables 

    #categorical variables already encoded in one hot encoding

    # get cat variables 

    cat_vars = df.select_dtypes(include='object').columns.tolist()

    df[cat_vars] = df[cat_vars].apply(preprocessing.LabelEncoder().fit_transform)

    #df.loc[:, "P1" :"P37"] = preprocessing.MinMaxScaler().fit_transform(df.loc[:, "P1" : "P37"])

    return (df,tar)
X_train,y_train = prepare_data_frame(train,'revenue')

X_test,y_test = prepare_data_frame(test,'revenue') ## there is no revnue in test\
X_train.shape
y_train.shape
X_test.shape
#! pip install tensorflow==2.0.0-beta1
#install tensorflow if not installed

## import tensorflow 

import tensorflow as tf

from tensorflow.keras import layers

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import cross_val_predict



from xgboost import XGBRegressor

#from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import GridSearchCV

from sklearn.svm import SVR

from sklearn.ensemble import RandomForestRegressor

from sklearn.linear_model import LassoCV

import warnings

warnings.filterwarnings('ignore')
def get_reg_mse(model,in_parameters):

    # Define the model

    my_model_1 = model # Your code here

    clf = GridSearchCV(my_model_1, in_parameters, cv=2, scoring='neg_mean_squared_error')

    # Fit the model

    clf.fit(X_train, y_train) # Your code here

    #predictions_1 = clf.predict(X_valid) # Your code here

    #mae_1 = mean_absolute_error(y_true=y_valid, y_pred=predictions_1) # Your code here

    print('best_params_',clf.best_params_)

    mae_1 = clf.best_score_ * -1

    return mae_1
def baseline_model():

    dim = len(X_train.columns.tolist())

    if not isinstance(dim, int):

        return 0

    model = tf.keras.Sequential([

        layers.Dense(dim, input_dim=dim, kernel_initializer='normal', activation='relu'),

        layers.Dense(int(round(dim/2)), kernel_initializer='normal', activation='relu'), ## 

        layers.Dense(int(round(dim/4)), kernel_initializer='normal', activation='relu'), ## 

        layers.Dense(int(round(dim/8)), kernel_initializer='normal', activation='relu'), ## 

        layers.Dense(1, kernel_initializer='normal')

    ])

    #model.compile(loss='mean_squared_error', optimizer='adam')111

    model.compile(optimizer='adam',

                  loss='mse',

                  metrics=['mse'])

    return model
from sklearn.model_selection import cross_val_score, KFold

from sklearn.preprocessing import StandardScaler



def get_reg_keras_mse(in_parameters):

    # Define the model

    # evaluate model with standardized dataset

    my_model_1 = tf.keras.wrappers.scikit_learn.KerasRegressor(build_fn=baseline_model, verbose=1)

    reg = GridSearchCV(my_model_1, in_parameters, cv=2, scoring='neg_mean_squared_error')

    X_ktrain = StandardScaler().fit_transform(X_train.values)  

    arr = sav_train['rev_save'].values 

    y_ktrain = StandardScaler().fit_transform(arr[:, np.newaxis])

    

    reg.fit(X_ktrain, y_train.values) # Your code here

    # Fit the model

    print('best_params_',reg.best_params_)

    mse_1 = reg.best_score_ * -1

    return mse_1



# A utility method to create a tf.data dataset from a Pandas Dataframe

def df_to_dataset(dataframe, shuffle=True, batch_size=32, target_name='target'):

    dataframe = dataframe.copy()

    labels = dataframe.pop(target_name)

    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))

    if shuffle:

        ds = ds.shuffle(buffer_size=len(dataframe))

    ds = ds.batch(batch_size)

    return ds



def input_fn(features, labels, training=True, batch_size=32):

    """An input function for training or evaluating"""

    # Convert the inputs to a Dataset.

    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))



    # Shuffle and repeat if you are in training mode.

    if training:

        #dataset = dataset.shuffle(1000).repeat()

        dataset = dataset.shuffle(buffer_size=len(features))

    

    return dataset.batch(batch_size)
#Parameters for tuning model 

#n_esitmators = list(range(100, 1001, 100))

#learning_rates = [x / 100 for x in range(5, 101, 5)]

#parameters = {'n_esitmators':n_esitmators, 'learning_rates':learning_rates}

#mse_1 = get_reg_mse(XGBRegressor(),parameters)

# Uncomment to print MAE

#print("Mean Squared Error:" , mse_1)
#c = list(np.arange(1, 10, 0.1))

#cache_size=list(range(100, 1001, 100))

#parameters = {'kernel':('linear', 'rbf'), 'C':c,'cache_size':cache_size}

#mse_1 = get_reg_mse(SVR(gamma='scale'),parameters)

# Uncomment to print MAE

#print("Mean Squared Error:" , mse_1)
#n_estimators=list(range(100, 501, 100))

#max_depth = list(range(1, 5, 1))

#parameters = {'n_estimators':n_estimators,'max_depth':max_depth}

#mse_1 = get_reg_mse(RandomForestRegressor(random_state=400),parameters)

# Uncomment to print MAE

#print("Mean Squared Error:" , mse_1)
#random_state=list(range(0, 100, 1))

#parameters = {'random_state':random_state}

#mse_1 = get_reg_mse(LassoCV(cv=2),parameters)

# Uncomment to print MAE

#print("Mean Squared Error:" , mse_1)
#seed = 1

#epochs=list(range(100, 101, 100))

#batch_size = [1,5]

#parameters = {'epochs':epochs,'batch_size':batch_size}

#mse_1 = get_reg_keras_mse(parameters)

# Uncomment to print MAE

#print("Mean Squared Error:" , mse_1)
samp = pd.read_csv('/kaggle/input/restaurant-revenue-prediction/sampleSubmission.csv')

samp.head()
# other regressors

#reg = RandomForestRegressor(random_state = 400, max_depth = 1, n_estimators = 500)

#reg.fit(X_train,y_train)

#print (reg)
## keras regressor

#reg = tf.keras.wrappers.scikit_learn.KerasRegressor(build_fn=baseline_model,batch_size=5,epochs=100, verbose=1)

#X_ktrain = StandardScaler().fit_transform(X_train.values)  

#arr = sav_train['rev_save'].values 

#y_ktrain = StandardScaler().fit_transform(arr[:, np.newaxis])

#reg.fit(X_ktrain,y_ktrain)

#طprint (reg)
## deal with null values in test dataset

#cols3 = X_test.columns[X_test.isna().any()].tolist()

#print(cols3)

#X_test['Days_from_open'] = X_test['Days_from_open'].fillna(int(round(X_test['Days_from_open'].mean())))
#Predict 

#id_vals = X_test['Id'].values

#output = reg.predict(X_test) #data=X_test for other Regressors

#output = np.exp(output)

#final_df = pd.DataFrame()

#final_df["Id"] = id_vals

#final_df["Prediction"] = output.round(1)

#final_df.to_csv("Output_Keras.csv", index=False)

#print('Check for Na : ',final_df.isna().sum())

#print('Check for Inf : ',np.isfinite(final_df).sum())
final_df.head()