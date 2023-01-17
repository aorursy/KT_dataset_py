import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

from scipy.stats import norm

from scipy import stats

import warnings

warnings.filterwarnings('ignore')

%matplotlib inline

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

# Input data files are available in the "../input/" directory on kaggle.

#Data importing

data = pd.read_csv('../input/data.csv')

data.reset_index(inplace=True)
data.shape
data.dtypes
data.isnull().any()
#horsepower and acceleration features are not cleaned because of null values

horsepower_nan_index = data[data['horsepower'].isnull()].index

acceleration_nan_index = data[data['acceleration'].isnull()].index
data.loc[horsepower_nan_index]
data.loc[acceleration_nan_index]
#Replacing NaN with mean value so that other features shoud be contributing in calculations



data.loc[horsepower_nan_index, 'horsepower'] = round(data.horsepower.mean(),1)
data.loc[horsepower_nan_index]
data.shape
data.loc[acceleration_nan_index, 'acceleration'] = round(data.acceleration.mean(), 1)
data.loc[acceleration_nan_index]
data.horsepower.unique()
data['model year'].unique()
#181 and 182 year are not possible hence we need to remove rows corresponding to those years

data = data[~data['model year'].isin(['181','182'])]
data['model year'].unique()
data['cylinders'].unique()
#Clearly 80,60 and 400 cylinders are outliers as those are impossible

data = data[~data['cylinders'].isin(['80','60','400'])]
data['cylinders'].unique()
data.shape
#data.acceleration.unique()

data.dtypes
data=data.drop(columns=['index'])
data.shape
data=data.set_index('car name') #setting "car name as index"
data.shape
data.dtypes
data.describe()
data.mpg.describe()
#So the minimum value is 9 and maximum is 46, but on average it is 23.51 with a variation of 7.8

sns.distplot(data['mpg'])
print("Skewness: %f" % data['mpg'].skew())

print("Kurtosis: %f" % data['mpg'].kurt())
#Lets visualise some relationships between these data points, but before we do, we need to scale them to same the same range of [0,1]

#Normalization of data

def normalize(a):

    b = (a-a.min())/(a.max()-a.min())

    return b

data_scale = data.copy()
data_scale ['displacement'] = normalize(data_scale['displacement'])

data_scale['horsepower'] = normalize(data_scale['horsepower'])

data_scale ['acceleration'] = normalize(data_scale['acceleration'])

data_scale ['weight'] = normalize(data_scale['weight'])

data_scale['mpg'] = normalize(data_scale['mpg'])
data_scale.head()
#All our data is now scaled to the same range of [0,1]. This will help us visualize data better. We used a copy of the original data-set for this as we will use the data-set later when we build regression models.

data['Country_code'] = data.origin.replace([1,2,3],['USA','Europe','Japan'])

data_scale['Country_code'] = data.origin.replace([1,2,3],['USA','Europe','Japan'])
data_scale.head()
#Lets look at MPG's relation to categories
var = 'Country_code'

data_plt = pd.concat([data_scale['mpg'], data_scale[var]], axis=1)

f, ax = plt.subplots(figsize=(8, 6))

fig = sns.boxplot(x=var, y="mpg", data=data_plt)

fig.axis(ymin=0, ymax=1)

plt.axhline(data_scale.mpg.mean(),color='r',linestyle='dashed',linewidth=2)
#Conclusion : -The red line marks the average of the set. From the above plot we can observe:



#Majority of the cars from USA (almost 75%) have MPG below global average.

#Majority of the cars from Japan and Europe have MPG above global average.
#Let's look at the year wise distribution of MPG
var = 'model year'

data_plt = pd.concat([data_scale['mpg'], data_scale[var]], axis=1)

f, ax = plt.subplots(figsize=(8, 6))

fig = sns.boxplot(x=var, y="mpg", data=data_plt)

fig.axis(ymin=0, ymax=1)

plt.axhline(data_scale.mpg.mean(),color='r',linestyle='dashed',linewidth=2)
#And MPG distribution for cylinders
var = 'cylinders'

data_plt = pd.concat([data_scale['mpg'], data_scale[var]], axis=1)

f, ax = plt.subplots(figsize=(8, 6))

fig = sns.boxplot(x=var, y="mpg", data=data_plt)

fig.axis(ymin=0, ymax=1)

plt.axhline(data_scale.mpg.mean(),color='r',linestyle='dashed',linewidth=2)
#Now that we have looked at the distribution

corrmat = data.corr()

f, ax = plt.subplots(figsize=(12, 9))

sns.heatmap(corrmat, square=True);
factors = ['cylinders','displacement','horsepower','acceleration','weight','mpg']

corrmat = data[factors].corr()

f, ax = plt.subplots(figsize=(12, 9))

sns.heatmap(corrmat, square=True);
#So far, we have seen the data to get a feel for it, we saw the spread of the desired variable MPG along the various discrete variables, namely, Origin, Year of Manufacturing or Model and Cylinders.

#Now lets extract an additional discrete variable company name and add it to this data. We will use regular expressions and str.extract() function of pandas data-frame to make this new column

data.index
#Lets look at some extremes

var='mpg'

data[data[var]== data[var].min()]
data[data[var]== data[var].max()]
var='displacement'

data[data[var]== data[var].min()]
data[data[var]== data[var].max()]
var='horsepower'

data[data[var]== data[var].min()]
data[data[var]== data[var].max()]
var='weight'

data[data[var]== data[var].min()]
data[data[var]== data[var].max()]
var='acceleration'

data[data[var]== data[var].min()]
data[data[var]== data[var].max()]
#Now that we have looked at the distribution of the data along discrete variables and we saw some scatter-plots using the seaborn pairplot. Now let's try to find some logical causation for variations in mpg. We will use the lmplot() function of seaborn with scatter set as true. This will help us in understanding the trends in these relations. We can later verify what we see with ate correlation heat map to find if the conclusions drawn are correct. We prefer lmplot() over regplot() for its ability to plot categorical data better. We will split the regressions for different origin countries.

var = 'horsepower'

plot = sns.lmplot(var,'mpg',data=data,hue='Country_code')

plot.set(ylim = (0,50))
var = 'displacement'

plot = sns.lmplot(var,'mpg',data=data,hue='Country_code')

plot.set(ylim = (0,50))
var = 'weight'

plot = sns.lmplot(var,'mpg',data=data,hue='Country_code')

plot.set(ylim = (0,50))
var = 'acceleration'

plot = sns.lmplot(var,'mpg',data=data,hue='Country_code')

plot.set(ylim = (0,50))
data['Power_to_weight'] = ((data.horsepower*0.7457)/data.weight)
data.sort_values(by='Power_to_weight',ascending=False ).head()
# So far, we have a looked at our data using various pandas methods and visualized it using seaborn package. We looked at



# MPGs relation with discrete variables

# MPG distribution over given years if manufacturing

# MPG distribution by country of origin

# MPG distribution by number of cylinders

# MPGs relation to other continuous variables:

# Pair wise scatter plot of all variables in data. ### Correlation

# We looked at the correlation heat map of all columns in our data

# Lets look at some regression models:

# Now that we know what our data looks like, lets use some machine learning models to predict the value of MPG given the values of the factors. We will use pythons scikit learn to train test and tune various regression models on our data and compare the results. We shall use the following regression models:-



# Linear Regression



# GBM Regression
data.head()
from sklearn.preprocessing import StandardScaler

from sklearn.tree import DecisionTreeRegressor

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error

from math import sqrt

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import KFold
factors = ['cylinders','displacement','horsepower','acceleration','weight','origin','model year']

#independent variables

X = pd.DataFrame(data[factors].copy())

#output variable

y = data['mpg'].copy()
X = StandardScaler().fit_transform(X)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size = 0.33,random_state=324)

X_train.shape[0] == y_train.shape[0]
regressor = LinearRegression()
regressor.get_params()
regressor.fit(X_train,y_train)
y_predicted = regressor.predict(X_test)
rmse = sqrt(mean_squared_error(y_true=y_test,y_pred=y_predicted))

rmse
gb_regressor = GradientBoostingRegressor(n_estimators=4000)

gb_regressor.fit(X_train,y_train)
gb_regressor.get_params()
y_predicted_gbr = gb_regressor.predict(X_test)
rmse_bgr = sqrt(mean_squared_error(y_true=y_test,y_pred=y_predicted_gbr))

rmse_bgr
fi= pd.Series(gb_regressor.feature_importances_,index=factors)

fi.plot.barh()
#Good, so our initial models work well, but these metrics were performed on test set and cannot be used for tuning the model, as that will cause bleeding of test data into training data, hence, we will use K-Fold to create Cross Validation sets and use grid search to tune the model.

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca.fit(data[factors])
pca.explained_variance_ratio_
pca1 = pca.components_[0]

pca2 = pca.components_[1]
transformed_data = pca.transform(data[factors])
pc1 = transformed_data[:,0]

pc2 = transformed_data[:,1]
plt.scatter(pc1,pc2)
c = pca.inverse_transform(transformed_data[(transformed_data[:,0]>0 )& (transformed_data[:,1]>250)])
factors
c
data[(data['model year'] == 70 )&( data.displacement>400)]
#This seems logical as the weight data given in the data set seems to be incorrect. The weight for the vehicle is given to be 3086 lbs, however, on research it can be found that the car weight is 4727-4775 lbs. These values are based on internet search



# Now we use K-fold to create a new K-fold object called 'cv_sets' that contains index values for training and cross validation and use these sets in GridSearchCV to tune our model so that it does not over fit or under fit the data

# We will also define a dictionary called 'params' with the hyper-parameters we want to tune

# Lastly we define 'grid' which is a GridSearchCV object which we will provide the parameters to tune and the K folds of data created by using the Kfold in sklearn.model_selection

cv_sets = KFold(n_splits=10, shuffle= True,random_state=100)

params = {'n_estimators' : list(range(40,61)),

         'max_depth' : list(range(1,10)),

         'learning_rate' : [0.1,0.2,0.3] }

grid = GridSearchCV(gb_regressor, params,cv=cv_sets,n_jobs=4)
grid = grid.fit(X_train, y_train)

grid
grid.best_estimator_
gb_regressor_t = grid.best_estimator_
gb_regressor_t.fit(X_train,y_train)
gb_regressor_t
y_predicted_gbr_t = gb_regressor_t.predict(X_test)
y_predicted_gbr_t
rmse = sqrt(mean_squared_error(y_true=y_test,y_pred=y_predicted_gbr_t))

rmse
data.duplicated().any()