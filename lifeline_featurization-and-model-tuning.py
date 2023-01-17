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
import seaborn as sns

import matplotlib.pyplot as plt

sns.set(color_codes=True)

%matplotlib inline

import warnings

warnings.filterwarnings('ignore')



class color:

   PURPLE = '\033[95m'

   CYAN = '\033[96m'

   DARKCYAN = '\033[36m'

   BLUE = '\033[94m'

   GREEN = '\033[92m'

   YELLOW = '\033[93m'

   RED = '\033[91m'

   BOLD = '\033[1m'

   UNDERLINE = '\033[4m'

   END = '\033[0m'
ds = pd.read_csv('../input/regression-with-neural-networking/concrete_data.csv');

ds.info()

ds.shape
#Check for missing values

print(ds.isna().sum())

print(ds.isnull().sum())
#Little more statistical understanding

ds.describe().T
#Let's rename the columns for better handling

ds.rename(columns = {'Strength': 'strength', 'Cement': 'cement', 'Blast Furnace Slag': 'slag', 'Fly Ash': 'ash' ,'Water': 'water', 'Superplasticizer': 'superplastic', 'Coarse Aggregate': 'coarseagg' ,'Fine Aggregate': 'fineagg', 'Age': 'age' }, inplace=True)
# Let's split the data before doing further analysis:

# train test split 70:30

#Remove the target varible

target = ds[['strength']]

data_set_buffer = ds.drop('strength',axis=1)



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(data_set_buffer, target, test_size = 0.3, random_state = 123)
data_set_buffer.tail()
sns.relplot(x="cement", y="strength",  data=ds);

sns.relplot(x="slag", y="strength",  data=ds);

sns.relplot(x="ash", y="strength",  data=ds);

sns.relplot(x="water", y="strength",  data=ds);

sns.relplot(x="superplastic", y="strength",  data=ds);

sns.relplot(x="coarseagg", y="strength",  data=ds);

sns.relplot(x="fineagg", y="strength",  data=ds);

sns.relplot(x="age", y="strength",  data=ds);

#Looks like superplastic and cement have +ve correlation with the strength rest all are not straight forward to infer.
sns.relplot(x="cement", y="strength", alpha=.5, palette="muted",

            height=6, data=ds)

#distplot

sns.relplot(x="water", y="strength", alpha=.5, palette="muted",

            height=6, data=ds)



#histogram

sns.relplot(x="slag", y="strength", alpha=.5, palette="muted",

            height=6, data=ds)



sns.relplot(x="ash", y="strength", alpha=.5, palette="muted",

            height=6, data=ds)

# Let's analyse the strength against water and fly-ash level



# create figure and axis objects with subplots()

fig,ax = plt.subplots()

# make a plot

ax.scatter(ds.strength,ds.ash,  color="red", marker=",")

# set x-axis label

ax.set_xlabel("Strength",fontsize=14)

# set y-axis label

ax.set_ylabel("Ash",color="red",fontsize=14)



# twin object for two different y-axis on the sample plot

ax2=ax.twinx()

# make a plot with different y-axis using second axis object

ax2.scatter(ds.strength,ds.water,color="blue",marker="+")

ax2.set_ylabel("Water",color="blue",fontsize=14)

plt.show()

# save the plot as a file

sns.relplot(x="water", y="ash", alpha=.5, palette="muted",

            height=6, data=ds)
from pandas.plotting import scatter_matrix



scatter_matrix(ds.loc[:],figsize=(12, 12),diagonal="kde")

plt.tight_layout()

plt.show()
ds.corr().T
#Strength is the target column while rest are our input variable.

#Let's do box plot to find the outliers if there any

plt.figure(figsize=(35,15))

sns.boxplot(data=ds)
import itertools

plt.figure(figsize=(10,20))

j=0;

for i, c in zip(ds.columns , list(ds.columns.values) ):

    j=j+1

    plt.subplot(3,3,j)

    sns.swarmplot( y = ds[i]);

    plt.title= c

    plt.show
# Let's replace the outliers with median values

import numpy as np

outliers =  []

from scipy import stats

zscore = ds.apply(stats.zscore)

ds_columns = ds.columns.values

for key, value in zscore.iteritems():

    row=0

    medianV = np.median(ds[[key]])

    columnnum = ds.columns.get_loc(key)

    for v in value:

        if v>3 or v < -3 :

            outliers.append(v)

            ds.iloc[row,columnnum] = medianV

            print (ds.iloc[row,columnnum] )

        row = row + 1

     

print("outlier size ", len(outliers))
from sklearn.tree import DecisionTreeRegressor

dtr_model = DecisionTreeRegressor()

dtr_model.fit(X_train , y_train)
print('The feature importances starting from hightest:')

for (i, j) in zip(X_train.columns , reversed(sorted(dtr_model.feature_importances_))): 

    print(i,j)
print(dtr_model.score(X_train, y_train))

print(dtr_model.score(X_test, y_test))
# Regularizing the Decision tree classifier to avoid over fitting situation described above

reg_dtr_model = DecisionTreeRegressor( max_depth = 5,random_state=1,min_samples_leaf=6)

reg_dtr_model.fit(X_train, y_train)
print(reg_dtr_model.score(X_train, y_train))

print(reg_dtr_model.score(X_test, y_test))
print('The feature importances:')

for (i, j) in zip(X_train.columns ,reversed(sorted(reg_dtr_model.feature_importances_))): 

    print(i,j)
import matplotlib.pyplot as plt



# Pie chart, where the slices will be ordered and plotted counter-clockwise:

labels = X_train.columns

sizes = reg_dtr_model.feature_importances_

fig1, ax1 = plt.subplots()

wedges, texts, autotexts = ax1.pie(sizes, labels=labels, autopct='%1.01f%%',

        shadow=True, startangle=90,pctdistance=1.2, labeldistance=1.5 )

ax1.axis('equal') 

ax1.legend(wedges, labels,

          title="Feature Importance:",

          loc="center left",

          bbox_to_anchor=(1, 0, 0.5, 1))

plt.setp(autotexts, size=10, weight="bold")



ax1.set_title("Feature Importance Pie")
#Let's confirm the correlation with correlation matrix

#Find correlation

plt.figure(figsize=(10,10))

corr =train_data_Set.corr()

matrix = np.triu(corr)

ax=sns.heatmap(corr, annot=True, mask=matrix,cmap= sns.diverging_palette(20, 220, n=200), linewidths=2,linecolor='white',square=True)

plt.show()
#Let's drop the coarseagg column and use important features only to analyse further.

data_set_buffer_feature_imprtance = ds.drop('coarseagg',axis=1)

X_train, X_test, y_train, y_test = train_test_split(data_set_buffer_feature_imprtance, target, test_size = 0.3, random_state = 123)
import statsmodels.api as sm

lr_1 = sm.OLS(y_train, X_train).fit()
y_pred = lr_1.predict(X_test)
from sklearn import metrics

metrics.explained_variance_score(y_test, y_pred)
from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

# Import Linear Regression machine learning library

from sklearn.linear_model import LinearRegression

from sklearn.linear_model import Ridge

from sklearn.linear_model import Lasso

num_folds = 50

seed = 7



kfold = KFold(n_splits=num_folds, random_state=seed)

model = LinearRegression()

results = cross_val_score(model, data_set_buffer, target, cv=kfold)

print(results)

print("Accuracy: %.3f%% (%.3f%%)" % (results.mean()*100.0, results.std()*100.0))

print(color.RED)

print(color.BOLD)

print('Accuracy Score')

print(color.BLUE)

print('Average: ', results.mean())

print('Standard deviation ',results.std())
data_set_buffer_removed_outlier = data_set_buffer

zscore = data_set_buffer_removed_outlier.apply(stats.zscore)

ds_columns = data_set_buffer_removed_outlier.columns.values

for key, value in zscore.iteritems():

    row=0

    medianV = np.median(data_set_buffer[[key]])

    columnnum = data_set_buffer_removed_outlier.columns.get_loc(key)

    for v in value:

        if v>3 or v < -3 :

            data_set_buffer_removed_outlier.iloc[row,columnnum] = medianV

        row = row + 1



target_removed_outlier = target

zscore = target_removed_outlier.apply(stats.zscore)

ds_columns = target_removed_outlier.columns.values

for key, value in zscore.iteritems():

    row=0

    medianV = np.median(target[[key]])

    columnnum = target_removed_outlier.columns.get_loc(key)

    for v in value:

        if v>3 or v < -3 :

            target_removed_outlier.iloc[row,columnnum] = medianV

        row = row + 1

num_folds = 6

seed = 7



kfold = KFold(n_splits=num_folds, random_state=seed)

model = LinearRegression()

results = cross_val_score(model, data_set_buffer_removed_outlier, target_removed_outlier, cv=kfold)

print(results)

print("Accuracy: %.3f%% (%.3f%%)" % (results.mean()*100.0, results.std()*100.0))

print(color.RED)

print(color.BOLD)

print('Accuracy Score after replacing outliers with median')

print(color.BLUE)

print('Average: ', results.mean())

print('Standard deviation ',results.std())
ridge = Ridge(alpha=.3)

ridge.fit(X_train,y_train)

print ("Ridge model:", (ridge.coef_))
lasso = Lasso(alpha=0.1)

lasso.fit(X_train,y_train)

print ("Lasso model:", (lasso.coef_))
print(lasso.score(X_train, y_train))

print(lasso.score(X_test, y_test))
print(ridge.score(X_train, y_train))

print(ridge.score(X_test, y_test))
from sklearn import preprocessing

from sklearn.preprocessing import PolynomialFeatures



X_scaled = preprocessing.scale(data_set_buffer_removed_outlier)

X_scaled = pd.DataFrame(X_scaled, columns=data_set_buffer_removed_outlier.columns)  



y_scaled = preprocessing.scale(y_train)

y_scaled = pd.DataFrame(y_scaled, columns=target.columns)  



poly = PolynomialFeatures(degree = 2, interaction_only=True)

X_poly = poly.fit_transform(X_scaled)

X_train_poly, X_test_poly, y_train_poly, y_test_poly = train_test_split(X_poly, target, test_size=0.30, random_state=1)

X_train_poly.shape



from sklearn.linear_model import LinearRegression

regression_model = LinearRegression()

regression_model.fit(X_train_poly, y_train_poly)

print(regression_model.coef_[0])

ridge = Ridge(alpha=.3)

ridge.fit(X_train_poly,y_train_poly)

print ("Ridge model:", (ridge.coef_))
print(ridge.score(X_train_poly, y_train_poly))

print(ridge.score(X_test_poly, y_test_poly))

lasso = Lasso(alpha=0.01)

lasso.fit(X_train_poly,y_train_poly)

print ("Lasso model:", (lasso.coef_))

print(lasso.score(X_train_poly, y_train_poly))

print(lasso.score(X_test_poly, y_test_poly))
regression_model = LinearRegression()

regression_model.fit(X_train, y_train)

y_pred = regression_model.predict(X_test)

from sklearn import metrics

metrics.explained_variance_score(y_test, y_pred)
from sklearn.ensemble import RandomForestRegressor

randomf = RandomForestRegressor(n_estimators=10, max_depth = 3)

randomf.fit(X_train, y_train)

randomf.score(X_test, y_test)
from sklearn.ensemble import GradientBoostingRegressor

gbr = GradientBoostingRegressor()

gbr.fit(X_train, y_train)

gbr.score(X_test, y_test)
from sklearn.neighbors import KNeighborsRegressor

knnr = KNeighborsRegressor(n_neighbors=10)

knnr.fit(X_train, y_train)

knnr.score(X_test, y_test)