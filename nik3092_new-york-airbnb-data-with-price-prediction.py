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

plt.rcParams['figure.figsize'] = 15,10

import missingno as msno



import warnings

warnings.filterwarnings('ignore')
df = pd.read_csv('/kaggle/input/new-york-city-airbnb-open-data/AB_NYC_2019.csv')

df.head()
df.shape
df.info()
sns.pairplot(df)
msno.matrix(df)
print("The Total Missing values presents in \"last_review\" columns is: {}".format(df['last_review'].isnull().sum()))

print("The Total Missing values presents in \"reviews_per_month\" columns is: {}".format(df['reviews_per_month'].isnull().sum()))
df.isnull().sum()
df = df.drop(['id','name','host_name','last_review','reviews_per_month'],axis=1)

df.head()
sns.set_context("talk",font_scale=1.0)

sns.scatterplot('neighbourhood','price',data = df.sort_values('price',ascending=False).head(20))

plt.grid()

plt.title("Neighbourhood Vs Price",color='magenta')

plt.xlabel("Neighbourhood",color='r')

plt.ylabel("Price($)",color='r')

plt.xticks(rotation=90)
sns.set_context("talk",font_scale=1.0)

ax=sns.barplot('neighbourhood','price',data = df.nlargest(20,'price'),ci=None)

plt.xlabel("Neighbourhood",color='r')

plt.ylabel("Price($)",color='r')

plt.xticks(rotation=90)
host = df['host_id'].value_counts().head(15)

host
fig_1 = host.plot(kind='bar')

fig_1.set_title("Top 15 Host with Listing in NYC")

fig_1.set_xlabel('Host_ID')

fig_1.set_ylabel('Count of Listing')

fig_1.set_xticklabels(fig_1.get_xticklabels(),rotation=45)

plt.show()
df['neighbourhood_group'].unique()
sns.set_context('talk')

ax=sns.barplot(x = df['neighbourhood_group'],y = df['price'],ci=None)

plt.xlabel("Neighbourhood_Group",color='r')

plt.ylabel("Price($)",color='g')

plt.title("Prices in Major Region",color='magenta',size=15)



for p in ax.patches:

    ax.annotate(int(p.get_height()), (p.get_x()+0.30, p.get_height()+1), va='bottom',

                    color= 'black')
sns.set_context('poster')

ax = sns.barplot(df['room_type'],df['price'],ci=None)

plt.xlabel("Room Types",color='r')

plt.ylabel("Price($)",color='r')

plt.title("Prices for Each RoomType",color='magenta',size=15)



for p in ax.patches:

    ax.annotate(int(p.get_height()), (p.get_x()+0.30, p.get_height()-0.05), va='bottom',

                    color= 'black')
sns.violinplot(df['room_type'],df['availability_365'])

plt.xlabel("Room Types")

plt.ylabel("Availability")

plt.title("Room Availability",color = 'magenta')
sns.set_context('talk')

sns.violinplot(df['neighbourhood_group'],df['availability_365'])

plt.xlabel("Neighbourhood Group")

plt.ylabel("Availability")

plt.title("Availability in each Region",color = 'magenta')
sns.scatterplot(x=df['latitude'],y=df['longitude'],hue=df['neighbourhood_group'],data=df,palette='Set2')
sns.set_style('white')

sns.scatterplot(x=df['latitude'],y=df['longitude'],hue=df['room_type'],data=df,sizes='size',

               markers="markers")
sns.set_style('darkgrid')

sns.scatterplot(x=df['latitude'],y=df['longitude'],hue=df['availability_365'],data=df,markers="markers")
df.columns
df = df.drop(['host_id','neighbourhood','calculated_host_listings_count'],axis=1)

df.head()
print(df['room_type'].unique())

print(df['neighbourhood_group'].unique())



cols = ['room_type','neighbourhood_group']
from sklearn.preprocessing import LabelEncoder

end = LabelEncoder()

for col in cols:

    df[col] = end.fit_transform(df[col])

    mapping = dict(zip(end.classes_,end.transform(end.classes_)))

    print("column : ", col)

    print("Mapping is : ", mapping)

    

df.head()
df.boxplot(rot=45)
print("Latitude")

print(30*'-')

print(df['latitude'].quantile(0.25))

print(df['latitude'].quantile(0.75))

print('\n')



print('Longitude')

print(30*'-')

print(df['longitude'].quantile(0.25))

print(df['longitude'].quantile(0.75))

print('\n')



print('Minimum Nights')

print(30*'-')

print(df['minimum_nights'].quantile(0.25))

print(df['minimum_nights'].quantile(0.75))

print('\n')



print("Number of reviews")

print(30*'-')

print(df['number_of_reviews'].quantile(0.25))

print(df['number_of_reviews'].quantile(0.75))

print('/n')



print('Price')

print(30*'-')

print(df['price'].quantile(0.25))

print(df['price'].quantile(0.75))
df['latitude'] = np.where(df['latitude']<40.6901,40.6901,df['latitude'])

df['latitude'] = np.where(df['latitude']>40.763115,40.763115,df['latitude'])



df['longitude'] = np.where(df['longitude']<-73.98307,-73.98307,df['longitude'])

df['longitude'] = np.where(df['longitude']> -73.936275,-73.936275,df['longitude'])



df['minimum_nights'] = np.where(df['minimum_nights'] < 1.0,1.0,df['minimum_nights'])

df['minimum_nights'] = np.where(df['minimum_nights'] > 5.0,5.0,df['minimum_nights'])





df['number_of_reviews'] = np.where(df['number_of_reviews'] < 1.0 , 1.0 , df['number_of_reviews'])

df['number_of_reviews'] = np.where(df['number_of_reviews'] > 24.0 , 24.0 , df['number_of_reviews'])



df['price'] = np.where(df['price'] < 69.0 , 69.0 , df['price'])

df['price'] = np.where(df['price'] > 175.0 , 175.0 , df['price'])
df.boxplot(rot=45)
sns.heatmap(df.corr(),annot=True)
df.corr()['price'].sort_values(ascending=False)
X = df.drop('price',axis=1)

y = df['price']
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaler.fit_transform(X)
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state = 8)
from sklearn.ensemble import ExtraTreesClassifier

feature_model = ExtraTreesClassifier(n_estimators=50)

feature_model.fit(X_train,end.fit_transform(y_train))



feat_importances = pd.Series(feature_model.feature_importances_ , index = X.columns).plot(kind='barh')
from sklearn.linear_model import LinearRegression

from sklearn.metrics import r2_score

from sklearn.metrics import mean_squared_error,mean_absolute_error



from sklearn.ensemble import RandomForestRegressor

from sklearn.ensemble import GradientBoostingRegressor
lr = LinearRegression()

lr.fit(X_train,y_train)

print("LinearRegression")

print(35 * '-')

print("Train Score: ", (lr.score(X_train,y_train)*100))

pred_1 = lr.predict(X_test)

print("Test Score: ", r2_score(y_test,pred_1) * 100)



print('\n')

print(50*'-')

print("mean_absolute_error: ", mean_absolute_error(y_test,pred_1))

print("mean_squared_error: ", mean_squared_error(y_test,pred_1))

print("Root_mean_squared_error: ",mean_squared_error(y_test,pred_1,squared=False))
rf = RandomForestRegressor()

rf.fit(X_train,y_train)

print("RandomForestRegressor")

print(35 * '-')

print("Train Score: ", (rf.score(X_train,y_train)*100))

pred_2 = rf.predict(X_test)

print("Test Score: ", r2_score(y_test,pred_2) * 100)



print('\n')

print(50*'-')

print("mean_absolute_error: ", mean_absolute_error(y_test,pred_2))

print("mean_squared_error: ", mean_squared_error(y_test,pred_2))

print("Root_mean_squared_error: ",mean_squared_error(y_test,pred_2,squared=False))
gbr = GradientBoostingRegressor()

gbr.fit(X_train,y_train)

print("GradientBoostingRegressor")

print(35 * '-')

print("Train Score: ", gbr.score(X_train,y_train)*100)

pred_3 = gbr.predict(X_test)

print("Test Score: ", r2_score(y_test,pred_3) *100)



print('\n')

print(50*'-')

print("mean_absolute_error: ", mean_absolute_error(y_test,pred_3))

print("mean_squared_error: ", mean_squared_error(y_test,pred_3))

print("Root_mean_squared_error: ",mean_squared_error(y_test,pred_3,squared=False))
# Number of trees in random forest

n_estimators = [600,800,900,1200]

# Number of features to consider at every split

max_features = ['auto', 'sqrt','log2']

# Maximum number of levels

max_depth = [1,3,7]

# Minimum number of samples required to split a node

min_samples_split = [2, 5, 10,14]

# Minimum number of samples required at each leaf node

min_samples_leaf = [1, 2, 4,6,8]

# Create the random grid

random_grid = {'n_estimators': n_estimators,

               'max_features': max_features,

               'max_depth': max_depth,

               'min_samples_split': min_samples_split,

               'min_samples_leaf': min_samples_leaf,

              'criterion':['friedman_mse', 'mse']}

print(random_grid)
from sklearn.model_selection import RandomizedSearchCV

random = RandomizedSearchCV(estimator=gbr,param_distributions=random_grid,cv=5,n_jobs=-1)

random.fit(X_train,y_train)
random.best_params_
random_gbr = random.best_estimator_

pred_4 = random_gbr.predict(X_test)



print("Test Score: ", r2_score(y_test,pred_4)*100)