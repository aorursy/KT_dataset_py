import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import sklearn

import scipy
project_data=pd.read_csv('../input/new-york-city-airbnb-open-data/AB_NYC_2019.csv')

project_data.head(5)
# finding Number of rows and columns

print("Number of data points in resources data", project_data.shape)

print("="*100)

# columns/features in the dataset

print("Names of features are:- ")

print(project_data.columns.values)
#to check about the data type of the columns

project_data.info()
#to check for null values

np.sum(project_data.isna())
#Neighbourhood Group

project_data.neighbourhood_group.unique()
# to find the correlation between the features

project_data.corr()
#to get the details like number of observations, min,max,25%,50%,75% ,mean,std

project_data.describe()
#distribution of price less than 2000

sns.distplot(project_data[project_data.price<2000].price)

plt.title('Distribution of price (only where price<2000)')

plt.show()
#to check the distribution of minimum nights

sns.distplot(project_data[project_data.minimum_nights<20].minimum_nights)

plt.title('Distribution of Minimum nights ')

plt.show()
# Histograms for univariate Analysis

plt.hist(project_data['number_of_reviews'])

plt.show()
sns.distplot(project_data[project_data.number_of_reviews<50].number_of_reviews)

plt.title('Distribution of number_of_reviews ')

plt.show()
# Histograms for univariate Analysis

plt.hist(project_data['availability_365'])

plt.show()
#https://www.kaggle.com/adikeshri/what-s-up-with-new-york

sns.countplot(project_data['neighbourhood_group'])

plt.title('boroughs wise listings in NYC')

plt.xlabel('boroughs name')

plt.ylabel('Count')

plt.show()
sns.countplot(project_data.sort_values('room_type').room_type)

plt.title('Room type count')

plt.xlabel('Room type')

plt.ylabel('Count')

plt.show()
# Histograms for univariate Analysis

plt.hist(project_data['reviews_per_month'])

plt.show()
project_data.plot(kind='scatter', x='price', y='minimum_nights') ;

plt.show()
project_data.plot(kind='scatter', y='price', x='calculated_host_listings_count') ;

plt.show()
project_data.plot(kind='scatter', y='price', x='number_of_reviews') ;

plt.show()
project_data.plot(kind='scatter', y='price', x='availability_365') ;

plt.show()
project_data.plot(kind='scatter', y='price', x='reviews_per_month') ;

plt.show()
# soure: previous project

sns.set_style("whitegrid");

sns.FacetGrid(project_data,hue='room_type',size=5).map(plt.scatter,'price','minimum_nights').add_legend()

plt.show()
# pairwise scatter plot: Pair-Plot.

plt.close();

sns.set_style("whitegrid");

sns.pairplot(project_data,hue='room_type',vars=['price','minimum_nights','availability_365'],size=6,diag_kind='kde');

plt.legend()

plt.show() 

# NOTE: the diagnol elements are PDFs for each feature. PDFs are expalined below.
# pairwise scatter plot: Pair-Plot.

plt.close();

sns.set_style("whitegrid");

sns.pairplot(project_data,hue='neighbourhood_group',vars=['price','minimum_nights','availability_365'],size=6,diag_kind='kde');

plt.legend()

plt.show() 

# NOTE: the diagnol elements are PDFs for each feature. PDFs are expalined below.
counts,bin_edges=np.histogram(project_data['minimum_nights'],bins=10,density=True)

pdf=counts/(sum(counts))

cdf=np.cumsum(pdf)

plt.plot(bin_edges[1:],pdf)

plt.plot(bin_edges[1:],cdf)

plt.show()

print(np.percentile(project_data['minimum_nights'],95))
project_data['price_500']=project_data[project_data.price<500].price

sns.FacetGrid(project_data,hue='neighbourhood_group',size=5).map(sns.distplot,'price_500').add_legend()

plt.show()
project_data.columns
project_data.drop(['id', 'name', 'host_id', 'host_name', 'neighbourhood', 'last_review', 'reviews_per_month','price_500'], axis=1, inplace=True)
project_data.head(2)
from sklearn.model_selection import train_test_split

from sklearn import preprocessing



Y = project_data['price']

X = project_data[['neighbourhood_group', 'longitude', 'room_type', 'minimum_nights','availability_365','latitude', 'calculated_host_listings_count']]



X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3)

X_cv,X_test,Y_cv,Y_test = train_test_split(X_train, Y_train, test_size = 0.3)

print("Train set shape:")

print(X_train.shape)

print(Y_train.shape)

print("="*50)

print("Test set shape:")

print(X_test.shape)

print(Y_test.shape)
from sklearn.preprocessing import LabelBinarizer



lb = LabelBinarizer()

lb_train_ng= lb.fit_transform(X_train['neighbourhood_group'])

lb_test_ng= lb.transform(X_test['neighbourhood_group'])



lb_train_ng = pd.DataFrame(lb_train_ng, columns=lb.classes_)

lb_test_ng = pd.DataFrame(lb_test_ng, columns=lb.classes_)



print("After vectorizations")

print(lb_train_ng.shape, Y_train.shape)

print(lb_test_ng.shape, Y_test.shape)
from sklearn.preprocessing import LabelBinarizer



lb = LabelBinarizer()

lb_train_rt= lb.fit_transform(X_train['room_type'])

lb_test_rt= lb.transform(X_test['room_type'])



lb_train_rt = pd.DataFrame(lb_train_rt, columns=lb.classes_)

lb_test_rt = pd.DataFrame(lb_test_rt, columns=lb.classes_)



print("After vectorizations")

print(lb_train_rt.shape, Y_train.shape)

print(lb_test_rt.shape, Y_test.shape)
#source Kaggle

from sklearn.preprocessing import StandardScaler

standard_vec = StandardScaler(with_mean = False)

# this will rise an error Expected 2D array, got 1D array instead: 

# array=[105.22 215.96  96.01 ... 368.98  80.53 709.67].

# Reshape your data either using 

# array.reshape(-1, 1) if your data has a single feature 

# array.reshape(1, -1)  if it contains a single sample.

standard_vec.fit(X_train['availability_365'].values.reshape(-1,1))



X_train_av_std = standard_vec.transform(X_train['availability_365'].values.reshape(-1,1))

X_test_av_std = standard_vec.transform(X_test['availability_365'].values.reshape(-1,1))



print("After vectorizations")

print(X_train_av_std.shape, Y_train.shape)

print(X_test_av_std.shape, Y_test.shape)

#source Kaggle

from sklearn.preprocessing import StandardScaler

standard_vec = StandardScaler(with_mean = False)



standard_vec.fit(X_train['calculated_host_listings_count'].values.reshape(-1,1))



X_train_chl_std = standard_vec.transform(X_train['calculated_host_listings_count'].values.reshape(-1,1))

X_test_chl_std = standard_vec.transform(X_test['calculated_host_listings_count'].values.reshape(-1,1))



print("After vectorizations")

print(X_train_chl_std.shape, Y_train.shape)

print(X_test_chl_std.shape, Y_test.shape)

#source Kaggle

from sklearn.preprocessing import StandardScaler

standard_vec = StandardScaler(with_mean = False)

standard_vec.fit(X_train['minimum_nights'].values.reshape(-1,1))



X_train_mn_std = standard_vec.transform(X_train['minimum_nights'].values.reshape(-1,1))



X_test_mn_std = standard_vec.transform(X_test['minimum_nights'].values.reshape(-1,1))



print("After vectorizations")

print(X_train_mn_std.shape, Y_train.shape)

print(X_test_mn_std.shape, Y_test.shape)

#source Kaggle

from sklearn.preprocessing import StandardScaler

standard_vec = StandardScaler(with_mean = False)



standard_vec.fit(X_train['latitude'].values.reshape(-1,1))



X_train_l_std = standard_vec.transform(X_train['latitude'].values.reshape(-1,1))

X_test_l_std = standard_vec.transform(X_test['latitude'].values.reshape(-1,1))



print("After vectorizations")

print(X_train_l_std.shape, Y_train.shape)

print(X_test_l_std.shape, Y_test.shape)

#source Kaggle

from sklearn.preprocessing import StandardScaler

standard_vec = StandardScaler(with_mean = False)

standard_vec.fit(X_train['longitude'].values.reshape(-1,1))



X_train_lo_std = standard_vec.transform(X_train['longitude'].values.reshape(-1,1))

X_test_lo_std = standard_vec.transform(X_test['longitude'].values.reshape(-1,1))



print("After vectorizations")

print(X_train_lo_std.shape, Y_train.shape)

print(X_test_lo_std.shape, Y_test.shape)

# merge two sparse matrices: https://stackoverflow.com/a/19710648/4084039

from scipy.sparse import hstack

X_tr = hstack((lb_train_ng,lb_train_rt,X_train_av_std,X_train_chl_std,X_train_mn_std,X_train_l_std,X_train_lo_std)).tocsr()

X_te = hstack((lb_test_ng,lb_test_rt,X_test_av_std,X_test_chl_std,X_test_mn_std,X_test_l_std,X_test_lo_std)).tocsr()



print("Final Data matrix")

print(X_tr.shape, Y_train.shape)

print(X_te.shape, Y_test.shape)

print("="*100)
from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error,r2_score



lin_model = LinearRegression().fit(X_tr, Y_train)

y_train_predict = lin_model.predict(X_tr)

rmse = (np.sqrt(mean_squared_error(Y_train, y_train_predict)))

r2 = r2_score(Y_train, y_train_predict)


print("The model performance for training set")

print("--------------------------------------")

print('R2 score is {}'.format(r2*100))

print("\n")



# model evaluation for testing set

y_test_predict = lin_model.predict(X_te)

rmse = (np.sqrt(mean_squared_error(Y_test, y_test_predict)))

r2 = r2_score(Y_test, y_test_predict)



print("The model performance for testing set")

print("--------------------------------------")

print('R2 score is {}'.format(r2*100))
error_frame = pd.DataFrame({'Actual': np.array(Y_test).flatten(), 'Predicted': y_test_predict.flatten()})

error_frame.head(10)


print("Actul Vs Predicted")

plt.scatter(Y_test, y_test_predict)

plt.xlabel("Prices: $Y_i$")

plt.ylabel("Predicted prices: $\hat{Y}_i$")

plt.title("Prices vs Predicted prices: $Y_i$ vs $\hat{Y}_i$")

plt.show()