import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O 

import matplotlib.pyplot as plt # various graphs

from sklearn.model_selection import train_test_split

from pandas.plotting import scatter_matrix

import os #system i/o





project_dir="../input"

housing = pd.read_csv(project_dir+"/housing.csv")
housing.head()
housing.info()
housing.describe()
housing["ocean_proximity"].value_counts()
%matplotlib inline

housing.hist(bins=50,figsize=(30,24))

plt.show()
%matplotlib inline

housing['ocean_proximity'].value_counts().plot(kind='bar')

plt.show()

housing.plot(kind="scatter",x="longitude",y="latitude",alpha=0.3,s=housing["population"]/100,label="population",figsize=(15,12),c="median_house_value",cmap=plt.get_cmap("jet"),colorbar=True)

plt.show()
corr_mat=housing.corr()

corr_mat
corr_mat["median_house_value"].sort_values(ascending=False)
sub_attributes=["median_house_value","median_income","total_rooms","housing_median_age"]

scatter_matrix(housing[sub_attributes],figsize=(20,15))

plt.show()