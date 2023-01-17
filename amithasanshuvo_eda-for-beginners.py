

import pandas as pd



housing = pd.read_csv("../input/hands-on-machine-learning-housing-dataset/housing.csv")
#Showing the first 5 rows

housing.head()
#info of dataset

housing.info
#Value counts of ocean_proximity column

housing['ocean_proximity'].value_counts()
#shape of our data

housing.shape
#Summary of each numerical attributes

housing.describe()
#Showwing the correlations

housing.corr()

#Showing the columns

housing.columns
#Plotting histograms for each numerical attributes

%matplotlib inline



import matplotlib.pyplot as plt

housing.hist(bins =50, figsize=(20,15))



plt.show()



#Slighly over 1000 distrcts have a median_house_value about 500000 usd
housing.hist(column='population')

# random_state parameter always generate the same shuffle indices. If the dataset 

# is not big enough then there's a chance of sampling bias

from sklearn.model_selection import train_test_split

train_set, test_set = train_test_split(housing, test_size = 0.2, random_state=42)



# Geographical data of all districts

housing.plot (kind ='scatter', x='longitude', y= 'latitude')
housing.plot (kind ='scatter', x='longitude', y= 'latitude', alpha = 0.1)

# adding alpha for better visualization. This helps to visualize the high density data points
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,

    s=housing["population"]/100, label="population", figsize=(10,10),

    c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True,

    sharex=False)

plt.legend()



# This represents that housing prices are varies with locations and population density

corr_matrix = housing.corr()
print (corr_matrix)
# Correlation of median_house_value with other attributes

corr_matrix['median_house_value'].sort_values(ascending = False)
# another way to check correlation using pandas



from pandas.plotting import scatter_matrix

attributes = ['median_house_value', 'median_income','total_rooms','housing_median_age']

scatter_matrix(housing[attributes], figsize = (12,8))
housing.plot (kind='scatter', x="median_income",y='median_house_value',alpha = 0.1)



# It means corr is very strong. 
# Creating new attributes



housing['room_per-_household'] = housing['total_rooms']/housing['households']

housing['bedrooms_per_rooms'] = housing['total_bedrooms']/housing['total_rooms']

housing['population_per_household'] = housing['population']/housing['households']
# Now the corr matrix will look something like this

corr_matrix = housing.corr()

corr_matrix['median_house_value'].sort_values(ascending = False)





# This means bedrooms_per_room attributee is much more correlated with mediamn house value than total num of rooms

# Lower bedrooms has high price
housing.plot(kind="scatter",

             x="room_per-_household",

             y="median_house_value",

             alpha=0.2)

plt.axis([0, 5, 0, 520000])



# Data Cleaning and handling missing values



#housing.dropna(subset=['total_bedrooms'])



# we can use this. but sklearn also provide e good function
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy ='median')





# this help to take care of missing values. 

# medians only be computed on numerical values ,so we need a copy of data without text attributes i.e ocean_proximity
housing_num = housing.drop('ocean_proximity', axis=1)
imputer.fit(housing_num)

# fitting the imputer instances to training data. It only computed the median of each attributes.
housing_num.median().values

X = imputer.transform(housing_num)



# Transforming the train set by replacing the missing values with new medians
imputer.strategy

housing_tr = pd.DataFrame(X, columns=housing_num.columns, index=housing.index)
housing_tr.loc[housing_num.index.values]

# Putting back into Pandas df
housing_tr = pd.DataFrame(X, columns=housing_num.columns,

                          index=housing_num.index)
housing_tr.head()

housing_cat = housing[["ocean_proximity"]]

housing_cat.head(10)
# Earlier we left the text attribute (ocean_proximity), now we have to work on that



# Converting them into labels



from sklearn.preprocessing import OrdinalEncoder

ordinal_encoder = OrdinalEncoder()

housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)

housing_cat_encoded[:10]
ordinal_encoder.categories_

# Convert int to categorical values into onehot vectors

from sklearn.preprocessing import OneHotEncoder



cat_encoder = OneHotEncoder()

housing_cat_1hot = cat_encoder.fit_transform(housing_cat)

housing_cat_1hot
housing_cat_1hot.toarray()
