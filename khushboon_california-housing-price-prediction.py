import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
# Read the dataset

housing = pd.read_csv("../input/california-housing-prices/housing.csv")
housing.shape



# describe function gives a summary like mean, quartiles, median, std, count, etc for the numeric columns

housing.describe()



# %% [code]

# info functions helps us to understand the data type of all the columns

housing.info()
# lets check if there are missing values in the data

housing.isnull().sum()
plt.figure(figsize=(5,5))

plt.hist(housing[housing["total_bedrooms"].notnull()]["total_bedrooms"],bins=30,color="purple")

#histogram of totalbedrooms

#data has some outliers..??

(housing["total_bedrooms"]>4000).sum()

plt.title("Historgram")

plt.xlabel("Total Bedrooms")

plt.ylabel("Frequency")



# We can clearly see there are some outliers in the column, but let check with the help of box plot once more
plt.figure(figsize=(15,5))

sns.boxplot(y="total_bedrooms",data=housing, orient="h", palette="plasma")

plt.plot



#As we can see there are a lot of outliers, so to fill them we should be using ``Median`` instead of ``Mean``, 

# as the mean would vary a lot because of outliers and can affect the accuracy of our model
# Fill missing values

housing['total_bedrooms'] = housing['total_bedrooms'].fillna((housing['total_bedrooms'].median()))
# Lets plot and see what our dependent variable ie; "Y" column - ("median house price") looks like

# Histogram would be the best way to do it



plt.figure(figsize=(20,5))

sns.set_color_codes(palette="bright")

sns.distplot(housing['median_house_value'],color='g')



# We can see there is sudden increase in the median house value at >= 5,00,000, 

# & this could be outliers. We should definately be removing them.
housing[housing["median_house_value"]>300000]["median_house_value"].value_counts().head(10)

housing = housing.loc[housing["median_house_value"]<500001,:]

plt.figure(figsize=(20,5))

sns.set_color_codes(palette="bright")

sns.distplot(housing["median_house_value"], color="r")
#The bins parameter is used to customize the number of bins shown on the plots.

housing.hist(bins=50,figsize=(10,10))
# Since we have some geographical data, lets see if get some meaning insights from it..



plt.figure(figsize=(10,5))

plt.scatter(housing["longitude"],housing["latitude"],c=housing["median_house_value"],

            s=housing["population"]/50, alpha=0.1, cmap="Oranges")

plt.colorbar()

plt.xlabel("Longitude")

plt.ylabel("Latitude")

plt.title("House price based of geographical co-ordinates")



# We can see there are some high density areas in california, so we can say the price of house is a bit realted to

# location as well. 



# Earlier when I saw the data, I thought longitude & latitude would not be weak predictors

# but after plotting this, we can conclude even they are useful features.``

# So never judge it by visually seeing the data just in the first time.
# Before we split our data, we can also see that the feature - total_rooms has no significance, as this talks 

# about the rooms in the entire district. 

# Instead, we should find out, how many rooms are there in individual household, that would be more informative

# for our analysis...



housing["rooms_household"] = housing.total_rooms / housing.households



# now we can remove this feature

housing.drop("total_rooms", axis=1, inplace=True)

# We have one categorical column ("Ocean Proximity") in the data set, lets see if we should keep this column or remove it



# Barplot of categorical column

plt.figure(figsize=(7,4))

sns.countplot(data=housing,x='ocean_proximity', palette = "YlOrBr_r")



# It is very definate we should be keeping this feautre, but since this is a categorical feature, we should perform 

# preprocessing on it to convert it into numerical data.

# to conviently split the data into x & y part, I am rearranging the output column and bring it in the last



housing=housing[["longitude", "latitude", "housing_median_age", "total_bedrooms", "population", 

                 "households", "median_income", "ocean_proximity", "rooms_household", "median_house_value"]]
# Spliting the data

x = housing.iloc[:,0:9].values

y = housing.iloc[:,9].values
# Converting Categorical attribute to numeric



from sklearn.preprocessing import LabelEncoder, OneHotEncoder

label_encode_x = LabelEncoder()

x[:, 7] = label_encode_x.fit_transform(x[:, 7])



onehot = OneHotEncoder(categories="auto")

x = onehot.fit_transform(x).toarray()
# Spliting the train & test data set 

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)
# Feature Scaling /  Normalization



# from sklearn.preprocessing import StandardScaler

# scale  = StandardScaler()

# x_train = scale.fit_transform(x_train)

# x_test = scale.transform(x_test)
# from sklearn.linear_model import LinearRegression

# lin_reg = LinearRegression()

# lin_reg.fit(x_train, y_train)
# model for future prediction

# y_pred = lin_reg.predict(x_test)
# rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))

# score = lin_reg.score(x_test, y_test)



# Output of score = 0.6524213016981026

# Plot Actual vs. Predicted



# test = pd.DataFrame({'Predicted':y_pred,'Actual':y_test})

# fig= plt.figure(figsize=(16,8))

# test = test.reset_index()

# test = test.drop(['index'],axis=1)

# plt.plot(test[:80])

# plt.legend(['Actual','Predicted'])

# sns.jointplot(x='Actual',y='Predicted',data=test,kind='reg',color="grey")