#importing packages

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
#Reading the csv files from all three category

#Restaurants

chefmozaccepts_df=pd.read_csv('../input/restaurant-data-with-consumer-rating/chefmozaccepts.csv')

chefmozcuisine_df=pd.read_csv('../input/restaurant-data-with-consumer-rating/chefmozcuisine.csv')

chefmozhours4_df=pd.read_csv('../input/restaurant-data-with-consumer-rating/chefmozhours4.csv')

chefmozparking_df=pd.read_csv('../input/restaurant-data-with-consumer-rating/chefmozparking.csv')

geoplaces2_df=pd.read_csv('../input/restaurant-data-with-consumer-rating/geoplaces2.csv')
#Getting column name

chefmozaccepts_df.columns
#Getting information of shape 

chefmozaccepts_df.shape
#Getting infoamation about columns

chefmozaccepts_df.info()
#checking for null value

chefmozaccepts_df.isnull().sum()
#Getting dtypes of this columns

chefmozaccepts_df.dtypes
#Describe the dataset

chefmozaccepts_df['Rpayment'].describe()
#Getting first five rows

#chefmozaccepts_df.head()

chefmozaccepts_df['Rpayment'] = chefmozaccepts_df['Rpayment'].factorize()[0]
chefmozaccepts_df.head()
# chefmozcuisine.csv dataset

#Getting column infomation

chefmozcuisine_df.columns
#Getting information of shape and size

chefmozcuisine_df.shape
#Getting information about dataframe

chefmozcuisine_df.info()
#Describe the dataset

chefmozcuisine_df['Rcuisine'].describe()
#Checking for null value

chefmozcuisine_df.isnull().sum()
for column in chefmozcuisine_df.columns:

    chefmozcuisine_df[column].fillna(chefmozcuisine_df[column].mode()[0], inplace=True)
chefmozcuisine_df['Rcuisine'] = chefmozcuisine_df['Rcuisine'].factorize()[0]
#Getting top five rows of dataset

chefmozcuisine_df.head()
#chefmozhours4.csv dataset

#Getting the column information

chefmozhours4_df.columns
#Getting shape 

chefmozhours4_df.shape
#Getting information about dataset

chefmozhours4_df.info()
#Describe the dataset

chefmozhours4_df['days'].describe()
chefmozhours4_df['days'].head()
#Split the date

chefmozhours4_df[['day1','day2','day3','day4','day5','day6']]= chefmozhours4_df.days.str.split(';', expand = True)
#checking for null value

chefmozhours4_df.isnull().sum()
#Getting first five rows of this dataset

chefmozhours4_df.head()
# chefmozparking.csv dataset

#Columns information

chefmozparking_df.columns
#Getting information of shape 

chefmozparking_df.shape
#Getting information about dataset

chefmozparking_df.info()
#Describe the dataset

chefmozparking_df['parking_lot'].describe()
chefmozparking_df['parking_lot'] = chefmozparking_df['parking_lot'].factorize()[0]
#Getting first five rows

chefmozparking_df.head()
# geoplaces2.csv dataset

#Column information

geoplaces2_df.columns
#Information about columns

geoplaces2_df.info()
#Check for null value

geoplaces2_df.isnull().sum()
#Getting information of shape

geoplaces2_df.shape
#Getting Top five rows

geoplaces2_df.head(1)
#Print the column name with total number of unknown value ('\?')

#str.contains(' \?') -String into it contains unknown values

for col in geoplaces2_df.columns:

    if geoplaces2_df[col].dtype == object:

            print(col, (geoplaces2_df[col].str.contains('\?') == True).sum())
#dropping columns with more than 50% missing values

geoplaces2_df.drop(['fax','zip','url'],axis=1,inplace = True)

#and replacing remaining colvalues with mode

geoplaces2_df.replace('?', np.nan,inplace = True)

for column in geoplaces2_df.columns:

    geoplaces2_df[column].fillna(geoplaces2_df[column].mode()[0], inplace=True)

#geoplaces2_df.head(20)

geoplaces2_df.city.value_counts()
geoplaces2_df.city=geoplaces2_df.city.apply(lambda x: x.lower())

geoplaces2_df['city']=geoplaces2_df['city'].replace(['san luis potos','san luis potosi','slp','san luis potosi ','s.l.p.','s.l.p'],'san luis potosi' )

geoplaces2_df['city']=geoplaces2_df['city'].replace(['victoria','cd victoria','victoria ','cd. victoria'],'ciudad victoria' )

geoplaces2_df.city.value_counts()
#clean n cnt of state

geoplaces2_df.state=geoplaces2_df.state.apply(lambda x: x.lower())

#replacing state with unique.

geoplaces2_df['state']=geoplaces2_df['state'].replace(['san luis potos','san luis potosi','slp','s.l.p.'],'san luis potosi' )

geoplaces2_df.state.value_counts()

#clean n cnt of country

geoplaces2_df.country=geoplaces2_df.country.apply(lambda x: x.lower())

geoplaces2_df.country.value_counts()
geoplaces2_df['the_geom_meter'] = geoplaces2_df['the_geom_meter'].factorize()[0]

geoplaces2_df['name'] = geoplaces2_df['name'].factorize()[0]

geoplaces2_df['address'] = geoplaces2_df['address'].factorize()[0]

geoplaces2_df['city'] = geoplaces2_df['city'].factorize()[0]

geoplaces2_df['state'] = geoplaces2_df['state'].factorize()[0]

geoplaces2_df['country'] = geoplaces2_df['country'].factorize()[0]

geoplaces2_df['alcohol'] = geoplaces2_df['alcohol'].factorize()[0]

geoplaces2_df['smoking_area'] = geoplaces2_df['smoking_area'].factorize()[0]

geoplaces2_df['dress_code'] = geoplaces2_df['dress_code'].factorize()[0]

geoplaces2_df['accessibility'] = geoplaces2_df['accessibility'].factorize()[0]

geoplaces2_df['price'] = geoplaces2_df['price'].factorize()[0]

geoplaces2_df['Rambience'] = geoplaces2_df['Rambience'].factorize()[0]

geoplaces2_df['franchise'] = geoplaces2_df['franchise'].factorize()[0]

geoplaces2_df['area'] = geoplaces2_df['area'].factorize()[0]

geoplaces2_df['other_services'] = geoplaces2_df['other_services'].factorize()[0]

geoplaces2_df.head(5)
#Check for null value

geoplaces2_df.isnull().sum()
# user cuisine

usercuisine_df=pd.read_csv('../input/restaurant-data-with-consumer-rating/usercuisine.csv')

userpayment_df=pd.read_csv('../input/restaurant-data-with-consumer-rating/userpayment.csv')

userprofile_df=pd.read_csv('../input/restaurant-data-with-consumer-rating/userprofile.csv')

rating_final_df = pd.read_csv('../input/restaurant-data-with-consumer-rating/rating_final.csv')
usercuisine_df.head()
usercuisine_df.info()
usercuisine_df['Rcuisine'] = usercuisine_df['Rcuisine'].factorize()[0]

usercuisine_df.head(5)
#userpayment_df.head(5)

userpayment_df.info()
userpayment_df['Upayment'] = userpayment_df['Upayment'].factorize()[0]

userpayment_df.head(5)
#userprofile_df.head(5)

userprofile_df.info()
userprofile_df.isnull().sum()
# as data contains unknown value, we are replacinf with Nan.

userprofile_df.replace('?', np.nan)

#Print the column name with total number of unknown value ('\?')

#str.contains(' \?') -String into it contains unknown values

for col in userprofile_df.columns:

    if userprofile_df[col].dtype == object:         

            print(col, (userprofile_df[col].str.contains('\?') == True).sum())

#userprofile_df.head(5)
#since the missing value pernt is very low in each variables, we are replacing with mode of that individual column.

for column in userprofile_df.columns:

    userprofile_df[column].fillna(userprofile_df[column].mode()[0], inplace=True)



userprofile_df['smoker'] = userprofile_df['smoker'].factorize()[0]

userprofile_df['drink_level'] = userprofile_df['drink_level'].factorize()[0]

userprofile_df['dress_preference'] = userprofile_df['dress_preference'].factorize()[0]

userprofile_df['ambience'] = userprofile_df['ambience'].factorize()[0]

userprofile_df['transport'] = userprofile_df['transport'].factorize()[0]

userprofile_df['marital_status'] = userprofile_df['marital_status'].factorize()[0]

userprofile_df['hijos'] = userprofile_df['hijos'].factorize()[0]

userprofile_df['interest'] = userprofile_df['interest'].factorize()[0]

userprofile_df['personality'] = userprofile_df['personality'].factorize()[0]

userprofile_df['religion'] = userprofile_df['religion'].factorize()[0]

userprofile_df['activity'] = userprofile_df['activity'].factorize()[0]

userprofile_df['color'] = userprofile_df['color'].factorize()[0]

userprofile_df['budget'] = userprofile_df['budget'].factorize()[0]

userprofile_df.head(20)
#Merging multiple files into one

# merging rating file with userprofile

f1_merge =pd.merge(rating_final_df,userprofile_df)

f1_merge.head(5)
#merging f1 with userpayments

f2_merge=pd.merge(f1_merge,userpayment_df,how='left',on=['userID'])

f2_merge.head(5)
#merging B with usercuisine(F5)

f3_merge=pd.merge(f2_merge,usercuisine_df,how='left',on=['userID'])

f3_merge.head(5)
#merging f3_merge with geoplaces2(F8)

f4_merge=pd.merge(f3_merge,geoplaces2_df,how='left',on=['placeID'])

f4_merge.head(5)
#merging f4.merge with chefmozparking

f5_merge=pd.merge(f4_merge,chefmozparking_df,how='left',on=['placeID'])

f5_merge.head(5)
#merging f5_merge with chefmozcuisine

f6_merge=pd.merge(f5_merge,chefmozcuisine_df,how='left',on=['placeID'])

f6_merge.head(5)
#merging f6_merge with chefmozaccepts(F1)

f7_merge=pd.merge(f6_merge,chefmozaccepts_df,how='left',on=['placeID'])

f7_merge.head(5)
len(f7_merge)
f7_merge.info()
f7_merge.shape
f7_merge.isnull().any()
f7_merge=f7_merge.fillna(0)

f7_merge.isnull().values.any()
f7_merge['userID'] = f7_merge['userID'].factorize()[0]

f7_merge = f7_merge.drop(['country'],axis= 'columns')
def correlation_heatmap(f7_merge):

    _, ax = plt.subplots(figsize = (25,25))

    colormap= sns.diverging_palette(220, 10, as_cmap = True)

    sns.heatmap(f7_merge.corr(), annot=True, cmap = colormap)



correlation_heatmap(f7_merge)
#packages for modelling

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor
# splitting train and test data as 75/25.

X_linear_reg=f7_merge.drop(['placeID','rating','food_rating','service_rating','marital_status','address','parking_lot'],axis=1)

y=f7_merge['rating']

X_train,X_test,y_train,y_test = train_test_split(X_linear_reg,y,test_size=0.25)
reg = LinearRegression()

reg.fit(X_train, y_train)
from sklearn import metrics



pred = reg.predict(X_test)

print('multiple linear Model')

mean_squared_error = metrics.mean_squared_error(y_test, pred)

print('Mean Squared Error (MSE) ', round(np.sqrt(mean_squared_error), 2))

print('R-squared (training) ', round(reg.score(X_train, y_train), 3))

print('R-squared (testing) ', round(reg.score(X_test, y_test), 3))

print('Intercept: ', reg.intercept_)

print('Coefficient:', reg.coef_)
reg.score(X_test,y_test)
#Finding out mean, median & mode

print('Mean', round(f7_merge['rating'].mean(), 2))

print('Median', f7_merge['rating'].median())

print('Mode', f7_merge['rating'].mode()[0])
#predicting label values for test set

reg.predict(X_test)
#model building using Decision tree Regression.

model2 =  DecisionTreeRegressor()

# splitting train and test data as 75/25.

#X_dec_tree = f7_merge.drop(['placeID','rating','food_rating','service_rating'],axis=1)

X_dec_tree=f7_merge.drop(['placeID','rating','food_rating','service_rating','marital_status','address','parking_lot'],axis=1)

y=f7_merge['rating']

X_train,X_test,y_train,y_test = train_test_split(X_dec_tree,y,test_size=0.25)



#Model fitting

model2.fit(X_train, y_train)



from sklearn import metrics



#predicting on test data.

pred2= model2.predict(X_test)



print('Decision Tree Regressor Model')

mean_squared_error = metrics.mean_squared_error(y_test, pred2)

print('Mean Squared Error (MSE) ', round(np.sqrt(mean_squared_error), 2))

print('R-squared (training) ', round(model2.score(X_train, y_train), 3))

print('R-squared (testing) ', round(model2.score(X_test, y_test), 3))
model2.score(X_test,y_test)
#model building by random forest.

model3 = RandomForestRegressor(max_depth=2, random_state=0) 
# splitting train and test data as 75/25.

X_Rand_forest = f7_merge.drop(['placeID','rating','food_rating','service_rating'],axis=1)

y=f7_merge['rating']

X_train,X_test,y_train,y_test = train_test_split(X_Rand_forest,y,test_size=0.25)



#Model fitting

model3.fit(X_train, y_train)

#predicting on test data.

pred3= model3.predict(X_test)



print('Random Forest Regressor Model')

mean_squared_error = metrics.mean_squared_error(y_test, pred3)

print('Mean Squared Error (MSE) ', round(np.sqrt(mean_squared_error), 2))

print('R-squared (training) ', round(model3.score(X_train, y_train), 3))

print('R-squared (testing) ', round(model3.score(X_test, y_test), 3))
model3.score(X_test,y_test)