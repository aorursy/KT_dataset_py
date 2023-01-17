### This dataset contains house sale prices for King County, which includes Seattle. 

### It includes homes sold between May 2014 and May 2015.



### It's a great dataset for evaluating simple regression models.
import warnings



warnings.simplefilter('ignore')
# importing libraries

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV

from sklearn.linear_model import LinearRegression, SGDRegressor

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor

from sklearn.metrics import r2_score, mean_squared_error
#importing data



df= pd.read_csv('../input/kc_house_data.csv')
# Exploring the dataset



print(df.shape)



df.head()
# we'll create a copy of the DataFrame just to fall back if anything goes wrong



df_copy= df.copy()
print(df.id.nunique())



# we see that the id is almost unique we will drop this feature



df.drop('id', axis=1, inplace= True)
# we will check for any missing values



print(df.isna().sum())

# we see that there are no missing values
# Let's now convert the date column to look more like a date column



df['Date']= pd.to_datetime(df.date)
print(df.Date.dt.year.nunique()) # we see that there are only two values for year, matches with our description

print(df.Date.dt.month.nunique())

print(df.Date.dt.week.nunique())

print(df.Date.dt.dayofweek.nunique())



# we can conver these as new variables
df['Year']= df.Date.dt.year

df['Month']= df.Date.dt.month

df['quarter']= df.Date.dt.quarter

df['weekday']= df.Date.dt.dayofweek





# making date columns
df.sample(7)
# let's now check the other features in the dataset

print(f'number of unique values in "lat" are: {df.lat.nunique()}') # there seems to be 5k locations, we will drop this variable for now. 



print(f'number of uniques values in "long" are: {df.long.nunique()}') # we will drop this variable as well for now.



df.drop(['lat', 'long'], axis= 1, inplace= True)
plt.figure(figsize= (24, 9))



plt.subplot(2, 1, 1)



sns.distplot(df.price)



plt.subplot(2, 1, 2)



sns.distplot(df.loc[df.price < 1000000, 'price'], )



# we see that most house prices are below a million and in that below 400 thousand
# we see that there are a number of different sqft listed, we will create a single feature by adding them



sqft_cols = ['sqft_living', 'sqft_above', 'sqft_basement'] 



print(pd.concat([df[sqft_cols].sum(axis= 1), df[sqft_cols]], axis=1).head()) 

# we see that the calculation is now much more realistic



df['Total_sqft']= df[sqft_cols].sum(axis= 1)



#let's check the unique values in the floor feature



print(f'Unique values in floors: {sorted(df.floors.unique())}')

print(f'number of unique values in floors: {df.floors.nunique()}')



# we see that there are 0.5 floors in the variable, let's explore these observations more carefully
# we will have to convert the datatype to object first



df['floors']= df['floors'].astype('object')



df.loc[(df.floors == 1.5) | (df.floors == 2.5) | (df.floors == 3.5), df.columns[df.columns.str.contains(r'Total|above|basement|living|floors')]].sample(20)



# a little research and I found out that .5 is usally a compromise between between one and two floors,

# we will go without making any changed for now and make this a categorical variable
print(df.condition.nunique())

print(df.condition.unique()) # there are 5 unique values, however can't say which is best and worst, 

# but we use the feature anyway
print(df.grade.nunique())

print(df.grade.unique()) # we see there are 12 unique variables, similiar to condition
print(df['yr_built'].nunique())

print(df['yr_built'].unique()) #there are 116 unique values, let's plot this to get a better picture
plt.figure(figsize= (24, 4))

plt.subplot(1, 2, 1)

sns.distplot(df['Year'].sub(df['yr_built'])) #this will give the age of the house in the year it was sold



plt.subplot(1, 2, 2)

plt.hist(df['Year'].sub(df['yr_built']), cumulative= True)



# we see that most houses are less than 60 years old. We will create bins for age of houses instead of years
df['Year'].sub(df['yr_built']).min() # there is a negative value, which we will investigate this now.



print(len(df.loc[df['Year'].sub(df['yr_built']) == -1]))



# there are 442 observations with negative value for age of house



df.loc[df['Year'].sub(df['yr_built']) == -1].sample(10)



## the observations show that the year built is a year later than the year sold, this could mean that 

## the house could have been while still under construction bought in the constrcution phase,

## but we will change these values to 1 for now.
df['age_of_house']= df['Year'].sub(df['yr_built']) 

# here we are creating a new variable to calculate the age of the house and then create bins for them later



print(df['age_of_house'].nunique())

print(df['age_of_house'].unique())



# let's proceed to create bins. we'll check the minimum and maximum values for our reference



print(f'Lowest value in age variable is: {df["age_of_house"].min()}')

print(f'Highest value in age variable is: {df["age_of_house"].max()}')

# highest value in the field is 115, and lowest is -1, this means that the house was pre-booked
age_cond = [

    ((df['age_of_house']>= 0) & (df['age_of_house']<= 20)),

    ((df['age_of_house']> 20) & (df['age_of_house']<= 40)),

    ((df['age_of_house']> 40) & (df['age_of_house']<= 60)),

    ((df['age_of_house']> 60) & (df['age_of_house']<= 80)),

    ((df['age_of_house']> 80) & (df['age_of_house']<= 100)),

    (df['age_of_house']> 100),

    (df['age_of_house']< 0)

]



age_choices = ['0-20', '21-40', '41-60', '61-80', '81-100', '100+', 'Pre-booked']





age_bins= np.select(age_cond, choicelist=age_choices, default='Missed')

# let's now check the variable to see if everything went right
# pd.concat([pd.Series(age_bins), df['age_of_house']], axis= 1).sample(20)

# # we see that all the sample bins were created correctly.



# #we'll procees and add this to the DataFrame



df['age_bins']= age_bins



df['age_bins'].value_counts(dropna= False)

# Perfecto!

# we see that there are only 12 pre-booked houses.
df.head()
# let's check the yr_renovated feature now.



print(df['yr_renovated'].nunique())

# we see that there are 70 values, let's create a variable for renovation age also a binary column 

# indicating if any renovation was done to the house.



modification_age = np.where(df['yr_renovated']== 0, 0, df['Year'].sub(df['yr_built']))



modification_age[:50]

# We see that there are very less modifications done, so we need not create a separate binary column to 

# idicate this, we can create one column with the age which can again be a bin and 0s can be converted to 

# string sying 'not renovated'
reno_cond = [

    ((modification_age>0) & (modification_age<= 20)),

    ((modification_age>20) & (modification_age<= 40)),

    ((modification_age>40) & (modification_age<= 60)),

    ((modification_age>60) & (modification_age<= 80)),

    ((modification_age>80) & (modification_age<= 100)),

    (modification_age> 100), (modification_age == 0)

] # we can copy paste this condition from age choice and make necessary changes to it



reno_choices = ['0-20', '21-40', '41-60', '61-80', '81-100', '100+', 'Not Renovated']



reno_variable= np.select(reno_cond, reno_choices, default='Missed')



print(reno_variable[:50])



print(modification_age[:50])



# we see that the conditions were applied correctly.
df['renovations']= reno_variable



print(df['renovations'].value_counts(dropna= False))

# perfecto!

# Here however we see that not many houses have been renovated, so we have limited data, we can club these 

# ages to just say renovated making the column a binary variable.

df['renovations']= np.where(df['renovations']== 'Not Renovated', 'Not Renovated', 'Renovated')



print(df['renovations'].value_counts(dropna= False)) # we just have 914 values for renovated this might 

# not make any difference in the model, we'll check this later
# let's check zipcode



df['zipcode'].nunique() # We have 70 values for zipcodes, we will drop them later or check if we can use 

# them somehow.
print(pd.concat([df[df.columns[df.columns.str.contains('living')]], df['renovations']], axis= 1).sample(10))



# we see that sqft_living in 2015 has changed(decreased or increased) in some observations but not 

# marked as renovated. We do not know why this is, so we will create a categorical variable that says 

# "decreased", "increased" or "unchanged"



sqft_living_conds= [

    (df['sqft_living15'].sub(df['sqft_living']) < 0),

    (df['sqft_living15'].sub(df['sqft_living']) > 0),

    (df['sqft_living15'].sub(df['sqft_living']) == 0)

]



sqft_living_choices= ['Decreased', 'Increased', 'Unchanged']



sqft_change= np.select(sqft_living_conds, sqft_living_choices, 'Missed')



print(pd.concat([pd.Series(sqft_change), df[df.columns[df.columns.str.contains('living')]]], axis= 1).head(7))

# we see that the conds were applied correctly.
df['sqft_liv_change']= sqft_change

# adding to the DataFrame



df['sqft_liv_change'].value_counts(dropna= False)
# we'll check the min and max values in sqft_basement, sqft_living and sqft_above features



for i in ['sqft_living', 'sqft_above', 'sqft_basement']:

    

    print(f'Min value in {i} is: {df[i].min()}')

    print(f'Max value in {i} is: {df[i].max()}')

    

    ## we see that there are 0 values in basement, which we saw earlier and used to create a new variable

    

    ## we can create new variables from sqft_living and sqft_above features which we will see now         
df.loc[df['sqft_basement'] > 0, 'sqft_basement'].describe([.25, .5, .75, .95])



# we see that min value in the basement feature is 10, we will not delve deeper into that but 

# let's create categories for now
# let's check percentiles in sqft columns

for i in ['sqft_living', 'sqft_above', 'sqft_basement']:

    

    print(f'25th, 50th, 75th and 95th percentiles in {i} are: {np.percentile(df[i], [25,50,75,95])}')
#sqft_basement

basement_conds= [

    ((df['sqft_basement'] > 0) & (df['sqft_basement'] <= 450)),

    ((df['sqft_basement'] > 450) & (df['sqft_basement'] <= 700)),

    ((df['sqft_basement'] > 700) & (df['sqft_basement'] <= 980)),

    ((df['sqft_basement'] > 980) & (df['sqft_basement'] <= 1450)),

    (df['sqft_basement'] > 1450),

    (df['sqft_basement'] == 0)

]



basement_choice= ['Small', 'Moderate', 'Large', 'Huge', 'Humungous','No Basement']



df['basement_size']= np.select(basement_conds, basement_choice, 'Missed')





#sqft_living

living_conds= [

    ((df['sqft_living'] > 0) & (df['sqft_living'] <= 1427)),

    ((df['sqft_living'] > 1427) & (df['sqft_living'] <= 1910)),

    ((df['sqft_living'] > 1910) & (df['sqft_living'] <= 2550)),

    ((df['sqft_living'] > 2550) & (df['sqft_living'] <= 3760)),

    (df['sqft_living'] > 3760),

    (df['sqft_living'] == 0)

]



basement_choice= ['Small', 'Moderate', 'Large', 'Huge', 'Humungous','No living area']



df['living_size']= np.select(living_conds, basement_choice, 'Missed')



#sqft_above



above_conds= [

    ((df['sqft_above'] > 0) & (df['sqft_above'] <= 1190)),

    ((df['sqft_above'] > 1190) & (df['sqft_above'] <= 1560)),

    ((df['sqft_above'] > 1560) & (df['sqft_above'] <= 2210)),

    ((df['sqft_above'] > 2210) & (df['sqft_above'] <= 3400)),

    (df['sqft_above'] > 3400),

    (df['sqft_above'] == 0)

]



basement_choice= ['Small', 'Moderate', 'Large', 'Huge', 'Humungous','Nothing Above']



df['above_size']= np.select(above_conds, basement_choice, 'Missed')
df.sample(7)
for i in ['living_size', 'above_size', 'basement_size']:

    

    print(f'Values in {i} are \n {df[i].value_counts(dropna= False)}')

    print("")

    

    # we see there are no unwanted results in the dataFrame
lot_change= pd.Series(np.where(df['sqft_lot15']==0, 0,df['sqft_lot15'].sub(df['sqft_lot'])))
lot_change[:20]



df['sqft_lot_change']= pd.Series(np.where(lot_change < 0, 'Decreased', np.where(lot_change > 0, 'Increased', 'Unchanged')))



df['sqft_lot_change'].value_counts(dropna= False)

# looks like the code worked just fine.
df.sample(5)
# we'll now check the size of the houses.



df['Total_sqft'].describe([.25, .5, .75, .95])
house_conds= [

    (df['Total_sqft'] <= 2854),

    ((df['Total_sqft'] > 2854) & (df['Total_sqft'] <= 3820)),

    ((df['Total_sqft'] > 3820) & (df['Total_sqft'] <= 5100)),   

    ((df['Total_sqft'] > 5100) & (df['Total_sqft'] <= 7520)),

    (df['Total_sqft'] > 7520),

]



house_choice= ['Small', 'Moderate', 'Large', 'Huge', 'Humungous']



df['house_size']= np.select(house_conds, house_choice, 'Missed')



df['house_size'].value_counts(dropna= False)
df['zipcode'].unique()
zips_location= {'98178': 'Seattle', '98125': 'Seattle', '98028': 'Kenmore', '98136': 'Seattle', '98074': 'Sammamish'

 ,'98053': 'Redmond', '98003': 'Federal', '98198': 'Seattle', '98146': 'Seattle', '98038': 'Maple Valley'

 ,'98007': 'Bellevue', '98115': 'Seattle', '98107': 'Seattle', '98126': 'Seattle', '98019': 'Duvall'

 ,'98103': 'Seattle', '98002': 'Auburn', '98133': 'Seattle', '98040': 'Mercer Island','98092': 'Auburn'

 ,'98030': 'Kent', '98119': 'Seattle', '98112': 'Seattle', '98052': 'Redmond', '98027': 'Issaquah', '98117': 'Seattle',

'98058': 'Renton', '98001': 'Auburn', '98056': 'Renton', '98166': 'Seattle', '98023': 'Federal'

 ,'98070': 'Vashon', '98148': 'Seattle', '98105': 'Seattle', '98042': 'Kent', '98008': 'Bellevue', '98059': 'Renton'

 ,'98122': 'Seattle', '98144': 'Seattle', '98004': 'Bellevue', '98005': 'Bellevue', '98034': 'Kirkland' 

 ,'98075': 'Sammamish', '98116': 'Seattle', '98010': 'Black Diamond', '98118': 'Seattle', '98199': 'Seattle'

 ,'98032': 'Kent', '98045': 'North Bend', '98102': 'Seattle', '98077': 'Woodinville', '98108': 'Seattle'

 ,'98168': 'Seattle', '98177': 'Seattle', '98065': 'Snoqualmie', '98029': 'Issaquah', '98006': 'Bellevue'

 ,'98109': 'Seattle', '98022': 'Enumclaw', '98033': 'Kirkland', '98155': 'Seattle', '98024':'Fall City'

 ,'98011': 'Bothell', '98031':'Kent', '98106': 'Kent', '98072': 'Woodinville', '98188': 'Seattle'

 ,'98014': 'Carnation', '98055': 'Renton', '98039':'Medina'}



# this was done using a zipcode search!
df['zipcode']= df['zipcode'].apply(str)
df['zipcode']= df['zipcode'].replace(zips_location)
print(df['zipcode'].nunique())

print(df['zipcode'].value_counts())
plt.figure(figsize= (24, 9))

sns.boxplot(x='zipcode', y='price', data= df)



# we see that locations do have a influence over the target variable, so we will keep the variable in the df
# Now that we have taken care of lot the features we will explore the relations between the predictors 

# and the target variable



# we will drop a few features before EDA



df.drop(['date', 'yr_built', 'yr_renovated', 'Date', 'age_of_house'], axis= 1, inplace= True)
df.sample(5)
objs_cols= ['bedrooms', 'bathrooms', 'floors', 'waterfront', 'view', 'condition', 'grade', 'Year', 'Month', 'dayofweek'

, 'age_bins', 'renovations', 'sqft_liv_change', 'basement_size', 'living_size', 'above_size'

, 'sqft_lot_change', 'house_size', 'quarter', 'zipcode']



num_cols= ['sqft_living', 'sqft_lot', 'sqft_above', 'sqft_basement', 'sqft_living15', 'sqft_lot15'

           , 'Total_sqft', ]
plt.figure(figsize= (24, 6))

sns.boxplot(x= 'bedrooms', y= 'price', data= df)



# we see that there are significant difference in price with higher number of bedrooms, except for 

# houses with greater than 8 rooms, this could also be due to locations
plt.figure(figsize= (24, 6))

sns.boxplot(x= 'bathrooms', y= 'price', data= df)

# see there are differences in house prices with higher number of bathrooms
plt.figure(figsize= (24, 6))

sns.boxplot(x= 'floors', y= 'price', data= df)

# see there are differences in house prices with higher number of bathrooms
plt.figure(figsize= (24, 12))



for idx, col in enumerate(['waterfront', 'view', 'condition', 'grade', 'Year', 'Month']):

    

    plt.subplot(3, 2, idx+ 1)

    

    sns.boxplot(x= col, y= 'price', data= df)
# we'll check the condition and months features a little more closely



plt.figure(figsize= (24, 9))

sns.violinplot(x= 'condition', y= 'price', data= df)



# condition does not seem to be such a good feature, but we will keep it and check feature importance
# we'll check the condition and months features a little more closely



plt.figure(figsize= (24, 9))

sns.violinplot(y= 'price', x= 'Month', data= df)



# Month also does'nt seem to be such a good feature, but we will keep it and check feature importance later
plt.figure(figsize= (24, 18))



for idx, col in enumerate(['age_bins', 'renovations', 'sqft_liv_change', 'basement_size', 'living_size', 'above_size','sqft_lot_change', 'house_size', 'quarter']):

    

    plt.subplot(3, 3, idx+1)

    

    sns.boxplot(x= col, y= 'price', data= df)
num_cols
plt.figure(figsize= (24, 18))



count= 1



for idx, col in enumerate(num_cols):

    

    plt.subplot(4, 4, count)

    

    sns.regplot(x= col, y= 'price', data= df)

    plt.xlabel(col)

    

    count+=1

    

    plt.subplot(4, 4, count)

    sns.distplot(df[col])

    count+=1
sns.heatmap(df[num_cols].corr())
# we'll drop the numerical columns from the DataFrame for now



df.sample(3)
df.drop(num_cols, axis= 1, inplace= True) #dropped
df.columns
obj= ['bedrooms', 'bathrooms', 'floors', 'waterfront', 'view','condition', 'grade', 'zipcode','Year'

      , 'Month', 'quarter','weekday', 'age_bins', 'renovations', 'sqft_liv_change', 'basement_size'

      , 'living_size', 'above_size', 'sqft_lot_change', 'house_size']
df[obj]= df[obj].applymap(str) #creating object from all the other features to make dummies
df.info()
df_train= pd.get_dummies(df, drop_first=True)
print(df_train.shape)

df_train.head()
# we have 112 columns we'll go with it for now



X, y= df_train.drop('price', axis= 1),df_train['price']
print(y.min(), y.max())
y= y.div(1000)

y.min(), y.max()
X_train, X_test, y_train, y_test= train_test_split(X, y, test_size= 0.2, random_state= 123)



X_train.shape, X_test.shape, y_train.shape, y_test.shape
print(y_train.head())

print(y_test.head())
y_test.reset_index(drop= True, inplace= True)
y_test.head()
lr= LinearRegression()



lr.fit(X_train, y_train)



lr_pred= lr.predict(X_test)



print(mean_squared_error(y_test, lr_pred))

print(r2_score(y_test, lr_pred))
pd.concat([y_test, pd.Series(lr_pred)], axis= 1).sample(10)
d_tree= DecisionTreeRegressor()



d_tree.fit(X_train, y_train)



d_pred= d_tree.predict(X_test)



print(np.sqrt(mean_squared_error(y_test, d_pred)))

print(r2_score(y_test, d_pred))



# we have better results here than the linear model

# we will no try for Random Forest

pd.concat([y_test, pd.Series(d_pred)], axis= 1).head(10)
rf_reg= RandomForestRegressor() # we'll just fit of the book RandomForest model



rf_reg.fit(X_train, y_train)



rf_pred= rf_reg.predict(X_test)



print(np.sqrt(mean_squared_error(y_test, rf_pred)))

print(r2_score(y_test, rf_pred))



# we have a slightly better model with the RandomForest model

# let's check gradient boosting and extra trees and SGD regressors and then we will start fine tuning our 

# model
pd.concat([y_test, pd.Series(rf_pred)], axis= 1).head(8)
gbm_reg = GradientBoostingRegressor() #we'll again use off the book model here then fine tune it later



et_reg= ExtraTreesRegressor()



sgd_reg= SGDRegressor()
gbm_reg.fit(X_train, y_train)



gbm_pred= gbm_reg.predict(X_test)



print(np.sqrt(mean_squared_error(y_test, gbm_pred)))

print(r2_score(y_test, gbm_pred))

# we get better score than RandomForest
et_reg.fit(X_train, y_train)



et_pred= et_reg.predict(X_test)



print(np.sqrt(mean_squared_error(y_test, et_pred)))

print(r2_score(y_test, et_pred))



# this didn't perform as well as the others
sgd_reg.fit(X_train, y_train)



sgd_pred= sgd_reg.predict(X_test)



print(np.sqrt(mean_squared_error(y_test, sgd_pred)))

print(r2_score(y_test, sgd_pred))



# we get a better model than a simple linear model

# we will see if we can stack this model later
# for now we will use Grid search for hyper parameter tuning



gb_params= {'alpha':[.8, .7, .9],

            'max_depth': np.arange(1, 10, 1), 

            'max_features':[.7, .8]

            }



gb_grid= GridSearchCV(gbm_reg, gb_params, scoring= 'r2', cv= 5, n_jobs=-1, verbose= 1)



gb_grid.fit(X_train, y_train)



gb_grid_pred= gb_grid.predict(X_test)



print(np.sqrt(mean_squared_error(y_test, gb_grid_pred)))

print(r2_score(y_test, gb_grid_pred))



# we did get a better score, we will train a model with the best estimator from our grid search and train

# the model again

gbm_reg= gb_grid.best_estimator_
gbm_reg.fit(X_train, y_train)



gbm_pred = gbm_reg.predict(X_test)



r2_score(y_test, gbm_pred)



# we see that we have a better score.
gbm_reg.fit(X, y)



gbm_pred1= gbm_reg.predict(X)



r2_score(y, gbm_pred1)



# we see that the model is giving 85% r2score

# we will stop model building here.