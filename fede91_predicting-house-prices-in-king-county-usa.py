# Visualisation

import matplotlib.pyplot as plt

import seaborn as sns

import missingno

import folium

%matplotlib inline



# Data and Statistics

import pandas as pd

import numpy as np

from scipy import stats



# Train and Test Preparation

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import train_test_split



# Preprocessing

from sklearn import preprocessing

from sklearn.preprocessing import StandardScaler



# Models

from sklearn.pipeline import Pipeline

from sklearn import linear_model

from sklearn.preprocessing import PolynomialFeatures

from sklearn.tree import DecisionTreeRegressor

from sklearn.neural_network import MLPRegressor

from sklearn.model_selection import RandomizedSearchCV



# Evaluation metrics

from sklearn.metrics import explained_variance_score

from sklearn.metrics import max_error

from sklearn.metrics import mean_absolute_error

from sklearn.metrics import mean_squared_error

from sklearn.metrics import mean_squared_log_error

from sklearn.metrics import median_absolute_error

from sklearn.metrics import r2_score

from sklearn.metrics import mean_poisson_deviance

from sklearn.metrics import mean_gamma_deviance



df = pd.read_csv("../input/housesalesprediction/kc_house_data.csv")



evaluation = pd.DataFrame({'Model': [],

                           'Details':[],

                           'Max Error':[],

                           'Mean Absolute Error' : [],

                           'Mean Squared Error' : [],

                           'Mean Squared Log Error' : [],

                           'Median Absolute Error' : [],

                           'Mean Poisson Deviance' : [],

                           'Mean Gamma Deviance': [],

                           'Root Mean Squared Error (RMSE)':[],

                           'R-squared (training)':[],

                           'Adjusted R-squared (training)':[],

                           'R-squared (test)':[],

                           'Adjusted R-squared (test)':[],

                           '12-Fold Cross Validation':[]})

def adjustedR2(r2,n,k):

    return r2-(k-1)/(n-k)*(1-r2)
df.head()
print("The dataset has", df.shape[0], "rows and", df.shape[1], "features.")
df.dtypes
# Plot graphic of missing values

missingno.matrix(df, figsize = (30,5))
# Use isnull() function to convert the missing values in boolean and sum them.

df.isnull().sum()
# Let's define a list of columns and the dataset in boolean version

col_list = df.columns.to_list()

df_isnull = df.isnull()



# Now is time to create a loop that print the informations we are looking for.

for col in col_list:

    print(col)

    print(df_isnull[col].value_counts())

    print("")
df.describe()
df_2 = df[(df['bedrooms'] == 0) | (df['bathrooms'] == 0)]

df_2
# Mean Calculation

print("The average number of bedrooms is:" , df['bedrooms'].mean(axis=0))

print("The average number of bathrooms is" , df['bathrooms'].mean(axis=0))
# Most Frequent Value

print("The most frequent number of bedrooms is: " , df['bedrooms'].value_counts().idxmax())

print("The most frequent number of bathrooms is" , df['bathrooms'].value_counts().idxmax())
# Most Frequent Value

freq_bed = df['bedrooms'].value_counts().idxmax()

freq_bath = df['bathrooms'].value_counts().idxmax()



# Replace the values

df['bedrooms'].replace(0, freq_bed, inplace=True)

df['bathrooms'].replace(0, freq_bed, inplace=True)



# Double check if there are other 0 values

# df[(df['bedrooms'] == 0) | (df['bathrooms'] == 0)].head()
df.loc[df['bedrooms'] > 12]
m = folium.Map(

    location=[47.6878, -122.331], zoom_start=25, tiles="OpenStreetMap", attr='Mapbox')

m
df['date'] =  pd.to_datetime(df['date'], format='%Y%m%dT%H%M%S%f')

df.head()
df.dtypes
# Drop the row where the 'badrooms' value is 33

df.drop(df[df['bedrooms'] == 33].index, inplace=True)
df.reset_index(drop=True, inplace=True)
# Let's define a personal colour palette. This is something that I'm still working on in order to identify my graphic design as Data Scientist.

p_palette = ['#FCBB6D', "#D8737F", "#AB6C82", "#685D79", "#475C7A", "#F18C8E", "#F0B7A4", "#F1D1B5", "#568EA6", "#305F72"]

d_palette = ['#568EA6']



# Plot a bar chart with the number of bedrooms 

n_bedr = df['bedrooms'].unique()

plt.figure(figsize = (12, 6))

sns.barplot(x = n_bedr, y = df['bedrooms'].value_counts(), palette = p_palette, data = df)

plt.xlabel("Number of bedrooms", fontsize = 14)

plt.ylabel("Count of Houses", fontsize = 14)

plt.title("Houses - Number of bedrooms distribution", fontsize = 18)

plt.show()
# Plot a bar chart with the number of Bathrooms 



# Define a new DataFrame with the number of bathrooms and the frequency

bath_dic = df['bathrooms'].value_counts()

bath_df = bath_dic.to_frame()

bath_df.reset_index(inplace=True)

bath_df.rename(columns={'index': 'bathrooms', 'bathrooms': 'freq_b'}, inplace=True)



# Plot the bar chart

plt.figure(figsize = (12, 6))



sns.barplot(x = bath_df['bathrooms'], y = bath_df['freq_b'], palette = p_palette, data = df)

plt.xlabel("Number of bathrooms", fontsize = 14)

plt.ylabel("Count of Houses", fontsize = 14)

plt.title("Houses - Number of bathrooms distribution", fontsize = 18)

plt.show()
# Let's plot a pairplot to show the different distributions.

# sns.pairplot(data=df, x_vars=df[['bedrooms', 'bathrooms']], y_vars = df['price'], kind='scatter')



plt.figure(figsize=(26,6))

sns.set_palette(d_palette)



# First plot - Bathrooms - Price

plt.subplot(1,2,1)

sns.scatterplot(x=df['bathrooms'], y=df['price'], data=df, palette=p_palette)

plt.xlabel('Bathrooms', fontsize=14)

plt.ylabel('Price', fontsize=14)

plt.title("Price Distribution by bathrooms", fontsize=18)



# Second plot Bedrooms - Price

plt.subplot(1,2,2)

sns.scatterplot(x=df['bedrooms'], y=df['price'], data=df, palette=p_palette)

plt.xlabel('Bedrooms', fontsize=14)

plt.ylabel('Price', fontsize=14)

plt.title('Price Distribution by Bedrooms', fontsize=18)

# Group by number of bedrooms

df_mean_bed = df[['price','bedrooms']].groupby('bedrooms').mean()



# Reset index

df_mean_bed.reset_index(inplace=True)



# Calculate the price per bedroom

df_mean_bed['rate'] = df_mean_bed['price']/df_mean_bed['bedrooms']



# Define the figure dimensions

plt.figure(figsize=(26,6))



# First Plot

plt.subplot(1,2,1)

sns.barplot(x = df_mean_bed['bedrooms'], y = df_mean_bed['price'], palette = p_palette, data = df_mean_bed)

plt.xlabel('Number of bedrooms', fontsize = 14)

plt.ylabel('Price', fontsize = 14)

plt.title('Average price - bedrooms',fontsize = 18)



# Second Plot

plt.subplot(1,2,2)

sns.barplot(x = df_mean_bed['bedrooms'], y = df_mean_bed['rate'], palette = p_palette, data = df_mean_bed)

plt.xlabel('Number of bedrooms', fontsize = 14)

plt.ylabel('Price/bedrooms', fontsize = 14)

plt.title('Price/bedrooms rate',fontsize = 18)

# 
# Group by bathrooms

df_mean_bath = df[['price','bathrooms']].groupby('bathrooms').mean()



# Reset index

df_mean_bath.reset_index(inplace=True)



# Calculate the price per bedrooms

df_mean_bath['rate'] = df_mean_bath['price']/df_mean_bath['bathrooms']



# Define the figure dimensions

plt.figure(figsize=(26,6))



# Third Plot

plt.subplot(1,2,1)

sns.barplot(x = df_mean_bath['bathrooms'], y = df_mean_bath['price'], palette = p_palette, data = df_mean_bath)

plt.xlabel('Number of bathrooms', fontsize = 14)

plt.ylabel('Price', fontsize = 14)

plt.title('Average price - bathrooms',fontsize = 18)



# Fourth Plot

plt.subplot(1,2,2)

sns.barplot(x = df_mean_bath['bathrooms'], y = df_mean_bath['rate'], palette = p_palette, data = df_mean_bath)

plt.xlabel('Number of bathrooms', fontsize = 14)

plt.ylabel('Price/bathrooms', fontsize = 14)

plt.title('Price/bathrooms rate',fontsize = 18)
# Define the dataset

df_bb_comb = df[['price', 'bedrooms', 'bathrooms']]

df_bb_comb.reset_index(drop=True, inplace=True)



# Sum of bathrooms and bedrooms

df_bb_comb['bath_bed'] = df_bb_comb['bathrooms'] + df_bb_comb['bedrooms']

# Price rate for number of bathrooms+bedrooms

df_bb_comb['bb_rate'] = round((df_bb_comb['price']/df_bb_comb['bath_bed']), 1)



# Rate bathrooms/bedrooms

df_bb_comb['bath_bed_rate'] = round((df_bb_comb['bathrooms']/df_bb_comb['bedrooms']), 1)

# Price rate for bathrooms/bedrooms rate

df_bb_comb['price_bath_bed_rate'] = round((df_bb_comb['price']/df_bb_comb['bath_bed_rate']), 1)



df_bb_comb.head()
plt.figure(figsize=(26,8))



plt.subplot(2,1,1)

sns.boxplot(x = df_bb_comb['bath_bed'], y = df_bb_comb['bb_rate'], palette = p_palette, data = df)

plt.xlabel('Number of bathrooms and bedrooms', fontsize = 14)

plt.ylabel('Price rate', fontsize = 14)

plt.title('Price rate for bathrooms and bedrooms', fontsize=18)

plt.subplots_adjust(hspace = 0.5)



plt.subplot(2,1,2)

sns.boxplot(x = df_bb_comb['bath_bed_rate'], y = df_bb_comb['price_bath_bed_rate'], palette = p_palette, data = df)

plt.xlabel('Bathrooms/bedrooms rate', fontsize = 14)

plt.ylabel('Price rate', fontsize = 14)

plt.title('Price rate for bathrooms per bedrooms rate', fontsize=18)

df.date.dt.year

df['age'] = df.date.dt.year - df['yr_built']

df[['date', 'yr_built', 'yr_renovated', 'age']].head()
age_bins = [-2,10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 100000] 

labels = ['10-', '11-20', '21-30', '31-40', '41-50', '51-60', '61-70', '71-80', '81-90', '91-100', '100+']



df['age_binned'] = pd.cut(df['age'], age_bins, labels=labels, include_lowest=True)
df_age_binned = df.groupby(df.age_binned).mean()

df_age_binned.reset_index(inplace = True)



plt.figure(figsize=(26,6))



sns.barplot(x = df_age_binned['age_binned'], y = df_age_binned['price'], palette = p_palette, data = df_age_binned)

plt.xlabel('Age of the building', fontsize = 14)

plt.ylabel('Average price', fontsize = 14)

plt.title('Average price per building age', fontsize=18)
columns_name = ['price', 'bedrooms', 'bathrooms', 'sqft_living',

       'sqft_lot', 'floors', 'waterfront', 'view', 'condition', 'grade',

       'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode',

       'lat', 'long', 'sqft_living15', 'sqft_lot15']



df_stand = df[columns_name]



scaler = StandardScaler()

scaler.fit(df_stand)

df_stand = scaler.transform(df_stand)

print(scaler)



df_stand = pd.DataFrame(df_stand,columns = columns_name)

df_stand.head()
corr = df_stand.corr()



# Generate a mask for the upper triangle

mask = np.triu(np.ones_like(corr, dtype=np.bool))



# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(18, 16))



# Generate a custom diverging colormap

cmap = sns.diverging_palette(220, 10, as_cmap=True)



# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr, mask=mask, cmap=cmap, center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot = True)

plt.title('Heatmap of correlations', fontsize=18)
sns.set_palette(d_palette)

h = df[columns_name].hist(bins=25,figsize=(26,26), xlabelsize='10', ylabelsize='10')

sns.despine(left=True, bottom=True)

[x.title.set_size(14) for x in h.ravel()];

[x.yaxis.tick_left() for x in h.ravel()];
pearson_an = pd.DataFrame(columns=['variable', 'pearson_coef', 'p_value'])



for col in columns_name[1:]:

    pearson_coef, p_value = stats.pearsonr(df_stand[col], df_stand['price'])

    #print(col)

    #print('The Pearson coefficient is', pearson_coef, 'and the P_value is', p_value)

    #print('')

    to_append = pd.Series([col, pearson_coef, p_value], index = pearson_an.columns)

    pearson_an = pearson_an.append(to_append, ignore_index=True)



pearson_an.sort_values(by='pearson_coef', ascending=False)
# Train_test split using the original dataframe

train_data,test_data = train_test_split(df, train_size = 0.8, random_state = 22)



# Initialize a new LinearRegression model

lr = linear_model.LinearRegression()



# Identify the X_train and convert it to a Numpy Array

X_train = np.array(train_data['sqft_living'], dtype=pd.Series).reshape(-1,1)



# Identify the y_train and convert it to a Numpy Array

y_train = np.array(train_data['price'], dtype=pd.Series)



# Train the model on X_train and y_train

lr.fit(X_train,y_train)



# Define X_test and y_test

X_test = np.array(test_data['sqft_living'], dtype=pd.Series).reshape(-1,1)

y_test = np.array(test_data['price'], dtype=pd.Series)



# Make a prediction on X_test

Yhat = lr.predict(X_test)
# Evaluation Metrics

max_err = float(format(max_error(y_test, Yhat),'.3f'))

mabserr = float(format(mean_absolute_error(y_test, Yhat),'.3f'))

msqerr = float(format(mean_squared_error(y_test, Yhat),'.3f'))

msqlogerr = float(format(mean_squared_log_error(y_test, Yhat),'.3f'))

medabserror = float(format(median_absolute_error(y_test, Yhat),'.3f'))

mpoisdev = float(format(mean_poisson_deviance(y_test, Yhat),'.3f'))

mgamdev = float(format(mean_gamma_deviance(y_test, Yhat),'.3f'))

rmsesm = float(format(np.sqrt(mean_squared_error(y_test, Yhat)),'.3f'))

rtrsm = float(format(lr.score(X_train, y_train),'.3f'))

rtesm = float(format(lr.score(X_test, y_test),'.3f'))

cv = float(format(cross_val_score(lr,df[['sqft_living']],df['price'],cv=12).mean(),'.3f'))



print ("Average Price for Test Data: {:.3f}".format(y_test.mean()))

print('Intercept: {}'.format(lr.intercept_))

print('Coefficient: {}'.format(lr.coef_))



r = evaluation.shape[0]



evaluation.loc[r] = ['Simple Linear Regression','Best Feature', max_err, mabserr, msqerr, msqlogerr, medabserror,mpoisdev, mgamdev, rmsesm,rtrsm,'-',rtesm,'-',cv]

evaluation
plt.figure(figsize=(12,6))

plt.scatter(X_test,y_test,color="DarkBlue", label="Actual values", alpha=.1)

plt.plot(X_test,lr.predict(X_test),color='Coral', label="Predicted Regression Line")

plt.xlabel("Living Space (sqft)", fontsize=15)

plt.ylabel("Price ($)", fontsize=15)

plt.xticks(fontsize=13)

plt.yticks(fontsize=13)

plt.legend()



plt.gca().spines['right'].set_visible(False)

plt.gca().spines['top'].set_visible(False)
# We have train_data,test_data that include all the columns of our dataset.

top_5 = ['sqft_living', 'grade', 'sqft_above', 'sqft_living15', 'bathrooms']

# Select X_train and X_test

X_train = train_data[['sqft_living', 'grade', 'sqft_above', 'sqft_living15', 'bathrooms']]

X_test = test_data[['sqft_living', 'grade', 'sqft_above', 'sqft_living15', 'bathrooms']]



# Select y_train and y_test

y_train = train_data[['price']]

y_test = test_data[['price']]



# Initialize a LinearRegression model and fit it with the train data

mlr = linear_model.LinearRegression().fit(X_train, y_train)



# Make a prediction

Yhat = mlr.predict(X_test)
# Evaluation Metrics

max_err = float(format(max_error(y_test, Yhat),'.3f'))

mabserr = float(format(mean_absolute_error(y_test, Yhat),'.3f'))

msqerr = float(format(mean_squared_error(y_test, Yhat),'.3f'))

#msqlogerr = float(format(mean_squared_log_error(y_test, Yhat),'.3f'))

medabserror = float(format(median_absolute_error(y_test, Yhat),'.3f'))

#mpoisdev = float(format(mean_poisson_deviance(y_test, Yhat),'.3f'))

#mgamdev = float(format(mean_gamma_deviance(y_test, Yhat),'.3f'))

rmsesm = float(format(np.sqrt(mean_squared_error(y_test, Yhat)),'.3f'))

rtrsm = float(format(mlr.score(train_data[['sqft_living', 'grade', 'sqft_above', 'sqft_living15', 'bathrooms']],train_data['price']),'.3f'))

artrcm = float(format

               (adjustedR2

                (mlr.score

                 (train_data[['sqft_living', 'grade', 'sqft_above', 'sqft_living15', 'bathrooms']],

                  train_data['price']),train_data.shape[0],

                 len(['sqft_living', 'grade', 'sqft_above', 'sqft_living15', 'bathrooms'])

                ),'.3f')

              )

rtesm = float(format(mlr.score(X_test, y_test),'.3f'))

artecm = float(format

               (adjustedR2

                (mlr.score

                 (test_data[['sqft_living', 'grade', 'sqft_above', 'sqft_living15', 'bathrooms']],

                  test_data['price']),

                 test_data.shape[0],

                 len(['sqft_living', 'grade', 'sqft_above', 'sqft_living15', 'bathrooms'])

                ),'.3f')

              )

cv = float(format(cross_val_score(mlr,df[['sqft_living', 'grade', 'sqft_above', 'sqft_living15', 'bathrooms']],df['price'],cv=12).mean(),'.3f'))



print ("Average Price for Test Data:", y_test.mean())

print('Intercept: {}'.format(lr.intercept_))

print('Coefficient: {}'.format(lr.coef_))



r = evaluation.shape[0]



evaluation.loc[r] = ['Multiple Linear Regression','Top 5 Features by Pearson_coef', max_err, mabserr, msqerr, '-', medabserror,'-', '-', rmsesm,rtrsm,artrcm,rtesm,artecm,cv]

evaluation.sort_values(by = '12-Fold Cross Validation', ascending=False)
plt.figure(figsize = (26,6))



ax1 = sns.distplot(y_test, label = 'Actual values', color = 'DarkBlue', hist=False, bins=50)

sns.distplot(Yhat, color='Orange', label = 'Predicted values', hist=False, bins=50, ax=ax1)

plt.xlabel('Price distribution', fontsize=14)

plt.ylabel('Count', fontsize=14)

plt.title('Yhat and y_test distribution comparison - Multiple Linear Regression - Top 5 Features', fontsize=18)
# columns_name

all_features = columns_name[1:]



# Define X_train and X_test

X_train = train_data[all_features]

X_test = test_data[all_features]



# Define y_train and y_test

y_train = train_data['price']

y_test = test_data['price']



# Initiate a LinearRegression Model and Train it

aflrm = linear_model.LinearRegression().fit(X_train, y_train)



# Make a prediction

Yhat = aflrm.predict(X_test)
# Evaluation Metrics

max_err = float(format(max_error(y_test, Yhat),'.3f'))

mabserr = float(format(mean_absolute_error(y_test, Yhat),'.3f'))

msqerr = float(format(mean_squared_error(y_test, Yhat),'.3f'))

#msqlogerr = float(format(mean_squared_log_error(y_test, Yhat),'.3f'))

medabserror = float(format(median_absolute_error(y_test, Yhat),'.3f'))

#mpoisdev = float(format(mean_poisson_deviance(y_test, Yhat),'.3f'))

#mgamdev = float(format(mean_gamma_deviance(y_test, Yhat),'.3f'))

rmsesm = float(format(np.sqrt(mean_squared_error(y_test, Yhat)),'.3f'))

rtrsm = float(format(aflrm.score(train_data[all_features],train_data['price']),'.3f'))

artrcm = float(format

               (adjustedR2

                (aflrm.score

                 (train_data[all_features],

                  train_data['price']),train_data.shape[0],

                 len(all_features)

                ),'.3f')

              )

rtesm = float(format(aflrm.score(X_test, y_test),'.3f'))

artecm = float(format

               (adjustedR2

                (aflrm.score

                 (test_data[all_features],

                  test_data['price']),

                 test_data.shape[0],

                 len(all_features)

                ),'.3f')

              )

cv = float(format(cross_val_score(aflrm,df[all_features],df['price'],cv=12).mean(),'.3f'))



print ("Average Price for Test Data:", y_test.mean())

print('Intercept: {}'.format(lr.intercept_))

print('Coefficient: {}'.format(lr.coef_))



r = evaluation.shape[0]



evaluation.loc[r] = ['Multiple Linear Regression','All Features from Pearson_coef table', max_err, mabserr, msqerr, '-', medabserror,'-', '-', rmsesm,rtrsm,artrcm,rtesm,artecm,cv]

evaluation.sort_values(by = '12-Fold Cross Validation', ascending=False)
plt.figure(figsize = (26,6))



ax1 = sns.distplot(y_test, label = 'Actual values', color = 'DarkBlue', hist=False, bins=50)

sns.distplot(Yhat, color='Orange', label = 'Predicted values', hist=False, bins=50, ax=ax1)

plt.xlabel('Price distribution', fontsize=14)

plt.ylabel('Count', fontsize=14)

plt.title('Yhat and y_test distribution comparison - Multiple Linear Regression - All Features', fontsize=18)
# Define X_train, y_train, X_test, y_test

X_train = train_data[['sqft_living']]

y_train = train_data['price']

X_test = test_data[['sqft_living']]

y_test = test_data['price']



# Define the pipeline input

Input = [('standardscaler', StandardScaler()), ('polynomial', PolynomialFeatures(degree=2, include_bias=False)), ('model', linear_model.LinearRegression())]



# Prepare the pipeline

pipe = Pipeline(Input)



# Fit the pipeline

pipe.fit(X_train, y_train)



# Make a prediction

Yhat = pipe.predict(X_test)
# Evaluation Metrics

max_err = float(format(max_error(y_test, Yhat),'.3f'))

mabserr = float(format(mean_absolute_error(y_test, Yhat),'.3f'))

msqerr = float(format(mean_squared_error(y_test, Yhat),'.3f'))

msqlogerr = float(format(mean_squared_log_error(y_test, Yhat),'.3f'))

medabserror = float(format(median_absolute_error(y_test, Yhat),'.3f'))

mpoisdev = float(format(mean_poisson_deviance(y_test, Yhat),'.3f'))

mgamdev = float(format(mean_gamma_deviance(y_test, Yhat),'.3f'))

rmsesm = float(format(np.sqrt(mean_squared_error(y_test, Yhat)),'.3f'))

rtrsm = float(format(pipe.score(train_data[['sqft_living']],train_data['price']),'.3f'))

artrcm = float(format

               (adjustedR2

                (pipe.score

                 (train_data[['sqft_living']],

                  train_data['price']),train_data.shape[0],

                 len(['sqft_living'])

                ),'.3f')

              )

rtesm = float(format(pipe.score(X_test, y_test),'.3f'))

artecm = float(format

               (adjustedR2

                (pipe.score

                 (test_data[['sqft_living']],

                  test_data['price']),

                 test_data.shape[0],

                 len(['sqft_living'])

                ),'.3f')

              )

cv = float(format(cross_val_score(pipe,df[['sqft_living']],df['price'],cv=12).mean(),'.3f'))



print ("Average Price for Test Data:", y_test.mean())

print('Intercept: {}'.format(lr.intercept_))

print('Coefficient: {}'.format(lr.coef_))



r = evaluation.shape[0]



evaluation.loc[r] = ['Polynomial Regression','Best Feature', max_err, mabserr, msqerr, msqlogerr, medabserror,mpoisdev, mgamdev, rmsesm,rtrsm,artrcm,rtesm,artecm,cv]

evaluation.sort_values(by = '12-Fold Cross Validation', ascending=False)
plt.figure(figsize = (26,6))



ax1 = sns.distplot(y_test, label = 'Actual values', color = 'DarkBlue', hist=False, bins=50)

sns.distplot(Yhat, color='Orange', label = 'Predicted values', hist=False, bins=50, ax=ax1)

plt.xlabel('Price distribution', fontsize=14)

plt.ylabel('Count', fontsize=14)

plt.title('Yhat and y_test distribution comparison - Polynomial Regression - Best Feature', fontsize=18)
# The top 5 features are stored into the top_5 list



# Train and test split

X_train = train_data[top_5]

y_train = train_data['price']

X_test = test_data[top_5]

y_test = test_data['price']



# Define the pipe's input

Input = [('scale', StandardScaler()), ('polynomial', PolynomialFeatures(degree = 2, include_bias = False)), ('linearRegression', linear_model.LinearRegression())]



# Define the pipe

pipe = Pipeline(Input)



# Train the model

pipe.fit(X_train, y_train)



# Make a prediction

Yhat = pipe.predict(X_test)
# Evaluation Metrics

max_err = float(format(max_error(y_test, Yhat),'.3f'))

mabserr = float(format(mean_absolute_error(y_test, Yhat),'.3f'))

msqerr = float(format(mean_squared_error(y_test, Yhat),'.3f'))

msqlogerr = float(format(mean_squared_log_error(y_test, Yhat),'.3f'))

medabserror = float(format(median_absolute_error(y_test, Yhat),'.3f'))

mpoisdev = float(format(mean_poisson_deviance(y_test, Yhat),'.3f'))

mgamdev = float(format(mean_gamma_deviance(y_test, Yhat),'.3f'))

rmsesm = float(format(np.sqrt(mean_squared_error(y_test, Yhat)),'.3f'))

rtrsm = float(format(pipe.score(train_data[top_5],train_data['price']),'.3f'))

artrcm = float(format

               (adjustedR2

                (pipe.score

                 (train_data[top_5],

                  train_data['price']),train_data.shape[0],

                 len(top_5)

                ),'.3f')

              )

rtesm = float(format(pipe.score(X_test, y_test),'.3f'))

artecm = float(format

               (adjustedR2

                (pipe.score

                 (test_data[top_5],

                  test_data['price']),

                 test_data.shape[0],

                 len(top_5)

                ),'.3f')

              )

cv = float(format(cross_val_score(pipe,df[top_5],df['price'],cv=12).mean(),'.3f'))



print ("Average Price for Test Data:", y_test.mean())

print('Intercept: {}'.format(lr.intercept_))

print('Coefficient: {}'.format(lr.coef_))



r = evaluation.shape[0]



evaluation.loc[r] = ['Multivariate Polynomial Regression','Top 5 Features by Pearson_coef', max_err, mabserr, msqerr, msqlogerr, medabserror,mpoisdev, mgamdev, rmsesm,rtrsm,artrcm,rtesm,artecm,cv]

evaluation.sort_values(by = '12-Fold Cross Validation', ascending=False)
plt.figure(figsize = (26,6))



ax1 = sns.distplot(y_test, label = 'Actual values', color = 'DarkBlue', hist=False, bins=50)

sns.distplot(Yhat, color='Orange', label = 'Predicted values', hist=False, bins=50, ax=ax1)

plt.xlabel('Price distribution', fontsize=14)

plt.ylabel('Count', fontsize=14)

plt.title('Yhat and y_test distribution comparison - Multivariate Polynomial Regression - Top 5 Features', fontsize=18)
# The variable that contain the list of features is all_features



# X_train, y_train, X_test, y_test

X_train = train_data[all_features]

y_train = train_data['price']

X_test = test_data[all_features]

y_test = test_data['price']



# Let's define the pipe's input

Input = [('scaler', StandardScaler()), ('plynomial', PolynomialFeatures(degree=2, include_bias=False)), ('LinearRegression', linear_model.LinearRegression())]



# Initialize the pipeline

pipe = Pipeline(Input)



# Train the pipeline

pipe.fit(X_train, y_train)



# Make a prediction

Yhat = pipe.predict(X_test)
# Evaluation Metrics

max_err = float(format(max_error(y_test, Yhat),'.3f'))

mabserr = float(format(mean_absolute_error(y_test, Yhat),'.3f'))

msqerr = float(format(mean_squared_error(y_test, Yhat),'.3f'))

#msqlogerr = float(format(mean_squared_log_error(y_test, Yhat),'.3f'))

medabserror = float(format(median_absolute_error(y_test, Yhat),'.3f'))

#mpoisdev = float(format(mean_poisson_deviance(y_test, Yhat),'.3f'))

#mgamdev = float(format(mean_gamma_deviance(y_test, Yhat),'.3f'))

rmsesm = float(format(np.sqrt(mean_squared_error(y_test, Yhat)),'.3f'))

rtrsm = float(format(pipe.score(train_data[all_features],train_data['price']),'.3f'))

artrcm = float(format

               (adjustedR2

                (pipe.score

                 (train_data[all_features],

                  train_data['price']),train_data.shape[0],

                 len(all_features)

                ),'.3f')

              )

rtesm = float(format(pipe.score(X_test, y_test),'.3f'))

artecm = float(format

               (adjustedR2

                (pipe.score

                 (test_data[all_features],

                  test_data['price']),

                 test_data.shape[0],

                 len(all_features)

                ),'.3f')

              )

cv = float(format(cross_val_score(pipe,df[all_features],df['price'],cv=12).mean(),'.3f'))



print ("Average Price for Test Data:", y_test.mean())

print('Intercept: {}'.format(lr.intercept_))

print('Coefficient: {}'.format(lr.coef_))



r = evaluation.shape[0]



evaluation.loc[r] = ['Multivariate Polynomial Regression','All Features from Pearson_coef', max_err, mabserr, msqerr, '-', medabserror,'-', '-', rmsesm,rtrsm,artrcm,rtesm,artecm,cv]

evaluation.sort_values(by = '12-Fold Cross Validation', ascending=False)
plt.figure(figsize = (26,6))



ax1 = sns.distplot(y_test, label = 'Actual values', color = 'DarkBlue', hist=False, bins=50)

sns.distplot(Yhat, color='Orange', label = 'Predicted values', hist=False, bins=50, ax=ax1)

plt.xlabel('Price distribution', fontsize=14)

plt.ylabel('Count', fontsize=14)

plt.title('Yhat and y_test distribution comparison - Multivariate Polynomial Regression - All Features', fontsize=18)
# Define train and test

X_train = train_data[['sqft_living']]

y_train = train_data['price']

X_test = test_data[['sqft_living']]

y_test = test_data['price']



# Input pipeline

Input = [('scaler', StandardScaler()), ('Ridge', linear_model.Ridge(alpha = 0.5, fit_intercept = True, random_state = 22))]



# Initialize the Pipeline

pipe = Pipeline(Input)



# Fit the pipeline

pipe.fit(X_train, y_train)



# Make a prediction

Y_hat = pipe.predict(X_test)
# Evaluation Metrics

max_err = float(format(max_error(y_test, Yhat),'.3f'))

mabserr = float(format(mean_absolute_error(y_test, Yhat),'.3f'))

msqerr = float(format(mean_squared_error(y_test, Yhat),'.3f'))

#msqlogerr = float(format(mean_squared_log_error(y_test, Yhat),'.3f'))

medabserror = float(format(median_absolute_error(y_test, Yhat),'.3f'))

#mpoisdev = float(format(mean_poisson_deviance(y_test, Yhat),'.3f'))

#mgamdev = float(format(mean_gamma_deviance(y_test, Yhat),'.3f'))

rmsesm = float(format(np.sqrt(mean_squared_error(y_test, Yhat)),'.3f'))

#rtrsm = float(format(pipe.score(train_data[all_features],train_data['price']),'.3f'))

artrcm = float(format

               (adjustedR2

                (pipe.score

                 (train_data[['sqft_living']],

                  train_data['price']),train_data.shape[0],

                 len(['sqft_living'])

                ),'.3f')

              )

rtesm = float(format(pipe.score(X_test, y_test),'.3f'))

artecm = float(format

               (adjustedR2

                (pipe.score

                 (test_data[['sqft_living']],

                  test_data['price']),

                 test_data.shape[0],

                 len(['sqft_living'])

                ),'.3f')

              )

cv = float(format(cross_val_score(pipe,df[['sqft_living']],df['price'],cv=12).mean(),'.3f'))



print ("Average Price for Test Data:", y_test.mean())

print('Intercept: {}'.format(lr.intercept_))

print('Coefficient: {}'.format(lr.coef_))



r = evaluation.shape[0]



evaluation.loc[r] = ['Ridge Regression','Best Feature', max_err, mabserr, msqerr, '-', medabserror,'-', '-', rmsesm,rtrsm,artrcm,rtesm,artecm,cv]

evaluation.sort_values(by = '12-Fold Cross Validation', ascending=False)
plt.figure(figsize = (26,6))



ax1 = sns.distplot(y_test, label = 'Actual values', color = 'DarkBlue', hist=False, bins=50)

sns.distplot(Yhat, color='Orange', label = 'Predicted values', hist=False, bins=50, ax=ax1)

plt.xlabel('Price distribution', fontsize=14)

plt.ylabel('Count', fontsize=14)

plt.title('Yhat and y_test distribution comparison - Ridge Regression - Best Feature', fontsize=18)
# Define Train and test

X_train = train_data[top_5]

y_train = train_data['price']

X_test = test_data[top_5]

y_test = test_data['price']



# Define the Input for the pipeline

Input = [('scaler', StandardScaler()), ('Ridge_regression', linear_model.Ridge(alpha = 0.5, fit_intercept = True, random_state = 22))]



# Initialize the pipeline

pipe = Pipeline(Input)



# Train the Pipeline

pipe.fit(X_train, y_train)



# Make a prediction

Yhat = pipe.predict(X_test)
# Evaluation Metrics

max_err = float(format(max_error(y_test, Yhat),'.3f'))

mabserr = float(format(mean_absolute_error(y_test, Yhat),'.3f'))

msqerr = float(format(mean_squared_error(y_test, Yhat),'.3f'))

#msqlogerr = float(format(mean_squared_log_error(y_test, Yhat),'.3f'))

medabserror = float(format(median_absolute_error(y_test, Yhat),'.3f'))

#mpoisdev = float(format(mean_poisson_deviance(y_test, Yhat),'.3f'))

#mgamdev = float(format(mean_gamma_deviance(y_test, Yhat),'.3f'))

rmsesm = float(format(np.sqrt(mean_squared_error(y_test, Yhat)),'.3f'))

rtrsm = float(format(pipe.score(train_data[top_5],train_data['price']),'.3f'))

artrcm = float(format

               (adjustedR2

                (pipe.score

                 (train_data[top_5],

                  train_data['price']),train_data.shape[0],

                 len(top_5)

                ),'.3f')

              )

rtesm = float(format(pipe.score(X_test, y_test),'.3f'))

artecm = float(format

               (adjustedR2

                (pipe.score

                 (test_data[top_5],

                  test_data['price']),

                 test_data.shape[0],

                 len(top_5)

                ),'.3f')

              )

cv = float(format(cross_val_score(pipe,df[top_5],df['price'],cv=12).mean(),'.3f'))



print ("Average Price for Test Data:", y_test.mean())

print('Intercept: {}'.format(lr.intercept_))

print('Coefficient: {}'.format(lr.coef_))



r = evaluation.shape[0]



evaluation.loc[r] = ['Ridge Regression','Top 5 Features by Pearson_coef', max_err, mabserr, msqerr, '-', medabserror,'-', '-', rmsesm,rtrsm,artrcm,rtesm,artecm,cv]

evaluation.sort_values(by = '12-Fold Cross Validation', ascending=False)
plt.figure(figsize = (26,6))



ax1 = sns.distplot(y_test, label = 'Actual values', color = 'DarkBlue', hist=False, bins=50)

sns.distplot(Yhat, color='Orange', label = 'Predicted values', hist=False, bins=50, ax=ax1)

plt.xlabel('Price distribution', fontsize=14)

plt.ylabel('Count', fontsize=14)

plt.title('Yhat and y_test distribution comparison - Ridge Regression - Top 5 Features', fontsize=18)
# Train and test

X_train = train_data[all_features]

y_train = train_data['price']

X_test = test_data[all_features]

y_test = test_data['price']



# Define the Pipeline Input

Input = [('scale', StandardScaler()), ('Ridge', linear_model.Ridge(alpha = 0.5, fit_intercept = True, random_state = 22))]



# Initialize the Pipeline

pipe = Pipeline(Input)



# Fit the model

pipe.fit(X_train, y_train)



# Make a prediction

Yhat = pipe.predict(X_test)
# Evaluation Metrics

max_err = float(format(max_error(y_test, Yhat),'.3f'))

mabserr = float(format(mean_absolute_error(y_test, Yhat),'.3f'))

msqerr = float(format(mean_squared_error(y_test, Yhat),'.3f'))

#msqlogerr = float(format(mean_squared_log_error(y_test, Yhat),'.3f'))

medabserror = float(format(median_absolute_error(y_test, Yhat),'.3f'))

#mpoisdev = float(format(mean_poisson_deviance(y_test, Yhat),'.3f'))

#mgamdev = float(format(mean_gamma_deviance(y_test, Yhat),'.3f'))

rmsesm = float(format(np.sqrt(mean_squared_error(y_test, Yhat)),'.3f'))

rtrsm = float(format(pipe.score(train_data[all_features],train_data['price']),'.3f'))

artrcm = float(format

               (adjustedR2

                (pipe.score

                 (train_data[all_features],

                  train_data['price']),train_data.shape[0],

                 len(all_features)

                ),'.3f')

              )

rtesm = float(format(pipe.score(X_test, y_test),'.3f'))

artecm = float(format

               (adjustedR2

                (pipe.score

                 (test_data[all_features],

                  test_data['price']),

                 test_data.shape[0],

                 len(all_features)

                ),'.3f')

              )

cv = float(format(cross_val_score(pipe,df[top_5],df['price'],cv=12).mean(),'.3f'))



print ("Average Price for Test Data:", y_test.mean())

print('Intercept: {}'.format(lr.intercept_))

print('Coefficient: {}'.format(lr.coef_))



r = evaluation.shape[0]



evaluation.loc[r] = ['Ridge Regression','All Features from Pearson_coef', max_err, mabserr, msqerr, '-', medabserror,'-', '-', rmsesm,rtrsm,artrcm,rtesm,artecm,cv]

evaluation.sort_values(by = '12-Fold Cross Validation', ascending=False)
plt.figure(figsize = (26,6))



ax1 = sns.distplot(y_test, label = 'Actual values', color = 'DarkBlue', hist=False, bins=50)

sns.distplot(Yhat, color='Orange', label = 'Predicted values', hist=False, bins=50, ax=ax1)

plt.xlabel('Price distribution', fontsize=14)

plt.ylabel('Count', fontsize=14)

plt.title('Yhat and y_test distribution comparison - Ridge Regression - All Features', fontsize=18)
# Train and test

X_train = train_data[['sqft_living']]

y_train = train_data['price']

X_test = test_data[['sqft_living']]

y_test = test_data['price']



# Define the Input for the pipeline

Input = [('scale', StandardScaler()), ('Lasso', linear_model.Lasso(alpha = 0.5, precompute = False, random_state = 22))]



# Initialize the pipeline

pipe = Pipeline(Input)



# Train the model

pipe.fit(X_train, y_train)



# Make a prediction

Yhat = pipe.predict(X_test)
# Evaluation Metrics

max_err = float(format(max_error(y_test, Yhat),'.3f'))

mabserr = float(format(mean_absolute_error(y_test, Yhat),'.3f'))

msqerr = float(format(mean_squared_error(y_test, Yhat),'.3f'))

msqlogerr = float(format(mean_squared_log_error(y_test, Yhat),'.3f'))

medabserror = float(format(median_absolute_error(y_test, Yhat),'.3f'))

mpoisdev = float(format(mean_poisson_deviance(y_test, Yhat),'.3f'))

mgamdev = float(format(mean_gamma_deviance(y_test, Yhat),'.3f'))

rmsesm = float(format(np.sqrt(mean_squared_error(y_test, Yhat)),'.3f'))

#rtrsm = float(format(pipe.score(train_data[all_features],train_data['price']),'.3f'))

artrcm = float(format

               (adjustedR2

                (pipe.score

                 (train_data[['sqft_living']],

                  train_data['price']),train_data.shape[0],

                 len(['sqft_living'])

                ),'.3f')

              )

rtesm = float(format(pipe.score(X_test, y_test),'.3f'))

artecm = float(format

               (adjustedR2

                (pipe.score

                 (test_data[['sqft_living']],

                  test_data['price']),

                 test_data.shape[0],

                 len(['sqft_living'])

                ),'.3f')

              )

cv = float(format(cross_val_score(pipe,df[['sqft_living']],df['price'],cv=12).mean(),'.3f'))



print ("Average Price for Test Data:", y_test.mean())

print('Intercept: {}'.format(lr.intercept_))

print('Coefficient: {}'.format(lr.coef_))



r = evaluation.shape[0]



evaluation.loc[r] = ['Lasso Regression','Best Feature', max_err, mabserr, msqerr, msqlogerr, medabserror,mpoisdev, mgamdev, rmsesm,rtrsm,artrcm,'-',artecm,cv]

evaluation.sort_values(by = '12-Fold Cross Validation', ascending=False)
plt.figure(figsize = (26,6))



ax1 = sns.distplot(y_test, label = 'Actual values', color = 'DarkBlue', hist=False, bins=50)

sns.distplot(Yhat, color='Orange', label = 'Predicted values', hist=False, bins=50, ax=ax1)

plt.xlabel('Price distribution', fontsize=14)

plt.ylabel('Count', fontsize=14)

plt.title('Yhat and y_test distribution comparison - Lasso Regression - Best Feature', fontsize=18)
# Train and test

X_train = train_data[top_5]

y_train = train_data['price']

X_test = test_data[top_5]

y_test = test_data['price']



# Define the Input for the pipeline

Input = [('scale', StandardScaler()), ('Lasso', linear_model.Lasso(alpha = 0.5, precompute = False, random_state = 22))]



# Initialize the pipeline

pipe = Pipeline(Input)



# Train the model

pipe.fit(X_train, y_train)



# Make a prediction

Yhat = pipe.predict(X_test)
# Evaluation Metrics

max_err = float(format(max_error(y_test, Yhat),'.3f'))

mabserr = float(format(mean_absolute_error(y_test, Yhat),'.3f'))

msqerr = float(format(mean_squared_error(y_test, Yhat),'.3f'))

#msqlogerr = float(format(mean_squared_log_error(y_test, Yhat),'.3f'))#

medabserror = float(format(median_absolute_error(y_test, Yhat),'.3f'))

#mpoisdev = float(format(mean_poisson_deviance(y_test, Yhat),'.3f'))#

#mgamdev = float(format(mean_gamma_deviance(y_test, Yhat),'.3f'))#

rmsesm = float(format(np.sqrt(mean_squared_error(y_test, Yhat)),'.3f'))

rtrsm = float(format(pipe.score(train_data[top_5],train_data['price']),'.3f'))

artrcm = float(format

               (adjustedR2

                (pipe.score

                 (train_data[top_5],

                  train_data['price']),train_data.shape[0],

                 len(top_5)

                ),'.3f')

              )

rtesm = float(format(pipe.score(X_test, y_test),'.3f'))

artecm = float(format

               (adjustedR2

                (pipe.score

                 (test_data[top_5],

                  test_data['price']),

                 test_data.shape[0],

                 len(top_5)

                ),'.3f')

              )

cv = float(format(cross_val_score(pipe,df[top_5],df['price'],cv=12).mean(),'.3f'))



print ("Average Price for Test Data:", y_test.mean())

print('Intercept: {}'.format(lr.intercept_))

print('Coefficient: {}'.format(lr.coef_))



r = evaluation.shape[0]



evaluation.loc[r] = ['Lasso Regression','Top 5 Features by Pearson_coef', max_err, mabserr, msqerr, '-', medabserror,'-', '-', rmsesm,rtrsm,artrcm,rtesm,artecm,cv]

evaluation.sort_values(by = '12-Fold Cross Validation', ascending=False)
plt.figure(figsize = (26,6))



ax1 = sns.distplot(y_test, label = 'Actual values', color = 'DarkBlue', hist=False, bins=50)

sns.distplot(Yhat, color='Orange', label = 'Predicted values', hist=False, bins=50, ax=ax1)

plt.xlabel('Price distribution', fontsize=14)

plt.ylabel('Count', fontsize=14)

plt.title('Yhat and y_test distribution comparison - Lasso Regression - Top 5 Features', fontsize=18)
# Train and test

X_train = train_data[all_features]

y_train = train_data['price']

X_test = test_data[all_features]

y_test = test_data['price']



# Define the Input for the pipeline

Input = [('scale', StandardScaler()), ('Lasso', linear_model.Lasso(alpha = 1, precompute = False, max_iter = 50000, random_state = 22))]



# Initialize the pipeline

pipe = Pipeline(Input)



# Train the model

pipe.fit(X_train, y_train)



# Make a prediction

Yhat = pipe.predict(X_test)
# Evaluation Metrics

max_err = float(format(max_error(y_test, Yhat),'.3f'))

mabserr = float(format(mean_absolute_error(y_test, Yhat),'.3f'))

msqerr = float(format(mean_squared_error(y_test, Yhat),'.3f'))

#msqlogerr = float(format(mean_squared_log_error(y_test, Yhat),'.3f'))

medabserror = float(format(median_absolute_error(y_test, Yhat),'.3f'))

#mpoisdev = float(format(mean_poisson_deviance(y_test, Yhat),'.3f'))

#mgamdev = float(format(mean_gamma_deviance(y_test, Yhat),'.3f'))

rmsesm = float(format(np.sqrt(mean_squared_error(y_test, Yhat)),'.3f'))

rtrsm = float(format(pipe.score(train_data[all_features],train_data['price']),'.3f'))

artrcm = float(format

               (adjustedR2

                (pipe.score

                 (train_data[all_features],

                  train_data['price']),train_data.shape[0],

                 len(all_features)

                ),'.3f')

              )

rtesm = float(format(pipe.score(X_test, y_test),'.3f'))

artecm = float(format

               (adjustedR2

                (pipe.score

                 (test_data[all_features],

                  test_data['price']),

                 test_data.shape[0],

                 len(all_features)

                ),'.3f')

              )

cv = float(format(cross_val_score(pipe,df[top_5],df['price'],cv=12).mean(),'.3f'))



print ("Average Price for Test Data:", y_test.mean())

print('Intercept: {}'.format(lr.intercept_))

print('Coefficient: {}'.format(lr.coef_))



r = evaluation.shape[0]



evaluation.loc[r] = ['Lasso Regression','All Features from Pearson_coef', max_err, mabserr, msqerr, '-', medabserror,'-', '-', rmsesm,rtrsm,artrcm,rtesm,artecm,cv]

evaluation.sort_values(by = '12-Fold Cross Validation', ascending=False)
plt.figure(figsize = (26,6))



ax1 = sns.distplot(y_test, label = 'Actual values', color = 'DarkBlue', hist=False, bins=50)

sns.distplot(Yhat, color='Orange', label = 'Predicted values', hist=False, bins=50, ax=ax1)

plt.xlabel('Price distribution', fontsize=14)

plt.ylabel('Count', fontsize=14)

plt.title('Yhat and y_test distribution comparison - Lasso Regression - All Features', fontsize=18)
# Train and test.

X_train = train_data[['sqft_living']]

y_train = train_data['price']

X_test = test_data[['sqft_living']]

y_test = test_data['price']



# The Decision Tree Regression Model doesn't need data normalisation.



tree_depth = pd.DataFrame({'Model': [],

                           'Depth':[],

                           'Max Error':[],

                           'Mean Absolute Error' : [],

                           'Mean Squared Error' : [],

                           'Mean Squared Log Error' : [],

                           'Median Absolute Error' : [],

                           'Mean Poisson Deviance' : [],

                           'Mean Gamma Deviance': [],

                           'Root Mean Squared Error (RMSE)':[],

                           'R-squared (training)':[],

                           'Adjusted R-squared (training)':[],

                           'R-squared (test)':[],

                           'Adjusted R-squared (test)':[],

                           '12-Fold Cross Validation':[]})



# Initialize the model

for depth in range(1,20):

    tree = DecisionTreeRegressor(max_depth = depth)



    # Train the model

    tree.fit(X_train, y_train)



    # Evaluation Metrics

    max_err = float(format(max_error(y_test, Yhat),'.3f'))

    mabserr = float(format(mean_absolute_error(y_test, Yhat),'.3f'))

    msqerr = float(format(mean_squared_error(y_test, Yhat),'.3f'))

    #msqlogerr = float(format(mean_squared_log_error(y_test, Yhat),'.3f'))

    medabserror = float(format(median_absolute_error(y_test, Yhat),'.3f'))

    #mpoisdev = float(format(mean_poisson_deviance(y_test, Yhat),'.3f'))

    #mgamdev = float(format(mean_gamma_deviance(y_test, Yhat),'.3f'))

    rmsesm = float(format(np.sqrt(mean_squared_error(y_test, Yhat)),'.3f'))

    rtrsm = float(format(tree.score(train_data[['sqft_living']],train_data['price']),'.3f'))

    artrcm = float(format

                   (adjustedR2

                    (tree.score

                     (train_data[['sqft_living']],

                      train_data['price']),train_data.shape[0],

                     len(['sqft_living'])

                    ),'.3f')

                  )

    rtesm = float(format(tree.score(X_test, y_test),'.3f'))

    artecm = float(format

                   (adjustedR2

                    (tree.score

                     (test_data[['sqft_living']],

                      test_data['price']),

                     test_data.shape[0],

                     len(['sqft_living'])

                    ),'.3f')

                  )

    cv = float(format(cross_val_score(tree,df[top_5],df['price'],cv=12).mean(),'.3f'))



    r = tree_depth.shape[0]



    tree_depth.loc[r] = ['Decision Tree Regression',depth, max_err, mabserr, msqerr, '-', medabserror,'-', '-', rmsesm,rtrsm,artrcm,rtesm,artecm,cv]



tree_depth.sort_values(by = '12-Fold Cross Validation', ascending=False, inplace=True)

tree_depth.reset_index(drop = True, inplace = True)

tree_depth.head(3)
plt.figure(figsize=(10,6))



max_depth = int(max(tree_depth['Depth']))

best_depth = tree_depth['Depth'][0]

max_cv_score = max(tree_depth['12-Fold Cross Validation'])





ax1 = sns.lineplot(x = tree_depth['Depth'], y = tree_depth['12-Fold Cross Validation'], color = 'Red', label="Cross Valudation")

sns.lineplot(x = tree_depth['Depth'], y = tree_depth['R-squared (test)'], label='R-squared (test)', color='Green')

sns.lineplot(x = tree_depth['Depth'], y = tree_depth['R-squared (training)'], label='R-squared (training)', color="orange")



plt.xlabel('Max Depth Level', fontsize = 14)

plt.ylabel('Evaluation Score', fontsize = 14)

plt.title('Cross Validation Score per Depth Level', fontsize = 18)
# Train and test. Does it make sense to trtain a decision tree with one feature only?

X_train = train_data[['sqft_living']]

y_train = train_data['price']

X_test = test_data[['sqft_living']]

y_test = test_data['price']



# The Decision Tree Regression Model doesn't need data normalisation.

# Initialize the model

tree = DecisionTreeRegressor(max_depth = best_depth)



# Train the model

tree.fit(X_train, y_train)



# Make a prediction

tree.predict(X_test)
# Evaluation Metrics

max_err = float(format(max_error(y_test, Yhat),'.3f'))

mabserr = float(format(mean_absolute_error(y_test, Yhat),'.3f'))

msqerr = float(format(mean_squared_error(y_test, Yhat),'.3f'))

#msqlogerr = float(format(mean_squared_log_error(y_test, Yhat),'.3f'))

medabserror = float(format(median_absolute_error(y_test, Yhat),'.3f'))

#mpoisdev = float(format(mean_poisson_deviance(y_test, Yhat),'.3f'))

#mgamdev = float(format(mean_gamma_deviance(y_test, Yhat),'.3f'))

rmsesm = float(format(np.sqrt(mean_squared_error(y_test, Yhat)),'.3f'))

rtrsm = float(format(tree.score(train_data[['sqft_living']],train_data['price']),'.3f'))

artrcm = float(format

               (adjustedR2

                (tree.score

                 (train_data[['sqft_living']],

                  train_data['price']),train_data.shape[0],

                 len(['sqft_living'])

                ),'.3f')

              )

rtesm = float(format(tree.score(X_test, y_test),'.3f'))

artecm = float(format

               (adjustedR2

                (tree.score

                 (test_data[['sqft_living']],

                  test_data['price']),

                 test_data.shape[0],

                 len(['sqft_living'])

                ),'.3f')

              )

cv = float(format(cross_val_score(tree,df[top_5],df['price'],cv=12).mean(),'.3f'))



print ("Average Price for Test Data:", y_test.mean())

print('Intercept: {}'.format(lr.intercept_))

print('Coefficient: {}'.format(lr.coef_))



r = evaluation.shape[0]



evaluation.loc[r] = ['Decision Tree Regression','Max Depth = {} Best Feature'.format(best_depth), max_err, mabserr, msqerr, '-', medabserror,'-', '-', rmsesm,rtrsm,artrcm,rtesm,artecm,cv]

evaluation.sort_values(by = '12-Fold Cross Validation', ascending=False)
# Plot the results

plt.figure(figsize=(16,6))

plt.scatter(df[['sqft_living']], df[['price']],

            color="DarkBlue", label="Actual Values", alpha=0.1)

plt.plot(X_test, Yhat, color="Coral",

         label="Decision Tree", linewidth=1)

#plt.plot(X_test, y_2, color="yellowgreen", label="max_depth=5", linewidth=2)

plt.xlabel("data")

plt.ylabel("target")

plt.title("Decision Tree Regression")

plt.legend()

plt.show()





#plt.scatter(X_test,y_test,color="DarkBlue", label="Actual values", alpha=.1)

#plt.plot(X_test,lr.predict(X_test),color='Coral', label="Predicted Regression Line")
# Train and test.

X_train = train_data[top_5]

y_train = train_data['price']

X_test = test_data[top_5]

y_test = test_data['price']



# The Decision Tree Regression Model doesn't need data normalisation.



tree_depth = pd.DataFrame({'Model': [],

                           'Depth':[],

                           'Max Error':[],

                           'Mean Absolute Error' : [],

                           'Mean Squared Error' : [],

                           'Mean Squared Log Error' : [],

                           'Median Absolute Error' : [],

                           'Mean Poisson Deviance' : [],

                           'Mean Gamma Deviance': [],

                           'Root Mean Squared Error (RMSE)':[],

                           'R-squared (training)':[],

                           'Adjusted R-squared (training)':[],

                           'R-squared (test)':[],

                           'Adjusted R-squared (test)':[],

                           '12-Fold Cross Validation':[]})



# Initialize the model

for depth in range(1,20):

    tree = DecisionTreeRegressor(max_depth = depth)



    # Train the model

    tree.fit(X_train, y_train)



    # Evaluation Metrics

    max_err = float(format(max_error(y_test, Yhat),'.3f'))

    mabserr = float(format(mean_absolute_error(y_test, Yhat),'.3f'))

    msqerr = float(format(mean_squared_error(y_test, Yhat),'.3f'))

    #msqlogerr = float(format(mean_squared_log_error(y_test, Yhat),'.3f'))

    medabserror = float(format(median_absolute_error(y_test, Yhat),'.3f'))

    #mpoisdev = float(format(mean_poisson_deviance(y_test, Yhat),'.3f'))

    #mgamdev = float(format(mean_gamma_deviance(y_test, Yhat),'.3f'))

    rmsesm = float(format(np.sqrt(mean_squared_error(y_test, Yhat)),'.3f'))

    rtrsm = float(format(tree.score(train_data[top_5],train_data['price']),'.3f'))

    artrcm = float(format

                   (adjustedR2

                    (tree.score

                     (train_data[top_5],

                      train_data['price']),train_data.shape[0],

                     len(top_5)

                    ),'.3f')

                  )

    rtesm = float(format(tree.score(X_test, y_test),'.3f'))

    artecm = float(format

                   (adjustedR2

                    (tree.score

                     (test_data[top_5],

                      test_data['price']),

                     test_data.shape[0],

                     len(top_5)

                    ),'.3f')

                  )

    cv = float(format(cross_val_score(tree,df[top_5],df['price'],cv=12).mean(),'.3f'))



    r = tree_depth.shape[0]



    tree_depth.loc[r] = ['Decision Tree Regression',depth, max_err, mabserr, msqerr, '-', medabserror,'-', '-', rmsesm,rtrsm,artrcm,rtesm,artecm,cv]



tree_depth.sort_values(by = '12-Fold Cross Validation', ascending=False, inplace=True)

tree_depth.reset_index(drop = True, inplace = True)

tree_depth.head(3)
plt.figure(figsize=(10,6))



max_depth = int(max(tree_depth['Depth']))

best_depth = tree_depth['Depth'][0]

max_cv_score = max(tree_depth['12-Fold Cross Validation'])





ax1 = sns.lineplot(x = tree_depth['Depth'], y = tree_depth['12-Fold Cross Validation'], color = 'Red', label="Cross Valudation")

sns.lineplot(x = tree_depth['Depth'], y = tree_depth['R-squared (test)'], label='R-squared (test)', color='Green')

sns.lineplot(x = tree_depth['Depth'], y = tree_depth['R-squared (training)'], label='R-squared (training)', color="orange")



plt.xlabel('Max Depth Level', fontsize = 14)

plt.ylabel('Evaluation Score', fontsize = 14)

plt.title('Cross Validation Score per Depth Level', fontsize = 18)
# Train and test. Does it make sense to trtain a decision tree with one feature only?

X_train = train_data[top_5]

y_train = train_data['price']

X_test = test_data[top_5]

y_test = test_data['price']



# The Decision Tree Regression Model doesn't need data normalisation.

# Initialize the model

tree = DecisionTreeRegressor(max_depth = best_depth)



# Train the model

tree.fit(X_train, y_train)



# Make a prediction

tree.predict(X_test)
# Evaluation Metrics

max_err = float(format(max_error(y_test, Yhat),'.3f'))

mabserr = float(format(mean_absolute_error(y_test, Yhat),'.3f'))

msqerr = float(format(mean_squared_error(y_test, Yhat),'.3f'))

#msqlogerr = float(format(mean_squared_log_error(y_test, Yhat),'.3f'))

medabserror = float(format(median_absolute_error(y_test, Yhat),'.3f'))

#mpoisdev = float(format(mean_poisson_deviance(y_test, Yhat),'.3f'))

#mgamdev = float(format(mean_gamma_deviance(y_test, Yhat),'.3f'))

rmsesm = float(format(np.sqrt(mean_squared_error(y_test, Yhat)),'.3f'))

rtrsm = float(format(tree.score(train_data[top_5],train_data['price']),'.3f'))

artrcm = float(format

               (adjustedR2

                (tree.score

                 (train_data[top_5],

                  train_data['price']),train_data.shape[0],

                 len(top_5)

                ),'.3f')

              )

rtesm = float(format(tree.score(X_test, y_test),'.3f'))

artecm = float(format

               (adjustedR2

                (tree.score

                 (test_data[top_5],

                  test_data['price']),

                 test_data.shape[0],

                 len(top_5)

                ),'.3f')

              )

cv = float(format(cross_val_score(tree,df[top_5],df['price'],cv=12).mean(),'.3f'))



print ("Average Price for Test Data:", y_test.mean())

print('Intercept: {}'.format(lr.intercept_))

print('Coefficient: {}'.format(lr.coef_))



r = evaluation.shape[0]



evaluation.loc[r] = ['Decision Tree Regression','Depth = {} - Top 5 Features by Pearson_coef'.format(best_depth), max_err, mabserr, msqerr, '-', medabserror,'-', '-', rmsesm,rtrsm,artrcm,rtesm,artecm,cv]

evaluation.sort_values(by = '12-Fold Cross Validation', ascending=False)
plt.figure(figsize = (26,6))



ax1 = sns.distplot(y_test, label = 'Actual values', color = 'DarkBlue', hist=False, bins=50)

sns.distplot(Yhat, color='Orange', label = 'Predicted values', hist=False, bins=50, ax=ax1)

plt.xlabel('Price distribution', fontsize=14)

plt.ylabel('Count', fontsize=14)

plt.title('Yhat and y_test distribution comparison - Decision Tree Regression - Top 5 Features', fontsize=18)
# Train and test.

X_train = train_data[all_features]

y_train = train_data['price']

X_test = test_data[all_features]

y_test = test_data['price']



# The Decision Tree Regression Model doesn't need data normalisation.



tree_depth = pd.DataFrame({'Model': [],

                           'Depth':[],

                           'Max Error':[],

                           'Mean Absolute Error' : [],

                           'Mean Squared Error' : [],

                           'Mean Squared Log Error' : [],

                           'Median Absolute Error' : [],

                           'Mean Poisson Deviance' : [],

                           'Mean Gamma Deviance': [],

                           'Root Mean Squared Error (RMSE)':[],

                           'R-squared (training)':[],

                           'Adjusted R-squared (training)':[],

                           'R-squared (test)':[],

                           'Adjusted R-squared (test)':[],

                           '12-Fold Cross Validation':[]})



# Initialize the model

for depth in range(1,20):

    tree = DecisionTreeRegressor(max_depth = depth)



    # Train the model

    tree.fit(X_train, y_train)



    # Evaluation Metrics

    max_err = float(format(max_error(y_test, Yhat),'.3f'))

    mabserr = float(format(mean_absolute_error(y_test, Yhat),'.3f'))

    msqerr = float(format(mean_squared_error(y_test, Yhat),'.3f'))

    #msqlogerr = float(format(mean_squared_log_error(y_test, Yhat),'.3f'))

    medabserror = float(format(median_absolute_error(y_test, Yhat),'.3f'))

    #mpoisdev = float(format(mean_poisson_deviance(y_test, Yhat),'.3f'))

    #mgamdev = float(format(mean_gamma_deviance(y_test, Yhat),'.3f'))

    rmsesm = float(format(np.sqrt(mean_squared_error(y_test, Yhat)),'.3f'))

    rtrsm = float(format(tree.score(train_data[all_features],train_data['price']),'.3f'))

    artrcm = float(format

                   (adjustedR2

                    (tree.score

                     (train_data[all_features],

                      train_data['price']),train_data.shape[0],

                     len(all_features)

                    ),'.3f')

                  )

    rtesm = float(format(tree.score(X_test, y_test),'.3f'))

    artecm = float(format

                   (adjustedR2

                    (tree.score

                     (test_data[all_features],

                      test_data['price']),

                     test_data.shape[0],

                     len(all_features)

                    ),'.3f')

                  )

    cv = float(format(cross_val_score(tree,df[all_features],df['price'],cv=12).mean(),'.3f'))



    r = tree_depth.shape[0]



    tree_depth.loc[r] = ['Decision Tree Regression',depth, max_err, mabserr, msqerr, '-', medabserror,'-', '-', rmsesm,rtrsm,artrcm,rtesm,artecm,cv]



tree_depth.sort_values(by = '12-Fold Cross Validation', ascending=False, inplace=True)

tree_depth.reset_index(drop = True, inplace = True)

tree_depth.head(3)
plt.figure(figsize=(10,6))



max_depth = int(max(tree_depth['Depth']))

best_depth = tree_depth['Depth'][0]

max_cv_score = max(tree_depth['12-Fold Cross Validation'])





ax1 = sns.lineplot(x = tree_depth['Depth'], y = tree_depth['12-Fold Cross Validation'], color = 'Red', label="Cross Valudation")

sns.lineplot(x = tree_depth['Depth'], y = tree_depth['R-squared (test)'], label='R-squared (test)', color='Green')

sns.lineplot(x = tree_depth['Depth'], y = tree_depth['R-squared (training)'], label='R-squared (training)', color="orange")



plt.xlabel('Max Depth Level', fontsize = 14)

plt.ylabel('Evaluation Score', fontsize = 14)

plt.title('Cross Validation Score per Depth Level', fontsize = 18)
# Train and test. Does it make sense to trtain a decision tree with one feature only?

X_train = train_data[all_features]

y_train = train_data['price']

X_test = test_data[all_features]

y_test = test_data['price']



# The Decision Tree Regression Model doesn't need data normalisation.

# Initialize the model

tree = DecisionTreeRegressor(max_depth = best_depth)



# Train the model

tree.fit(X_train, y_train)



# Make a prediction

tree.predict(X_test)
# Evaluation Metrics

max_err = float(format(max_error(y_test, Yhat),'.3f'))

mabserr = float(format(mean_absolute_error(y_test, Yhat),'.3f'))

msqerr = float(format(mean_squared_error(y_test, Yhat),'.3f'))

#msqlogerr = float(format(mean_squared_log_error(y_test, Yhat),'.3f'))

medabserror = float(format(median_absolute_error(y_test, Yhat),'.3f'))

#mpoisdev = float(format(mean_poisson_deviance(y_test, Yhat),'.3f'))

#mgamdev = float(format(mean_gamma_deviance(y_test, Yhat),'.3f'))

rmsesm = float(format(np.sqrt(mean_squared_error(y_test, Yhat)),'.3f'))

rtrsm = float(format(tree.score(train_data[all_features],train_data['price']),'.3f'))

artrcm = float(format

               (adjustedR2

                (tree.score

                 (train_data[all_features],

                  train_data['price']),train_data.shape[0],

                 len(all_features)

                ),'.3f')

              )

rtesm = float(format(tree.score(X_test, y_test),'.3f'))

artecm = float(format

               (adjustedR2

                (tree.score

                 (test_data[all_features],

                  test_data['price']),

                 test_data.shape[0],

                 len(all_features)

                ),'.3f')

              )

cv = float(format(cross_val_score(tree,df[all_features],df['price'],cv=12).mean(),'.3f'))



print ("Average Price for Test Data:", y_test.mean())

print('Intercept: {}'.format(lr.intercept_))

print('Coefficient: {}'.format(lr.coef_))



r = evaluation.shape[0]



evaluation.loc[r] = ['Decision Tree Regression','Depth = {} - All Features from Pearson_coef'.format(best_depth), max_err, mabserr, msqerr, '-', medabserror,'-', '-', rmsesm,rtrsm,artrcm,rtesm,artecm,cv]

evaluation.sort_values(by = '12-Fold Cross Validation', ascending=False)
plt.figure(figsize = (26,6))



ax1 = sns.distplot(y_test, label = 'Actual values', color = 'DarkBlue', hist=False, bins=50)

sns.distplot(Yhat, color='Orange', label = 'Predicted values', hist=False, bins=50, ax=ax1)

plt.xlabel('Price distribution', fontsize=14)

plt.ylabel('Count', fontsize=14)

plt.title('Yhat and y_test distribution comparison - Decision Tree Regression - All Features', fontsize=18)
# train and test

X_train = train_data[all_features]

y_train = train_data['price']

X_test = test_data[all_features]

y_test = test_data['price']



# Define a pipeline

Input = [('scaler', StandardScaler()), ('MLPR', MLPRegressor(activation = 'tanh',

                                                            solver='sgd',

                                                            learning_rate = 'adaptive',

                                                            max_iter = 2000))]

pipe = Pipeline(Input)



# Train the model

pipe.fit(X_train, y_train)



# Make a prediction

Yhat = pipe.predict(X_test)
# Evaluation Metrics

max_err = float(format(max_error(y_test, Yhat),'.3f'))

mabserr = float(format(mean_absolute_error(y_test, Yhat),'.3f'))

msqerr = float(format(mean_squared_error(y_test, Yhat),'.3f'))

#msqlogerr = float(format(mean_squared_log_error(y_test, Yhat),'.3f'))

medabserror = float(format(median_absolute_error(y_test, Yhat),'.3f'))

#mpoisdev = float(format(mean_poisson_deviance(y_test, Yhat),'.3f'))

#mgamdev = float(format(mean_gamma_deviance(y_test, Yhat),'.3f'))

rmsesm = float(format(np.sqrt(mean_squared_error(y_test, Yhat)),'.3f'))

rtrsm = float(format(pipe.score(train_data[all_features],train_data['price']),'.3f'))

artrcm = float(format

               (adjustedR2

                (pipe.score

                 (train_data[all_features],

                  train_data['price']),train_data.shape[0],

                 len(all_features)

                ),'.3f')

              )

rtesm = float(format(pipe.score(X_test, y_test),'.3f'))

artecm = float(format

               (adjustedR2

                (pipe.score

                 (test_data[all_features],

                  test_data['price']),

                 test_data.shape[0],

                 len(all_features)

                ),'.3f')

              )

cv = float(format(cross_val_score(pipe,df[all_features],df['price'],cv=12).mean(),'.3f'))



print ("Average Price for Test Data:", y_test.mean())

print('Intercept: {}'.format(lr.intercept_))

print('Coefficient: {}'.format(lr.coef_))



r = evaluation.shape[0]



evaluation.loc[r] = ['Multi_layer Perceptron Regressor','All Features from Pearson_coef'.format(best_depth), max_err, mabserr, msqerr, '-', medabserror,'-', '-', rmsesm,rtrsm,artrcm,rtesm,artecm,cv]

evaluation.sort_values(by = '12-Fold Cross Validation', ascending=False)

plt.figure(figsize = (26,6))



ax1 = sns.distplot(y_test, label = 'Actual values', color = 'DarkBlue', hist=False, bins=50)

sns.distplot(Yhat, color='Orange', label = 'Predicted values', hist=False, bins=50, ax=ax1)

plt.xlabel('Price distribution', fontsize=14)

plt.ylabel('Count', fontsize=14)

plt.title('Yhat and y_test distribution comparison - Decision Tree Regression - All Features', fontsize=18)
evaluation.sort_values(by = '12-Fold Cross Validation', ascending=False)