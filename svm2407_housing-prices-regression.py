## to do
# import libraries



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from scipy import stats

from sklearn import datasets, linear_model, metrics

from statsmodels.stats.outliers_influence import variance_inflation_factor
# import data



path = "/kaggle/input/housesalesprediction/kc_house_data.csv"



df_house = pd.read_csv(path)

df_house
display(df_house.describe())

display(df_house.info())
df_house['has_basement'] = np.where(df_house['sqft_basement'] > 0, 1, 0)



df_house.head()
from math import radians, sin, cos, asin, sqrt



def haversine(lon1, lat1, lon2, lat2):

    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    dlon = lon2 - lon1

    dlat = lat2 - lat1

    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2

    return 2 * 6371 * asin(sqrt(a))



latitude_city = 47.610515

longitude_city = -122.33465413



df_house['distance_to_city'] = df_house.apply(lambda row: haversine(latitude_city, longitude_city, row['lat'], row['long']), axis=1)



df_house
# check if prices are higher or lower er certain zipcodes



f, axe = plt.subplots(1, 1,figsize=(25,5))

sns.boxplot(x=df_house['zipcode'],y=df_house['price'], ax=axe)

sns.despine(left=True, bottom=True)

axe.yaxis.tick_left()

axe.set(xlabel='Zipcode', ylabel='Price')
# create new variables for top x and bottom x zipcodes by price

# trial and error to get a better x



med_price_zip = df_house.groupby(['zipcode']).agg({'price': 'median', 'id': "count"}).sort_values('price', ascending = False)



zipcode_topx = np.array([med_price_zip[c].nlargest(30).index.values for c in med_price_zip])[0]

zipcode_bottomx = np.array([med_price_zip[c].nsmallest(30).index.values for c in med_price_zip])[0]



print(zipcode_topx)

print(zipcode_bottomx)



df_house["is_topx_zipcode"] = [1 if x in list(zipcode_topx) else 0 for x in df_house["zipcode"]]

df_house["is_bottomx_zipcode"] = [1 if x in list(zipcode_bottomx) else 0 for x in df_house["zipcode"]]



df_house
# histograms for all variables

# copied from: https://www.kaggle.com/burhanykiyakoglu/predicting-house-prices



h = df_house.drop(['id', 'date'], axis = 1).hist(bins = 25, figsize = (16,16), xlabelsize = '10', ylabelsize = '10', xrot = -15)

sns.despine(left = True, bottom = True)

[x.title.set_size(12) for x in h.ravel()];

[x.yaxis.tick_left() for x in h.ravel()];
# graph distribution



for col in list(df_house.drop(['id', 'date'], axis = 1).columns):    

    plt.figure(figsize=(12,4))

    plt.subplot(1,2,1)

    sns.distplot(df_house[col].dropna(), fit=stats.norm);

    plt.subplot(1,2,2)

    _=stats.probplot(df_house[col].dropna(), plot=plt)



plt.show()
# joinplots for all x variables in relation to x, to visualize the bivariate distribution



for col in list(df_house.drop(['id', 'date','price'], axis = 1).columns):

    sns.jointplot(x = col, y = "price", data = df_house, kind = 'reg', size = 5)



plt.show()
# create correlation matrix

corr_matrix = df_house.corr()





# set up mask to hide upper triangle



mask = np.zeros_like(corr_matrix, dtype = np.bool)

mask[np.triu_indices_from(mask)] = True





# create seaborn heatmap



f, ax = plt.subplots(figsize = (16, 10)) 



heatmap = sns.heatmap(corr_matrix, 

                      mask = mask,

                      #square = True, # Makes each cell square-shaped

                      linewidths = .5, # set width of the lines that will divide each cell to .5

                      cmap = "coolwarm", # map data values to the coolwarm color space

                      cbar_kws = {'shrink': .4, # shrink the legend size and label tick marks at [-1, -.5, 0, 0.5, 1]

                                "ticks" : [-1, -.5, 0, 0.5, 1]},

                      vmin = -1, # Set min value for color bar

                      vmax = 1, # Set max value for color bar

                      annot = True, # Turn on annotations for the correlation values

                      fmt='.2f', # String formatting code to use when adding annotations

                      annot_kws = {"size": 12}) # Set annotations to size 12



# add title

plt.title('House Sales King County - Correlation Heatmap', 

              fontsize=14, 

              fontweight='bold')



# add the column names as labels

ax.set_xticklabels(corr_matrix.columns, rotation = 90) # Add column names to the x labels and rotate text to 90 degrees

ax.set_yticklabels(corr_matrix.columns, rotation = 0) # Add column names to the y labels and rotate text to 0 degrees



sns.set_style({'xtick.bottom': True}, {'ytick.left': True}) # Show tickmarks on bottom and left of heatmap
# check for skewed data



skew = df_house.skew(axis = 0, skipna = True) 



# show all variables which are skewed to the right (skew > 1)

print('Variables which are skewed to the right:')

print(skew[skew > 1].sort_values(ascending = False))

print()



# show all variables which are skewed to the left (skew < -1)

print('Variables which are skewed to the left:')

print(skew[skew < -1].sort_values())
df_house['price_log'] = np.log(df_house['price'])

df_house['sqft_lot_log'] = np.log(df_house['sqft_lot'])

df_house['distance_to_city_log'] = np.log(df_house['distance_to_city'])



df_house
# taken from https://www.kaggle.com/burhanykiyakoglu/predicting-house-prices



# add the age of the buildings when the houses were sold as a new column

age = df_house['date'].astype(str).str[:4].astype(int) - df_house['yr_built']



# partition the age into bins

bins = [-2, 0, 5, 10, 25, 50, 75, 100, 100000]

labels = ['<1', '1-5', '6-10', '11-25', '26-50', '51-75', '76-100', '>100']

df_house['age_binned'] = pd.cut(age, bins = bins, labels = labels)



# histograms for the binned columns

plot = sns.countplot(df_house['age_binned'])

for p in plot.patches:

    height = p.get_height()

    plot.text(p.get_x() + p.get_width() / 2, height + 50, height, ha = "center")   



ax.set(xlabel='Age')

ax.yaxis.tick_left()



# transform the factor values to be able to use in the model

df_house = pd.get_dummies(df_house, columns=['age_binned'])



df_house
# joinplots for all x variables in relation to x, to visualize the bivariate distribution



for col in list(df_house.drop(['id', 'date','price'], axis = 1).columns):

    sns.jointplot(x = col, y = "price_log", data = df_house, kind = 'reg', size = 5)



plt.show()
# create correlation matrix

corr_matrix = df_house.corr()

#display(corr_matrix)





# set up mask to hide upper triangle



mask = np.zeros_like(corr_matrix, dtype = np.bool)

mask[np.triu_indices_from(mask)] = True





# create seaborn heatmap



f, ax = plt.subplots(figsize = (16, 10)) 



heatmap = sns.heatmap(corr_matrix, 

                      mask = mask,

                      #square = True, # Makes each cell square-shaped

                      linewidths = .5, # set width of the lines that will divide each cell to .5

                      cmap = "coolwarm", # map data values to the coolwarm color space

                      cbar_kws = {'shrink': .4, # shrink the legend size and label tick marks at [-1, -.5, 0, 0.5, 1]

                                "ticks" : [-1, -.5, 0, 0.5, 1]},

                      vmin = -1, # Set min value for color bar

                      vmax = 1, # Set max value for color bar

                      annot = True, # Turn on annotations for the correlation values

                      fmt='.2f', # String formatting code to use when adding annotations

                      annot_kws = {"size": 12}) # Set annotations to size 12



# add title

plt.title('House Sales King County - Correlation Heatmap', 

              fontsize=14, 

              fontweight='bold')



# add the column names as labels

ax.set_xticklabels(corr_matrix.columns, rotation = 90) # Add column names to the x labels and rotate text to 90 degrees

ax.set_yticklabels(corr_matrix.columns, rotation = 0) # Add column names to the y labels and rotate text to 0 degrees



sns.set_style({'xtick.bottom': True}, {'ytick.left': True}) # Show tickmarks on bottom and left of heatmap
# split in train and test dataset, and x and y



from sklearn.model_selection import train_test_split



X = df_house.drop(['id', 'date', # variables not used from start

                   'price', 'price_log', # exclude dependent variables

                   'sqft_lot', 'distance_to_city', # exclude original variables after creating log transformatian

                   'sqft_basement','yr_built', 'zipcode', # exclude original variables after transformation

                   'sqft_above'], # remove to avoid multicoliniarity

                  axis = 1)

y = df_house['price_log']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)



print(X_train.shape)

print(X_test.shape)
# create linear model



from sklearn.linear_model import LinearRegression



lr = LinearRegression()

lrmodel1 = lr.fit(X_train, y_train)
# print model coefficients



coef_lr = pd.Series(lrmodel1.coef_, index = X_train.columns)

print('Intercept:', lrmodel1.intercept_)

print()

print('Coefficients:')

print(coef_lr.round(4))
# get p values for coefficients with statsmodel



import statsmodels.api as sm



#Fitting sm.OLS model

X_1 = sm.add_constant(X_train)

model = sm.OLS(y_train,X_1).fit()

print(model.summary())
# predict y values based on model coefficients



pred_train = lrmodel1.predict(X_train)

pred_test = lr.predict(X_test)



plt.scatter(y_test,pred_test)

plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], "r")

plt.xlabel('y actual')

plt.ylabel('y predicted')
# return predictions obtained for each element when it was in the test set



from sklearn.model_selection import cross_val_predict



y_cross = cross_val_predict(lrmodel1, X, y, cv = 5)



plt.scatter(y, y_cross)

plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], "r")

plt.xlabel('y actual')

plt.ylabel('y predicted (cross)')
# create probability plot



residuals = y_train - pred_train.reshape(-1)



plt.figure(figsize=(7,7))

stats.probplot(residuals, dist="norm", plot=plt)

plt.title("Normal Q-Q Plot")

print("If the residuals (blue dots) fall on the red line, the residuals are approximately normally distributed")

plt.show()

print()





# Kolmogorov-Smirnov test

# if the test is significant, this indicates that the model’s residuals are not normally distributed



kstest = stats.kstest(residuals, 'norm')



print("Kolmogorov-Smirnov:")

print(kstest)

if kstest[1] < 0.05:

    print("Evidence that the residuals are not normally distributed")

    print('Assumption not satisfied')

else:

    print("No evidence that the residuals are not normally distributed")

    print('Assumption satisfied')

print()





# Shapiro Wilk test

# if the test is significant, this indicates that the model’s residuals are not normally distributed



shapiro = stats.shapiro(residuals)



print("Shapiro Wilk:")

print(shapiro)

if shapiro[1] < 0.05:

    print("Evidence that the residuals are not normally distributed")

    print('Assumption not satisfied')

else:

    print("No evidence that the residuals are not normally distributed")

    print('Assumption satisfied')
import statsmodels.stats.api as sms



dw = sms.durbin_watson(residuals)



print('Durbin-Watson: {:.3f}'.format(dw))

if dw < 1.5:

    print('Signs of positive autocorrelation')

    print('Assumption not satisfied')

elif dw > 2.5:

    print('Signs of negative autocorrelation')

    print('Assumption not satisfied')

else:

    print('Little to no autocorrelation')

    print('Assumption satisfied')
# residual plot



sns.residplot(pred_train, y_train, lowess = True,

                                  line_kws = {'color': 'red', 'lw': 1, 'alpha': 1})

plt.xlabel("Predicted Y")

plt.ylabel("Residual")

plt.title('Residual plot')

plt.show()
# show all correlations above 0.8



cor = X_train.corr().round(2).abs() # create correlation df with rounded absolute values

cor = cor.unstack() # unstack to one row per correlation

cor = cor.reset_index(drop=False) # create new index instead of variable names as index

cor = cor[cor['level_0'] != cor['level_1']] # remove correlation for each variable with itsself

cor = cor[cor[0] >= 0.80] # show only correlations above this level

cor = cor.iloc[0:-1:2] # there are double rows for each correlation. take only one row. this is not fool proof !

cor.sort_values([0], ascending = False, inplace = True) # sort from high correlation to low

print(cor)
# calculate vif

# situations in which a high VIF is not a problem and can be safely ignored:

# 1. The variables with high VIFs are control variables, and the variables of interest do not have high VIFs.

# 2. The high VIFs are caused by the inclusion of powers or products of other variables. 

# 3. The variables with high VIFs are indicator (dummy) variables that represent a categorical variable with three+ categories.



vif = pd.DataFrame()

vif["VIF Factor"] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]

vif["features"] = X_train.columns

vif["VIF Factor"] = vif["VIF Factor"].round(1)

vif.sort_values(['VIF Factor'], ascending = False, inplace = True)

vif["VIF Factor"] = vif["VIF Factor"].astype(str)

vif
# calculate MSQE, R2, and R2 adjusted, for train and test model



from sklearn.metrics import mean_squared_error, r2_score 

from sklearn.model_selection import cross_val_score



print("Train model scores")



mse = np.sqrt(mean_squared_error(y_train, pred_train)) 

print('Root mean square error:', mse) 



r2 = r2_score(y_train, pred_train)

print('R2: {:.2%}'.format(r2))



n = X_train.shape[0]

p = X_train.shape[1]

r2_adj = 1 - (1 - r2) * (n - 1) / (n - p - 1)

print('R2 Adjusted: {:.2%}'.format(r2_adj))

print()





print("Test model scores")



mse = np.sqrt(mean_squared_error(y_test,pred_test)) 

print('Root mean square error:', mse) 



r2 = r2_score(y_test, pred_test)

print('R2: {:.2%}'.format(r2))



n = X_test.shape[0]

p = X_test.shape[1]

r2_adj = 1 - (1 - r2) * (n - 1) / (n - p - 1)

print('R2 Adjusted: {:.2%}'.format(r2_adj))

print()





print("Cross validation scores")



scores = cross_val_score(lrmodel1, X, y, cv = 5)

print("All scores:",(scores * 100).round(2))

print('Average cross validation score: {:.2%}'.format(np.mean(scores)))
# feature selection with RFE



from sklearn.feature_selection import RFE



# choose optimal # of features



#no of features

columns_count = len(X_train.columns)

nof_list = np.arange(1, columns_count + 1)

high_score = 0



#Variable to store the optimum features

nof = 0

score_list = []



for n in range(len(nof_list)):

    #X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = 0)

    model = LinearRegression()

    rfe = RFE(model, nof_list[n])

    X_train_rfe = rfe.fit_transform(X_train, y_train)

    X_test_rfe = rfe.transform(X_test)

    model.fit(X_train_rfe, y_train)

    score = model.score(X_test_rfe, y_test)

    score_list.append(score)

    if(score > high_score):

        high_score = score

        nof = nof_list[n]

print("Optimum number of features: %d" %nof)

print("Score with %d features: %f" % (nof, high_score))

print()





# run RFE 



cols = list(X.columns)

model = LinearRegression()



#Initializing RFE model

rfe = RFE(model, nof)



#Transforming data using RFE

X_rfe = rfe.fit_transform(X,y)



#Fitting the data to model

model.fit(X_rfe,y)

temp = pd.Series(rfe.support_,index = cols)

selected_features_rfe = temp[temp==True].index

print("Selected features:", list(selected_features_rfe))