import pandas as pd

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split

from matplotlib import pyplot as plt

import numpy as np

import warnings

warnings.simplefilter('ignore') #supress all future warnings

#from sklearn.tree import DecisionTreeRegressor



#load training and test data

home_data = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")

test_data = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")





corr_vals = home_data.corrwith(home_data.SalePrice) #correlating variables

corr_vals.sort_values(ascending = False,inplace = True) #sorting them from high to low



plt.rcParams["figure.figsize"]=25,5; fig, ax = plt.subplots(2)

ax[0].bar(corr_vals.index[1:21],corr_vals[1:21]);ax[0].set_ylabel('Correlation')

ax[1].bar(corr_vals.index[1:21],(home_data.shape[0]- home_data[corr_vals.index[1:21]].count()),color = 'tab:red');plt.ylabel('# of Missing Values') #total entries - present entries

plt.show()
plt.rcParams["figure.figsize"]=25,2.5;fig2,ax2 = plt.subplots(); 

ax2.bar(corr_vals.index[1:21],(test_data.shape[0]- test_data[corr_vals.index[1:21]].count()),color = 'tab:red');plt.ylabel('# of Missing Values') ;plt.show()
def train_by_feat(y,features,home_data):

    #Trains model based on list of features, splits data into training and validation by the features.

    X = home_data[features]

    train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

    

    model = RandomForestRegressor(random_state = 1)

    model.fit(train_X,train_y)

    feat_val_predictions = model.predict(val_X)

    feat_val_mae = mean_absolute_error(feat_val_predictions, val_y)

    

    return feat_val_mae
first_opti = pd.Series({num:train_by_feat(home_data.SalePrice,corr_vals.index[1:num+1],home_data) for num in np.arange(10)+1}) #storing MEA vs # of model params



plt.rcParams["figure.figsize"]=8,5;plt.plot(first_opti.index,first_opti.data);plt.xlabel('# of parameters');plt.ylabel('MEA');plt.title('MEA vs # of parameters')
from datetime import date

# creating date time parameters as a serial value

all_datetime_serial= pd.Series({idx:date.toordinal(pd.to_datetime(str(home_data.YrSold[idx])+'/'+str(home_data.MoSold[idx]))) for idx in home_data.index})



print("Correlation of combine Year and Month: %.3f" %(all_datetime_serial.corr(home_data.SalePrice)))

print("Correlation of Year %.3f" %(home_data.YrSold.corr(home_data.SalePrice)))

print("Correlation of Month %.3f" %(home_data.MoSold.corr(home_data.SalePrice)))
all_datetime = pd.Series({idx:pd.to_datetime(str(home_data.YrSold[idx])+'/'+str(home_data.MoSold[idx])) for idx in home_data.index})

plt.rcParams["figure.figsize"]=16,5; plt.scatter(all_datetime,home_data.SalePrice); plt.xlabel('Date') ; plt.ylabel('Sale Price USD$');plt.title('House sale price over time');
#calculating average house price of each month

avg_month = pd.Series({mo:home_data.SalePrice[home_data.MoSold == mo].mean() for mo in pd.Series(home_data.MoSold.unique()).sort_values()}) 

plt.rcParams["figure.figsize"]=7,5;plt.scatter(avg_month.index,avg_month.data);plt.xlabel('Month');plt.ylabel('Average Sale Price');plt.title('Avg. sale price over 12 months');
category =  np.setdiff1d(home_data.columns,corr_vals.index)

category
numerical_home_data = home_data[corr_vals.index].copy() # creating a new dataframe but with only numerical data, to which we will append our dummy variables
numerical_home_data.shape 
num_home_data_dummy = numerical_home_data.add(pd.get_dummies(home_data[category]),fill_value =0) # creating dummy variables
num_home_data_dummy.shape
#all the top 20 correlations

dummy_corr = num_home_data_dummy.corrwith(num_home_data_dummy.SalePrice)

plt.rcParams["figure.figsize"]=28,4

fig, ax = plt.subplots()

ax = plt.bar(dummy_corr.sort_values(ascending = False).iloc[1:20].index,dummy_corr.sort_values(ascending = False).iloc[1:20]) ; plt.xlabel('Variable') ; plt.ylabel('Correlation');

for bar in ax.patches:

    bar.set_facecolor('#888888')

ax.patches[7].set_facecolor('#aa3333');ax.patches[11].set_facecolor('#aa3333');ax.patches[12].set_facecolor('#aa3333');ax.patches[16].set_facecolor('#aa3333');ax.patches[17].set_facecolor('#aa3333');ax.patches[18].set_facecolor('#aa3333')

def smart_fill(to_fill,frame):

    '''

    Takes dataframe and specific variable in it. Fills the missing variables in to_fill through intepolation with correlated variables.

    '''

    top_corr = frame.corrwith(frame[to_fill]).sort_values(ascending = False).index[1:4] #top 3 correlating values to variable being

    frame.sort_values(by=list(top_corr),inplace= True) #sorting frame with highest correlating variables with missing series

    frame[to_fill].interpolate(inplace = True) #filling nans with interpolation, inferring position by the sorting

    frame[to_fill].fillna(method = 'bfill',inplace= True)

    frame[to_fill].fillna(method = 'ffill',inplace= True)

    #bfill & ffill needed as you can't interpolate for edge values, and missing edge values are left NaN by interpolate method

    return frame  
filled_cols = []

for col in num_home_data_dummy.columns:

    if num_home_data_dummy[col].isnull().any()==1: #only run for columns with missing data

        filled_cols.append(col)

        smart_fill(col,num_home_data_dummy)
filled_cols # variables which have had their missing data filled!
print("Missing data in test data: {0}".format(test_data.isnull().any().count()))


test_cat = test_data[corr_vals.index.drop(corr_vals.index[0])].add(pd.get_dummies(test_data[category]),fill_value = 0).copy()

#train_data = home_data[corr_vals.index].add(pd.get_dummies(home_data[category]),fill_value = 0).copy()

filled_cols2 = []

for col in test_cat.columns:

    if test_cat[col].isnull().any()==1: #only run for columns with missing data

        filled_cols2.append(col)

        test_cat = smart_fill(col,test_cat)
print("The number of missing data filled is {0}".format(len(filled_cols2))) # a lot of missing data was filled in the categorical data!
#creating training data

train_data = home_data[corr_vals.index].add(pd.get_dummies(home_data[category]),fill_value = 0).copy()



filled_cols2 = []

for col in train_data.columns:

    if train_data[col].isnull().any()==1: #only run for columns with missing data

        filled_cols2.append(col)

        train_data = smart_fill(col,train_data)
def train_by_feat(y,features,home_data):

    '''

    Trains a model based on a specified list of features and returns the MAE value for that model.

    '''

    X = home_data[features]

    train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

    

    model = RandomForestRegressor(random_state = 1)

    model.fit(train_X,train_y)

    feat_val_predictions = model.predict(val_X)

    feat_val_mae = mean_absolute_error(feat_val_predictions, val_y)

    

    return feat_val_mae
features = train_data.corrwith(train_data.SalePrice).sort_values(ascending = False).keys()
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
opti = pd.Series({num:train_by_feat(train_data.SalePrice,features[1:num+2],train_data) for num in np.arange(100)}) #need +2 to offset the fact we are starting at index 1
plt.rcParams["figure.figsize"]=7,5;plt.plot((1+np.arange(100)),opti);plt.xlabel('Number of Features');plt.ylabel('MAE');plt.title('Optimization of Random Forrest Regressor');
print("Mininmum MAE of {0} with {1} features".format(opti[opti.idxmin()],opti.idxmin()))
#Creating model of the 85 features

y = train_data.SalePrice

X = train_data[features[1:87]]

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

    

model = RandomForestRegressor(random_state = 1)

model.fit(train_X,train_y)

feat_val_predictions = model.predict(val_X)

mean_absolute_error(feat_val_predictions, val_y) # shows we are assuming a low error in our predictions
X_test = test_cat[features[1:87]]

y_pred = model.predict(X_test)
output = pd.DataFrame({'Id': test_data.Id,'SalePrice': y_pred})



good_result = pd.read_csv("../input/erincb-house-submission/submission_good.csv")

good_result.rename(columns ={'SalePrice':'SalePriceCorrect'},inplace = True)

good_result['SalePriceWrong'] = y_pred

good_result.head()
#create categorical dummy sets

train_cat = pd.get_dummies(home_data)

test_cat = pd.get_dummies(test_data)



#creating list of indexes in correlating order

cat_corr = train_cat.corrwith(train_cat.SalePrice).sort_values(ascending = False)



#filling missing values

train_cat.fillna(method ='pad',inplace= True)

test_cat.fillna(method = 'pad',inplace = True)



#determining optimum number of features ()

opti = pd.Series({num:train_by_feat(train_cat.SalePrice,cat_corr.index[1:num+2],train_cat) for num in np.arange(30)}) #30 chosen arbitrarily from trial and error on kaggle score



print("Number of optimum features {0}".format(opti.idxmin()))

#creatingthe model

X = train_cat[cat_corr.index[1:28]]

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

    



model = RandomForestRegressor(random_state = 1)

model.fit(train_X,train_y)

feat_val_predictions = model.predict(val_X)

feat_val_mae = mean_absolute_error(feat_val_predictions, val_y)
#making predictions

predictions = model.predict(test_cat[cat_corr.index[1:28]])

outy = pd.DataFrame({'Id': test_data.Id,'SalePrice': predictions})

outy.head()
outy.to_csv('submission.csv',index = False)