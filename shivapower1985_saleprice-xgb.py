import pandas as pd

from numpy import nan



train_data=pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

test_data=pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')



print("Training shape ", train_data.shape)

print("Test shape ", test_data.shape)









#print(train_data["Id"].dtype)

#print(train_data["MSZoning"].dtype)



train_data=train_data.drop(['Id'], axis=1)

test_data=test_data.drop(['Id'], axis=1)



print(train_data.shape)

print(test_data.shape)



train_data.describe()

train_data.info()
total = train_data.isnull().sum().sort_values(ascending=False)

percent = ((train_data.isnull().sum()/train_data.isnull().count())*100).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

print(missing_data.head(50))







for x in missing_data.index:

    a=missing_data.loc[x]

    if a[1]>=16:

        print(x)

        train_data=train_data.drop(columns=[x])

        test_data=test_data.drop(columns=[x])

        





print(train_data.shape)

print(test_data.shape)

total = train_data.isnull().sum().sort_values(ascending=False)

print(total)
train_data=train_data.fillna(method='ffill')

test_data=test_data.fillna(method='ffill')

total_train = train_data.isnull().sum().sort_values(ascending=False)

print(total_train)

total_test = test_data.isnull().sum().sort_values(ascending=False)

print(total_test)



#Check How many are catagory and numerical variables

cat_cols = [x for x in train_data.columns if train_data[x].dtype == 'object']

num_cols = [x for x in train_data.columns if train_data[x].dtype != 'object']



print("Number of column features in Training data set: ", train_data.shape[1])

print("Categorical Variables in 'train.csv' file: ", len(cat_cols))

print("Numerical Variables in 'train.csv' file: ", len(num_cols))



#Copying categorical variables 

cat_df = train_data.select_dtypes(include=['object']).copy()





#Copying numerical variables 

num_df = train_data.select_dtypes(include=['int64','float64']).copy()



#num_df.head()

cat_df.head()

from sklearn.preprocessing import LabelEncoder

le=LabelEncoder()



#for i in cat_df.columns:

#    cat_df[i]=le.fit_transform(cat_df[i])



#cat_df.head(50)



for i in train_data.columns:

    if i!="SalePrice":

        if train_data[i].dtype=='object':

            train_data[i]=le.fit_transform(train_data[i])

        elif test_data[i].dtype=='object':

            test_data[i]=le.fit_transform(test_data[i])







print(train_data.shape)

print(test_data.shape)



train_data.head()

#test_data.head()

#Feature Selection Technique 3: Correlation analysis

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt



X = train_data.iloc[:,0:-1]  #independent columns

y = train_data.iloc[:,-1]    #target column i.e price range

#get correlations of each features in dataset

corrmat = train_data.corr()

top_corr_features = corrmat.index

plt.figure(figsize=(81,81))

#plot heat map

g=sns.heatmap(train_data[top_corr_features].corr(),annot=True,cmap="RdYlGn")
k = 15 

  

cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index 

  

cm = np.corrcoef(train_data[cols].values.T) 

f, ax = plt.subplots(figsize =(12, 10)) 

  

sns.heatmap(cm, ax = ax, cmap ="YlGnBu", 

            linewidths = 0.1, yticklabels = cols.values,  

                              xticklabels = cols.values) 
#Feature Selection Technique 3: Correlation analysis

corrmat = train_data.corr()

corrmat.sort_values(["SalePrice"], ascending = False, inplace = True)

cor_SalePrice=corrmat.SalePrice





for x in cor_SalePrice.index:

    if cor_SalePrice[x] < 0.4:

        cor_SalePrice=cor_SalePrice.drop([x])



print(cor_SalePrice)
import seaborn as sns

#for a in cor_SalePrice.index:

#train_set=train_data

#test_set=test_data



df_train=pd.DataFrame()

df_test=pd.DataFrame()



for n in cor_SalePrice.index:

    #rint(train_data[n])

    df_train=df_train.append(train_data[n])

    if n != "SalePrice":

        df_test=df_test.append(test_data[n])

#   df = pd.DataFrame(data, index =['rank1', 'rank2', 'rank3', 'rank4']) 



df_train=df_train.T

df_test=df_test.T



df_train.head()

#df_test.head()








YearBuilt=df_train["YearBuilt"]

GarageYrBlt=df_train["GarageYrBlt"]

YearRemodAdd=df_train["YearRemodAdd"]



for b in range(0,len(YearBuilt)):

    if YearBuilt[b]>=2000:

        YearBuilt[b]=1

    elif YearBuilt[b]>=1950 and YearBuilt[b]<2000:

        YearBuilt[b]=2

    else:

        YearBuilt[b]=3

        

    if GarageYrBlt[b]>=2000:

        GarageYrBlt[b]=1

    elif GarageYrBlt[b]>=1950 and GarageYrBlt[b]<2000:

        GarageYrBlt[b]=2

    else:

        GarageYrBlt[b]=3

    

    if YearRemodAdd[b]>=2000:

        YearRemodAdd[b]=1

    elif YearRemodAdd[b]>=1950 and YearRemodAdd[b]<2000:

        YearRemodAdd[b]=2

    else:

        YearRemodAdd[b]=3



        

        

YearBuilt_test=df_test["YearBuilt"]

GarageYrBlt_test=df_test["GarageYrBlt"]

YearRemodAdd_test=df_test["YearRemodAdd"]



for b in range(0,len(YearBuilt_test)):

    if YearBuilt_test[b]>=2000:

        YearBuilt_test[b]=1

    elif YearBuilt_test[b]>=1950 and YearBuilt_test[b]<2000:

        YearBuilt_test[b]=2

    else:

        YearBuilt_test[b]=3



    if GarageYrBlt_test[b]>=2000:

        GarageYrBlt_test[b]=1

    elif GarageYrBlt_test[b]>=1950 and GarageYrBlt_test[b]<2000:

        GarageYrBlt_test[b]=2

    else:

        GarageYrBlt_test[b]=3

    

    if YearRemodAdd_test[b]>=2000:

        YearRemodAdd_test[b]=1

    elif YearRemodAdd_test[b]>=1950 and YearRemodAdd_test[b]<2000:

        YearRemodAdd_test[b]=2

    else:

        YearRemodAdd_test[b]=3



        

        

df_test.head()

df_train.head()

        



#####Basic Scatter Plot#########

fig, ax = plt.subplots()

plt.scatter(df_train["SalePrice"],df_train["OverallQual"])

ax.set_xlabel('Sale Price')

ax.set_ylabel('Overall Quality Rating')

ax.set_title('Sale Price vs Overall Qual')

plt.show()





############ Bubble Chart using Scatter Plot ##################

fig1, ax1 = plt.subplots()



scatter = ax1.scatter(df_train["SalePrice"],df_train["OverallQual"], c=df_train["YearBuilt"], s=df_train["YearBuilt"]*100)



# produce a legend with the unique colors from the scatter

legend1 = ax1.legend(*scatter.legend_elements(),

                    loc="lower right", title="Built Year Classes",frameon=True)

ax1.add_artist(legend1)

ax1.set_xlabel('Sale Price')

ax1.set_ylabel('Overall Quality Rating')

ax1.set_title('Sale Price vs Overall Qual vs YearBuilt')

# produce a legend with a cross section of sizes from the scatter

#handles, labels = scatter.legend_elements(prop="sizes", alpha=0.6)

#legend2 = ax.legend(handles, labels, loc="upper right", title="Sizes")

plt.show()



# Basics in feature scaling

sns.distplot(df_train['SalePrice'], kde = False, color ='red', bins = 30)



from scipy.stats import kurtosis, skew

print(kurtosis(df_train['SalePrice']))

print(skew(df_train['SalePrice']))



print(df_train['SalePrice'].describe())
#Inverse transform

#inversed = scaler.inverse_transform(data_scaled)

#print(inversed)
#MIN-MAX scaler

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler() 

data_min_max_scaled = scaler.fit_transform(df_train)



sns.distplot(data_min_max_scaled[:,2], kde = False, color ='red', bins = 30)

sample=data_min_max_scaled[:,2]

sample=pd.DataFrame(sample)

sample.describe()
# Log+1 function standardization



norm_log=np.log(df_train+1)

sns.distplot(norm_log.iloc[:,0], kde = False, color ='red', bins = 30)

#for n in norm_log.columns:

#    sns.distplot(norm_log[n], kde = False, color ='red', bins = 30)

print(norm_log.iloc[:,0].describe())
# standardScaler()

#NOTE:  standardization (or Z-score normalization) means centering the variable at zero and standardizing the variance at 1.

#print(df_train.describe())

from sklearn.preprocessing import StandardScaler

scaler_train = StandardScaler() 

scaler_test = StandardScaler() 



train_data_scaled = scaler_train.fit_transform(df_train) #Standardscaler on training_data

test_data_scaled = scaler_test.fit_transform(df_test)#Standardscaler on testing_data



saleprice_scaled=pd.DataFrame(train_data_scaled[:,0])

print(saleprice_scaled.describe())



sns.distplot(train_data_scaled[:,0], kde = False, color ='red', bins = 30)





#Inverse transform

#inversed = scaler.inverse_transform(data_scaled)

#print(inversed)

print(train_data_scaled.shape)

print(train_data_scaled[:,1::].shape)
from sklearn.model_selection import train_test_split

import statistics as st

#X=df_train.iloc[:,1::]

#Y=df_train.iloc[:,0]



#print(X.shape)

#print(Y.shape)



train, test= train_test_split(df_train, test_size=0.3, shuffle=False,random_state=42)



print(train.shape)

print(test.shape)



train_mean=train.mean()

train_std=np.std(train)



scaled_train =  (train - train.mean()) / np.std(train)



scaled_test = (test - train.mean()) / np.std(train)



x_train=scaled_train.iloc[:,1::]

y_train=scaled_train.iloc[:,0]



x_test=scaled_test.iloc[:,1::]

y_test=scaled_test.iloc[:,0]



train.head()

test.head()



#print(scaled_train.min(), scaled_train.max())

#df_test

#train, test = split(df_train)

# Gradient boost regressor



#train_y=train_data_scaled[:,0]

#train_x=train_data_scaled[:,1::]

#print(train_x.shape)

#train_y=train_y.reshape((len(train_y),1))

#print(train_y.shape)



#SVR on Scaled Data

from sklearn import svm



from sklearn import metrics



from sklearn.metrics import mean_squared_error

from sklearn.metrics import r2_score





# Fit regression model

regr_rbf = svm.SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)

regr_rbf.fit(x_train, y_train)

predict_scaled_output_svr_rbf=regr_rbf.predict(x_test)

# Fit regression model

#svr_rbf = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)

#svr_lin = SVR(kernel='linear', C=100, gamma='auto')

#svr_poly = SVR(kernel='poly', C=100, gamma='auto', degree=3, epsilon=.1,coef0=1)

#Inverse transform

test_predict_svr_rbf=(predict_scaled_output_svr_rbf*train_std.iloc[0])+ train_mean.iloc[0] 

actual=(y_test*train_std.iloc[0])+ train_mean.iloc[0]



print("R2_Score", r2_score(actual, test_predict_svr_rbf))



#print("MAE", metrics.mean_absolute_error(actual, test_predict_svr_rbf))

#print("MSE", metrics.mean_squared_error(actual, test_predict_svr_rbf))

print("RMSE",np.sqrt(metrics.mean_squared_error(actual, test_predict_svr_rbf)))

#plt.plot(actual, test_predict_svr_rbf)

#plt.show()



#print(actual.iloc[0], test_predict_svr_rbf[0])

from yellowbrick.regressor import PredictionError

visualizer_regr_rbf = PredictionError(regr_rbf)



visualizer_regr_rbf.fit(x_train, y_train)  # Fit the training data to the visualizer

visualizer_regr_rbf.score(x_test, y_test)  # Evaluate the model on the test data

visualizer_regr_rbf.show()                 # Finalize and render the figure

#SVR on Not scaled data

from sklearn import svm

x_train_raw=train.iloc[:,1::]

y_train_raw=train.iloc[:,0]



x_test_raw=test.iloc[:,1::]

y_test_raw=test.iloc[:,0]



# Fit regression model

regr_rbf_raw = svm.SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)

regr_rbf_raw.fit(x_train_raw, y_train_raw)

predict_raw_output_svr_rbf=regr_rbf_raw.predict(x_test_raw)

# Fit regression model

#svr_rbf = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)

#svr_lin = SVR(kernel='linear', C=100, gamma='auto')

#svr_poly = SVR(kernel='poly', C=100, gamma='auto', degree=3, epsilon=.1,coef0=1)



#print(mean_squared_error(y_test_raw, predict_raw_output_svr_rbf))

print("R2_Score", r2_score(y_test_raw, predict_raw_output_svr_rbf))



#print("MAE", metrics.mean_absolute_error(y_test_raw, predict_raw_output_svr_rbf))

#print("MSE", metrics.mean_squared_error(y_test_raw, predict_raw_output_svr_rbf))

print("RMSE",np.sqrt(metrics.mean_squared_error(y_test_raw, predict_raw_output_svr_rbf)))



#plt.plot(actual, test_predict_svr_rbf)

#plt.show()



#print(y_test_raw.iloc[0], predict_raw_output_svr_rbf[0])



from yellowbrick.regressor import PredictionError

visualizer_rbf_raw = PredictionError(regr_rbf_raw)



visualizer_rbf_raw.fit(x_train_raw, y_train_raw)  # Fit the training data to the visualizer

visualizer_rbf_raw.score(x_test_raw, y_test_raw)  # Evaluate the model on the test data

visualizer_rbf_raw.show()                 # Finalize and render the figure





# XGB on scaled data

import xgboost as xgb



from sklearn import metrics





#Fitting XGB regressor 

model = xgb.XGBRegressor(base_score=0.5, booster=None, colsample_bylevel=1,

             colsample_bynode=1, colsample_bytree=1, gamma=0, gpu_id=-1,

             importance_type='gain', interaction_constraints=None,

             learning_rate=0.100000012, max_delta_step=0, max_depth=6,

             min_child_weight=1, missing=nan, monotone_constraints=None,

             n_estimators=100, n_jobs=0, num_parallel_tree=1,

             objective='reg:squarederror', random_state=0, reg_alpha=0,

             reg_lambda=1, scale_pos_weight=1, subsample=1, tree_method=None,

             validate_parameters=False, verbosity=None)



#model = xgb.XGBRegressor()#default - learning rate=0.3, base_score=0.5, booster=None

model.fit(x_train,y_train)

print (model)









#print(test_x.shape)

#print(train_x.shape)

predict_scaled_output = model.predict(data=x_test)

#print(predict_scaled_output.shape)



#print(train_mean.iloc[0])

#print(train_std.iloc[0])

#print(np.std(train).shape)

#print(train.mean().shape)



#Inverse transform

test_predict=(predict_scaled_output*train_std.iloc[0])+ train_mean.iloc[0] 

actual=(y_test*train_std.iloc[0])+ train_mean.iloc[0]



print(test_predict.shape)

print(actual.shape)



from sklearn.metrics import mean_squared_error

from sklearn.metrics import r2_score



print("R2_Score", r2_score(actual, test_predict))



#print("MAE", metrics.mean_absolute_error(actual, test_predict))

#print("MSE", metrics.mean_squared_error(actual, test_predict))

print("RMSE",np.sqrt(metrics.mean_squared_error(actual, test_predict)))



print("Mean squared logarithmic error regression loss: ", metrics.mean_squared_log_error(actual, test_predict))

#print('RMSE in % is', (np.sqrt(metrics.mean_squared_error(actual, test_predict))/max(15)*100)

#actual=actual.reset_index()

#plt.plot(test_predict)

#plt.plot(actual)

#plt.ylabel('Sale Price in $')

#plt.show()

from yellowbrick.regressor import PredictionError

visualizer = PredictionError(model)



visualizer.fit(x_train, y_train)  # Fit the training data to the visualizer

visualizer.score(x_test, y_test)  # Evaluate the model on the test data

visualizer.show()                 # Finalize and render the figure



#print(x_train.shape)

#print(y_train.shape)

#print(x_test.shape)

#print(y_test.shape)

#variance_xgb= np.var(test_predict)

#print("The Variance of the XGBoost model is: ", variance_xgb)
import xgboost as xgb

from sklearn.model_selection import GridSearchCV

from sklearn import metrics



########Parameter tuning starts here

# ROUND 1: Tuning max_depth and min_child_weight

param_grid1 = {

    'max_depth':np.arange(1,11,2),

    'min_child_weight':np.arange(1,11,2)

}



# First Search

gsearch1 = GridSearchCV(xgb.XGBRegressor(),

    param_grid1, 

    scoring='neg_mean_squared_error',

    cv=4)



gsearch1 = gsearch1.fit(x_train, y_train)



best_params = gsearch1.best_params_

#best_params, -gsearch1.best_score_





# ROUND 2: Tuning Gamma



param_grid2 = {'gamma':[i/10.0 for i in range(0,6)]}



gsearch2 = GridSearchCV(

    xgb.XGBRegressor(**best_params),

    param_grid2,

    scoring='neg_mean_squared_error', 

    cv=5

)



gsearch2 = gsearch2.fit(x_train, y_train)



best_params.update(gsearch2.best_params_)

#best_params, -gsearch2.best_score_



# ROUND 3: Tuning subsamle and colsample_bytree

param_grid3 = {

    'colsample_bytree':[i/10.0 for i in range(0,11)],

    'subsample':[i/10.0 for i in range(0,11)]

}



gsearch3 = GridSearchCV(

    xgb.XGBRegressor(**best_params),

    param_grid3,

    scoring='neg_mean_squared_error',

    cv=3   

)



gsearch3 = gsearch3.fit(x_train, y_train)

best_params.update(gsearch3.best_params_)

#best_params, -gsearch3.best_score_



# TUNING Alpha & Lambda

param_grid4 = {

    'alpha':[i/10 for i in range(0,11)],

    'lambda':[i/10 for i in range(0,11)]

}



gsearch4 = GridSearchCV(

    xgb.XGBRegressor(**best_params),

    param_grid4, 

    scoring='neg_mean_squared_error',

    cv=5

)



gsearch4 = gsearch4.fit(x_train, y_train)



best_params.update(gsearch4.best_params_)

#best_params, -gsearch4.best_score_



# Tuning: the number of trees and learning rate

param_grid5 = {

    'n_estimators':np.arange(50, 450, 50),

    'learning_rate':[0.01, 0.05, 0.1, .5, 1]

}



gsearch5 = GridSearchCV(

    xgb.XGBRegressor(**best_params),

    param_grid5, 

    scoring='neg_mean_squared_error',

    cv=5

)



gsearch5 = gsearch5.fit(x_train, y_train)



best_params.update(gsearch5.best_params_)

#best_params, -gsearch5.best_score_
print(best_params)
# XGB Tuned Model

modeltuned = xgb.XGBRegressor(**best_params)







modeltuned.fit(x_train,y_train)

print (modeltuned)









#print(test_x.shape)

#print(train_x.shape)

predict_scaled_output = modeltuned.predict(data=x_test)

#print(predict_scaled_output.shape)



#print(train_mean.iloc[0])

#print(train_std.iloc[0])

#print(np.std(train).shape)

#print(train.mean().shape)



#Inverse transform

test_predict=(predict_scaled_output*train_std.iloc[0])+ train_mean.iloc[0] 

actual=(y_test*train_std.iloc[0])+ train_mean.iloc[0]



print(test_predict.shape)

print(actual.shape)



from sklearn.metrics import mean_squared_error

from sklearn.metrics import r2_score



print("R2_Score", r2_score(actual, test_predict))



#print("MAE", metrics.mean_absolute_error(actual, test_predict))

#print("MSE", metrics.mean_squared_error(actual, test_predict))

print("RMSE",np.sqrt(metrics.mean_squared_error(actual, test_predict)))



print("Mean squared logarithmic error regression loss: ", metrics.mean_squared_log_error(actual, test_predict))

#print('RMSE in % is', (np.sqrt(metrics.mean_squared_error(actual, test_predict))/max(15)*100)

#actual=actual.reset_index()

#plt.plot(test_predict)

#plt.plot(actual)

#plt.ylabel('Sale Price in $')

#plt.show()

from yellowbrick.regressor import PredictionError

visualizer = PredictionError(modeltuned)



visualizer.fit(x_train, y_train)  # Fit the training data to the visualizer

visualizer.score(x_test, y_test)  # Evaluate the model on the test data

visualizer.show()                 # Finalize and render the figure



#print(x_train.shape)

#print(y_train.shape)

#print(x_test.shape)

#print(y_test.shape)

#variance_xgb= np.var(test_predict)

#print("The Variance of the XGBoost model is: ", variance_xgb)
#Scale [df_test] for scaling



train_mean=train.mean()

SalePrice_train_mean=train_mean[0]

train_mean=train_mean.drop(['SalePrice'])



train_std=np.std(train)

SalePrice_train_std=train_std[0]

train_std=train_std.drop(['SalePrice'])



df_test_scaled=(df_test - train_mean)/train_std







#Prediction the scales 'SalePrice' 

# Use the loaded base XGBRegressor model to make predictions 

predict_output_scaled=model.predict(df_test_scaled) 



# Scaling back the scaled prediction value to normal value

predicted_SalePrice=(predict_output_scaled*SalePrice_train_std)+ SalePrice_train_mean 



print(predicted_SalePrice)

#print(predicted_SalePrice.shape)
