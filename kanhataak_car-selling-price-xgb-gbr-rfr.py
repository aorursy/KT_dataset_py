import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import numpy as np

import seaborn as sn
df=pd.read_csv('../input/car-data/car data.csv')

df.head()
df.shape
for i in df.columns:

    print(i)
print(df['Seller_Type'].unique())

print(df['Fuel_Type'].unique())

print(df['Transmission'].unique())

print(df['Owner'].unique())
##check missing values

df.isnull().sum()
df.describe()
final_dataset=df[['Year','Selling_Price','Present_Price','Kms_Driven','Fuel_Type','Seller_Type','Transmission','Owner']]
final_dataset.head()
# showing the selling price based on Year

plt.figure(1, figsize=(8, 6))

plt.bar(final_dataset.Year,final_dataset.Selling_Price, color='blue',alpha=0.4)

plt.xlabel("Year")

plt.ylabel("Selling price")

plt.show()
plt.figure(figsize = (18,5))

sn.boxplot(data=final_dataset)

plt.show()
q1 = final_dataset['Kms_Driven'].quantile(0.25)

q3 = final_dataset['Kms_Driven'].quantile(0.75)

iqr = q3-q1



UL = q3 + (1.5 * iqr)

LL = q1 - (1.5 * iqr)

print(f"IQR: {iqr}, UL: {UL}, LL: {LL}")
final_dataset[final_dataset['Kms_Driven']>UL]
plt.figure(figsize = (18,5))

sn.boxplot(data=final_dataset[final_dataset['Kms_Driven']>UL])

plt.show()
#outlier removal from Kms_Driven



final_dataset = final_dataset[final_dataset['Kms_Driven']<UL]

final_dataset.head()

plt.figure(figsize = (18,5))

sn.boxplot(data=final_dataset[final_dataset['Kms_Driven']<UL])

plt.show()
sn.distplot(final_dataset['Year'])
final_dataset['Current Year']=2020
final_dataset.head()
# How many years car is old so that how you can do currentYear subtract Year of buying[Year]

final_dataset['no_year']=final_dataset['Current Year']- final_dataset['Year']
final_dataset.head()
count_fuelTye = pd.value_counts(final_dataset['Fuel_Type'], sort = True)



count_fuelTye.plot(kind = 'bar', rot=0,)



plt.title("Distribution of Based on Fuel Type")



plt.xticks(range(3))



plt.xlabel("Fuel Type")



plt.ylabel("Frequency Count")
print(final_dataset.Fuel_Type.value_counts())

ax = sn.barplot(x="Fuel_Type", y="Selling_Price", data=final_dataset)
count_Transmission = pd.value_counts(final_dataset['Transmission'], sort = True)



count_Transmission.plot(kind = 'bar', rot=0,)



plt.title("Distribution of Transmission")



plt.xticks(range(3))



plt.xlabel("Transmission Type")



plt.ylabel("Frequency Count")
print("How many cars are avaliable Dealer and Individual: \n",final_dataset.Seller_Type.value_counts())

ax = sn.barplot(x="Seller_Type", y="Selling_Price", data=final_dataset)
# now we dont need to Year and also Current Year feature so drop it

final_dataset.drop(['Year','Current Year'],axis=1,inplace=True)
final_dataset.head()
final_dataset=pd.get_dummies(final_dataset,drop_first=True)
final_dataset.head()
final_dataset.corr()
import seaborn as sn
sn.pairplot(final_dataset)
#get correlations of each features in dataset

corrmat = final_dataset.corr()

top_corr_features = corrmat.index

plt.figure(figsize=(10,10))

#plot heat map

g=sn.heatmap(final_dataset[top_corr_features].corr(),annot=True,cmap="RdYlGn")
X=final_dataset.iloc[:,1:]

y=final_dataset.iloc[:,0]
X['Owner'].unique()
X.head()
y.head()
### Feature Importance



from sklearn.ensemble import ExtraTreesRegressor

import matplotlib.pyplot as plt

model = ExtraTreesRegressor()

model.fit(X,y)
print(model.feature_importances_)
#plot graph of feature importances for better visualization

feat_importances = pd.Series(model.feature_importances_, index=X.columns)

feat_importances.nlargest(6).plot(kind='barh')

plt.show()
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
from sklearn.ensemble import RandomForestRegressor
regressor=RandomForestRegressor()
from sklearn.model_selection import RandomizedSearchCV
 #Randomized Search CV



# Number of trees in random forest

n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)]

# Number of features to consider at every split

max_features = ['auto', 'sqrt']

# Maximum number of levels in tree

max_depth = [int(x) for x in np.linspace(5, 30, num = 6)]

# max_depth.append(None)

# Minimum number of samples required to split a node

min_samples_split = [2, 5, 10, 15, 100]

# Minimum number of samples required at each leaf node

min_samples_leaf = [1, 2, 5, 10]

# Create the random grid

random_grid = {'n_estimators': n_estimators,

               'max_features': max_features,

               'max_depth': max_depth,

               'min_samples_split': min_samples_split,

               'min_samples_leaf': min_samples_leaf}



print(random_grid)
# Use the random grid to search for best hyperparameters

# First create the base model to tune

rf = RandomForestRegressor()
# Random search of parameters, using 3 fold cross validation, 

# search across 100 different combinations

rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid,scoring='neg_mean_squared_error', n_iter = 10, cv = 5, verbose=2, random_state=42, n_jobs = 1)
# It will take some time for selecting best param

rf_random.fit(X_train,y_train)
rf_random.best_estimator_
rf_random.best_params_
rfr = RandomForestRegressor(max_depth=20, min_samples_leaf=2,

                      min_samples_split=15, n_estimators=1100)
model_rf=rfr.fit(X_train,y_train)
predict_=model_rf.predict(X_test)
model_rf.score(X_test,y_test)
from sklearn.metrics import r2_score

R2 = r2_score(y_test,predict_)

R2
# Plotting y_test and predictions to understand the spread



fig = plt.figure()

plt.scatter(y_test,predict_, alpha=.5)

fig.suptitle('y_test vs predict_', fontsize = 20) 

plt.xlabel('y_test', fontsize = 18)                          

plt.ylabel('predict_', fontsize = 16) 

plt.show()
df = pd.DataFrame({'Actual':y_test,"Predicted":predict_})

df.head()
# difference between the Actual and predicted value



df1 = df.head(25)

df1.plot(kind='bar',figsize=(10,5))

plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')

plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')

plt.show()
rf_random.best_score_

# -4.430592047602699
predictions=rf_random.predict(X_test)
from sklearn.metrics import r2_score

R2 = r2_score(y_test,predictions)

R2

# 0.8678892897805003
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score

R2 = r2_score(y_test,predictions)

R2

# 0.8678892897805003
df = pd.DataFrame({'Actual':y_test,"Predicted":predictions})

df.head()
# difference between the Actual and predicted value



df1 = df.head(25)

df1.plot(kind='bar',figsize=(10,5))

plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')

plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')

plt.show()
# Plotting y_test and predictions to understand the spread



fig = plt.figure()

plt.scatter(y_test, predictions, alpha=.5)

fig.suptitle('y_test vs y_pred', fontsize = 20) 

plt.xlabel('y_test', fontsize = 18)                          

plt.ylabel('y_pred', fontsize = 16) 

plt.show()
rf = y_test-predictions

# Plot the histogram of the error terms

fig = plt.figure()

sn.distplot((rf), bins = 20)

fig.suptitle('Error Terms', fontsize = 20)                  # Plot heading 

plt.xlabel('Errors', fontsize = 18) 
sn.distplot(y_test-predictions)
from sklearn import metrics
print('MAE:', metrics.mean_absolute_error(y_test, predictions))

print('MSE:', metrics.mean_squared_error(y_test, predictions))

print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))
# with outliers show result and also no tuning param

# MAE: 0.8830906595298466

# MSE: 3.9509103915752966

# RMSE: 1.987689712096759
from sklearn.ensemble import GradientBoostingRegressor

gboost_model = GradientBoostingRegressor()
gboost_model.fit(X_train, y_train)
y_pred = gboost_model.predict(X_test)
print(r2_score(y_test, y_pred))
sn.distplot(y_test-y_pred)
fig = plt.figure()

plt.scatter(y_test, y_pred, alpha=.5)

fig.suptitle('y_test vs y_pred', fontsize = 20) 

plt.xlabel('y_test', fontsize = 18)                          

plt.ylabel('y_pred', fontsize = 16) 

plt.show()
df_gb = pd.DataFrame({'Actual':y_test,"Predicted":y_pred})

df_gb.head()
# difference between the Actual and predicted value



df_gb = df_gb.head(25)

df_gb.plot(kind='bar',figsize=(10,5))

plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')

plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')

plt.show()
#Randomized search CV for gradient boosting

#Number of treers in random forest

n_estimators =[int(x) for x in np.linspace(100,1200,num = 12)]

#Learning rate

learning_rate = [0.01, 0.02, 0.05, 0.1, 0.2]

subsample = [0.05, 0.06, 0.08, 0.09, 0.1]

criterion = ['mse', 'rmse', 'friedman_mse']

#Number of features to consider at every split

max_features =["auto", "sqrt"]
#creating gradient boosting grid

gb_grid = {'n_estimators' : n_estimators,

           'learning_rate' : learning_rate,

           'subsample' : subsample,

           'max_depth' : max_depth,

           'max_features' : max_features}

print(gb_grid)
final_gb_model = RandomizedSearchCV(estimator = gboost_model, param_distributions=gb_grid,

                                 scoring='neg_mean_squared_error', n_iter = 20,

                                 cv = 5, verbose = 2, random_state = 42, n_jobs =1)
final_gb_model.fit(X_train,y_train)
final_gb_model.best_estimator_
final_gb_model.best_params_
gbr_model = GradientBoostingRegressor(learning_rate=0.05, max_depth=5, max_features='auto',

                          n_estimators=900, subsample=0.08)
gbr_model.fit(X_train,y_train)
pred= gbr_model.predict(X_test)


print(r2_score(y_test, pred))
gbr_model.score(X_test,y_test)
df_1 = pd.DataFrame({'Actual':y_test,"Predicted":pred})

df_1.head()
# difference between the Actual and predicted value



df_1 = df_1.head(25)

df_1.plot(kind='bar',figsize=(10,5))

plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')

plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')

plt.show()
print('MAE:', metrics.mean_absolute_error(y_test, pred))

print('MSE:', metrics.mean_squared_error(y_test, pred))

print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, pred)))
# Plotting y_test and pred to understand the spread



fig = plt.figure()

plt.scatter(y_test, pred, alpha=.5)

fig.suptitle('y_test vs pred', fontsize = 20) 

plt.xlabel('y_test', fontsize = 18)                          

plt.ylabel('pred', fontsize = 16) 

plt.show()
# Plot the histogram of the error terms

fig = plt.figure()

sn.distplot((y_test-pred), bins = 20)

fig.suptitle('Error Terms', fontsize = 20)                  # Plot heading 

plt.xlabel('Errors', fontsize = 18)
# import pickle

# # open a file, where you ant to store the data

# file = open('Gradient_Boosting_Regressor_model.pkl', 'wb')



# # dump information to that file

# pickle.dump(gboost_model, file)
import xgboost as xgb
#Fitting XGB regressor 

model = xgb.XGBRegressor()

model.fit(X_train,y_train)
xpred = model.predict(X_test)
plt.plot(X_test, xpred)
sn.distplot(y_test-xpred)
plt.scatter(y_test, xpred, alpha=.5)
print(r2_score(y_test, xpred))
df_x = pd.DataFrame({'Actual':y_test,"Predicted":xpred})

df_x.head()
# difference between the Actual and predicted value



df_x = df_x.head(25)

df_x.plot(kind='bar',figsize=(10,5))

plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')

plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')

plt.show()