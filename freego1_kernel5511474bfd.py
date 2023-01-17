"""

When faced with a Machine Learning task, finding the right algorithm for the problem could be the difference between success and failure of the assignment.

This project is about finding the most suitable Machine learning algorithms to solve a given regression problem 

"""
# Importing the important libraries

# =============================================================================

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

from pandas.plotting import scatter_matrix

from sklearn import preprocessing

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import MinMaxScaler

from sklearn.preprocessing import RobustScaler

import statsmodels.api as sm
# =============================================================================

# Reading data and Data Preprocessing

# =============================================================================

df1 = pd.read_csv('/kaggle/input/bmi-data/bmi_data.csv', skiprows = 1, names = ['Sex', 'Age','Height', 'Weight', 'BMI'])
# Data exploration

# =============================================================================

df1.info()

df1.columns

df1.head(10)

#checking for missing values

sns.heatmap(df1.isnull(), cbar =False)  #Too small to show on the map?

df1.isnull().sum()
# Checking for outliers

ax = sns.boxplot(data=df1, orient="h", palette="Set2")

#Null values are only 0.2% of the dataset. Since the affected variables are all numeric, we can fill them with their respective means.

#df2 = df1.dropna(axis=0)             

df2 = df1.fillna(df1.mean())  #checked using Paired T Test and Chi Square, but found no statistical difference between the mean and standard deviations before and after the fill.

df2.isnull().sum()
#Encoding the categorical variable "Sex"

df = pd.get_dummies(df2,columns=['Sex'],drop_first=True)

#describe() method to show the summary statistics.

df.describe()
#To get a feel for the type of data we are dealing with, we shall plot a histogram.

df.hist(bins=50, figsize=(20,15))

plt.show()
#The variable we are going to predict is the “BMI”. So let’s look at how much each independent variable correlates with this dependent variable

corr = df.corr()

ax = sns.heatmap(

    corr, 

    vmin=-1, vmax=1, center=0,

    cmap=sns.diverging_palette(20, 220, n=200),

    square=True

)

ax.set_xticklabels(

    ax.get_xticklabels(),

    rotation=45,

    horizontalalignment='right'

)
#using the corr matrix for better understanding of the correlation 

corr_matrix = df.corr()

corr_matrix['BMI'].sort_values(ascending=False)           

#We can clearly see from the corr map and/or matrix that there's negligible linear relationship between

#age and BMI, and similarly Sex and BMI. Later, we will see with OLS what their p_values might be.



#Let’s create scatter plots for each independent variable to visualize the data:

plt.scatter(df.Height, df.BMI, c='red', label='Height vs BMI')

plt.xlabel('Height(inches)')

plt.ylabel('BMI(kg/sqm')

plt.legend()

plt.show()



plt.scatter(df.Weight, df.BMI, c='green', label='Weight vs BMI')

plt.xlabel('Weight(pounds)')

plt.ylabel('BMI(kg/sqm')

plt.legend()

plt.show()
#Let's visualize the scatter plots between variables by using Pandas’ scatter_matrix function

attributes = ['BMI', 'Height', 'Weight']

scatter_matrix(df[attributes], figsize=(12, 8))

plt.show()
#There appears to be some linear relationship between the dependent and independent variables

# So let's do Regression plot

sns.regplot(x=df['Height'], y=df['BMI'], color = 'orange')
sns.regplot(x=df['Weight'], y=df['BMI'], color = 'blue')
#The scales of measurement and range of the independent variables differ, so we will do scaling

#We will try different scales to see which suits best

x = pd.DataFrame({

    'Height': df.Height,

    'Weight': df.Weight,

})



scaler = preprocessing.MinMaxScaler()

minmax_scaled_df = scaler.fit_transform(x)

minmax_scaled_df = pd.DataFrame(minmax_scaled_df, columns=['Height', 'Weight'])



scaler = preprocessing.StandardScaler()

scaled_df = scaler.fit_transform(x)

scaled_df = pd.DataFrame(scaled_df, columns=['Height', 'Weight'])

    

scaler = preprocessing.RobustScaler()

robust_scaled_df = scaler.fit_transform(x)

robust_scaled_df = pd.DataFrame(robust_scaled_df, columns=['Height', 'Weight'])





fig, (ax1, ax2, ax3, ax4) = plt.subplots(ncols=4, figsize=(9, 5))

ax1.set_title('Before Scaling')

sns.kdeplot(x['Height'], ax=ax1)

sns.kdeplot(x['Weight'], ax=ax1)



ax2.set_title('After Min-Max Scaling')

sns.kdeplot(minmax_scaled_df['Height'], ax=ax2)

sns.kdeplot(minmax_scaled_df['Weight'], ax=ax2)



ax3.set_title('After Standard Scaler')

sns.kdeplot(scaled_df['Height'], ax=ax3)

sns.kdeplot(scaled_df['Weight'], ax=ax3)



ax4.set_title('After Robust Scaling')

sns.kdeplot(robust_scaled_df['Height'], ax=ax4)

sns.kdeplot(robust_scaled_df['Weight'], ax=ax4)



plt.show()
#Based on the result, Standard Scaler is preferred. We shall be using it to scale for knn later.

# =============================================================================

# Splitting data set for modelling

# =============================================================================

# Scaling is not required for Linear regression

X = df.drop(['BMI'], axis = 1)

y = df['BMI']
# Splitting the data set 

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=0, test_size=0.3)

# =============================================================================

# Backward Elimination

# =============================================================================



X1 = sm.add_constant(X_train)

ols = sm.OLS(y_train,X1)

lr = ols.fit()



cols = list(X.columns)

pmax = 1

while (len(cols)>0):

    p= []

    X_1 = X[cols]

    X_1 = sm.add_constant(X_1)

    model = sm.OLS(y,X_1).fit()

    p = pd.Series(model.pvalues.values[1:],index = cols)      

    pmax = max(p)

    feature_with_p_max = p.idxmax()

    if(pmax>0.05):

        cols.remove(feature_with_p_max)

    else:

        break
selected_features_BE = cols

print(selected_features_BE)         #Weight and Height

print(model.summary())         

model.pvalues

# =============================================================================

# Modelling

# =============================================================================

from sklearn.linear_model import LinearRegression

import xgboost as xgb

from sklearn.ensemble import AdaBoostRegressor

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.neighbors import KNeighborsRegressor

from sklearn.svm import SVR



from sklearn.metrics import mean_squared_error, r2_score

from sklearn.model_selection import GridSearchCV





# =============================================================================

# 1.Fitting linear regressor on the test set

# =============================================================================

regressor = LinearRegression()

regressor.fit(X_train, y_train)



# Predicting the Test set results

y_pred1 = regressor.predict(X_test)



MSE_lr = mean_squared_error(y_test,y_pred1)

print(MSE_lr)
print(r2_score(y_test,y_pred1))

print(r2_score(y_train,regressor.predict(X_train)))

print('intercept:', regressor.intercept_)

print('slope:', regressor.coef_)
#We use K-Fold Validation to check the performance of our model

from sklearn.model_selection import cross_val_score

clf = LinearRegression()

cross_val_score(clf,X,y, cv=4).mean()
#Lets plot the actual vs predicted BMI and see

plt.scatter(y_test, y_pred1)

plt.xlabel("Actual BMI")

plt.ylabel("Predicted BMI")
#Regularisation:

from sklearn.linear_model import LassoCV, RidgeCV, ElasticNet

from sklearn.model_selection import cross_val_score
#Implementation of LassoCV

lasso = LassoCV(alphas=[0.0001, 0.001, 0.01, 0.1, 1, 10, 100])

print("Root Mean Squared Error (Lasso): ", np.sqrt(-cross_val_score(lasso, X, y, cv=4, scoring='neg_mean_squared_error')).mean())

print ('MSE_Lasso: ', (np.sqrt(-cross_val_score(lasso, X, y, cv=4, scoring='neg_mean_squared_error')).mean())**2)
#Implementation of ElasticNet

elastic = ElasticNet(alpha=0.001)

print("Root Mean Squared Error (ElasticNet): ", np.sqrt(-cross_val_score(elastic, X, y, cv=4, scoring='neg_mean_squared_error')).mean())

print('MSE_ElasticNet: ', (np.sqrt(-cross_val_score(elastic, X, y, cv=4, scoring='neg_mean_squared_error')).mean())**2)
#Implementation of RidgeCV

ridge = RidgeCV(alphas=[0.0001, 0.001, 0.01, 0.1, 1, 10, 100])

print("Root Mean Squared Error (Ridge): ", np.sqrt(-cross_val_score(ridge, X, y, cv=4, scoring='neg_mean_squared_error')).mean())

print('MSE_Ridge: ', (np.sqrt(-cross_val_score(ridge, X, y, cv=4, scoring='neg_mean_squared_error')).mean())**2)
# =============================================================================

# 2. Implementation of xgboost

# =============================================================================

#Tuning parameters for xgboost

tuned_parameters = [{'max_depth': [5,10, 15, 20, 25, 30],'learning_rate':[0.001, 0.01, 0.1, 0.5], 'n_estimators': [10,20, 50, 100,150,200]}]

MSE_xgb = ['mean_squared_error(y_test,y_pred2)']



for value in MSE_xgb:

    regr = GridSearchCV(xgb.XGBRegressor(), tuned_parameters, cv=4)

    regr.fit(X_train, y_train)

    y_true, y_pred2 = y_test, regr.predict(X_test)
regr.best_params_   # we best accuracy at learning_rate=0.1, max_depth =10  and n_estimators = 200
regr.best_score_
#The tuned parameters can be passed directly into the model

regr = xgb.XGBRegressor(learning_rate=0.1, max_depth=10, n_estimators=200, random_state = 0)

regr.fit(X_train, y_train)



#Predicting with Xgboost

y_pred2 = regr.predict(X_test)



MSE_xgb = mean_squared_error(y_test,y_pred2)
print(MSE_xgb)
print(r2_score(y_test,y_pred2))
print(r2_score(y_train,regr.predict(X_train)))
plt.scatter(y_test, y_pred2)

plt.xlabel("Actual BMI")

plt.ylabel("Predicted BMI")

plt.title("xgboost")
# =============================================================================

# 3. Adaboost

# =============================================================================

# Finding the best hyper-parameters for AdaBoost

tuned_parameters = [{'learning_rate': [0.1, 0.5,1,2,3,4,5], 'n_estimators': [25, 50, 100,200,300]}]

MSE_ada = ['mean_squared_error(y_test,y_pred3)']



for value in MSE_ada:

    adaregr = GridSearchCV(AdaBoostRegressor(), tuned_parameters, cv=4)

    adaregr.fit(X_train, y_train)

    y_true, y_pred3 = y_test, adaregr.predict(X_test)
adaregr.best_params_
adaregr.best_score_
#Now you can plug in the best hyper-parameter value and run the model straightaway

adaregr = AdaBoostRegressor(random_state=0, learning_rate = 2, n_estimators=50)

adaregr.fit(X, y)

#Predicting with Adaboost

y_pred3 = adaregr.predict(X_test)
MSE_ada = mean_squared_error(y_test,y_pred3)

print(r2_score(y_test,y_pred3))
print(MSE_ada)
plt.scatter(y_test, y_pred3)

plt.xlabel("Actual BMI")

plt.ylabel("Predicted BMI")

plt.title("Adaboost")
# =============================================================================

# 4. For Decision Tree

# =============================================================================

# finding the best depth

tuned_parameters = [{'max_depth': [1,2,3,4,5,10, 15, 20, 25, 50, 100,200]}]

MSE_dt = ['mean_squared_error(y_test,y_pred4)']

for value in MSE_dt:

    regressor_dt = GridSearchCV(DecisionTreeRegressor(), tuned_parameters, cv=4)

    regressor_dt.fit(X_train, y_train)

    y_true, y_pred4 = y_test, regressor_dt.predict(X_test)
regressor_dt.best_params_ 
regressor_dt.best_score_
#We found our best result on max_depth = 20. Now let's plug that in and run our model

regressor_dt = DecisionTreeRegressor(random_state=0, max_depth = 20)

regressor_dt.fit(X,y)



#Predicting with Decision Tree

y_pred4 = regressor_dt.predict(X_test)



MSE_dt = mean_squared_error(y_test,y_pred4)

print(MSE_dt)
print(r2_score(y_test,y_pred4))
print(r2_score(y_train,regressor_dt.predict(X_train)))
plt.scatter(y_test, y_pred4)

plt.xlabel("Actual BMI")

plt.ylabel("Predicted BMI")

plt.title("Decision Tree")
# =============================================================================

# 5. Random Forest

# =============================================================================

#First, let's find the best tuned_parameters for our model



tuned_parameters = [{'max_depth': [5,10, 15, 20, 50, 70], 'n_estimators': [10, 25, 50, 100,150, 200, 250]}]

MSE_rf = ['mean_squared_error(y_test, y_pred5)']



for value in MSE_rf:

    regr_rf = GridSearchCV(RandomForestRegressor(), tuned_parameters, cv=4)

    regr_rf.fit(X_train, y_train)



    y_true, y_pred5 = y_test, regr_rf.predict(X_test)
regr_rf.best_params_
#Now, let's set the best parameters and run our model

regr_rf = RandomForestRegressor(max_depth=70, random_state=0,

                             n_estimators=250)

regr_rf.fit(X, y)  

#Predicting with Random Forest

y_pred5 = regr_rf.predict(X_test)



MSE_rf = mean_squared_error(y_test,y_pred5)
print(MSE_rf)
print(r2_score(y_test,y_pred5))
print(r2_score(y_train,regr_rf.predict(X_train)))
plt.scatter(y_test, y_pred5)

plt.xlabel("Actual BMI")

plt.ylabel("Predicted BMI")

plt.title("Random Forest")
# =============================================================================

# 6. KNN

# =============================================================================

# Feature Scaling is required for Distance-based algorithms like KNN

sc = StandardScaler()

X_train_scaled = sc.fit_transform(X_train)

X_test_scaled = sc.transform(X_test)
# Let us use GridSearchCV to find the best hyper-parameters for our algorithm

tuned_parameters = [{'n_neighbors': [1,2,3,4,5,10,15,20], 'p': [1,2]}]

MSE_knn = ['mean_squared_error(y_test,y_pred)']



for i in MSE_knn:

    model = GridSearchCV(KNeighborsRegressor(), tuned_parameters, cv=4)

    model.fit(X_train_scaled, y_train)

 

    y_true, y_pred6 = y_test, model.predict(X_test_scaled)
model.best_params_  
model.best_score_
#Implementing knn with the best hyper-parameters

#Fitting knn on the training set

neigh = KNeighborsRegressor(n_neighbors = 5, metric = 'minkowski', p = 2)

neigh.fit(X_train, y_train)

# Predicting the Test set results

y_pred6 = neigh.predict(X_test)



MSE_knn = mean_squared_error(y_test,y_pred6)

print(MSE_knn)
print(r2_score(y_test,y_pred6))
print(r2_score(y_train,neigh.predict(X_train)))
plt.scatter(y_test, y_pred6)

plt.xlabel("Actual BMI")

plt.ylabel("Predicted BMI")

plt.title("KNN")
# =============================================================================

# 7. SVM

# =============================================================================

#Due to high computational cost, we will reduce the size of the datasets for our convenience

X_train7 = X_train_scaled[:1000]

y_train7 = y_train[:1000]

X_test7 = X_test[:1000]

y_test7 = y_test[:1000]
tuned_parameters = [{'kernel': ['linear', 'rbf', 'poly'], 'C':[0.1, 1], 'gamma': [0.1, 1]}]

MSE_svm = ['mean_squared_error(y_test,y_pred7)']



for value in MSE_svm:

    svr_regr = GridSearchCV(SVR(), tuned_parameters, cv=4)

    svr_regr.fit(X_train7, y_train7)

    y_true, y_pred7 = y_test7, svr_regr.predict(X_test7)
svr_regr.best_params_
svr_regr.best_score_
#Using the hyper-parameters to run the model on the entire dataset for best results

svr_regr = SVR(gamma=0.1, kernel = 'linear', C =1)

svr_regr.fit(X_train_scaled, y_train) 



#Predicting with SVM

y_pred7 = svr_regr.predict(X_test_scaled)



MSE_svm = mean_squared_error(y_test,y_pred7)
print(MSE_svm)
print(r2_score(y_test,y_pred7))
plt.scatter(y_test, y_pred7)

plt.xlabel("Actual BMI")

plt.ylabel("Predicted BMI")

plt.title("SVM")
# =============================================================================

# Since we set out to compare the models, let,s bring it all together

# =============================================================================

mse_lr = print(mean_squared_error(y_test,y_pred1))

mse_xgb = print(mean_squared_error(y_test,y_pred2))

mse_ada = print(mean_squared_error(y_test,y_pred3))

mse_dt = print(mean_squared_error(y_test,y_pred4))

mse_rf = print(mean_squared_error(y_test,y_pred5))

mse_knn = print(mean_squared_error(y_test,y_pred6))

mse_svm = print(mean_squared_error(y_test,y_pred7))
"""As can be seen from the results above, all the regressors performed well on this data set.

However, Decision Tree showed the lowest MSE and almost perfect scatter plot of predicted vs actual values thereby showing the best fit.

Similarly, it returned R2 value of approximately 100%. On the other hand, 

Adaboost had the highest MSE and the least R2 score to emerge as the relatively worst fit model for this problem."""