import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



from scipy.stats import norm

from scipy import stats

from sklearn import preprocessing

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import StandardScaler



from sklearn.model_selection import train_test_split

from sklearn.model_selection import KFold

from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import ExtraTreesClassifier



from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import PolynomialFeatures

from sklearn.linear_model import Ridge

from sklearn.linear_model import Lasso

from sklearn.linear_model import ElasticNet



from sklearn import metrics

from sklearn.metrics import mean_squared_error

from sklearn.metrics import mean_absolute_error

from math import sqrt

from sklearn.metrics import r2_score
nyc_data = pd.read_csv('../input/new-york-city-airbnb-open-data/AB_NYC_2019.csv')
nyc_data.info()
nyc_data.head(10)
nyc_data.isnull().sum()
plt.figure(figsize=(15,12))

sns.scatterplot(x='room_type', y='price', data=nyc_data)



plt.xlabel("Room Type", size=13)

plt.ylabel("Price", size=13)

plt.title("Room Type vs Price",size=15, weight='bold')
plt.figure(figsize=(20,15))

sns.scatterplot(x="room_type", y="price",

            hue="neighbourhood_group", size="neighbourhood_group",

            sizes=(50, 200), palette="Dark2", data=nyc_data)



plt.xlabel("Room Type", size=13)

plt.ylabel("Price", size=13)

plt.title("Room Type vs Price vs Neighbourhood Group",size=15, weight='bold')
plt.figure(figsize=(20,15))

sns.set_palette("Set1")



sns.lineplot(x='price', y='number_of_reviews', 

             data=nyc_data[nyc_data['neighbourhood_group']=='Brooklyn'],

             label='Brooklyn')

sns.lineplot(x='price', y='number_of_reviews', 

             data=nyc_data[nyc_data['neighbourhood_group']=='Manhattan'],

             label='Manhattan')

sns.lineplot(x='price', y='number_of_reviews', 

             data=nyc_data[nyc_data['neighbourhood_group']=='Queens'],

             label='Queens')

sns.lineplot(x='price', y='number_of_reviews', 

             data=nyc_data[nyc_data['neighbourhood_group']=='Staten Island'],

             label='Staten Island')

sns.lineplot(x='price', y='number_of_reviews', 

             data=nyc_data[nyc_data['neighbourhood_group']=='Bronx'],

             label='Bronx')

plt.xlabel("Price", size=13)

plt.ylabel("Number of Reviews", size=13)

plt.title("Price vs Number of Reviews vs Neighbourhood Group",size=15, weight='bold')
nyc_data['neighbourhood_group']= nyc_data['neighbourhood_group'].astype("category").cat.codes

nyc_data['neighbourhood'] = nyc_data['neighbourhood'].astype("category").cat.codes

nyc_data['room_type'] = nyc_data['room_type'].astype("category").cat.codes

nyc_data.info()
plt.figure(figsize=(10,10))

sns.distplot(nyc_data['price'], fit=norm)

plt.title("Price Distribution Plot",size=15, weight='bold')
nyc_data['price_log'] = np.log(nyc_data.price+1)
plt.figure(figsize=(12,10))

sns.distplot(nyc_data['price_log'], fit=norm)

plt.title("Log-Price Distribution Plot",size=15, weight='bold')
plt.figure(figsize=(7,7))

stats.probplot(nyc_data['price_log'], plot=plt)

plt.show()
nyc_model = nyc_data.drop(columns=['name','id' ,'host_id','host_name', 

                                   'last_review','price'])

nyc_model.isnull().sum()
mean = nyc_model['reviews_per_month'].mean()

nyc_model['reviews_per_month'].fillna(mean, inplace=True)

nyc_model.isnull().sum()
plt.figure(figsize=(15,12))

palette = sns.diverging_palette(20, 220, n=256)

corr=nyc_model.corr(method='pearson')

sns.heatmap(corr, annot=True, fmt=".2f", cmap=palette, vmax=.3, center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .5}).set(ylim=(11, 0))

plt.title("Correlation Matrix",size=15, weight='bold')
nyc_model_x, nyc_model_y = nyc_model.iloc[:,:-1], nyc_model.iloc[:,-1]
f, axes = plt.subplots(5, 2, figsize=(15, 20))

sns.residplot(nyc_model_x.iloc[:,0],nyc_model_y, lowess=True, ax=axes[0, 0], 

                          scatter_kws={'alpha': 0.5}, 

                          line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})

sns.residplot(nyc_model_x.iloc[:,1],nyc_model_y, lowess=True, ax=axes[0, 1],

                          scatter_kws={'alpha': 0.5}, 

                          line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})

sns.residplot(nyc_model_x.iloc[:,2],nyc_model_y, lowess=True, ax=axes[1, 0], 

                          scatter_kws={'alpha': 0.5}, 

                          line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})

sns.residplot(nyc_model_x.iloc[:,3],nyc_model_y, lowess=True, ax=axes[1, 1], 

                          scatter_kws={'alpha': 0.5}, 

                          line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})

sns.residplot(nyc_model_x.iloc[:,4],nyc_model_y, lowess=True, ax=axes[2, 0], 

                          scatter_kws={'alpha': 0.5}, 

                          line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})

sns.residplot(nyc_model_x.iloc[:,5],nyc_model_y, lowess=True, ax=axes[2, 1], 

                          scatter_kws={'alpha': 0.5}, 

                          line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})

sns.residplot(nyc_model_x.iloc[:,6],nyc_model_y, lowess=True, ax=axes[3, 0], 

                          scatter_kws={'alpha': 0.5}, 

                          line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})

sns.residplot(nyc_model_x.iloc[:,7],nyc_model_y, lowess=True, ax=axes[3, 1], 

                          scatter_kws={'alpha': 0.5}, 

                          line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})

sns.residplot(nyc_model_x.iloc[:,8],nyc_model_y, lowess=True, ax=axes[4, 0], 

                          scatter_kws={'alpha': 0.5}, 

                          line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})

sns.residplot(nyc_model_x.iloc[:,9],nyc_model_y, lowess=True, ax=axes[4, 1], 

                          scatter_kws={'alpha': 0.5}, 

                          line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})

plt.setp(axes, yticks=[])

plt.tight_layout()
#Eigen vector of a correlation matrix.

multicollinearity, V=np.linalg.eig(corr)

multicollinearity
scaler = StandardScaler()

nyc_model_x = scaler.fit_transform(nyc_model_x)
X_train, X_test, y_train, y_test = train_test_split(nyc_model_x, nyc_model_y, test_size=0.3,random_state=42)
lab_enc = preprocessing.LabelEncoder()



feature_model = ExtraTreesClassifier(n_estimators=50)

feature_model.fit(X_train,lab_enc.fit_transform(y_train))



plt.figure(figsize=(7,7))

feat_importances = pd.Series(feature_model.feature_importances_, index=nyc_model.iloc[:,:-1].columns)

feat_importances.nlargest(10).plot(kind='barh')

plt.show()
### Linear Regression ###



def linear_reg(input_x, input_y, cv=5):

    ## Defining parameters

    model_LR= LinearRegression()



    parameters = {'fit_intercept':[True,False], 'normalize':[True,False], 'copy_X':[True, False]}



    ## Building Grid Search algorithm with cross-validation and Mean Squared Error score.



    grid_search_LR = GridSearchCV(estimator=model_LR,  

                         param_grid=parameters,

                         scoring='neg_mean_squared_error',

                         cv=cv,

                         n_jobs=-1)



    ## Lastly, finding the best parameters.



    grid_search_LR.fit(input_x, input_y)

    best_parameters_LR = grid_search_LR.best_params_  

    best_score_LR = grid_search_LR.best_score_ 

    print(best_parameters_LR)

    print(best_score_LR)





# linear_reg(nyc_model_x, nyc_model_y)
### Ridge Regression ###



def ridge_reg(input_x, input_y, cv=5):

    ## Defining parameters

    model_Ridge= Ridge()



    # prepare a range of alpha values to test

    alphas = np.array([1,0.1,0.01,0.001,0.0001,0])

    normalizes= ([True,False])



    ## Building Grid Search algorithm with cross-validation and Mean Squared Error score.



    grid_search_Ridge = GridSearchCV(estimator=model_Ridge,  

                         param_grid=(dict(alpha=alphas, normalize= normalizes)),

                         scoring='neg_mean_squared_error',

                         cv=cv,

                         n_jobs=-1)



    ## Lastly, finding the best parameters.



    grid_search_Ridge.fit(input_x, input_y)

    best_parameters_Ridge = grid_search_Ridge.best_params_  

    best_score_Ridge = grid_search_Ridge.best_score_ 

    print(best_parameters_Ridge)

    print(best_score_Ridge)

    

# ridge_reg(nyc_model_x, nyc_model_y)
### Lasso Regression ###



def lasso_reg(input_x, input_y, cv=5):

    ## Defining parameters

    model_Lasso= Lasso()



    # prepare a range of alpha values to test

    alphas = np.array([1,0.1,0.01,0.001,0.0001,0])

    normalizes= ([True,False])



    ## Building Grid Search algorithm with cross-validation and Mean Squared Error score.



    grid_search_lasso = GridSearchCV(estimator=model_Lasso,  

                         param_grid=(dict(alpha=alphas, normalize= normalizes)),

                         scoring='neg_mean_squared_error',

                         cv=cv,

                         n_jobs=-1)



    ## Lastly, finding the best parameters.



    grid_search_lasso.fit(input_x, input_y)

    best_parameters_lasso = grid_search_lasso.best_params_  

    best_score_lasso = grid_search_lasso.best_score_ 

    print(best_parameters_lasso)

    print(best_score_lasso)



# lasso_reg(nyc_model_x, nyc_model_y)
### ElasticNet Regression ###



def elastic_reg(input_x, input_y,cv=5):

    ## Defining parameters

    model_grid_Elastic= ElasticNet()



    # prepare a range of alpha values to test

    alphas = np.array([1,0.1,0.01,0.001,0.0001,0])

    normalizes= ([True,False])



    ## Building Grid Search algorithm with cross-validation and Mean Squared Error score.



    grid_search_elastic = GridSearchCV(estimator=model_grid_Elastic,  

                         param_grid=(dict(alpha=alphas, normalize= normalizes)),

                         scoring='neg_mean_squared_error',

                         cv=cv,

                         n_jobs=-1)



    ## Lastly, finding the best parameters.



    grid_search_elastic.fit(input_x, input_y)

    best_parameters_elastic = grid_search_elastic.best_params_  

    best_score_elastic = grid_search_elastic.best_score_ 

    print(best_parameters_elastic)

    print(best_score_elastic)



# elastic_reg(nyc_model_x, nyc_model_y)
kfold_cv=KFold(n_splits=5, random_state=42, shuffle=False)

for train_index, test_index in kfold_cv.split(nyc_model_x,nyc_model_y):

    X_train, X_test = nyc_model_x[train_index], nyc_model_x[test_index]

    y_train, y_test = nyc_model_y[train_index], nyc_model_y[test_index]
Poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)

X_train = Poly.fit_transform(X_train)

X_test = Poly.fit_transform(X_test)
##Linear Regression

lr = LinearRegression(copy_X= True, fit_intercept = True, normalize = True)

lr.fit(X_train, y_train)

lr_pred= lr.predict(X_test)



#Ridge Model

ridge_model = Ridge(alpha = 0.01, normalize = True)

ridge_model.fit(X_train, y_train)             

pred_ridge = ridge_model.predict(X_test) 



#Lasso Model

Lasso_model = Lasso(alpha = 0.001, normalize =False)

Lasso_model.fit(X_train, y_train)

pred_Lasso = Lasso_model.predict(X_test) 



#ElasticNet Model

model_enet = ElasticNet(alpha = 0.01, normalize=False)

model_enet.fit(X_train, y_train) 

pred_test_enet= model_enet.predict(X_test)

nyc_model_xx= nyc_model.drop(columns=['neighbourhood_group', 'room_type'])
nyc_model_xx, nyc_model_yx = nyc_model_xx.iloc[:,:-1], nyc_model_xx.iloc[:,-1]

X_train_x, X_test_x, y_train_x, y_test_x = train_test_split(nyc_model_xx, nyc_model_yx, test_size=0.3,random_state=42)
scaler = StandardScaler()

nyc_model_xx = scaler.fit_transform(nyc_model_xx)
### Linear Regression ###

# linear_reg(nyc_model_xx, nyc_model_yx, cv=4)
### Ridge Regression ###

# ridge_reg(nyc_model_xx, nyc_model_yx, cv=4)
### Lasso Regression ###

# lasso_reg(nyc_model_xx, nyc_model_yx, cv=4)
### ElasticNet Regression ###

# elastic_reg(nyc_model_xx, nyc_model_yx, cv=4)
kfold_cv=KFold(n_splits=4, random_state=42, shuffle=False)

for train_index, test_index in kfold_cv.split(nyc_model_xx,nyc_model_yx):

    X_train_x, X_test_x = nyc_model_xx[train_index], nyc_model_xx[test_index]

    y_train_x, y_test_x = nyc_model_yx[train_index], nyc_model_yx[test_index]
Poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)

X_train_x = Poly.fit_transform(X_train_x)

X_test_x = Poly.fit_transform(X_test_x)
###Linear Regression

lr_x = LinearRegression(copy_X= True, fit_intercept = True, normalize = True)

lr_x.fit(X_train_x, y_train_x)

lr_pred_x= lr_x.predict(X_test_x)



###Ridge

ridge_x = Ridge(alpha = 0.01, normalize = True)

ridge_x.fit(X_train_x, y_train_x)           

pred_ridge_x = ridge_x.predict(X_test_x) 



###Lasso

Lasso_x = Lasso(alpha = 0.001, normalize =False)

Lasso_x.fit(X_train_x, y_train_x)

pred_Lasso_x = Lasso_x.predict(X_test_x) 



##ElasticNet

model_enet_x = ElasticNet(alpha = 0.01, normalize=False)

model_enet_x.fit(X_train_x, y_train_x) 

pred_train_enet_x= model_enet_x.predict(X_train_x)

pred_test_enet_x= model_enet_x.predict(X_test_x)

print('-------------Lineer Regression-----------')



print('--Phase-1--')

print('MAE: %f'% mean_absolute_error(y_test, lr_pred))

print('RMSE: %f'% np.sqrt(mean_squared_error(y_test, lr_pred)))   

print('R2 %f' % r2_score(y_test, lr_pred))



print('--Phase-2--')

print('MAE: %f'% mean_absolute_error(y_test_x, lr_pred_x))

print('RMSE: %f'% np.sqrt(mean_squared_error(y_test_x, lr_pred_x)))   

print('R2 %f' % r2_score(y_test_x, lr_pred_x))



print('---------------Ridge ---------------------')



print('--Phase-1--')

print('MAE: %f'% mean_absolute_error(y_test, pred_ridge))

print('RMSE: %f'% np.sqrt(mean_squared_error(y_test, pred_ridge)))   

print('R2 %f' % r2_score(y_test, pred_ridge))



print('--Phase-2--')

print('MAE: %f'% mean_absolute_error(y_test_x, pred_ridge_x))

print('RMSE: %f'% np.sqrt(mean_squared_error(y_test_x, pred_ridge_x)))   

print('R2 %f' % r2_score(y_test_x, pred_ridge_x))



print('---------------Lasso-----------------------')



print('--Phase-1--')

print('MAE: %f' % mean_absolute_error(y_test, pred_Lasso))

print('RMSE: %f' % np.sqrt(mean_squared_error(y_test, pred_Lasso)))

print('R2 %f' % r2_score(y_test, pred_Lasso))



print('--Phase-2--')

print('MAE: %f' % mean_absolute_error(y_test_x, pred_Lasso_x))

print('RMSE: %f' % np.sqrt(mean_squared_error(y_test_x, pred_Lasso_x)))

print('R2 %f' % r2_score(y_test_x, pred_Lasso_x))



print('---------------ElasticNet-------------------')



print('--Phase-1 --')

print('MAE: %f' % mean_absolute_error(y_test,pred_test_enet)) #RMSE

print('RMSE: %f' % np.sqrt(mean_squared_error(y_test,pred_test_enet))) #RMSE

print('R2 %f' % r2_score(y_test, pred_test_enet))



print('--Phase-2--')

print('MAE: %f' % mean_absolute_error(y_test_x,pred_test_enet_x)) #RMSE

print('RMSE: %f' % np.sqrt(mean_squared_error(y_test_x,pred_test_enet_x))) #RMSE

print('R2 %f' % r2_score(y_test_x, pred_test_enet_x))
fig, ((ax1, ax2, ax3, ax4), (ax5, ax6, ax7, ax8)) = plt.subplots(2, 4, figsize=(30, 20))

fig.suptitle('True Values vs Predictions')



ax1.scatter(y_test, lr_pred)

ax1.set_title('Linear Regression - Phase-1')



ax2.scatter(y_test, pred_ridge)

ax2.set_title('Ridge - Phase-1')



ax3.scatter(y_test, pred_Lasso)

ax3.set_title('Lasso - Phase-1')



ax4.scatter(y_test, pred_test_enet)

ax4.set_title('ElasticNet - Phase-1')



ax5.scatter(y_test_x, lr_pred_x)

ax5.set_title('Linear Regression - Phase-2')



ax6.scatter(y_test_x, pred_ridge_x)

ax6.set_title('Ridge - Phase-2')



ax7.scatter(y_test_x, pred_Lasso_x)

ax7.set_title('Lasso - Phase-2')



ax8.scatter(y_test_x, pred_test_enet_x)

ax8.set_title('ElasticNet - Phase-2')



for ax in fig.get_axes():

    ax.set(xlabel='True Values', ylabel='Predictions')
