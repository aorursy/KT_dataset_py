# load packages and read in the data



from subprocess import check_output

import numpy as np

import xgboost as xgb

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt

import matplotlib



data = np.genfromtxt('../input/Train/train.csv', delimiter=',',skip_header=1,usecols=(1,2,3,4,5))

X = data[:,:4]

Y = data[:,4]
blobs = np.sum(X,axis=1)



plt.scatter(blobs,Y)

plt.xlabel('#non-pup sea lions')

plt.ylabel('#pups')

plt.show()
# a function to do the training and prediction

def train_pred(n_sims,X,Y,f_names,test_size):

    RMSE = np.zeros(n_sims)

    f_imp = np.zeros([n_sims,np.shape(X)[1]])



    for i in range(n_sims):



        # split the data

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size)

        # initialize XGBRegressor

        GB = xgb.XGBRegressor()



        # the parameter grid below was too much on the kaggle kernel

        #param_grid = {"learning_rate": [0.01,0.03,0.1],

        #              "objective": ['reg:linear'],

        #              "n_estimators": [300,1000,3000]}

        # do GridSearch

        #search_GB = GridSearchCV(GB,param_grid,cv=4,n_jobs=-1).fit(X_train,Y_train)

        # the best parameters should not be on the edges of the parameter grid

        #print('   ',search_GB.best_params_)

        # train the best model

        #xgb_pups = xgb.XGBRegressor(**search_GB.best_params_).fit(X_train, Y_train)



        # preselected parameters

        param_grid = {"learning_rate": 0.03,

                      "objective": 'reg:linear',

                      "n_estimators": 300}

        xgb_pups = xgb.XGBRegressor(**param_grid).fit(X_train, Y_train)



        # predict on the test set

        preds = xgb_pups.predict(X_test)



        # feature importance

        b = xgb_pups.booster()

        f_imp[i,:] = list(b.get_fscore().values())



        # rmse of prediction

        RMSE[i] = np.sqrt(mean_squared_error(Y_test, preds))

    

    # visualize the prediction of the last model

    plt.scatter(Y_test,preds,label = 'regression model')

    plt.plot(np.arange(np.max(Y_test)),np.arange(np.max(Y_test)),color='k',label='perfect prediction')

    plt.title('predictions of the last model')

    plt.legend(loc='best')

    plt.xlabel('true #pups')

    plt.ylabel('predicted #pups')

    plt.show()

    

    return RMSE, f_imp

f_names = ['adult males','subadult males','adult females','juveniles']

RMSE, f_imp = train_pred(10,X,Y,f_names,test_size=0.2)



print('RMSE = ',np.around(np.mean(RMSE),1),'+/-',np.around(np.std(RMSE),1))



plt.bar(range(len(f_names)),np.mean(f_imp,axis=0),width=0.8,yerr = np.std(f_imp,axis=0))

plt.ylabel('f score')

plt.xticks(range(len(f_names)), f_names)

plt.show()
X_sum = np.sum(X,axis=1)

X_new = np.hstack((X/X_sum[:,None],X_sum[:,None]))

f_names_new = np.append(f_names,'sum')



RMSE_new, f_imp_new = train_pred(10,X_new,Y,f_names_new,test_size=0.2)



print('RMSE = ',np.around(np.mean(RMSE_new),1),'+/-',np.around(np.std(RMSE_new),1))



plt.bar(range(len(f_names_new)),np.mean(f_imp_new,axis=0),width=0.8,yerr = np.std(f_imp_new,axis=0))

plt.ylabel('f score')

plt.xticks(range(len(f_names_new)), f_names_new,rotation='vertical')

plt.show()