# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt # data visualisation
#load dataset

data = pd.read_csv("../input/bostonhoustingmlnd/housing.csv")

prices = data['MEDV']

features = data.drop('MEDV', axis=1)

data.head()
minimum_price = np.min(prices) #minimum price in dataset

maximum_price = np.max(prices) #maximum price in dataset

mean_price = round(np.mean(prices), 2) #mean price rounded to 2 decimal places

median_price = np.median(prices) #median price

std_price = round(np.std(prices), 2) # standard deviation of price rounded to 2 decimal places



#printing results

print('Stats for Boston Housing Prices: \n')

print('Minimum price: ' + str(minimum_price) + '\n')

print('Maximum price: ' + str(maximum_price) + '\n')

print('Mean price: ' + str(mean_price) + '\n')

print('Median price: ' + str(median_price) + '\n')

print('Standard Deviation of Prices: ' + str(std_price) + '\n')

plt.figure(figsize=(20,5))



#i: index, col: column

for i, col in enumerate(features.columns):

    plt.subplot(1, 3, i+1)

    x = data[col]

    y = prices

    plt.plot(x, y, 'o')

    

    #creating regression line

    plt.plot(np.unique(x), np.poly1d(np.polyfit(x,y,1))(np.unique(x)))

    plt.title(col)

    plt.xlabel(col)

    plt.ylabel('Prices')
#import 'r2_score'

def performance_metric(y_true, y_predict):

    """ Calculates and returns the performance score between true and predicted values based on the metric chosen. """

    #calculate performance score between 'y_true' and 'y_predict'

    from sklearn.metrics import r2_score

    score = r2_score(y_true, y_predict)

    

    #return calculated score

    return score
#calculate performance of given model

score = round(performance_metric([3, -0.5, 2, 7, 4.2], [2.5, 0.0, 2.1, 7.8, 5.3]), 3)

print("Model has co-efficient of determination, R^2, of " + str(score))
true, pred = [3, -0.5, 2, 7, 4.2], [2.5, 0.0, 2.1, 7.8, 5.3]



#plot true values

true_handle = plt.scatter(true, true, alpha=0.6, color='blue', label='True')



#reference line

fit = np.poly1d(np.polyfit(true, true, 1))

lims = np.linspace(min(true)-1, max(true)+1)

plt.plot(lims, fit(lims), alpha=0.3, color='black')



#plot predicted values

pred_handle = plt.scatter (true, pred, alpha=0.6, color='red', label='Predicted')



#specify legend and show plot

plt.legend(handles=[true_handle, pred_handle], loc='upper left')

plt.show()
#importing model_selection for full set of required functions

from sklearn.model_selection import train_test_split



#shuffling and splitting data into training and testing subsets

X_train, X_test, y_train, y_test = train_test_split(features, prices, test_size=0.2, random_state=42)

print('Training and testing split successful')
import warnings

warnings.filterwarnings("ignore", category = UserWarning, module = "sklearn")



from sklearn.model_selection import ShuffleSplit, train_test_split

from sklearn.model_selection import learning_curve, validation_curve

from sklearn.tree import DecisionTreeRegressor



def ModelLearning(X, y):

    

    """ Calculates the performance of several models with varying sizes of training data. 

        The learning and testing scores for each model are then plotted. """

    

    #create 10 cross-validation sets for training and testing

    cv = ShuffleSplit(n_splits=10, test_size=0.1, train_size=None, random_state=None)

    

    #generate training set sizes increasing by 50

    train_sizes = np.rint(np.linspace(1, X.shape[0]*0.8-1, 9)) .astype(int)

    

    #create figure window

    fig = plt.figure(figsize=(10,7))

    

    #create 3 different models depending on max_depth

    for k, depth in enumerate([1,3,6,10]):

        

        #create DecisionTreeRegressor with max_depth = depth

        regressor = DecisionTreeRegressor(max_depth=depth)

        

        #calculate training and testing scores

        sizes, train_scores, test_scores = learning_curve(regressor, X, y, cv=cv, train_sizes= train_sizes, scoring = 'r2')



        #mean and standard deviation for smoothing of plot

        train_std = np.std(train_scores, axis=1)

        train_mean = np.mean(train_scores, axis=1)

        test_std = np.std(test_scores, axis=1)

        test_mean = np.std(test_scores, axis=1)



        #subplot learning curve

        ax = fig.add_subplot(2, 2, k+1)

        ax.plot(sizes, train_mean, 'o-', color='r', label='Training Score')

        ax.plot(sizes, test_mean, 'o-', color='g', label='Testing Score')

        ax.fill_between(sizes, train_mean-train_std, train_mean+train_std, alpha=0.15, color='r')

        ax.fill_between(sizes, test_mean-test_std, test_mean+test_std, alpha=0.15, color='r')



        #setting labels

        ax.set_title('Max Depth: ' + str(depth))

        ax.set_xlabel('Number of Training Points')

        ax.set_ylabel('Score')

        ax.set_xlim([0, X.shape[0]*0.8])

        ax.set_ylim([-0.05, 1.05])



    #visual changes

    ax.legend(bbox_to_anchor=(1.05, 2.05), loc='lower left', borderaxespad=0)

    fig.suptitle('Decision Tree Regressor Learning Performances', fontsize= 16, y=1.03)

    fig.tight_layout()

    fig.show()



def ModelComplexity(X, y):

    

    """ Calculates the performance of the model as model complexity increases. 

        The learning and testing errors rates are then plotted. """

    

    #create 10 cross-validation sets for training and testing

    cv = ShuffleSplit(n_splits=10, test_size=0.1, train_size=None, random_state=None)



    #vary max_depth from 1 to 10

    max_depth = np.arange(1,11)



    #calculate training and testing scores

    train_scores, test_scores = validation_curve(DecisionTreeRegressor(), X, y, param_name='max_depth', param_range=max_depth, cv=cv, scoring='r2')

    

    #mean and standard deviation for smoothing of plot

    train_std = np.std(train_scores, axis=1)

    train_mean = np.mean(train_scores, axis=1)

    test_std = np.std(test_scores, axis=1)

    test_mean = np.std(test_scores, axis=1)



    #plot validation curve

    plt.figure(figsize=(7,5))

    plt.title('Decision Tree Regressor Complexity Performance')

    plt.plot(max_depth, train_mean, 'o-', color='r', label='Training Score')

    plt.plot(max_depth, test_mean, 'o-', color='g', label='Validation Score')

    plt.fill_between(max_depth, train_mean-train_std, train_mean+train_std, alpha=0.15, color='r')

    plt.fill_between(max_depth, test_mean-test_std, test_mean+test_std, alpha=0.15, color='g')



    #legends

    plt.legend(loc='lower right')

    plt.xlabel('Maximum Depth')

    plt.ylabel('Score')

    plt.ylim([-0.05, 1.05])

    plt.show()



def PredictTrials(X, y, fitter, data):

    

    """ Performs trials of fitting and predicting data. """

    

    #store predicted prices in empty list

    prices = []

    for k in range(10):

        

        #splitting data

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=k)

 



        #fitting data

        reg = fitter(X_train, y_train)



        #make prediction and store

        pred = reg.predict([data[0]])[0]

        prices.append(pred)



        #print result

        print('Trial '+  str(k+1) + ' ' + str((pred)))



    #print price range

    print('\nRange in prices: ' + str(max(prices)-min(prices)))
#produce learning curves for varying training set sizes and maximum depths

ModelLearning(features, prices)
ModelComplexity(X_train, y_train)
#import 'make_scorer' and 'GridSearchCV'

from sklearn.metrics import make_scorer

from sklearn.model_selection import GridSearchCV



def fit_model(X, y):

    

    """ Performs grid search over the 'max_depth' parameter for a decision tree regressor trained on the input data [X, y]. """



    #create cross-validation sets from training data

    cv_sets = ShuffleSplit(n_splits=10, test_size=0.1, train_size=None, random_state=None)



     #create decision tree regressor object

    regressor = DecisionTreeRegressor(random_state=1001)



    #create dictionary for parameter 'max_depth' ranging from 1 to 10

    tree_range = range(1, 11)

    params = dict(max_depth = [1,2,3,4,5,6,7,8,9,10])



    #transform 'performance_metric' into scoring function using 'make_scorer'

    scoring_fnc = make_scorer(performance_metric)



    #creating grid search cross-validation object

    grid = GridSearchCV(regressor, params, scoring=scoring_fnc, cv=cv_sets)

    

    #fit grid search object to the data to compute optimal model

    grid = grid.fit(X, y)

    return grid.best_estimator_
#fit training data to the model using grid search

reg = fit_model(X_train, y_train)

#produce value for 'max_depth'

print("Parameter 'max_depth' is " + str(reg.get_params()['max_depth']) + ' for the optimal model.')
#produce matrix for client data

client_data = [[5, 17, 15], #client1

               [4, 32, 22], #client2

               [8, 3, 12]]  #client3



#show predictions

for i, price in enumerate(reg.predict(client_data)):

    print("Predicted selling price for Client " + str(i+1) + "'s home: $" + str(round(price, 2)))
clients = np.transpose(client_data)

pred= reg.predict(client_data)



for i, feat in enumerate(['RM', 'LSTAT', 'PTRATIO']):

    plt.scatter(features[feat], prices, alpha=0.25, c=prices)

    plt.scatter(clients[i], pred, color='black', marker='x', linewidths=2)

    plt.xlabel(feat)

    plt.ylabel('MEDV')

    plt.show()
reg = fit_model(X_train, y_train)

pred = reg.predict(X_test)

score= performance_metric(y_test, pred)

print('R^2 value: ' + str(score))
plt.hist(prices, bins=20)



for price in reg.predict(client_data):

    plt.axvline(price, lw=5, c='r')
PredictTrials(features, prices, fit_model, client_data)