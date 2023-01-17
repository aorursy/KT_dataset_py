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
# importing modules and data



import numpy as np

import pandas as pd

pd.set_option('display.max_columns', 70)

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score

from xgboost import XGBClassifier

#from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import GridSearchCV

from sklearn import preprocessing

#from sklearn.preprocessing import StandardScaler

#from sklearn.metrics import mean_squared_error

from sklearn.model_selection import learning_curve

from sklearn.model_selection import ShuffleSplit

import warnings



warnings.simplefilter(action='ignore')



train = pd.read_csv('../input/learn-together/train.csv')

test = pd.read_csv('../input/learn-together/test.csv')



cover_type = {1:'Spruce/Fir', 2:'Lodgepole Pine',3:'Ponderosa Pine',4:'Cottonwood/Willow',5:'Aspen',6:'Douglas-fir',7:'Krummholz'}

train['Cover_type_description'] = train['Cover_Type'].map(cover_type)



# I put together train and test data to work on both dataset at the same time



combined_data = [train, test]



def distance(a,b):

    return np.sqrt(np.power(a,2)+np.power(b,2))



# now I can classify different soil type according to their stoniness



extremely_stony = [1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1]

stony = [0,1,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

rubbly = [0,0,1,1,1,0,0,0,0,1,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

other = [0,0,0,0,0,0,1,1,0,0,0,0,0,1,0,1,1,0,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0]

   

# grouping for Aspect, Elevation and Slope 



for data in combined_data:

    

    # grouping of Aspect feature

    data['Azimut'] = 0

    data['Azimut'].loc[(data['Aspect'] == 0)] = 'north'

    data['Azimut'].loc[(data['Aspect']>0 ) & (data['Aspect']< 90)] = 'north_east'

    data['Azimut'].loc[data['Aspect']==90] = 'east'

    data['Azimut'].loc[(data['Aspect'] >90) & (data['Aspect'] <180)] = 'south_east'

    data['Azimut'].loc[data['Aspect']==180] = 'south'

    data['Azimut'].loc[(data['Aspect']>180) & (data['Aspect']<270)] = 'south_west'

    data['Azimut'].loc[data['Aspect']== 270] = 'west'

    data['Azimut'].loc[(data['Aspect']> 270) & (data['Aspect']< 360)] = 'noth_west'

    data['Azimut'].loc[data['Aspect']== 360] = 'north'

    

    #grouping of Elevation feature

    

    data['Elevation_bins'] = 0

    data['Elevation_bins'].loc[data['Elevation']<= 2000] = 'less than 2000'

    data['Elevation_bins'].loc[(data['Elevation'] > 2000) & (data['Elevation']< 2500)] = 'between 2000 and 2500'

    data['Elevation_bins'].loc[(data['Elevation'] > 2500) & (data['Elevation'] <= 3000)] = 'between 2000 and 3000'

    data['Elevation_bins'].loc[(data['Elevation'] > 3000) & (data['Elevation'] <= 3500)] = 'between 3000 and 3500'

    data['Elevation_bins'].loc[data['Elevation'] > 3500] = 'greater than 3500'

    

    # grouping for slope

    data['Slope_category'] = 0

    data['Slope_category'].loc[(data['Slope'] <= 10)] = 'slope less than 10'

    data['Slope_category'].loc[(data['Slope'] > 10) & (data['Slope'] <= 20)] = 'slope between 10 and 20'

    data['Slope_category'].loc[(data['Slope'] > 20) & (data['Slope'] <= 30)] = 'slope between 20 and 30'

    data['Slope_category'].loc[(data['Slope'] > 30)] = 'slope greater than 30'



    # for hillshade I take the mean and for I create a new variable Distance to hydrology



    data['mean_Hillshade'] = (data['Hillshade_9am']+ data['Hillshade_Noon']+data['Hillshade_3pm'])/3

    data['Distance_to_hidrology'] = distance(data['Horizontal_Distance_To_Hydrology'],data['Vertical_Distance_To_Hydrology'])



    data['extremely_stony_level'] = data[[col for col in data.columns if col.startswith("Soil")]]@extremely_stony

    data['stony'] = data[[col for col in data.columns if col.startswith("Soil")]]@stony

    data['rubbly'] = data[[col for col in data.columns if col.startswith("Soil")]]@rubbly

    data['other'] = data[[col for col in data.columns if col.startswith("Soil")]]@other



    data['Hillshade_noon_3pm'] = data['Hillshade_Noon']- data['Hillshade_3pm']

    data['Hillshade_3pm_9am'] = data['Hillshade_3pm']- data['Hillshade_9am']

    data['Hillshade_9am_noon'] = data['Hillshade_9am']- data['Hillshade_Noon']



    # as I discovered in my previous kernel distance (from firepoints, roads and from hidrology) is an important variable so I will create new features based on that ones



    data['Up_the_water'] = data['Vertical_Distance_To_Hydrology'] > 0

    data['Total_horizontal_distance'] = data['Horizontal_Distance_To_Hydrology']+ data['Horizontal_Distance_To_Roadways']+ data['Horizontal_Distance_To_Fire_Points']

    data['Elevation_of_hydrology'] = data['Elevation']+ data['Vertical_Distance_To_Hydrology']

    data['Distance_to_firepoints plus Distance_to_roads'] = data['Horizontal_Distance_To_Fire_Points']+ data['Horizontal_Distance_To_Roadways']

    data['Distance_to_roads plus distance_to_hydrology'] = data['Horizontal_Distance_To_Roadways'] + data['Horizontal_Distance_To_Hydrology']

    data['Distance_to_firepoints minus Distance_to_roads'] = data['Horizontal_Distance_To_Fire_Points']- data['Horizontal_Distance_To_Roadways']

    data['Distance_to_roads minus distance_to_hydrology'] = data['Horizontal_Distance_To_Roadways'] - data['Horizontal_Distance_To_Hydrology']



 
soil_columns = [col for col in train.columns if col.startswith('Soil')]





# we can drop soil columns because we have group each soil according to their stoniness



for data in combined_data:

    data.drop(columns = soil_columns, inplace=True)
# for categorical values we can encode them into dummy values 

    

columns_to_encode = ['Azimut','Elevation_bins','Slope_category','Up_the_water']

    

train = pd.get_dummies(train,columns = columns_to_encode)

test = pd.get_dummies(test,columns = columns_to_encode)



# once we have encoded the columns we can drop them from both training and testing dataset



for data in combined_data:

    data.drop(columns= columns_to_encode, inplace=True)
# scaling variables





columns_to_scale = ['Elevation', 'Aspect', 'Slope',

       'Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology',

       'Horizontal_Distance_To_Roadways', 'Hillshade_9am', 'Hillshade_Noon',

       'Hillshade_3pm', 'Horizontal_Distance_To_Fire_Points',

       'mean_Hillshade', 'Distance_to_hidrology','Hillshade_noon_3pm', 'Hillshade_3pm_9am',

       'Hillshade_9am_noon', 'Total_horizontal_distance',

       'Elevation_of_hydrology',

       'Distance_to_firepoints plus Distance_to_roads',

       'Distance_to_roads plus distance_to_hydrology',

       'Distance_to_firepoints minus Distance_to_roads',

       'Distance_to_roads minus distance_to_hydrology']

    



s_scaler = preprocessing.StandardScaler()

train_columns_to_scale = train[columns_to_scale]

test_columns_to_scale = test[columns_to_scale]



train_scaled = s_scaler.fit_transform(train_columns_to_scale)

test_scaled = s_scaler.fit_transform(test_columns_to_scale)



train_scaled_df = pd.DataFrame(data = train_scaled, columns = columns_to_scale)

test_scaled_df = pd.DataFrame(data = test_scaled, columns = columns_to_scale)



# dropping columns scaled from training and testing dataset

train.drop(columns = columns_to_scale, inplace = True)

test.drop(columns = columns_to_scale, inplace = True)



#now we can concatenate scaled columns to both training and testing dataset

    

train_final = pd.concat([train , train_scaled_df],axis = 1)



test_final = pd.concat([test  ,test_scaled_df],axis = 1)





#defining targer variable and features

target = 'Cover_Type'

features = [ col for col in train_final.columns if col not in ['Id','Cover_Type','Cover_type_description']]





X = train_final[features]

y = train_final[target]



X_train, X_test,y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 1)





from sklearn.model_selection import KFold

from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier,BaggingClassifier,GradientBoostingClassifier,ExtraTreesClassifier

from sklearn import tree, neighbors, svm

from sklearn.naive_bayes import GaussianNB



ML_algo = [

        #Ensemble methods

        AdaBoostClassifier(),

        RandomForestClassifier(),

        BaggingClassifier(),

        GradientBoostingClassifier(),

        ExtraTreesClassifier(),

        #XGBoost

        XGBClassifier(),

        #Tree

        tree.DecisionTreeClassifier(),

        #Nearest Neighbour

        neighbors.KNeighborsClassifier(),

        # support vector machine

        svm.SVC(),

        #Gaussian Naive Bayes

        GaussianNB()

                

        ]





Kfold = KFold(n_splits= 10, random_state= 1)



# create a dataframe to compare ML_algo based on neagative mean absolute error



ML_algo_columns = ['ML_name', 'ML_parameters', 'ML_negative_mean_square_error','ML_standard_deviation']



ML_compare = pd.DataFrame(columns=ML_algo_columns)





row = 0



for algo in ML_algo:

    name = algo.__class__.__name__

    ML_compare.loc[row, 'ML_name'] = str(name)

    ML_compare.loc[row, 'ML_parameters'] = str(algo.get_params())

    # we use cross validation with KFold to score models

    cv_results = cross_val_score(estimator =algo, X= X_train,scoring='neg_mean_squared_error', y=y_train,cv=Kfold)

    ML_compare.loc[row, 'ML_negative_mean_square_error'] = str(cv_results.mean())

    ML_compare.loc[row,'ML_standard_deviation'] = str(cv_results.std())

    row = row +1



   

print(ML_compare)

def plot_learning_curve(estimator, title, X,y, ylim =None, cv = None, n_jobs = None, train_sizes = np.linspace(.1,1.0,5)):

    plt.figure()

    plt.title(title)

    if ylim is not None:

        plt.ylim(*ylim)

    plt.xlabel("Training samples")

    plt.ylabel("Score")

    train_sizes, train_scores, test_scores = learning_curve(estimator, X,y,cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)

    train_scores_mean = np.mean(train_scores, axis = 1)

    train_scores_std = np.std(train_scores, axis = 1)

    test_scores_mean = np.mean(test_scores, axis = 1)

    test_scores_std = np.mean(test_scores, axis =1)

    plt.grid()

    plt.fill_between(train_sizes,train_scores_mean-train_scores_std,train_scores_mean+train_scores_std,alpha = 0.1, color ="r")

    plt.fill_between(train_sizes,test_scores_mean-test_scores_std,test_scores_mean+test_scores_std,alpha = 0.1, color ="g")

    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",label="Training score")

    plt.plot(train_sizes,test_scores_mean,'o-',color = 'g', label ="Cross_validation score")

    plt.legend(loc ="best")

    return plt
# Params decision for RandomForest



random_forest  = RandomForestClassifier()

params_decision_random_forest = {'criterion':['gini','entropy'],'max_features':['auto','sqrt','log2']}

grid_search_random_forest = GridSearchCV(random_forest, param_grid =params_decision_random_forest, cv=Kfold,scoring= 'neg_mean_squared_error', n_jobs=-1)

grid_search_random_forest.fit(X_train, y_train)

print('Best parameters for random forest classifier:', grid_search_random_forest.best_params_)





decision_random_forest = RandomForestClassifier(criterion = grid_search_random_forest.best_params_['criterion'], max_features = grid_search_random_forest.best_params_['max_features']) 

decision_random_forest.fit(X_train,y_train)





if False:

    title = "Random Forest learning curve"

    cv_shuffle = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)

    plot_learning_curve(decision_random_forest, title, X, y, cv=cv_shuffle, n_jobs=-1)

    plt.show()
#Best parameters for bagging classifier



bagging_classifier = BaggingClassifier()

params_decision_for_bagging = {'n_estimators':[10,50,75],'max_samples' : [0.05, 0.1, 0.2, 0.5]}

grid_search_bagging= GridSearchCV(bagging_classifier, param_grid =params_decision_for_bagging, cv=Kfold,scoring= 'neg_mean_squared_error', n_jobs=-1)

grid_search_bagging.fit(X_train, y_train)

print('Best parameters for bagging classifier:', grid_search_bagging.best_params_)





decision_bagging_classifier = BaggingClassifier(n_estimators = grid_search_bagging.best_params_['n_estimators'], max_samples = grid_search_bagging.best_params_['max_samples']) 

decision_bagging_classifier.fit(X_train,y_train)



if False:

    title = "Bagging classifier learning curve"

    cv_shuffle = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)

    plot_learning_curve(decision_bagging_classifier, title, X, y, cv=cv_shuffle, n_jobs=-1)

    plt.show()

#Best parameters for ExtraTreeClassifier

 

extra_tree = ExtraTreesClassifier()

params_decision_for_extra_tree= {'n_estimators':[10,50,75],'criterion' : ['gini','entropy']}

grid_search_extra_tree= GridSearchCV(extra_tree, param_grid =params_decision_for_extra_tree, cv=Kfold,scoring= 'neg_mean_squared_error', n_jobs=-1)

grid_search_extra_tree.fit(X_train, y_train)

print('Best parameters for extra tree classifier:', grid_search_extra_tree.best_params_)







decision_extra_tree_classifier = ExtraTreesClassifier(n_estimators = grid_search_extra_tree.best_params_['n_estimators'], criterion = grid_search_extra_tree.best_params_['criterion']) 

decision_extra_tree_classifier.fit(X_train,y_train)



if False:

    title = "Extra tree classifier learning curve"

    cv_shuffle = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)

    plot_learning_curve(decision_extra_tree_classifier, title, X, y, cv=cv_shuffle, n_jobs=-1)

    plt.show()
#Best parameters for GradientBoostingClassifier



gradient_boosting = GradientBoostingClassifier()

params_decision_gradient_boosting = { 'learning_rate':[0.03,0.05], 'n_estimators':[10,50,75]}

grid_search_gradient_boosting = GridSearchCV(gradient_boosting, param_grid =params_decision_gradient_boosting , cv=Kfold,scoring= 'neg_mean_squared_error', n_jobs=-1)

grid_search_gradient_boosting.fit(X_train, y_train)

print('Best parameters for decision gradient boosting:', grid_search_gradient_boosting.best_params_)



decision_gradient_boosting = GradientBoostingClassifier(n_estimators= grid_search_gradient_boosting.best_params_['n_estimators'], learning_rate= grid_search_gradient_boosting.best_params_['learning_rate']) 

decision_gradient_boosting.fit(X_train,y_train)



if False:

    title = "Gradient Boosting learning curve"

    cv_shuffle = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)

    plot_learning_curve(decision_gradient_boosting, title, X, y, cv=cv_shuffle, n_jobs=-1)

    plt.show()
#Best parameters for Decision tree classifier

decision_tree = tree.DecisionTreeClassifier()

params_decision_tree= {'criterion':['gini','entropy'], 'max_features':['auto','sqrt','log2']}

grid_search_decision_tree = GridSearchCV(decision_tree, param_grid =params_decision_tree , cv=Kfold,scoring= 'neg_mean_squared_error', n_jobs=-1)

grid_search_decision_tree.fit(X_train, y_train)

print('Best parameters for decision tree classifier:', grid_search_decision_tree.best_params_)





decision_tree_model = tree.DecisionTreeClassifier(criterion = grid_search_decision_tree.best_params_['criterion'], max_features = grid_search_decision_tree.best_params_['max_features']) 

decision_tree_model.fit(X_train,y_train)



if False:

    title = "Decision tree classifier"

    cv_shuffle = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)

    plot_learning_curve(decision_tree_model, title, X, y, cv=cv_shuffle, n_jobs=-1)

    plt.show()
#Best parameters for XGBClassifier

xgbclassifier = XGBClassifier()  

params_xgbclassifier = {'n_estimators': [50,100,150], 'learning_rate':[0.01,0.03,0.05]}

grid_search_xgboost = GridSearchCV(xgbclassifier, param_grid =params_xgbclassifier,cv=Kfold,scoring= 'neg_mean_squared_error', n_jobs=-1 )

grid_search_xgboost.fit(X_train, y_train)

print('Best parameters for xgboost classifier :', grid_search_xgboost.best_params_)





#creare modello con i parametri ottimali

xgboost_model = XGBClassifier(n_estimators = grid_search_xgboost.best_params_['n_estimators'],learning_rate = grid_search_xgboost.best_params_['learning_rate'])

xgboost_model.fit(X_train,y_train)





if False:

    title = "XGboost classifier"

    cv_shuffle = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)

    plot_learning_curve(xgboost_model, title, X, y, cv=cv_shuffle, n_jobs=-1)

    plt.show()
#Best parameters for SCV



SVC = svm.SVC()

params_decision_SVC = {'C':[0.5,1.0]}

grid_search_SVC = GridSearchCV(SVC, param_grid =params_decision_SVC , cv=Kfold,scoring= 'neg_mean_squared_error', n_jobs=-1)

grid_search_SVC.fit(X_train, y_train)

print('best parameters for SVC:', grid_search_SVC.best_params_)



decision_SVC = svm.SVC(C = grid_search_SVC.best_params_['C']) 

decision_SVC.fit(X_train,y_train) 





if False:

    title = "SVC classifier"

    cv_shuffle = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)

    plot_learning_curve(decision_SVC, title, X, y, cv=cv_shuffle, n_jobs=-1)

    plt.show()
#Best parameters for KNeighborsClassifier



KN_classifier = neighbors.KNeighborsClassifier()

params_KN = {'weights':['uniform','distance']}

grid_search_KN = GridSearchCV(KN_classifier, param_grid = params_KN, cv=Kfold,scoring = 'neg_mean_squared_error' ,n_jobs=-1)

grid_search_KN.fit(X_train, y_train)

print('Best parameters for KN classifier :', grid_search_KN.best_params_)





KN_model = neighbors.KNeighborsClassifier(weights= grid_search_KN.best_params_['weights'])

KN_model.fit(X_train,y_train)





if False:

    title = "KN classifier"

    cv_shuffle = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)

    plot_learning_curve(KN_model, title, X, y, cv=cv_shuffle, n_jobs=-1)

    plt.show()
from sklearn.ensemble import VotingClassifier



#list of models

estimators = [('Random forest',decision_random_forest),('Bagging',decision_bagging_classifier),('Extra tree classifier', decision_extra_tree_classifier),('Gradient boosting',decision_gradient_boosting),('Decision tree',decision_tree_model),('Xgboost',xgboost_model),('SCV',decision_SVC),('KNeighbors',KN_model)]

ensemble = VotingClassifier(estimators, voting = 'hard')



ensemble.fit(X_train, y_train)



title = "Ensemble classifier learning curve"

cv_shuffle = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)

plot_learning_curve(ensemble, title, X, y, cv=cv_shuffle, n_jobs=-1)

plt.show()



X = test_final[features]

y_pred = ensemble.predict(X)
#submitting data

sub = pd.DataFrame({'ID': test_final.Id, 'Cover_Type': y_pred})

sub.to_csv('submission_csv')