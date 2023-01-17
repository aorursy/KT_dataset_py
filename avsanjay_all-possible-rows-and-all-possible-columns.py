# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.



train = pd.read_csv('input/train.csv')

test = pd.read_csv('input/test.csv')



#subset out datasets to an intial train and test by input and output variables

train_y = train['y']

train_x = train.drop(['y'], axis=1)

test_x = test





train_x['Source'] = 'train'

test_x['Source'] = 'test'

traintest_x = train_x.append(test_x)

traintest_x = pd.get_dummies(traintest_x) 

#Split back data into train_x and test_x now that multicategory split is done

train_x = traintest_x[traintest_x['Source_train']==1]

del train_x['Source_train']

del train_x['Source_test']

test_x = traintest_x[traintest_x['Source_test']==1]

del test_x['Source_train']

del test_x['Source_test']





# Start to make pipeline

from sklearn.pipeline import make_pipeline

# Import Elastic Net, Ridge Regression, and Lasso Regression

from sklearn.linear_model import ElasticNet, Ridge, Lasso



# Import Random Forest and Gradient Boosted Trees

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

pipelines = {

    'lasso' : make_pipeline(Lasso(random_state=123)),

   

}





# Set hyperparameters for each model that will use to fit each model.  This will allow for finding an optimial model



# Lasso hyperparameters 

lasso_hyperparameters = { 

    'lasso__alpha' : [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10] 

}



# Create hyperparameters dictionary

hyperparameters = {

    

    'lasso' : lasso_hyperparameters,

   

}



from sklearn.metrics import r2_score

from sklearn.model_selection import GridSearchCV



# Create empty dictionary called fitted_models

fitted_models = {}



#split the train set into a train and val dataset since the test dataset doesn't have label.  start with test size of 0.2

from sklearn.cross_validation import train_test_split

X_train, X_val, y_train, y_val = train_test_split(train_x, train_y, test_size=0.2, random_state=42)



len = X_train.shape[1]

row = X_train.shape[0]



for x in range(len):

    X_train2 = X_train.iloc[:,0:x]

    X_val2 = X_val.iloc[:,0:x]

    

    for name, pipeline in pipelines.items():

    # Create cross-validation object from pipeline and hyperparameters

        model = GridSearchCV(pipeline, hyperparameters[name], cv=10, n_jobs=-1)

    

        # Fit model on X_train, y_train

        model.fit(X_train2, y_train)

    

        # Store model in fitted_models[name] 

        #fitted_models[name] = model

        pred = model.predict(X_val2)

    

        # Print '{name} has been fitted'

        print("x value is ", x)

        #print(name, 'has been fitted.')

        print( 'R^2:', r2_score(y_val, pred ))





X_train2 = X_train.iloc[:,139:554]

X_val2 = X_val.iloc[:, 139:554]



bestR2Value = -100







for a in range(5, row,500):

   

    for y in range( a + 15,row):

        if y > a:

            X_train2 = X_train.iloc[a:y,139:554]

            X_val2 = X_val.iloc[a:y,139:554]

            y_train2 = y_train.iloc[a:y]

            y_val2 = y_val.iloc[a:y]

    

            for name, pipeline in pipelines.items():

    # Create cross-validation object from pipeline and hyperparameters

                model = GridSearchCV(pipeline, hyperparameters[name], cv=10, n_jobs=-1)

    

        # Fit model on X_train, y_train

                model.fit(X_train2, y_train2)

    

        # Store model in fitted_models[name] 

        #fitted_models[name] = model

                pred = model.predict(X_val2)

                

                r2 = r2_score(y_val2, pred )

    

        # Print '{name} has been fitted'

                print("a value is",a)

                print("y value is ", y)

        #print(name, 'has been fitted.')

                print( 'R^2:', r2)

                        

                if r2 > bestR2Value:

                    bestR2Value = r2

                    a1 = a

                    y1 = y

                    print("bestR2Value is", bestR2Value)

                    print("a value is", a)

                    print("y value is", y)



print("bestR2Value is", bestR2Value)

print("a1 value is", a1)

print("y1 value is", y1)
