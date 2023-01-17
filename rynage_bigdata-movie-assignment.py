import numpy as np 

#import pandas as pd 



#file import, preprocessing, cross-validation and accuracy check

from sklearn.datasets import load_svmlight_file

from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.metrics import accuracy_score

from sklearn.model_selection import GridSearchCV

from sklearn.feature_selection import SelectKBest, chi2



#three classification models that will be demostrated

from sklearn.linear_model import LogisticRegression

from sklearn.svm import LinearSVC

from sklearn.tree import DecisionTreeClassifier





#Please change the path to the location of your train/test files

train_path = '/kaggle/input/bigdatamovie/labeledBow.feat'

test_path = '/kaggle/input/bigdatamovie/test_data.feat'



#Please change the path to the desired output path of the prediction files

LR_outpath = './LR_predict.csv'

LSVC_outpath = './LSVC_predict.csv'

DTC_outpath = './DTC_predict.csv'



#this block process data into tfidf values and select 5000 best features for classification

#for simplicity, the four arrays will be named X_train, y_train, X_test, y_test and they will be used in subsequent blocks

movie_train = load_svmlight_file(train_path, dtype='int')

movie_matrix = movie_train[0]

#True for positive reviews and False for negative

y_train = np.array([x > 5 for x in movie_train[1]])

movie_test = load_svmlight_file(test_path, dtype='int', n_features=89527)

test_matrix = movie_test[0]

y_test = np.array([x > 5 for x in movie_train[1]])





tfid = TfidfTransformer()

movie_matrix_tfid = tfid.fit_transform(movie_matrix)

test_matrix_tfid = tfid.transform(test_matrix)



#to enhance speed of the program, 5000 features are chosen based on chi-square

#not necessarily best for prediction accuracy

KBest = SelectKBest(chi2, 5000)

X_train = KBest.fit_transform(movie_matrix_tfid, y_train)

X_test = KBest.transform(test_matrix_tfid)
#prepare the function to print prediction results to csv file

def write_prediction(prediction, outpath):

    with open(outpath, 'w') as fid:

        for i in range(len(prediction)):

            fid.write(str(i))

            if prediction[i]:

                fid.write(', Positive\n')

            else:

                fid.write(', Negative\n')

    
#search for best C parameter for logistic regression, with 5-fold cross validation. max_iter is increased to ensure convergence

#define a function for repeat searching

def find_bestLR(grid):

    param_grid = {'C':grid}

    LR = LogisticRegression(max_iter=2000)

    LR_cv = GridSearchCV(LR, param_grid, cv=5)

    LR_cv.fit(X_train, y_train)

    print('The best C value is', LR_cv.best_params_['C'])

    print('Training time is', LR_cv.refit_time_, 'seconds')

    return LR_cv.best_params_['C']
#logspace is a good first step for finding likely parameter in a wide range

#dummy variable to mute return value

c_logspace = np.logspace(-3, 3, 10)

_ = find_bestLR(c_logspace)
#the best C value is 2.1544, let's try to narrow down the search

c_linspace = np.linspace(1.5, 2.5, 20)

_ = find_bestLR(c_linspace)
#the best value of C is the upper bound (2.5), maybe try shifting the range up

c_linspace = np.linspace(2.5, 3.5, 20)

_ = find_bestLR(c_linspace)
#the value is still too close to the upper bound for comfort, let's try to shift up a bit more

c_linspace = np.linspace(3.4, 4.5, 20)

_ = find_bestLR(c_linspace)
#now we have decide the C value, we can perform the training and prediction

LR_best = LogisticRegression(max_iter=2000, C=4.210526)

LR_best.fit(X_train, y_train)

print('The training accuracy is:', accuracy_score(LR_best.predict(X_train), y_train))

print('The testing accuracy is:', accuracy_score(LR_best.predict(X_test), y_test))

write_prediction(LR_best.predict(X_test), LR_outpath)

#similar to what we have done, but this time we use LinearSVC. max_iter is again increased to ensure convergence

def find_bestLSVC(grid):

    param_grid = {'C':grid}

    LSVC = LinearSVC(max_iter=10000)

    LSVC_cv = GridSearchCV(LSVC, param_grid, cv=5)

    LSVC_cv.fit(X_train, y_train)

    print('The best C value is', LSVC_cv.best_params_['C'])

    print('Training time is', LSVC_cv.refit_time_, 'seconds')

    return LSVC_cv.best_params_['C']
#overly large value of C will give rise to convergence problem, so this time the upper bound is lower

c_logspace = np.logspace(-3, 2, 10)

_ = find_bestLSVC(c_logspace)
#from last search C=0.59948 is the best. Again let us try to narrow down the search

c_linspace = np.linspace(0.3, 0.9, 20)

_ = find_bestLSVC(c_linspace)
#the last value was the lower bound (C=0.3), so let's shift the range downward

c_linspace = np.linspace(0.1, 0.4, 20)

_ = find_bestLSVC(c_linspace)
LSVC_best = LinearSVC(C=0.28947, max_iter = 10000)

LSVC_best.fit(X_train, y_train)

print('The training accuracy is:', accuracy_score(LSVC_best.predict(X_train), y_train))

print('The testing accuracy is:', accuracy_score(LSVC_best.predict(X_test), y_test))

write_prediction(LSVC_best.predict(X_test), LSVC_outpath)
#for decision tree, we should first run without tuning, to learn about its maximum depth

#random state is fixed to reproduce result

DTC = DecisionTreeClassifier(random_state = 31)

DTC.fit(X_train,y_train)

print('The training accuracy is:',accuracy_score(DTC.predict(X_train), y_train))

print('The testing accuracy is:', accuracy_score(DTC.predict(X_test), y_test))

print('Maximum depth is:', DTC.get_depth())
#the model suffers from overfitting

#again we try to find optimal max_depth parameter for the model

#random_state is fixed to reproduce result

def find_bestDTC(grid):

    param_grid = {'max_depth':grid}

    DTC = DecisionTreeClassifier(random_state = 31)

    DTC_cv = GridSearchCV(DTC, param_grid, cv=5)

    DTC_cv.fit(X_train, y_train)

    print('The best max_depth value is', DTC_cv.best_params_['max_depth'])

    print('Training time is', DTC_cv.refit_time_, 'seconds')

    return DTC_cv.best_params_['max_depth']
#first search

_ = find_bestDTC([10,20,30,40,50])
#again narrow down the search

_ = find_bestDTC(range(1,20))
#perform the prediction using max_depth = 17

DTC_best = DecisionTreeClassifier(max_depth = 17, random_state = 31)

DTC_best.fit(X_train, y_train)

print('The training accuracy is:', accuracy_score(DTC_best.predict(X_train), y_train))

print('The testing accuracy is:', accuracy_score(DTC_best.predict(X_test), y_test))

write_prediction(DTC_best.predict(X_test), DTC_outpath)