# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import subprocess

from scipy.sparse import csr_matrix, hstack

from sklearn.metrics import mean_absolute_error

from sklearn.preprocessing import StandardScaler

#from sklearn.cross_validation import KFold

from sklearn import preprocessing

from sklearn.ensemble import RandomForestRegressor

import time



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

y = train['loss']

shift = 200

log_y = np.log(y+shift)

print(np.mean(y))

print(np.median(y))
def preprocess(df):

    

    #Assume categorical/continuous variables named as cati/contj

    #Assume no missing data, etc. 

    cat_headers = [ x for x in df.columns.values if "cat" in x] ; #print cat_headers

    cont_headers = [ x for x in df.columns.values if "cont" in x] ; #print cont_headers

    df3 = df[cat_headers].apply(preprocessing.LabelEncoder().fit_transform) 

    #print(df[['cat1','cat2','cont1']].head(5))

    return pd.concat([df[cont_headers],df3],axis=1)



train = preprocess(train)

x = train.values
def scorer(x,y):

    return mean_absolute_error(np.exp(x),np.exp(y))



from sklearn.metrics import fbeta_score, make_scorer

custom_scorer = make_scorer(scorer,greater_is_better=False)
from sklearn import tree

from sklearn.linear_model import LinearRegression

from sklearn.ensemble import RandomForestRegressor

from sklearn.linear_model import Lasso

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.model_selection import cross_val_score

from numpy.random import RandomState

prng = RandomState(1234567890)



cv_tree = False

cv_rf = True



if cv_tree:

    for max in [None,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]:

        t0 = time.time()

        reg = tree.DecisionTreeRegressor(max_depth=max)

        scores = cross_val_score(reg, x,log_y, cv=3,scoring=custom_scorer)

        scores10 = cross_val_score(reg, x,log_y, cv=10,scoring=custom_scorer)

        d = time.time()-t0

        print("tree:",max,abs(scores.mean()),abs(scores10.mean()),d)



if cv_rf:

    t0 = time.time()

    reg = RandomForestRegressor(n_estimators = 100,criterion='mse',random_state=prng,n_jobs=-1)

    scores = cross_val_score(reg, x,log_y, cv=3,scoring=custom_scorer)

    #scores10 = cross_val_score(reg, x,log_y, cv=10,scoring=custom_scorer)

    d = time.time()-t0

    #print("rf:",max,abs(scores.mean()),abs(scores10.mean()),d)

    print("rf:",max,abs(scores.mean()),d)
from numpy.random import RandomState

prng = RandomState(1234567890)



forest = RandomForestRegressor(n_estimators = 100,criterion='mse',random_state=prng,n_jobs=-1)

forest = GradientBoostingRegressor(learning_rate=0.4,criterion='mse')

regressors = [ GradientBoostingRegressor(learning_rate=rate,criterion='mse') for rate in [0.1,0.2,0.3,0.4,0.5]]

find_features = False
if find_features:

    t0 = time.time()

    forest = forest.fit(x,log_y)

    d = time.time()-t0 ; print("Time to fit",d)

    feature_importances =  sorted(zip(forest.feature_importances_,train.columns.values),reverse=True)

    sorted_features = [ feature for score,feature in feature_importances] ; print(sorted_features)

else:

    sorted_features = ['cat80', 'cont14', 'cat101', 'cont7', 'cont2', 'cat103', 'cat79', 'cat100', 'cat111', 'cat112', 'cont8', 'cat12', 'cont5', 'cat53', 'cont3', 'cont6', 'cont4', 'cont1', 'cat110', 'cont13', 'cat81', 'cont12', 'cont11', 'cont10', 'cont9', 'cat114', 'cat1', 'cat113', 'cat57', 'cat116', 'cat72', 'cat83', 'cat107', 'cat82', 'cat91', 'cat105', 'cat115', 'cat93', 'cat106', 'cat73', 'cat84', 'cat108', 'cat109', 'cat92', 'cat87', 'cat94', 'cat4', 'cat75', 'cat97', 'cat31', 'cat39', 'cat104', 'cat95', 'cat5', 'cat50', 'cat6', 'cat99', 'cat36', 'cat2', 'cat27', 'cat37', 'cat38', 'cat44', 'cat102', 'cat23', 'cat26', 'cat49', 'cat9', 'cat96', 'cat25', 'cat52', 'cat76', 'cat98', 'cat66', 'cat77', 'cat13', 'cat10', 'cat11', 'cat90', 'cat74', 'cat40', 'cat3', 'cat41', 'cat54', 'cat8', 'cat28', 'cat71', 'cat19', 'cat24', 'cat45', 'cat29', 'cat88', 'cat43', 'cat16', 'cat65', 'cat86', 'cat89', 'cat85', 'cat7', 'cat30', 'cat78', 'cat51', 'cat18', 'cat17', 'cat14', 'cat67', 'cat42', 'cat46', 'cat59', 'cat61', 'cat32', 'cat33', 'cat21', 'cat68', 'cat47', 'cat34', 'cat60', 'cat63', 'cat35', 'cat22', 'cat48', 'cat58', 'cat56', 'cat69', 'cat20', 'cat70', 'cat55', 'cat15', 'cat62', 'cat64']

    print(sorted_features)
tune_params = False

if tune_params:

    from sklearn.model_selection import train_test_split

    from sklearn.metrics import mean_absolute_error



    tuned_parameters = {'n_features':[60,65,70,75,80,85,90,95,100,105,110]}

    tuned_parameters = {'n_features':[80]}



    for key in tuned_parameters.keys():

        for value in tuned_parameters[key]:

            t0 = time.time()

            features = sorted_features[0:value]

            x_ = train[features].values

            x_train, x_test, log_y_train, log_y_test = train_test_split(x_, log_y, test_size=0.5, random_state=prng)

            forest.fit(x_train,log_y_train)

            log_y_predict = forest.predict(x_test)

            y_predict = np.exp(log_y_predict)-shift

            y_true = np.exp(log_y_test)-shift

            score = mean_absolute_error(y_predict,y_true)

            d = (time.time()-t0)/60.0 

            print(key,value,score,d)



#n_features 80 1212...
tune_shift = False

if tune_shift:

    from sklearn.model_selection import train_test_split

    from sklearn.metrics import mean_absolute_error

    

    shifts = [200]

    features = sorted_features[0:80]

    

    for shift in shifts:

        t0 = time.time()

        x_ = train[features].values

        log_y = np.log(y+shift)

        x_train, x_test, log_y_train, log_y_test = train_test_split(x_, log_y, test_size=0.33, random_state=prng)

        forest.fit(x_train,log_y_train)

        log_y_predict = forest.predict(x_test)

        y_predict = np.exp(log_y_predict)-shift

        y_true = np.exp(log_y_test)-shift

        score = mean_absolute_error(y_predict,y_true)

        d = (time.time()-t0)/60.0 

        print(shift,score,d)
find_regressor = False

if find_regressor:

    from sklearn.model_selection import train_test_split

    from sklearn.metrics import mean_absolute_error

   

    for reg in regressors:

        t0 = time.time()

        x_ = train[features].values

        log_y = np.log(y+shift)

        x_train, x_test, log_y_train, log_y_test = train_test_split(x_, log_y, test_size=0.33, random_state=prng)

        reg.fit(x_train,log_y_train)

        log_y_predict = reg.predict(x_test)

        y_predict = np.exp(log_y_predict)-shift

        y_true = np.exp(log_y_test)-shift

        score = mean_absolute_error(y_predict,y_true)

        d = (time.time()-t0)/60.0 

        print(shift,score,d)
predict = False

if predict:

    top_features = sorted_features[0:80]

    x2 = train[top_features].values

    t0 = time.time()

    forest = forest.fit(x2,log_y)

    d = time.time()-t0 ; print("Time to fit",d/60.0)

    xx = preprocess(test)[top_features].values

    log_yy = forest.predict(xx)

    yy = np.exp(log_yy)-shift
publish = False

if publish:

    sub_name = 'random_forest.csv'

    sub = pd.DataFrame()

    sub['id'] = test['id']

    #sub['loss'] = [ int(res) for res in yy]

    sub['loss'] = yy

    sub.to_csv(sub_name, index=False)

    print(sub.head(5))

    sub_example = pd.read_csv('../input/sample_submission.csv')

    print(sub_example.head(5))