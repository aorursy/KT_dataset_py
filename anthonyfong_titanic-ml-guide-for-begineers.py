'''IMPORTING MODULES/LOADING/UNLOADING DATA'''

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

train = pd.read_csv('../input/titanic/train.csv')
test = pd.read_csv('../input/titanic/test.csv')

'''DATA CLEANSING/DATA EXPLORATION'''

print(train.Embarked.value_counts())
ix  = train.Embarked.isnull().nonzero()
train.loc[ix[0],"Embarked"] = 'S' #since the number of S is the most we will fill the NaN values with S

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Imputer,LabelBinarizer, StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin



#function deletes column according to what you feed in
class data_cleanse(BaseEstimator,TransformerMixin):
    def __init__(self,string_param='None'):
        self.string_param = string_param
    def fit(self,X,y=None):
        try:
            type(self.string_param) == list
        except TypeError:
            print('Please input the string_params as a list')
        return self
    def transform(self,X):
        X = X.drop(self.string_param,axis=1)
        return X

#this code does a one hot encoding and adds into the training data
def one_hot_into_df (df,column_name=None):

    def zero_to_one_array(array):
        intermediate= np.equal(array, np.zeros(len(df_encode)).reshape(-1,1))
        return 1*intermediate

    #from IPython.core.debugger import set_trace
    #set_trace()
    if type(df) != pd.DataFrame:
        return 'Only datatypes of dataframe is allowed'
    if type(column_name) != str:
        return "only column_name of type str is allowed"
    
    from sklearn.preprocessing import LabelBinarizer 
    encoder = LabelBinarizer()
    df_encode = encoder.fit_transform(df[column_name])
    uniq_col= sorted(df[column_name].unique())
    num_uniq_col = len(uniq_col)
    
    #add the binarized into he training data
    if num_uniq_col>2:
        for i in range(num_uniq_col):
            df[uniq_col[i]]= df_encode[:,i]

    elif num_uniq_col==2:
        df[uniq_col[0]]=zero_to_one_array(df_encode) 
        df[uniq_col[1]]=df_encode
        
        
    '''
    Embarked_encode = pd.DataFrame(df)
    for x in [uniq_col]:
        df[x] = Embarked_encode[x]'''
        
    df= df.drop([column_name],axis=1)
    return df

class one_hot_encode(BaseEstimator,TransformerMixin):
    def __init__(self,column_name=None):
        self.column_name = column_name
    def fit(self,X,y=None):
        try:
            type(self.column_name) == str
        except TypeError:
            print('Please input the column name as a str')
        return self
    def transform(self,X):
        for term in self.column_name:
            X=one_hot_into_df (X,term)
        return X

estimator = Pipeline([('data_cleanse',data_cleanse(['Cabin','Ticket','Name'])),
    ('one_hot_encode',one_hot_encode(['Embarked','Sex'])),
    ('imputer',Imputer(strategy="median")) #imputer returns a numpy array
                     ])

result = estimator.fit_transform(train)

#imput the headings into the numpy array(imputer function converts it into an numpy array)
headings = train.columns.values
headings = np.delete(headings,[3,4,8,10,11],0)
headings = np.append(headings,["C",'Q','S',"female",'male'])
result_df =pd.DataFrame(result)
result_df.columns = headings

'''
from sklearn.preprocessing import StandardScaler
for x in ['Age','Fare']:
    #import pdb;pdb.set_trace()
    scaled = StandardScaler().fit_transform(result_df[x].reshape(-1,1))
    result_df[x] = scaled
'''

result_df.drop('PassengerId',axis=1)
train.Survived.value_counts()
#the average survival rate is 0.3838 , any combination of results which produce a higher 
#survival rate than that is worth looking at

import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(1)
plt.subplot(1,2,1)
sns.set_style("whitegrid")
g_1 = sns.countplot(x=train.Pclass,hue=train.Survived,palette="Set3")

plt.subplot(1,2,2)
g_2 = sns.countplot(x=train.Parch,hue=train.Survived,palette="Set3")


#assuming rich female woman will survive etc,
combinator = {"Pclass":[1],'Sex':['female']}
ric_fem = train.isin(combinator).sum(axis=1)
ix = (ric_fem == 2).nonzero()
train.iloc[ix]

result_df.drop("Survived",axis=1)
#splitting the train dataset into train and cv
from sklearn.model_selection import train_test_split
x_df = result_df.drop('Survived',axis=1)
y_df = result_df.Survived

#x_train, x_cv , y_train , y_cv = train_test_split(x,y,test_size=0.3)



from sklearn import linear_model
from sklearn.model_selection import GridSearchCV
from time import time
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform as sp_rand
import scipy.stats as sp
# Utility function to report best scores
def report(results, n_top=20):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")
logistic = linear_model.LogisticRegression(penalty = 'l2')

#the hyper-parameters include the number of features,  type of regularization 
#and the alpha parameter of the regularizer
param_grid= {'penalty':['l1','l2'],'C':sp_rand()}

num_iter = 2000
rand_search_cv = RandomizedSearchCV(logistic,param_distributions = param_grid
                                    ,n_iter = num_iter)

start = time()
rand_search_cv.fit(x_df, y_df)
print("RandomizedSearchCV took %.2f seconds for %d candidates"
      " parameter settings." % ((time() - start), num_iter))
report(rand_search_cv.cv_results_)

logistic.set_params(**rand_search_cv.best_params_)
#note the ** means to unpack all the iterations in the named argument 
#(either in dictionary form or in named pair eg "c" = 0.1231)
#note the * means to unpack all the positional arguments
#applying data cleansing on the test data too
test_copy = estimator.fit_transform(test)

temp = list(headings)
temp.remove('Survived')
headings = np.asarray(temp)

test_copy =pd.DataFrame(test_copy)
test_copy.columns = headings

'''
for x in ['Age','Fare']:
    #import pdb;pdb.set_trace()
    scaled = StandardScaler().fit_transform(test_copy[x].values.reshape(-1,1))
    test_copy[x] = scaled
'''
test_copy.drop('PassengerId',axis=1)
#submitting the results
y_pred = rand_search_cv.predict(test_copy).astype(int)


submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": y_pred
    })
submission.to_csv('titaniclogistic.csv', index=False)
#we will now train a SVM
from sklearn import svm
svm_model = svm.LinearSVC(dual=False)
param_grid_svm= {'penalty':['l1','l2'],'C':sp_rand()}

num_iter_svm = 2000
rand_search_svm = RandomizedSearchCV(svm_model,param_distributions = param_grid_svm
                                    ,n_iter = num_iter_svm)
start = time()
rand_search_svm.fit(x_df, y_df)
print("RandomizedSearchCV took %.2f seconds for %d candidates"
      " parameter settings." % ((time() - start), num_iter_svm))
report(rand_search_svm.cv_results_)

svm_model.set_params(**rand_search_svm.best_params_)

#submitting the results
y_pred = rand_search_svm.predict(test_copy).astype(int)
submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": y_pred
    })
submission.to_csv('titanicsvm.csv', index=False)


#we will now train with a random tree forest algorithm
from sklearn.ensemble import RandomForestClassifier
params_grid = {'n_estimators':sp.randint(1,200),
              'criterion':['gini','entropy'],
               'max_depth':sp.randint(1,8),
               'min_samples_leaf':sp.randint(10,50),
               'max_features':np.arange(0.1,0.7,0.1)
               }
num_iterations = 2000
ran_forest_mnist = RandomForestClassifier(random_state = 42, verbose =1,n_jobs=-1)
ran_forest_mnist_cv = RandomizedSearchCV(ran_forest_mnist,param_distributions=params_grid,
                                         n_iter=num_iterations,verbose=1,n_jobs=-1)
start = time()
ran_forest_mnist_cv.fit(x_df,y_df)
print("RandomizedSearchCV took %.2f seconds for %d candidates"
      " parameter settings." % ((time() - start), num_iterations))

report(ran_forest_mnist_cv.cv_results_)

ran_forest_mnist.set_params(**ran_forest_mnist_cv.best_params_)

ran_forest_mnist.fit(x_df,y_df)

ran_forest_mnist.score(x_df,y_df)
from sklearn.model_selection import learning_curve
start = time()
train_size,train_score,cv_score = learning_curve(ran_forest_mnist,x_df,y_df,
                                                 train_sizes=np.linspace(0.2,1,num=21,dtype= float),
                                                 cv = 10)
#the train_score,cv_score are all arra of number of different train size x number of cv folds
print('The process took {0}seconds'.format(time()-start))

train_score_mean , cv_score_mean = np.mean(train_score,axis=1) , np.mean(cv_score,axis=1)

plt.plot(train_size,train_score_mean,label='Training score')

plt.plot(train_size,cv_score_mean,label="Cross validation score")
plt.xlabel('number of training examples')
plt.ylabel('Accuracy score')
plt.title("Learning curve")
plt.legend()
#submitting the results
y_pred_ran_forest = ran_forest_mnist.predict(test_copy).astype(int)
submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": y_pred_ran_forest
    })
submission.to_csv('titanicran1.csv', index=False)
