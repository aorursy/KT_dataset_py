import pandas as pd                   #for data handling (espically with data_frames)
from sklearn import preprocessing     #for labeling cuisines (apply numerical labels or ids on cuisines )
from sklearn.feature_extraction.text import TfidfVectorizer #to convert text data to numericals 
                                                             #without losing any property or parameter

from sklearn.model_selection import train_test_split  #for spilitting data set into test data set and train data set 
from sklearn.metrics import accuracy_score, confusion_matrix  #for confusion matrix and accuracy score
from sklearn.svm import SVC
from sklearn import svm #our training algorithm 
from sklearn.model_selection import GridSearchCV # optional(for hyper parameter tuning)
import os
print(os.listdir("../input"))
print("done")
#retrieving data from .json files
df_train = pd.read_json("../input/train.json")
df_test = pd.read_json("../input/test.json")
df_train
df_train['ingredients'] = df_train['ingredients'].apply(','.join) # this converts ingredients columns into arrays
df_test['ingredients'] = df_test['ingredients'].apply(','.join)   # for test data, above one is for train data
X_train = df_train['ingredients']  # assigning a new variable for ingredients array(for train data)
X_test = df_test['ingredients']    # assigning a new variable for ingredients array(for test data)
print("done")
print(X_train)
#all our parameters(i.e cuisine,ingredients) are in text format
#most of the machine learning algorithms cannot handle text data forms
#so we have to convert the text data into some numerical data without losing its quality 

encoder = preprocessing.LabelEncoder() #encoder for our y-values i.e cuisine
y_train_transformed = encoder.fit_transform(df_train['cuisine'])#this labels each y_value(which is in text) to a number 

#as our X-values i.e ingredients are grouped as array for each cuisine 
#we need to vectorize or simply assign a set of numerical value to each element in array without losing its quality.
#one of the effective tool for this is TfidVectorizer

vec = TfidfVectorizer(binary = True).fit(X_train.values) #assign our vectoriser 
X_train_transformed = vec.transform(X_train.values) #applying vectorizer for x-values of train set 
X_test_transformed = vec.transform(X_test.values)   #applying vectorizer for x-values of test set 
print("done")
#spiliting test-data set into further test-set and train-set
#In my view This spiliting process is for calculating accuracy_score 
X_for_train, X_for_test, y_for_train, y_for_test = train_test_split(X_train_transformed, y_train_transformed ,test_size= 0.25, random_state = 0)
#best set algorithm for current problem 
clf = svm.LinearSVC(C=0.5, max_iter=100, random_state=20, tol=0.5) #these are hyper parameters best suitable for this problem 
                                                                #I will post the code at the end for hyper parameter tuning

clf.fit(X_for_train, y_for_train) #training our train-set
y_pred = clf.predict(X_for_test)  #predicting our test-set
print("done")

#accuracy calculation
accuracy = accuracy_score(y_for_test, y_pred)
print('accuracy_score = ', accuracy)
#predicting values for test set
y_pred = clf.predict(X_test_transformed)
y_pred_transformed = encoder.inverse_transform(y_pred) #result will be encoded by label encoder, so it should be 
                                                    #decoded to view or pass into dataframe 

predictions = pd.DataFrame({'id': df_test['id'], 'cuisine': y_pred_transformed}) #constructing a data frame with ids
                                                                                   #and predictions as columns

predictions.to_csv('submit.csv', index = False)
print('done')
#for hyper parameter tuning 
#using grid search
random_state = []
for i in range(1, 110, 10):
    random_state.append(i)
     
param_grid = {'max_iter': [10, 100, 1000],
               'random_state': random_state, 'tol':[0.01, 0.1, 0.5 ], 'C':[0.5, 1, 1.5]}

# all the numbers in param_grid are for an optimal parameter tuning 
# tested with many possibilities and provided the best amongest them 
#note: these params vary from problem to problem. These are some of best suitable for this problem 
           
optimal_clf = svm.LinearSVC()
param = optimal_clf.get_params().keys()

grid_search = GridSearchCV(optimal_clf, param_grid)
grid_search.fit(X_for_train, y_for_train)
print(grid_search.best_params_)

better_model = grid_search.best_estimator_
better_pred = better_model.predict(X_for_test)
better_accuracy = accuracy_score(y_for_test, better_pred)
print(better_accuracy)