import numpy as np 

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

df = pd.read_csv("/kaggle/input/wineuci/Wine.csv", names = ['class','Alcohol','MalicAcid','Ash','AshAlcalinity','Magnesium','Phenol',

                                      'Flavanoid','NonFlavanoid','Proanthocyanins','ColorIntensity',

                                      'Hue','DilutedWines','Proline'])

df.head()
df.info()
# to check the if the class is balance

print(df.groupby('class').size())
from imblearn.over_sampling import SMOTE



# Resample the minority class. You can change the strategy to 'auto' if you are not sure.

sm = SMOTE(random_state=7)



# Fit the model to generate the data.

oversampled_trainX, oversampled_trainY = sm.fit_sample(df.drop('class', axis=1), df['class'])

oversampled_train = pd.concat([pd.DataFrame(oversampled_trainY), pd.DataFrame(oversampled_trainX)], axis=1)

oversampled_train.columns = df.columns

oversampled_train.info()
# to chech the class

print(oversampled_train.groupby('class').size())
def importdata():

    df = oversampled_train

    return df
def splitdataset(df):

    

    # Seperating the target variable

    X = df.values[:,1:]  

    y = df.values[:,0] 



    #Split data into training and test datasets (training will be based on 70% of data)

    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0, stratify =y)

    

    # transform data so its distribution will have a mean value 0 and standard deviation of 1

    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()

    scaler.fit(X_train)

    

    X_train = scaler.transform(X_train)

    X_test = scaler.transform(X_test)

    

    #test_size: if integer, number of examples into test dataset; if between 0.0 and 1.0, means proportion

    print('There are {} samples in the training set and {} samples in the test set'.format(X_train.shape[0], X_test.shape[0]))

    

    return X, y, X_train, X_test, y_train, y_test
from sklearn.tree import DecisionTreeClassifier,export_graphviz

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score 

from sklearn.metrics import classification_report

from IPython.display import SVG

from graphviz import Source

from IPython.display import display

from sklearn import tree
# Function to perform training with giniIndex. 

def train_using_gini(X_train, X_test, y_train): 

  

    # Creating the classifier object 

    #clf_gini = DecisionTreeClassifier(criterion = "gini", 

    #        random_state = 100,max_depth=8, min_samples_leaf=3)

    clf_gini = DecisionTreeClassifier(criterion = "gini", random_state = 100)

  

    # Performing training 

    clf_gini.fit(X_train, y_train) 

    return clf_gini
# Function to perform training with entropy. 

def tarin_using_entropy(X_train, X_test, y_train): 

  

    # Decision tree with entropy 

    clf_entropy = DecisionTreeClassifier(criterion = "entropy", random_state = 100) 

  

    # Performing training 

    clf_entropy.fit(X_train, y_train) 

    return clf_entropy
# Function to make predictions 

def prediction(X_test, clf_object): 

  

    # Predicton on test with giniIndex 

    y_pred = clf_object.predict(X_test) 

    return y_pred
# Function to show prediction values

def pred_result (df,y_test,y_pred_clf):



    df_new = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred_clf})

    df_new['result'] = np.where(df_new['Actual'] == df_new['Predicted'], 'correct', 'wrong')

    print(df_new)
# Function to calculate accuracy 

def cal_accuracy(y_test, y_pred): 

      

    print("Confusion Matrix: \n", 

    confusion_matrix(y_test, y_pred)) 

      

    print ("Accuracy : \n", 

    accuracy_score(y_test,y_pred)*100) 

      

    print("Report : \n", 

    classification_report(y_test, y_pred))
# Function to draw decision tree

def draw_dt (df,clf_object):

    

    graph = Source(tree.export_graphviz(clf_object, out_file=None

                                        , feature_names= df.iloc[:, 1:].columns, class_names=['1', '2', '3'] 

                                        , filled = True))

    display(SVG(graph.pipe(format='svg')))
# Driver code 

def main(): 

      

    # Building Phase 

    data = importdata() 

    X, y, X_train, X_test, y_train, y_test = splitdataset(data) 

    clf_gini = train_using_gini(X_train, X_test, y_train) 

    clf_entropy = tarin_using_entropy(X_train, X_test, y_train) 

      

    # Operational Phase 

    print("\n\033[1m"+"Results Using Gini Index:"+"\033[0;0m") 

      

    # Prediction using gini 

    y_pred_gini = prediction(X_test, clf_gini) 

    cal_accuracy(y_test, y_pred_gini)

    

    #Prediction result

    pred_result(data,y_test,y_pred_gini)

    

     # Draw tree

    draw_dt (data,clf_gini)

    

    print("\n\n\033[1m" + "Results Using Entropy:"+"\033[0;0m") 

    # Prediction using entropy 

    y_pred_entropy = prediction(X_test, clf_entropy) 

    cal_accuracy(y_test, y_pred_entropy)

    

    #Prediction result

    pred_result(data,y_test,y_pred_entropy)



    # Draw tree

    draw_dt (data,clf_entropy)
import sys

np.set_printoptions(threshold=sys.maxsize)

# Calling main function 

if __name__=="__main__": 

    main()