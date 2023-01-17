# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#Preprocessing 

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.preprocessing import OneHotEncoder



#Classifier 

from sklearn.model_selection import StratifiedKFold

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.model_selection import GridSearchCV



import tensorflow as tf

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense

from keras import backend as K

from sklearn.metrics import recall_score,precision_score,f1_score
#Read files

titanic_train = pd.read_csv('/kaggle/input/titanic/train.csv')
#Let's look on the train dataset

titanic_train.head()
#Check information about train dataset 

titanic_train.info()
#Check correlations

plt.figure(figsize=(10, 10))

sns.heatmap(titanic_train.corr(), vmax=1,

            square=True,annot=True,cmap='coolwarm')

plt.title('Correlations')
#Look at title on the name of the dataset - split the title from the name

def split_title(dataset):

    dataset['Title'] = dataset['Name'].str.extract('([a-zA-Z]+)\.')

    return dataset
split_title(titanic_train)
titanic_train['Title'].value_counts()
#4 groups of Title: Mr, Miss,Mrs, other titles

title_mapping = {'Mr':0,"Miss":1,'Mrs':2}

titanic_train['Title'] = titanic_train['Title'].map(title_mapping).replace(np.nan,3).astype('int')
titanic_train.dtypes
#Check missing values on train dataset - Age, Cabin, Embarked have missing values 

titanic_train.isnull().sum()
#Missing values arrangement

#For Train dataset 



#Fill Age with mean with the same title 

#Drop missing values on Embarked (2 Values)

#Drop unuse column



titanic_train['Age'].fillna(round(titanic_train.groupby('Title')["Age"].transform('mean')),inplace = True)

titanic_train = titanic_train.dropna(subset = ['Embarked'])

titanic_train = titanic_train.drop(['Name','Ticket','PassengerId','Cabin'],axis = 1)



#reset the index

titanic_train = titanic_train.reset_index()
#One hot encoding

def cvt_onehot(dataset):

    obj_columns = dataset.select_dtypes(include = [object])

    for col in obj_columns:

        onehot = OneHotEncoder(handle_unknown='ignore')

        onehot.fit(dataset[[col]])

        onehot_df = pd.DataFrame(onehot.transform(dataset[[col]]).toarray(),columns=onehot.get_feature_names())

        onehot.index = dataset.index

        dataset = dataset.drop([col],axis = 1)

        dataset = pd.concat([dataset,onehot_df],axis = 1)

    return dataset
cvt_onehot(titanic_train)
#We can build the function to preprocess model 



def preprocess(dataset):

    split_title(dataset)

    

    title_mapping = {'Mr':0,"Miss":1,'Mrs':2}

    dataset['Title'] = dataset['Title'].map(title_mapping).replace(np.nan,3).astype('int')

    

    dataset['Age'].fillna(round(dataset.groupby('Title')["Age"].transform('mean')),inplace = True)

    

    dataset = dataset.dropna(subset = ['Embarked'])

    dataset = dataset.drop(['Name','Ticket','PassengerId','Cabin'],axis = 1)

    dataset = dataset.reset_index(drop = True)

    dataset = cvt_onehot(dataset)

    return dataset
#Finding hyperparameters

titanic_train = pd.read_csv('/kaggle/input/titanic/train.csv')

titanic_train = preprocess(titanic_train)

X = titanic_train.drop(['Survived'],axis = 1).values

Y = titanic_train["Survived"].values



model_params = {

    'decision_tree':{

        'model':DecisionTreeClassifier(),

        'params':{

            'max_depth' : [10,15,20,50,500],

            'min_samples_split' : [2, 5, 10, 15, 100],

            'min_samples_leaf' : [1, 2, 5, 10]  

        }

    },

    'naive_bayes':{

        'model':GaussianNB(),

        'params':{}

    }

}





scores = []

for model_name,mp in model_params.items():

    clf = GridSearchCV(estimator=mp['model'],

                       param_grid=mp['params'],

                       cv = 5)

    clf.fit(X,Y)

    scores.append({

        'model':model_name,

        'best_score':clf.best_score_,

        'best_params':clf.best_params_

    })

pd.set_option('display.max_colwidth', None)

pd.DataFrame(scores)
# Find recall, Precision, F-Measure, Average F-Measure and Neura



def recall_m(y_true, y_pred):

    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))

    recall = true_positives / (possible_positives + K.epsilon())

    return recall



def precision_m(y_true, y_pred):

    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))

    precision = true_positives / (predicted_positives + K.epsilon())

    return precision



def f1_m(y_true, y_pred):

    precision = precision_m(y_true, y_pred)

    recall = recall_m(y_true, y_pred)

    return 2*((precision*recall)/(precision+recall+K.epsilon()))
#Neural Network model

def NN_model():

    d_in = (11,)

    model = Sequential()

    model.add(Dense(32, input_shape = d_in, activation='relu'))

    model.add(Dense(32, activation='relu'))

    model.add(Dense(32, activation='relu'))

    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss=tf.keras.losses.MeanSquaredError(), 

        optimizer='adam', metrics=['acc',f1_m,precision_m, recall_m])

    return model
#5-fold cross validation, use max_depth, min_sample_leaf, min_samples_split from hyperparameters finding 



titanic_train = pd.read_csv('/kaggle/input/titanic/train.csv')

titanic_train = preprocess(titanic_train)

X = titanic_train.drop(['Survived'],axis = 1).values

Y = titanic_train["Survived"].values



ann_score = {}

decision_tree_score = {}

naive_bayes_score = {}



folds = 5

kfold = StratifiedKFold(shuffle=True,random_state=0)



for train_index,test_index in kfold.split(X,Y):

    Xtrain, Xtest, Ytrain, Ytest = X[train_index],X[test_index],Y[train_index],Y[test_index]

    

    ann_model = NN_model()

    ann_model.fit(Xtrain,Ytrain,epochs=100,verbose = 0)

    _, accuracy_, f1_score_, precision_, recall_ = ann_model.evaluate(Xtest, Ytest, verbose=1)

    ann_score.setdefault('recall',[]).append(recall_)

    ann_score.setdefault('precision',[]).append(precision_)

    ann_score.setdefault('f1_score',[]).append(f1_score_)

    

    dt_model = DecisionTreeClassifier(max_depth=10,min_samples_leaf=5,min_samples_split=15)

    dt_model.fit(Xtrain,Ytrain)

    y_pred = dt_model.predict(Xtest)

    decision_tree_score.setdefault('recall',[]).append(recall_score(Ytest,y_pred))

    decision_tree_score.setdefault('precision',[]).append(precision_score(Ytest,y_pred))

    decision_tree_score.setdefault('f1_score',[]).append(f1_score(Ytest,y_pred))



    nb_model = GaussianNB()

    nb_model.fit(Xtrain,Ytrain)

    y_pred = nb_model.predict(Xtest)

    naive_bayes_score.setdefault('recall',[]).append(recall_score(Ytest,y_pred))

    naive_bayes_score.setdefault('precision',[]).append(precision_score(Ytest,y_pred))

    naive_bayes_score.setdefault('f1_score',[]).append(f1_score(Ytest,y_pred)) 
print('Decision Tree\n', pd.DataFrame(decision_tree_score).T.mean(axis = 1))
print('Naive Bayes\n', pd.DataFrame(naive_bayes_score).T.mean(axis = 1))
print('Neural Network\n', pd.DataFrame(ann_score).T.mean(axis = 1))
modeldts_f1 = np.mean(decision_tree_score['f1_score'])

modelnbs_f1 = np.mean(naive_bayes_score['f1_score'])

modelNN_f1 = np.mean(ann_score['f1_score'])

mean = np.mean([modeldts_f1,modelnbs_f1,modelNN_f1])



print(f"The average f Measure from classifiers {mean}")