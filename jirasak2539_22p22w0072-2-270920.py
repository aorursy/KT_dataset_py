import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt
titanic = pd.read_csv('../input/titanic/train.csv')
titanic.head()
titanic.info()
titanic.describe(include = 'all').T
titanic.isnull().sum()
ncols = 5

nrows = 1

cols = ['Pclass','Sex','SibSp','Parch','Embarked']

fig, axs = plt.subplots(nrows, ncols, figsize = (3.2 * ncols, 3.2 * nrows))

for r in range(nrows):

    for c in range(ncols):

        i = r * ncols + c

        ax = axs[c]

        sns.countplot(titanic[cols[i]],hue = titanic['Survived'],ax = ax)

        ax.set_title(cols[i])

plt.tight_layout()
titanic.head()
# Extract title from person's name

def extract_title(dataset):

    dataset['Title'] = dataset["Name"].str.extract('([a-zA-Z]+)\.')

    return dataset



extract_title(titanic)
titanic['Title'].value_counts()
# The title will be divided into 4 main groups

# - 'Mr' : 0

# - 'Miss' : 1

# - 'Mrs' : 2

# - Others : 3

title_mapping = {'Mr':0,"Miss":1,'Mrs':2}

titanic['Title'] = titanic['Title'].map(title_mapping).replace(np.nan,3).astype('int')
# fill the missing age values with the rounded mean of the person with the same 'Title'

titanic['Age'].fillna(round(titanic.groupby('Title')["Age"].transform('mean')),inplace = True)
print(titanic.isnull().sum())

print(titanic.isnull().mean())
# Deal with missing values



# The column "Cabin" will be dropped since it contain more than 70 % of missing values

titanic = titanic.drop(['Cabin'],axis = 1)

# The row with 2 missing value from column 'Embarked' will also be removed

titanic = titanic.dropna(subset = ['Embarked'])



# Drop the unwanted column

titanic = titanic.drop(['Name','Ticket','PassengerId'],axis = 1)



#reset the index

titanic = titanic.reset_index()
print(titanic.isnull().sum())

print(titanic.isnull().mean())

# no more missing value
titanic.dtypes
# Convert the "Sex" and "Embarked" column into categorical type



def convert_to_onehot(dataset):

    from sklearn.preprocessing import OneHotEncoder

    

    obj_cols = dataset.select_dtypes(include = [object])

    for col in obj_cols:

        ohe = OneHotEncoder(handle_unknown='ignore')

        ohe.fit(dataset[[col]])

        ohe_df = pd.DataFrame(ohe.transform(dataset[[col]]).toarray(),columns=ohe.get_feature_names())

        ohe.index = dataset.index

        dataset = dataset.drop([col],axis = 1)

        dataset = pd.concat([dataset,ohe_df],axis = 1)

    return dataset
convert_to_onehot(titanic)
# overall, we can write a main function to do all the steps above, make it more repeatable 

def preprocessing(dataset):

    extract_title(dataset)

    

    title_mapping = {'Mr':0,"Miss":1,'Mrs':2}

    dataset['Title'] = dataset['Title'].map(title_mapping).replace(np.nan,3).astype('int')

    

    dataset['Age'].fillna(round(dataset.groupby('Title')["Age"].transform('mean')),inplace = True)

    

    dataset = dataset.drop(['Cabin'],axis = 1)

    dataset = dataset.dropna(subset = ['Embarked'])

    dataset = dataset.drop(['Name','Ticket','PassengerId'],axis = 1)

    dataset = dataset.reset_index(drop = True)

    dataset = convert_to_onehot(dataset)

    return dataset
from sklearn.model_selection import StratifiedKFold

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.model_selection import GridSearchCV



import tensorflow as tf

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense
# hyperparameters seaching

titanic = pd.read_csv('../input/titanic/train.csv')

titanic = preprocessing(titanic)

X = titanic.drop(['Survived'],axis = 1).values

Y = titanic["Survived"].values



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
# according to our finding, we will use 

# max_depth = 10

# min_samples_leaf = 5

# min_samples_split = 15

# for the decision tree model
# Define evaluation score to use in the keras's neural network model

from keras import backend as K



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



# define the neural network model



def ANN_model():

    d_in = (11,)

    model = Sequential()

    model.add(Dense(32, input_shape = d_in, activation='relu'))

    model.add(Dense(32, activation='relu'))

    model.add(Dense(32, activation='relu'))

    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss=tf.keras.losses.MeanSquaredError(), 

        optimizer='adam', metrics=['acc',f1_m,precision_m, recall_m])

    return model


from sklearn.metrics import recall_score,precision_score,f1_score

titanic = pd.read_csv('../input/titanic/train.csv')

titanic = preprocessing(titanic)

X = titanic.drop(['Survived'],axis = 1).values

Y = titanic["Survived"].values



ann_score = {}

decision_tree_score = {}

naive_bayes_score = {}



folds = 5

kfold = StratifiedKFold(shuffle=True,random_state=0)



for train_index,test_index in kfold.split(X,Y):

    Xtrain, Xtest, Ytrain, Ytest = X[train_index],X[test_index],Y[train_index],Y[test_index]

    

    ann_model = ANN_model()

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
print("evaluation score from neural network model")

print(pd.DataFrame(ann_score).T.mean(axis = 1))
print("evaluation score from decision tree model")

print(pd.DataFrame(decision_tree_score).T.mean(axis = 1))
print("evaluation score from naive bayes model")

print(pd.DataFrame(naive_bayes_score).T.mean(axis = 1))
model1_f1 = np.mean(ann_score['f1_score'])

model2_f1 = np.mean(decision_tree_score['f1_score'])

model3_f1 = np.mean(naive_bayes_score['f1_score'])

mean = np.mean([model1_f1,model2_f1,model3_f1])



print(f"The average f1 score from all three models is {mean}")