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
#Importing libraries



from sklearn.feature_extraction.text import TfidfVectorizer

import string

import pickle

import re

import matplotlib.pyplot as plt

%matplotlib inline



#NTLK Libraries - Text Pre-processing

import nltk

from nltk.tokenize import word_tokenize

import string

from nltk.corpus import stopwords

from nltk.stem.porter import PorterStemmer



#Model Libraries

from sklearn.linear_model import LogisticRegression, SGDClassifier

from sklearn.model_selection import StratifiedKFold

from sklearn.base import clone

import xgboost

from sklearn.linear_model import SGDClassifier

from sklearn.cluster import MiniBatchKMeans

from sklearn.ensemble import RandomForestClassifier

from sklearn.pipeline import Pipeline



#Model Perfomance

from sklearn.metrics import precision_recall_curve, confusion_matrix, precision_score, recall_score, f1_score, roc_curve
#Reading the dataset

def get_data():    

    df_data = pd.read_csv('/kaggle/input/sentiment140/training.1600000.processed.noemoticon.csv', 

                          encoding='latin-1',

                          names=['Sentiment','ID', 'Timestamp','Query', 'TwitterID','Tweet'])

    return df_data
#Vectorizing the text



tfidf = TfidfVectorizer(sublinear_tf = True,

                        min_df=5,

                        analyzer='word',

                        norm= 'l2',

                        stop_words = 'english',

                        encoding= 'latin-1',

                        strip_accents='unicode',

                        ngram_range =(1,1), 

                        max_features=50000,

                        decode_error='strict',

                        lowercase='True'

                       )
#One-Hot encode

df_data = get_data()

dummy_data =  pd.get_dummies(df_data['Sentiment'], prefix='Sent')

df_data = pd.concat([df_data, dummy_data], axis= 1)



#Remove Other Columns



new_df = df_data.sample(frac = 1).reset_index(drop= True)[:25000] # Testing the scores using the first 25000

new_df = new_df.drop(columns=['ID','Timestamp','Query','TwitterID', 'Sentiment', 'Sent_0']) # Removing all the other columns except tweet and Sentiment
class preprocessing:



    """

    Preprocesing - Preprocess and cleans the data by removing special characters, urls, stemming. Returns list of stemmed and cleaned words

    :Params data:  - Inputs a Pandas series and replace String by List of keywords

    data = "I love Springs"

    >>>preprocessing(data)

    ['i','love','spring']

    """

    def __init__(self, data):

        self.data = data

        self.data = self.data.apply(lambda x : self._stemmer(x))

        

    def _cleanTokenize(self,x):

        x = re.sub(r'http\S+', ' ', x) # Removes URLs

        x = re.sub('[^A-Za-z]+', ' ', x) # Removes special characters and Numbers

        regextoken = x.split() # Tokenization of Words

        return regextoken

    

    def _stemmer(self, data): # Stemming of words - Changing then to their lowest form

        regextoken= self._cleanTokenize(data)

        porter = PorterStemmer()

        stemmed  = [porter.stem(word) for word in regextoken]

        return stemmed

    
#Randomly divide the samples and perform classification on them - Using Skfold Split



class modelling:

    """

    Class Modelling: Applies model supplied to the data supplied and provides performance metrics 

    :Params Model: Model to be used for classification

    :Params Data: Dataframe on which predictions needs to be applied

    

    >>>modelling(logreg, dataframe)

    """

    def __init__(self, model, data):

        self.model= model # Creating model as a class/instance variable

        self.data= data # Initializing the data variable

        y_test, y_pred = self._apply_model(self.data)

        self._metrics(y_test, y_pred)

    

    def _apply_model(self, data): # Apply model function to dataset, no test to divide train, test and validation sets 

        X = data.Tweet

        X = preprocessing(data.Tweet)

        X = X.data

        y = data.Sent_4

        skfold = StratifiedKFold(n_splits=3, random_state=42)

        i = 0



        for train_index, test_index in skfold.split(X,y):

            clone_clf = clone(self.model)

            

            if i== 0:

                

                vectorizer = tfidf.fit(X[train_index].astype(str))

                X_train_folds = vectorizer.transform(X[train_index].astype(str))

                

            else:



                X_train_folds = vectorizer.transform(X[train_index].astype(str)) 

            

            y_train_folds = y[train_index]

            X_test_folds = vectorizer.transform(X[test_index].astype(str))

            y_test_fodls = y[test_index]

            i = i+1

            

            clone_clf.fit(X_train_folds, y_train_folds)

            y_pred = clone_clf.predict(X_test_folds)

            n_correct = sum(y_pred == y_test_fodls)

            print('Accuracy is: ',n_correct/len(y_pred))

        return y_test_fodls, y_pred

            

            

    def _metrics(self, y_test_fodls, y_pred): # Internal function for performance matrix

        print('Confusion matrix: ', confusion_matrix(y_test_fodls, y_pred)) # Confusion matrix

        print('Recall Score: ', recall_score(y_test_fodls, y_pred)) # Recall Score

        print('Precision Score: ', precision_score(y_test_fodls, y_pred)) # Precision Score

        print('F1 Score: ', f1_score(y_test_fodls, y_pred)) # F1 Score



        precisions, recalls, thresholds = precision_recall_curve(y_test_fodls, y_pred)



        def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):



            plt.plot(thresholds, precisions[:-1], "b--", label="Precision")

            plt.plot(thresholds, recalls[:-1], "g-", label="Recall")





        plot_precision_recall_vs_threshold(precisions, recalls, thresholds)

        plt.legend()

        plt.show()



        #ROC Curve

        fpr, tpr, thresholds = roc_curve(y_test_fodls, y_pred)



        def plot_roc_curve(fpr, tpr, label=None):

            plt.plot(fpr, tpr, linewidth=2, label=label)

            plt.plot([0, 1], [0, 1], 'k--') # dashed diagonal

            [...] # Add axis labels and grid

        plot_roc_curve(fpr, tpr)

        plt.legend()

        plt.show()
# Applying Logistic Regression Classisifier

logreg =  LogisticRegression()

modelling(model= logreg, data= new_df) # modelling.apply_model()
#XGBoost classifier

xgboost =  xgboost.XGBClassifier()

modelling(model= xgboost, data= new_df)# modelling.apply_model()
#Applying Random Forest Classifier

pipeline = Pipeline([

    ("r_forest", RandomForestClassifier(n_estimators= 200, max_leaf_nodes= 2000, n_jobs=-1)),

])

modelling(pipeline,new_df)
#Applying SVM Classifier

sgd = SGDClassifier()

modelling(sgd, new_df)