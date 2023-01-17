# Import of Necessary libraries 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer,TfidfTransformer 

from sklearn.linear_model import LogisticRegression, LinearRegression

from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import accuracy_score

from sklearn import svm

from sklearn.svm import SVC

from sklearn.svm import LinearSVC

from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.model_selection import GridSearchCV



from sklearn.pipeline import FeatureUnion

from sklearn.model_selection import train_test_split 

from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.decomposition import TruncatedSVD





import os

print(os.listdir("../input"))

%matplotlib inline
category= pd.read_csv('../input/category.csv').set_index('category_id')

category.head()
#Reading of data 

df = pd.read_csv('../input/train.csv')

test_set = pd.read_csv('../input/test.csv')

train_data = df.drop(['category_id','item_id'],axis=1) # drop columns not useful for training 

train_target = df['category_id']
df.head(5) # Familiarisation with dataset
df.shape, test_set.shape  
test_set.head()
df.isnull().sum() # Check for missing data
len(df['description'][0]), len(df['title'][0])
len(df['description'][4]), len(df['title'][4])
lens = df.description.str.len()

print (lens.mean(), lens.std(), lens.max())

lens.hist(); 
lens1 = df.title.str.len()

print(lens1.mean(), lens1.std(), lens1.max())

lens1.hist(); 
# Concatenate the text features from the train and test datasets so that  

#a unique bag of words contains words from both datasets 



train_text = df['title'] + ' ' + df['description'] 

test_text = test_set['title'] + ' ' + test_set['description']

all_text = pd.concat([train_text, test_text,])
from nltk.corpus import stopwords

from nltk.stem.snowball import RussianStemmer, SnowballStemmer



stemmer = SnowballStemmer("russian", ignore_stopwords=False)   #This converts all words to their stem format  )



stop_words = stopwords.words('russian') # russian stop words from nltk library 





class StemmedTfidfVectorizer(TfidfVectorizer):

    

    def __init__(self, stemmer, *args, **kwargs):

        super(StemmedTfidfVectorizer, self).__init__(*args, **kwargs)

        self.stemmer = stemmer

        

    def build_analyzer(self):

        analyzer = super(StemmedTfidfVectorizer, self).build_analyzer()

        return lambda doc: (self.stemmer.stem(word) for word in analyzer(doc.replace('\n', ' ')))



# build the feature matrices

word_vectorizer = TfidfVectorizer(stop_words= stop_words, 

                                  sublinear_tf= True,

                                  lowercase= True, 

                                  analyzer= 'word',

                                  max_features= 150000)
train = word_vectorizer.fit_transform(train_text) #Now we use our vectors to transform our training set

test  = word_vectorizer.transform(test_text)
#In order to avoid overfitting, we split the training data into train and test sets

X_train, X_test,y_train, y_test=train_test_split(train,train_target, 

                                                 test_size= 0.3, random_state=17)
import time



svc = LinearSVC() #.fit(X_train, y_train)

param_grid = {'C':(0.01,0.1,1,10)}

a=GridSearchCV(svc,param_grid,cv=5)

start = time.time()

a.fit(X_train,y_train)

# print (a.best_scores_)

print (a.best_params_)

end = time.time()

print ('Training classifier takes:', end-start, 'ms')
print (a.best_params_)
import time

#train the classifier

start = time.time()

svc  = LinearSVC().fit(X_train, y_train)

clf = LogisticRegression(solver='sag',multi_class= 'multinomial').fit(X_train,y_train)

# test the classifier

pred  = svc.predict(X_test)

pred1 = clf.predict(X_test)

# Accuracy scores

score  = accuracy_score(y_test,pred)

score_a = accuracy_score(y_train,svc.predict(X_train))

score1 = accuracy_score(y_test,pred1)

score1_a = accuracy_score(y_train,clf.predict(X_train) )

end = time.time()

print ('Training the classifiers takes:', end-start, 'ms')

print ('Logistic Regression Score:','Train:',score1_a,'Test:',score1,'\n',

       'Support Vector Machine Score:','Train:',score_a,'Test:',score)
#Check accuracy on each class

y_pred_=[pred[i] for i in range(len(pred))] 

y_test_=[y_test.iloc[i] for i in range(len(y_test))]

 

right_predictions=[0 for i in range(54)] 

wrong_predictions=[0 for i in range(54)]

 

for i in range(len(y_pred_)): 

    if y_pred_[i] == y_test_[i]: 

        right_predictions[y_test_[i]] += 1 

    else: 

        wrong_predictions[y_test_[i]] += 1

#Calculate accuracy for each class

accur_score = [0 for i in range(54)] 

for i in range(len(accur_score)): 

    accur_score[i] = right_predictions[i]/(right_predictions[i]+wrong_predictions[i])



class_score = pd.DataFrame(columns=['category_id','accuracy'])

class_score['category_id'],class_score['accuracy'] = range(54),accur_score

class_score.head()
#Heirachy Accuracy 

def add_cat(array):

    while len(array) !=4:

        array.append(' '.join(array))

    return array



cat_df = pd.DataFrame(category['name'].apply(lambda s: s.split('|')).apply(add_cat).tolist())

heir = {i:cat_df[i].to_dict() for i in range(4)}

def replace(array,mask):

    return pd.DataFrame(array).replace(mask).values.ravel()



for level, adapter in heir.items():

    acc = accuracy_score(replace(y_test,adapter), replace(pred,adapter))

    

    print (f'level={level}, accuracy={acc:.3f}')
## Implementation using sklearn pipeline

# Pipeline allows to combine all the transformations before fitting it to our data
from sklearn.pipeline import Pipeline, FeatureUnion



class ColumnSelector(BaseEstimator, TransformerMixin):

    def __init__(self, columns):

        self.columns = columns



    def fit(self, X, y=None):

        return self



    def transform(self, X):

        assert isinstance(X, pd.DataFrame)



        try:

            return X[self.columns]

        except KeyError:

            cols_error = list(set(self.columns) - set(X.columns))

            raise KeyError("The DataFrame does not include the columns: %s" % cols_error)

            

class TypeSelector(BaseEstimator, TransformerMixin):

    def __init__(self, dtype):

        self.dtype = dtype

    def fit(self, X, y=None):

        return self

    def transform(self, X):

        assert isinstance(X, pd.DataFrame)

        return X.select_dtypes(include=[self.dtype])
X_train, X_test,y_train, y_test = train_test_split(train_data,train_target,test_size= 0.2, random_state=17)



pipeline = Pipeline([

         # Use FeatureUnion to combine the features from title and description

    ('union', FeatureUnion(

        transformer_list=[



            # Pipeline for pulling features from title

            ('title', Pipeline([

                ('selector', ColumnSelector(columns='title')),

                ('tfidf', TfidfVectorizer(min_df=50,

                                         stop_words= stop_words, 

                                         sublinear_tf= True,

                                         lowercase= True, 

                                         analyzer= 'word',

                                         max_features= 150000)),

            ])),



            # Pipeline for standard bag-of-words model for description

            ('description', Pipeline([

                ('selector', ColumnSelector(columns='description')),

                ('tfidf', TfidfVectorizer(stop_words= stop_words, 

                                          sublinear_tf= True,

                                          lowercase= True, 

                                          analyzer= 'word',

                                          max_features= 150000)),

                ('best', TruncatedSVD(n_components=50)),

            ])),



            # Pipeline for pulling price features 

            ('price', Pipeline([

            ('selector', TypeSelector(np.float)),

            ('scaler', StandardScaler())

            ])),

        ],

    )),

    # Use a SVC classifier on the combined features

    ('svc', LinearSVC()),

])



start1 = time.time()

pipeline.fit(X_train, y_train)

pred = pipeline.predict(X_test)

score1 = accuracy_score(y_test,pred)

end1= time.time()



print ('Training SVC with pipeline  takes:', end1-start1, 'ms')

print ('Support Vector Machine Score:',score1)

# Result on the provided test set using the trained SVC classifier

result = pd.DataFrame()   

result['item_id']= test_set['item_id']

result['category_id'] =  svc.predict(test)

result.to_csv('submission_1.csv', index=False)
pipeline.predict(test_set)

result1=  pd.DataFrame()

result1['item_id'] = test_set['item_id']

result1['category_id']= pipeline.predict(test_set)

result1.to_csv('submission_2.csv')
result.head(10)
result1.head(10)