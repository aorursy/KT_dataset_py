## HOW WE TRAINED THE MODEL 



import os
import joblib
import numpy as np 
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt
import pickle
import re


import nltk.corpus
from nltk.stem import WordNetLemmatizer
from nltk import tokenize
from nltk.tokenize import sent_tokenize,word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import tokenize


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression

# nltk.download("all")

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

#LOADING DATA
train = pd.read_csv("/kaggle/input/twitter-sentiment-analysis-hatred-speech/train.csv")
test = pd.read_csv("/kaggle/input/twitter-sentiment-analysis-hatred-speech/test.csv")


#PLOTTING HISTOGRAM
train['label'].hist()
plt.show()


#TURNING LABELS INTO CATEGORIES
train['label'] = train['label'].astype('category')
train.drop(columns=['id'], inplace=True)


train_2 = pd.read_csv("/kaggle/input/hatespeechlabeleddata/labeled_data.csv")
train_2['class'].hist()
plt.show()

#create new columns containing 
train_2['label'] = 1
# this dataset contains three levels of hate speech level 0,1,2
# We have added all hate tweets to our existing data so that we can get a balanced data


#Creating dataframe
train_2 = pd.DataFrame(train_2[['label',"tweet"]])
#Concatinating both dataframes
train = [train, train_2]
result = pd.concat(train)

# shuffling data and reseting index
result=shuffle(result)
result = result.reset_index(drop=True)

#Plotting histogram
result['label'].hist()
#checking for null values
result.info()

# Storing stopwords of english language from nltk library
sw = set(stopwords.words("english"))

# remove stop words
def filter_words(word_list):
    useful_words = [ w for w in word_list if w not in sw ]
    return(useful_words)



def preprocess_data(dataset):
    data = dataset.copy()

    #Removing punctuations, special characters and lemmatizing words to their base form
    data['text_lem'] = [''.join([WordNetLemmatizer().lemmatize(re.sub('[^A-Za-z]',' ',text)) for text in lis]) for lis in data['tweet']]
    
    a=[]
    for text in data['text_lem']:
        word_list = word_tokenize(text)
        text=filter_words(word_list)
        a.append(text)  
    
    train_text = []
    for i in a:
        sent=''
        for  j in i:
            sent += str(j) + ' '
        train_text.append(sent)

    data['cleaned_tweets'] = train_text
    
    #Using TF-IDF vectorizer
    vect = TfidfVectorizer(ngram_range = (1,3)).fit(data['cleaned_tweets'])
    
    #Transforming our data using the vector trained on training data.  
    vectorized_tweets = vect.transform(data['cleaned_tweets'])
    
    return vectorized_tweets, vect


    
#storing preprocessed data in data_train and vector in vect
data_train,vect  = preprocess_data(result)
data_target = np.array(result["label"])

print(data_train.shape, data_target.shape)
X_train, X_test, y_train, y_test = train_test_split(data_train, data_target, test_size=0.2, random_state=42)
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
# Helping_Function to show Cross Val Scores
def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())

# we are using logistic regression  
lg_reg_clf = LogisticRegression(C=50, penalty='l2', solver='lbfgs')
#Calculating cross-val score
score = cross_val_score(lg_reg_clf, X_train, y_train, cv=7)
#Display CV score:
display_scores(score)
model = lg_reg_clf.fit(X_train, y_train)
print("Accuracy   :\t",lg_reg_clf.score(X_test,y_test))
sns.heatmap(confusion_matrix(lg_reg_clf.predict(X_test), y_test),annot=True)
plt.show()


with open('model','wb') as f:
    pickle.dump(model,f)
    
with open('vector','wb') as f:
    pickle.dump(vect,f)
    

