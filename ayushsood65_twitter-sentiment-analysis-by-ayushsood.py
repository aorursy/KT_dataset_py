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
dataset_encoder="ISO-8859-1"
dataset_column=['sentiments','id','date','flag','user','text']
dataset=pd.read_csv("../input/sentiment140/training.1600000.processed.noemoticon.csv",encoding=dataset_encoder,names=dataset_column)
dataset
# remove the unneccesory columns
dataset=dataset[['sentiments','text']]
dataset
dataset['sentiments'].unique()
dataset['sentiments']=dataset['sentiments'].replace(4,1)
dataset['sentiments'].unique()
dataset.tail(10)
#Ploting
plt=dataset.groupby('sentiments').count().plot(kind='bar',title='Data Distribution',legend=True)
plt.set_xticklabels(["Negative","Positive"],rotation=0)
# store the data in list
text,sentiment=list(dataset['text']),list(dataset['sentiments'])
# PreProcess the data
# Defining dictionary containing all emojis with their meanings.
emojis = {':)': 'smile', ':-)': 'smile', ';d': 'wink', ':-E': 'vampire', ':(': 'sad', 
          ':-(': 'sad', ':-<': 'sad', ':P': 'raspberry', ':O': 'surprised',
          ':-@': 'shocked', ':@': 'shocked',':-$': 'confused', ':\\': 'annoyed', 
          ':#': 'mute', ':X': 'mute', ':^)': 'smile', ':-&': 'confused', '$_$': 'greedy',
          '@@': 'eyeroll', ':-!': 'confused', ':-D': 'smile', ':-0': 'yell', 'O.o': 'confused',
          '<(-_-)>': 'robot', 'd[-_-]b': 'dj', ":'-)": 'sadsmile', ';)': 'wink', 
          ';-)': 'wink', 'O:-)': 'angel','O*-)': 'angel','(:-D': 'gossip', '=^.^=': 'cat'}
# the data i am using here is taken from google you can simply go through this on google and get this much emojies in text form .

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stopwords.words('english')
# here we get all the words of stopwords 
# Defining set containing all stopwords in english.
stopwordlist = ['a', 'about', 'above', 'after', 'again', 'ain', 'all', 'am', 'an',
             'and','any','are', 'as', 'at', 'be', 'because', 'been', 'before',
             'being', 'below', 'between','both', 'by', 'can', 'd', 'did', 'do',
             'does', 'doing', 'down', 'during', 'each','few', 'for', 'from', 
             'further', 'had', 'has', 'have', 'having', 'he', 'her', 'here',
             'hers', 'herself', 'him', 'himself', 'his', 'how', 'i', 'if', 'in',
             'into','is', 'it', 'its', 'itself', 'just', 'll', 'm', 'ma',
             'me', 'more', 'most','my', 'myself', 'now', 'o', 'of', 'on', 'once',
             'only', 'or', 'other', 'our', 'ours','ourselves', 'out', 'own', 're',
             's', 'same', 'she', "shes", 'should', "shouldve",'so', 'some', 'such',
             't', 'than', 'that', "thatll", 'the', 'their', 'theirs', 'them',
             'themselves', 'then', 'there', 'these', 'they', 'this', 'those', 
             'through', 'to', 'too','under', 'until', 'up', 've', 'very', 'was',
             'we', 'were', 'what', 'when', 'where','which','while', 'who', 'whom',
             'why', 'will', 'with', 'won', 'y', 'you', "youd","youll", "youre",
             "youve", 'your', 'yours', 'yourself', 'yourselves']
def preprocess(textdata):
    processedText = []
    
    # Create Lemmatizer and Stemmer.
    wordLemm = WordNetLemmatizer()
    
    # Defining regex patterns.
    import re
    urlPattern        = r"((http://)[^ ]*|(https://)[^ ]*|( www\.)[^ ]*)"
    userPattern       = '@[^\s]+'
    alphaPattern      = "[^a-zA-Z0-9]"
    sequencePattern   = r"(.)\1\1+"
    seqReplacePattern = r"\1\1"
    
    for tweet in textdata:
        tweet = tweet.lower()
        
        # Replace all URls with 'URL'
        tweet = re.sub(urlPattern,' URL',tweet)
        # Replace all emojis.
        for emoji in emojis.keys():
            tweet = tweet.replace(emoji, "EMOJI" + emojis[emoji])  
        
        # Replace @USERNAME to 'USER'.
        tweet = re.sub(userPattern,' USER', tweet)        
        # Replace all non alphabets.
        tweet = re.sub(alphaPattern, " ", tweet)
        # Replace 3 or more consecutive letters by 2 letter.
        tweet = re.sub(sequencePattern, seqReplacePattern, tweet)

        tweetwords = ''
        for word in tweet.split():
            # Checking if the word is a stopword.
            #if word not in stopwordlist:
            if len(word)>1:
                # Lemmatizing the word.
                word = wordLemm.lemmatize(word)
                tweetwords += (word+' ')
            
        processedText.append(tweetwords)
        
    return processedText
 
    
from nltk.stem import WordNetLemmatizer
processedtext = preprocess(text)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(processedtext, sentiment,
                                                    test_size = 0.05, random_state = 0)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, classification_report
vectorization=TfidfVectorizer(ngram_range=(1,2),max_features=500000)
vectorization.fit(X_train)
X_train = vectorization.transform(X_train)
X_test  = vectorization.transform(X_test)
# data evaluation
def model_Evaluate(model):
    
    # Predict values for Test dataset
    y_pred = model.predict(X_test)

    # Print the evaluation metrics for the dataset.
    print(classification_report(y_test, y_pred))
    
    cf_matrix = confusion_matrix(y_test, y_pred)
    print(cf_matrix)
from sklearn.linear_model import LogisticRegression
regressor = LogisticRegression(C = 2, max_iter = 1000, n_jobs=-1)
regressor.fit(X_train, y_train)
model_Evaluate(regressor)
import pickle
file=open('vetorizer.pickle','wb')
pickle.dump(vectorization,file)
file.close()
file1=open('Regressor.pickle','wb')
pickle.dump(regressor,file1)
file1.close()
def load_models():
    file=open('./Regressor.pickle','rb')
    regression=pickle.load(file)
    file.close()
    
    file=open('./vetorizer.pickle','rb')
    vectorizer=pickle.load(file)
    file.close()
    
    return vectorizer,regression
    
    
def predict(vectoriser, model, text):
    # Predict the sentiment
    textdata = vectorization.transform(preprocess(text))
    sentiment = model.predict(textdata)
    
    # Make a list of text with sentiment.
    data = []
    for text, pred in zip(text, sentiment):
        data.append((text,pred))
        
    # Convert the list into a Pandas DataFrame.
    df = pd.DataFrame(data, columns = ['text','sentiment'])
    df = df.replace([0,1], ["Negative","Positive"])
    return df

if __name__=="__main__":
    # Loading the models.
    #vectoriser, LRmodel = load_models()
    
    # Text to classify should be in a list.
    text = ["I hate twitter",
            "May the Force be with you.",
            "Mr. Stark, I don't feel so good"]
    
    df = predict(vectorization, regressor, text)
    print(df.head())
def predict(vectorization,model,text):
    textdata=vectorization.transform(preprocess(text))
    sentiment=model.predict(textdata)
    
    data=[]
    for text,pred in zip(text,sentiment):
        data.append((text,pred))
    
    #converting list into df
    dataset_created=pd.DataFrame(data,columns=['text','sentiment'])
    dataset_created= dataset_created.replace([0,1],['Negative','Positive'])
    return dataset_created

if __name__=='__main__':
    text=["I Love my India","I am very sad","Good to know You are Fine","See You Soon","Ayush have a Lot Of Patience"]
    dataset_Predict=predict(vectorization,regressor,text)
    dataset_Predict
    
        
dataset_Predict.to_csv("Predicted_result.csv")
dataset_Predict
