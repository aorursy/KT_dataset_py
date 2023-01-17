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
#pip install comet_ml
# Import comet_ml in the top of your file

#from comet_ml import Experiment



#experiment = Experiment(api_key="1T5FVvyOGYzMIIYf6dyXqFLcJ",

                        #project_name="classification-predict", workspace="juandreliebenberg")
# Context manager allows logging og parameters



#experiment.context_manager("validation")
# URL where experiments can be found



#experiment.url
# Analysis Libraries

import pandas as pd

import numpy as np

from collections import Counter



# Visualisation Libraries

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style("whitegrid")

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

from PIL import Image



# Language Processsing Libraries

import nltk

#nltk.download(['punkt','stopwords'])

#nltk.download('vader_lexicon')

#nltk.download('popular')

from sklearn.utils import resample

from nltk.stem import WordNetLemmatizer 

from nltk.tokenize import sent_tokenize, word_tokenize

from nltk.tokenize import word_tokenize, TreebankWordTokenizer 

import re

import string

from nltk import SnowballStemmer

import spacy



# ML Libraries

from sklearn.pipeline import Pipeline

from sklearn.svm import LinearSVC,SVC

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.metrics import classification_report,confusion_matrix,accuracy_score,recall_score,precision_score

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn import metrics

from nltk import SnowballStemmer



# Code for hiding seaborn warnings

import warnings

warnings.filterwarnings("ignore")
X_df = pd.read_csv('/kaggle/input/climate-change-belief-analysis/train.csv')

X_df.head(5)
# Inspect structure of dataset



X_df.info()
# Generate dataframe to indicate unique values



number_of_unique=[X_df[i].nunique() for i in X_df.columns] # Number of unique values per column

column_names=[i for i in X_df.columns]

unique_zip=list(zip(column_names,number_of_unique))

unique_df=pd.DataFrame(unique_zip,columns=['Column_Feature','Unique_Values'])

unique_df
# A function to remove duplicate rows from the message column



def delete_dup(df):

  df=df.copy()

  df = df.drop_duplicates(subset='message') #messges specified as subset to evaluate

  return df
X_df=delete_dup(X_df)
# Recheck for duplicates

number_of_unique=[X_df[i].nunique() for i in X_df.columns]

unique_df=pd.DataFrame(unique_zip,columns=['Column_Feature','Unique_Values'])

unique_df
# A function to add the text version of 'sentiment'. This is just for graphing purposes

# and should be droped.

def add_text_sent(df):



  

    # Copy the input DataFrame



    out_df = df.copy()

    

    sentiment_text = []

    

    # Loop though the sentiments and assign the text version. 

    # Pro: 1, News: 2, Neutral: 0, Anti: -1

    for sent in df['sentiment']:

        

        if sent == 1:

            sentiment_text.append('Pro')

            

        elif sent == 2:

            sentiment_text.append('News')

            

        elif sent == 0:

            sentiment_text.append('Neutral')

            

        elif sent == -1:

            sentiment_text.append('Anti')

            

    out_df['sentiment_text'] = sentiment_text

    

    out_df.drop(['message', 'tweetid'], axis = 1, inplace = True)

        

    return out_df
# Function to arrange the DataFrame to show percentage of classes

def class_table(df):

    out_df = df.groupby(['sentiment_text']).count()

    

    class_perc = [round(100 * x / len(df), 1) for x in out_df['sentiment']]

    

    out_df['% of Total Classes'] = class_perc

    

    return out_df
# Create a new DataFrame for graphing purposes. Show the sentiment classes as a 

# percentage.

new_X_df = add_text_sent(X_df)

new_X_df_t = class_table(new_X_df)

new_X_df_t
# Show the ditribution of the classes as a graph



f, ax = plt.subplots(figsize=(10, 8))

sns.set(style="whitegrid")

ax = sns.countplot(x="sentiment_text", data=new_X_df)

plt.title('Message Count', fontsize =20)

plt.show()
# Add a column of length of tweets



new_X_df['message_length'] = X_df['message'].str.len()

new_X_df.head()
# Display the boxplot of the length of tweets.

plt.figure(figsize=(12.8,6))

sns.boxplot(data=new_X_df, x='sentiment_text', y='message_length');
# Plot of distribution of scores for building categories

plt.figure(figsize=(12.8,6))

    

# Density plot of Energy Star scores

sns.kdeplot(new_X_df[new_X_df['sentiment_text'] == 'Pro']['message_length'], label = 'Pro', shade = False, alpha = 0.8);

sns.kdeplot(new_X_df[new_X_df['sentiment_text'] == 'News']['message_length'], label = 'News', shade = False, alpha = 0.8);

sns.kdeplot(new_X_df[new_X_df['sentiment_text'] == 'Neutral']['message_length'], label = 'Neutral', shade = False, alpha = 0.8);

sns.kdeplot(new_X_df[new_X_df['sentiment_text'] == 'Anti']['message_length'], label = 'Anti', shade = False, alpha = 0.8);



# label the plot

plt.xlabel('message length (char)', size = 15); plt.ylabel('Density', size = 15); 

plt.title('Density Plot of Message Length by Sentiment Class ', size = 20);
# Function to remove/replace unwanted text such as characters,URLs etc



def clean(text):



  text=text.replace("'",'')

  text=text.replace(".",' ')

  text=text.replace("  ",'')

  text=text.replace(",",' ')

  text=text.replace("_",' ')

  text=text.replace("!",' ')

  text=text.replace("RT",'retweet') #Replace RT(Retweet) with relay

  text=text.replace(r'\d+','')

  text=re.sub('((www\.[^\s]+)|(https?://[^\s]+)|(https?//[^\s]+))','weblink',text)

  text=re.sub('((co/[^\s]+)|(co?://[^\s]+)|(co?//[^\s]+))','',text)

  text=text.lower()  # Lowercase tweet

  text =text.lstrip('\'"') # Remove extra white space

  

  return text
#Function 3

def rm_punc(text):

  

  clean_text=[]

  for i in str(text).split():

    rm=i.strip('\'"?,.:_/<>!')

    clean_text.append(rm)

  return ' '.join(clean_text)
X_df['message']=X_df['message'].apply(clean)

X_df['message']=X_df['message'].apply(rm_punc)

X_df.head(5)
# Function replaces the @ symbol with the word at



def at(text):

 

  return ' '.join(re.sub("(@+)","at ",text).split())
# Function replaces the # symbol with the word tag



def hashtag(text):



  return ' '.join(re.sub("(#+)"," tag ",text).split())
# Remove hashtags and replace @



X_df['message']=X_df['message'].apply(at)

X_df['message']=X_df['message'].apply(hashtag)

X_df.head(5)
# Tokenise each tweet messge



tokeniser = TreebankWordTokenizer()

X_df['tokens'] = X_df['message'].apply(tokeniser.tokenize)

X_df.head(5)
# Function performs lemmatization in the tokens column



def lemma(text):

  lemma = WordNetLemmatizer() 

  return [lemma.lemmatize(i) for i in text]

X_df['lemma'] =X_df['tokens'].apply(lemma)

X_df.head(5)
# Insert new clean message column

X_df['clean message'] = X_df['lemma'].apply(lambda i: ' '.join(i))

X_df.head(5)
# Create copy of X_df to generate word cloud DataFrame



word_df=X_df.copy()
# Remove small words that will clutter word cloud and have no significant meaning



def remove_small(text):

  output=[]

  for i in text.split():

  

    if len(i)>3:

      output.append(i)

    else:

      pass

  return ' '.join(output)



word_df['clean message']=word_df['clean message'].apply(remove_small)
# Create and generate a word cloud image:



# Display the generated image:



fig, axs = plt.subplots(2, 2, figsize=(18,10))





# Anti class word cloud



anti_wordcloud = WordCloud(width=1800, height = 1200,background_color="white").generate(' '.join(i for i in word_df[word_df['sentiment']==-1]['clean message']))

axs[0, 0].imshow(anti_wordcloud, interpolation='bilinear')

axs[0, 0].set_title('Anti Tweets')

axs[0, 0].axis('off')



# Neutral cloud word cloud



neutral_wordcloud = WordCloud(width=1800, height = 1200,background_color="white").generate(' '.join(i for i in word_df[word_df['sentiment']==0]['clean message']))

axs[0, 1].imshow(neutral_wordcloud, interpolation='bilinear')

axs[0, 1].set_title('Neutral Tweets')

axs[0, 1].axis('off')



# Positive class word cloud



positive_wordcloud = WordCloud(width=1800, height = 1200).generate(' '.join(i for i in word_df[word_df['sentiment']==1]['clean message']))

axs[1, 0].imshow(positive_wordcloud, interpolation='bilinear')

axs[1, 0].set_title('Positive Tweets')

axs[1, 0].axis('off')



# News class word cloud



news_wordcloud = WordCloud(width=1800, height = 1200).generate(' '.join(i for i in word_df[word_df['sentiment']==2]['clean message']))

axs[1, 1].imshow(news_wordcloud, interpolation='bilinear')

axs[1, 1].set_title('News Tweets')

axs[1, 1].axis('off')



plt.show()
# Spacy will be used to generate entities

nlp = spacy.load('en_core_web_sm')
# A new dataframe NER_df is created for the following visualisations

NER_df=pd.DataFrame(X_df['clean message'])
# Function generates docs to get Name Entity Recognitions



def doc(text):

  doc=nlp(text)

  return doc
# Create a new column containing the nlp transformed text

NER_df['doc']=NER_df['clean message'].apply(doc)
#Functions below extract persons and organisations from the input parameter text. If entity is not found 'None' is populated in cell



def person(doc):

    if doc.ents:

        for ent in doc.ents:

          if ent.label_=='PERSON':

            return (ent.text)

    else:

      return ('None')



def org(doc):

    if doc.ents:

        for ent in doc.ents:

          if ent.label_=='ORG':

            return (ent.text)

    else:

      return ('None')
# Generate new columns 'persons' and 'organisation'



NER_df['persons']=NER_df['doc'].apply(person)

NER_df['organisation']=NER_df['doc'].apply(org)
# Retrive all the PERSON labels from the NER_df and generate a new dataframe person_df for analysis

persons=[i for i in NER_df['persons']]

person_counts = Counter(persons).most_common(20)

person_df=pd.DataFrame(person_counts,columns=['persons name','count'])

person_df.drop([0,1,7,8,13,15,16],axis=0,inplace=True) # rows removed due to 'None' entries, incorrect classification or different entry of a same entity (repetition)



# Plot top persons tweeted

f, ax = plt.subplots(figsize=(30, 10))

sns.set(style='white',font_scale=1.2)

sns.barplot(x=person_df[person_df['count'] <1000].iloc[:,0],y=person_df[person_df['count'] <1000].iloc[:,1])

plt.xlabel('Persons Name') 

plt.ylabel('Mentions')   

plt.show()
# Retrive all the ORG labels from the NER_df and generate a new dataframe org_df for analysis

org=[i for i in NER_df['organisation']]



org_counts = Counter(org).most_common(15)

org_df=pd.DataFrame(org_counts,columns=['organisation name','count'])

org_df.drop([0,1,3,8,12],axis=0,inplace=True) # rows removed due to 'None' entries, incorrect classification or different entry of a same entity (repetition)



# Plot top organisations tweeted

f, ax = plt.subplots(figsize=(30, 10))

sns.set(style='white',font_scale=2)

org_bar=sns.barplot(x=org_df[org_df['count'] <1000].iloc[:,0],y=org_df[org_df['count'] <1000].iloc[:,1])

plt.xlabel('Organisation Name') 

plt.ylabel('Mentions')   

plt.show()
# Feature and label split 



X=X_df['clean message']

y=X_df['sentiment']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
#pipeline = Pipeline([('tfidf', TfidfVectorizer()),('clf', SVC())])



#parameters = {

 #   'tfidf__max_df': (0.25, 0.5, 0.75),'tfidf__ngram_range': [(1, 1), (1, 2), (1, 3)],

   # 'tfidf__max_features':(500,2500,5000),'clf__C':(0.1,1,10),'clf__gamma':(1,0.1,0.001)}



#svc = GridSearchCV(pipeline, parameters, cv=2, n_jobs=2, verbose=3)

#svc.fit(X_train, y_train)
#svc.best_params_
#svc_predictions = svc.predict(X_test)
#Pipeline 



svc = Pipeline(

    [('tfidf', TfidfVectorizer(analyzer='word', max_df=0.75,max_features=5000,ngram_range=(1,1)))

    ,('clf', SVC(C=10,gamma=1))])



# Train model

model=svc.fit(X_train, y_train)



# Form a prediction set

predictions = model.predict(X_test)
# Print Results

#Confusion matrix

confusion = 'Confusion Matrix'.center(100, '*')

print(confusion)

matrix=confusion_matrix(y_test,predictions)

print(confusion_matrix(y_test,predictions))

print('')



#Classification report

report='Classification Report'.center(100,'*')

print(report)

print('')

print(classification_report(y_test,predictions))

print('')



#Model Performance

performance='Performance Metrics'.center(100,'*')

print(performance)

print('The model accuracy is :',accuracy_score(y_test,predictions))

print('The model recall is :',recall_score(y_test, predictions,average='weighted'))



F1 = 2 * (precision_score(y_test,predictions,average='weighted') * recall_score(y_test, predictions,average='weighted')) / (precision_score(y_test,predictions,average='weighted') + recall_score(y_test, predictions,average='weighted'))



print('The model F1score is : ',F1)
import pickle

model_save_path = "SVC.pkl"

with open(model_save_path,'wb') as file:

    pickle.dump(svc,file)
#import tes.csv



test=pd.read_csv('/kaggle/input/climate-change-belief-analysis/test.csv')
# Text cleaning



test['message']=test['message'].apply(clean) #clean data

test['message']=test['message'].apply(rm_punc) #remove punctuation

test['message']=test['message'].apply(at) #replace @

test['message']=test['message'].apply(hashtag) #remove #
# Tokenize messages



tokeniser = TreebankWordTokenizer()

test['tokens'] = test['message'].apply(tokeniser.tokenize)
# Lemmatize tokens column

 

test['lemma'] = test['tokens'].apply(lemma)
# Generate clean message column



test['clean message'] = test['lemma'].apply(lambda i: ' '.join(i))
test.head(5)
#Drop columns not needed for predictions

drop_list=['message','tokens','lemma']

test.drop(drop_list,axis=1,inplace=True)
test.head(5)
model_load_path = "SVC.pkl"

with open(model_load_path,'rb') as file:

    pickle_rick = pickle.load(file)
# Perfom predictions on test set



kaggle_predictions = pickle_rick.predict(test['clean message'])

kaggle_predictions = pd.DataFrame(kaggle_predictions)

kaggle_predictions.rename(columns={0: "sentiment"}, inplace=True)

kaggle_predictions["tweetid"] = test['tweetid']

cols = ['tweetid','sentiment']

kaggle_predictions = kaggle_predictions[cols]
kaggle_predictions.to_csv(path_or_buf='upload_kaggle_pred.csv',index=False)
#prediction output



pred_df=pd.read_csv('upload_kaggle_pred.csv')

pred_df
# Log metrics



#experiment = Experiment(api_key="1T5FVvyOGYzMIIYf6dyXqFLcJ")

#with experiment.context_manager("validation"):

  #svc.fit(X_train, y_train)

  #accuracy = accuracy_score(y_test,predictions)

  #recall = recall_score(y_test, predictions,average='weighted')

  #F1 = 2 * (precision_score(y_test,predictions,average='weighted') * recall_score(y_test, predictions,average='weighted')) / (precision_score(y_test,predictions,average='weighted') + recall_score(y_test, predictions,average='weighted'))

  # returns the validation accuracy,recall and F1 score

  #experiment.log_metric("accuracy", accuracy)

  #experiment.log_metric("recall", recall)

  #experiment.log_metric("F1", F1)
# End the experiment



#experiment.end()
# Display comet experiment



#experiment.display()