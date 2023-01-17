# Imports go here

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import glob
import nltk
import json
from pprint import pprint
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
%matplotlib inline

# Some downloads, separate from the imports
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('vader_lexicon')

from nltk.sentiment.vader import SentimentIntensityAnalyzer
# Starter code, filters articles from another kernel.  Source: https://www.kaggle.com/mlconsult/summary-page-covid-19-risk-factors
# I chose to do this to both focus my results on what matters (COVID-19) as well as limit overhead.  This cuts the dataset down from over 100k to about 23k articles.



def search_focus(df):
    dfa = df[df['abstract'].str.contains('covid')]
    dfb = df[df['abstract'].str.contains('-cov-2')]
    dfc = df[df['abstract'].str.contains('cov2')]
    dfd = df[df['abstract'].str.contains('ncov')]
    frames=[dfa,dfb,dfc,dfd]
    df = pd.concat(frames)
    df=df.drop_duplicates(subset='title', keep="first")
    return df
#Same source as above, edited for my own purposes
# Read in CSV
df=pd.read_csv('/kaggle/input/CORD-19-research-challenge/metadata.csv', usecols=['title','journal','abstract','authors','doi','publish_time', 'pdf_json_files', 'sha', 'pmcid', 'cord_uid', 'mag_id', 'who_covidence_id', 'url'])

# Fill empty fields
df=df.fillna('xyz') #using a string that will not occur naturally here, for reasons that will make sense later.

# Drop duplicate titles
df = df.drop_duplicates(subset='title', keep="first")

# Keep only 2020 dated papers
df=df[df['publish_time'].str.contains('2020')]

# Convert abstracts to lowercase + adds title
df["abstract"] = df["abstract"].str.lower()+df["title"].str.lower()

# Filter down to our focused list using the above method
df=search_focus(df)

print ("COVID-19 focused docuemnts ",df.shape)
df.head()
def remove_xyz(row):
    row.abstract = row.abstract.replace('xyz', '')
    return row
df.apply(remove_xyz, axis=1)

df.head()
df.query('pdf_json_files != "xyz" and sha != "xyz"')
def get_json_path(row):
    return row['pdf_json_files'] #will return 'xyz' if no filepath, can get rid of later
    
json_paths = []
x = df.apply(get_json_path, axis=1) #apply function to each row in df, result returns as Series


#will have plenty of rows that say 'xyz' still, now to remove those and get our paths in a list
for y in x:
    if y != 'xyz':
        json_paths.append(y)

        
#some of these paths are actually paths to two files - need to separate them out
dupe_list = []
for path in json_paths:
    if ';' in path:
        dupe_list.append(path)
        json_paths.remove(path)

        

#for some reason I need to do this twice, I do not know why.  If I only do it once, 33 items remain.  It does not make sense.  I'm leaving it in.
for path in json_paths:
    if ';' in path:
        dupe_list.append(path)
        json_paths.remove(path)
        
for item in dupe_list: #split and append them back in
    temp_list = item.split(';')
    for i in temp_list:
        i = i.replace(' ', '') #get rid of spaces left from splitting/however they made it in
        json_paths.append(i)
        

        
#paths are incomplete, need to add a bit more to them
json_paths_complete = []
for path in json_paths:
    json_paths_complete.append('/kaggle/input/CORD-19-research-challenge/{}'.format(path))

#now we have a nice list in order!
#Next step is to use our list to read in a second dataframe.
#This was a major roadblock for a while, this notebook was incredibly helpful: https://www.kaggle.com/fmitchell259/create-corona-csv-file
#I've added in a few columns I won't fill here, to fill with metadata later
#NOTE TO SELF: THIS TAKES A WHILE TO RUN, DON'T RUN MORE THAN YOU NEED TO

cols = ['doc_id', 'title', 'abstract', 'text_body', 'publish_time', 'authors', 'journal', 'url']
df2 = pd.DataFrame(columns=cols)



#get all json data
for path in json_paths_complete:
    row = {"doc_id": None, "title": None, "abstract": None, "text_body": None, "publish_time": None, "authors": None, "journal": None, "url": None} #create dict to store the data from JSON
    
    with open(path) as json_data: #read in JSON data
        data = json.load(json_data)
        json_data.close()
        
        row['doc_id'] = data['paper_id'] #load in doc id and title
        row['title'] = data['metadata']['title']

        abstract_list = [data['abstract'][x]['text'] for x in range(len(data['abstract']) - 1)] #use list comp to get all abstract data
        abstract = "\n ".join(abstract_list) #join with newlines for more readability
        row['abstract'] = abstract
        
        body_list = [] #use for loop to get all body data (per the original workbook, list comps seem to break here, so using their less pythonic solution)
        for _ in range(len(data['body_text'])):
            try:
                body_list.append(data['body_text'][_]['text'])
            except:
                pass
        
        body = "\n ".join(body_list) #join with newlines for more readability
        row['text_body'] = body
        
        
        df2 = df2.append(row, ignore_index=True)


df2.shape

#create a method to update a single row (gonna use apply later)
#NOTE: THIS ALSO TAKES A LONG TIME TO RUN.  LIMIT YOUR USE OF THIS.

#pseudocode:
    #create final empty DF with all columns
    #create dict
    #add in doc_id, text_body from df2
    #search for row from df using doc_id to get metadata
    #update publish_time, authors, and journal, title, abstract, and url from metadata
    #append new row to list
    #use list to create final DF, cov_data



df_dicts= []
for index, row in df2.iterrows(): #loop through rows in df (yes I know this is bad practice, will refactor later if i have time)
    new_row = {"doc_id": row['doc_id'], "title": 'xyz', "abstract": 'xyz', "text_body": row['text_body'], "publish_time": 'xyz', "authors": 'xyz', "journal": 'xyz'} #create the dict that will become a new row in cov_data
    try: #was having some value errors here, so encased this in a Try-Except block so it at least completes.
        temp_row = df[df['sha'].str.contains(new_row['doc_id'], na=False)] #using sha and doc_id as the shared key, get the original metadata
        new_row['publish_time'], new_row['authors'], new_row['journal'], new_row['title'], new_row['abstract'], new_row['url'] = temp_row['publish_time'], temp_row['authors'],temp_row['journal'], temp_row['title'], temp_row['abstract'], temp_row['url'] #assign from the metadata
    except ValueError:
        print("Error: no such value")
    
    #print(new_row)
    df_dicts.append(new_row) #and append to our list of dicts
 

cov_data = pd.DataFrame(df_dicts)

cov_data.head()

#and we now have our compiled DF!!!
def strip_journal(row):
    s = str(row['journal'])
    return s[6:-29]


def strip_title(row):
    s = str(row['title'])
    return s[6:-27]

def strip_authors(row):
    s = str(row['authors'])
    return s[6:-29]

def strip_abstract(row):
    s = str(row['abstract'])
    return s[6:-30]
    

def strip_publish_time(row):
    s = str(row['publish_time'])
    return s[6:-34]
    
def strip_url(row):
    s = str(row['url'])
    return s[6:-25]
cov_data['journal'] = cov_data.apply(strip_journal, axis=1).astype(str)
cov_data['title'] = cov_data.apply(strip_title, axis=1).astype(str)
cov_data['abstract'] = cov_data.apply(strip_abstract, axis=1).astype(str)
cov_data['authors'] = cov_data.apply(strip_authors, axis=1).astype(str)
cov_data['publish_time'] = cov_data.apply(strip_publish_time, axis=1).astype(str)
cov_data['url'] = cov_data.apply(strip_url, axis=1).astype(str)
#remove spaces from datetime and url cols
def remove_spaces(row):
    row['publish_time'] = row.publish_time.replace(' ', '')
    row['url'] = row.url.replace(' ', '')
    return row
cov_data.apply(remove_spaces, axis=1)
cov_data.head()
#there are a couple date issues to clean as well:
def clean_dates(row):
    if len(row.publish_time) > 10:
        #print(row['publish_time'])
        row['publish_time'] = row.publish_time[:10]
    return row
cov_data.apply(clean_dates, axis=1)
cov_data.head()
#now convert our date format to datetime
cov_data['publish_time'] = pd.to_datetime(cov_data['publish_time'], format ='%Y-%m-%d')

#and remove any lingering spaces from the front of the journal col
def journal_space(row):
    s = str(row['journal'])
    s = s.strip()
    return s
cov_data['journal'] = cov_data['journal'].astype(str)
cov_data['journal'] = cov_data.apply(journal_space, axis=1)
cov_data.journal.value_counts()[1:11].sort_values(ascending=False) #skipping 0, as it is our null value XYZ
#pickling the dataframe so I don't have to run all the above later
#I've commented this so I don't accidentally run it again and mess up my backup!  I also have a local version saved just in case.

cov_data.to_pickle('./cov_data.pkl')

#unpickling for future use - all future work will be on the df Dataframe

df = pd.read_pickle('./cov_data.pkl')

df.head()
df.info()
df.publish_time.hist(bins=12, figsize=(25,10)); 
plt.title('Histogram of published work'); 
plt.xlabel('Month');
plt.ylabel('Frequency');
temp = df.journal.value_counts()
temp2 = temp.head(16)
temp2 = temp2.drop(temp2.index[0])
plt.figure(figsize=(22,10))
plt.xticks(rotation=45)
temp2.plot(kind='bar', x = 'Journal Name', y = '# Articles');
#bioRxiv is the most prolific journal by far, with almost 10x the articles of anyone else.
#Reading in CSV files from the folder

#pseudocode:
#get file names in list
#create master DF
#for each file in list:
#    read in temp df
#    add column to temp df with filename for tracking purposes
#    append to master DF


file_list = glob.glob("/kaggle/input/CORD-19-research-challenge/Kaggle/target_tables/8_risk_factors/*.csv") #get File List
master = pd.DataFrame(columns = ['Date', 'Study', 'Study link', 'Journal', 'Sample Size', 'y'])

#temp = pd.read_csv(file_list[0], usecols = ['Date', 'Study', 'Study link', 'Journal', 'Sample Size'])
#temp.head()

for file in file_list: #loop through list
    temp = pd.read_csv(file, usecols = ['Date', 'Study', 'Study link', 'Journal', 'Sample Size']) #there are a ton of cols, pulling in just a few here.  Will add more if needed.
    temp['y'] = file
    #print(temp.head())
    master = master.append(temp)

    
master.tail()
def clean_y(row): 
    return row.y[77:-4] #quick method to strip out the path and just leave the name of the csv
master['y'] = master.apply(clean_y, axis=1) #apply our method
master.rename(columns={'Study link': 'study_link', 'Sample Size': 'sample_size'}, inplace=True) #rename columns to remove spaces
master.head()
master.y.value_counts()
#here is how our study list breaks down
#so obviously, our y set is only 562 studies, whereas our full data set is over 10,000.  This means we will only be able to know the truth of a subset of our data.
#However, if we can be confident in the accuracy of our results on this small test set, we can feel confident that running our model on the rest of the set will also be accurate.


keywords = master.y.unique() #our list of key words is here

keywords = [word.lower() for word in keywords]

#removing common and otherwise useless words so they are not used as keywords
keywords = [word.replace(' diseases', '') for word in keywords] #there might be a better way to do this, but a series of list comps is fast if inelegant
keywords = [word.replace(' disease', '') for word in keywords]
keywords = [word.replace(' disorders', '') for word in keywords]
keywords = [word.replace(' status', '') for word in keywords]
keywords = [word.replace('- and cerebro', '') for word in keywords]
keywords = [word.replace(' system', '') for word in keywords]
keywords = [word.replace(' or', '') for word in keywords]
keywords = [word.replace(' vs.', '') for word in keywords]
keywords = [word.replace('_ black white', '') for word in keywords]
keywords = [word.replace('ethnicity_ ', '') for word in keywords]
keywords = [word.replace(' non-hispanic', '') for word in keywords]
keywords = [word.replace('overweight ', '') for word in keywords]
keywords = [word.replace('male ', '') for word in keywords]
keywords = [word.replace('chronic ', '') for word in keywords]
keywords = [word.replace(' failure', '') for word in keywords]
keywords.remove('heart') #get rid of the dupe
keywords

#something I didn't do initially was normalize my y as well down to the keywords - I am fixing that now.

master.y.replace('Male gender', 'gender', inplace=True)
master.y.replace('Diabetes', 'diabetes', inplace=True)
master.y.replace('Hypertension', 'hypertension', inplace=True)
master.y.replace('Heart Disease', 'heart', inplace=True)
master.y.replace('Smoking Status', 'smoking', inplace=True)
master.y.replace('COPD', 'copd', inplace=True)
master.y.replace('Age', 'age', inplace=True)
master.y.replace('Overweight or obese', 'obese', inplace=True)
master.y.replace('Chronic respiratory diseases', 'respiratory', inplace=True)
master.y.replace('Chronic kidney disease', 'kidney', inplace=True)
master.y.replace('Cancer', 'cancer', inplace=True)
master.y.replace('Cerebrovascular disease', 'cerebrovascular', inplace=True)
master.y.replace('Heart Failure', 'heart', inplace=True)
master.y.replace('Cardio- and cerebrovascular disease', 'cardiovascular', inplace=True)
master.y.replace('Race_ Black vs. White', 'race', inplace=True)
master.y.replace('Chronic liver disease', 'liver', inplace=True)
master.y.replace('Respiratory system diseases', 'respiratory', inplace=True)
master.y.replace('Ethnicity_ Hispanic vs. non-Hispanic', 'hispanic', inplace=True)
master.y.replace('Immune system disorders', 'immune', inplace=True)
master.y.replace('Chronic digestive disorders', 'digestive', inplace=True)
master.y.replace('Endocrine diseases', 'endocrine', inplace=True)
master.y.replace('Dementia', 'dementia', inplace=True)
master.y.replace('Asthma', 'asthma', inplace=True)
master.y.replace('Drinking', 'drinking', inplace=True)

master.y.value_counts()
master.reset_index(inplace=True)
#Let's also take the time to get a subset df that we can match y's to.  Using the url column, we should be able to get matches
def match_url(row):
    for i in range(len(master)): #loop through master df
        if str(master.loc[i, 'study_link']) == str(row['url']):
            #print('before: {}'.format(row['y']))
            row['y'] = master.loc[i, 'y']
            #print('after: {}'.format(row['y']))
            return row #ending it early might help it run a little faster
        
    return row
#Let's also take the time to get a subset df that we can match y's to.  Using the url column, we should be able to get matches
#takes a few minutes to complete

df['y'] = pd.Series(['xyz' for x in range(len(df.index))])

df = df.apply(match_url, axis=1)

df.head()


print(df['y'].value_counts())
df.shape
#apply method to count word occurrences
def key_words(row):
    for word in keywords:
        word2 = word.replace(' ', '_') #in order to be able to assign to our column
        row[word2] = row.text_body.count(word)
    #print(row)
    return row
#Now that we have our final list of keywords, lets add them to our df and count occurrences

#pseudocode
#add a count column to df for each of our keywords
#make sure all text_body is lowercase
#for each row in df,
#    for each new keyword column
#        step through text body, count occurrences of each keyword and store in relevant column




df['text_body'] = df['text_body'].str.lower()  #lowercase the text_body column   

for word in keywords: #add count columns
    word2 = word.replace(' ', '_') #remove spaces for column name
    df[word2] = pd.Series([0 for x in range(len(df.index))])

#even with apply, this is S L O W - be warned!
df = df.apply(key_words, axis=1)    

      
df.head()
#let's take a look at what our keywords turned up

sum_keywords = {}

for word in keywords:
    sum_keywords[word] = 0

for word in keywords:
    word2 = word.replace(' ', '_')
    sum_keywords[word] = df[word2].sum()
    

keys = sum_keywords.keys()
vals = sum_keywords.values()
plt.figure(figsize=(22,10))
plt.xticks(rotation=45)
plt.bar(keys, vals); # Quick plot by category
plt.title('Keyword Occurrence Frequency');
plt.xlabel('Keyword');
plt.ylabel('Frequency');
#create our X and y datasets.  It's going to be a small dataset, which is far from ideal


#create copy of dataset

df_copy =  df[df['y'] != 'xyz'] 

df_copy.reset_index(inplace=True)

y = df_copy[['y']]
X = df_copy[['hispanic', 'respiratory', 'copd', 'hypertension', 'digestive', 'obese', 'kidney', 'asthma', 'drinking','diabetes', 
             'heart', 'smoking', 'race', 'immune', 'age', 'gender','dementia', 'liver', 'cancer', 'endocrine', 'cerebrovascular','cardiovascular']]




#using a random forest, we are going to try to predict what each article is about in relation to risk factor!

rf = RandomForestClassifier()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.5, random_state = 42)

rf.fit(X_train, y_train)


y_preds = rf.predict(X_test)

#confusion_matrix(y_test, y_preds)

acc_randomforest = round(accuracy_score(y_preds, y_test) * 100, 2)
print(acc_randomforest)
#This initial test off of keywords didn't prove to be very accurate or precise.  Perhaps not surprising, as I did a pretty simplistic COUNT vectorizer of keywords.  
#Going to take a shot with a Lemmatizer over the whole document, and TFIDF next.
X_temp = df_copy['text_body']

X_temp.head()
from nltk.stem import WordNetLemmatizer 
from nltk import word_tokenize    
class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, articles):
        return [self.wnl.lemmatize(t) for t in word_tokenize(articles)]
tf = TfidfVectorizer(stop_words='english',
                     tokenizer=LemmaTokenizer())
X2 = tf.fit_transform(X_temp)
#train_test split again


X_train, X_test, y_train, y_test = train_test_split(X2, y, test_size = 0.20, random_state = 1989)
rf.fit(X_train, y_train)
y_preds = rf.predict(X_test)

#confusion_matrix(y_test, y_preds)

acc_randomforest = round(accuracy_score(y_preds, y_test) * 100, 2)
print(acc_randomforest)
#Lemmatizer actually does way worse!

y_preds, y_test
#Clearly here, gender is taking over the model (slightly shocking, as I thought Age would be doing that).
#Unfortunately, at this point, I've run out of time, so we do have to go with this.


X_final = df[['hispanic', 'respiratory', 'copd', 'hypertension', 'digestive', 'obese', 'kidney', 'asthma', 'drinking','diabetes', 'gender',
                   'heart', 'smoking', 'race', 'immune', 'age','dementia', 'liver', 'cancer', 'endocrine', 'cerebrovascular','cardiovascular']]


X_final.shape
y_preds_final = rf.predict(X_final)

df['y_preds'] = y_preds_final #add our predictions back into the dataframe

df.y_preds.value_counts() #gender gets massively overfit here.

#and here is it graphed.  Full disclosure: This is most likely not accurate.

df.y_preds.value_counts().plot(kind='bar',title= 'Articles Grouped by Risk Factor', x = 'Risk Factor', y = '# Articles', figsize=(22,10));


#One thing to consider - besides the obvious gender overfit, all of these risk factors line up with what I know through osmosis about Coronavirus.
#Given we did filter down the dataset to COVID-related articles, it is possible that some of the other risk factors listed do not show up in our test dataset.
#Again, given more time, I would explore this in more detail, but time is not something I had enough of.
#Now that we have all of our dataset classified, let's use Vader to predict whether or not each has a positive or negative correlation.

#obviously, since the model is wonky, we are only looking at the 7 risk factors it narrowed down to.

#create a smaller df for Vader processing
df_vader = df[['title', 'journal', 'text_body', 'y_preds']]

sia = SentimentIntensityAnalyzer()

df_vader.head()
neg_scores = []
neu_scores = []
pos_scores = []
cpd_scores = []
for text in df_vader['text_body']:
    scores = sia.polarity_scores(text)
    neg_scores.append(scores['neg'])
    pos_scores.append(scores['pos'])
    neu_scores.append(scores['neu'])
    cpd_scores.append(scores['compound'])
    

df_vader['negative'] = neg_scores
df_vader['neutral'] = neu_scores
df_vader['positive'] = pos_scores
df_vader['compound'] = cpd_scores
df_vader.head()
#let's see how our vader scores ended up!
df_vader.describe()
#initialize our sentiment column
df_vader['sentiment'] = pd.Series(['neutral' for x in range(len(df.index))])
df_vader.head()
def find_sentiment(row):
    if row['positive'] > row['negative']:
        row['sentiment'] = 'positive'
    elif row['positive'] < row['negative']:
        row['sentiment'] = 'negative'
    #else leave it as neutral    
    return row
df_vader = df_vader.apply(find_sentiment, axis=1)
df_vader.sentiment.value_counts().plot(kind='bar', figsize=(22,10));
#let's split them up by risk factor.  First, one-hot our sentiment

df2 = pd.get_dummies(df_vader['sentiment'])

df2.rename(columns={'negative': 'neg', 'neutral': 'neu', 'positive': 'pos'}, inplace=True)

df3 = df_vader.join(df2)

df3
df_print = df3[['y_preds', 'neg', 'neu', 'pos']]

df_print.groupby('y_preds').sum().plot(kind='bar', figsize=(22,10));