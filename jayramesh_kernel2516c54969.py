#import comet_ml in the top of your file

#from comet_ml import Experiment



#Inspecting

import numpy as np 

import pandas as pd 

pd.set_option('display.max_colwidth', -1)

import emoji

import re

import string





#Visuals

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

from wordcloud import WordCloud



import warnings

warnings.filterwarnings('ignore')



#Text preprocessing and cleaning

import nltk

from nltk.tokenize import word_tokenize, TreebankWordTokenizer

from nltk.stem import WordNetLemmatizer

from nltk.corpus import stopwords

from nltk.probability import FreqDist

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer



#Modelling

from sklearn.pipeline import Pipeline

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import AdaBoostClassifier

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.svm import LinearSVC

from sklearn.naive_bayes import MultinomialNB

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV,train_test_split,cross_val_score,KFold

from sklearn.metrics import fbeta_score, make_scorer





#Metrics for analysis

from sklearn.metrics import mean_squared_error,mean_absolute_error,f1_score,classification_report,confusion_matrix,accuracy_score
train = pd.read_csv('../input/climate-change-belief-analysis/train.csv')

test = pd.read_csv('../input/climate-change-belief-analysis/test.csv')
train.head()
test.head()
train.info()
test.info()
#bar graph plot to show the count for each sentiment class 

cnt_srs = train['sentiment'].value_counts()



plt.figure(figsize=(6,7))

sns.barplot(cnt_srs.index, cnt_srs.values, alpha=0.8,palette='GnBu_d')

plt.title('Number of tweets for each sentiment class', fontsize=14)

plt.ylabel('Number of Tweets', fontsize=12)

plt.xlabel('Sentiment Class', fontsize=12)

plt.ylabel("Sentiment Count")

plt.xticks([0,1,2,3],['Anti(-1)','Neutral(0)','Pro(1)','News(2)']);

plt.show()

#calc total number of emojis in the dataset and total number of emojis in each sentiment class.

no_emoji=train['message'].apply(lambda x: emoji.demojize(x)).apply(lambda x: len(re.findall(r':[a-z_&]+:',x))).sum()

no_emoji1=train['message'].apply(lambda x: emoji.demojize(x)).apply(lambda x: len(re.findall(r':[a-z_&]+:',x))).groupby(train['sentiment']).sum()

print("The total number of emojis in the dataset is : " + str(no_emoji)+"\n")

print("The no. of emojis in each sentiment :")

print(no_emoji1)
#find the total number of hashtags as well as total nuumber of hashtags for each sentiment class.

no_hash=train['message'].apply(lambda x: len(re.findall(r'#\w+',x))).sum()

no_hash1=train['message'].apply(lambda x: len(re.findall(r'#\w+',x))).groupby(train['sentiment']).sum()

print("The number of hashtags in the dataset is : " + str(no_hash)+"\n")

print("The no. of hashtags in each sentiment :")

print(no_hash1)
#calc the total number of urls in dataset and total number of urls for each sentiment class.

num_url=train['message'].apply(lambda x: (len(re.findall(r'http.?://[^\s]+[\s]?',x)))).sum()

num_url1=train['message'].apply(lambda x: (len(re.findall(r'http.?://[^\s]+[\s]?',x)))).groupby(train['sentiment']).sum()

print("The number of urls in the dataset is : " + str(num_url)+"\n")

print("The no. of urls in each sentiment :")

print(num_url1)
word_df= pd.DataFrame()

word_df['words'] = train['message'].apply(lambda x: len(re.findall(r'\w+',x)))

word_df['sentiment'] = train.sentiment

neg = word_df[word_df['sentiment']==-1]

neu = word_df[word_df['sentiment']==0]

pro = word_df[word_df['sentiment']==1]

news = word_df[word_df['sentiment']==2]
fig, axis = plt.subplots(ncols=2,nrows=2,figsize=(11,11))

fig.suptitle('Distribution of words in each sentiment class', fontsize=16)



axis[1,1].hist(neg['words'], bins=12,rwidth=0.95)

axis[1,1].set_title('Anti(-1)')

axis[1,1].set_xlabel('Word Count')

axis[1,1].set_ylabel('Frequency')



axis[1,0].hist(neu['words'], bins=12,rwidth=0.95)

axis[1,0].set_title('Neutral(0)')

axis[1,0].set_ylabel('Frequency')

axis[1,0].set_xlabel('Word Count')



axis[0,1].hist(pro['words'], bins=12,rwidth=0.95)

axis[0,1].set_title('Pro(1)')

axis[0,1].set_ylabel('Frequency')

axis[0,1].set_xlabel('Word Count')



axis[0,0].hist(news['words'], bins=12,rwidth=0.95)

axis[0,0].set_title('News(2)')

axis[0,0].set_ylabel('Frequency')

axis[0,0].set_xlabel('Word Count')



plt.show()
print('Average no. of words for each sentiment class:')

print('Anti(-1): '+str(round(neg['words'].mean(),2)))

print('Neutral(0): '+str(round(neu['words'].mean(),2)))

print('Pro(1): '+str(round(pro['words'].mean(),2)))

print('News(2): '+str(round(news['words'].mean(),2)))
def vader_score(df):

    """

    Computes the compund score of each tweet using Vader Analysis and converts the score

    into a string representing the sentiment which is then added to the tweet.

    

    Parameters

    ------------

    df: dataframe

        Takes in a dataframe

        

    Output

    ------------

    output: dataframe

        Returns a dataframe

    """

    analyser = SentimentIntensityAnalyzer()

    com_score=[]

    for sentence in train['message']:

        score = analyser.polarity_scores(sentence)['compound']

        if score >= 0.05:

            com_score.append(' Positive')

        elif score <= -0.05 : 

            com_score.append(' Negative')

        else:

            com_score.append(' Neutral')

    df['compound score'] = pd.DataFrame(com_score)

    return df



train = vader_score(train)

test = vader_score(test)
def correct_spelling(text):

    

    """

    This function takes a string of text as input. It corrects the spelling by applying textblob's correction

    method and returns the modified text string.

    """

    

    # instantiate TextBlob object

    blob = TextBlob(text)

    

    # correct spelling and return modified string

    return str(blob.correct())
def contn_replace(df):

    

    """

    Expand contraction words in a dataframe

    

    Parameters

    ------------

    df: dataframe

        Takes in a dataframe

        

    Output

    ------------

    output: dataframe

        Returns a dataframe

    """

    

    df['message'] = df['message'].str.replace(r'\'ve', ' have')

    df['message'] = df['message'].str.replace(r'\'ll', ' will')

    df['message'] = df['message'].str.replace(r'\'d', ' would')

    df['message'] = df['message'].str.replace(r'n\'t', ' not')

    df['message'] = df['message'].str.replace(r'\'s', ' is')

    df['message'] = df['message'].str.replace(r'\'m', ' am')

    df['message'] = df['message'].str.replace(r'\'re', ' are')

    

    return df



train=contn_replace(train)

test=contn_replace(test)
def url_replace(df):

    """

    Replace url links in a dataframe with the string 'url'

    

    Parameters

    ------------

    df: dataframe

        Takes in a dataframe

        

    Output

    ------------

    output: dataframe

        Returns a dataframe

    """

    df['message'] = df['message'].str.replace(r'http.?://[^\s]+[\s]?', 'url ')

    return df



train=url_replace(train)

test=url_replace(test)
def emoji_rep(df):

    """

    Replace emoticons with the word 'emoji'

    

    Parameters

    ------------

    df: dataframe

        Takes in a dataframe

        

    Output

    ------------

    output: dataframe

        Returns a dataframe

    """

    df['message']= df['message'].apply(lambda x: emoji.demojize(x)).apply(lambda x: re.sub(r':[a-z_&]+:','emoji ',x))

    return df



train=emoji_rep(train)

test=emoji_rep(test)
def noise_removal(df):

    """

    Removes noise such as special characters from text.

    

    Parameters

    ------------

    df: dataframe

        Takes in a dataframe

        

    Output

    ------------

    output: dataframe

        Returns a dataframe

    """

    

    df['message'] = df['message'].apply(lambda x: re.sub(r'[^\x00-\x7F]+','',x)) #removes unicode characters

    df['message'] = df['message'].str.replace(r"[',.():|-]", " ")                #removes some special characters

    df['message'] = df['message'].str.replace(r'^(RT|rt)( @\w*)?', '')           #removes twitter handles

    df['message'] = df['message'].apply(lambda x: re.sub(r'\d','',x))            #removes digits

    df['message'] = df['message'].apply(lambda x: re.sub('[^a-zA-z\s]','',x))    #removes punctuation

    df['message'] = df['message'].str.lower()                                    #converts text to lowercase

    return df



train=noise_removal(train)

test=noise_removal(test)
#initalize a tokenizer and lemmatizer object and apply it to on the cleaned tweets

from nltk.tokenize import TweetTokenizer

from nltk.stem import WordNetLemmatizer

get_tokens = TweetTokenizer()

get_lemmas = WordNetLemmatizer()



def tokenize_lemmatize(df):

    """

    Tokenize and Lemmatize the message column in the dataframe

    

    Parameters

    ------------

    df: dataframe

        Takes in a dataframe

        

    Output

    ------------

    output: dataframe

        Returns a dataframe

    """

    df['message'] = df.apply(lambda row: [get_lemmas.lemmatize(w) for w in get_tokens.tokenize(row['message'])], axis=1)

    

    return df



train=tokenize_lemmatize(train)

test=tokenize_lemmatize(test)
stopwords=[ 'i','me','my','myself','we','our','ours','ourselves','you','your','yours','yourself','yourselves','he','him','his','himself','she','her','hers','herself','it','itself','they','them','their','theirs','themselves','what','which','who','whom','this','that','these','those','am','is','are','was','were','be','been','being','have','has','had','having','do','does','did','doing','a','an','the','and','but','if','or','because','as','until','while','of','at','by','for','with','about','between','into','through','during','before','after','above','below','to','from','up','down','in','out','on','off','over','under','again','further','then','once','here','there','when','where','why','how','all','any','both','each','few','more','most','other','some','such','only','own','same','so','than','too','very','s','t','can','will','just','should','now','d','ll','m','o','re','ve','y','rt','u','doe','going','ha','wa','#','&','%','+','v','*','$','http',';','/']
#remove the stop words from our data

def remove_stopwords(df,stoplist):

    """

    Remove words contained in a list from a text column.

    

    Parameters

    ------------

    df: dataframe

        Takes in a dataframe

    stoplist : list   

        Takes in a list of words

        

    Output

    ------------

    output: dataframe

        Returns a dataframe

    """

    df['message']=df['message'].apply(lambda x: [item for item in x if item not in stoplist])

    return df



train=remove_stopwords(train,stopwords)

test=remove_stopwords(test,stopwords)
#join words in a list to make it a string line

train['message']=train['message'].apply(lambda x: ' '.join(str(v) for v in x ))

test['message']=test['message'].apply(lambda x: ' '.join(str(v) for v in x ))
#convert messages in df into one string for each sentiment 

anti_str =' '.join([line for line in train['message'][train['sentiment'] == -1]])

neu_str =' '.join([line for line in train['message'][train['sentiment'] == 0]])

pro_str =' '.join([line for line in train['message'][train['sentiment'] == 1]])

news_str =' '.join([line for line in train['message'][train['sentiment'] == 2]])
#Create wordcloud for each sentiment 

fig, axis = plt.subplots(nrows=2,ncols=2,figsize=(18,12))

anti_wc = WordCloud(width=900, height=600, background_color='black', colormap='summer').generate(anti_str)



axis[0,0].imshow(anti_wc)

axis[0,0].set_title('Anti(-1)',fontsize=16)

axis[0,0].axis("off") 



neu_wc = WordCloud(width=900, height=600, background_color='black', colormap='summer').generate(neu_str)

axis[0,1].imshow(neu_wc)

axis[0,1].set_title('Neutral(0)',fontsize=16)

axis[0,1].axis("off") 



pro_wc = WordCloud(width=900, height=600, background_color='black', colormap='summer').generate(pro_str)

axis[1,0].imshow(pro_wc)

axis[1,0].set_title('Pro(1)',fontsize=16)

axis[1,0].axis("off") 



news_wc = WordCloud(width=900, height=600, background_color='black', colormap='summer').generate(news_str)

axis[1,1].imshow(news_wc)

axis[1,1].set_title('News(2)',fontsize=16)

axis[1,1].axis("off") 



plt.show()
#select the columns that are going to be used to train the models

X = train['message']

y = train['sentiment']



y_df = test['message']
#split train dataset into train and test components

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#Creating our vectorizer and applying it to our data

vectorizer=TfidfVectorizer(min_df=1, max_df=0.9, stop_words='english', decode_error='ignore')

X_train=vectorizer.fit_transform(X_train)

X_test=vectorizer.transform(X_test)

X_val = vectorizer.fit_transform(y_df)
#Multinomial Naive Bayes

mnb = MultinomialNB()

mnb.fit(X_train,y_train)

prediction_mnb = mnb.predict(X_test)



print(classification_report(prediction_mnb,y_test))

print("Accuracy score: "+str(accuracy_score(prediction_mnb,y_test)))



#multinb_f1 = round(f1_score(y_val, NB_predictions, average='weighted'),2)

#Random Forest Classifier

rfc = RandomForestClassifier()

rfc.fit(X_train,y_train)

prediction_rfc = rfc.predict(X_test)



print(classification_report(prediction_rfc,y_test))

print("Accuracy score: "+str(accuracy_score(prediction_rfc,y_test)))
#Logistic regression

lr = LogisticRegression()

lr.fit(X_train,y_train)

prediction_lr = lr.predict(X_test)



print(classification_report(y_test,prediction_lr))

print("Accuracy score: "+str(accuracy_score(y_test,prediction_lr)))
# Linear SVC

lsvc = LinearSVC()

lsvc.fit(X_train,y_train)

prediction_lsvc = lsvc.predict(X_test)



print(classification_report(y_test,prediction_lsvc))

print("Accuracy score: "+str(accuracy_score(y_test,prediction_lsvc )))
#Gradient Boosting Classifier 

gbc = GradientBoostingClassifier(n_estimators=200,max_depth=6,random_state=10)

gbc.fit(X_train,y_train)

prediction_gbc = gbc.predict(X_test)



print(classification_report(y_test,prediction_gbc))

print("Accuracy score: "+str(accuracy_score(y_test,prediction_gbc)))
#K-Nearest Neighbours

knc = KNeighborsClassifier(n_neighbors = 10)

knc.fit(X_train,y_train)

prediction_knc = knc.predict(X_test)



print(classification_report(y_test,prediction_knc))

print("Accuracy score: "+str(accuracy_score(y_test,prediction_knc)))
#Tuning parameters for Linear Support Vector Classifier 

C_list = [0.1, 0.5, 0.75, 1, 5, 10, 25]

penalty_list = ['l1','l2']

loss =['hinge','squared_hinge']

   

scorer = make_scorer(f1_score, average = 'weighted')    

   

parameters = {'C':C_list,'penalty': penalty_list,'loss' : loss}

#fitting model with new parameters

tune = GridSearchCV(lsvc, parameters, scoring = scorer, cv=5, n_jobs=-1, verbose=3)

lsvc_tune = tune.fit(X_train,y_train)

prediction_lsvc_tune = lsvc_tune.predict(X_test)

#Displaying best parameters

print("Best parameters set:")

print(lsvc_tune.best_estimator_)

#showing summary statisitics 

print(classification_report(y_test,prediction_lsvc_tune))

print("Accuracy score: "+str(accuracy_score(y_test,prediction_lsvc_tune)))
#choosing parameters and it's values to test

C_list = [0.01, 0.1, 0.5, 0.75, 1, 5, 10, 25]

penalty_list = ['l1','l2']

random_state = ['random_state', 1, 10]

tol = ['tol', 1e-10, 1]

    

scorer = make_scorer(f1_score, average = 'weighted')

    

parameters = {'C':C_list,'penalty': penalty_list,

              'random_state' : random_state,'tol': tol}

#fitting model with new parameters

tune = GridSearchCV(lr, parameters, scoring = scorer,cv=5, n_jobs=-1, verbose=3)

lr_tune=tune.fit(X_train,y_train)

prediction_lr_tune = lr_tune.predict(X_test)
#Displaying best parameters

print("Best parameters set:")

print(lr_tune.best_estimator_)

#showing summary statisitics 

print(classification_report(y_test,prediction_lr_tune))

print("Accuracy score: "+str(accuracy_score(y_test,prediction_lr_tune)))

#Calculating our weighted f1 scores for each model tested

mnbf1 = round(f1_score(y_test,prediction_mnb,average='weighted'),3)

rfcf1 = round(f1_score(y_test,prediction_rfc,average='weighted'),3)

lrf1 = round(f1_score(y_test,prediction_lr,average='weighted'),3)

lsvcf1 = round(f1_score(y_test,prediction_lsvc,average='weighted'),3)

gbcf1 = round(f1_score(y_test,prediction_gbc,average='weighted'),3)

kncf1 = round(f1_score(y_test,prediction_knc,average='weighted'),3)
#Visualising and comparing the weighted f1 scores of the various models

plt.subplots(figsize=(12, 6))

x_var = ['Multinomial Naive Bayes','Random Forest Classifier','Logistic Regression','Linear SVC','Gradient Boosting','KN Neighbors Classifier']

y_var = [mnbf1,rfcf1,lrf1,lsvcf1,gbcf1,kncf1]

ax = sns.barplot(x=x_var,y=y_var,palette='GnBu_d')

plt.title('Weighted F1 score for each model',fontsize=14)

plt.xticks(rotation=90)

plt.xlabel('')

plt.ylabel('Weighted F1 score')

for a in ax.patches:

    ax.text(a.get_x() + a.get_width()/2, a.get_y() + a.get_height(),round(a.get_height(),3),fontsize=12, ha="center", va='bottom')

    

plt.show()
#calculating new f1 scores for the trained models

lsvcf1_tuned = round(f1_score(y_test,prediction_lsvc_tune,average='weighted'),3)

lrf1_tuned = round(f1_score(y_test,prediction_lr_tune,average='weighted'),3)
#Visualising new weighted F1 scores after tuning

x = [u'Linear SVC', u'Linear SVC tuned', u'Logistic Regression', u'Logistic Regression Tuned']

y = [lsvcf1, lsvcf1_tuned, lrf1, lrf1_tuned]



fig, ax = plt.subplots(figsize=(12, 5))    

width = 0.5 # the width of the bars 

ind = np.arange(len(y))  # the x locations for the groups

ax.barh(ind, y, width, color="skyblue")

ax.set_yticks(ind+width/2)

ax.set_yticklabels(x, minor=False)

plt.title('Comparison of weighted F1 scores before and after tuning')

plt.xlabel('Weighted F1 Score')

plt.ylabel('Models')

for i, v in enumerate(y):

    ax.text(v , i , str(v), color='blue', fontweight='bold')

plt.show()  
train1 = pd.read_csv('../input/climate-change-belief-analysis/train.csv')

test1 = pd.read_csv('../input/climate-change-belief-analysis/test.csv')



X1 = train1['message']

y1 = train1['sentiment']



y_df1 = test1['message']



vectorizer=TfidfVectorizer(min_df=1, max_df=0.9, stop_words='english', decode_error='ignore')

X1=vectorizer.fit_transform(X1)

X_val = vectorizer.transform(y_df1)



lr_best=LogisticRegression(C=5, random_state=1, tol=1e-10)

lr_best_model = lr_best.fit(X1,y1)

pred_lr_best = lr_best_model.predict(X_val)
my_submission = pd.DataFrame({'tweetid': test1['tweetid'], 'sentiment': pred_lr_best})

# you could use any filename. We choose submission here

my_submission.to_csv('lr_best.csv', index=False)