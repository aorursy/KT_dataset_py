import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import string, re
sns.set()

modelling = False

subsampling = False # For speed up the iteration of development
#Check the dataset sizes(in MB)

!du -lh ../input/emoji-sentiment-data/*
# Multiple output printing in each cell

from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"
if not modelling:

    full_comments_df = pd.read_csv("../input/commentscsv/test_data.csv")
if not modelling:

    emoji_sent_df = pd.read_csv("../input/emoji-sentiment-data/Emoji_Sentiment_Data_v1.0.csv")
if not modelling:

    full_comments_df.shape
if not modelling:

    if subsampling:

        full_comments_df = full_comments_df.sample(5000, random_state=1234).reset_index()

        full_comments_df.shape
if not modelling:

    full_comments_df.head()
if not modelling:

    full_comments_df.tail()
if not modelling:

    full_comments_df.drop_duplicates(inplace=True)

    full_comments_df.shape
if not modelling:

    full_comments_df.isnull().sum(axis = 0)
if not modelling:

    full_comments_df[full_comments_df['translation'].isnull()]['language']
if not modelling:

    # Just copy the original text into the translation for the ones with empty translation field

    full_comments_df['translation'].fillna(full_comments_df['comment'], inplace = True)

    full_comments_df.isnull().sum(axis = 0)

if not modelling:

    f'[{string.punctuation}]' # Just to check what symbols are included

    full_comments_df['translation']=full_comments_df['translation'].str.replace(f'[{string.punctuation}]','') # Removing punctuation symbols (Maybe exclamations of questions marks have any impact?)

    full_comments_df['translation']=full_comments_df['translation'].str.replace('[‘’“”…]', '') # Removing more potential weird charactersa
if not modelling:

    full_comments_df['translation'].str.len().mean()

    full_comments_df['translation'].str.len().std()

    full_comments_df['translation'].str.len().quantile(q = [0,0.25,0.5,0.75,0.9,0.95,0.99,0.999,0.9999,1])
if not modelling:

    full_comments_df = full_comments_df[full_comments_df['translation'].str.len() < 500]

    full_comments_df.shape
if not modelling:

    sns.countplot(x = 'harmful', data = full_comments_df)
if not modelling:

    sns.countplot(x = 'harmful_type', data = full_comments_df[full_comments_df['harmful_type'] != 0])
if not modelling:

    full_comments_df['contains_numbers'] = full_comments_df['translation'].str.contains('\s\d+\s',regex=True) # Pure number words

    full_comments_df['translation']=full_comments_df['translation'].str.replace('[0-9]',' ',regex=True) # Remove all numbers (even the ones between letters)

    full_comments_df['contains_numbers'].mean()
if not modelling:

    pd.crosstab(full_comments_df['contains_numbers'],full_comments_df['harmful'])
if not modelling:

    #def perc_capital_letters(text): return sum(1 for c in text if c.isupper())/len(text)

    full_comments_df['perc_capital'] = full_comments_df['translation'].apply(lambda text: sum(1 for c in text if c.isupper())/(len(text)+1))
if not modelling:

    full_comments_df['perc_capital'].describe()
if not modelling:

    sns.distplot(full_comments_df[(full_comments_df['harmful'] == 0)]['perc_capital'].dropna(),label='No harm')

    sns.distplot(full_comments_df[(full_comments_df['harmful'] == 1)]['perc_capital'].dropna(),label='Harm')

    plt.legend()

    plt.xlim([0,0.4])

    plt.show()
if not modelling:

    full_comments_df['translation'] = full_comments_df['translation'].str.lower().str.strip() # Lower case and striping (Maybe capital letters have any predictive impact?f)
if not modelling:

    full_comments_df['translation']
if not modelling:

    full_comments_df['char_length'] = full_comments_df['translation'].str.len()

    full_comments_df['char_length_norm'] = full_comments_df['char_length']/full_comments_df['char_length'].max()
if not modelling:

    plot1 = sns.distplot(full_comments_df[(full_comments_df['harmful'] == 0)]['char_length'].dropna(),label='No harm')

    plot2 = sns.distplot(full_comments_df[(full_comments_df['harmful'] == 1)]['char_length'].dropna(),label='Harm')

    #plot1.set(xscale="log")

    #plot2.set(xscale="log")

    plt.legend()

    plt.show()


if not modelling:

    with sns.axes_style('white'):

        g = sns.jointplot("perc_capital", "char_length", full_comments_df, kind='kde')

        #g.ax_joint.set_xscale('log')

        #g.ax_joint.set_yscale('log')

        g.ax_joint.set_xlim([0,0.2]) 

        g.ax_joint.set_ylim([0,70])

if not modelling:

    with sns.axes_style('white'):

        g = sns.jointplot("perc_capital", "char_length", full_comments_df[full_comments_df['harmful'] == 0], kind='kde')

        #g.ax_joint.set_xscale('log')

        #g.ax_joint.set_yscale('log')

        g.ax_joint.set_xlim([0,0.2]) 

        g.ax_joint.set_ylim([0,70])

if not modelling:

    with sns.axes_style('white'):

        g = sns.jointplot("perc_capital", "char_length", full_comments_df[full_comments_df['harmful'] == 1], kind='kde')

        #g.ax_joint.set_xscale('log')

        #g.ax_joint.set_yscale('log')

        g.ax_joint.set_xlim([0,0.2]) 

        g.ax_joint.set_ylim([0,70])

if not modelling:

    c = 5

    print('Harmful comment:')

    print(full_comments_df[full_comments_df['harmful'] == 1].reset_index(drop=True)['translation'][c])

    print('Non-Harmful comment:')

    print(full_comments_df[full_comments_df['harmful'] == 0].reset_index(drop=True)['translation'][c])
if not modelling:

    emoji_sent_df.head()

    emoji_sent_df.dtypes
if not modelling:

    def emoji_sent(text):

        negative = 0

        neutral = 0

        positive = 0

        position = -1

        has = 0

        total = 0

        for emoji in emoji_sent_df['Emoji']:

            n_times = text.count(emoji)

            if n_times > 0:

                has += n_times

                position += n_times*emoji_sent_df[emoji_sent_df['Emoji']==emoji]['Position'].values[0]

                negative += n_times*emoji_sent_df[emoji_sent_df['Emoji']==emoji]['Negative'].values[0]/10000

                neutral += n_times*emoji_sent_df[emoji_sent_df['Emoji']==emoji]['Neutral'].values[0]/10000

                positive += n_times*emoji_sent_df[emoji_sent_df['Emoji']==emoji]['Positive'].values[0]/10000

                total += n_times

        if total != 0:

            negative = negative/total

            neutral = neutral/total

            positive = positive/total

            position = (position +1)/total

        return pd.Series([has,negative,neutral,positive,position])

    

    full_comments_df[['has_emoji','emoji_neg_sent','emoji_neut_sent','emoji_pos_sent','emoji_position']]=full_comments_df['translation'].apply(emoji_sent)
if not modelling:

    full_comments_df.head(300)
if not modelling:

    pd.crosstab(full_comments_df['has_emoji'] > 0,full_comments_df['harmful'])
if not modelling:

    #plt.rcParams['figure.figsize'] = [16, 6]

    emoji_columns = ['emoji_neg_sent','emoji_neut_sent','emoji_pos_sent']

    for emoji_col  in emoji_columns:

        plot1 = sns.distplot(full_comments_df[(full_comments_df['harmful'] == 0) & (full_comments_df['has_emoji'] > 0)][emoji_col].dropna(),label='No harm')

        plot2 = sns.distplot(full_comments_df[(full_comments_df['harmful'] == 1) & (full_comments_df['has_emoji'] > 0)][emoji_col].dropna(),label='Harm')

        #plot1.set(xscale="log")

        #plot2.set(xscale="log")

        plt.legend()

        plt.show()
if not modelling:

    from textblob import TextBlob

    polarity = lambda x: TextBlob(x).sentiment.polarity

    subjectivity = lambda x: TextBlob(x).sentiment.subjectivity

    full_comments_df['polarity']=full_comments_df['translation'].apply(polarity)

    full_comments_df['subjectivity']=full_comments_df['translation'].apply(subjectivity)

if not modelling:

    sentiment_columns = ['polarity','subjectivity']

    for sent_col  in sentiment_columns:

        plot1 = sns.distplot(full_comments_df[(full_comments_df['harmful'] == 0)][sent_col].dropna(),label='No harm', kde = False)

        plot2 = sns.distplot(full_comments_df[(full_comments_df['harmful'] == 1)][sent_col].dropna(),label='Harm', kde = False)

        #plot1.set(xscale="log")

        #plot2.set(xscale="log")

        plt.legend()

        plt.show()
from wordcloud import WordCloud

from sklearn.feature_extraction import text 



if not modelling:

    stop_words = text.ENGLISH_STOP_WORDS



    # Set two wordclouds images,one per class

    plt.rcParams['figure.figsize'] = [16, 6]

    wc = WordCloud(stopwords=stop_words, background_color="white", colormap="Dark2",max_font_size=150, random_state=12457)



    wc.generate(full_comments_df[full_comments_df['harmful']==0]['translation'].str.cat(sep=' '))

    plt.subplot(1,2,1)

    plt.imshow(wc,interpolation='bilinear')

    plt.axis("off")

    plt.title('Non Harmful')

    wc.generate(full_comments_df[full_comments_df['harmful']==1]['translation'].str.cat(sep=' '))

    plt.subplot(1,2,2)

    plt.imshow(wc,interpolation='bilinear')

    plt.axis("off")

    plt.title('Harmful')



    plt.show()
from sklearn.model_selection import train_test_split

import os, pickle



exists = os.path.isfile('train_df.pkl') # Check if there is one of the supposed stored files

if exists and modelling:

    train_df = pd.read_pickle('train_df.pkl')

    dev_df = pd.read_pickle('dev_df.pkl')

    test_df = pd.read_pickle('test_df.pkl')   

else:

    train_df, dev_test_df = train_test_split(full_comments_df, test_size=0.25, random_state = 4321) # 70% for training

    dev_df, test_df = train_test_split(dev_test_df, test_size=0.4, random_state = 6789) # 15% for developping and 10% for final testing

       

    pickle.dump(train_df, open('train_df.pkl', 'wb'))

    pickle.dump(dev_df, open('dev_df.pkl', 'wb'))

    pickle.dump(test_df, open('test_df.pkl', 'wb'))
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfVectorizer



# COUNTs

count_vec = CountVectorizer(stop_words=stop_words,strip_accents = 'unicode',ngram_range = [1,1], min_df = 3)



train_dtm = count_vec.fit_transform(train_df['translation'])

dev_dtm = count_vec.transform(dev_df['translation'])

test_dtm = count_vec.transform(test_df['translation'])



train_dtm_df = pd.DataFrame(train_dtm.toarray(), columns=count_vec.get_feature_names())

dev_dtm_df = pd.DataFrame(dev_dtm.toarray(), columns=count_vec.get_feature_names())

test_dtm_df = pd.DataFrame(test_dtm.toarray(), columns=count_vec.get_feature_names())



train_dtm_df.index = train_df.index

dev_dtm_df.index = dev_df.index

test_dtm_df.index = test_df.index



train_dtm_df = pd.concat([train_df,train_dtm_df], axis = 1).reset_index()

dev_dtm_df = pd.concat([dev_df,dev_dtm_df], axis = 1).reset_index()

test_dtm_df = pd.concat([test_df,test_dtm_df], axis = 1).reset_index()



#TF-IDF

count_idf_vec = TfidfVectorizer(stop_words=stop_words,strip_accents = 'unicode',ngram_range = [1,1], min_df = 3)



train_dtm_idf = count_idf_vec.fit_transform(train_df['translation'])

dev_dtm_idf = count_idf_vec.transform(dev_df['translation'])

test_dtm_idf = count_idf_vec.transform(test_df['translation'])



train_dtm_idf_df = pd.DataFrame(train_dtm_idf.toarray(), columns=count_idf_vec.get_feature_names())

dev_dtm_idf_df = pd.DataFrame(dev_dtm_idf.toarray(), columns=count_idf_vec.get_feature_names())

test_dtm_idf_df = pd.DataFrame(test_dtm_idf.toarray(), columns=count_idf_vec.get_feature_names())



train_dtm_idf_df.index = train_df.index

dev_dtm_idf_df.index = dev_df.index

test_dtm_idf_df.index = test_df.index



train_dtm_idf_df = pd.concat([train_df,train_dtm_idf_df], axis = 1).reset_index()

dev_dtm_idf_df = pd.concat([dev_df,dev_dtm_idf_df], axis = 1).reset_index()

test_dtm_idf_df = pd.concat([test_df,test_dtm_idf_df], axis = 1).reset_index()
harmful_words_serie = train_dtm_df[train_dtm_df['harmful']==0].sum(numeric_only = True).iloc[18:] # The index plus the other non proper word features are skipped

non_harmful_words_serie = train_dtm_df[train_dtm_df['harmful']==1].sum(numeric_only = True).iloc[18:] # The index plus the other non proper word features are skipped

harm_top_serie = harmful_words_serie.sort_values(ascending=False).head(30)

non_harm_top_serie = non_harmful_words_serie.sort_values(ascending=False).head(30)



harm_top_words = set(harm_top_serie.keys())

non_harm_top_words = set(non_harm_top_serie.keys())

add_stop_words = harm_top_words.intersection(non_harm_top_words)

harm_top_serie

non_harm_top_serie



add_stop_words

   
# Computing the DTM_DFs (count and tf-idf representations) again adding this new stopwords



stop_words_new = text.ENGLISH_STOP_WORDS.union(add_stop_words)



# COUNTs

count_vec = CountVectorizer(stop_words=stop_words_new,strip_accents = 'unicode',ngram_range = [1,1], min_df = 3)



train_dtm = count_vec.fit_transform(train_df['translation'])

dev_dtm = count_vec.transform(dev_df['translation']) # Check that dev and test are only applying the transformation did on the train set

test_dtm = count_vec.transform(test_df['translation'])



train_dtm_df = pd.DataFrame(train_dtm.toarray(), columns=count_vec.get_feature_names())

dev_dtm_df = pd.DataFrame(dev_dtm.toarray(), columns=count_vec.get_feature_names())

test_dtm_df = pd.DataFrame(test_dtm.toarray(), columns=count_vec.get_feature_names())



train_dtm_df.index = train_df.index

dev_dtm_df.index = dev_df.index

test_dtm_df.index = test_df.index



train_dtm_df = pd.concat([train_df,train_dtm_df], axis = 1).reset_index()

dev_dtm_df = pd.concat([dev_df,dev_dtm_df], axis = 1).reset_index()

test_dtm_df = pd.concat([test_df,test_dtm_df], axis = 1).reset_index()   



#TF-IDF

count_idf_vec = TfidfVectorizer(stop_words=stop_words_new,strip_accents = 'unicode',ngram_range = [1,1], min_df = 3)



train_dtm_idf = count_idf_vec.fit_transform(train_df['translation'])

dev_dtm_idf = count_idf_vec.transform(dev_df['translation'])

test_dtm_idf = count_idf_vec.transform(test_df['translation'])



train_dtm_idf_df = pd.DataFrame(train_dtm_idf.toarray(), columns=count_idf_vec.get_feature_names())

dev_dtm_idf_df = pd.DataFrame(dev_dtm_idf.toarray(), columns=count_idf_vec.get_feature_names())

test_dtm_idf_df = pd.DataFrame(test_dtm_idf.toarray(), columns=count_idf_vec.get_feature_names())



train_dtm_idf_df.index = train_df.index

dev_dtm_idf_df.index = dev_df.index

test_dtm_idf_df.index = test_df.index



train_dtm_idf_df = pd.concat([train_df,train_dtm_idf_df], axis = 1).reset_index()

dev_dtm_idf_df = pd.concat([dev_df,dev_dtm_idf_df], axis = 1).reset_index()

test_dtm_idf_df = pd.concat([test_df,test_dtm_idf_df], axis = 1).reset_index()    
print(train_dtm.sum(0))
# First compute ad-hoc function the 

def words_prob_vector(class_0_1):

    p = X[y==class_0_1].sum(0)

    return (p+1) / ((y==class_0_1).sum()+1)



def nb_prediction(dtm_matrix,w,b):

    predictions = (dtm_matrix @ w +b)>0

    return predictions
X = train_dtm.toarray()

y = train_df['harmful']

w_nb = np.log(words_prob_vector(1)/words_prob_vector(0)) # This weights in NB are learn based on "first" principles, assuming the "naive" of conditional independence given a comment is harmful or not

bias = np.log((y ==1).mean()/(y==0).mean())

# Now given a comment, the prediction is = (term_representation*wb + b) and assuming a threshold of 50% of harmful/non-harmful if this is greater than 0 => Class 1 else 0

train_error=(nb_prediction(train_dtm.toarray(),w_nb,bias)==train_df['harmful']).mean()

dev_error=(nb_prediction(dev_dtm.toarray(),w_nb,bias)==dev_df['harmful']).mean()

print(f'Naive Bayes ==> Train_Accuracy:{train_error} - Dev_Accuracy:{dev_error}')
X = train_dtm.sign().toarray() # All entries will be 0 or 1

y = train_df['harmful']

w_nb = np.log(words_prob_vector(1)/words_prob_vector(0)) # This weights in NB are learn based on "first" principles, assuming the "naive" of conditional independence given a comment is harmful or not

bias = np.log((y ==1).mean()/(y==0).mean())

# Now given a comment, the prediction is = (term_representation*wb + b) and assuming a threshold of 50% of harmful/non-harmful if this is greater than 0 => Class 1 else 0

train_error=(nb_prediction(train_dtm.sign().toarray(),w_nb,bias)==train_df['harmful']).mean()

dev_error=(nb_prediction(dev_dtm.sign().toarray(),w_nb,bias)==dev_df['harmful']).mean()

print(f'Naive Bayes binary input ==> Train_Accuracy:{train_error} - Dev_Accuracy:{dev_error}')
from sklearn.linear_model import LogisticRegression

m = LogisticRegression(C=1e8, dual=True) # C is the inverse of the reg coefficient, bigger values imply less regularization (this is like no regularization)

m.fit(train_dtm.toarray(), train_df['harmful']) # It is already loaded the train values from the NB model

train_error = (m.predict(train_dtm.toarray())==train_df['harmful']).mean()

dev_error = (m.predict(dev_dtm.toarray())==dev_df['harmful']).mean()

print(f'Logistic Regression ==> Train_Accuracy:{train_error} - Dev_Accuracy:{dev_error}')
from sklearn.linear_model import LogisticRegression

m = LogisticRegression(C=0.5, dual=True)

m.fit(train_dtm.toarray(), train_df['harmful'])

train_error = (m.predict(train_dtm.toarray())==train_df['harmful']).mean()

dev_error = (m.predict(dev_dtm.toarray())==dev_df['harmful']).mean()

print(f'Logistic Regression ==> Train_Accuracy:{train_error} - Dev_Accuracy:{dev_error}')
m = LogisticRegression(C=0.5, dual=True)

m.fit(train_dtm.sign().toarray(), train_df['harmful'])

train_error = (m.predict(train_dtm.sign().toarray())==train_df['harmful']).mean()

dev_error = (m.predict(dev_dtm.sign().toarray())==dev_df['harmful']).mean()

print(f'Logistic Regression binary input ==> Train_Accuracy:{train_error} - Dev_Accuracy:{dev_error}')
m = LogisticRegression(C=0.5, dual=True)

X = train_dtm.toarray() 

y = train_df['harmful']

w_nb = np.log(words_prob_vector(1)/words_prob_vector(0)).reshape([1,train_dtm.toarray().shape[1]])

X_new = X*w_nb

m.fit(X_new, train_df['harmful'])

train_error = (m.predict(train_dtm.toarray()*w_nb)==train_df['harmful']).mean()

dev_error = (m.predict(dev_dtm.toarray()*w_nb)==dev_df['harmful']).mean()

print(f'NB_LR ==> Train_Accuracy:{train_error} - Dev_Accuracy:{dev_error}')
from sklearn.ensemble import RandomForestClassifier



target = 'harmful'

non_word_features = ['contains_numbers', 'perc_capital', 'char_length','char_length_norm', 'has_emoji',

                     'emoji_neg_sent', 'emoji_neut_sent','emoji_pos_sent', 'emoji_position', 'polarity', 'subjectivity']



X_train_rf = train_df[non_word_features]

y_train = train_df['harmful']

X_dev_rf = dev_df[non_word_features]

y_dev = dev_df['harmful']

import sklearn

from sklearn.metrics import recall_score



def print_score(m):

    if isinstance(m, sklearn.ensemble.forest.RandomForestClassifier):

        res = [recall_score(m.predict(X_train_rf), y_train), recall_score(m.predict(X_dev_rf), y_dev),m.score(X_train_rf, y_train), m.score(X_dev_rf, y_dev)]

        if hasattr(m, 'oob_score_'): 

            res.append(m.oob_score_)

            out_text = f'Recall_Train:{res[0]} - Recall_Dev:{res[1]} - Accuracy_train::{res[2]} - Accuracy_Dev:{res[3]} - OOB:{res[4]}'

        else:

            out_text = f'Recall_Train:{res[0]} - Recall_Dev:{res[1]} - Accuracy_train::{res[2]} - Accuracy_Dev:{res[3]}'

            

    print(out_text)
m = RandomForestClassifier(n_estimators=20, n_jobs=-1)

m.fit(X_train_rf, y_train)

print_score(m)
m = RandomForestClassifier(n_estimators=40, n_jobs=-1)

m.fit(X_train_rf, y_train)

print_score(m)
m = RandomForestClassifier(n_estimators=80, n_jobs=-1)

m.fit(X_train_rf, y_train)

print_score(m)
m = RandomForestClassifier(n_estimators=40, n_jobs=-1, oob_score=True)

m.fit(X_train_rf, y_train)

print_score(m)
m = RandomForestClassifier(n_estimators=40, n_jobs=-1, oob_score=True, min_samples_leaf= 3)

m.fit(X_train_rf, y_train)

print_score(m)
m = RandomForestClassifier(n_estimators=40, n_jobs=-1, oob_score=True, min_samples_leaf= 5)

m.fit(X_train_rf, y_train)

print_score(m)
m = RandomForestClassifier(n_estimators=40, n_jobs=-1, oob_score=True, min_samples_leaf= 10)

m.fit(X_train_rf, y_train)

print_score(m)
m = RandomForestClassifier(n_estimators=40, n_jobs=-1, oob_score=True, min_samples_leaf= 25)

m.fit(X_train_rf, y_train)

print_score(m)
m = RandomForestClassifier(n_estimators=40, n_jobs=-1, oob_score=True, min_samples_leaf= 100)

m.fit(X_train_rf, y_train)

print_score(m)
m = RandomForestClassifier(n_estimators=40, n_jobs=-1, oob_score=True, min_samples_leaf= 15, max_features=0.5)

m.fit(X_train_rf, y_train)

print_score(m)
m = RandomForestClassifier(n_estimators=40, n_jobs=-1, oob_score=True, min_samples_leaf= 15, max_features='sqrt')

m.fit(X_train_rf, y_train)

print_score(m)
m = RandomForestClassifier(n_estimators=40, n_jobs=-1, oob_score=True, min_samples_leaf= 15, max_features='log2')

m.fit(X_train_rf, y_train)

print_score(m)
fi = m.feature_importances_

fi

X_train_rf.shape

X_train_rf = X_train_rf.loc[:,fi > 0.02] # Let's remove the less important ones

X_dev_rf = X_dev_rf.loc[:,fi > 0.02] # Let's remove the less important ones

m = RandomForestClassifier(n_estimators=40, n_jobs=-1, oob_score=True, min_samples_leaf= 15, max_features='log2')

m.fit(X_train_rf, y_train)

print_score(m)