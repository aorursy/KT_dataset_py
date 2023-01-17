import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split

from catboost import Pool, cv, CatBoostRegressor,CatBoostClassifier

from sklearn.metrics import classification_report,mean_squared_error



import shap

shap.initjs()



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

columns = ['parent_id', 'subreddit_id', 'text', 'score',

#            'ups', 'parent_ups',  ## these appear to be redundnat with score, especially in this dataset (normally would allow info about upvotes * downvotes)

       'author', 'controversiality',  'parent_text',

       'parent_score', 'parent_author',

       'parent_controversiality']

# we'll use a sample of the data to speed things up

df_pos = pd.read_csv("../input/reddit-comment-score-prediction/comments_positive.csv",nrows=2e5, usecols=columns)

df_pos["binary_label"] = 1

df_neg = pd.read_csv("../input/reddit-comment-score-prediction/comments_negative.csv",nrows=2e5, usecols=columns)

df_neg["binary_label"] = 0



df = pd.concat([df_pos,df_neg], ignore_index=True).drop_duplicates(['text', 'parent_text','binary_label']).sample(frac=1) # concat and drop duplicates. We'll still have duplicate texts, but this should help clean  bot autoposts at least



df = df.loc[(~df.text.isna()) & (df['parent_text'].notnull())] # drop empty comments

del df_pos, df_neg



df
## feature to mark parent or post deleted (rather than getting it via proxy, since ["deleted"] is very common):

df["deleted"] = (df["parent_author"].str.contains("deleted",case=False) | df["author"].str.contains("deleted",case=False)).astype(int)



df["deleted"].sum()
df.nunique()
df["author"].value_counts().nunique()
(df["author"].value_counts()<2).sum() # majority of authors appear just once in the data. 
### we could replace only authors with less than 3 posts with their count, but there are so few that I won't bother. 

### https://stackoverflow.com/questions/37239627/change-values-in-pandas-dataframe-according-to-value-counts

## # df.where(df.apply(lambda x: x.map(x.value_counts()))>=2, "other")



## replace values with their counts/occurences #

df["author"] = df["author"].map(df["author"].value_counts())

df["parent_id"] = df["parent_id"].map(df["parent_id"].value_counts())

df["parent_author"] = df["parent_author"].map(df["parent_author"].value_counts())
# df["subreddit_id"].where(df["subreddit_id"].map(df["subreddit_id"].value_counts())<=5,df["subreddit_id"].map(df["subreddit_id"].value_counts()),df["subreddit_id"]) ## todo - fix for replacing rare subreddits (if any)
df["subreddit_counts"] = df["subreddit_id"].map(df["subreddit_id"].value_counts())

print("min subreddit occurence before filtering",df["subreddit_counts"].min())



df = df.loc[df["subreddit_counts"]>40 ] # drop posts from very rare subreddits (a minor fraction)
## Ideally we should combine these, but that means dropping duplicate parent posts then merging. This is good enough for now. 

df["subreddit_mean_post_score"] = df.groupby("subreddit_id")["score"].transform("mean")

df["subreddit_mean_post_parent_score"] = df.groupby("subreddit_id")["parent_score"].transform("mean")



# df["subreddit_mean_binary_label"] = 100*df.groupby("subreddit_id")["binary_label"].transform("mean")



## we could also normalize/z-score by sub-reddit as a target transforamtion!! 



# sum controversiality from post and parent post. silly but good enough for now

df["subreddit_mean_controversiality"] = df.groupby("subreddit_id")["controversiality"].transform("mean")

df["subreddit_mean_controversiality"] = 100*(df["subreddit_mean_controversiality"] + df.groupby("subreddit_id")["parent_controversiality"].transform("mean"))/2



df["subreddit_mean_deleted"] = 100*df.groupby("subreddit_id")["deleted"].transform("mean")
df
categorical_cols = ["subreddit_id"]

text_cols = [ 'text', 'parent_text',]



## columns to drop 

leak_cols = ['score', 'binary_label'] # I'd be suspicious of 'parent_score', but it can be left in, arguably



target_col = ['binary_label'] ###  'binary_label' if we want binary classification by dataset's authgor's prior ranking
X = df.drop(text_cols+leak_cols,axis=1)

y = df[target_col]



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=42)
train_pool = Pool(X_train,y_train,

                 cat_features=categorical_cols,

                 )





test_pool = Pool(X_test,y_test,

                 cat_features=categorical_cols,

                 )



model = CatBoostClassifier(iterations=150,

#                             task_type="GPU",

                           custom_metric=['Logloss',"Precision",

                                          'AUC'])



## we can accellerate training greatly if we run with GPU. 

### using early stopping with the test set and also evaluating on it is "cheating", but we don't care about exact numbers as much here. 

model.fit(train_pool,

          eval_set=test_pool,

          use_best_model=True,

          verbose=False,plot=True,

         ) # use less iterations to speed things up , especially when not running on GPU



print(model.get_best_score())
explainer = shap.TreeExplainer(model)

shap_values = explainer.shap_values(test_pool) # get explanations on test data. Could do so on the train data



# feature importance plot

shap.summary_plot(shap_values, X_test, plot_type="bar")



# summarize the effects of all the features

shap.summary_plot(shap_values, X_test)



### grey is missing values, although we are not missing any subreddit_ids. It's a bug in catboost/shap. #df.subreddit_id.isna().sum() ## =0
train_pool = Pool(X_train.drop(['parent_id','parent_score'],axis=1),y_train,

                 cat_features=categorical_cols)





test_pool = Pool(X_test.drop(['parent_id','parent_score'],axis=1),y_test,

                 cat_features=categorical_cols)



model2 = CatBoostClassifier(iterations=100,

                           custom_metric=['Logloss',"Precision",

                                          'AUC'])

model2.fit(train_pool,

          eval_set=test_pool,

          use_best_model=True,

          verbose=False,plot=True,

         ) 



print(model2.get_best_score())



explainer = shap.TreeExplainer(model2)

shap_values = explainer.shap_values(test_pool) # get explanations on test data. Could do so on the train data



# feature importance plot

shap.summary_plot(shap_values, X_test.drop(['parent_id','parent_score'],axis=1), plot_type="bar")



# summarize the effects of all the features

shap.summary_plot(shap_values, X_test.drop(['parent_id','parent_score'],axis=1))

categorical_cols = ["subreddit_id"]

text_cols = [ 'text']#[ 'text', 'parent_text',] ## we won't featurize the parent text for now. overkill (and can let leaks thorough, in the form of popular/upvoted parent topics)



## columns to drop 

drop_cols = ['score', 'binary_label','parent_score', 'parent_text'] # I'd be suspicious of , but it can be left in, arguably



import string

from string import punctuation

from nltk import pos_tag

from nltk.corpus import stopwords

from nltk.tokenize import TweetTokenizer

stop_words = set(stopwords.words('english')) 



# !pip install TextBlob

from textblob import TextBlob



# functions to get polatiy and subjectivity of text using the module textblob

def get_polarity(text):

    try:

        textblob = TextBlob(unicode(text, 'utf-8'))

        pol = textblob.sentiment.polarity

    except:

        pol = 0.0

    return pol



def get_subjectivity(text):

    try:

        textblob = TextBlob(unicode(text, 'utf-8'))

        subj = textblob.sentiment.subjectivity

    except:

        subj = 0.0

    return subj





def tag_part_of_speech(text):

    text_splited = text.split(' ')

    text_splited = [''.join(c for c in s if c not in string.punctuation) for s in text_splited]

    text_splited = [s for s in text_splited if s]

    pos_list = pos_tag(text_splited)

    noun_count = len([w for w in pos_list if w[1] in ('NN','NNP','NNPS','NNS')])

    adjective_count = len([w for w in pos_list if w[1] in ('JJ','JJR','JJS')])

    verb_count = len([w for w in pos_list if w[1] in ('VB','VBD','VBG','VBN','VBP','VBZ')])

    return[noun_count, adjective_count, verb_count]



def text_features(df:pd.DataFrame,text:str="text", get_pos_feats=False,get_textblob_sentiment=True) -> pd.DataFrame:

    """

    Extract and add in place many text/NLP features on a pandas dataframe for a given column.

    Functions are generic, but some were basedon  :  https://www.kaggle.com/shivamb/extensive-text-data-feature-engineering , 

    https://www.kaggle.com/shaz13/feature-engineering-for-nlp-classification

    I modified to use vectorized functions - many, many times faster, Can be optimized further easily.

    """





    # https://www.kaggle.com/shaz13/feature-engineering-for-nlp-classification

    df[f'{text}_char_count'] = df[text].str.len()

    df[f'{text}_num_words'] = df[text].str.split().str.len()



    df['capitals'] = df[text].apply(lambda comment: sum(1 for c in comment if c.isupper()))

    df['caps_vs_length'] = df.apply(lambda row: float(row['capitals'])/float(row[f'{text}_char_count']),axis=1)

    df['num_exclamation_marks'] = df[text].str.count('!')

    df['num_question_marks'] = df[text].str.count('\?')

    df['num_punctuation'] = df[text].apply(lambda comment: sum(comment.count(w) for w in '.,;:'))

    df['num_symbols'] = df[text].apply(lambda comment: sum(comment.count(w) for w in r'*&$%/:;'))



    df['num_unique_words'] = df[text].apply(lambda comment: len(set(w for w in comment.split())))

    df['words_vs_unique'] = df['num_unique_words'] / df[f'{text}_num_words']

    df['num_smilies'] = df[text].apply(lambda comment: sum(comment.count(w) for w in (':-)', ':)', ';-)', ';)')))

    df['num_sad'] = df[text].apply(lambda comment: sum(comment.count(w) for w in (':-<', ':()', ';-()', ';(')))



#     df['char_count'] = df['text'].apply(len)

#     df['word_count'] = df['text'].apply(lambda x: len(x.split()))

    df['word_density'] = df[f'{text}_char_count'] / (df[f'{text}_num_words']+1)

    df['punctuation_count'] = df['text'].apply(lambda x: len("".join(_ for _ in x if _ in punctuation))) 



    df['upper_case_word_count'] = df[text].apply(lambda x: len([wrd for wrd in x.split() if wrd.isupper()]))

    df['stopword_count'] = df[text].apply(lambda x: len([wrd for wrd in x.split() if wrd.lower() in stop_words]))

#     df['title_word_count'] = df['text'].apply(lambda x: len([wrd for wrd in x.split() if wrd.istitle()]))

    df["count_words_title"] = df[text].apply(lambda x: len([w for w in str(x).split() if w.istitle()]))

    df["mean_word_len"] = df[text].apply(lambda x: np.mean([len(w) for w in str(x).split()]))

    df['punct_percent']= df['num_punctuation']*100/df[f'{text}_num_words']

    

    if get_textblob_sentiment:

        df['polarity'] = df[text].apply(get_polarity)

        df['subjectivity'] = df[text].apply(get_subjectivity)

    

    if get_pos_feats:

        df['nouns'], df['adjectives'], df['verbs'] = zip(*df[text].apply(

            lambda comment: tag_part_of_speech(comment)))

        df['nouns_vs_length'] = df['nouns'] / df[f'{text}_char_count']

        df['adjectives_vs_length'] = df['adjectives'] / df[f'{text}_char_count']

        df['verbs_vs_length'] = df['verbs'] /df[f'{text}_char_count']

        df['nouns_vs_words'] = df['nouns'] / df[f'{text}_num_words']

        df['adjectives_vs_words'] = df['adjectives'] / df[f'{text}_num_words']

        df['verbs_vs_words'] = df['verbs'] / df[f'{text}_num_words']



        df.drop(['nouns','adjectives','verbs'],axis=1,inplace=True) # drop the count of POS, keep only the percentages. Can change to keep them..



        

    df["ends_on_alphanumeric"] = df[text].str.strip().str[-1].str.isalpha() # does word end on alphanumeric, vs ".". Interesting for comments. Note the strip. 

        

    return df
%%time

df = text_features(df, get_pos_feats=False)



df.tail()
X = df.drop(drop_cols,axis=1)

y = df[target_col]



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=42)
train_pool = Pool(X_train,y_train,

                 cat_features=categorical_cols,

                  text_features=text_cols

                 )



test_pool = Pool(X_test,y_test,

                 cat_features=categorical_cols,

                 text_features=text_cols

                 )



## catboost text featurizer params (e.g. tokenizer, n-grams, BoW..) - https://catboost.ai/docs/features/text-features.html



model = CatBoostClassifier(iterations=350,custom_metric=['Logloss',"Precision", 'AUC'])



model.fit(train_pool,

          eval_set=test_pool,

          use_best_model=True,

          verbose=False,plot=True,

         )



print(model.get_best_score())
explainer = shap.TreeExplainer(model)

shap_values = explainer.shap_values(test_pool) # get explanations on test data



# feature importance plot

shap.summary_plot(shap_values, X_test, plot_type="bar")



# summarize the effects of all the features

shap.summary_plot(shap_values, X_test)
display(df.score.describe())

df.score.hist();
display(df["subreddit_mean_post_score"].describe())

df["subreddit_mean_post_score"].hist();
## what are subreddits with a mean negative score?



df.loc[df["subreddit_mean_post_score"]<-20].drop_duplicates(["parent_id","subreddit_id"])