# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
from sklearn.linear_model import PassiveAggressiveClassifier, RidgeClassifier

from sklearn.svm import LinearSVC

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.preprocessing import RobustScaler

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.metrics import jaccard_similarity_score, make_scorer, brier_score_loss, confusion_matrix



import nltk

import re

from itertools import combinations

from collections import defaultdict, namedtuple

import seaborn as sns
## Load Datasets



df_kim = pd.read_csv("../input/KimKardashianTweets.csv")

df_neil = pd.read_csv("../input/NeildeGrasseTysonTweets.csv")

df_fte = pd.read_csv("../input/FiveThirtyEightTweets.csv")

df_adam = pd.read_csv("../input/AdamSavageTweets.csv")
####### Regular expressions for removal

at = r'@'

hashtag = r'#'

bitly = r'bit\.ly.*\s?'

instagram = r'instagr\.am.*\s?'

url = r'https?:.*\s?'

tweeturl = r't\.co.*\s?'

pic = r'pic\.twitter\.com.+\s?'
def munger(data):

    for index, row in data.iterrows():

        text = row['text']

        text = re.sub("@","",text)

        text = re.sub("#","",text)

        text = re.sub("bit\.ly.*\s?","",text)

        text = re.sub("instagr\.am.*\s?","",text)

        text = re.sub("https?:.*\s?","",text)

        text = re.sub("t\.co.*\s?","",text)

        text = re.sub("pic\.twitter\.com\S*\s?","",text)

        #### set_value is considered the new preferred way of setting values

        #### It is also extremely fast when used with iterrows()

        data.set_value(index,"text",text)

   

    return data
### Due to memory and CPU limits, it's often required to sample the dataframes and run in batches rather than

##### run a single model with all the tweets at once. This will come in handy when working with all of NASA's tweets.

def sample_dfs(num_of_samples,num_of_dfs,df,random_state):

    sampled_dfs = []

    for i in range(num_of_dfs):

        sampled_df = df.sample(num_of_samples,random_state=random_state*i)

        munged_df = munger(sampled_df)

        sampled_dfs.append(munged_df)

    return sampled_dfs



def munge_dfs(df):

    hashtag = re.compile(r"#")

    at = re.compile(r"\.?@.+\s?")

    for index, row in df.iterrows():

        text = row["text"]

        text = hashtag.sub("",text)

        text = at.sub("",text)

        df.set_value(index,"text",text)
### Scorers since Ridge, Linear SVC, and Passive Aggressive do not have predict_proba

def score_decision_function_model(X_test, y_test,model, class_one,class_two):

    predicted_proba_y = model.decision_function(X_test)

    predicted_proba_y = (predicted_proba_y - predicted_proba_y.min()) / (

        predicted_proba_y.max() - predicted_proba_y.min())



    predicted_y = model.predict(X_test)



    type_count = count_matrix(y_test, predicted_y, class_one, class_two)



    # Brier predicts the QUALITY of the prediction

    q_score = brier_score_loss(y_test, predicted_proba_y[:, class_two])



    # Jaccard predicts the VALUE of the prediction

    jaccard_score = jaccard_similarity_score(y_test, predicted_y)



    return jaccard_score, q_score, type_count
### Included are the calculations for accuracy and f1 scores should you feel inclined to use those instead.

def count_matrix(y_true, y_pred, class_one, class_two):

    matrix = confusion_matrix(y_true, y_pred, [class_one, class_two])

    TP = matrix[1, 1]

    TN = matrix[0, 0]

    FP = matrix[1, 0]

    FN = matrix[0, 1]



    numerator = TP * TN - FP * FN

    denominator = (TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)

    denominator = np.sqrt(denominator)



    matthews_coef = np.divide(numerator, denominator)



    ## Accuracy

    # numerator = TP + TN

    # denominator = TP + FP + TN + FN

    # correct = numerator / denominator



    ## F1-score

    # precision = np.divide(TP, (TP + FP))

    # recall = np.divide(TP, (TP + FN))

    # numerator = 2 * precision * recall

    # denominator = precision + recall

    # f1 = np.divide(numerator,denominator)



    return matthews_coef
### Build the testing dataframes - pass lists

def build_and_type(dfs,classes):

    assert len(dfs) == len(classes)

    for i in range(len(classes)):

        dfs[i]['class'] = classes[i]

    

    returned_df = dfs.pop()

    for df in dfs:

        returned_df = returned_df.append(df)

    

    return returned_df
def store_and_score(results, scores):

    scores['model_primary_jaccard'].append(results.m_primary_jaccard)

    scores['model_primary_q'].append(results.m_primary_q)

    scores['model_primary_count'].append(results.m_primary_count)

    

    scores['model_contrast_jaccard'].append(results.m_contrast_jaccard)

    scores['model_contrast_q'].append(results.m_contrast_q)

    scores['model_contrast_count'].append(results.m_contrast_count)



    scores['model_control_jaccard'].append(results.m_control_jaccard)

    scores['model_control_q'].append(results.m_control_q)

    scores['model_control_count'].append(results.m_control_count)    

    

    scores['primary_jaccard'].append(results.primary_jaccard)

    scores['primary_q'].append(results.primary_q)

    scores['primary_count'].append(results.primary_count)

    

    scores['contrast_jaccard'].append(results.contrast_jaccard)

    scores['contrast_q'].append(results.contrast_q)

    scores['contrast_count'].append(results.contrast_count)

    

    scores['control_jaccard'].append(results.control_jaccard)

    scores['control_q'].append(results.control_q)

    scores['control_count'].append(results.control_count)
scorer = make_scorer(jaccard_similarity_score)
class terminal_output_decision_fuction_model():



    def __init__(self, ngram, classifier, params, target, primary, contrast, control):

        self.ngram = ngram

        self.params = params

        self.classifier = classifier

        self.stop_words = nltk.corpus.stopwords.words('english')

        self.control = control

        self.primary = primary

        self.contrast = contrast

        self.target = target

        

    #### Vectorization is broken up into two distinct parts. The first operates on all of the data to build a total vocabulary

    def vect_model(self, df, ngram):

        truthvalues = df['class'].values

        df = df.text.astype(str)

        tfidf = TfidfVectorizer(ngram_range=ngram, stop_words=self.stop_words)

        tfidf.fit(df)

        df = tfidf.transform(df)

        return df, truthvalues, tfidf

    

    #### Vectorzation here operates on the smaller test portions of the data using the larger vocabulary

    ###### To not use the larger vocabulary can result in anomalies and bugs when scoring later on.

    def vect_test(self, df, tfidf):

        truthvalues = df['class'].values

        df = df.text.astype(str)

        df = tfidf.transform(df)

        return df, truthvalues

    

    def run(self):

        primary_target = build_and_type([self.primary,self.target],[1,0])

        contrast_target = build_and_type([self.contrast,self.target],[2,0])

        control_target = build_and_type([self.control,self.target],[3,0])

        X = build_and_type([self.control,self.contrast,self.primary,self.target],[3,2,1,0])

        

        X, y, tfidf = self.vect_model(X,self.ngram)

        primary_t_X, primary_t_y = self.vect_test(primary_target,tfidf)

        contrast_t_X, contrast_t_y = self.vect_test(contrast_target,tfidf)

        control_t_X, control_t_y = self.vect_test(control_target,tfidf)

        

        X_train, X_test, y_train, y_test = train_test_split(X,y,random_state = 42)

        

        _, pri_X_test, _, pri_y_test = train_test_split(primary_t_X, primary_t_y,random_state=42)

        _, contrast_X_test, _, contrast_y_test = train_test_split(contrast_t_X, contrast_t_y, random_state = 42)

        _, control_X_test, _, control_y_test = train_test_split(control_t_X, primary_t_y,random_state = 42)

        ### In a script setting, fiddle with n_jobs at 4 to 8 to make it run much much faster with mutlithreading

        gscv = GridSearchCV(self.classifier, self.params, scoring=scorer, n_jobs=-1)

        gscv.fit(X_train,y_train)

        

        print(gscv.best_score_)

        print(gscv.best_params_)

        

        jaccard, q, count = score_decision_function_model(X_test, y_test,gscv.best_estimator_,0,1)

        print("\n<<<<<<>>>>>>\n")

        

        print("Overall Model Scoring of Primary vs Target")

        print("Jaccard score: " + str(jaccard))

        print("Brier Score: " + str(q))

        print("Matthews Score: " + str(count))

        

        jaccard, q, count = score_decision_function_model(X_test, y_test,gscv.best_estimator_,0,2)

        print("\n<<<<<<>>>>>>\n")

        

        print("Overall Model Scoring of Contrast vs Target")

        print("Jaccard score: " + str(jaccard))

        print("Brier Score: " + str(q))

        print("Matthews Score: " + str(count))

        

        jaccard, q, count = score_decision_function_model(pri_X_test, pri_y_test,gscv.best_estimator_,0,1)

        print("\n<<<<<<>>>>>>\n")

        

        print("Scoring of Primary vs Target")

        print("Jaccard score: " + str(jaccard))

        print("Brier Score: " + str(q))

        print("Matthews Score: " + str(count))

        print("\n<<<<<<>>>>>>\n")

        

        jaccard, q, count = score_decision_function_model(contrast_X_test, contrast_y_test,gscv.best_estimator_,0,2)

        print("Scoring of Contrast vs Target")

        print("Jaccard score: " + str(jaccard))

        print("Brier Score: " + str(q))

        print("Matthews Score: " + str(count))

        print("\n<<<<<<>>>>>>\n")

        jaccard, q, count = score_decision_function_model(control_X_test, control_y_test,gscv.best_estimator_,0,3)

        print("Scoring of Control vs Target")

        print("Jaccard score: " + str(jaccard))

        print("Brier Score: " + str(q))

        print("Matthews Score: " + str(count))
class graphing_output_decision_fuction_model():



    def __init__(self, ngram, classifier, params, target, primary, contrast, control, scores_dict):

        self.ngram = ngram

        self.params = params

        self.classifier = classifier

        self.stop_words = nltk.corpus.stopwords.words('english')

        self.control = control

        self.primary = primary

        self.contrast = contrast

        self.target = target

        self.scores_dict = scores_dict

        

    #### Vectorization is broken up into two distinct parts. The first operates on all of the data to build a total vocabulary

    def vect_model(self, df, ngram):

        truthvalues = df['class'].values

        df = df.text.astype(str)

        tfidf = TfidfVectorizer(ngram_range=ngram, stop_words=self.stop_words)

        tfidf.fit(df)

        df = tfidf.transform(df)

        return df, truthvalues, tfidf

    

    #### Vectorzation here operates on the smaller test portions of the data using the larger vocabulary

    ###### To not use the larger vocabulary can result in anomalies and bugs when scoring later on.

    def vect_test(self, df, tfidf):

        truthvalues = df['class'].values

        df = df.text.astype(str)

        df = tfidf.transform(df)

        return df, truthvalues

        

    def run(self):

        primary_target = build_and_type([self.primary,self.target],[1,0])

        contrast_target = build_and_type([self.contrast,self.target],[2,0])

        control_target = build_and_type([self.control,self.target],[3,0])

        X = build_and_type([self.control,self.contrast,self.primary,self.target],[3,2,1,0])

        

        X, y, tfidf = self.vect_model(X,self.ngram)

        primary_t_X, primary_t_y = self.vect_test(primary_target,tfidf)

        contrast_t_X, contrast_t_y = self.vect_test(contrast_target,tfidf)

        control_t_X, control_t_y = self.vect_test(control_target,tfidf)

        

        X_train, X_test, y_train, y_test = train_test_split(X,y,random_state = 42)

        

        _, pri_X_test, _, pri_y_test = train_test_split(primary_t_X, primary_t_y,random_state=42)

        _, contrast_X_test, _, contrast_y_test = train_test_split(contrast_t_X, contrast_t_y, random_state = 42)

        _, control_X_test, _, control_y_test = train_test_split(control_t_X, primary_t_y,random_state = 42)

        ### In a script setting, fiddle with n_jobs at 4 to 8 to make it run much much faster with mutlithreading

        gscv = GridSearchCV(self.classifier, self.params, scoring=scorer, n_jobs=-1)

        gscv.fit(X_train,y_train)

        

        print(gscv.best_score_)

        print(gscv.best_params_)

        

        m_primary_jaccard, m_primary_q, m_primary_count = score_decision_function_model(X_test, y_test,gscv.best_estimator_,0,1)

        

        m_contrast_jaccard, m_contrast_q, m_contrast_count = score_decision_function_model(X_test, y_test,gscv.best_estimator_,0,2)



        m_control_jaccard, m_control_q, m_control_count = score_decision_function_model(X_test, y_test,gscv.best_estimator_,0,3)

        

        primary_jaccard, primary_q, primary_count = score_decision_function_model(pri_X_test, pri_y_test,gscv.best_estimator_,0,1)

        

        contrast_jaccard, contrast_q, contrast_count = score_decision_function_model(contrast_X_test, contrast_y_test,gscv.best_estimator_,0,2)



        control_jaccard, control_q, control_count = score_decision_function_model(control_X_test, control_y_test,gscv.best_estimator_,0,3)

        

        scores_tuple = namedtuple("scores_tuple",["m_primary_jaccard", "m_primary_q", "m_primary_count",

                           "m_contrast_jaccard", "m_contrast_q", "m_contrast_count",

                           "m_control_jaccard", "m_control_q", "m_control_count",

                           "primary_jaccard", "primary_q", "primary_count",

                           "contrast_jaccard", "contrast_q", "contrast_count",

                           "control_jaccard", "control_q", "control_count"])

        

        scores = scores_tuple(m_primary_jaccard, m_primary_q, m_primary_count,

                             m_contrast_jaccard, m_contrast_q, m_contrast_count,

                             m_control_jaccard, m_control_q, m_control_count,

                             primary_jaccard, primary_q, primary_count,

                             contrast_jaccard, contrast_q, contrast_count,

                             control_jaccard, control_q, control_count)

        

        store_and_score(scores,self.scores_dict)
kim = sample_dfs(400,20,df_kim,13)

fte = sample_dfs(400,20,df_fte,13)

adam = sample_dfs(400,20,df_adam,13)

neil = sample_dfs(400,20,df_neil,13)
classifier = PassiveAggressiveClassifier()

params = dict(C = [1.0])

terminal_output_decision_fuction_model((1,2),classifier,params,kim[0], adam[0], neil[0], fte[0]).run()
PA_scores = defaultdict(list)
classifier = PassiveAggressiveClassifier()

params = dict(C = [1.0])

for i in range(len(adam)):

    graphing_output_decision_fuction_model((1,2),classifier,params,kim[i], neil[i], adam[i], fte[i],PA_scores).run()
PA_scores_df = pd.DataFrame.from_dict(PA_scores,orient="columns")

PA_scores_df.head()
PA_scores_df[['contrast_count','model_contrast_count','primary_count','model_primary_count']].describe()
PA_scores_df[['contrast_count','model_contrast_count','primary_count','model_primary_count']].plot.hist(alpha=0.5,figsize=(10,8),colormap='rainbow',bins=16)
PA_scores_df[['contrast_jaccard','model_contrast_jaccard','primary_jaccard','model_primary_jaccard']].plot.hist(alpha=0.5,figsize=(8,6),colormap='spring',bins=16)
PA_scores_df[['contrast_count','model_contrast_count','primary_count','model_primary_count']].plot.kde(alpha=0.5,colormap='rainbow')
sns.violinplot(PA_scores_df[['contrast_count','model_contrast_count','primary_count','model_primary_count']])
PA_zscores = (PA_scores_df - PA_scores_df.mean()) / PA_scores_df.std()
sns.violinplot(PA_zscores[['contrast_count','model_contrast_count','primary_count','model_primary_count']])
classifier = RidgeClassifier()

params = dict(alpha = [0.0001])

Ridge_scores = defaultdict(list)

for i in range(len(adam)):

    graphing_output_decision_fuction_model((1,2),classifier,params,kim[i], neil[i], adam[i], fte[i],Ridge_scores).run()

    

Ridge_scores_df = pd.DataFrame.from_dict(Ridge_scores,orient="columns")

Ridge_zscores = (Ridge_scores_df - Ridge_scores_df.mean()) / Ridge_scores_df.std()
Ridge_scores_df[['contrast_count','model_contrast_count','primary_count','model_primary_count']].plot.kde(alpha=0.5,colormap='rainbow')
sns.violinplot(Ridge_scores_df[['contrast_count','model_contrast_count','primary_count','model_primary_count']])
sns.violinplot(Ridge_zscores[['contrast_count','model_contrast_count','primary_count','model_primary_count']])
classifier = LinearSVC()

params = dict(C = [1.0,0.1,0.01,0.001])

Linear_scores = defaultdict(list)

for i in range(len(adam)):

    graphing_output_decision_fuction_model((1,2),classifier,params,kim[i], neil[i], adam[i], fte[i],Linear_scores).run()

    

Linear_scores_df = pd.DataFrame.from_dict(Linear_scores,orient="columns")

Linear_zscores = (Linear_scores_df - Linear_scores_df.mean()) / Linear_scores_df.std()
Linear_scores_df[['contrast_count','model_contrast_count','primary_count','model_primary_count']].plot.kde(alpha=0.5,colormap='rainbow')
sns.violinplot(Linear_scores_df[['contrast_count','model_contrast_count','primary_count','model_primary_count']])
sns.violinplot(Linear_zscores[['contrast_count','model_contrast_count','primary_count','model_primary_count']])
### PASSIVE AGGRESSIVE Q

PA_scores_df[['contrast_q','model_contrast_q','primary_q','model_primary_q']].plot.kde(alpha=0.5,colormap='rainbow')
### RIDGE Q

Ridge_scores_df[['contrast_q','model_contrast_q','primary_q','model_primary_q']].plot.kde(alpha=0.5,colormap='rainbow')
### LINEAR SVC Q

Linear_scores_df[['contrast_q','model_contrast_q','primary_q','model_primary_q']].plot.kde(alpha=0.5,colormap='rainbow')