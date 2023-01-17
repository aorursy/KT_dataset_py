import numpy as np 

import pandas as pd 



import os

from pathlib import Path



import string

import nltk

from nltk.corpus import stopwords



import scipy.io

import scipy.linalg

from scipy.sparse import csr_matrix, vstack, lil_matrix 

from sklearn.base import TransformerMixin

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.feature_selection import SelectFromModel

from sklearn.pipeline import FeatureUnion

from sklearn.pipeline import Pipeline

from sklearn.metrics.pairwise import cosine_similarity

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

from sklearn.model_selection import train_test_split



from sklearn.svm import LinearSVC, SVC

from sklearn.linear_model import LogisticRegression

from sklearn.neural_network import MLPClassifier



import plotly.express as px

import plotly.figure_factory as ff

from yellowbrick.text import TSNEVisualizer

def get_data():

    data_path = "/kaggle/input/hate-speech-offensive-tweets-by-davidson-et-al/data/labeled_data.csv"

    df = pd.read_csv(data_path, index_col=0) 

    df = df.sample(frac=1).reset_index(drop=True)  

    return df 
tdf = get_data()

tdf.head(10)
# Remove stop words, special chars 

# stem the word tokens

# re.sub(r'^https?:\/\/.*[\r\n]*', '', text)

def clean_tweet(sent):

    stemmer = nltk.PorterStemmer()        

    tknzr = nltk.RegexpTokenizer(r'[a-zA-Z0-9]+')



    exclp = list(string.punctuation)     

    exclc = [

        "'re", "n't", "'m", "'s", "n't", "'s", 

        "``", "''", "'ve", "'m", "'ll", "'ve", 

        "...", "http", "https"]    

    sw = set(stopwords.words("english") + exclp + exclc)    



    tokens = tknzr.tokenize(sent.lower())

    words = [stemmer.stem(token) for token in tokens if not token in sw]

    return " ".join(words)     
def clean_tweet_column(df):

    df["tweet"] = df["tweet"].apply(lambda t: clean_tweet(t))

    return df
tdf = clean_tweet_column(tdf)

tdf.head(10)
def get_summary(df):   



    content = df["tweet"].values        

    word_tok = [word.lower() for item in content for word in nltk.word_tokenize(item)]    

    st_words = set(word_tok)   

    

    fact = {

        "TotalCount": len(content),

        "TotalWords": len(word_tok),        

        "TotalUniqueWords": len(st_words),

        "MeanWordsPerTweet": len(word_tok) / len(content),

    }



    return fact, df.describe()
f, s = get_summary(tdf)

s
f
def show_wfreq_plot(df, label, labelDescr = ""):   

    xdf = df[df["class"] == label]

    content = xdf["tweet"].values        

    word_tok = [word.lower() for item in content for word in nltk.word_tokenize(item)]    

    st_words = set(word_tok)   

    freq_dist = nltk.FreqDist(word_tok)    

    ls_freq = [(word, frequency) for word, frequency in freq_dist.most_common(20)]

    twdf = pd.DataFrame(ls_freq, columns=["Word", "Frequency"])

    tfig = px.bar(twdf, x="Word", y="Frequency", title="Top 20 most frequent words - " + labelDescr)

    tfig.show()                  
show_wfreq_plot(tdf, 0, "hate speech")
show_wfreq_plot(tdf, 1, "offensive")
show_wfreq_plot(tdf, 2, "neither")
def show_tsne_plot(df):   

    tknzr = nltk.RegexpTokenizer(r'[a-zA-Z0-9]+')

    sents = df["tweet"].values

    labels = np.array(df["class"].values)

    vcrz = TfidfVectorizer(lowercase=True,stop_words='english',                      

                        analyzer="word",        

                        max_features=5000,                

                        tokenizer = tknzr.tokenize)     

    sents_vals = vcrz.fit_transform(sents)

    tsne = TSNEVisualizer(labels=[0,1,2])

    tsne.fit(sents_vals, labels)

    tsne.show()
show_tsne_plot(tdf)
def get_model_tfidf(df, fset=["c", "w"], max_feats=5000, ngram_range = (1,3)):   

    tknzr = nltk.RegexpTokenizer(r'[a-zA-Z0-9]+')

    sents = df["tweet"].values

    labels = np.array(df["class"].values)

    features = []



    if "w" in fset:

        wvcrz = TfidfVectorizer(lowercase=True,stop_words='english',

                            ngram_range = ngram_range,

                            analyzer="word",        

                            max_features=max_feats,                

                            tokenizer = tknzr.tokenize)        

        features.append(('wvect_features', Pipeline([("wvect", wvcrz)])))  

        

    if "c" in fset:

        cvcrz = TfidfVectorizer(lowercase=True,stop_words='english',

                            ngram_range = ngram_range,

                            analyzer="char",        

                            max_features=max_feats,                

                            tokenizer = tknzr.tokenize)        

        features.append(('cvect_features', Pipeline([("cvect", cvcrz)])))         

         

    merger = FeatureUnion(features)                    

    sents_vals = merger.fit_transform(sents)    

    sents_vals_lil = lil_matrix(sents_vals) 

    

    return sents_vals_lil, labels
def show_sim_plot(df):   

    

    lil_mat, labels = get_model_tfidf(df, fset=["w"])

    hate_vals = lil_mat[labels == 0,:]

    off_vals = lil_mat[labels == 1,:] 

    neit_vals = lil_mat[labels == 2,:]    



    sim_matrices = [

        ("Hate", 'In Class', cosine_similarity(hate_vals, hate_vals)),

        ("Offensive", 'In Class', cosine_similarity(off_vals, off_vals)),

        ("Neither",'In Class', cosine_similarity(neit_vals, neit_vals)),

        ("Hate v Offensive",'Other Class', cosine_similarity(hate_vals, off_vals)),

        ("Offensive v Neither",'Other Class', cosine_similarity(off_vals, neit_vals)),

        ("Neither v Hate",'Other Class', cosine_similarity(hate_vals, neit_vals)),        

    ]



    scores = []



    for lab, group, score_matrix in sim_matrices:

        sdf = pd.DataFrame(score_matrix)

        sdf.replace(0, np.nan, inplace=True)

        sdf.replace(1, np.nan, inplace=True)

        sdf["max"] = sdf.max(axis=1)

        score = sdf["max"].mean()

        scores.append((lab, group, score))



    simdf = pd.DataFrame(scores, columns=["Label", "Grouping", "Score"])

    tfig = px.bar(simdf[simdf["Grouping"] == "In Class"], x="Label", y="Score", color="Label", title="In class similarities")

    tfig.show()      

    

    tfig = px.bar(simdf[simdf["Grouping"] == "Other Class"], x="Label", y="Score", color="Label", title="Out of class similarities")

    tfig.show()     
show_sim_plot(tdf)
def get_train_test(df, fset=["c", "w"], tsize=0.25):

    

    x_values, y_values = get_model_tfidf(df, fset=fset, max_feats=10000, ngram_range = (1,3))

    

    sel_mod = SelectFromModel(

        LogisticRegression(penalty='l2', solver='saga', multi_class='multinomial', random_state=1), 

        threshold=-np.inf)

    sel_feats = sel_mod.fit_transform(x_values, y_values) 

    

    x_train, x_test, y_train, y_test = train_test_split(

        sel_feats, y_values, test_size=tsize, random_state=1, stratify=y_values

    )      

    return x_train, x_test, y_train, y_test           
def run_logreg(df):     

    x_train, x_test, y_train, y_test = get_train_test(df)     

    

    clf = LogisticRegression(penalty="none", random_state=0, solver='saga')      

    clf.fit(x_train, y_train)  

    predicted = clf.predict(x_test)           

    return clf, predicted, y_test



def run_linsvm(df):     

    x_train, x_test, y_train, y_test = get_train_test(df)     

    

    clf = LinearSVC()       

    clf.fit(x_train, y_train)  

    predicted = clf.predict(x_test)           

    return clf, predicted, y_test



def run_mlptron(df):     

    x_train, x_test, y_train, y_test = get_train_test(df)     

    

    clf = MLPClassifier()       

    clf.fit(x_train, y_train)  

    predicted = clf.predict(x_test)           

    return clf, predicted, y_test

def show_results(predicted, y_test, labels=[0,1,2],  label_descr=["Hate", "Offensive", "Neither"]):

    clsr = classification_report(y_test, predicted, target_names=labels, output_dict=True)

    cm = confusion_matrix(y_test, predicted, labels=labels)   

    

    cr_df = pd.DataFrame(clsr).transpose()    

    print(cr_df)

    

    fig = ff.create_annotated_heatmap(cm, x=label_descr, y=label_descr)

    fig.update_layout(title_text='Confusion Matrix')

    fig.show()

       
clf, pred, y_test = run_logreg(tdf)

show_results(pred, y_test)
clf, pred, y_test = run_linsvm(tdf)

show_results(pred, y_test)
clf, pred, y_test = run_mlptron(tdf)

show_results(pred, y_test)