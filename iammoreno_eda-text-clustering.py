# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
import seaborn as sns

from itertools import islice
from collections import Counter
import re

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from scipy.stats import pearsonr
import wordcloud
from textblob import TextBlob

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
def jaccard(chain_one, chain_two):
    """
        Returns the jaccard simalarity between two strings of text 0<=jaccard<=1
    chain_one: type string
    chain_two: type string
    """
    chain_one = chain_one.split()
    chain_two = chain_two.split()
    
    anb = set(chain_one).intersection(set(chain_two))
    
    
    return len(anb)/(len(chain_one) + len(chain_two) - len(anb) + .01)
def clean_text(text):
    text = re.sub(r"[^a-zA-Z]"," ",text)
    text =  re.sub(r"\b\w{1}\b", "", text)
    text = text.lower()
    text = re.sub(r"http", "",text)
    
    tokens = text.split()
    
    text = " ".join([word for word in tokens if word not in ENGLISH_STOP_WORDS])
    
    return text
    
def visualize_most_common_words(frame, text_col_name = "text", class_column = None, return_top = 10, cleanse_text = False):
    all_words = list()
    colors = ['orange','red', 'green']
    
    if cleanse_text:
        frame[text_col_name] = frame[text_col_name].fillna('').apply(lambda x: clean_text(x))
    
        
    if not class_column:
        for _, row in frame.fillna('').iterrows():
            for word in row[text_col_name].split():
                all_words.append(word)
        words_dict = Counter(all_words).most_common(return_top)
        
        words_dict = dict(words_dict)
        
        
        
        fig, ax =plt.subplots(nrows = 1, ncols = 1, figsize = (12,7))
        
        ax.barh(list(words_dict.keys()), list(words_dict.values()), edgecolor = 'black', color = 'steelblue')
    else:

        classes = frame[class_column].unique()
        fig, ax =plt.subplots(nrows = 1, ncols = classes.size, figsize = (20,9))
        
        for i, class_ in enumerate(classes):
            for _, row in frame[frame[class_column] == class_].fillna('').iterrows():
                for word in row[text_col_name].split():
                    all_words.append(word)
            words_dict = Counter(all_words).most_common(return_top)

            words_dict = dict(words_dict)

            ax[i].barh(list(words_dict.keys()), list(words_dict.values()), edgecolor = 'black', color = colors[i])
            ax[i].set_title(class_)
            
            all_words = list()
        
                
def visualize_text_lenght(frame, text_col_name = "text", class_column = None, cleanse_text = False, separated_plots = True):
    colors = ['orange','red', 'green']
    
    if cleanse_text:
        frame[text_col_name] = frame[text_col_name].fillna('').apply(lambda x: clean_text(x))
        
    frame["doc_lenght"] = frame[text_col_name].fillna('').apply(lambda x: len(x.split()))
    
    avg_length = np.mean(frame.doc_lenght)
    
    if not separated_plots:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize = (12,7))
        if not class_column:
            sns.kdeplot(frame["doc_lenght"],color = 'orange', alpha = .5, shade = True, ax = ax)
            ax.set_title(f"Documents lenght distribution\nMean words by document: {int(avg_length)}")
            ax.set_xlabel("Number of words")
        else:
            for color, class_ in zip(colors, frame[class_column].unique()):
                sns.kdeplot(frame[frame[class_column] == class_]["doc_lenght"],color = color, alpha = .5, shade = True, ax = ax, label = class_)
            ax.set_title(f"Documents lenght distribution by class\nMean words by document: {int(avg_length)}")
            ax.set_xlabel("Number of words")
    else:
        classes = frame[class_column].unique()
        nrows = 1
        ncols = classes.size
        fig, ax = plt.subplots(nrows = nrows, ncols = ncols, figsize = (20,7))
        for i, class_ in enumerate(classes):
            avg_length = np.mean(frame[frame[class_column] == class_].doc_lenght)
            sns.kdeplot(frame[frame[class_column] == class_]["doc_lenght"],color = colors[i], alpha = .5, shade = True, ax = ax[i], label = class_)
            ax[i].set_title(f"Class: {class_}\nMean words by document: {int(avg_length)}")
            ax[i].set_xlabel("Number of words")
            
            
            
                                                       
                                                       
    
def visualize_text_vs_selected_lenght(frame, text_col_name = "text", sel_text_col_name = "selected_text", class_column = None, cleanse_text = False, separated_plots = True):
    colors = ['orange','red', 'green']
    if cleanse_text:
        frame[text_col_name] = frame[text_col_name].fillna('').apply(lambda x: clean_text(x))
        frame[sel_text_col_name] = frame[sel_text_col_name].fillna('').apply(lambda x: clean_text(x))
    frame["doc_lenght"] = frame[text_col_name].fillna('').apply(lambda x: len(x.split()))
    frame["sel_doc_lenght"] = frame[sel_text_col_name].fillna('').apply(lambda x: len(x.split()))
    
    text_corr = pearsonr(frame["doc_lenght"], frame["sel_doc_lenght"])[0]
    
    if not separated_plots:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize = (12,7))
        if not class_column:
            ax.scatter(frame["doc_lenght"], frame["sel_doc_lenght"],facecolors = 'none',edgecolor = 'orange')
            ax.set_title(f"Documents lenght distribution\Correlation for words in selected and original text: {round(text_corr, 3)}")
            ax.set_xlabel("Text words lenght")
            ax.set_ylabel("Selected text words lenght")
        else:
            for color, class_ in zip(colors, frame[class_column].unique()):
                ax.scatter(frame["doc_lenght"], frame["sel_doc_lenght"],facecolors = 'none',c = frame[class_column].map({'positive':'green','negative':'red','neutral':'orange'}))
            ax.set_title(f"Documents lenght distribution by class\nCorrelation for words in selected and original text: {round(text_corr, 3)}")
            ax.set_xlabel("Text words lenght")
            ax.set_ylabel("Selected text words lenght")
    else:
        classes = frame[class_column].unique()
        nrows = 1
        ncols = classes.size
        fig, ax = plt.subplots(nrows = nrows, ncols = ncols, figsize = (20,7))
        for i, class_ in enumerate(classes):
            text_corr = pearsonr(frame[frame[class_column] == class_]["doc_lenght"], frame[frame[class_column] == class_]["sel_doc_lenght"])[0]
            ax[i].scatter(frame[frame[class_column] == class_]["doc_lenght"], frame[frame[class_column] == class_]["sel_doc_lenght"],facecolors = 'none',color = colors[i])
            ax[i].set_title(f"Class: {class_}\nMean words by document: {round(text_corr,3)}")
            ax[i].set_xlabel("Text words lenght")
            ax[i].set_ylabel("Selected text words lenght")
            
    
def visualize_jaccard_similarity(frame, document_one_col_name = "text", document_two_col_name = "selected_text", class_column = None, cleanse_text = False, separated_plots = True):
    colors = ['orange','red', 'green']
    if cleanse_text:
        frame[document_one_col_name] = frame[document_one_col_name].fillna('').apply(lambda x: clean_text(x))
        frame[document_two_col_name] = frame[document_two_col_name].fillna('').apply(lambda x: clean_text(x))
    frame["jaccard_similarity"] = frame.apply(lambda x: jaccard(x[document_one_col_name] ,x[document_two_col_name]), axis = 1)
    
    
    avg_js = np.mean(frame.jaccard_similarity)
    
    if not separated_plots:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize = (12,7))
        if not class_column:
            sns.kdeplot(frame["jaccard_similarity"],color = 'orange', alpha = .5, shade = True, ax = ax)
            ax.set_title(f"Documents lenght distribution\nMean words by document: {int(avg_js)}")
            ax.set_xlabel("Number of words")
        else:
            for color, class_ in zip(colors, frame[class_column].unique()):
                sns.kdeplot(frame[frame[class_column] == class_]["jaccard_similarity"],color = color, alpha = .5, shade = True, ax = ax, label = class_)
            ax.set_title(f"Documents Jaccard similarityon by class\nJaccard similarity by document: {int(avg_js)}")
            ax.set_xlabel("Jaccard similarity")
    else:
        classes = frame[class_column].unique()
        nrows = 1
        ncols = classes.size
        fig, ax = plt.subplots(nrows = nrows, ncols = ncols, figsize = (20,7))
        for i, class_ in enumerate(classes):
            avg_js = np.mean(frame[frame[class_column] == class_].jaccard_similarity)
            sns.kdeplot(frame[frame[class_column] == class_]["jaccard_similarity"],color = colors[i], alpha = .5, shade = True, ax = ax[i], label = class_)
            ax[i].set_title(f"Class: {class_}\nJaccard similarity: {round(avg_js, 3)}")
            ax[i].set_xlabel("Jaccard similarity")
def make_word_cloud(frame, text_col_name = 'text', class_column = None):
    colors = ['Oranges','Reds', 'Greens']
    if not class_column:
        wc = wordcloud.WordCloud(
            width= 900,
            height= 500,
            max_words = 500,
            max_font_size= 100,
            relative_scaling= .5,
            colormap = 'Oranges',
            normalize_plurals = True,
            stopwords = ENGLISH_STOP_WORDS
        ).generate(frame[text_col_name].to_string())
        
        plt.figure(figsize = (12,7))
        plt.imshow(wc, interpolation= 'bilinear')
        plt.axis("off")
    else:

        classes = frame[class_column].unique()
        fig, ax =plt.subplots(nrows = 1, ncols = classes.size, figsize = (20,20))
        
        for i, class_ in enumerate(classes):
            wc = wordcloud.WordCloud(
            width= 900,
            height= 500,
            max_words = 500,
            max_font_size= 100,
            relative_scaling= .5,
            colormap = colors[i],
            normalize_plurals = True,
            stopwords = ENGLISH_STOP_WORDS
        ).generate(frame[frame[class_column] == class_][text_col_name].to_string())
        
            ax[i].imshow(wc, interpolation= 'bilinear')
            ax[i].axis("off")
            ax[i].set_title(class_)
train = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/train.csv')
test = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/test.csv')
train.sample(7)
train.info()
train = train[train.text.notnull()]
train.sentiment.value_counts(normalize = True).plot.bar(edgecolor = 'black', color = 'steelblue')
plt.title("Class distribution");
visualize_text_lenght(train, cleanse_text = True, separated_plots= True, class_column= 'sentiment', text_col_name= 'selected_text')
visualize_jaccard_similarity(train, class_column= 'sentiment', separated_plots= True, cleanse_text= False)
visualize_text_vs_selected_lenght(train, class_column= 'sentiment', separated_plots= True)
visualize_most_common_words(train, cleanse_text= True, class_column= 'sentiment', return_top= 20)
make_word_cloud(train, class_column= 'sentiment')