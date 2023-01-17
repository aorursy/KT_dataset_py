! pip install mglearn

import os

from sklearn.pipeline import make_pipeline

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import confusion_matrix

from sklearn.linear_model import LogisticRegression

from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer

from sklearn.model_selection import train_test_split

import mglearn

import numpy as np

import pandas as pd

import seaborn as sn

import matplotlib as mpl

import matplotlib.pyplot as plt

import nltk

from nltk.corpus import stopwords 

from string import punctuation

from gensim.sklearn_api import W2VTransformer



from subprocess import check_output

from wordcloud import WordCloud, STOPWORDS