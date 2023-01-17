# word cloud
# semantic analysis
# graphs of countries mentioned
# network of which emails she sent to

import pandas as pd
from os import path
import numpy as np
import sqlite3
import pdb
import nltk.stem
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from PIL import Image
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt

#nltk.download("stopwords")

#con = sqlite3.connect('./data/database.sqlite')
con = sqlite3.connect('../input/database.sqlite')
e = pd.read_sql_query("Select ExtractedBodyText From Emails where ExtractedBodyText like '%President%' limit 20", con)
e_all = pd.read_sql_query("Select ExtractedBodyText From Emails", con)
e_all_limited = pd.read_sql_query("Select ExtractedBodyText From Emails limit 20", con)
subjects = pd.read_sql_query("Select ExtractedSubject From Emails", con)


# Any results you write to the current directory are saved as output.
print("hello")


