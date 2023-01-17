! pip install spacy
!python setup.py install
!python -m spacy.en.download all
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import base64
import string
import re
from collections import Counter
from nltk.corpus import stopwords
import random 
stopwords = stopwords.words('english')
nlp = pd.read_csv('../input/research_paper.csv')
nlp.head()
nlp.isnull().sum()
from sklearn.model_selection import train_test_split
train, test = train_test_split(nlp, test_size=0.33, random_state=42)
print('Research title sample:', train['Title'].iloc[0])
print('Conference of this paper:', train['Conference'].iloc[0])
print('Training Data Shape:', train.shape)
print('Testing Data Shape:', test.shape)
fig = plt.figure(figsize=(8,4))
sns.barplot(x = train['Conference'].unique(), y=train['Conference'].value_counts())
plt.show()
plt.xkcd()
import spacy
spacy.load('en')