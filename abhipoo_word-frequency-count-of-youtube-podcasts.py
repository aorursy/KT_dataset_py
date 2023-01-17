import pandas as pd
from collections import defaultdict
import operator
from nltk.corpus import stopwords

df = pd.read_excel('../input/transcripts-1/Untitled spreadsheet.xlsx')
words = df['text'][1].split()

tf = defaultdict(int) #defining frequency table
for word in words:
    if word not in (stopwords.words('english')): #removing stopwords
        tf[word] += 1
tf_sorted = sorted(tf.items(), key=operator.itemgetter(1),reverse=True)
len(tf_sorted)
tf_sorted