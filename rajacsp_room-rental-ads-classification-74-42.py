# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import re

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

import missingno as miss

from collections import Counter

from nltk.corpus import stopwords #removes and, in, the, a ... etc

import plotly.express as px

import matplotlib.pyplot as plt

#ignore warnings
import warnings
warnings.filterwarnings('ignore')
FILEPATH = '/kaggle/input/newyork-room-rentalads/room-rental-ads.csv'
df = pd.read_csv(FILEPATH)
df.head()
df.sample(3)
df.info()
df.describe()
miss.matrix(df)
miss.heatmap(df)
miss.dendrogram(df)
miss.bar(df)
# get the number of missing data points per column
missing_values_count = df.isnull().sum()

# missing points in the first 10 
missing_values_count[0:10]
def get_space(pre_content, total_space_count = 30):

    current_space_count = total_space_count - len(pre_content)
    
    return pre_content + (" " * current_space_count)
def show_missing_percentage(current_df):
    
    total_cells = np.product(current_df.shape)
    total_missing = missing_values_count.sum()
    
    total_space_count = 20

    print(get_space("Total cells", total_space_count)+": {}".format(total_cells))
    print(get_space("Total missing cells", total_space_count)+": {}".format(total_missing))

    missing_percentage = (total_missing / total_cells)

    print(get_space("Missing Percentage", total_space_count)+": {:.2%}".format(missing_percentage))
show_missing_percentage(df)
df.isnull().sum()
df = df.dropna(axis = 0)
df.isnull().sum()
df.head()
df.isnull().sum()
df = df.rename(columns = {'Vague/Not' : 'Low_Quality'})
df.head()
df['Low_Quality'] = df['Low_Quality'].astype('int32')
df.head()
# df['new_col'] = range(1, len(df) + 1)
df = df.reset_index()
df.head()
def show_donut_plot(col):
    
    cur_df = df
    
#     rating_data = cur_df.groupby(col)[['Complaint ID']].count().head(10)
    rating_data = cur_df.groupby(col)[['index']].count().head(10)
    plt.figure(figsize = (12, 8))
    plt.pie(rating_data[['index']], autopct = '%1.0f%%', startangle = 140, pctdistance = 1.1, shadow = True)

    # create a center circle for more aesthetics to make it better
    gap = plt.Circle((0, 0), 0.5, fc = 'white')
    fig = plt.gcf()
    fig.gca().add_artist(gap)
    
    plt.axis('equal')
    
    cols = []
    for index, row in rating_data.iterrows():
        cols.append(index)
    plt.legend(cols)
    
    plt.title('Donut Plot - ' + str(col) + '', loc='center')
    
    plt.show()
show_donut_plot('Low_Quality')
# Clean the data
def clean_text_simple(text):
    text = text.lower()
    text = re.sub(r'[^(a-zA-Z)\s]','', text)
    text = text.strip()
    text = re.sub("\n", "", text)

    return text
df['Description'] = df['Description'].apply(clean_text_simple)
df.head()
import spacy

nlp = spacy.load('en_core_web_sm') 
# Here we will remove noise in the string. 
# Sample noise: httpsyoutube, httpswwwyoutube, (string less than 3 characters)
def is_noise(content):
    
    if('httpsyoutube' in content or 'httpswwwyoutube' in content):
        return True
    
    if(len(content) < 3):
        return True
    
    return False
    
def get_NER(sentence):
  
    doc = nlp(sentence) 
    
    ner_set = set()
    
    for ent in doc.ents: 
        # print(ent.text, ent.start_char, ent.end_char, ent.label_) 
        # print(ent.text)
        
        current_ner = str(ent.text)
        
        if(not is_noise(current_ner)):
            ner_set.add(current_ner)
    
    return list(ner_set)
df['NER'] = df['Description'].apply(get_NER)
df['NER_count'] = df['NER'].apply(lambda x: len(x))
df_sub = df[['NER', 'NER_count']][0:50]
def highlight_max_custom(s, color = 'lightblue'):
    '''
    highlight the maximum in a Series yellow.
    '''
    is_max = s == s.max()
    return ['background-color: '+color if v else '' for v in is_max]
df_sub.style.apply(highlight_max_custom, color = '#CFFE96',  axis = 0, subset=['NER_count'])
stopwords1 = stopwords.words('english')

words_collection = Counter([item for sublist in df['NER'] for item in sublist if not item in stopwords1])
freq_word_df = pd.DataFrame(words_collection.most_common(30))
freq_word_df.columns = ['frequently_used_word','count']


freq_word_df.style.background_gradient(cmap='OrRd', low=0, high=0, axis=0, subset=None)

# Possible color map values
# 'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds', 'YlOrBr', 
# 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu','GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']
# As we need to keep the pie chart clean, we are using only top 15 rows
freq_word_df_small = freq_word_df[0:15]
fig = px.pie(freq_word_df_small, values='count', names='frequently_used_word', title='Rental ads - Frequently Used NER')
fig.show()
# Define how much percent data you wanna split
split_count = int(0.23 * len(df))
# Shuffles dataframe
df = df.sample(frac=1).reset_index(drop=True)

# Training Sets
train = df[split_count:]
trainX = train['Description']
trainY = train['Low_Quality'].values

# Test Sets
test = df[:split_count]
testX = test['Description']
testY = test['Low_Quality'].values

print(f"Training Data Shape: {testX.shape}\nTest Data Shape: {testX.shape}")
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the vectorizer, fit on training set, transform on test set
vectorizer = TfidfVectorizer()
trainX = vectorizer.fit_transform(trainX)
testX = vectorizer.transform(testX)
from sklearn.tree import DecisionTreeClassifier

dt_model = DecisionTreeClassifier()
dt_model = dt_model.fit(trainX, trainY)
from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier().fit(trainX, trainY)
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()

knn_model = knn.fit(trainX, trainY)
from sklearn.linear_model import LogisticRegression

lor = LogisticRegression(solver = "liblinear")
lor_model = lor.fit(trainX, trainY)
models = [
#     svm_model,
    dt_model,
    rf_model,
    knn_model,
    lor_model
]
best_model_accuracy = 0
best_model = None

for model in models:
    
    model_name = model.__class__.__name__
    
    predY = model.predict(testX)
    accuracy = accuracy_score(testY, predY)
    
    print("-" * 43)
    print(model_name + ": " )
    
    if(accuracy > best_model_accuracy):
        best_model_accuracy = accuracy
        best_model = model_name
    
    print("Accuracy: {:.2%}".format(accuracy))
print("Best Model : {}".format(best_model))
print("Best Model Accuracy : {:.2%}".format(best_model_accuracy))
