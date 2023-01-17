import json
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer
import numpy as np
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import plotly.graph_objs as go
import random
from plotly.offline import init_notebook_mode, iplot
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
init_notebook_mode(connected=True)

%matplotlib inline
df_train = pd.read_json('../input/train.json')
df_test = pd.read_json('../input/test.json')
print("Some rows of the training data are -:")
df_train.head()
print("Some rows of the test data are -:")
df_test.head()
print("The number of rows in the training dataset are-: ", df_train.shape[0])
print("The number of rows in the test dataset are-: ", df_test.shape[0])
# This is a function for generating random colors
def get_random_colors(n):
    
    lst = []
    for i in range(n):
        r = lambda: random.randint(0,255)
        lst.append('#%02X%02X%02X' % (r(),r(),r()))
    
    return lst
top_cuisines = df_train['cuisine'].value_counts()
trace = go.Bar(
    y = top_cuisines.index[::-1],
    x = (top_cuisines / top_cuisines.sum() * 100)[::-1],
    orientation = 'h',
     marker = dict(
        color = get_random_colors(len(top_cuisines)),
        line = dict(
            color = '#000000',
            width = 1)
    )
)

layout = go.Layout(
    xaxis = dict(
        title='Cuisines Count',
    ),
    yaxis = dict(
        title='Cuisines',
    ),
    title='Count of the Cuisines in the dataset'
)

data = [trace]
fig = go.Figure(data=data, layout=layout)
iplot(fig)
data = df_train[df_train['cuisine'] == 'italian']
data_joined = data['ingredients'].apply(lambda x: ' '.join(x).lower())
text = ' '.join(list(data_joined))
wordcloud = WordCloud(max_font_size=None, background_color='white').generate(text)
plt.figure(figsize=(15, 10))
plt.imshow(wordcloud)
plt.title('Top Ingedients for Italain Dishes')
plt.axis("off")
plt.show()
data = df_train[df_train['cuisine'] == 'mexican']
data_joined = data['ingredients'].apply(lambda x: ' '.join(x).lower())
text = ' '.join(list(data_joined))
wordcloud = WordCloud(max_font_size=None, background_color='white').generate(text)
plt.figure(figsize=(15, 10))
plt.imshow(wordcloud)
plt.title('Top Ingedients for Mexican Dishes')
plt.axis("off")
plt.show()
data = df_train[df_train['cuisine'] == 'southern_us']
data_joined = data['ingredients'].apply(lambda x: ' '.join(x).lower())
text = ' '.join(list(data_joined))
wordcloud = WordCloud(max_font_size=None, background_color='white').generate(text)
plt.figure(figsize=(15, 10))
plt.imshow(wordcloud)
plt.title('Top Ingedients for Southern_US Dishes')
plt.axis("off")
plt.show()
data = df_train[df_train['cuisine'] == 'indian']
data_joined = data['ingredients'].apply(lambda x: ' '.join(x).lower())
text = ' '.join(list(data_joined))
wordcloud = WordCloud(max_font_size=None, background_color='white').generate(text)
plt.figure(figsize=(15, 10))
plt.imshow(wordcloud)
plt.title('Top Ingedients for Indian Dishes')
plt.axis("off")
plt.show()
data = df_train[df_train['cuisine'] == 'chinese']
data_joined = data['ingredients'].apply(lambda x: ' '.join(x).lower())
text = ' '.join(list(data_joined))
wordcloud = WordCloud(max_font_size=None, background_color='white').generate(text)
plt.figure(figsize=(15, 10))
plt.imshow(wordcloud)
plt.title('Top Ingedients for Chinese Dishes')
plt.axis("off")
plt.show()
# Combining the words in the ingredients list for each row
df_train['joined_ing'] = df_train['ingredients'].apply(lambda x: ' '.join(x).lower())
df_test['joined_ing'] = df_test['ingredients'].apply(lambda x: ' '.join(x).lower())
# Label Encoding the target values
y_train = df_train['cuisine'].values.tolist()
le = LabelEncoder()
y_train = le.fit_transform(y_train)
# Creating Bag of Words features
cnt_vec = CountVectorizer()
cnt_vec.fit(df_train['joined_ing'].values.tolist())

x_train_cnt = cnt_vec.transform(df_train['joined_ing'].values.tolist())
x_test_cnt = cnt_vec.transform(df_test['joined_ing'].values.tolist())
#Creating tf-idf based features
tfidf_vec = TfidfVectorizer(binary=True)
tfidf_vec.fit(df_train['joined_ing'].values.tolist())

x_train_tfidf = tfidf_vec.transform(df_train['joined_ing'].values.tolist())
x_test_tfidf = tfidf_vec.transform(df_test['joined_ing'].values.tolist())
# Concatenating the BOW and tfidf features
x_train = np.hstack((x_train_cnt.todense(), x_train_tfidf.todense())) 
x_test = np.hstack((x_test_cnt.todense(), x_test_tfidf.todense()))
# One vs Rest Logistic Regression Classifier
model = OneVsRestClassifier(LogisticRegression())
model.fit(x_train, y_train)
# Getting the predictions on the new data
y_pred = model.predict(x_test)
id_test = df_test['id']
y_pred = le.inverse_transform(y_pred)
# Generating the submission file.
df_sub = pd.DataFrame()
df_sub['id'] = id_test
df_sub['cuisine'] = y_pred
df_sub.to_csv('ans_sub.csv', index=False)
