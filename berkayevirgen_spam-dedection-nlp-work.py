from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV

from sklearn.feature_extraction.text import CountVectorizer

from plotly.offline import init_notebook_mode, iplot, plot

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix

from nltk.corpus import stopwords

from wordcloud import WordCloud

import matplotlib.pyplot as plt

import plotly.graph_objs as go

from plotly import tools

import seaborn as sns

import plotly as py

import pandas as pd

import numpy as np

import nltk as nlp

import re

import warnings

warnings.filterwarnings('ignore')

init_notebook_mode(connected=True)



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
data = pd.read_csv('../input/spam-text-message-classification/SPAM text message 20170820 - Data.csv',encoding='latin1')
data.head()
sns.countplot(data.Category)

plt.title('Ham-Spam Visualizing',color='blue', fontsize=15)

plt.show()

data.Category.value_counts()
data.Category = [1 if each == 'spam' else 0 for each in data.Category] #marked it as 1 if it is a spam

data.head(10)
all_words = data['Message'].str.split(expand=True).unstack().value_counts()

data = [go.Bar(

    x = all_words.index.values[2:50],

    y = all_words.values[2:50],

    marker= dict(colorscale='Jet',

                 color = all_words.values[2:100]

                ),

    text='Word counts'

)]



layout = go.Layout(

    title = dict(

        text = '<b> Top 50 (Uncleaned) Word frequencies in the training dataset</b>',

        x = 0.49, y = .93,

        font = dict(

            family = 'Italic',

            size = 15,

            color = 'Black')

    ),

    xaxis = dict(

        title = dict(

            text = 'Words',

            font = dict(family = 'Italic',

                        size = 20,

                        color = 'Black')

        ),

        ticklen = 5,

        zeroline = False

    ),

    yaxis = dict(title = dict(text = 'Count',

                              font = dict(family = 'Italic',

                                          size = 20,

                                          color = 'Black')

                             ),

                 ticklen = 5,

                 zeroline = False

                )

)



fig = go.Figure(data=data, layout=layout)

iplot(fig)
data = pd.read_csv('../input/spam-text-message-classification/SPAM text message 20170820 - Data.csv',encoding='latin1')



# list comprehension

lemma = nlp.WordNetLemmatizer()

msg_list = []



for Message in data.Message:

    Message = re.sub('[^a-z A-Z]',' ',Message).lower()

    Message = nlp.word_tokenize(Message)

    Message = [word for word in Message if not word in set(stopwords.words('english'))]

    Message = [lemma.lemmatize(word) for word in Message]

    Message = ' '.join(Message)

    msg_list.append(Message)

    

msg_list[:10]
# extracting irrelevant words

count_vectorizer = CountVectorizer(stop_words='english')

sparce_matrix = count_vectorizer.fit_transform(msg_list).toarray()
plt.subplots(figsize=(8,8))

wordcloud = WordCloud(

    background_color='white',

    max_font_size = 300,

    width=512,

    height=384,

    max_words=50

).generate(' '.join(count_vectorizer.get_feature_names()))

plt.imshow(wordcloud)

plt.axis('off')

plt.savefig('graph.png')

plt.show()
# splitting data

y = data.iloc[:,0]

x = sparce_matrix

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
# training model

rf = RandomForestClassifier(n_estimators = 4, random_state=42)

rf.fit(x_train,y_train)

print("accuracy: %",rf.score(x_test,y_test)*100)
#confusion matrix

cm=confusion_matrix(y_test,rf.predict(x_test))

f,ax=plt.subplots(figsize=(6,6))

sns.heatmap(cm,annot=True,linewidth=.5,linecolor="r",fmt=".0f",ax=ax)

plt.xlabel("y_pred")

plt.ylabel("y_true")

ax.set_xticklabels(['Ham','Spam'])

ax.set_yticklabels(['Ham','Spam'])

plt.show()
# Grid Search CrossValidation with logistic regression

grid = {'C':np.logspace(-3,3,7),'penalty':['l1','l2']} # C = log regression regularization parameter

lr = LogisticRegression()

lr_cv = GridSearchCV(lr,grid, cv=10)

lr_cv.fit(x_train, y_train)



#  finding tuned hyperparameter and best score

print('tuned Hyperparameter:',lr_cv.best_params_)

print('best acc for tuned parameter',lr_cv.best_score_)
# Confusion Matrix for LR

lr.fit(x_train,y_train)

cm=confusion_matrix(y_test,lr.predict(x_test))

f,ax=plt.subplots(figsize=(6,6))

sns.heatmap(cm,annot=True,linewidth=.5,linecolor="r",fmt=".0f",ax=ax)

plt.xlabel("y_pred")

plt.ylabel("y_true")

ax.set_xticklabels(['Ham','Spam'])

ax.set_yticklabels(['Ham','Spam'])

plt.show()