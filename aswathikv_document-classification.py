!pip install pdfminer
!pip install docxpy
!pip install tika
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter

from pdfminer.converter import TextConverter

from pdfminer.layout import LAParams

from pdfminer.pdfpage import PDFPage

from io import StringIO

from os.path import splitext

import os

import re

import nltk

import pandas as pd

import numpy as np

import docxpy

from tika import parser

import warnings

import seaborn as sns

warnings.filterwarnings('ignore')


def splitext_(path):

    if len(path.split('.')) > 2:

        return path.split('.')[0],'.'.join(path.split('.')[-2:])

    return splitext(path)



def text_preprocess(text):

    cleaned_text =  re.sub(r"[^a-zA-Z]", ' ', text)  

    return cleaned_text



# extracting text from pdf file



def convert_pdf_to_txt(path):

    rsrcmgr = PDFResourceManager()

    retstr = StringIO()

    laparams = LAParams()

    device = TextConverter(rsrcmgr, retstr, laparams=laparams)

    fp = open(path, 'rb')

    interpreter = PDFPageInterpreter(rsrcmgr, device)

    password = ""

    maxpages = 0

    caching = True

    pagenos=set()

    try:

        for page in PDFPage.get_pages(fp, pagenos, maxpages=maxpages, password=password,\

                                  caching=caching, check_extractable=True):

            interpreter.process_page(page)

    except:

        print('This pdf won\'t allow text extraction!')        

    fp.close()

    device.close()

    str = retstr.getvalue()

    retstr.close()

    return str

 



extracted = []    



# Based on the extension of file, extracting text



for foldername,subfolders,files in os.walk(r"/kaggle/input/health_docs"):

    for file_ in files:

        dict_ = {}

        file_name,extension = splitext_(file_)

        if extension == '.pdf':

            converted = convert_pdf_to_txt(foldername +"/" + file_)

            converted = text_preprocess(converted)   

            dict_['Extracted'] = converted

            dict_['Label'] = foldername.split('/')[-1]

            extracted.append(dict_)

            

        elif extension == '.docx':

            doc = docxpy.process(foldername +'/'+ file_)

            doc = text_preprocess(doc)

            

            dict_['Extracted'] = doc

            dict_['Label'] = foldername.split('/')[-1]

            extracted.append(dict_)

        elif extension == '.ppt':

            parsed = parser.from_file(foldername +'/'+ file_)

            ppt = parsed["content"]

            ppt = text_preprocess(ppt)

            dict_['Extracted'] = ppt

            dict_['Label'] = foldername.split('/')[-1]

            extracted.append(dict_)   

        

            

        df =  pd.DataFrame(extracted)

        print(df)

        df.to_csv('labelled_data.csv')

            

import pandas as pd

data = pd.read_csv("/kaggle/input/labelled-data/bla.csv")

data.head()
data = data.drop('Unnamed: 0',axis=1)
data.isna().sum()
data=data.dropna(axis=0)

data.isna().sum()
import re



data['cleaned_text'] = data['text'].str.lower()

data['cleaned_text'] = data['cleaned_text'].apply(lambda text : re.sub(r'\d+','', text)) 

data['cleaned_text'] = data['cleaned_text'].apply(lambda text : re.sub('[^a-zA-Z0-9-_*.]', ' ', text) )

data['cleaned_text'] = data['cleaned_text'].str.strip()

data.head()
import nltk

from nltk.tokenize import word_tokenize

from nltk.corpus import stopwords



stop_words = set(stopwords.words('english'))

data['cleaned_text'] = data['cleaned_text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))

from nltk.stem import PorterStemmer

from nltk.tokenize import word_tokenize



stemmer= PorterStemmer()

data['cleaned_text'].apply(lambda x : stemmer.stem(x))

data['cleaned_text'].head()

import matplotlib.pyplot as plt

from wordcloud import WordCloud



text = str(data['cleaned_text'].values)

wordcloud  = WordCloud(stopwords=stop_words).generate(text)

plt.imshow(wordcloud)

plt.axis("off")

plt.tight_layout(pad=0)

plt.show()
data.head()
data.label.value_counts()
sns.set(font_scale=.9)

sns.countplot(data['label'])

sns.set_color_codes(palette='deep')

plt.show()
from sklearn.preprocessing import LabelEncoder



data['label'] = LabelEncoder().fit_transform(data['label'])
from sklearn.feature_extraction.text import TfidfVectorizer



vector = TfidfVectorizer()

vector.fit(data['cleaned_text'])

tfidf_text = vector.transform(data['cleaned_text'])

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score,classification_report,confusion_matrix



X = tfidf_text

y = data['label']



X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.2,random_state=42)

lr = LogisticRegression()

lr.fit(X_train,y_train)

ypred = lr.predict(X_test)

confusion_matrix(y_test,ypred)
accuracy_score(y_test,ypred)
from sklearn.naive_bayes import GaussianNB

import seaborn as sns



nb = GaussianNB()

nb.fit(X_train.toarray(),y_train)

ypred = nb.predict(X_test.toarray())

sns.heatmap(confusion_matrix(y_test,ypred),annot=True)
from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier 



classifier_dict = { 'DecisionTree' : DecisionTreeClassifier(), 'Random Forest': RandomForestClassifier(), 'Logistic Regression': LogisticRegression(),'KNN': KNeighborsClassifier()}



l = []

for key, clf in classifier_dict.items():

    dict_ = {}

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    dict_['class'] = clf.__class__.__name__

    dict_['acc'] = accuracy

    l.append(dict_)

    print(key+" : "+str(accuracy))
