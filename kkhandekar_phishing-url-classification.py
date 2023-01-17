#Install TLD Extract Library

!pip install tldextract -q
#Generic

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import re



#Extract

from tldextract import extract



#Garbage

import gc



#Warnings

import warnings

warnings.filterwarnings("ignore")



#SKLearn Libraries

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.metrics import accuracy_score, confusion_matrix,classification_report

from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.pipeline import Pipeline

from sklearn.naive_bayes import MultinomialNB



#WordCloud Generator

from wordcloud import WordCloud,STOPWORDS



#Plotting Library

import matplotlib.pyplot as plt

#Load

url = '../input/phishing-site-urls/phishing_site_urls.csv'

data = pd.read_csv(url, header='infer')
#Custom Explore Function

def explore(dataframe):

    # Shape

    print("Total Records: ", dataframe.shape[0])

          

    #Check Missing/Null

    x = dataframe.columns[dataframe.isnull().any()].tolist()   

    if not x:

        print("No Missing/Null Records")

    else:        

        print("Found Missing Records")
#Explore

explore(data)
#Custom Function to extract domains

def extract_domain(x):

    tsd, td, tsu = extract(x)

    y = td + '.' + tsu 

    return td
#Extract Domain

data['Domain'] = data['URL'].apply(lambda x: extract_domain(x))



#Drop URL Column

data.drop('URL', axis=1, inplace=True)



#Re-arranging Columns

data = data[['Domain','Label']]
#Encode the Polarity Label to convert it into numerical values

lab_enc = LabelEncoder()



#Applying to the dataset

data['Label'] = lab_enc.fit_transform(data['Label'])
#Inspect

data.head()
#Splitting the normalized data into train [90%] & test[10%] data

x_train,x_test,y_train,y_test = train_test_split(data['Domain'], data.Label, test_size=0.1, random_state=0)
# Constructing Pipeline to Extract Features, Transform Count Matrix & then build/train Model



pipe = Pipeline([('vect', CountVectorizer(min_df=0,max_df=1,binary=False,ngram_range=(1,3))),

                 ('tfidf', TfidfTransformer()),

                 ('model', MultinomialNB()) ])
#Train the Model

mnb_model = pipe.fit(x_train, y_train)
# Making Prediction on Test Data & Calculating Accuracy

mnb_pred = mnb_model.predict(x_test)

print("Multinomial Naive Bayes Model Accuracy: ",'{:.2%}'.format(accuracy_score(y_test,mnb_pred)))
# Function to plot word cloud

def plot_wordcloud(text, mask=None, max_words=2000, max_font_size=120, figure_size=(12.0,12.0), 

                   title = None, title_size=20, image_color=False):



    wordcloud = WordCloud(background_color='white',

                    max_words = max_words,

                    max_font_size = max_font_size, 

                    random_state = 42,

                    mask = mask)

    wordcloud.generate(text)

    

    plt.figure(figsize=figure_size)

    if image_color:

        image_colors = ImageColorGenerator(mask);

        plt.imshow(wordcloud.recolor(color_func=image_colors), interpolation="bilinear");

        plt.title(title, fontdict={'size': title_size,  

                                  'verticalalignment': 'bottom'})

    else:

        plt.imshow(wordcloud);

        plt.title(title, fontdict={'size': title_size, 'color': 'black', 

                                  'verticalalignment': 'top'})

    plt.axis('off');

    plt.tight_layout()  

    

d = '../input/masks/masks-wordclouds/'
#Creating Seperate Dataframe with Labels & Sampling 1% of the data

df_Bad = data[data['Label']==0].sample(frac=0.1, replace=True, random_state=1)

df_Good = data[data['Label']==1].sample(frac=0.1, replace=True, random_state=1)
txt = str(df_Bad.Domain)

plot_wordcloud(txt, max_words=1000, max_font_size=50, 

               title = 'Bad Domain Names', title_size=30)
txt = str(df_Good.Domain)

plot_wordcloud(txt, max_words=1000, max_font_size=50, 

               title = 'Good Domain Names', title_size=30)