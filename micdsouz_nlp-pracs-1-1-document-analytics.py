# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input/bbc-full-text-document-classification/bbc-fulltext (document classification)/bbc/'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import cross_val_score, StratifiedKFold

from sklearn.feature_extraction.text import TfidfVectorizer

from scipy.sparse import hstack

from matplotlib import pyplot as plt

import seaborn as sns

import eli5

from IPython.display import Image
# Step 1 - Get the file details

directory = []

file = []

title = []

text = []

label = []

datapath = '/kaggle/input/bbc-full-text-document-classification/bbc-fulltext (document classification)/bbc/' 

for dirname, _ , filenames in os.walk(datapath):

    #print('Directory: ', dirname)

    #print('Subdir: ', dirname.split('/')[-1])

    # remove the Readme.txt file

    # will not find file in the second iteration so we skip the error

    try:

        filenames.remove('README.TXT')

    except:

        pass

    for filename in filenames:

        directory.append(dirname)

        file.append(filename)

        label.append(dirname.split('/')[-1])

        #print(filename)

        fullpathfile = os.path.join(dirname,filename)

        with open(fullpathfile, 'r', encoding="utf8", errors='ignore') as infile:

            intext = ''

            firstline = True

            for line in infile:

                if firstline:

                    title.append(line.replace('\n',''))

                    firstline = False

                else:

                    intext = intext + ' ' + line.replace('\n','')

            text.append(intext)



#

fulldf = pd.DataFrame(list(zip(directory, file, title, text, label)), 

               columns =['directory', 'file', 'title', 'text', 'label'])



df = fulldf.filter(['title','text','label'], axis=1)



print("FullDf : ", fulldf.shape)

print("DF : ", df.shape)
from sklearn.model_selection import train_test_split

train, test = train_test_split(df, test_size=0.2)

print("Train DF: ",train.shape)

print("Test DF: ",test.shape)
df.head()
sns.countplot(df['label']);

plt.title('Data: Target distribution');
plt.subplots(1, 2)

plt.subplot(1, 2, 1)

df['text'].apply(lambda x: len(x.split())).plot(kind='hist');

plt.yscale('log');

plt.title('Text length in words');

plt.subplot(1, 2, 2)

df['title'].apply(lambda x: len(x.split())).plot(kind='hist');

plt.yscale('log');

plt.title('Title length in words');
from wordcloud import WordCloud, STOPWORDS



# Thanks : https://www.kaggle.com/aashita/word-clouds-of-various-shapes ##

def plot_wordcloud(text, mask=None, max_words=200, max_font_size=100, figure_size=(24.0,16.0), 

                   title = None, title_size=40, image_color=False):

    stopwords = set(STOPWORDS)

    more_stopwords = {'one', 'br', 'Po', 'th', 'sayi', 'fo', 'Unknown'}

    stopwords = stopwords.union(more_stopwords)



    wordcloud = WordCloud(background_color='black',

                    stopwords = stopwords,

                    max_words = max_words,

                    max_font_size = max_font_size, 

                    random_state = 42,

                    width=800, 

                    height=400,

                    mask = mask)

    wordcloud.generate(str(text))

    

    plt.figure(figsize=figure_size)

    if image_color:

        image_colors = ImageColorGenerator(mask);

        plt.imshow(wordcloud.recolor(color_func=image_colors), interpolation="bilinear");

        plt.title(title, fontdict={'size': title_size,  

                                  'verticalalignment': 'bottom'})

    else:

        plt.imshow(wordcloud);

        plt.title(title, fontdict={'size': title_size, 'color': 'black', 

                                  'verticalalignment': 'bottom'})

    plt.axis('off');

    plt.tight_layout()  
plot_wordcloud(df["title"], title="Word Cloud of Titles")
text_transformer = TfidfVectorizer(stop_words='english', 

                                   ngram_range=(1, 2), lowercase=True, max_features=150000)
%%time

X_train_text = text_transformer.fit_transform(train['text'])

X_test_text = text_transformer.transform(test['text'])
# if you wish to have a look at the output uncomment this line

#print(X_train_text)
X_train = X_train_text

X_test = X_test_text

print("X Train DF: ",X_train.shape)

print("X Test DF: ", X_test.shape)
logit = LogisticRegression(C=5e1, solver='lbfgs', multi_class='multinomial',

                           random_state=17, n_jobs=4)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=17)
%%time

cv_results = cross_val_score(logit, X_train, train['label'], cv=skf, scoring='f1_macro')
cv_results, cv_results.mean()
%%time

logit.fit(X_train, train['label'])
eli5.show_weights(estimator=logit, 

                  feature_names= text_transformer.get_feature_names(),top=(50, 5))