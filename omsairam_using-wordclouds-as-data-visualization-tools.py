# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



import numpy as np # linear algebra

import pandas as pd 

import matplotlib as mpl

import matplotlib.pyplot as plt

%matplotlib inline

mpl.rcParams['figure.figsize']=(9.0,7.0)    

mpl.rcParams['font.size']=12                #10 

mpl.rcParams['savefig.dpi']=100             #72 

mpl.rcParams['figure.subplot.bottom']=.1 

# Any results you write to the current directory are saved as output.
from wordcloud import WordCloud, STOPWORDS

stopwords = set(STOPWORDS)



data = pd.read_csv('../input/multipleChoiceResponses.csv', encoding="ISO-8859-1")
data.columns
data.Country.unique()
countries = data['Country'].apply(lambda x: 0 if pd.isnull(x) else x)
countries.head()
import random



def random_color_func(word=None, font_size=None, position=None,  orientation=None, font_path=None, random_state=None):

    h = 344

    s = int(100.0 * 255.0 / 255.0)

    l = int(100.0 * float(random_state.randint(60, 120)) / 255.0)

    return "hsl({}, {}%, {}%)".format(h, s, l)



wordcloud = WordCloud(

                          background_color='white',

                          stopwords=stopwords,

                          max_words=200,

                          max_font_size=40, 

                          random_state=42

                         ).generate(str(countries))

print(wordcloud)

fig = plt.figure(1)

plt.imshow(wordcloud.recolor(color_func= random_color_func, random_state=3),

           interpolation="bilinear")

plt.axis('off')

plt.show()


def random_color_func(word=None, font_size=None, position=None,  orientation=None, font_path=None, random_state=None):

    h = 130

    s = int(100.0 * 255.0 / 255.0)

    l = int(100.0 * float(random_state.randint(60, 120)) / 255.0)

    return "hsl({}, {}%, {}%)".format(h, s, l)



wordcloud = WordCloud(

                          background_color='white',

                          stopwords=stopwords,

                          max_words=200,

                          max_font_size=40, 

                          random_state=42

                         ).generate(str(data['CurrentJobTitleSelect'].apply(lambda x: 0 if pd.isnull(x) else x)))

print(wordcloud)

fig = plt.figure(1)

plt.imshow(wordcloud.recolor(color_func= random_color_func, random_state=3),

           interpolation="bilinear")

plt.axis('off')

plt.show()


def random_color_func(word=None, font_size=None, position=None,  orientation=None, font_path=None, random_state=None):

    h = 20

    s = int(100.0 * 255.0 / 255.0)

    l = int(100.0 * float(random_state.randint(60, 120)) / 255.0)

    return "hsl({}, {}%, {}%)".format(h, s, l)



wordcloud = WordCloud(

                          background_color='white',

                          stopwords=stopwords,

                          max_words=200,

                          max_font_size=40, 

                          random_state=42

                         ).generate(str(data['LanguageRecommendationSelect'].apply(lambda x: 0 if pd.isnull(x) else x)))

print(wordcloud)

fig = plt.figure(1)

plt.imshow(wordcloud.recolor(color_func= random_color_func, random_state=3),

           interpolation="bilinear")

plt.axis('off')

plt.show()


def random_color_func(word=None, font_size=None, position=None,  orientation=None, font_path=None, random_state=None):

    h = 180

    s = int(100.0 * 255.0 / 255.0)

    l = int(100.0 * float(random_state.randint(60, 120)) / 255.0)

    return "hsl({}, {}%, {}%)".format(h, s, l)



wordcloud = WordCloud(

                          background_color='white',

                          stopwords=stopwords,

                          max_words=200,

                          max_font_size=40, 

                          random_state=42

                         ).generate(str(data['FormalEducation'].apply(lambda x: 0 if pd.isnull(x) else x)))

print(wordcloud)

fig = plt.figure(1)

plt.imshow(wordcloud.recolor(color_func= random_color_func, random_state=3),

           interpolation="bilinear")

plt.axis('off')

plt.show()
def random_color_func(word=None, font_size=None, position=None,  orientation=None, font_path=None, random_state=None):

    h = 243

    s = int(100.0 * 255.0 / 255.0)

    l = int(100.0 * float(random_state.randint(60, 120)) / 255.0)

    return "hsl({}, {}%, {}%)".format(h, s, l)



wordcloud = WordCloud(

                          background_color='white',

                          stopwords=stopwords,

                          max_words=200,

                          max_font_size=40, 

                          random_state=42

                         ).generate(str(data['EmployerIndustry'].apply(lambda x: 0 if pd.isnull(x) else x)))

print(wordcloud)

fig = plt.figure(1)

plt.imshow(wordcloud.recolor(color_func= random_color_func, random_state=3),

           interpolation="bilinear")

plt.axis('off')

plt.show()
def random_color_func(word=None, font_size=None, position=None,  orientation=None, font_path=None, random_state=None):

    h = 310

    s = int(100.0 * 255.0 / 255.0)

    l = int(100.0 * float(random_state.randint(60, 120)) / 255.0)

    return "hsl({}, {}%, {}%)".format(h, s, l)



wordcloud = WordCloud(

                          background_color='white',

                          stopwords=stopwords,

                          max_words=200,

                          max_font_size=40, 

                          random_state=42

                         ).generate(str(data['WorkAlgorithmsSelect'].apply(lambda x: 0 if pd.isnull(x) else x)))

print(wordcloud)

fig = plt.figure(1)

plt.imshow(wordcloud.recolor(color_func= random_color_func, random_state=3),

           interpolation="bilinear")

plt.axis('off')

plt.show()
def random_color_func(word=None, font_size=None, position=None,  orientation=None, font_path=None, random_state=None):

    h = 77

    s = int(100.0 * 255.0 / 255.0)

    l = int(100.0 * float(random_state.randint(60, 120)) / 255.0)

    return "hsl({}, {}%, {}%)".format(h, s, l)



wordcloud = WordCloud(

                          background_color='white',

                          stopwords=stopwords,

                          max_words=200,

                          max_font_size=40, 

                          random_state=42

                         ).generate(str(data['JobFunctionSelect'].apply(lambda x: 0 if pd.isnull(x) else x)))

print(wordcloud)

fig = plt.figure(1)

plt.imshow(wordcloud.recolor(color_func= random_color_func, random_state=3),

           interpolation="bilinear")

plt.axis('off')

plt.show()
def random_color_func(word=None, font_size=None, position=None,  orientation=None, font_path=None, random_state=None):

    h = 36

    s = int(100.0 * 255.0 / 255.0)

    l = int(100.0 * float(random_state.randint(60, 120)) / 255.0)

    return "hsl({}, {}%, {}%)".format(h, s, l)



wordcloud = WordCloud(

                          background_color='white',

                          stopwords=stopwords,

                          max_words=200,

                          max_font_size=40, 

                          random_state=42

                         ).generate(str(data['MLSkillsSelect'].apply(lambda x: 0 if pd.isnull(x) else x)))

print(wordcloud)

fig = plt.figure(1)

plt.imshow(wordcloud.recolor(color_func= random_color_func, random_state=3),

           interpolation="bilinear")

plt.axis('off')

plt.show()
def random_color_func(word=None, font_size=None, position=None,  orientation=None, font_path=None, random_state=None):

    h = 203

    s = int(100.0 * 255.0 / 255.0)

    l = int(100.0 * float(random_state.randint(60, 120)) / 255.0)

    return "hsl({}, {}%, {}%)".format(h, s, l)



wordcloud = WordCloud(

                          background_color='white',

                          stopwords=stopwords,

                          max_words=200,

                          max_font_size=40, 

                          random_state=42

                         ).generate(str(data['PastJobTitlesSelect'].apply(lambda x: 0 if pd.isnull(x) else x)))

print(wordcloud)

fig = plt.figure(1)

plt.imshow(wordcloud.recolor(color_func= random_color_func, random_state=3),

           interpolation="bilinear")

plt.axis('off')

plt.show()