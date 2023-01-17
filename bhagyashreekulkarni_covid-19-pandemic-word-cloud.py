import requests

import numpy as np

import pandas as pd

from PIL import Image

import matplotlib.pyplot as plt

from nltk.corpus import stopwords

from nltk.stem.wordnet import WordNetLemmatizer

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator



## Import the dataset

data = pd.read_csv("../input/CORD-19-research-challenge/metadata.csv")

data.info()
print("Number of observations : {} \n".format(data.shape[0]))

print("Features in the dataset : {} \n".format(data.shape[1]))
## Fetch only abstract column from the data frame

abstract_data = data['abstract']



## Check for NAs and remove

abstract_data = pd.DataFrame(abstract_data)

abstract_data.isna().sum()

abstract_data = abstract_data.dropna()  



## Remove punctuations and special chars 

abstract_data = abstract_data['abstract']

abstract_data = abstract_data.str.replace("[^-a-zA-Z0-9]", " ")



## Remove stop words

stop = stopwords.words('english')

abstract_data = abstract_data.apply(lambda x: ' '.join([word for word in str(x).split() if word.lower() not in (stop)]))



## Remove common english words

common_words = requests.get('https://gist.githubusercontent.com/deekayen/4148741/raw/98d35708fa344717d8eee15d11987de6c8e26d7d/1-1000.txt', stream=True).text

common_words = common_words.split('\n')

abstract_data = abstract_data.apply(lambda x: ' '.join([word for word in str(x).split() if word.lower() not in (common_words)]))





## Lemmatization process

lem = WordNetLemmatizer()

abstract_data = abstract_data.apply(lambda x : ' '.join([lem.lemmatize(x) for x in str(x).split()]))



## Sanitize by removing unecessary words

remove_words = ["background","abstract","summary","many","chapter","report","three",

                "known","several","patient","among","although", "found", "use", "well", "two", "one",

                "show", "used", "based", "compared", "different", "may", "similar", "showed", "whereas",

                "method", "including", "various","furthermore", "case", "include", "addition",

                "result","change", "method", "suggest", "function", "conclusion", "effect", "evaluated",

                "determined", "individual", "associated", "data", "using", "present", "presence",

                "level", "observed", "group", "actvity", "approach", "number", "within", "reported",

                "first", "specific", "developed", "thu", "process", "respectively", "increase", "identified",

                "system", "specie", "although", "without", "CONCLUSION" , "Abstract", "population","Although",

                "likely", "increased", "Furthermore", "structure", "development", "Conclusion",

                "RESULT", "Moreover", "Unknown", "year", "timeRT", "day","month", "week"]



abstract_data = abstract_data.apply(lambda x: ' '.join([word for word in str(x).split() if word not in (remove_words)]))



## join all the rows to a single string

abstract_text = ' '.join([text for text in abstract_data])





mask = np.array(Image.open(requests.get('https://drive.google.com/uc?export=view&id=1P_CaLUX3tVO8neSiP9pAtFUXtiQomT6l', stream=True).raw))



# mask = np.array(Image.open('D:\\Python\\coronavirus.png'))

## Generate Word Cloud

word_cloud = WordCloud(

    width = 1920,

    height = 1632,

    background_color = 'black',

     mask=mask

    ).generate(str(abstract_text))



## Plot the Word Cloud

plt.figure(

    figsize = (10, 8),

    facecolor = 'k',

    edgecolor = 'k')



plt.imshow(word_cloud, interpolation='bilinear')

plt.axis('off')

plt.tight_layout(pad=0)

plt.show()