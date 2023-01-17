import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

from bs4 import BeautifulSoup   

import re

import seaborn as sns

from nltk.corpus import stopwords # Import the stop word list

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.metrics import accuracy_score

from sklearn import svm

from sklearn import preprocessing
training_variants = pd.read_csv('../input/training_variants')

test_variants = pd.read_csv('../input/test_variants')

training_text = pd.read_csv('../input/training_text',sep='\|\|',skiprows=1,engine='python',names=["ID","text"])

test_text = pd.read_csv('../input/test_text',sep='\|\|',skiprows=1,engine='python',names=["ID","text"])
training_text.head()
training_variants.head()
#First row

training_text["text"][0].split('.')[0:3]
training_variants.size
test_variants.size
sns.set(style="whitegrid", color_codes=True)



plt.figure(figsize=(12,8))

ax = sns.countplot(x="Class", data=training_variants,palette="GnBu_d")

plt.ylabel('Frequency')

plt.xlabel('Class')

plt.title('Frequency distribution of classes')

plt.show()
# Unique Genes

unique_genes = list(training_variants.Gene.unique())
len(unique_genes)
# Unique variation

unique_variation = list(training_variants.Variation.unique())
len(unique_variation)
num_classes = 9
y = training_variants["Class"]
#training_variants=training_variants.drop(['Class'], axis=1)
def cleaning_text( text ):

    # 1. Remove non-letters        

    letters_only = re.sub("[^a-zA-Z0-9.]", " ", text) 

    #

    # 2. Convert to lower case, split into individual words

    words = letters_only.lower().split()                             

    #

    # 3. stopwords

    stops = set(stopwords.words("english"))     

    ls = ['no','not','nor','neither','none','negative','never']             

    # 

    # 4. Remove stop words

    meaningful_words = [w for w in words if (not w in stops or w in ls)]   

    #print(meaningful_words)

    #

    # 5. Join the words back into one string separated by space, 

    # and return the result.

    return( " ".join( meaningful_words ))  
clean_text = cleaning_text( training_text["text"][0] )
clean_text.split('.')[0:3]
# Initialize an empty list to hold the clean text

clean_train_text = []

num_reviews = training_text["text"].size

for i in range(0, num_reviews):

    try:

        clean_train_text.append( cleaning_text( training_text["text"][i] ))

    except KeyError:

        clean_train_text.append(" 0 ")

        print("Value not found", i)
# Initialize the "CountVectorizer" object, which is scikit-learn's

# bag of words tool.  

vectorizer = CountVectorizer(analyzer = "word",   

                             tokenizer = None,    

                             preprocessor = None, 

                             stop_words = None,   

                             max_features = 100000) 



# fit_transform() does two functions: First, it fits the model

# and learns the vocabulary; second, it transforms our training data

# into feature vectors. The input to fit_transform should be a list of 

# strings.

train_data_features = vectorizer.fit_transform(clean_train_text)
train_data_features.shape
vocab = vectorizer.get_feature_names()

#print( vocab)
genes = training_variants[training_variants.columns[1:2]]

#print(genes)

variations = training_variants[training_variants.columns[2:3]]

#print(variations)



###   Label encoding for Gene column   ###



labelEncoderG = preprocessing.LabelEncoder()

labelEncoderG = labelEncoderG.fit(genes)

#print(labelEncoderG.classes_)

genes_array = labelEncoderG.transform(genes)

#print(genes_array)
 ###   Label encoding for Variation column   ###

    

labelEncoderV = preprocessing.LabelEncoder()

labelEncoderV = labelEncoderV.fit(variations)

#print(labelEncoderV.classes_)

variations_array = labelEncoderV.transform(variations)

#print(variations_array)