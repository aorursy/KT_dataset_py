#!pip install --upgrade pip
!pip install tensorflow --upgrade

#!pip uninstall tensorflow
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt # plotting
import numpy as np # linear algebra
import os # accessing directory structure
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing import sequence, text
from keras.layers import Input, Embedding
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from textblob import TextBlob
from collections import Counter #counting of words in the texts
import operator
import datetime as dt
import pandas as pd
import numpy as np
import warnings
import string
import re
warnings.filterwarnings('ignore')
print(tf.__version__)
#load data as tf.data.Dataset
seed=42
data_paths = '../input/sanad-dataset'
labels=os.listdir(data_paths) 
raw_data = tf.keras.preprocessing.text_dataset_from_directory(
    data_paths,
    labels="inferred",
    label_mode="int",
    #class_names=classes,
    #batch_size=1,
    max_length=None,
    shuffle=True,
    seed=seed,
    validation_split=None,
    subset=None,
    follow_links=False,
)
print("Article classes are:\n",raw_data.class_names)
x=[]
y=[]
for text_batch, label_batch in raw_data:
    for i in range(len(text_batch)):
        s=text_batch.numpy()[i].decode("utf-8") 
        x.append(s)
        y.append(raw_data.class_names[label_batch.numpy()[i]])
        #print(label_batch.numpy()[i])
print(len(x))
print(len(y))
unique, counts = np.unique(y, return_counts=True)
plt.figure("classe Pie", figsize=(10, 10))
plt.title("Pie plot of the class frequencies")
plt.pie(counts, labels=labels)
plt.legend(unique)
plt.show();
plt.bar( labels,counts)
plt.show();
#convert to DataFrame for EDA flexibility
data =pd.DataFrame({"text":x,"label":y}) 
stop_words = list(set(stopwords.words('arabic')))
print(stop_words)

#Function count stop words in text
def word_count(text, word_list):
    count_w = dict()
    for w in word_list:
        count_w[w] = 0
        words = text.lower().split()
        for word in words:
            _word = word.strip('.,:-)()')
            if _word in count_w:
                count_w[_word] +=1

    return count_w
#Count the stop word distribution in x=1000 examples of the dataset
lst=[]
sample=1000
A=word_count(data['text'][1],stop_words)
lst=list(A.values())[:]
for  i in range(1,sample):
    A=word_count(data['text'][i],stop_words)
    for j in range(len(A)-1):
        lst[j]=lst[j]+list(A.values())[j]
#visualize the distribution of 
plt.figure("stop words Pie", figsize=(10, 10))
plt.pie(lst, labels=stop_words)
plt.show();
#We conclude that the most stop words used in this dataset are:"إلى-من-في-أن-على". This to indicate place and causality.
# # Feature extraction/Text Mining
#Count features:
#Feature 1: Count the number of words in each example.

#Feature 2: Count the number of characters in each example.

#Feature 3: Average length of words used in statement.

#Feature 4: Count stop word per text.

#Feature 5: Getting top 50 used words.

#Feature 6: Count of punctuations in the input.



#Function for removing punctuations from string
def remove_punctuations_from_string(string1):
    string1 = string1.lower() #changing to lower case
    translation_table = dict.fromkeys(map(ord, string.punctuation), ' ') #creating dictionary of punc & None
    string2 = string1.translate(translation_table) #translating string1
    return string2
#Function for removing stopwords.
def remove_stopwords_from_string(string1):
    pattern = re.compile(r'\b(' + r'|'.join(stopwords.words('arabic')) + r')\b\s*') #compiling all stopwords.
    string2 = pattern.sub('', string1) #replacing the occurrences of stopwords in string1
    return string2
#Lets take backup of un-processed text, we might need it for future functions:
data['text_backup']=data['text']
data.head()
#We need to remove the stop words and the punctuation in the text:
data["text"] = data["text"].apply(lambda x:remove_punctuations_from_string(x))
data["text"] = data["text"].apply(lambda x:remove_stopwords_from_string(x))
print(data["text"][1])
#Feature 1: Count the number of words in each example:
data['Feature_1']= data["text_backup"].apply(lambda x: len(str(x).split()))
#Feature 2: Count the number of characters in each example:
data['Feature_2']= data["text_backup"].apply(lambda x: len(str(x)))
#Feature 3: Average length of words used in statement
data['Feature_3']= data["Feature_2"]/data['Feature_1']
data.head()

#Feature 4: Count stop word per text
data['Feature_4'] = data["text_backup"].apply(lambda x: len([w for w in str(x).lower().split() if w in stop_words]))
data.head()
#Getting top 50 used word in the dataset and thier frequency in each sample.
all_text_without_sw = ''
for i in data.itertuples():
    all_text_without_sw = all_text_without_sw +  str(i.text)
#getting counts of each word:
counts = Counter(re.findall(r"[\w']+", all_text_without_sw))
#deleting ' from counts
del counts["'"]
#getting top 50 used words:
sorted_x = dict(sorted(counts.items(), key=operator.itemgetter(1),reverse=True)[:50])
 #Feature 5:getting top 50 used words:
data['Feature_5'] = data['text'].apply(lambda x: len([w for w in str(x).lower().split() if w in sorted_x]))
data.head()
#Feature 6: Count of punctuations in the input.
data['Feature_6'] = data['text_backup'].apply(lambda x: len([w for w in str(x) if w in string.punctuation]) )
data.head()
#Lets visualize these 3 last extracted features:

                    #Feature 4: Count stop word per text.

                    #Feature 5: Getting top 50 used words.

                    #Feature 6: Count of punctuations in the input
            
            
def plot_bar_chart_from_dataframe(dataframe1,key_column,columns_to_be_plotted):
    import pandas as pd
    test_df1 = dataframe1.groupby(key_column).sum()
    test_df2 = pd.DataFrame()
    for column in columns_to_be_plotted:
        test_df2[column] = round(test_df1[column]/ test_df1[column].sum()*100,2)
    test_df2 = test_df2.T 
    
    ax = test_df2.plot(kind='bar', stacked=True, figsize =(10,5),legend = 'reverse',title = '% Distribution over classes')
    for p in ax.patches:
        a = p.get_x()+0.4
        ax.annotate(str(p.get_height()), (a, p.get_y()), xytext=(5, 10), textcoords='offset points')

key_column = 'label'
columns_to_be_plotted = ['Feature_4','Feature_5','Feature_6']
plot_bar_chart_from_dataframe(data,key_column,columns_to_be_plotted)
#Feature 7: Count of Most words start with.

starting_words = sorted(list(map(lambda word : word[:2],filter(lambda word : len(word) > 3,all_text_without_sw.split()))))
sw_counts = Counter(starting_words)
top_30_sw = dict(sorted(sw_counts.items(), key=operator.itemgetter(1),reverse=True)[:30])
data['Feature_7'] = data['text'].apply(lambda x: len([w for w in str(x).lower().split() if w[:2] in top_30_sw and w not in stop_words]) )

#Feature 8: Count of Most words end with.
ending_words = sorted(list(map(lambda word : word[-2:],filter(lambda word : len(word) > 3,all_text_without_sw.split()))))
ew_counts = Counter(ending_words)
top_30_ew = dict(sorted(sw_counts.items(), key=operator.itemgetter(1),reverse=True)[:30])
data['Feature_8'] = data['text'].apply(lambda x: len([w for w in str(x).lower().split() if w[:2] in top_30_ew and w not in stop_words]) )
data.head()
tokenized_all_text = word_tokenize(all_text_without_sw) #tokenize the text
list_of_tagged_words = nltk.pos_tag(tokenized_all_text) #adding POS Tags to tokenized words
set_pos  = (set(list_of_tagged_words))                  # set of POS tags & words
set_pos

nouns = ['NN','NNS','NNP','NNPS'] #POS tags of nouns
list_of_words = set(map(lambda tuple_2 : tuple_2[0], filter(lambda tuple_2 : tuple_2[1] in  nouns, set_pos)))
data['Feature_9'] = data['text'].apply(lambda x: len([w for w in str(x).lower().split() if w in list_of_words]) )

pronouns = ['PRP','PRP$','WP','WP$'] # POS tags of pronouns
list_of_words = set(map(lambda tuple_2 : tuple_2[0], filter(lambda tuple_2 : tuple_2[1] in  pronouns, set_pos)))
data['Feature_10'] = data['text'].apply(lambda x: len([w for w in str(x).lower().split() if w in list_of_words]) )
verbs = ['VB','VBD','VBG','VBN','VBP','VBZ'] #POS tags of verbs
list_of_words = set(map(lambda tuple_2 : tuple_2[0], filter(lambda tuple_2 : tuple_2[1] in  verbs, set_pos)))
data['Feature_11'] = data['text'].apply(lambda x: len([w for w in str(x).lower().split() if w in list_of_words]) )

adverbs = ['RB','RBR','RBS','WRB'] #POS tags of adverbs
list_of_words = set(map(lambda tuple_2 : tuple_2[0], filter(lambda tuple_2 : tuple_2[1] in  adverbs, set_pos)))
data['Feature_12'] = data['text'].apply(lambda x: len([w for w in str(x).lower().split() if w in list_of_words]) )

adjectives = ['JJ','JJR','JJS'] #POS tags of adjectives
list_of_words = set(map(lambda tuple_2 : tuple_2[0], filter(lambda tuple_2 : tuple_2[1] in  adjectives, set_pos)))
data['Feature_13'] = data['text'].apply(lambda x: len([w for w in str(x).lower().split() if w in list_of_words]) )

data.head()
data.to_csv(r'data.csv', index = False)
