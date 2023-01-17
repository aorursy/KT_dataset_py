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
import glob



all_scrapped_files = glob.glob('/kaggle/input/amazons-advertisements/scrapped_data/scrapped_data/*/*.csv')

arrayOfLines = []



for i in all_scrapped_files:

    count = 0

    with open(i, 'r') as file:

        for line in file:

            count += 1

    # print(i,  '|| LINES: ', count)

    arrayOfLines.append(count)

print('Amount of scrapped files: ', len(all_scrapped_files))

print('List of all lines in datasets: ', arrayOfLines)
new_scrapped_files = []

arrayOfLines_2 = []



for i in all_scrapped_files:

    count = 0

    with open(i, 'r') as file:

        for line in file:

            count += 1

    if count >= 1000: 

        print(i,  '|| LINES: ', count)

        new_scrapped_files.append(i)

        arrayOfLines_2.append(count)

    

    

print('Amount of scrapped files: ', len(new_scrapped_files))

print('List of all lines in datasets: ', sorted(arrayOfLines_2))
skipper = len('/kaggle/input/amazons-advertisements/scrapped_data/scrapped_data/')

check = set()

counter = 0



for i in new_scrapped_files:

    counter += 1

    stripped_line = i[skipper:]

    # print('stripped_line: ', stripped_line)

    indexOfCrossedLine = stripped_line.find('/')

    j = stripped_line[:indexOfCrossedLine]

    # print('name of future cluster/topic: ', j)

    check.add(j)

    

print('Counter/number of files: ', counter)

print('number of cluster / topics to be measured: ', len(check))

for i in sorted(check):

    print(i)
# writing paths to files explicitly

list_of_ten = ['/kaggle/input/amazons-advertisements/scrapped_data/scrapped_data/computers/amazon_computers_monitors.csv', 

               '/kaggle/input/amazons-advertisements/scrapped_data/scrapped_data/computers/amazon_computers_printers.csv', 

               '/kaggle/input/amazons-advertisements/scrapped_data/scrapped_data/electronics/amazon_electronics_headphones.csv', 

               '/kaggle/input/amazons-advertisements/scrapped_data/scrapped_data/beauty/amazon_beauty_hair-care.csv',

               '/kaggle/input/amazons-advertisements/scrapped_data/scrapped_data/women-fashion__jewelry/amazon_women-fashion_jewelry_rings.csv',

               '/kaggle/input/amazons-advertisements/scrapped_data/scrapped_data/luggage/amazon_luggage_backpacks.csv', 

               '/kaggle/input/amazons-advertisements/scrapped_data/scrapped_data/women-fashion__clothing/amazon_women-fashion_clothing_jeans.csv',

               '/kaggle/input/amazons-advertisements/scrapped_data/scrapped_data/women-fashion__shoes/amazon_women-fashion_shoes_sneakers.csv',

               '/kaggle/input/amazons-advertisements/scrapped_data/scrapped_data/women-fashion__watches/amazon_women-fashion_watches_wrist.csv',

               '/kaggle/input/amazons-advertisements/scrapped_data/scrapped_data/tools-home__safety-security/amazon_tools-home_safety-security_flashlights.csv'

              ]

   

import random





for i in range(len(list_of_ten)):

    with open(list_of_ten[i], 'r') as source:

        data = [ (random.random(), line) for line in source ]

        print('Reading {n} '.format(n=list_of_ten[i]))

    data.sort()

    with open('amazon_ten_topic_data_labeled.csv','a') as target:

        for _, line in data[:1000]:

            if line.startswith('"') and not line.endswith('"'):

                line + '"'

            _line = line.rstrip('\n') + ', ' + str(i)

            target.write( _line )

            target.write('\n')
import pandas as pd

pd.options.display.max_colwidth = 80



import numpy as np

import spacy

import re



import warnings

warnings.filterwarnings('ignore')



import matplotlib.pyplot as plt

%matplotlib inline



from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from sklearn.decomposition import LatentDirichletAllocation, NMF

from sklearn import pipeline, ensemble, preprocessing, feature_extraction, metrics



import spacy

nlp = spacy.load('en_core_web_lg')
ads = pd.read_csv('amazon_ten_topic_data_labeled.csv', names=['ad', 'label'])

ads.describe()
ads.head(15)
ads.info()
# Removing NaN values

ads.dropna(inplace=True)



# Shuffling rows

ads = ads.sample(frac=1)

ads['ad'].describe()
ads.head(10)
# count vetorizing object

count_vectorizer = CountVectorizer()



# fitting CV

count_vectorizer.fit(ads['ad'])



# collecting the vocabulary items used in vectorizer

dictionary = count_vectorizer.vocabulary_.items()



# Storing vocab and counts in a pandas DF

vocab = []

count = []



# iterating through each vocab and count append the value to designated list

for key, value in dictionary:

    vocab.append(key)

    count.append(value)

    

# storing the count in pandas DF with vocab as index

vocab_bef_stem = pd.Series(count, index = vocab)



# sorting the DF

vocab_bef_stem = vocab_bef_stem.sort_values(ascending=False)



top_vocab = vocab_bef_stem.head(10)



# Note, that since lines in dataset is always generated randlomly, dataframes may differ

# So,play with xlim parameter to gain visual representation of the dataset.

top_vocab.plot(kind = 'barh', figsize=(20, 15), xlim=(16320, 16355)) 
def length(text):

    """Function that returns the length of a text"""

    return len(text)
ads['length'] = ads['ad'].apply(length)

ads.head(10)
def plot_sample_length_distribution(sample_texts):

    """Plots the sample length distribution.



    # Arguments

        samples_texts: list, sample texts.

    """

    

    plt.figure(figsize=(20,10))

    plt.hist([len(s) for s in sample_texts], 100)

    plt.xlabel('Length of a sample')

    plt.ylabel('Number of samples')

    plt.title('Sample length distribution of the documents before FE')

    plt.show()

    



plot_sample_length_distribution(ads['ad'])
def stopwords(text):

    """

    Function for removing 

        - stopwords,

        - punctuation,

        - numbers / digits

        - words containing numbers

    """

    doc = nlp(text)

    for token in doc:

        text = [token.text for token in doc if 

                not token.is_stop 

                and not token.is_punct 

                and not token.is_digit]

        

        

    # joining the list of words with space separator

    joined_text = " ".join(text)

    # removing words that contain any sort of numbers, like 'G2420-BK' or 'G1W40A#BGJ '

    re_text = re.sub(r"\S*\d\S*", '', joined_text).strip()

    

    return re_text
ads['NO SW'] = ads['ad'].apply(stopwords)

ads.head()
# 1 Getting rid off of two whitespaces¶

ads['NO SW']=ads['NO SW'].str.replace("  "," ")



# 2 Replacing rows with no entries with NaN to dropna them

ads['NO SW'].replace('', np.nan, inplace=True)

ads.dropna(inplace=True)

ads['NO SW'].describe()
# count vetorizing object

count_vectorizer = CountVectorizer()



# fitting CV

count_vectorizer.fit(ads['NO SW'])



# collecting the vocabulary items used in vectorizer

dictionary = count_vectorizer.vocabulary_.items()



# Storing vocab and counts in a pandas DF

vocab = []

count = []



# iterating through each vocab and count append the value to designated list

for key, value in dictionary:

    vocab.append(key)

    count.append(value)

    

# storing the count in pandas DF with vocab as index

vocab_bef_stem = pd.Series(count, index = vocab)



# sorting the DF

vocab_bef_stem = vocab_bef_stem.sort_values(ascending=False)



top_vocab = vocab_bef_stem.head(10)

top_vocab.plot(kind = 'barh', figsize=(20, 15), xlim=(10850, 10970))
# ad is chosen randomly

print(ads['ad'][806], ' ====', ads['NO SW'][806])
ads.head()
tfidf = TfidfVectorizer(max_df=0.95, min_df=2)



dtm = tfidf.fit_transform(ads['NO SW'])



dtm
nmf_model = NMF(n_components=10,random_state=42)



# This can take awhile, we're dealing with a large amount of documents!

nmf_model.fit(dtm)
len(tfidf.get_feature_names())
for index,topic in enumerate(nmf_model.components_):

    print(f'THE TOP 5 WORDS FOR TOPIC #{index}')

    print([tfidf.get_feature_names()[i] for i in topic.argsort()[-5:]])

    print('\n')
topic_results = nmf_model.transform(dtm)
ads['topic label'] = topic_results.argmax(axis=1)



my_topic_dictionary = {0: 'Watches', 

                       1: 'Monitors', 

                       2: 'Printers', 

                       3: 'Ring', 

                       4: 'Jeans', 

                       5: 'Headphones', 

                       6: 'Flashlights',

                       7: 'Backpacks',

                       8: 'Sneakers',

                       9: 'Hair-Care'}



ads['topic name'] = ads['topic label'].map(my_topic_dictionary)

ads
def reassign_2_label(topic):

    switcher = {

        0: 8,

        1: 0,

        2: 1,

        3: 4,

        4: 6,

        5: 2,

        6: 9,

        7: 5,

        8: 7,

        9: 3

    }

    return switcher.get(topic)
ads['predicted label'] = ads['topic label'].apply(reassign_2_label)

ads.head()
confusion_matrix = pd.crosstab(ads['label'], ads['predicted label'])

confusion_matrix