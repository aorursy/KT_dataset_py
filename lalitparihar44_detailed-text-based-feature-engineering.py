#As usual, in the first step we will import the libraries which we are gonna use.

import pandas as pd #pandas will help us reading the csv data to dataframes(df) and then working on the df.

import matplotlib.pyplot as plt #matplotlib will be used to visualize the data in the form of graphs.

import numpy as np # linear algebra

import string #for text pre-processing

from nltk.corpus import stopwords #for removing stopwords

import re #Regular expression operations

import xgboost as xgb #For predicting the values

from sklearn.model_selection import KFold #for cross validations(CV)

from sklearn import metrics #for getting CV score

from collections import Counter #counting of words in the texts

import operator

from nltk import ngrams

import nltk

from nltk import word_tokenize



#Now we will read the input data and store them in dataframes for further processing.

training_df = pd.read_csv("../input/spooky-author-identification/train.csv")

testing_df = pd.read_csv("../input/spooky-author-identification/test.csv")



#Once the data is loaded, we need to check the data and if it is loaded correctly. Executing the below

#command will display the top 5 rows of the training_df dataframe.

training_df.head(5)
#similarly we will check the testing dataframe.

testing_df.head(5)
training_author_df = training_df.groupby('author',as_index=False).count()

training_author_df.head()
#Though it is clear that there is no much difference between the inputs for each classes(authors),

#we can still have a look at bar-chart for better visualisation.



objects = training_author_df['author']  #storing values of authors in objects

y_pos = np.arange(len(objects)) #creating numpy array for the count of authors

ids = training_author_df['id'] #assignig values of id for each author

 

plt.bar(y_pos, ids, align='center', alpha=0.4,color = 'bgk') #basic configuration for bar chart

plt.xticks(y_pos, objects) #assigning Labels to be displayed on X-axis

plt.ylabel('Input Count') #Labels for Y-axis

plt.title('Inputs per author') #Label for Chart

 

plt.show()

test_string = training_df.iloc[0]['text']



test_string
string.punctuation
#Function for removing punctuations from string

def remove_punctuations_from_string(string1):

    string1 = string1.lower() #changing to lower case

    translation_table = dict.fromkeys(map(ord, string.punctuation), ' ') #creating dictionary of punc & None

    string2 = string1.translate(translation_table) #translating string1

    return string2

#lets check the function on our test string.



print('After processing')

test_string = remove_punctuations_from_string(test_string)

test_string

#The punctuations have been removed from the string. Lets write a similar function for removing

#stopwords.



def remove_stopwords_from_string(string1):

    pattern = re.compile(r'\b(' + r'|'.join(stopwords.words('english')) + r')\b\s*') #compiling all stopwords.

    string2 = pattern.sub('', string1) #replacing the occurrences of stopwords in string1

    return string2



print('After processing')

test_string = remove_stopwords_from_string(test_string)

test_string

#This seems to be fine. Now we can apply above functions on our dataframe. Lets check our dataframe

#again.



training_df.head(5)
#Lets take backup of un-processed text, we might need it for future functions.

#We will perform all actions on testing_df aswell to avoid any errors in future.

training_df["text_backup"] = training_df["text"] #Creating new column text_backup same as text.

testing_df["text_backup"] = testing_df["text"] #Creating new column text_backup same as text.





#Applying above made functions on text.

training_df["text"] = training_df["text"].apply(lambda x:remove_punctuations_from_string(x))

training_df["text"] = training_df["text"].apply(lambda x:remove_stopwords_from_string(x))

testing_df["text"] = testing_df["text"].apply(lambda x:remove_punctuations_from_string(x))

testing_df["text"] = testing_df["text"].apply(lambda x:remove_stopwords_from_string(x))



training_df.head(5)
#Now we have processed and pre-processed text in our dataframe. Lets start making features from

#the above data.



#Initially we will create the basic features: 1 - Count of words in a statement(Vocab size), 

#2 - Count of characters in a statement & 3 - Diversity_score.



#In most of the cases above 3 features display the variations between writing styles of the authors.



#Feature 1 - Length of the input OR count of the words in the statement(Vocab size).

training_df['Feature_1']= training_df["text_backup"].apply(lambda x: len(str(x).split()))

testing_df['Feature_1']= testing_df["text_backup"].apply(lambda x: len(str(x).split()))



#Feature 2 - Count of characters in a statement

training_df['Feature_2'] = training_df["text_backup"].apply(lambda x: len(str(x)))

testing_df['Feature_2'] = testing_df["text_backup"].apply(lambda x: len(str(x)))



#Feature 3-Diversity_score i.e. Average length of words used in statement

training_df['Feature_3'] = training_df['Feature_2'] / training_df['Feature_1']

testing_df['Feature_3'] = testing_df['Feature_2'] / testing_df['Feature_1']



training_df.head(5)
#The usage of stop words can be another writing pattern. So the fourth feature is count of stopwords.

#Feature_4 = Count of stopwords in the sentence.

stop_words = set(stopwords.words('english'))

training_df['Feature_4'] = training_df["text_backup"].apply(lambda x: len([w for w in str(x).lower().split() if w in stop_words]) )

testing_df['Feature_4'] = testing_df["text_backup"].apply(lambda x: len([w for w in str(x).lower().split() if w in stop_words]) )



#Let us identify the highest used words(other than stopwords) in our input data for further feature generation.



#getting all text in single list: Though there are several other quicker options to do this, but

#this is the most accurate and convinient of them.

all_text_without_sw = ''

for i in training_df.itertuples():

    all_text_without_sw = all_text_without_sw +  str(i.text)

#getting counts of each words:

counts = Counter(re.findall(r"[\w']+", all_text_without_sw))

#deleting ' from counts

del counts["'"]

#getting top 50 used words:

sorted_x = dict(sorted(counts.items(), key=operator.itemgetter(1),reverse=True)[:50])



#Feature-5: The count of top used words.

training_df['Feature_5'] = training_df['text'].apply(lambda x: len([w for w in str(x).lower().split() if w in sorted_x]) )

testing_df['Feature_5'] = testing_df['text'].apply(lambda x: len([w for w in str(x).lower().split() if w in sorted_x]) )

#Similarly lets identify the least used words:

reverted_x = dict(sorted(counts.items(), key=operator.itemgetter(1))[:10000])

#Feature-6: The count of least used words.

training_df['Feature_6'] = training_df['text'].apply(lambda x: len([w for w in str(x).lower().split() if w in reverted_x]) )

testing_df['Feature_6'] = testing_df['text'].apply(lambda x: len([w for w in str(x).lower().split() if w in reverted_x]) )



#Feature-7: Count of punctuations in the input.

training_df['Feature_7'] = training_df['text_backup'].apply(lambda x: len([w for w in str(x) if w in string.punctuation]) )

testing_df['Feature_7'] = testing_df['text_backup'].apply(lambda x: len([w for w in str(x) if w in string.punctuation]) )

#Let's plot these features on a chart. To view these feature we will write a function:

def plot_bar_chart_from_dataframe(dataframe1,key_column,columns_to_be_plotted):

    import pandas as pd

    test_df1 = dataframe1.groupby(key_column).sum()

    test_df2 = pd.DataFrame()

    for column in columns_to_be_plotted:

        test_df2[column] = round(test_df1[column]/ test_df1[column].sum()*100,2)

    test_df2 = test_df2.T 

    ax = test_df2.plot(kind='bar', stacked=True, figsize =(10,5),legend = 'reverse',title = '% usage for each author')

    for p in ax.patches:

        a = p.get_x()+0.4

        ax.annotate(str(p.get_height()), (a, p.get_y()), xytext=(5, 10), textcoords='offset points')



key_column = 'author'

columns_to_be_plotted = ['Feature_4','Feature_5','Feature_6','Feature_7']

plot_bar_chart_from_dataframe(training_df,key_column,columns_to_be_plotted)
#Feature-8: Count of UPPER case words.

training_df['Feature_8'] = training_df['text_backup'].apply(lambda x: len([w for w in str(x).replace('I','i').replace('A','a').split() if w.isupper() == True]) )

testing_df['Feature_8'] = testing_df['text_backup'].apply(lambda x: len([w for w in str(x).replace('I','i').replace('A','a').split() if w.isupper() == True]) )



#Feature-9: Count of Title case words.

training_df['Feature_9'] = training_df['text_backup'].apply(lambda x: len([w for w in str(x).replace('I','i').replace('A','a').split() if w.istitle() == True]) )

testing_df['Feature_9'] = testing_df['text_backup'].apply(lambda x: len([w for w in str(x).replace('I','i').replace('A','a').split() if w.istitle() == True]) )



#The above features are common features which can indicate a writing pattern. There might be a possibility

#that a writer is using words which START WITH or END WITH particular characters. Lets try to identify them.



starting_words = sorted(list(map(lambda word : word[:2],filter(lambda word : len(word) > 3,all_text_without_sw.split()))))

sw_counts = Counter(starting_words)

top_30_sw = dict(sorted(sw_counts.items(), key=operator.itemgetter(1),reverse=True)[:30])



#Feature-10: Count of (Most words start with)

training_df['Feature_10'] = training_df['text'].apply(lambda x: len([w for w in str(x).lower().split() if w[:2] in top_30_sw and w not in stop_words]) )

testing_df['Feature_10'] = testing_df['text'].apply(lambda x: len([w for w in str(x).lower().split() if w[:2] in top_30_sw and w not in stop_words]) )



#Feature-11: Count of (Most words end with)

ending_words = sorted(list(map(lambda word : word[-2:],filter(lambda word : len(word) > 3,all_text_without_sw.split()))))

ew_counts = Counter(ending_words)

top_30_ew = dict(sorted(sw_counts.items(), key=operator.itemgetter(1),reverse=True)[:30])

training_df['Feature_11'] = training_df['text'].apply(lambda x: len([w for w in str(x).lower().split() if w[:2] in top_30_ew and w not in stop_words]) )

testing_df['Feature_11'] = testing_df['text'].apply(lambda x: len([w for w in str(x).lower().split() if w[:2] in top_30_ew and w not in stop_words]) )



list_of_cities_excel_file ='../input/city-country/list_of_cities.xlsx'

city_df = pd.read_excel(open(list_of_cities_excel_file,'rb'), sheet_name='Sheet1')



city_df.head(5)
city_list = city_df['City'].tolist()

city_list = [x.lower() for x in city_list]



#We have list of cities used in the input text. We will create our Feature 12 on the basis of this list.



#Feature-12: Count of City names used.

training_df['Feature_12'] = training_df['text'].apply(lambda x: len([w for w in str(x).lower().split() if w in city_list]) )

testing_df['Feature_12'] = testing_df['text'].apply(lambda x: len([w for w in str(x).lower().split() if w in city_list]) )



country_list = city_df['Country'].tolist()

country_list = [x.lower() for x in country_list]



#Feature-13: Count of Country names used.

training_df['Feature_13'] = training_df['text'].apply(lambda x: len([w for w in str(x).lower().split() if w in country_list]) )

testing_df['Feature_13'] = testing_df['text'].apply(lambda x: len([w for w in str(x).lower().split() if w in country_list]) )



#Function for getting Trigram from text:

def ngram_list_from_string(string1,count_of_words_in_ngram):

    string1 = string1.lower()

    string1 = string1.replace('.','. ')

    all_grams = ngrams(string1.split(), count_of_words_in_ngram)

    grams_list = []

    for grams in all_grams:

        grams_list.append(grams)

    return(grams_list)
#Getting Trigram for text:

ngram_list = ngram_list_from_string(all_text_without_sw,3)



#Getting count for every ngram:

ngram_counts = Counter(ngram_list)



#Getting top 10 ngram as per highest count:

sorted_ngram = dict(sorted(ngram_counts.items(), key=operator.itemgetter(1),reverse=True)[:10])



#Feature-14: Top 10 trigram occurence:

training_df['Feature_14'] = training_df['text_backup'].apply(lambda x: len([w for w in ngram_list_from_string(x,3)if w in sorted_ngram]) )

testing_df['Feature_14'] = testing_df['text_backup'].apply(lambda x: len([w for w in ngram_list_from_string(x,3)if w in sorted_ngram]) )



tokenized_all_text = word_tokenize(all_text_without_sw) #tokenize the text

list_of_tagged_words = nltk.pos_tag(tokenized_all_text) #adding POS Tags to tokenized words



set_pos  = (set(list_of_tagged_words)) # set of POS tags & words



nouns = ['NN','NNS','NNP','NNPS'] #POS tags of nouns

list_of_words = set(map(lambda tuple_2 : tuple_2[0], filter(lambda tuple_2 : tuple_2[1] in  nouns, set_pos)))

training_df['Feature_15'] = training_df['text'].apply(lambda x: len([w for w in str(x).lower().split() if w in list_of_words]) )

testing_df['Feature_15'] = testing_df['text'].apply(lambda x: len([w for w in str(x).lower().split() if w in list_of_words]) )



pronouns = ['PRP','PRP$','WP','WP$'] # POS tags of pronouns

list_of_words = set(map(lambda tuple_2 : tuple_2[0], filter(lambda tuple_2 : tuple_2[1] in  pronouns, set_pos)))

training_df['Feature_16'] = training_df['text'].apply(lambda x: len([w for w in str(x).lower().split() if w in list_of_words]) )

testing_df['Feature_16'] = testing_df['text'].apply(lambda x: len([w for w in str(x).lower().split() if w in list_of_words]) )



verbs = ['VB','VBD','VBG','VBN','VBP','VBZ'] #POS tags of verbs

list_of_words = set(map(lambda tuple_2 : tuple_2[0], filter(lambda tuple_2 : tuple_2[1] in  verbs, set_pos)))

training_df['Feature_17'] = training_df['text'].apply(lambda x: len([w for w in str(x).lower().split() if w in list_of_words]) )

testing_df['Feature_17'] = testing_df['text'].apply(lambda x: len([w for w in str(x).lower().split() if w in list_of_words]) )



adverbs = ['RB','RBR','RBS','WRB'] #POS tags of adverbs

list_of_words = set(map(lambda tuple_2 : tuple_2[0], filter(lambda tuple_2 : tuple_2[1] in  adverbs, set_pos)))

training_df['Feature_18'] = training_df['text'].apply(lambda x: len([w for w in str(x).lower().split() if w in list_of_words]) )

testing_df['Feature_18'] = testing_df['text'].apply(lambda x: len([w for w in str(x).lower().split() if w in list_of_words]) )



adjectives = ['JJ','JJR','JJS'] #POS tags of adjectives

list_of_words = set(map(lambda tuple_2 : tuple_2[0], filter(lambda tuple_2 : tuple_2[1] in  adjectives, set_pos)))

training_df['Feature_19'] = training_df['text'].apply(lambda x: len([w for w in str(x).lower().split() if w in list_of_words]) )

testing_df['Feature_19'] = testing_df['text'].apply(lambda x: len([w for w in str(x).lower().split() if w in list_of_words]) )
