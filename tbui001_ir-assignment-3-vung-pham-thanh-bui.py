from nltk.corpus import stopwords,words

from nltk.tokenize import word_tokenize

from nltk.stem import SnowballStemmer

from string import punctuation

stemmer = SnowballStemmer('english')



def process_sentence(sentence, tokens_to_remove, English_words, max_tokens=None):

    words = word_tokenize(sentence) # Tokenize

    if max_tokens is not None and len(words) < max_tokens:

        return None

    else:

        words = [w.lower() for w in words if not w.isdigit()] # Convert to lowercase and also remove digits

        filter_words = [stemmer.stem(w) for w in words if w not in tokens_to_remove and w in English_words] # remove tokens + check english words + stem

        return filter_words
%%time

import csv

max_docs = 30000 # test with this number of docs first. If would like to do for all docs, set this value to None

review_outfile = 'review_text.txt'



stop_words = set(stopwords.words('english'))

tokens_to_remove = stop_words.union(set(punctuation))

English_words = set(words.words())





doc_count = 0



with open('../input/amazon-review-testset/test.csv','rt',encoding='utf-8') as rf:

    with open(review_outfile, 'w') as outputfile:

        reader = csv.reader(rf, delimiter=',')

        for row in reader:

            score = int(row[0])-1 #fit into class for easier work later

            review = process_sentence(row[1], tokens_to_remove, English_words, 100)     

            if review is not None:

                outputfile.writelines(str(score) + ", " + " ".join(review) + '\n') # write the results

                doc_count += 1

                if  max_docs and doc_count >= max_docs: # if we do define the max_docs

                    break

    outputfile.close()

rf.close()
# View the file if needed

from IPython.display import FileLink

FileLink('review_text.txt')
from sklearn.model_selection import train_test_split

import numpy as np

from numpy import savetxt



with open("review_text.txt", "rt") as infile:

    data = infile.read().split('\n')

    data = np.array(data)



data = np.delete(data,-1) 



train_data,test_data = train_test_split(data,test_size=0.15,random_state = 1) #set the seed to 1 for reproducibility



#Save data into csv files

savetxt('train_set.csv', train_data, delimiter=',',fmt='%s')

savetxt('test_set.csv', test_data, delimiter=',', fmt='%s')
len(train_data)
len(test_data)
# View the file if needed

FileLink('train_set.csv')
# View the file if needed

FileLink('test_set.csv')
import wordcloud

import matplotlib.pyplot as plt

def show_wordcloud(text, title=None):

    # Create and generate a word cloud image:

    wc = wordcloud.WordCloud(background_color='white').generate(text)

    # Display the generated image:

    plt.figure(figsize=(10, 10))

    plt.imshow(wc, interpolation='bilinear')

    plt.axis("off")

    if title is not None:

        plt.title(title)

    plt.show()
# Read data

import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer



train_df = pd.read_csv("train_set.csv", header=None)

train_df.columns = ["label", "text"]

class_label = train_df['label'].values.astype(str)

train_document = train_df['text'].values
show_wordcloud(" ".join(train_df['text'].values), "Train set")
set(class_label)
for cls in set(class_label):

    show_wordcloud(" ".join(train_df[train_df['label']==int(cls)]['text'].values), f"Train set class {cls}")
feature_extraction = CountVectorizer()

trans_docs =  feature_extraction.fit_transform(train_document)
len(feature_extraction.vocabulary_)
len(feature_extraction.vocabulary_)
feature_set = pd.DataFrame(trans_docs.toarray(),columns=feature_extraction.get_feature_names())
feature_set.head()
top_words = pd.DataFrame(feature_set.sum()).sort_values(0, ascending=False)

top_words.columns=['counts']

top_words.head(10)
import matplotlib.pyplot as plt

plt.bar(list(top_words.head(10).index), sorted(top_words.head(10)['counts'].values))

plt.title("Top words in the training set")
categories, counts = np.unique(class_label, return_counts=True)

class_prob = counts/len(class_label)

print(class_prob)
import matplotlib.pyplot as plt

fig1, ax1 = plt.subplots()

ax1.pie(class_prob, labels=categories, autopct='%1.1f%%', shadow=True, startangle=90)

ax1.axis('equal')

plt.show()
class_code = sorted(categories)
class_code
def get_word_prob_for_class(cls):

    value1 = feature_set[class_label==cls].values

    P_w_c =((np.sum(value1,axis = 0)+1)/(np.sum(value1)+len(value1[0]))) #smoothing fomular => must be len of the vocabulary

    return P_w_c
word_prob=np.array([get_word_prob_for_class(cls) for cls in class_code])
print(word_prob)
for i, wp in enumerate(word_prob):

    plt.figure()

    plt.title(f'Probability distribution of top 50 words in class {categories[i]}')

    plt.hist([np.array(sorted(wp))[-50:]])

    plt.plot()
for i, wp in enumerate(word_prob):

    plt.figure()

    plt.title(f'Probability of top 10 words in class {categories[i]}')

    top_indices = np.argsort(wp)[-10:]

    top_words = feature_set.columns[top_indices]

    plt.bar(top_words, wp[top_indices])

    plt.plot()
saved_model = pd.DataFrame(word_prob, columns = feature_set.columns)
saved_model['class_prob'] = class_prob
saved_model.head()
saved_model.to_csv('model.csv')
# View the file if needed

FileLink('model.csv')
test_df = pd.read_csv('test_set.csv', header=None)

test_df.columns = ['label', 'text']

test_label = test_df['label']

test_document = test_df['text']

trans_test =  feature_extraction.transform(test_document)
show_wordcloud(" ".join(test_df['text'].values), "Test set")
for cls in set(class_label):

    show_wordcloud(" ".join(test_df[test_df['label']==int(cls)]['text'].values), f"Test set class {cls}")
categories, counts = np.unique(test_label, return_counts=True)

test_class_prob = counts/len(test_label)

print(test_class_prob)
fig1, ax1 = plt.subplots()

ax1.pie(test_class_prob, labels=categories, autopct='%1.1f%%', shadow=True, startangle=90)

ax1.axis('equal')

plt.show()
test_feature_set = pd.DataFrame(trans_test.toarray(),columns=feature_extraction.get_feature_names())

test_top_words = pd.DataFrame(test_feature_set.sum()).sort_values(0, ascending=False)

test_top_words.columns=['counts']

test_top_words.head(10)
plt.bar(list(test_top_words.head(10).index), sorted(test_top_words.head(10)['counts'].values))
test_feature_set.head()
feature_set.head()
saved_model
def get_prob_of_class(cls):

    return saved_model['class_prob'][int(cls)]

def get_word_freqency_from_doc(word, docIdx):

    return test_feature_set[word][docIdx]

def get_prob_word_given_class(word, cls):

    return saved_model[word][int(cls)]

def get_prob_for_words_in_doc_given_cls(docIdx, cls):

    return [get_prob_word_given_class(word, cls)**get_word_freqency_from_doc(word, docIdx) for word in list(test_feature_set.columns) if get_word_freqency_from_doc(word, docIdx)!=0]

# def get_prob_for_words_in_doc_given_cls(docIdx, cls):

#     return [get_prob_word_given_class(word, cls) for word in list(test_feature_set.columns) if get_word_freqency_from_doc(word, docIdx)!=0]

def get_prob_doc_class(docIdx, cls):

    class_prob = saved_model['class_prob'][int(cls)]

    prod_word_probs = np.prod(get_prob_for_words_in_doc_given_cls(docIdx, cls))

    return class_prob * prod_word_probs
job_tuples = []

for docIdx in range(len(test_feature_set)):

    for cls in class_code:

        job_tuples.append((docIdx, cls))
%%time

import multiprocessing

with multiprocessing.Pool(processes = multiprocessing.cpu_count()-1) as pool:

    probs1 = pool.starmap(get_prob_doc_class, job_tuples)

probs1 = np.array(probs1).reshape(int(len(probs1)/5), 5)
# %%time

# probs = [[get_prob_doc_class(docIdx, cls) for cls in class_code] for docIdx in range(len(test_feature_set))]
predicted_class = np.argmax(probs1, axis=1)
# store the predicted class for visualization later/on

import json

with open('predicted_probs.json', 'w') as f:

    json.dump([list(probs) for probs in probs1], f)
FileLink('predicted_probs.json')
# store the predicted class for visualization later/on

import json

with open('predicted_class.json', 'w') as f:

    json.dump(list(predicted_class.astype(str)), f)
FileLink('predicted_class.json')
test_label
predicted_class
test_label
def confusion_type_for_class(cls, true_label, predicted_label):

    result={}

    result["true_positive"] = [true_label[i] == cls and predicted_label[i] == cls for i in range(len(true_label))]

    result["false_positive"] = [true_label[i] != cls and predicted_label[i] == cls for i in range(len(true_label))]

    result["false_negative"] = [true_label[i] == cls and predicted_label[i] != cls for i in range(len(true_label))]

    result["true_negative"] = [true_label[i] != cls and predicted_label[i] != cls for i in range(len(true_label))]

    return result
def num_to_scientific_power(num):

    pw = int("{:e}".format(num).split("e-")[1])-1

    return pw
def analyze_doc_class(docIdx, cls):

    text = test_data[docIdx]

    print(text)

    words = text.split(', ')[1].split(' ')

    words = [w for w in words if w in saved_model.columns]

    unique_words = list(set(words))    

    word_probs = [get_prob_word_given_class(word, cls) for word in unique_words]

    sorted_indexes = np.argsort(word_probs)[-15:]

    top_prob_words = [unique_words[i] for i in sorted_indexes]

    top_probs = [word_probs[i] for i in sorted_indexes]

    plt.figure()

    plt.bar(top_prob_words, top_probs)

    plt.xticks(rotation=90)

    _ = plt.title(f'Top words in this review for class {cls}')



    plt.figure()

    pw = num_to_scientific_power(max(probs1[docIdx]))

    plt.bar(np.arange(0,5), probs1[docIdx]*10**pw)

    _= plt.title('Probability for all classes (after multiplying by 1e115)')    
# Analyze for class 0 true positive

cls = 0

result = confusion_type_for_class(cls, test_label.values, predicted_class)

docIdx = result['true_positive'].index(True)

analyze_doc_class(docIdx, cls)
# Analyze for class 0 false positive

cls = 0

result = confusion_type_for_class(cls, test_label.values, predicted_class)

docIdx = result['false_positive'].index(True)

analyze_doc_class(docIdx, cls)
# Its actual class is

test_label[docIdx]
# Analyze for class 0 true negative

cls = 0

result = confusion_type_for_class(cls, test_label.values, predicted_class)

docIdx = result['true_negative'].index(True)

analyze_doc_class(docIdx, cls)
# Analyze for class 0 true negative

cls = 0

result = confusion_type_for_class(cls, test_label.values, predicted_class)

docIdx = result['false_negative'].index(True)

analyze_doc_class(docIdx, cls)
# distribution of the posterier for each class

for i in range(5):

    plt.figure()

    _ = plt.hist(probs1[:, 0])
def visualize_confusion_results(cls, test_df, true_label, predicted_label):

    confusion_result = confusion_type_for_class(cls, true_label, predicted_label)

    for cf in confusion_result.keys():

        show_wordcloud(" ".join(test_df[confusion_result[cf]]['text'].values), f"Test set class {cls}, type {cf}")
for cls in set(class_label):

    visualize_confusion_results(int(cls), test_df, test_label, predicted_class)
from sklearn.metrics import classification_report



print(classification_report(predicted_class, test_label))