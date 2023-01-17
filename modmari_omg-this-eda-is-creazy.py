import os
import time
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from tqdm import tqdm
import math
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import f1_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from sys import getsizeof
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import string
from string import digits
import re
import operator 
plt.style.use('seaborn-darkgrid')
# https://www.kaggle.com/theoviel/improve-your-score-with-some-text-preprocessing
def load_embed(file):
    def get_coefs(word,*arr): 
        return word, np.asarray(arr, dtype='float32')
    
    if file == '../input/embeddings/wiki-news-300d-1M/wiki-news-300d-1M.vec':
        embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(file) if len(o)>100)
    else:
        embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(file, encoding='latin'))
        
    return embeddings_index
# https://www.kaggle.com/theoviel/improve-your-score-with-some-text-preprocessing
def build_vocab(texts):
    sentences = texts.apply(lambda x: x.split()).values
    vocab = {}
    for sentence in sentences:
        for word in sentence:
            try:
                vocab[word] += 1
            except KeyError:
                vocab[word] = 1
    return vocab
# https://www.kaggle.com/theoviel/improve-your-score-with-some-text-preprocessing
def check_coverage(vocab, embeddings_index):
    known_words = {}
    unknown_words = {}
    nb_known_words = 0
    nb_unknown_words = 0
    for word in vocab.keys():
        try:
            known_words[word] = embeddings_index[word]
            nb_known_words += vocab[word]
        except:
            unknown_words[word] = vocab[word]
            nb_unknown_words += vocab[word]
            pass

    print('Found embeddings for {:.2%} of vocab'.format(len(known_words) / len(vocab)))
    print('Found embeddings for  {:.2%} of all text'.format(nb_known_words / (nb_known_words + nb_unknown_words)))
    unknown_words = sorted(unknown_words.items(), key=operator.itemgetter(1))[::-1]

    return unknown_words
def plot_learning_curve(history,model_info):
    # summarize history for loss
    plt.figure(figsize=(9,7))
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title(f'Model {model_info} Loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig(f'Model_{model_info}_Loss.png')
    
    # summarize history for accuracy
    plt.figure(figsize=(9,7))
    plt.plot(history['acc'])
    plt.plot(history['val_acc'])
    plt.title(f'Model {model_info} Accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig(f'Model_{model_info}_Acc.png')
    print(f'Model_{model_info}')
def load_combined():
    train_data = pd.read_csv("../input/train.csv")
    test_data = pd.read_csv("../input/test.csv")
    
    # Get the number of samples in training data :
    train_size = train_data.shape[0]
    test_size =  test_data.shape[0]
    
    # Combine training and test data :
    combined = train_data.append(test_data, ignore_index  = True, sort = False)
    
    ## fill in nans :
    combined["question_text"].fillna('NAN', inplace = True)
    
    return combined, train_size, test_size

combined, train_size, test_size = load_combined()
question_text = combined["question_text"].copy()
%%time
# Get Words :
def get_words(samples):
    words = set()
    for sample in samples.values:
        for word in sample.split():
            words.add(word)
    num_words = len(words)
    print(f'Number of unique words : {num_words}')
    return words
num_toxic    = combined[combined['target'] == 1 ].count()[0]
num_nontoxic = combined[combined['target'] == 0 ].count()[0]
print(f'Number of toxic samples    : {num_toxic}')
print(f'Number of nontoxic samples : {num_nontoxic}')

print(f'{round((num_toxic/train_size)*100,2)}% of the samples in the training data is Toxic')
# Number of words without processing the dataset :
_ = get_words(samples = combined["question_text"])
# Convert text to lowercase :
combined["question_text"] = combined["question_text"].apply(lambda x: x.lower())
# Number of words after converting the samples to lower case :
_ = get_words(samples = combined["question_text"])
%%time
# Process commas :
combined["question_text"] = combined["question_text"].apply(lambda x: re.sub("'", '', x)).apply(lambda x: re.sub(",", ' COMMA', x))

# Getting rid of punctuation
exclude = set(string.punctuation)
combined["question_text"] = combined["question_text"].apply(lambda x: ''.join(ch for ch in x if ch not in exclude))
# Number of words after dealing with punctuation :
_ = get_words(samples = combined["question_text"])
# Getting rid of digits
remove_digits = str.maketrans('', '', digits)
combined["question_text"] = combined["question_text"].apply(lambda x: x.translate(remove_digits))
# Number of words without digits :
words = get_words(samples = combined["question_text"])
max_features = len(words)
# prepare tokenizer
tokenizer = Tokenizer(num_words = max_features)
tokenizer.fit_on_texts(combined['question_text'])
# integer encode the documents
encoded_samples = tokenizer.texts_to_sequences(combined['question_text'])
combined['encoded_samples'] = encoded_samples
# Get samples length :
combined['sample_len'] = combined['encoded_samples'].apply(len)
max_len = combined['sample_len'].max()
min_len = combined['sample_len'].min()
avg_len = int(combined['sample_len'].mean())

print(f'The longest  sample has {max_len} words')
print(f'The shortest sample has {min_len} words')
print(f'The average number of words in samples is {avg_len}')
empty_samples_idx = combined[combined['sample_len'] <= 0].index
question_text[empty_samples_idx]
oneWord_samples_idx = combined[combined['sample_len'] == 1].index
temp = pd.DataFrame({'question_text':question_text[oneWord_samples_idx],
              'target':combined['target'][oneWord_samples_idx]})
temp
# Percentage of toxic samples in the dataset :
toxic_perc = round((combined['target'].sum()/train_size)*100,2)
print(f'Percentage of toxic samples in the dataset : {toxic_perc}%')
t = round((temp['target'].sum()/combined['target'].sum())*100,4)
print(f'Percentage of toxic samples with one word only over the other toxic samples : {t}%')
def get_count(df,col,min_len,max_len):
    return df[(df[col]>=min_len) & (df[col]<=max_len)].count()[0]
def plot_sample_len(df,col,title = 'Lengths of Samples',sp=10000):
    # Get the range of lengths and number of samples for each range (for test data)
    ranges = [(0,5), (6,10), (11,15), (16,20), (21,30), (31,40),
              (41,50), (51,65), (66,80), (81,100), (101,150) ]
    range_name = []
    num_samps_in_range = []
    for r in ranges:
        num_samps_in_range.append(
            get_count(df = df, col = col, min_len=r[0], max_len=r[1]))
        range_name.append(f'{r[0]} -> {r[1]}')

        # Plot range of lengths and number of samples for each range :
    fig, ax = plt.subplots(figsize = (18, 10),)
    ax.set(title = title,
           xlabel = ' Length (# words)', ylabel = '# Samples')

    r1 = ax.bar(range_name,
                num_samps_in_range,
                alpha = 0.9,
                label = '# Samples')
    for idx in range(len(ranges)) : 
        if(num_samps_in_range[idx] > 10000):
            ax.text(range_name[idx],
                    num_samps_in_range[idx]+(sp),
                    num_samps_in_range[idx],
                    horizontalalignment='center',
                    size='large')
        else :
            ax.text(range_name[idx],
                num_samps_in_range[idx]+100,
                num_samps_in_range[idx],
                horizontalalignment='center',
                size='large')
# ranges of sample length for all combined data :
plot_sample_len(df = combined,col = 'sample_len',
                title = 'Lengths of All Samples',sp=10000)
# ranges of sample length for training data :
plot_sample_len(df = combined[:train_size],col = 'sample_len',
                title = 'Lengths of Training Samples', sp=10000)
# ranges of sample length for test data :
plot_sample_len(df = combined[train_size:],col = 'sample_len',
                title = 'Lengths of Test Samples', sp=500)
glove = '../input/embeddings/glove.840B.300d/glove.840B.300d.txt'
paragram =  '../input/embeddings/paragram_300_sl999/paragram_300_sl999.txt'
wiki_news = '../input/embeddings/wiki-news-300d-1M/wiki-news-300d-1M.vec'
%%time
print("Extracting GloVe embedding ...")
embed_glove = load_embed(glove)
print("Extracting Paragram embedding ...")
embed_paragram = load_embed(paragram)
print("Extracting FastText embedding ...")
embed_fasttext = load_embed(wiki_news)
print('Done!')
vocab = build_vocab(combined['question_text'])
print("Glove : ")
oov_glove = check_coverage(vocab, embed_glove)
print("Paragram : ")
oov_paragram = check_coverage(vocab, embed_paragram)
print("FastText : ")
oov_fasttext = check_coverage(vocab, embed_fasttext)
# Get : words and thier count and words that appeard only once :
word_freq  = []
all_words  = []
rare_words = []
for i,w in enumerate(tokenizer.word_counts):
    word_freq.append(tokenizer.word_counts[w])
    all_words.append(w)
    if(tokenizer.word_counts[w] <= 1):
        rare_words.append(w)
num_all_words  = len(all_words)
num_rare_words = len(rare_words)
print(f'Total number of words                     : {num_all_words}')
print(f'Number of words that appeared only once    : {num_rare_words}')
print(f'{round((num_rare_words/num_all_words)*100,2)}% of the words appeared only once')
# Sort the words by their frequency :
common_words       = []
common_words_count = []
for y,x in sorted(zip(word_freq, all_words),reverse = True):
    common_words.append(x)
    common_words_count.append(y)
# Plot the most 10 common words :
fig, ax = plt.subplots(figsize = (18, 10),)
ax.set(title = 'Common Words',
       xlabel = ' Word', ylabel = 'Count')

r1 = ax.bar(common_words[:10],
            common_words_count[:10],
            alpha = 0.9,
            label = '# Count')
for idx in range(10) : 
    ax.text(common_words[idx],
            common_words_count[idx]+10000,
            common_words_count[idx],
            horizontalalignment='center',
            size='small')
# The most 100 common words :
top100Words = pd.DataFrame({'Word':common_words[:100], 'Count':common_words_count[:100]})
print('The Top 10 Words :')
top100Words.head(10)
# Get a list of the words in the toxic samples:
toxic_samples = list(combined['encoded_samples'][combined['target'] == 1].values)
toxic_samples_words = [item for sublist in toxic_samples for item in sublist]
toxic_words = list(set(toxic_samples_words))
print(f'There are {len(toxic_words)} unique words in the toxic samples')
# Extract non-toxic samples :
nontoxic_samples = list(combined['encoded_samples'][combined['target'] == 0].values)

# Get all the words in the non-toxic samples : 
nontoxic_samples_words = [item for sublist in nontoxic_samples for item in sublist]
nontoxic_samples_words = list(set(nontoxic_samples_words))

# Get words that only appeared in toxic samples :
toxic_only_words = list(set(toxic_words) - set(nontoxic_samples_words))
print(f'Words that apeared only in toxic samples are {len(toxic_only_words)} words')
# Create a dictionary to convert word index into word :
idx2word = {}
for k, v in tokenizer.word_index.items():
    idx2word[v] = k

# Convert indecies to words :
toxic_only_words_idx = toxic_only_words.copy()
toxic_only_words = [idx2word[x] for x in toxic_only_words_idx]
# print some of the words that appeared only in the toxic samples :  
temp = pd.DataFrame({ '1-20' :toxic_only_words[:20],
                      '20-40':toxic_only_words[20:40]})
print('Some of the words that appeared only in the toxic samples :')
temp
# Deleting words that appeared only one time
toxic_only_words = list(set(toxic_only_words) - set(rare_words))
print(f'Number of toxic words after deleting the rare words : {len(toxic_only_words)} words')
# Get top 10 toxic words by their count :
temp = pd.DataFrame({'toxic word':toxic_only_words})
temp['count'] = temp['toxic word'].apply(lambda x: tokenizer.word_counts[x])
temp.sort_values(by=['count'], ascending = False, inplace = True)
temp = temp.reset_index(drop = True)[:10]
# Plot the most 10 common toxic words :
fig, ax = plt.subplots(figsize = (12, 10),)
ax.set(title = 'Common Toxic Words',
       xlabel = 'Count', ylabel = 'Toxic Word')

r1 = ax.barh(temp['toxic word'][:10],
            temp['count'][:10],
            alpha = 0.7,
            color = '#EF4A6D')
ax.invert_yaxis()
for idx in range(10) : 
    ax.text(temp['count'][idx]-0.2,
            idx,
            temp['count'][idx],
            horizontalalignment='right',
            size='medium')
