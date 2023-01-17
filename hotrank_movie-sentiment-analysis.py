import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt # 画图常用库


train_raw = pd.read_csv('../input/labeledTrainData.tsv', delimiter="\t")
test_raw = pd.read_csv('../input/testData.tsv', delimiter="\t")
train_raw.head()   
print (train_raw.shape)
print (test_raw.shape)
from tqdm import tqdm
import re
from nltk.corpus import stopwords # Import the stop word list

stop_words = set(stopwords.words('english')) # making the list to set improves the speed

def rm_stop_words(text):
    words = text.split()
    words = [word for word in words if not word in stop_words]
    new_text = ' '.join(words)
    return new_text

num_train = train_raw.shape[0]
num_test = test_raw.shape[0]
train = train_raw.copy(deep = True)
test = test_raw.copy(deep = True)

for i in tqdm(range(num_train)):   
    text = re.sub('[^a-zA-Z0-9]', ' ', train_raw.loc[i, 'review']).lower()
    train.loc[i,'review'] = rm_stop_words(text)
for i in tqdm(range(num_test)):
    text = re.sub('[^a-zA-Z0-9]', ' ', test_raw.loc[i, 'review']).lower()
    test.loc[i,'review'] = rm_stop_words(text)

# check result
print(train.shape)    
print(train_raw.loc[24900, 'review'])
print(train.loc[24900, 'review'])
# from time import time

def create_dict(data_df):
    vocab = {}
    i = 0
    for review in data_df['review']:
        words = review.split()
        for word in words:
            if word not in vocab.keys():
                vocab[word] = i
                i += 1
    return vocab

# start = time()
# create a dictionary using all the text in training data set.
vocab = create_dict(train)
# end = time()
# print('time = %s seconds' %(str(end - start)))


print(len(vocab.keys()))

        
def text2vector(text, vocab):
    vect = np.zeros(len(vocab.keys()))
    words = text.split()
    for word in words:
        if word in vocab.keys():
            vect[vocab[word]] += 1
    return vect  

text2vector(train.loc[0, 'review'], vocab)
from sklearn.model_selection import train_test_split

X_train, X_vali, y_train, y_vali = train_test_split(train['review'], train['sentiment'], test_size = 0.25, random_state = 0)

y_train.iloc[0]
X_train.shape
def naive_bayes_train(train, label, vocab):
    count_pos_review, count_neg_review = 0, 0
    pos_tf_vector = np.ones(len(vocab.keys()))
    neg_tf_vector = np.ones(len(vocab.keys()))
    df_count = np.ones(len(vocab.keys()))  # to record the number of documents that contain each word

    for i, text in tqdm(enumerate(train)):
        if label.iloc[i] == 1:
            text_vector = text2vector(text, vocab)
            pos_tf_vector += text_vector/text_vector.sum()
            df_count += np.array([1 if i else 0 for i in text_vector]) # if a word is in document, 1, else, 0
            count_pos_review += 1
        else:
            text_vector = text2vector(text, vocab)
            neg_tf_vector += text_vector/text_vector.sum()
            df_count += np.array([1 if i else 0 for i in text_vector]) # if a word is in document, 1, else, 0
            count_neg_review += 1


    p_pos = count_pos_review / (count_pos_review + count_neg_review)
    p_neg = 1 - p_pos
    idf_vector = np.log((count_pos_review + count_neg_review)/df_count)
    pos_tfidf_vector = pos_tf_vector * idf_vector
    neg_tfidf_vector = neg_tf_vector * idf_vector
    
    p_pos_vector = pos_tfidf_vector/pos_tfidf_vector.sum()
    p_neg_vector = neg_tfidf_vector/neg_tfidf_vector.sum()
    
    return p_pos, p_neg, p_pos_vector, p_neg_vector, idf_vector

p_pos, p_neg, p_pos_vector, p_neg_vector, idf_vector = naive_bayes_train(X_train, y_train, vocab)

print(p_pos, p_neg) 
idf_vector
from sklearn.metrics import accuracy_score

def predict(text, p_pos, p_neg, p_pos_vector, p_neg_vector, vocab, idf_vector):
    text_vector = text2vector(text, vocab)
    text_tfidf_vector = text_vector / text_vector.sum() * idf_vector
    pos = (text_tfidf_vector * np.log(p_pos_vector)).sum() + np.log(p_pos)
    neg = (text_tfidf_vector * np.log(p_neg_vector)).sum() + np.log(p_neg)
    if pos >= neg:
        return 1
    else:
        return 0
    
y_pred = []
for i, text in tqdm(enumerate(X_vali)):
    ans = predict(text, p_pos, p_neg, p_pos_vector, p_neg_vector, vocab, idf_vector)
    y_pred.append(ans)

score = accuracy_score(y_vali, y_pred)
print(score)

predictions = []

for text in tqdm(test['review']):
    predictions.append(predict(text, p_pos, p_neg, p_pos_vector, p_neg_vector, vocab, idf_vector))
    
print(test.shape, len(predictions))

output_df = pd.DataFrame({'id': test['id'], 'sentiment': predictions})
output_df.to_csv('submission_baseline_stopwords_tfidf.csv', index = False, header = True)