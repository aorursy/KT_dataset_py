import numpy as np
import nltk
from nltk.tokenize import word_tokenize
import pandas as pd
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import re
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import *
train_df = pd.read_csv("../input/tweet-sentiment-extraction/train.csv")
test_df = pd.read_csv("../input/tweet-sentiment-extraction/test.csv")
submission_df = pd.read_csv("../input/tweet-sentiment-extraction/sample_submission.csv")
train_df['text'] = train_df['text'].str.lower()
test_df['text'] = test_df['text'].str.lower()
train_df['selected_text'] = train_df['selected_text'].str.lower()
def jaccard(str1, str2): 
    a = set(str1.lower().split()) 
    b = set(str2.lower().split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))

def evaluation(actual_list, pred_list):
    score = 0
    for (actual, pred) in zip(actual_list, pred_list):
        score += jaccard(actual, pred)
    return score / len(pred)
lm = WordNetLemmatizer()
def Remove_Special_Char(text):
    text = re.sub('[-=+,#/\?:^$.@*\"※~&%ㆍ!』\\‘|\(\)\[\]\<\>`\'…》]', '', text) 
    return text
# remove special characters from selected text 
train_df['selected_text'] = train_df['selected_text'].astype(str).apply(Remove_Special_Char)
sentiment_value_counts = train_df['sentiment'].value_counts()

# initialize sentiment score dictionary
sentiment_score = {'neutral': dict(), 
                  'positive': dict(),
                  'negative': dict()}

# generate selected text corpus
selected_text_corpus = train_df['selected_text'].values.flatten()
selected_text_corpus = np.array(selected_text_corpus, dtype = str)
selected_text_corpus = ' '.join(selected_text_corpus)

for selected_text, sentiment in zip(train_df['selected_text'], train_df['sentiment']):
    word_list_in_selected_text = word_tokenize(selected_text)    
    for word in word_list_in_selected_text:
        lemmatized_word = lm.lemmatize(word)
        if lemmatized_word in sentiment_score[sentiment].keys():
            sentiment_score[sentiment][lemmatized_word] += 1
        else:
            sentiment_score[sentiment][lemmatized_word] = 1
                
for sentiment in sentiment_score.keys():
    expected_value = sentiment_value_counts[sentiment] / sum(sentiment_value_counts)
    for word in sentiment_score[sentiment].keys():
        word_frequency = sentiment_score['positive'].get(word, 0) + sentiment_score['neutral'].get(word, 0) + sentiment_score['negative'].get(word, 0)
        actual_value = sentiment_score[sentiment][word] / word_frequency
        sentiment_score[sentiment][word] = actual_value - expected_value
train_df['text'] = train_df['text'].astype(str).apply(Remove_Special_Char)
train_df['selected_text'] = train_df['selected_text'].astype(str).apply(Remove_Special_Char)

train_df['tokend_text'] = train_df['text'].apply(word_tokenize)
train_df['tokend_selected_text'] = train_df['selected_text'].apply(word_tokenize)
def find_neighbor(t, window_size):
    T = np.arange(0, 100)
    return np.argsort(np.abs(T - t))[:window_size]
def generate_dataset(df, window_size):
    X = []; Y = []
    for tokend_text, tokend_selected_text, sentiment in zip(df['tokend_text'], df['tokend_selected_text'], df['sentiment']):
        try:
            s, e = [(i, i+len(tokend_selected_text)) for i in range(len(tokend_text)) if tokend_text[i:i+len(tokend_selected_text)] == tokend_selected_text][0] #s: start point of tokend_selected_text in tokend_text // e: end point of tokend_selected_text in tokend_text
        except:
            s, e = (0, 0)
        y = [0] * s + [1] * (e-s) + [0] * (len(tokend_text) - e)
        x = []
        for word in tokend_text:
            lemmatized_word = lm.lemmatize(word)
            x.append(sentiment_score[sentiment].get(word, 0))        
        
        x = np.array(x)
        y = np.array(y)
        for t in range(len(x)):            
            neighbor = find_neighbor(t, window_size)
            try:
                X.append(x[neighbor])
                Y.append(y[t])
            except:
                pass

    return X, Y
X, Y = generate_dataset(df = train_df, window_size = 3)
model = SVC(kernel = 'linear').fit(X, Y)
def make_prediction(test_df, model, window_size):
    test_df['text'] = test_df['text'].astype(str).apply(Remove_Special_Char)
    test_df['tokend_text'] = test_df['text'].apply(word_tokenize)
    result = []
    for tokend_text, sentiment in zip(test_df['tokend_text'], test_df['sentiment']):
        x = []
        for word in tokend_text:
            lemmatized_word = lm.lemmatize(word)
            x.append(sentiment_score[sentiment].get(word, 0))

        X = []
        x = np.array(x)
        for t in range(len(x)):            
            neighbor = find_neighbor(t, window_size)
            try:
                X.append(x[neighbor])
            except:
                pass        
        
        try:
            pred_Y = model.predict(X)
            pred_sentence = ''
            for (word, y) in zip(tokend_text, pred_Y):
                if y == 1:
                    pred_sentence += word + ' '            
            while pred_sentence[-1] == ' ':
                pred_sentence = pred_sentence[:-1]
            result.append(pred_sentence)
        except:
            result.append('')
    
    return result
result = make_prediction(test_df, model, window_size = 3)
submission_df['selected_text'] = result
submission_df.to_csv('submission.csv', index=False)
