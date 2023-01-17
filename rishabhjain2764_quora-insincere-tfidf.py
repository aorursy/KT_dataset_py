import numpy as np
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm_notebook as tqdm
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from wordcloud import WordCloud 
import string
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report
%matplotlib inline
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

print(train.shape)
print(test.shape)

train.head()
train['target'].value_counts()
print('Train data : ')
print("% of sincere questions : {:.2f}".format(train.target.value_counts()[0] / len(train)))
print("% of insincere questions : {:.2f}".format(train.target.value_counts()[1] / len(train)))
no_insincere = len(train[train.target == 1])
insincere_index = train[train.target == 1].index
sincere_index = train[train.target == 0].index
chosen_sincere = np.random.choice(sincere_index, no_insincere, replace = False)
mix_index = np.concatenate([chosen_sincere, insincere_index])
sampled_df = train.loc[mix_index]
del train
contractions_dict = {
    "ain't": "are not",
    "aren't": "are not",
    "can't": "cannot",
    "can't've": "cannot have",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "couldn't've": "could not have",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hadn't've": "had not have",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'd've": "he would have",
    "he'll": "he will",
    "he'll've": "he will have",
    "he's": "he is",
    "how'd": "how did",
    "how'd'y": "how do you",
    "how'll": "how will",
    "how's": "how is",
    "i'd": "i would",
    "i'd've": "i would have",
    "i'll": "i will",
    "i'll've": "i will have",
    "i'm": "i am",
    "i've": "i have",
    "isn't": "is not",
    "it'd": "it would",
    "it'd've": "it would have",
    "it'll": "it will",
    "it'll've": "it will have",
    "it's": "it is",
    "let's": "let us",
    "ma'am": "madam",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "mightn't've": "might not have",
    "must've": "must have",
    "mustn't": "must not",
    "mustn't've": "must not have",
    "needn't": "need not",
    "needn't've": "need not have",
    "o'clock": "of the clock",
    "oughtn't": "ought not",
    "oughtn't've": "ought not have",
    "shan't": "shall not",
    "sha'n't": "shall not",
    "shan't've": "shall not have",
    "she'd": "she would",
    "she'd've": "she would have",
    "she'll": "she will",
    "she'll've": "she will have",
    "she's": "she is",
    "should've": "should have",
    "shouldn't": "should not",
    "shouldn't've": "should not have",
    "so've": "so have",
    "so's": "so is",
    "that'd": "that would",
    "that'd've": "that would have",
    "that's": "that is",
    "there'd": "there would",
    "there'd've": "there would have",
    "there's": "there is",
    "they'd": "they would",
    "they'd've": "they would have",
    "they'll": "they will",
    "they'll've": "they will have",
    "they're": "they are",
    "they've": "they have",
    "to've": "to have",
    "wasn't": "was not",
    "we'd": "we would",
    "we'd've": "we would have",
    "we'll": "we will",
    "we'll've": "we will have",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what will",
    "what'll've": "what will have",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",
    "when's": "when is",
    "when've": "when have",
    "where'd": "where did",
    "where's": "where is",
    "where've": "where have",
    "who'll": "who will",
    "who'll've": "who will have",
    "who's": "who is",
    "who've": "who have",
    "why's": "why is",
    "why've": "why have",
    "will've": "will have",
    "won't": "will not",
    "won't've": "will not have",
    "would've": "would have",
    "wouldn't": "would not",
    "wouldn't've": "would not have",
    "y'all": "you all",
    "y'all'd": "you all would",
    "y'all'd've": "you all would have",
    "y'all're": "you all are",
    "y'all've": "you all have",
    "you'd": "you would",
    "you'd've": "you would have",
    "you'll": "you will",
    "you'll've": "you shall have",
    "you're": "you are",
    "you've": "you have",
    "doin'": "doing",
    "goin'": "going",
    "nothin'": "nothing",
    "somethin'": "something",
}
sampled_df.question_text = sampled_df.question_text.apply(lambda question: question.strip().lower())
test.question_text = test.question_text.apply(lambda question: question.strip().lower())

sampled_df.question_text = sampled_df.question_text.apply(lambda question: re.sub("\s{2,}", " ",question))
test.question_text = test.question_text.apply(lambda question: re.sub("\s{2,}", " ",question))

sampled_df.question_text = sampled_df.question_text.apply(lambda question: question.replace("’", "'"))
test.question_text = test.question_text.apply(lambda question: question.replace("’", "'"))

sampled_df.question_text = sampled_df.question_text.apply(lambda question: re.sub("'s", "",question))
test.question_text = test.question_text.apply(lambda question: re.sub("'s", "",question))

sampled_df.question_text = sampled_df.question_text.apply(lambda question: re.sub('"', "",question))
test.question_text = test.question_text.apply(lambda question: re.sub('"', "",question))
def fix(word):
    val = ''
    if word != '':
        val = contractions_dict.get(word,word)
    return val

def remove_contractions(question):
    words = question.split(" ")
    new_words = [fix(word) for word in words]
    return " ".join(new_words)
sampled_df.question_text = sampled_df.question_text.apply(lambda question: remove_contractions(question))
sampled_df.question_text = sampled_df.question_text.apply(lambda question: re.sub("'", "",question))
test.question_text = test.question_text.apply(lambda question: re.sub("'", "",question))
sampled_df.question_text = sampled_df.question_text.apply(lambda question: re.sub("/", " ",question))
test.question_text = test.question_text.apply(lambda question: re.sub("/", " ",question))
sampled_df.question_text = sampled_df.question_text.apply(lambda question: re.sub("-", " ",question))
test.question_text = test.question_text.apply(lambda question: re.sub("-", " ",question))
string.punctuation+= '“”’-‘'+ "``" + "''"
string.punctuation
stop_words = set(stopwords.words('english'))
lem = WordNetLemmatizer()
def preprocess_question(question):
    words = word_tokenize(question)
    words = [lem.lemmatize(word) for word in words if (not word in stop_words) and (not word in string.punctuation)]
    return re.sub('\d+',''," ".join(words))
sampled_df['question_text'] = sampled_df.question_text.apply(lambda question: preprocess_question(question))
test['question_text'] = test.question_text.apply(lambda question: preprocess_question(question))
def plot_wordcloud(words, ttle):
    wordcloud = WordCloud(width = 800, height = 400, background_color ='black', 
                min_font_size = 6, max_font_size = 150).generate(str(words))  
    plt.figure(figsize = (18, 12), facecolor = None) 
    plt.imshow(wordcloud)
    plt.title(ttle, fontsize = 40)
    plt.axis("off") 
    plt.tight_layout(pad = 0) 
    plt.show() 
plot_wordcloud(sampled_df['question_text'],'Word Cloud of Questions')
plot_wordcloud(sampled_df.question_text[sampled_df.target == 1],'Word Cloud of Insincere Questions')
plot_wordcloud(sampled_df.question_text[sampled_df.target == 0],'Word Cloud of Sincere Questions')
def bar_plot(data, ttle, ntw = 15, fig_size = (12,8), ttle_size = 30):
    plt.figure(figsize = fig_size)
    sns.barplot(y = np.array(list(dict(data[:ntw]).keys()),dtype = object), x = np.array(list(dict(data[:ntw]).values())).astype(float))
    plt.title(ttle, fontsize = ttle_size)
def vocabulary(questions, Verbose = True):
    vocab = dict()
    for question in tqdm(questions, disable = (not Verbose)):
        for word in word_tokenize(question):
            if (word != " ") and (word not in stop_words) and (word not in '?!.,"#$%\'()*+-/:;<=>@[\\]^_`{|}~' + '“”’`"' + "'``"):
                vocab[word] = vocab.get(word,0) + 1
    return vocab

vocab = vocabulary(sampled_df.question_text)

sorted_vocab = sorted(vocab.items(), key=lambda kv: kv[1],reverse = True)
bar_plot(sorted_vocab, 'Most frequent words in all the Questions', 50,(15,12))
vocab = vocabulary(sampled_df.question_text[sampled_df.target == 1])
sorted_vocab = sorted(vocab.items(), key=lambda kv: kv[1],reverse = True)
bar_plot(sorted_vocab, 'Most frequent words in Insincere Questions', 50,(8,12))
vocab = vocabulary(sampled_df.question_text[sampled_df.target == 0])
sorted_vocab = sorted(vocab.items(), key=lambda kv: kv[1],reverse = True)
bar_plot(sorted_vocab, 'Most frequent words in Sincere Questions', 50,(8,12))
def ngram(questions, n):
    counts_ng = dict()
    for questions in tqdm(questions, disable = True):
        words = word_tokenize(questions)
        ngram_tuples = list(nltk.ngrams(words,n))
        for ngram_tuple in ngram_tuples:
            counts_ng[ngram_tuple] = counts_ng.get(ngram_tuple, 0) + 1
    return counts_ng
def plot_ngrams(questions, n,ttle, ntw = 25, ttle_size = 30, fig_size = (8,14)):
    counts_ng = ngram(questions, n)
    sorted_by_value_ng = sorted(np.array(list(counts_ng.items())), key=lambda kv: kv[-1],reverse = True)
    ng = sorted_by_value_ng[:ntw]
    ng_indx = [str(ng[i][0]) for i in range(ntw)] 
    ng_val = [ng[i][1] for i in range(ntw)] 
    plt.figure(figsize = fig_size)
    ax = sns.barplot(y = ng_indx, x = ng_val)
    plt.title(ttle, fontsize = ttle_size )
plot_ngrams(sampled_df.question_text[sampled_df.target == 1], 3, 'Frequent trigrams in Insincere questions', 50,30,(8,14))
plot_ngrams(sampled_df.question_text[sampled_df.target == 0], 3, 'Frequent trigrams in Sincere questions', 50,30,(8,14))
cv = TfidfVectorizer(sublinear_tf = True,stop_words = 'english', ngram_range = (1,2), max_features = 5000, token_pattern = '(\S+)') 
X_train, X_val, y_train, y_val = train_test_split(sampled_df['question_text'], sampled_df['target'], test_size = 0.25)
cvec = cv.fit(X_train)
df_train = pd.DataFrame(cvec.transform(X_train).todense(),columns = cvec.get_feature_names())
df_val = pd.DataFrame(cvec.transform(X_val).todense(), columns = cvec.get_feature_names())
loreg = LinearSVC()
loreg.fit(df_train, y_train)
y_pred = loreg.predict(df_val)
cm = confusion_matrix(y_val, y_pred)
labels = ['sincere', 'unsincere']
print(pd.DataFrame(cm, columns=labels, index=labels))
print(classification_report(y_val, y_pred))
print("Accuracy : {:.2f}".format(metrics.accuracy_score(y_val, y_pred)))
df_test = pd.DataFrame(cvec.transform(test.question_text).todense(),columns = cvec.get_feature_names())
test_pred = loreg.predict(df_test)
submission = pd.DataFrame()
submission['qid'] = test.qid
submission['prediction'] = test_pred
submission.head()
submission.to_csv('submission.csv', index=False)