!pip install gensim

!pip install keras==2.24

!pip install pandas==0.23.4
import os

import numpy as np

import pandas as pd

import gensim



from nltk.stem import WordNetLemmatizer

from nltk.tokenize import RegexpTokenizer

from nltk.corpus import stopwords

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

from wordcloud import WordCloud

from collections import Counter



from gensim.models import Word2Vec



import keras

from keras.utils import to_categorical

from keras.models import Sequential

from keras.layers import Dense

from keras.layers import Flatten

from keras.layers.embeddings import Embedding

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences



import math
pd.set_option('display.max_rows', 10)

pd.set_option('display.max_columns', 500)

pd.set_option('display.width', 1000)
EMBED_DIM=10

MAX_SEQ_LENGTH=100
# filename = os.listdir("../input")[1]

# path = os.path.join("..","input",filename)

# path
path = "../input/sentiment140/training.1600000.processed.noemoticon.csv"

df_input = pd.read_csv(path, encoding="ISO-8859-1", names=["target", "ids", "date", "flag", "user", "text"])
df_raw = df_input[['text','target']]

df_raw=df_raw.rename(columns={"target": "prediction"})



print(df_raw.head())
# dataset_filename = os.listdir("../input")[1]

# dataset_path = os.path.join("..","input",dataset_filename)

tweet_stock_l=[]

for i in range(8):

    tweet_stock_l.append(pd.DataFrame(columns=['Date','created_at', 'text', 'user_id_str']))
for dirname, _, filenames in os.walk('/kaggle/input/tweetpredictstock/tweet_train/'):

    for filename in filenames:        

        path = os.path.join(dirname,filename)

        sts = dirname.split('/')

        idx = int(sts[len(sts)-1][0])-1

        tweet_raw_df = pd.read_json(path, lines=True,orient='columns')

        tweet_raw_df["Date"] = filename

        #tweet_raw_df = tweet_raw_df[['Date','created_at', 'text', 'user_id_str']]

        tweet_stock_l[idx] = pd.concat([tweet_stock_l[idx], tweet_raw_df], ignore_index=True, sort=False)
test_l=[]

for i in range(8):

    test_l.append(pd.DataFrame(columns=['Date','created_at', 'text', 'user_id_str']))
for dirname, _, filenames in os.walk('/kaggle/input/tweet-testing/'):

    for filename in filenames:        

        path = os.path.join(dirname,filename)

        sts = path.split('/')

        idx = int(sts[len(sts)-2][0])-1

        test_df = pd.read_json(path, lines=True,orient='columns')

        test_df["Date"] = filename

        test_l[idx] = pd.concat([test_l[idx], test_df], ignore_index=True, sort=True).sort_values(by='Date')
print("Training Data:")

print(tweet_stock_l[1].iloc[0:2,0:3])



print("\nTesting Data:")

print(test_l[1].iloc[0:2,0:3])
for i in range(8):

    if(i<=2):

        print("(Train data) Stock="+str(i+1))

        print(tweet_stock_l[i].groupby(['Date'])['Date'].count().reset_index(name='count').sort_values(['count'], ascending=False).head(5))
for i in range(8):

    if(i<=2):

        print("(Test data) Stock="+str(i+1))

        print(test_l[i].groupby(['Date'])['Date'].count().reset_index(name='count').sort_values(['count'], ascending=False).head(5))
# dataset_filename = os.listdir("../input")[1]

# dataset_path = os.path.join("..","input",dataset_filename)



raw_price_train_l = []

for i in range(1,9,1):

    filename = "/kaggle/input/tweetpredictstock/raw_price_train/"+str(i)+"_r_price_train.csv"

    print(filename)

    raw_price_file = pd.read_csv(filename)

    price_train = pd.DataFrame(raw_price_file)

    raw_price_train_l.append(price_train)
def mapLabel(input_val):

    raw_val = int(input_val)

    if(raw_val==0):

        return 0

    elif(raw_val==2):

        return 99

    elif(raw_val==4):

        return 1



df_raw.prediction = df_raw.prediction.apply(lambda x: mapLabel(x))
df_raw = df_raw[df_raw.prediction != 99]

df_pos = df_raw[df_raw.prediction == 1]

df_pos=df_pos.sample(frac=0.1, replace=True, random_state=1)

df_neg = df_raw[df_raw.prediction == 0]

df_neg=df_neg.sample(frac=0.1, replace=True, random_state=1)

frame = [df_pos, df_neg]

df_negative_positive = pd.concat(frame)



print(df_negative_positive.shape[0])
#tokenizer that can filter out most punctuation, filtered out words e.g. "book", "didn't

reg_filter = "[\w']+"

toker = RegexpTokenizer(reg_filter) 



#lemmatizer return context of word, e.g."better->good", "rocks->rock"

wordnet_lemmatizer=WordNetLemmatizer()

stopWords = set(stopwords.words('english'))

stopWords.remove("hadn't")

stopWords.remove("didn't")
df_tokenized = df_negative_positive





def dataPreProcessing(row):

    row_lower=row.lower()

    

    #tokenization

    token_l=toker.tokenize(row_lower)

    

    #remove stopwrods

    token_l_no_stop_word=[x for x in token_l if x not in stopWords ]



    #lematization

    token_l_lemmatized=[wordnet_lemmatizer.lemmatize(x) for x in token_l_no_stop_word]

    

    return token_l_lemmatized

    

def joinToken(row):

    clean_text=" ".join(row)

    return clean_text



df_tokenized['token_list'] = df_tokenized.text.apply(lambda row: dataPreProcessing(row))

df_tokenized['clean_text'] = df_tokenized.token_list.apply(lambda row: joinToken(row))

df_tokenized=df_tokenized[['text','token_list','clean_text','prediction']]



stoke_tokenized_l=[]

for df in tweet_stock_l:

    df=df.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)

    df.text = df.text.apply(lambda row: joinToken(row))

    df['token_list'] = df.text.apply(lambda row: dataPreProcessing(row))

    df['clean_text'] = df.token_list.apply(lambda row: joinToken(row))

    stoke_tokenized_l.append(df)

    

test_tokenized_l=[]

for df in test_l:

    df=df.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)

    df.text = df.text.apply(lambda row: joinToken(row))

    df['token_list'] = df.text.apply(lambda row: dataPreProcessing(row))

    df['clean_text'] = df.token_list.apply(lambda row: joinToken(row))

    test_tokenized_l.append(df)
print(df_tokenized[["clean_text","token_list"]].head())
stoke_tokenized_l[0]
print(stoke_tokenized_l[0][["Date","clean_text"]].head())
test_tokenized_l[0]
print(test_tokenized_l[0][["Date","clean_text"]].head())
df_positive = df_tokenized[ df_tokenized['prediction'] == 1]

df_negative = df_tokenized[ df_tokenized['prediction'] == 0]





fig = plt.figure(1,figsize=(30, 20))

def makeWordCloud(df, title, a):

    c = Counter()

    for i, row in df.iterrows():

        c.update(row['token_list']) 



    cleaned_word = " ".join(list(df['clean_text']))

    wc = WordCloud(width=500, height=500, background_color='white',random_state=0)

    wc.generate_from_frequencies(c)

    axis_1 = fig.add_subplot(4,5,(a))

    axis_1.imshow(wc)

    axis_1.axis('off')

    plt.title(title)

    

makeWordCloud(df_positive, "Positive", 1)

makeWordCloud(df_negative, "Negative", 2)



for i in range(8):

    makeWordCloud(stoke_tokenized_l[i], "Training Data: Stock="+str(i+1), (4+i))

    

for i in range(8):

    makeWordCloud(test_tokenized_l[i], "Testing Data: Stock="+str(i+1), (13+i))
tokens_all = [ x for x in df_tokenized.token_list]



word_to_vec_model = gensim.models.word2vec.Word2Vec(size=EMBED_DIM, window=5, min_count=1, workers=4, sg=0)

word_to_vec_model.build_vocab(tokens_all)
word_to_vec_model.train(tokens_all, total_examples=len(tokens_all), epochs = 4)
word_to_vec_model.most_similar("great")
toker = Tokenizer()

toker.fit_on_texts(df_tokenized.clean_text)

wd_idx = toker.word_index
seq = toker.texts_to_sequences(df_tokenized.clean_text)
print(df_tokenized["token_list"].head(1))

print(seq[:1])
stock_seq_l = []

for i in range(8):

    stock_seq = toker.texts_to_sequences(stoke_tokenized_l[i].clean_text)

    stock_seq_l.append(stock_seq)
test_seq_l = []

for i in range(8):

    stock_seq = toker.texts_to_sequences(test_tokenized_l[i].clean_text)

    test_seq_l.append(stock_seq)
print(stoke_tokenized_l[0]["clean_text"].head(1))

print(stock_seq_l[0][:1])

print("index of 'top'="+str(wd_idx['top']))

print("index of 'story'="+str(wd_idx['story']))
print(test_tokenized_l[0]["clean_text"].head(1))

print(test_seq_l[0][:1])

print("index of 'top'="+str(wd_idx['word']))

print("index of 'story'="+str(wd_idx['tell']))
x_matrix = pad_sequences(seq, maxlen=MAX_SEQ_LENGTH)

print(x_matrix[0])
stock_matrix_l=[]

for i in range(8):

    matrix = pad_sequences(stock_seq_l[i], maxlen=MAX_SEQ_LENGTH)

    stock_matrix_l.append(matrix)

print(stock_matrix_l[0][0])
test_matrix_l=[]

for i in range(8):

    matrix = pad_sequences(test_seq_l[i], maxlen=MAX_SEQ_LENGTH)

    test_matrix_l.append(matrix)

print(test_matrix_l[0][0])
y_array = np.asarray(df_tokenized["prediction"])

print(y_array[:1])

y_matrix = to_categorical(y_array)

print(y_matrix[:1])
idxs = np.arange(x_matrix.shape[0])

print(idxs)

np.random.shuffle(idxs)

print(idxs)
x_rand = x_matrix[idxs]

y_rand = y_matrix[idxs]

split_idx = int(x_rand.shape[0] * 0.8)

print(split_idx)



x_train= x_rand[:split_idx]

y_train= y_rand[:split_idx]

x_valid= x_rand[split_idx:]

y_valid= y_rand[split_idx:]



print("Size of train data="+str(x_train.shape[0]))

print("Size of valid data="+str(x_valid.shape[0]))

print("Shape of x train feature matrix="+str(x_train.shape))

print("Shape of x valid feature matrix="+str(x_valid.shape))
embed_grid = np.zeros((len(wd_idx)+1, EMBED_DIM))

for ele, i in wd_idx.items():

    if ele in word_to_vec_model.wv.vocab:

        embed_grid[i] = word_to_vec_model.wv[ele]

        if(i<4):

            print(ele)

            print(embed_grid[i])
model = Sequential()

model.add(Embedding(len(embed_grid),EMBED_DIM,weights=[embed_grid], input_length=MAX_SEQ_LENGTH))                        

model.add(Flatten())

model.add(Dense(units=2, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.summary()
model.fit(x_train, y_train, epochs=4)



train_loss, train_accuracy = model.evaluate(x_train, y_train)

print("* Accurancy of training data ="+str(train_accuracy)+", it is reasonable")
valid_loss, valid_accuracy = model.evaluate(x_valid, y_valid)

print("* Accurancy of validate data ="+str(valid_accuracy)+", it is reasonable")
y_validate_l=[]

for i in range(8):

    y_validate = model.predict(stock_matrix_l[i])

    y_validate_l.append(y_validate)
y_test_l=[]

for i in range(8):

    y_test = model.predict(test_matrix_l[i])

    y_test_l.append(y_test)
x_look = (stoke_tokenized_l[0])

y_look = y_validate_l[0]



for i in range(100,105,1):

    clean_txt = x_look.loc[i, "clean_text"]

    score = y_look[i]

    print(clean_txt)

    print(score)
x_look = (test_tokenized_l[0])

y_look = y_test_l[0]



for i in range(0,5,1):

    clean_txt = x_look.loc[i, "clean_text"]

    score = y_look[i]

    print(clean_txt)

    print(score)
original_size_l = []

for i in range(8):

    original_size_l.append(stoke_tokenized_l[i].shape[0])

print("(Training stock tweet data) no of tweets of each stock="+str(original_size_l))
original_test_size_l = []

for i in range(8):

    original_test_size_l.append(test_tokenized_l[i].shape[0])

print("(Testing stock tweet data) no of tweets of each stock="+str(original_test_size_l))
y_score_l = []

for i in range(8):

    y_validate = y_validate_l[i]

    score = []

    for j in range(len(y_validate)):

        if(y_validate[j][1]>0.5):

            score.append(1)

        else:

            score.append(0)

    y_score_l.append(score)

print("(Training stock tweet data) score result of some example:"+ str(y_score_l[0][:5]))
y_test_score_l = []

for i in range(8):

    y_test = y_test_l[i]

    score = []

    for j in range(len(y_test)):

        if(y_test[j][1]>0.5):

            score.append(1)

        else:

            score.append(0)

    y_test_score_l.append(score)

print("(Testing stock tweet data) score result of some example:"+ str(y_test_score_l[0][:5]))
def groupByDate(x_df, y_l):

    x_df["score"]=y_l

    date_dict ={}

    for idx,row in x_df.iterrows():

        date = row["Date"]

        score = row["score"]

        sentiment_l = date_dict.get(date, [0,0,0,0])

        if score==1:

            sentiment_l[0] += 1

            date_dict[date] = sentiment_l

        else:

            sentiment_l[1] += 1

            date_dict[date] = sentiment_l

    for key in date_dict:

        senti_l = date_dict[key]

        senti_l[2]=senti_l[0]+senti_l[1]

        senti_l[3]=senti_l[0]/senti_l[2]

        date_dict[key]= senti_l

    df_out = pd.DataFrame.from_dict(date_dict, orient='index', columns=['no_of_positive_tweet', 'no_of_negative_tweet', 'no_of_tweet', 'score'])

    df_out = df_out.sort_index()

    return df_out



def output_csv(df_out, name, i):

    name = "sentiment_output_"+name +"_"+ str(i+1) + ".csv"

    df_out.to_csv(name)
metrics_size_l=[]

for i in range(8):

    df_out = groupByDate(stoke_tokenized_l[i], y_score_l[i])

    output_csv(df_out,"train", i)

    metrics_size_l.append(df_out.shape[0])

    if(i<=2):

        print("(Training stock tweet data) Result of stock="+str(i+1))

        print(df_out.head())

print("(Training stock tweet data) size of metrics of each stock="+str(metrics_size_l))
test_metrics_size_l=[]

for i in range(8):

    df_out = groupByDate(test_tokenized_l[i], y_test_score_l[i])

    output_csv(df_out, "test", i)

    test_metrics_size_l.append(df_out.shape[0])

    if(i<=2):

        print("(Testing stock tweet data) Result of stock="+str(i+1))

        print(df_out.head())

print("(Testing stock tweet data) size of metrics of each stock="+str(test_metrics_size_l))