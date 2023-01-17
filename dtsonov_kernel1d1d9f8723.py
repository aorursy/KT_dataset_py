import numpy as np

import pandas as pd

import sqlite3

import re

import nltk

from nltk.corpus import stopwords

from nltk.tokenize import word_tokenize

from nltk.corpus import stopwords

import matplotlib.pyplot as plt

import seaborn as sns

from wordcloud import WordCloud

import squarify

import random



random.seed(100)
df = pd.read_csv("/kaggle/input/twitter-airline-sentiment/Tweets.csv")



stop_words = set(stopwords.words('english'))

def clean_text(text):

    # Remove http links

    res = re.sub("(?<!\S)https?://\S+", "", text)

    

    # Remove numbers

    res = re.sub("(?<!\S)[-+]?(?=.*?\d)\d*[.,]?\d*(?!\S)", "", res)

    

    # Remove special characters and make everything lowercase. This can be improved because this way we lose emojis like :(, :) etc

    res = re.sub('[^A-Za-z0-9]+', ' ', res).lower().strip()

    

    # We do not expand contracted spelling forms i.e. forms like "he's" = "he is" because we accept the user spelling style as is and keep them intact.

    # Otherwise the trained model should handle this and choose between expanded/contrаcted forms which is an additional issue.

    

    # Remove stopwords

    words = [word for word in nltk.word_tokenize(res) if not word in stop_words]

    res = ' '.join(words)

    

    return res



df['text_cleaned'] = df['text'].apply(clean_text)
# The following code prepares data for the word cloud (that shows the most frequent words by sentiment).



from sklearn.feature_extraction.text import CountVectorizer



# Labels of each column

print("Columns in the data base:\n")

print(df.keys())

print("\n")

print("We have " + str(df.shape[0]) + " data points spread between " + str(df.shape[1]) + " features.")



# Popular words

pos = df[df['airline_sentiment'] == 'positive']['text_cleaned'].head(25000).str.cat(sep=' ')

neg = df[df['airline_sentiment'] == 'negative']['text_cleaned'].head(25000).str.cat(sep=' ')

neut = df[df['airline_sentiment'] == 'neutral']['text_cleaned'].head(25000).str.cat(sep=' ')



# Generate a word cloud image

poswc = WordCloud(width=1280, height=800).generate(pos)

negwc = WordCloud(width=1280, height=800).generate(neg)

neutwc = WordCloud(width=1280, height=800).generate(neut)



def plwordcl(fig_num, word_cloud, title):

    plt.figure(fig_num, figsize=(20,10), facecolor='w')

    plt.imshow(word_cloud)

    plt.axis("off")

    plt.tight_layout(pad=0)

    plt.title(title)



plwordcl(1, poswc, "Word cloud for POSITIVE sentiment")

plwordcl(2, negwc, "Word cloud for NEGATIVE sentiment")

plwordcl(3, neutwc, "Word cloud for NEUTRAL sentiment")

plt.show()

  
#Airlines chart

df_ac = df.groupby('airline').size()

df_ac.plot(kind='pie', subplots=True, figsize=(8, 8))

plt.title("Airlines data chart")

plt.ylabel("")

plt.show()





#Sentiment treemap chart

df_sent = df.groupby('airline_sentiment').size().reset_index(name='counts')

labels = df_sent.apply(lambda x: str(x[0]) + "\n (" + str(x[1]) + ")", axis=1)

sizes = df_sent['counts'].values.tolist()

colors = [plt.cm.Spectral(i/float(len(labels))) for i in range(len(labels))]

plt.figure(figsize=(6,4), dpi= 80)

squarify.plot(sizes=sizes, label=labels, color=colors, alpha=.8)





#Negative reasons

df_negative_airlines = df.groupby('negativereason').size().reset_index(name='counts')

n = df_negative_airlines['negativereason'].unique().__len__()+1

all_colors = list(plt.cm.colors.cnames.keys())

random.seed(100)

c = random.choices(all_colors, k=n)



# Plot Bars

plt.figure(figsize=(12,8))

plt.bar(df_negative_airlines['negativereason'], df_negative_airlines['counts'], color=c, width=.5)

for i, val in enumerate(df_negative_airlines['counts'].values):

    plt.text(i, val, float(val), horizontalalignment='center', verticalalignment='bottom', fontdict={'fontweight':500, 'size':12})



# Decoration

plt.gca().set_xticklabels(df_negative_airlines['negativereason'], rotation=60, horizontalalignment= 'right')

plt.title("Negative reasons", fontsize=22)

plt.ylabel('# Tweets')

plt.ylim(0, 3500)

plt.show()

import tensorflow as tf



from scipy.spatial.distance import cdist

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, GRU, Embedding

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.preprocessing.text import Tokenizer

from tensorflow.keras.preprocessing.sequence import pad_sequences



from sklearn.preprocessing import LabelEncoder

from keras.utils import to_categorical

from sklearn.model_selection import train_test_split, cross_val_score



data_text = df['text_cleaned'].tolist()

y_orig = df['airline_sentiment'].tolist()



le = LabelEncoder()

neg_label_ind, pos_label_ind, neut_label_ind = le.fit_transform(['negative', 'positive', 'neutral'])

print(f'negative encoded as {neg_label_ind}, positive encoded as {pos_label_ind}, neutral encoded as {neut_label_ind}')



y_le = le.transform(y_orig)

y = to_categorical(y_le)



x_train_text, x_test_text, y_train, y_test = train_test_split(data_text, y, test_size=0.20, random_state=12)



num_words = 10000

tokenizer = Tokenizer(num_words=num_words)

tokenizer.fit_on_texts(x_train_text)



x_train_tokens = tokenizer.texts_to_sequences(x_train_text)

x_test_tokens = tokenizer.texts_to_sequences(x_test_text)



num_tokens = [len(tokens) for tokens in x_train_tokens + x_test_tokens]

num_tokens = np.array(num_tokens)

np.mean(num_tokens)



np.max(num_tokens)



max_tokens = np.mean(num_tokens) + 2 * np.std(num_tokens)

max_tokens = int(max_tokens)

max_tokens



np.sum(num_tokens < max_tokens) / len(num_tokens)



pad = 'pre'



x_train_pad = pad_sequences(x_train_tokens, maxlen=max_tokens,

                            padding=pad, truncating=pad)



x_test_pad = pad_sequences(x_test_tokens, maxlen=max_tokens,

                           padding=pad, truncating=pad)



print(x_train_pad.shape)

print(x_test_pad.shape)



model_rnn = Sequential()

embedding_size = 50



model_rnn.add(Embedding(input_dim=num_words,

                    output_dim=embedding_size,

                    input_length=max_tokens

                   ))



model_rnn.add(GRU(units=20, return_sequences=True))



model_rnn.add(GRU(units=10, return_sequences=True))



model_rnn.add(GRU(units=5))



model_rnn.add(Dense(3, activation='softmax'))



optimizer = Adam(0.01)



model_rnn.compile(loss='categorical_crossentropy',

              optimizer=optimizer,

              metrics=['accuracy'])



model_rnn.summary()

modelh = model_rnn.fit(x_train_pad, y_train,

          validation_split=0.07, epochs=5, batch_size=200)



result = model_rnn.evaluate(x_test_pad, y_test, verbose=2)

print("\n\n")

print("Accuracy on validation set: {0:.2}".format(result[1]))
y_pred = model_rnn.predict(x=x_test_pad)



y_pred_unenc = [np.argmax(p) for p in y_pred]

y_test_unenc = [np.argmax(p) for p in y_test]



from sklearn.metrics import confusion_matrix, recall_score, precision_score



conf_mat_dl = confusion_matrix(y_test_unenc, y_pred_unenc, labels=[0,1,2])

cm_normalized_dl = conf_mat_dl.astype('float') / conf_mat_dl.sum(axis=1)[:, np.newaxis]



print("Confusion matrix:")

print(conf_mat_dl)

print("Normalized confusion matrix:")

print(cm_normalized_dl)



def plot_confusion_matrix(array, columns, title):

    import seaborn as sn

    import pandas as pd

    import matplotlib.pyplot as plt



      

    df_cm = pd.DataFrame(array, index=columns, columns=columns)

    df_cm.round(2)

    #plt.figure(figsize = (10,7))

    sn.set(font_scale=1.4)#for label size

    sn.heatmap(df_cm, annot=True,annot_kws={"size": 16})# font size



    plt.title(title)

    

    plt.show()

    

plot_confusion_matrix(cm_normalized_dl, le.inverse_transform([0,1,2]), "Normalized Conf. Matrix, RNN")
def draw_loss(history, string):

    plt.plot(history.history[string])

    plt.plot(history.history['val_'+string], '')

    plt.xlabel("epochs")

    plt.ylabel(string)

    plt.legend([string, 'val_'+string])

    plt.show()

    

draw_loss(modelh, 'accuracy')

draw_loss(modelh, 'loss')
print("Out of " + str(np.sum(conf_mat_dl[neg_label_ind])) + "  negative the RNN predicts " + str(conf_mat_dl[neg_label_ind][neg_label_ind]) + " as negative.")



print("Out of " + str(np.sum(conf_mat_dl[neut_label_ind])) + " neutral the RNN predicts " + str(conf_mat_dl[neut_label_ind][neut_label_ind]) + " as neutral.")



print("Out of " + str(np.sum(conf_mat_dl[pos_label_ind])) + " positive the RNN predicts " + str(conf_mat_dl[pos_label_ind][pos_label_ind]) + " as positive.")
print("Negative sentiment precision: " + str(conf_mat_dl[neg_label_ind][neg_label_ind]/np.sum(conf_mat_dl.T[neg_label_ind])))



print("Neutral sentiment precision: " + str(conf_mat_dl[neut_label_ind][neut_label_ind]/np.sum(conf_mat_dl.T[neut_label_ind])))



print("Positive sentiment precision: " + str(conf_mat_dl[pos_label_ind][pos_label_ind]/np.sum(conf_mat_dl.T[pos_label_ind])))
print("Negative sentiment recall: " + str(conf_mat_dl[neg_label_ind][neg_label_ind]/np.sum(conf_mat_dl[neg_label_ind])))



print("Neutral sentiment recall: " + str(conf_mat_dl[neut_label_ind][neut_label_ind]/np.sum(conf_mat_dl[neut_label_ind])))



print("Positive sentiment recall: " + str(conf_mat_dl[pos_label_ind][pos_label_ind]/np.sum(conf_mat_dl[pos_label_ind])))
accuracy = (np.trace(conf_mat_dl) )/(sum(sum(conf_mat_dl)))

print(f"We conclude that {accuracy:.2%} of the predicted outputs should be correctly classified using RNN.")
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, BaggingClassifier, ExtraTreesClassifier, GradientBoostingClassifier

from sklearn.feature_extraction.text import CountVectorizer

from nltk.corpus import stopwords

from collections import Counter, OrderedDict

from sklearn.model_selection import cross_validate, cross_val_predict

from sklearn.metrics import f1_score, accuracy_score

from sklearn.metrics import confusion_matrix, recall_score, precision_score

from sklearn.svm import SVC

from sklearn.multiclass import OneVsRestClassifier

from sklearn.preprocessing import MultiLabelBinarizer

import numpy as np



count_vect = CountVectorizer( max_df=0.5,stop_words=stopwords.words('english'), max_features=10000)



X = count_vect.fit_transform(df['text_cleaned'])



model_rf = RandomForestClassifier(n_estimators=5, class_weight='balanced', random_state=0)

model_ada = AdaBoostClassifier(n_estimators=50, learning_rate=1.0, algorithm='SAMME.R', random_state=56)

model_xgb = GradientBoostingClassifier(loss='deviance', 

                                       learning_rate=0.1, n_estimators=50, 

                                       subsample=1.0, 

                                       criterion='friedman_mse', min_samples_split=5, 

                                       min_samples_leaf=1, 

                                       min_weight_fraction_leaf=0.0, 

                                       max_depth=3, 

                                       min_impurity_decrease=0.0, 

                                       min_impurity_split=None, 

                                       random_state=56, 

                                       max_features=None, 

                                       verbose=0, 

                                       max_leaf_nodes=None, 

                                       warm_start=False, 

                                       validation_fraction=0.1, tol=0.0001)



models = { 'random forest': model_rf,

          'ada boost classifier': model_ada,

          'gradient boosting classifier': model_xgb

         }



models = { k: OneVsRestClassifier(m) for k, m in models.items()}



for key, model in models.items():

    print('-----------')

    print(f"Results for {key}:")

    y_pred = cross_val_predict(model, X, y, cv=4)

    

    y_pred_unenc = [np.argmax(p) for p in y_pred]

    y_unenc = [np.argmax(p) for p in y]

    

    conf_mat = confusion_matrix(y_unenc, y_pred_unenc, labels=[0,1,2])

    cm_normalized = conf_mat.astype('float') / conf_mat.sum(axis=1)[:, np.newaxis]

    acc = accuracy_score(y_pred_unenc, y_unenc, normalize=True)

    print("Confusion matrix:")

    print(conf_mat)

    print("Normalized confusion matrix:")

    print(cm_normalized)

    print(f"Accuracy: {acc}")

    plot_confusion_matrix(cm_normalized, le.inverse_transform([0,1,2]), "Normalized Conf. Matrix, %s" % key)
