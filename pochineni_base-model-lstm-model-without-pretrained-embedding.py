# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from wordcloud import WordCloud ,STOPWORDS

from keras.preprocessing import text, sequence

from keras import backend as K

from keras.models import load_model

import keras

import pickle

print(K.tensorflow_backend._get_available_gpus())

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

sub = pd.read_csv('../input/sample_submission.csv')
train_data = train["comment_text"]

label_data = train["target"]

test_data = test["comment_text"]

train_data.shape, label_data.shape, test_data.shape
test.head()
tokenizer = text.Tokenizer()

tokenizer.fit_on_texts(list(train_data) + list(test_data))
train_data = tokenizer.texts_to_sequences(train_data)

test_data = tokenizer.texts_to_sequences(test_data)
MAX_LEN = 200

train_data = sequence.pad_sequences(train_data, maxlen=MAX_LEN)

test_data = sequence.pad_sequences(test_data, maxlen=MAX_LEN)
max_features = None
max_features = max_features or len(tokenizer.word_index) + 1

max_features
type(train_data), type(label_data.values), type(test_data)

label_data = label_data.values
#Word cloud of train file



from wordcloud import WordCloud, STOPWORDS



# Thanks : https://www.kaggle.com/aashita/word-clouds-of-various-shapes ##

def plot_wordcloud(text, mask=None, max_words=200, max_font_size=100, figure_size=(24.0,16.0), 

                   title = None, title_size=40, image_color=False):

    stopwords = set(STOPWORDS)

    more_stopwords = {'one', 'br', 'Po', 'th', 'sayi', 'fo', 'Unknown'}

    stopwords = stopwords.union(more_stopwords)



    wordcloud = WordCloud(background_color='black',

                    stopwords = stopwords,

                    max_words = max_words,

                    max_font_size = max_font_size, 

                    random_state = 42,

                    width=800, 

                    height=400,

                    mask = mask)

    wordcloud.generate(str(text))

    

    plt.figure(figsize=figure_size)

    if image_color:

        image_colors = ImageColorGenerator(mask);

        plt.imshow(wordcloud.recolor(color_func=image_colors), interpolation="bilinear");

        plt.title(title, fontdict={'size': title_size,  

                                  'verticalalignment': 'bottom'})

    else:

        plt.imshow(wordcloud);

        plt.title(title, fontdict={'size': title_size, 'color': 'black', 

                                  'verticalalignment': 'bottom'})

    plt.axis('off');

    plt.tight_layout()  

    

plot_wordcloud(train["comment_text"], title="Word Cloud of Train Comments")
#Word cloud of train file of toxic comments

train['target'] = np.where(train['target'] >= 0.5, 1, 0)

train_toxic = train[train.target == 1]





from wordcloud import WordCloud, STOPWORDS



# Thanks : https://www.kaggle.com/aashita/word-clouds-of-various-shapes ##

def plot_wordcloud(text, mask=None, max_words=200, max_font_size=100, figure_size=(24.0,16.0), 

                   title = None, title_size=40, image_color=False):

    stopwords = set(STOPWORDS)

    more_stopwords = {'one', 'br', 'Po', 'th', 'sayi', 'fo', 'Unknown'}

    stopwords = stopwords.union(more_stopwords)



    wordcloud = WordCloud(background_color='black',

                    stopwords = stopwords,

                    max_words = max_words,

                    max_font_size = max_font_size, 

                    random_state = 42,

                    width=800, 

                    height=400,

                    mask = mask)

    wordcloud.generate(str(text))

    

    plt.figure(figsize=figure_size)

    if image_color:

        image_colors = ImageColorGenerator(mask);

        plt.imshow(wordcloud.recolor(color_func=image_colors), interpolation="bilinear");

        plt.title(title, fontdict={'size': title_size,  

                                  'verticalalignment': 'bottom'})

    else:

        plt.imshow(wordcloud);

        plt.title(title, fontdict={'size': title_size, 'color': 'black', 

                                  'verticalalignment': 'bottom'})

    plt.axis('off');

    plt.tight_layout()  

    

plot_wordcloud(train_toxic["comment_text"], title="Word Cloud of Toxic Comments")
#Word cloud of train file of non-toxic comments

train['target'] = np.where(train['target'] >= 0.5, 1, 0)

train_nontoxic = train[train.target == 0]

plot_wordcloud(train_nontoxic["comment_text"], title="Word Cloud of Non-Toxic Comments")
# Keras Model

# Model Parameters

NUM_HIDDEN = 256

EMB_SIZE = 256

LABEL_SIZE = 1

MAX_FEATURES = max_features

DROP_OUT_RATE = 0.2

DENSE_ACTIVATION = "sigmoid"

NUM_EPOCH = 1



# Optimization Parameters

BATCH_SIZE = 1000

LOSS_FUNC = "binary_crossentropy"

OPTIMIZER_FUNC = "adam"

METRICS = ["accuracy"]



class LSTMModel:

    

    def __init__(self):

        self.model = self.build_graph()

        self.compile_model()

    

    def build_graph(self):

        model = keras.models.Sequential([

            keras.layers.Embedding(MAX_FEATURES, EMB_SIZE),

            keras.layers.CuDNNLSTM(NUM_HIDDEN),

            keras.layers.Dropout(rate=DROP_OUT_RATE),

            keras.layers.Dense(LABEL_SIZE, activation=DENSE_ACTIVATION)])

        return model

    

    def compile_model(self):

        self.model.compile(

            loss=LOSS_FUNC,

            optimizer=OPTIMIZER_FUNC,

            metrics=METRICS)
model = LSTMModel().model

model.fit(

    train_data, 

    label_data, 

    batch_size = BATCH_SIZE, 

    epochs = NUM_EPOCH)
from sklearn.metrics import auc

auc_keras = auc(fpr_keras, tpr_keras)
plt.xlim(0, 0.2)

plt.ylim(0.8, 1)

plt.plot([0, 1], [0, 1], 'k--')

plt.plot(fpr_keras, tpr_keras, label='Keras (area = {:.3f})'.format(auc_keras))

plt.xlabel('False positive rate')

plt.ylabel('True positive rate')

plt.title('ROC curve (zoomed in at top left)')

plt.legend(loc='best')

plt.show()
## Loss and accuracy graphs



import matplotlib.pyplot as plt



history = model.fit(

    train_data, 

    label_data, 

    validation_split=0.25, epochs=5, batch_size=1000, verbose=1)


# Plot training & validation accuracy values

plt.xlim(0, 4)

plt.ylim(0.6, 0.8)

plt.plot(history.history['acc'])

plt.plot(history.history['val_acc'])

plt.title('Model accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()



# Plot training & validation loss values

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('Model loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()
#Plot ROC curve

import sklearn.metrics as metrics

# calculate the fpr and tpr for all thresholds of the classification

y_pred = model.predict(train_data)
# Tag label_data as binary

# y_pred=np.where(y_pred>=0.5,1,0)

label_data=np.where(label_data>=0.5,1,0)

fpr, tpr, threshold = metrics.roc_curve(label_data, y_pred)

roc_auc = metrics.auc(fpr, tpr)



# method I: plt

import matplotlib.pyplot as plt

plt.title('Receiver Operating Characteristic')

plt.plot(fpr, tpr, 'b')

# label = 'AUC = %0.2f' % roc_auc)

plt.legend(loc = 'lower right')

plt.plot([0, 1], [0, 1],'r--')

plt.xlim([-0.02, 1])

plt.ylim([0, 1.05])

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')

plt.show()
#Tag both label and predicted as binary (with margin as 0.5)



y_pred=np.where(y_pred>=0.5,1,0)

label_data=np.where(label_data>=0.5,1,0)

fpr, tpr, threshold = metrics.roc_curve(label_data, y_pred)

roc_auc = metrics.auc(fpr, tpr)



# method I: plt

import matplotlib.pyplot as plt

plt.title('Receiver Operating Characteristic')

plt.plot(fpr, tpr, 'b',label = 'AUC = %0.2f' % roc_auc)

plt.legend(loc = 'lower right')

plt.plot([0, 1], [0, 1],'r--')

plt.xlim([-0.02, 1])

plt.ylim([0, 1.05])

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')

plt.show()
#Extract frequency of transactions along with toxicity for different probability buckets
submission_in = '../input/sample_submission.csv'

submission_out = 'submission.csv'
result = model.predict(test_data)
submission = pd.read_csv(submission_in, index_col='id')

submission['prediction'] = result

submission.reset_index(drop=False, inplace=True)
submission.to_csv(submission_out, index=False)
#Get data for train file so that we can evaluate false positives

submission_in = '../input/train.csv'

#Filter ID and pre

submission_out = 'submission_train.csv'

#Make test data as train so that we can check false positives of classification

test1_data = train["comment_text"]
tokenizer = text.Tokenizer()

tokenizer.fit_on_texts(list(test1_data))
test1_data = tokenizer.texts_to_sequences(test1_data)
MAX_LEN = 200

test1_data = sequence.pad_sequences(test1_data, maxlen=MAX_LEN)
result1 = model.predict(test1_data)
submission_train = pd.read_csv(submission_in, index_col='id')

submission_train['prediction'] = result1

submission_train.reset_index(drop=False, inplace=True)
#submisson file has a prediction column as well



submission_train.head()
#filter only the ID, target value and prediction value

filtered_submission=submission_train.filter(items=['id', 'target','prediction'])
filtered_submission.head()
filtered_submission['prediction'] = [ 1 if prediction>=0.5 else 0 for prediction in filtered_submission['prediction'] ]

filtered_submission['target'] = [ 1 if target>=0.5 else 0 for target in filtered_submission['target'] ]

filtered_submission.head(10)
filtered_submission.groupby(["target","prediction"]).count()[['id']]
# FN are more than FPs. Precision 13%, Recall 3%



#Word cloud of train file of false positives and f

filtered_submission_toxic = filtered_submission[filtered_submission.target == 1]



plot_wordcloud(train_toxic["comment_text"], title="Word Cloud of Toxic Comments")
def toxicwordcloud(subset=train[train.target>0.7], title = "Words Frequented"):

    stopword=set(STOPWORDS)

#     toxic_mask=np.array(Image.open(picture))

#     toxic_mask=toxic_mask[:,:,1]

    text=subset.comment_text.values

    wc= WordCloud(background_color="black",max_words=4000,stopwords=stopword)

    wc.generate(" ".join(text))

    plt.figure(figsize=(8,8))

    plt.xticks([])

    plt.yticks([])

    plt.axis('off')

    plt.title(title, fontsize=20)

    plt.imshow(wc.recolor(colormap= 'gist_earth' , random_state=244), alpha=0.98)

    

toxicwordcloud(subset=train[train.target>0.7], title = "Words Frequented")