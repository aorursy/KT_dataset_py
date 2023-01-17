import pandas as pd
import numpy as np
import string
import re
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style='whitegrid', palette='muted',
        rc={'figure.figsize': (15,10)})
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
train.head()
test.head()
print(train.shape, test.shape)
pres = {'deKlerk': 0,
        'Mandela': 1,
        'Mbeki': 2,
        'Motlanthe': 3,
        'Zuma': 4,
        'Ramaphosa': 5}

train.replace({'president': pres}, inplace=True)
# speech number: intro lines
starts = {
    0: 1,
    1: 1,
    2: 1,
    3: 12,
    4: 12,
    5: 5,
    6: 1,
    7: 1,
    8: 8,
    9: 9,
    10: 12,
    11: 14,
    12: 14,
    13: 15,
    14: 15,
    15: 15,
    16: 15,
    17: 15,
    18: 15,
    19: 15,
    20: 20,
    21: 1,
    22: 15,
    23: 20,
    24: 20,
    25: 15,
    26: 15,
    27: 20,
    28: 20,
    29: 15,
    30: 18
}
def divide_on(df, char):
    
    # iterate over text column of DataFrame, splitting at each occurrence of char

    sentences = []
    # let's split the data into senteces
    for i, row in df.iterrows():
        
        # skip the intro lines of the speech
        for sentence in row['text'].split(char)[starts[i]:]:
            sentences.append([row['president'], sentence])

    df = pd.DataFrame(sentences, columns=['president', 'text'])
    
    return df[df['text'] != '']
train = divide_on(train, '.')
train.head(5)
train.shape
train['president'].value_counts()
# proportion of total
train['president'].value_counts()/train.shape[0]
train['sentence'] = None
test['president'] = None

df = pd.concat([train, test], axis=0, sort=False)
# reorder columns
df = df[['sentence', 'text', 'president']]
df.tail()
def fixup(text):
    
    # remove punctuation
    text = ''.join([char for char in text if char == '-' or char not in string.punctuation])
    # remove special characters
    text = text.replace(r'^[*-]', '')
    # remove numbers
    text = ''.join([char for char in text if not char.isdigit()])
    # lowercase
    text = text.lower()
    
    # remove hanging whitespace
    text = " ".join(text.split())
    
    return text


df['text'] = df['text'].apply(fixup)
df.head(5)
# get length of sentence as variable
df['length'] = df['text'].apply(len)
# what are our longest sentences?
df.sort_values(by='length', ascending=False).head(10)
df.loc[3930][1]
# what are our shortest sentences?
df.sort_values(by='length').head(5)
# let's check the shortest sentences in our test set
df[pd.isnull(df['president'])].sort_values(by='length').head()
# sentences with just a few characters are of no use to us
df = df[df['length']>10]
# what are our shortest sentences now?
df.sort_values(by='length').head(5)
df['president'].value_counts()
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(strip_accents='unicode', ngram_range=(1,3), stop_words='english', min_df=6)
X = tfidf.fit_transform(df['text']).todense()
X.shape
tfidf.get_feature_names()
X = pd.DataFrame(data=X, columns=tfidf.get_feature_names())
df = df.drop(columns=['text', 'length'], axis=1)
df.head()
X = pd.DataFrame(np.hstack((df, X)))
X.shape
X.columns = ['sentence_id', 'president_id'] + tfidf.get_feature_names()
X.head()
X.shape
train = X[pd.isnull(X['sentence_id'])]
test = X[pd.notnull(X['sentence_id'])]
train.shape
X_train = train.drop(['sentence_id', 'president_id'], axis=1)
X_test = test.drop(['sentence_id', 'president_id'], axis=1)
def one_hot_encode(label):
    
    # initialize zero array
    vec = [0, 0, 0, 0, 0, 0]
    
    # set index of array corresponding to label = 1
    vec[label] = 1
    
    return vec

# save encoded labels as target for model
y_train = np.vstack(row for row in train['president_id'].apply(one_hot_encode).values)
y_train[600]
print('Train size:', X_train.shape)
print('Test size:', X_test.shape)
def create_model(lyrs=[X_train.shape[1], 1028, 512, 256], act='relu', opt='Adam', dr=0.25):
    
    model = Sequential()
    
    # create first hidden layer
    model.add(Dense(lyrs[0], input_dim=X_train.shape[1], activation=act))
    
    # create additional hidden layers
    for i in range(1,len(lyrs)):
        model.add(Dense(lyrs[i], activation=act))
    
    # add dropout, default is none
    model.add(Dropout(dr))
    
    # create output layer
    model.add(Dense(6, activation='softmax'))  # output layer
    
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    
    return model
model = create_model()
print(model.summary())
# train model on full train set, with 80/20 CV split
training = model.fit(X_train, y_train, epochs=5, batch_size=64, validation_split=0.2)
val_acc = np.mean(training.history['val_acc'])
print("\n%s: %.2f%%" % ('val_acc', val_acc*100))
# summarize history for accuracy
plt.plot(training.history['acc'])
plt.plot(training.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
predictions = model.predict(X_test)
pred_lbls = []
for pred in predictions:
    pred = list(pred)
    max_value = max(pred)
    max_index = pred.index(max_value)
    pred_lbls.append(max_index)

predictions = np.array(pred_lbls)
predictions.shape
test['president_id'] = predictions
test['president_id'].value_counts()
submission = test[['sentence_id','president_id']]
submission.columns = ['sentence', 'president']
submission.to_csv('rnn_1.csv', index=False)
submission.president.value_counts()