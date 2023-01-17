from keras.layers import Input, Dense, Activation, merge, Conv1D, Dropout, Embedding, GlobalMaxPooling1D

from keras.models import Model

from keras.callbacks import Callback

from keras.optimizers import Adam
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss
%matplotlib inline

import seaborn as sns

import matplotlib.pyplot as plt
import pandas as pd

import numpy as np
train_df = pd.read_csv("../input/train.csv")

test_df = pd.read_csv("../input/test.csv")
labels = train_df.Survived.values

train_df.drop('Survived', axis=1, inplace=True)
train_df = train_df.fillna(0)

test_df = test_df.fillna(0)
train_df["Sex"] = train_df["Sex"].apply(lambda x: 1 if x == "male" else 0)

test_df["Sex"] = train_df["Sex"].apply(lambda x: 1 if x == "male" else 0)

train_df["Cabin"] = train_df["Cabin"].apply(lambda x: 1 if x != 0 else 0)

test_df["Cabin"] = train_df["Cabin"].apply(lambda x: 1 if x != 0 else 0)
train_df.head()
numeric_features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Cabin"]
X_numeric = train_df[numeric_features].values
X_numeric.shape
text_features = ["Name", "Ticket", "Embarked"]
def load_data(symbols):

    vocab = {}

    words = list(symbols.lower())

    for i, word in enumerate(words):

        if word not in vocab:

            vocab[word] = len(vocab)

    print('corpus length:', len(words))

    print('vocab size:', len(vocab))

    return vocab
all_symbols = ""

for x in train_df[text_features].values:

    all_symbols += " ".join(map(str, x)) + " "
vocab = load_data(all_symbols)
max_name_length = train_df.Name.apply(len).max()

max_ticket_length = train_df.Ticket.apply(len).max()

train_df["Embarked"] = train_df.Embarked.apply(lambda x: "s" if x == 0 else x.lower())

max_embarked_length = train_df.Embarked.apply(len).max()
X_text = np.zeros((891, max_name_length + max_ticket_length + max_embarked_length))
for e, i in enumerate(train_df[text_features].iterrows()):

    name = i[1]["Name"].lower()

    ticket = i[1]["Ticket"].lower()

    emb = i[1]["Embarked"].lower()

    for p, w in enumerate(name):

        X_text[e, p] = vocab[w]

    for p, w in enumerate(ticket):

        X_text[e, p + max_name_length] = vocab[w]

    for p, w in enumerate(emb):

        X_text[e, p + max_name_length + max_ticket_length] = vocab[w] 
X_text.shape
split_n = int(0.25 * len(train_df))
X_text_train, X_text_test = X_text[split_n:], X_text[:split_n]

X_numeric_train, X_numeric_test = X_numeric[split_n:], X_numeric[:split_n]

y_train, y_test = labels[split_n:], labels[:split_n]
y_train.mean(), y_test.mean()
numeric_input = Input(shape=(7,), name='numeric_input')

y = Dense(3)(numeric_input)



text_input = Input(shape=(101,), name='text_input')

x = Embedding(len(vocab), 64, input_length=101) (text_input)

x = Conv1D(16, 4, activation='relu', subsample_length=1)(x)

x = GlobalMaxPooling1D()(x)

x = Dense(16)(x)

x = Dropout(0.5)(x)



conv_output = Dense(1, activation='sigmoid', name='conv_output')(x)



x = merge([x, y], mode='concat')



preds = Dense(1, activation='sigmoid', name='main_output')(x)
adam = Adam(lr=0.0001)
model = Model(input=[numeric_input, text_input], output=[preds, conv_output])

model.compile(loss='binary_crossentropy', 

              optimizer=adam,

              metrics=["accuracy"],

              loss_weights=[1, 0.2])
model.summary()
N_EPOCHS = 100
#The code below runs about 22 sec on my Titan X
%%time

train_scores = []

test_scores = []

for epoch in range(N_EPOCHS):

    model.fit([X_numeric_train, X_text_train], [y_train, y_train], nb_epoch=1, batch_size=8, verbose=0)

    probas = model.predict([X_numeric_train, X_text_train])[0]

    a, r, l = accuracy_score(y_train, probas > 0.5), roc_auc_score(y_train, probas), log_loss(y_train, probas)

    train_scores.append((a, r, l))

    probas = model.predict([X_numeric_test, X_text_test])[0]

    a, r, l = accuracy_score(y_test, probas > 0.5), roc_auc_score(y_test, probas), log_loss(y_test, probas)

    test_scores.append((a, r, l))
probas = model.predict([X_numeric_test, X_text_test])[0]

probas.mean()
train_scores = pd.DataFrame([x for x in train_scores], columns=["accuracy", "roc_auc", "log_loss"])

test_scores = pd.DataFrame([x for x in test_scores], columns=["accuracy", "roc_auc", "log_loss"])

train_scores["phase"] = "train"

test_scores["phase"] = "test"

scores = pd.concat([train_scores, test_scores])

scores["epoch"] = scores.index
plt.plot(scores[scores.phase=="train"].epoch, scores[scores.phase=="train"].roc_auc)

plt.plot(scores[scores.phase=="test"].epoch, scores[scores.phase=="test"].roc_auc)

plt.show()
plt.plot(scores[scores.phase=="train"].epoch, scores[scores.phase=="train"].accuracy)

plt.plot(scores[scores.phase=="test"].epoch, scores[scores.phase=="test"].accuracy)

plt.show()
plt.plot(scores[scores.phase=="train"].epoch, scores[scores.phase=="train"].log_loss)

plt.plot(scores[scores.phase=="test"].epoch, scores[scores.phase=="test"].log_loss)

plt.show()
X_numeric_submit = test_df[numeric_features].values
X_numeric_submit.shape
X_text_submit = np.zeros((418, max_name_length + max_ticket_length + max_embarked_length))
for e, i in enumerate(test_df[text_features].iterrows()):

    name = i[1]["Name"].lower()

    ticket = i[1]["Ticket"].lower()

    emb = i[1]["Embarked"].lower()

    for p, w in enumerate(name):

        X_text_submit[e, p] = vocab[w]

    for p, w in enumerate(ticket):

        X_text_submit[e, p + max_name_length] = vocab[w]

    for p, w in enumerate(emb):

        X_text_submit[e, p + max_name_length + max_ticket_length] = vocab[w] 
X_text_submit.shape
submit_probas = model.predict([X_numeric_submit, X_text_submit])[0]
np.mean(submit_probas > 0.5)
test_df.head()
#print("PassengerId,Survived")

#for i, s in zip(test_df.PassengerId.values, submit_probas):

#    print(i, 1 if s > 0.5 else 0, sep=",")