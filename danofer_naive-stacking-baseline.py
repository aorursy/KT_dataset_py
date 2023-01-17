import pandas as pd

import numpy as np

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.linear_model import LogisticRegression, LogisticRegressionCV

from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing import sequence

from keras.models import Model, Input

from keras.layers import Dense, Embedding, GlobalAveragePooling1D, Dropout, SpatialDropout1D

from keras.preprocessing.text import Tokenizer
train_df = pd.read_csv("../input/train.csv")

test_df = pd.read_csv("../input/test.csv")
df = pd.concat([train_df['comment_text'], test_df['comment_text']], axis=0).fillna("BLANK")  # concat data for "cheating" in vectorizing
train_df.head()
(train_df.iloc[:,2:].apply(sum,axis=1)>1).sum()
print(train_df.comment_text.str.len().describe())
print(train_df.comment_text.str.split().str.len().describe())
print(test_df.comment_text.str.split().str.len().describe())
X_train = train_df["comment_text"].fillna("BLANK").values

y_train = train_df[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]].values

X_test = test_df["comment_text"].fillna("BLANK").values
i = 0

print("Comment: {}".format(X_train[i]))

print("Label: {}".format(y_train[i]))
# Set parameters:

max_features = 95000

maxlen = 84

batch_size = 32

embedding_dims = 60 #50

epochs = 3
print('Tokenizing data...')

tok = Tokenizer(num_words=max_features)

tok.fit_on_texts(list(X_train) + list(X_test))

x_train = tok.texts_to_sequences(X_train)

x_test = tok.texts_to_sequences(X_test)

print(len(x_train), 'train sequences')

print(len(x_test), 'test sequences')

print('Average train sequence length: {}'.format(np.mean(list(map(len, x_train)), dtype=int)))

print('Average test sequence length: {}'.format(np.mean(list(map(len, x_test)), dtype=int)))
print('Pad sequences (samples x time)')

x_train = sequence.pad_sequences(x_train, maxlen=maxlen)

x_test = sequence.pad_sequences(x_test, maxlen=maxlen)

print('x_train shape:', x_train.shape)

print('x_test shape:', x_test.shape)
print('Build model...')

comment_input = Input((maxlen,))



# we start off with an  embedding layer

comment_emb = Embedding(max_features, embedding_dims, input_length=maxlen)(comment_input)

# We see that we overfit straight away, so dropout may be useful

drp = SpatialDropout1D(0.1)(comment_emb)

# we add a GlobalAveragePooling1D, which will average the embeddings

# of all words in the document

main = GlobalAveragePooling1D()(drp)



# We project onto a single unit output layer, and squash it with a sigmoid:

output = Dense(6, activation='softmax')(main)



model = Model(inputs=comment_input, outputs=output)



model.compile(loss='categorical_crossentropy',

              optimizer='adam',

              metrics=['accuracy'])
hist = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)
# print('Build model...')

# comment_input = Input((maxlen,))



# # we start off with an  embedding layer

# comment_emb = Embedding(max_features, embedding_dims, input_length=maxlen)(comment_input)

# # We see that we overfit straight away, so dropout may be useful

# drp = Dropout(0.15)(comment_emb)

# # we add a GlobalAveragePooling1D, which will average the embeddings

# # of all words in the document

# main = GlobalAveragePooling1D()(drp)



# drp2 =  Dropout(0.25)(main)

# # We project onto a single unit output layer, and squash it with a sigmoid:

# output = Dense(6, activation='softmax')(drp2)



# model2 = Model(inputs=comment_input, outputs=output)



# model2.compile(loss='categorical_crossentropy',

#               optimizer='adam',

#               metrics=['accuracy'])



# hist2 = model2.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)
nrow_train = train_df.shape[0]



vectorizer = CountVectorizer(stop_words='english',min_df=3, max_df=0.97,max_features = 40000)

data = vectorizer.fit_transform(df)



col = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']



lr_preds = np.zeros((test_df.shape[0], len(col)))



X_train = data[:nrow_train]

X_test = data[nrow_train:]



for i, j in enumerate(col):

    print('fit '+j)

    lr_model = LogisticRegression(C=0.1, dual=True)

    lr_model.fit(X_train, train_df[j])

    lr_preds[:,i] = lr_model.predict_proba(X_test)[:,1]

print("done")
for i, j in enumerate(col):

    print(j,lr_preds[:,i].mean())
# Get predictions from our keras/fasttext model

ft_pred = model.predict(x_test)
# get mean of both submissions



y_pred = lr_preds+ft_pred

y_pred = y_pred/2.0

submission = pd.read_csv("../input/sample_submission.csv")
submission[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]] = y_pred
submission.head()
submission.to_csv("submission_fasttext_1.csv", index=False)