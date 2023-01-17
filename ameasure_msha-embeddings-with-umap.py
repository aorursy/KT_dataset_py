# enable interactive plots in jupyter notebooks

%matplotlib notebook

# download the MSHA data

!wget --no-clobber 'https://github.com/ameasure/autocoding-class/raw/master/msha.xlsx'

# install the UMAP library which allows us to generate 2 dimensional projections of high dimensional vectors

!pip install umap-learn
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.preprocessing import LabelBinarizer



import pandas as pd



# read in the data and split it into training and validation

df = pd.read_excel('msha.xlsx')

df['ACCIDENT_YEAR'] = df['ACCIDENT_DT'].apply(lambda x: x.year)

df['ACCIDENT_YEAR'].value_counts()

df_train = df[df['ACCIDENT_YEAR'].isin([2010, 2011])].copy()

df_valid = df[df['ACCIDENT_YEAR'] == 2012].copy()

print('training rows:', len(df_train))

print('validation rows:', len(df_valid))



# create bag of words features

vectorizer = CountVectorizer()

vectorizer.fit(df_train['NARRATIVE'])

X_train = vectorizer.transform(df_train['NARRATIVE'])

X_valid = vectorizer.transform(df_valid['NARRATIVE'])



# keras only accepts a one-hot encoding of the training labels

# we do that here

label_encoder = LabelBinarizer().fit(df_train['INJ_BODY_PART'])

y_train = label_encoder.transform(df_train['INJ_BODY_PART'])

y_valid = label_encoder.transform(df_valid['INJ_BODY_PART'])

n_codes = len(label_encoder.classes_)
from keras.models import Model

from keras.layers import Dense, Input, Dropout

from keras.optimizers import Adam



text_input = Input(shape=(X_train.shape[1],))

layer1 = Dense(units=100, activation='relu')(text_input)

do1 = Dropout(0.5)(layer1)

output = Dense(units=n_codes, activation='softmax')(do1)



model = Model(inputs=[text_input], outputs=[output])



optimizer = Adam(lr=.001)

model.compile(optimizer=optimizer, 

              loss='categorical_crossentropy', 

              metrics=['accuracy'])
# validation_data - the data we evaluate the model on at the end of each epoch

model.fit(x=X_train, y=y_train,

          validation_data=(X_valid, y_valid),

          batch_size=32, epochs=5)
new_model = Model(inputs=[text_input], outputs=[layer1])
preds = new_model.predict(X_valid)
preds.shape
import umap



embedding = umap.UMAP().fit_transform(preds)
embedding.shape
import matplotlib.pyplot as plt

import seaborn as sns



sns.set(context='paper', style='white')

fig, ax = plt.subplots(figsize=(12,10))

# use colors to indicate the injured part in each narrative

color = y_valid.argmax(axis=1)

# generate a scatter plot of our 2d embeddings, each point represents a narrative

sc = plt.scatter(embedding[:, 0], embedding[:, 1], c=color, cmap='Spectral', s=0.1)

# add the color bar to the side of our graph

cbar = fig.colorbar(sc)

# set the tick marks and labels on our color bar

cbar.set_ticks(range(len(label_encoder.classes_)))

cbar.ax.set_yticklabels(label_encoder.classes_)

# remove the x and y ticks from our scatter plot

plt.setp(ax, xticks=[], yticks=[])
# identify examples with a label index < 20 (i.e. one of first 20 labels in label_encoder.classes_)

cond = y_valid.argmax(axis=1) < 20

# create X_samp as subset of X_valid that meets this condition

X_samp = X_valid[cond]

# create y_samp as subset of y_valid that meets this condition

y_samp = y_valid[cond]
# generate dense embeddings for each of these examples

preds = new_model.predict(X_samp)

print(preds.shape)

# project these 100 dimensional embeddings to 2 dimensional space using UMAP

embedding = umap.UMAP().fit_transform(preds)

print(embedding.shape)
#!pip install mpld3

#import mpld3

#mpld3.enable_notebook()
%matplotlib notebook

sns.set(context='paper', style='white')

# create a 12 inch by 10 inch plot

fig, ax = plt.subplots(figsize=(12,10))

# map each label in y_samp to its corresponding index position

color = y_samp.argmax(axis=1)



# generate a scatter plot, the 0th dimension of the umap on the x axis, the 1st dim on the y

# c=color says create a separate color for each "body part"

# cmap='tab20' means use the 20 colors matplotlib calls tab20

# s=2 says set the size of each point to 2

sc = plt.scatter(embedding[:, 0], embedding[:, 1], c=color, cmap='tab20', s=10)

# add the corresponding colorbar / legend to the plot (the vertical bar on the right)

cbar = fig.colorbar(sc)

# set 20 ticks on the colorbar, one for each of our 20 codes

cbar.set_ticks(range(0,20))

# put 20 "body part" labels in the colorbar, one for each of the first 20 codes

cbar.ax.set_yticklabels(label_encoder.classes_[0:20])

# remove the x and y tick marks from our scatterplot

plt.setp(ax, xticks=[], yticks=[])