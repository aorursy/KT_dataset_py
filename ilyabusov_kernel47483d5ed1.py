import pandas as pd

import numpy as np



%pylab inline
df_train = pd.read_csv('../input/train.tsv', sep='\t')

df_test = pd.read_csv('../input/test.tsv', sep='\t')

df = pd.concat([df_train, df_test], sort=False)
df.head(10)
labels_count = df_train.Sentiment.value_counts()

plt.pie(labels_count, labels=labels_count.index)
from sklearn.feature_extraction.text import TfidfVectorizer



vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(df_train['Phrase'])
vectorized_test = vectorizer.transform(df_test['Phrase'])
from keras.models import Sequential

from keras.layers import Dense, Conv2D, Dropout

from keras.utils import to_categorical
model = Sequential()



model.add(Dense(128, input_shape=(X.shape[1],), activation='relu'))



model.add(Dropout(0.4))

model.add(Dense(64, activation = "relu"))

model.add(Dropout(0.4))

model.add(Dense(32, activation = "relu"))

          

model.add(Dense(5, activation='sigmoid'))



model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X, to_categorical(df_train['Sentiment']), epochs=5, batch_size=32, validation_split=0.25)
y_pred = model.predict(vectorized_test)

df_test['Sentiment']=y_pred.argmax(axis=-1)
df_test[['PhraseId', 'Sentiment']].to_csv('result.csv', index=False)