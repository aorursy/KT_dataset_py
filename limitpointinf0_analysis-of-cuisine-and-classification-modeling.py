import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")

with open('../input/train.json', 'r') as f:
    txt = f.read()
df = pd.DataFrame(json.loads(txt))
df.head()
plt.figure(figsize=(15,5))
plt.title('Recipe Count per Cuisine in Data')
ax = df.cuisine.value_counts().plot()
plt.xticks(np.arange(len(df.cuisine.unique())), df.cuisine.value_counts().index, rotation=80)
plt.show()
from wordcloud import WordCloud

def plot_wordcloud(text, title=None, max = 1000, size=(10,5), title_size=16):
    """plots wordcloud"""
    wordcloud = WordCloud(max_words=max).generate(text)
    plt.figure(figsize=size)
    plt.title(title, size=title_size)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")

tok_list = sum([x for x in df.ingredients],[])
tok_list = ' '.join(tok_list)
plot_wordcloud(tok_list, title='Ingredients')
df['num_in'] = df.ingredients.map(lambda x: len(x))
plt.figure(figsize=(15,5))
plt.title('Number of Ingredients per Cuisine')
ax = sns.boxplot(x="cuisine", y="num_in", data=df)
plt.xticks(rotation=80)
plt.show()
for c in df.cuisine.unique():
    temp = df[df.cuisine == c]
    txt = ' '.join(sum([x for x in temp.ingredients], []))
    plot_wordcloud(txt, title=c)
    break
df['joined'] = df.ingredients.map(lambda x: ' '.join(x))
df_nb = df[['cuisine','joined']]
df_nb.head()
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

count_vect = CountVectorizer()
tfidf_transformer = TfidfTransformer()
X = count_vect.fit_transform(df_nb.joined)
X = tfidf_transformer.fit_transform(X)
X.shape
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score

clf = MultinomialNB()
scores = cross_val_score(clf, X, df_nb.cuisine, cv=5)
print('accuracy CV:',scores)
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.optimizers import SGD, Adam
from keras.callbacks import History
from sklearn.model_selection import train_test_split

def simple_NN(input_shape, nodes_per=[60], hidden=0, out=2, act_out='softmax', act_hid='relu', drop=True, d_rate=0.1):
  """Generate a keras neural network with arbitrary number of hidden layers, activation functions, dropout rates, etc"""
  model = Sequential()
  #adding first hidden layer with 60 nodes (first value in nodes_per list)
  model.add(Dense(nodes_per[0],activation=act_hid,input_shape=input_shape))
  if drop:
      model.add(Dropout(d_rate))
  try:
    if hidden != 0:
      for i,j in zip(range(hidden), nodes_per[1:]):
          model.add(Dense(j,activation=act_hid))
          if drop:
              model.add(Dropout(d_rate))
    model.add(Dense(out,activation=act_out))
    return(model)
  except:
    print('Error in generating hidden layers')

ch_dict = dict([(y,x) for x,y in enumerate(set(df_nb.cuisine))])
y = np.array([ch_dict[x] for x in df_nb.cuisine])
y = to_categorical(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

print('creating and training model...')
model = simple_NN(input_shape=(X.shape[1],), nodes_per=[100, 100], hidden=1, out=y.shape[1], drop=True)
model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['acc'])
early_stopping_monitor = EarlyStopping(patience=3)
history = model.fit(X_train,y_train, epochs=1000, validation_data=(X_test, y_test), callbacks=[early_stopping_monitor], verbose=True)
print('model trained')

with open('../input/test.json', 'r') as f:
    txt = f.read()
df_test = pd.DataFrame(json.loads(txt))
df_test['joined'] = df_test.ingredients.map(lambda x: ' '.join(x))
df_test = df_test.drop(['ingredients'], axis=1)
df_test.head()
dec_dict = dict([(x,y) for y,x in ch_dict.items()])
X_test = np.array(df_test.joined)
X_test = count_vect.transform(X_test)
X_test = tfidf_transformer.transform(X_test)

preds = model.predict(X_test)
y_test = [dec_dict[np.argmax(x)] for x in preds]
df_test['cuisine'] = y_test
df_test.head()
df_test = df_test.drop('joined', axis=1)
df_test.to_csv('result.csv', index=False)
print('written to csv.')