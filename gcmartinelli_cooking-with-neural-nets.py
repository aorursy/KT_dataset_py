%matplotlib inline
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MultiLabelBinarizer

from tensorflow.python import keras
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, Softmax
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.activations import relu

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
df_train = pd.read_json('../input/train.json')
df_test = pd.read_json('../input/test.json')
df_train.head()
df_test.head()
num_cuisines = df_train.cuisine.unique().shape[0]
num_cuisines
def tt_split(df_train, df_test, train_size):  
    X = np.array(df_train.drop(['id', 'cuisine'], axis=1))
    
    cuisine_vector = [[c] for c in df_train.cuisine]
    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(cuisine_vector)
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=train_size)
    return X_train, X_val, y_train, y_val, mlb

def vect_train_test(dftrain, dftest, n_words=1000, words=None, vect=None):
    if vect == 'tfid':
        vectorizer = TfidfVectorizer(max_features = n_words)
    else:
        vectorizer = CountVectorizer(max_features = n_words)
    ingredients_train = dftrain.ingredients
    words_train = [' '.join(x) for x in ingredients_train]
    ingredients_test = dftest.ingredients
    words_test = [' '.join(x) for x in ingredients_test]
    if isinstance(words, pd.Series):
        bag_of_words = vectorizer.fit(words)
    else:
        bag_of_words = vectorizer.fit(words_train)

    ing_array_train = bag_of_words.transform(words_train).toarray()
    ing_array_test = bag_of_words.transform(words_test).toarray()

    df_ing_train = pd.DataFrame(ing_array_train, columns=vectorizer.vocabulary_)
    df_ing_test = pd.DataFrame(ing_array_test, columns=vectorizer.vocabulary_)

    df_train = dftrain.merge(df_ing_train, 
                          left_index=True, 
                          right_index=True).drop('ingredients', axis=1)
    df_test= dftest.merge(df_ing_test, 
                          left_index=True, 
                          right_index=True).drop('ingredients', axis=1)
    return df_train, df_test
max_ing = 1000
df_train_new, df_test_new = vect_train_test(df_train, df_test, n_words=max_ing)
X_train, X_val, y_train, y_val, mlb = tt_split(df_train_new, df_test_new, 0.85)
# m = Sequential([
#     Dense(500, input_dim=1000, activation='relu'),
#     Dropout(0.15),
#     Dense(250, input_dim=1000, activation='relu'),
#     Dense(20, activation='softmax')
# ])

# m.compile(loss='categorical_crossentropy',
#         metrics=['accuracy'],
#               optimizer=Adam(lr=0.002))
# m.fit(
#     X_train, y_train,
#     epochs=5,
#     verbose=1,
#     validation_data=(X_val, y_val)
# )
# pred = m.predict(X_val)
# plt.scatter(pred.argmax(axis=1), y_val.argmax(axis=1), alpha=0.005)
max_ing = 1000
df_train_new, df_test_new = vect_train_test(df_train, df_test, 
                                            n_words=max_ing, vect='tfid')
X_train, X_val, y_train, y_val, mlb = tt_split(df_train_new, df_test_new, 0.85)
m = Sequential([
    Dense(500, input_dim=1000, activation='relu'),
    Dropout(0.15),
    Dense(250, input_dim=1000, activation='relu'),
    Dense(20, activation='softmax')
])

m.compile(loss='categorical_crossentropy',
        metrics=['accuracy'],
              optimizer=Adam(lr=0.002))

m.fit(
    X_train, y_train,
    epochs=5,
    verbose=1,
    validation_data=(X_val, y_val)
)
y_pred = m.predict(df_test_new.drop('id', axis=1))
y_cat = mlb.classes_[y_pred.argmax(axis=1)]
y_cat
df_sub = pd.DataFrame(np.array([df_test.id, y_cat]).T, 
                      columns=['id', 'cuisine']).set_index('id')

df_sub.to_csv('submission_nn.csv')
