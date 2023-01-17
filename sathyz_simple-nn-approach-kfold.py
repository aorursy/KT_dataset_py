# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

import matplotlib.pyplot as plt

from keras.utils import to_categorical
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from sklearn.model_selection import train_test_split

import nltk
from nltk import wordpunct_tokenize

%matplotlib inline
%config InlineBackend.figure_format='retina'
train_df = pd.read_json("../input/train.json").set_index("id")
train_df.cuisine = train_df.cuisine.astype("category")
train_df.head()
train_df.ingredients.apply(len).describe().astype(int)
train_df.cuisine.describe()
train_df.cuisine.value_counts().plot(kind="barh", color="steelblue");
from collections import Counter

cuisines = train_df.cuisine.cat.categories.values.tolist()

texts = []
labels = []

label2index = { cuisine: i for i, cuisine in enumerate(cuisines)}
for i, row in train_df.iterrows():
    texts.append("\n".join(row.ingredients))
    labels.append(label2index[row.cuisine])

labels = to_categorical(np.asarray(labels, dtype=np.int32))

word_count = Counter()

def lemmatize(texts):
    global word_count
    wnl = nltk.WordNetLemmatizer()
    for text in texts:
        tokens_recipe = []
        for sentence in text.split("\n"):
            tokens_ingredient = [ wnl.lemmatize(w) for w in wordpunct_tokenize(sentence.lower()) if w.isalpha() ]
            word_count.update(tokens_ingredient)
            tokens_recipe.append(" ".join(tokens_ingredient))
        yield " ".join(tokens_recipe)

def preprocess(texts):
    processed_texts = list(lemmatize(texts))
    black_list = [ word for word, count in word_count.items() if count < 5 ]
    return [[ word for word in sentence.split() if word not in black_list] for sentence in processed_texts ]
    
tokenizer = Tokenizer(oov_token="<UNK>")
processed_texts = preprocess(texts)
tokenizer.fit_on_texts(processed_texts)
# sequences = tokenizer.texts_to_sequences(processed_texts)
# padded_sequences = pad_sequences(sequences, maxlen=100)
feature_matrix = tokenizer.texts_to_matrix(processed_texts)
word2index = tokenizer.word_index

print("Unique tokens: {}".format(len(word2index)))
feature_matrix.shape
from keras.layers import Dense, Dropout
from keras.models import Sequential

def build_model(hidden_units, dropout):
    model = Sequential()
    model.add(Dense(hidden_units, input_shape=[1722,], activation="relu", name="hidden"))
    model.add(Dropout(dropout, name="dropout"))
    model.add(Dense(20, name="output"))
    
    model.compile("adam", "categorical_hinge", metrics=["accuracy"])
    return model
## do a grid search for hyper parameters
# from keras.wrappers.scikit_learn import KerasClassifier
# from sklearn.model_selection import GridSearchCV

# np.random.seed(7)

# model = KerasClassifier(build_fn=build_model, epochs=10, verbose=0)

# param_grid = dict(
#     batch_size = [128,],  # (32, 64, 128, 256, 512,),
#     hidden_units = [2048,] , # [32, 64, 256, 512, 1024, 2048,],
#     dropout = [0.8,], # [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,],
#     )

# cv = [train_test_split( np.arange(39774), test_size=0.2), train_test_split( np.arange(39774), test_size=0.2), ]
# grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1, verbose=10, cv=cv)
# grid_result = grid.fit(feature_matrix, labels)
# print("best: {} using {}".format(grid_result.best_score_, grid_result.best_params_))
X_train, X_val, y_train, y_val = train_test_split(feature_matrix, labels, test_size=0.2, random_state=42)
test_df = pd.read_json("../input/test.json")
test_texts = []

for i, row in test_df.iterrows():
    test_texts.append("\n".join(row.ingredients))

processed_texts = preprocess(test_texts)
test_feature_matrix = tokenizer.texts_to_matrix(processed_texts)

test_feature_matrix.shape
from keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.metrics import hinge_loss, accuracy_score
from sklearn.model_selection import KFold

kf = KFold(n_splits=5, shuffle=True)

models = []
early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1)

oos_y = []
oos_pred = []

for i, (train_index, test_index) in enumerate(kf.split(feature_matrix), start=1):
    X_train, y_train = feature_matrix[train_index], labels[train_index]
    X_val, y_val = feature_matrix[test_index], labels[test_index]
    model = build_model(2048, 0.8)
    chk_point = ModelCheckpoint("best-model-{}.h5".format(i), monitor='val_loss', save_best_only=True, save_weights_only=True)
    model.fit(X_train, y_train, 
        validation_data=(X_val, y_val),
        epochs=50,
        callbacks=[early_stopping, chk_point],
        verbose=0,
        batch_size=128)

    # use the best model to predict
    model.load_weights("best-model-{}.h5".format(i))
    y_pred = model.predict(X_val)
    loss = hinge_loss(y_val.argmax(axis=1), y_pred)
    accuracy = accuracy_score(y_val.argmax(axis=1), y_pred.argmax(axis=1))
    
    oos_y.append(y_val)
    oos_pred.append(y_pred)
    models.append(model)
        
    print("Fold {}, loss: {}, accuracy: {:.2%}".format(i, loss, accuracy))
    
y_true = np.vstack(oos_y)
y_pred = np.vstack(oos_pred)

loss = hinge_loss(y_true.argmax(axis=1), y_pred)
accuracy = accuracy_score(y_true.argmax(axis=1), y_pred.argmax(axis=1))

print("loss: {}, accuracy: {:.2%}".format(loss, accuracy))
summary = np.zeros((20, 20), dtype=np.int32)
for y_true_i, y_pred_i in zip(y_true.argmax(axis=1), y_pred.argmax(axis=1)):
    summary[y_true_i, y_pred_i] += 1

summary_df = pd.DataFrame(summary, 
                          columns=cuisines, 
                          index=cuisines)

summary_df
import seaborn as sns 

summary_norm = ( summary / y_true.sum(axis=0) )
sns.heatmap( summary_norm, 
            vmin=0, vmax=1, center=0.5, 
            xticklabels=cuisines,
            yticklabels=cuisines);
test_pred = np.zeros((9944, 5), dtype=np.int32)

for i, model in enumerate(models):
    y_pred = model.predict(test_feature_matrix)
    test_pred[:, i] = y_pred.argmax(axis=1)
    
def voting(arr):
    return np.bincount(arr).argmax()

predictions = np.apply_along_axis(voting, 1, test_pred)
result = pd.Series( pd.Categorical.from_codes(predictions, cuisines), test_df.id, name="cuisine")
result.to_csv("submission.csv", header=True)
result.value_counts()
