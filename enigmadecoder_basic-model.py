import numpy as np
import pandas as pd
import os

from sklearn.model_selection import GridSearchCV
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dropout, Dense, Flatten
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn import metrics

# print(os.listdir("../input"))
train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')
submission_df = pd.read_csv('../input/sample_submission.csv')
print(train_df.columns, test_df.columns, submission_df.columns)
print(train_df.shape, test_df.shape, submission_df.shape)
max_features = 5000
tokenizer = Tokenizer(num_words= max_features)
x_train, x_val, y_train, y_val = train_test_split(train_df.question_text, 
                                                  train_df.target, 
                                                  test_size=0.33, 
                                                  random_state=6122018, 
                                                  stratify = train_df.target)
tokenizer.fit_on_texts(x_train)
maxlen = 50 # max words in a question
x_train = tokenizer.texts_to_sequences(x_train)
x_train = pad_sequences(x_train, maxlen=maxlen)

x_val = tokenizer.texts_to_sequences(x_val)
x_val = pad_sequences(x_val, maxlen=maxlen)

x_test = tokenizer.texts_to_sequences(test_df.question_text)
x_test = pad_sequences(x_test, maxlen=maxlen)
print(x_train.shape, x_val.shape, y_train.shape, y_val.shape)
batch_size = 1024
hidden_size = 32
use_dropout = True
def create_model(hidden_size, dropout_rate):
    model = Sequential()
    model.add(Embedding(max_features, 128, input_length=maxlen))
    model.add(LSTM(hidden_size, return_sequences=True))
    model.add(Dropout(dropout_rate))
    model.add(LSTM(int(hidden_size/2), return_sequences=True))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
    return model

model = KerasClassifier(build_fn=create_model, epochs=2, batch_size=batch_size, verbose=1)
# del(train_df)
# del(test_df)
# del(submission_df)
%env JOBLIB_TEMP_FOLDER=/tmp
# define the grid search parameters
hidden_size = [32]
dropout_rate = [0.5]
# dropout_rate = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
param_grid = dict(hidden_size=hidden_size, dropout_rate=dropout_rate)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1)
grid_result = grid.fit(x_train, y_train)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
# help(grid_result.predict)
# pred_noemb_val_y = grid.predict([x_val], batch_size=1024, verbose=1)
pred_noemb_val_y = grid.predict([x_val])
for thresh in np.arange(0.1, 0.501, 0.01):
    thresh = np.round(thresh, 2)
    print("F1 score at threshold {0} is {1}".format(thresh, metrics.f1_score(y_val, (pred_noemb_val_y>thresh).astype(int))))
pred_noemb_val_y
# push these two statemets tothe begining of the notebook
x_test = tokenizer.texts_to_sequences(test_df.question_text)
x_test = pad_sequences(x_test, maxlen=maxlen)

# pred_noemb_val_y = model.predict([x_test], batch_size=1024, verbose=1)
pred_noemb_val_y = grid.predict([x_test])
import seaborn as sns
sns.distplot(pred_noemb_val_y)
# threshold = 0.29
# submission_df.prediction = (pred_noemb_val_y[:,0] > threshold).astype(np.int)
submission_df.prediction = pred_noemb_val_y

submission_df.prediction.value_counts()
submission_df.to_csv('submission.csv', index=False)