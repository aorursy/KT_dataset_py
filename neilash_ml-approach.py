# ALL imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style; style.use('ggplot')
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
# Create dataframes train and test
train = pd.read_csv('../input/drugsComTrain_raw.csv')
test = pd.read_csv('../input/drugsComTest_raw.csv')
"""
Notes on CountVectorizer hyperparameters:
- binary=True: all nonzero counts set to 1 (rather than integer counts)
- stop_words=stopwords.words('english'): removes words like 'a', 'the', 'be', etc.
- lowercase=True: all text lowercase
- min_df=3: ignore terms that appear in less than 3 documents
- max_df=0.9: ignore terms that appear in more than 90% of the documents
- max_features=5000: feature vector size is 5000
"""

# Get review text
reviews = np.vstack((train.review.values.reshape(-1, 1), 
                     test.review.values.reshape(-1, 1)))

# Set up function to vectorize reviews
vectorizer = CountVectorizer(binary=False, stop_words=stopwords.words('english'),
                             lowercase=True, min_df=3, max_df=0.9, max_features=500)

# Vectorize reviews
X = vectorizer.fit_transform(reviews.ravel()).toarray()

# Get ratings
ratings = np.concatenate((train.rating.values, test.rating.values)).reshape(-1, 1)

# # Set up one-hot encoder
# ohe = OneHotEncoder(categories='auto')

# # One-hot encode ratings
# y = ohe.fit_transform(ratings).toarray()
y = ratings
# stopwords.words('english')
# ratings[:10], y[:10]
# Train/test split
X_train, X_test = X[:train.values.shape[0], :], X[train.values.shape[0]:, :] 
y_train, y_test = y[:train.values.shape[0]], y[train.values.shape[0]:]
X.shape, y.shape, X_train.shape, y_train.shape, X_test.shape, y_test.shape
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X_train[:5000], y_train[:5000])
pred = lin_reg.predict(X_train[5000:])
np.sum(np.abs(y_train[5000:] - pred[:])) / (161297 - 5000)
from sklearn.metrics import mean_absolute_error
mean_absolute_error(y_train[5000:], pred)
# import keras as K
# from keras.models import Sequential
# from keras.layers import Dense, LSTM, Embedding
# from keras import metrics, losses

# model = Sequential()
# """ TRY units=2500 """
# model.add(Dense(units=250, activation='relu', input_dim=5000))
# model.add(Dense(units=10, activation='relu'))

# model.compile(loss=losses.categorical_crossentropy, optimizer='adam', metrics=[metrics.categorical_accuracy])
# model.summary()
# model.fit(X_train, y_train, epochs=1, batch_size=128, verbose=1, validation_data=(X_test, y_test));
# correct_count = 0

# for i in range(y_test.shape[0]):
# #     print(model.predict_classes(X_test[i].reshape(1, -1))[0] + 1, np.where(y_test[i] == 1)[0][0] + 1)
#     if (model.predict_classes(X_test[i].reshape(1, -1))[0] + 1) == (np.where(y_test[i] == 1)[0][0] + 1):
#         correct_count += 1
        
# correct_count / y_test.shape[0]
# np.where(y_test[0] == 1)[0][0]
# # Create neural net model
# nn = MLPClassifier(hidden_layer_sizes=(2500,), activation='relu', max_iter=1000)
# nn.fit(X_train, y_train)
# accuracy_score(nn.predict(X), y)
# from sklearn.linear_model import LogisticRegression
# log_reg = LogisticRegression(solver='lbfgs', multi_class='auto')
# log_reg.fit(X_train, y_train)
# accuracy_score(log_reg.predict(X_test), y_test)
# from sklearn.svm import SVC
# svm = SVC(kernel='rbf')
# svm.fit(X_train, y_train)
# accuracy_score(svm.predict(X_test), y_test)