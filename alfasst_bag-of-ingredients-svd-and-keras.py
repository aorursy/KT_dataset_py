import pandas as pd
import numpy as np
from collections import Counter
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.optimizers import SGD

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
Train = pd.read_json('../input/train.json')
Test = pd.read_json('../input/test.json')
TrainIngredients = list(Train['ingredients'].values)
Ingredients = set([item for sublist in TrainIngredients for item in sublist])

print('Unique Ingredients: ', len(Ingredients))
def BagOfWords(BigList, Vocab):
    Matrix = []
    for List in BigList:
        counter = Counter(List)
        Row = [counter.get(w, 0) for w in Vocab]
        Matrix.append(Row)
    Matrix = np.array(Matrix)
    return Matrix
TrainMatrix = BagOfWords(TrainIngredients, Ingredients)

print('Train Feature Matrix: ', TrainMatrix.shape)
n = 600

SVD = TruncatedSVD(n_components=n)

X = SVD.fit_transform(TrainMatrix)

y = pd.get_dummies(Train['cuisine'])
Model = Sequential()
Model.add(Dense(32, activation='relu', input_dim=n))
Model.add(Dropout(0.2))
Model.add(Dense(64, activation='relu'))
Model.add(Dropout(0.2))
Model.add(Dense(32, activation='relu'))
Model.add(Dropout(0.2))
Model.add(Dense(20, activation='softmax'))

Stock = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
Model.compile(loss='categorical_crossentropy', optimizer=Stock, metrics=['accuracy'])

Model.fit(X, y, epochs=100, batch_size=32)

print(Model)
TestIngredients = list(Test['ingredients'].values)
UTI = set([item for sublist in TestIngredients for item in sublist])
NewIngredients = [i for i in UTI if i not in Ingredients]
TestMatrix = BagOfWords(TestIngredients, Ingredients)

print('Unique Ingredients of test: ', len(UTI))
print('New Ingredients: ', len(NewIngredients))
print('Test Feature Matrix: ', TestMatrix.shape)
TestMatrix = SVD.transform(TestMatrix)

Predicted = Model.predict(TestMatrix)
Predicted = Predicted.argmax(axis=-1)
Predicted = to_categorical(Predicted).astype(np.int64)
Predicted = pd.DataFrame(Predicted, columns = list(y))

print(Predicted.shape)
Predicted = list(Predicted.idxmax(axis=1))
IDs = list(Test['id'].values)
Out = pd.DataFrame({'id':IDs, 'cuisine':Predicted})

Out.to_csv('sample_submission.csv', index = False)