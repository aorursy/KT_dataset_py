import numpy as np 

import pandas as pd 

from sklearn.linear_model import LogisticRegression, SGDClassifier

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import train_test_split

from sklearn.metrics import f1_score, accuracy_score

import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap

from sklearn.ensemble import BaggingClassifier

from keras.callbacks import LearningRateScheduler
# Вы можете тестить свои алгоритмы прямо на kaggle, данные здесь уже есть, их нужно только загрузить

train_x = pd.read_csv('../input/bird-or-aircraft-2020/train_x.csv', index_col=0, header=None)

train_y = pd.read_csv('../input/bird-or-aircraft-2020/train_y.csv', index_col=0)

test_x = pd.read_csv('../input/bird-or-aircraft-2020/test_x.csv', index_col=0, header=None)
# Этот блок был бы нужен, если бы у нас классы назывались 'Bird' и 'Airplane'

# mappping_type = {'Bird': 0, 'Airplane': 1}

# train_y = train_y.replace({"target": mappping_type})
# в train_y у нас лежат лейблы (птиццы и самолеты = 0 и 1)

train_y
# 7200 картинок 32х32х3

train_x
# посмотрим на картинки

_train_x = train_x.values.reshape(7200,32,32,3)

_test_x = test_x.values.reshape(4800,32,32,3)
i = np.random.randint(7200)

print(train_y.iloc[i])

plt.imshow(_train_x[i])

plt.show()
# нормируем данные, этого можно и не делать, почему?

train_feature_matrix, test_feature_matrix = train_x/255, test_x/255
# здесь должен быть ваш классификатор

clf = SGDClassifier(max_iter=1000, tol=1e-3, loss='log', penalty='l2')

clf.fit(train_feature_matrix, np.ravel(train_y))
accuracy_score(clf.predict(train_feature_matrix), np.ravel(train_y)) # np.ravel - разворачиваем массив в строку
predict_y = clf.predict_proba(test_x)
# это можно не менять, просто упаковываются данные в csv

sample = pd.DataFrame(np.array([[i, x.argmax()] for i, x in enumerate(predict_y)]), columns=['id', 'target'])



# mappping_type_inv = {0: 'Bird', 1: 'Airplane'}

# sample = sample.replace({'target': mappping_type_inv})
sample.to_csv('submit.csv', index=False)

# на этом месте формируется ответ в виде файла submit.csv на kaggle,

# вы должны сделать commit своего кернела, а затем перейти в раздел data и нажать submit to competition