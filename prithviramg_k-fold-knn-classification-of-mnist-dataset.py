#import necessary libraries
import warnings
warnings.filterwarnings("ignore")#ignore warnings message displayed

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random

from sklearn.model_selection import train_test_split
from sklearn import model_selection
from sklearn.model_selection import cross_val_score

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score
dataframe = pd.read_csv("../input/MNIST.csv")
dataframe
print('Each numerical image has below samples')
print(dataframe['label'].value_counts())
test_data = pd.DataFrame(columns=dataframe.columns)
rows = pd.concat([dataframe[dataframe.label == 0].sample(1000),\
                 dataframe[dataframe.label == 1].sample(1000),\
                 dataframe[dataframe.label == 2].sample(1000),\
                 dataframe[dataframe.label == 3].sample(1000),\
                 dataframe[dataframe.label == 4].sample(1000),\
                 dataframe[dataframe.label == 5].sample(1000),\
                 dataframe[dataframe.label == 6].sample(1000),\
                 dataframe[dataframe.label == 7].sample(1000),\
                 dataframe[dataframe.label == 8].sample(1000),\
                 dataframe[dataframe.label == 9].sample(1000),\
                  ])
test_data = test_data.append(rows,ignore_index=True)
input_data = dataframe
input_data = input_data.drop(rows.index)
print('Each numerical image in input data has below samples')
print(input_data['label'].value_counts())
print('Each numerical image in test data has below samples')
print(test_data['label'].value_counts())
input_data.head()
test_data.head()
input_data_label = input_data['label']
test_data_label = test_data['label']
input_data = input_data.drop(columns="label")
test_data = test_data.drop(columns="label")
input_data.head()
test_data_label.head()
plt.figure(figsize=(15,15))

for i in range(1,7):
    plt.subplot(1,6,i)
    idx = random.randint(1,32000)
    grid_data = input_data.iloc[idx].as_matrix().reshape(28,28)  # reshape from 1d to 2d pixel array
    plt.imshow(grid_data, interpolation = "none", cmap = "gray")
    plt.title(str(input_data_label.iloc[idx]))

plt.show()
input_data_std = StandardScaler(with_mean='False').fit_transform(input_data)
test_data_std = StandardScaler(with_mean='False').fit_transform(test_data)
cv_scores = [];
count = 5;
max_score = 0;
neighbors = list(filter(lambda x: x % 2 != 0, range(1,100)))
for k in neighbors:
    knn = KNeighborsClassifier(n_neighbors=k,n_jobs=-1)
    print('selected k is %d' % (k));
    scores = cross_val_score(knn,input_data_std,input_data_label,cv=4, scoring='accuracy')
    print('\bmean score is %f' % (scores.mean()))
    cv_scores.append(scores.mean())
    if max_score<scores.mean():
        max_score = scores.mean();
        count = 5;
    else:
        count = count - 1;
    if count == 0:
        break;
error = [1-x for x in cv_scores]
optimal_k = neighbors[error.index(min(error))]
knn = KNeighborsClassifier(n_neighbors=optimal_k,n_jobs=-1)
knn.fit(input_data_std,input_data_label)
for i in range(1,10):
    idx = random.randint(1,10000)
    y_pred = knn.predict(test_data_std[idx].reshape(1,784))
    print('predicted value is '+str(y_pred))
    print('ground truth value is '+str(test_data_label.iloc[idx]))
y_pred = knn.predict(test_data_std)
mismatch = 0
for i in range(0,len(y_pred)-1):
    if y_pred[i] == test_data_label.loc[i]:
        continue;
    else:
        mismatch = mismatch + 1;
accuracy = (1-(mismatch/len(y_pred)))*100;        
print('\nThe accuracy of the knn classifier for k = %d is %f%%' % (optimal_k, accuracy))

