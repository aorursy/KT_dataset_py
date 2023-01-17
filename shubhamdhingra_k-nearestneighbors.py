import numpy as np

from sklearn import preprocessing, model_selection, neighbors

import pandas as pd
col = ['id','clump_thickness','unif_cell_size','unif_cell_shape','marg_adhesion','single_epith_cell_size','bare_nuclei','bland_chrom','norm_nucleoli','mitoses','class']

df = pd.read_csv(r'../input/breast-cancer-wisconsin.data.txt', names = col)



df.replace('?', -99999, inplace= True) # k-NN handles outliers very badly

df.drop(['id'], 1, inplace=True)  # dropping the column(1) which has no role in the algorithm



df.head()

x = np.array(df.drop(['class'], 1)) # Features

y = np.array(df['class']) # Labels
x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size = 0.2, random_state = 20)

                                                         # random_state fixes the values of data which algorithm takes to train and test so that accuracy value remains same everytime it gets executed. 
clf = neighbors.KNeighborsClassifier()

clf.fit(x_train, y_train)
accuracy = clf.score(x_test, y_test)

accuracy
'''example_measure = np.array([4,2,1,1,1,2,3,2,1])

example_measure = example_measure.reshape(1, -1)'''

# a deprication or value error will occur if you will not reshape



example_measure = np.array([[1,2,3,4,5,4,3,2,1], [10,11,43,77,1,23,11,21,11]])

# if there are multiple numbers of input(lists in list), then

example_measure = example_measure.reshape(len(example_measure), -1)



prediction = clf.predict(example_measure) 

prediction
from math import sqrt
plot1 = [1,3]

plot2 = [2,5]



euclideans_dictance = sqrt( (plot1[0] - plot2[0])**2 + (plot1[1] - plot2[1])**2 )

euclideans_dictance
import numpy as np

import matplotlib.pyplot as plt

import warnings   # to warn the user when they are using not a suitale number for k.(Basically, even numer)

from collections import Counter

from matplotlib import style



style.use('fivethirtyeight')
dataset = {'k': [[1,2], [2,3],[3,1]], 'r': [[6,5],[7,7],[8,6]]}   # coordinates as features

new_features = [3,8]



for i in dataset:    # Iterating through i in dataset

    for ii in dataset[i]:   # iterating through ii in each feature set

        plt.scatter(ii[0], ii[1], s = 100, color = i)

        

# OR

# [[plt.scatter(ii[0], ii[1], s = 100, color = i) for ii in dataset[i]] for i in dataset]





plt.scatter(new_features[0], new_features[1], s = 100)
def kNN(data, predict, k=3):

    if len(data)>=k:

        warnings.warn('not a great k value')

    distances = []

    for group in data:

        for features in data[group]:

            # euclideans_dictance = sqrt( (features[0] - predict[0])**2 + (features[1] - predict[1])**2 )  correct algo, ut not an efficient, fast way to make predictions. Also, this is only for 2D data

            # euclideans_distance = np.sqrt(np.sum((np.array(features) - np.array(predict))**2))

            euclideans_dictance2 = np.linalg.norm(np.array(features) - np.array(predict))

            distances.append([euclideans_dictance2, group])

                

    votes = [i[1] for i in sorted(distances)[:k]]   # i[1] in distances is euclid_dist and we are forming groups in k numbera

    vote_result = Counter(votes).most_common(1)[0][0]  # since most common comes in array of list, therefore defining [0][0]th element

    confidence = (Counter(votes).most_common(1)[0][1]/k)

    

    return vote_result, confidence
result = kNN(dataset, new_features, k=3)

result
import random
full_data = df.astype(float).values.tolist()  # sometimes data may come in quotes, which might not be treated as datapoints. So, everything in the dataframe ought to be int or float for better accuracy in prediction



print(full_data[:5])
random.shuffle(full_data)



print(full_data[:5])
test_size = 0.2

train_set = {2:[], 4:[]}

test_set = {2:[], 4:[]}

train_data = full_data[:-int(test_size*len(full_data))]  # first 80% of data

test_data = full_data[:-int(test_size*len(full_data)): ] # last 20% of data



for i in train_data:

    train_set[i[-1]].append(i[:-1])  # i[-1] means we are entering the last column of the data, which is either 2(enign) or 4(malignant) where we are appemding its features values till -1, i.e. last value

    

for i in test_data:

    test_set[i[-1]].append(i[:-1])
correct = 0

total = 0



for group in test_set:

    for data in test_set[group]:

        vote, conf = kNN(train_set, data, k=5)

        if group == vote:

            correct += 1

        else:

            print(conf)

        total += 1

        

print('accuracy:',  correct/total, 'Confidence:', conf)
correct = 0

total = 0



for group in test_set:

    for data in test_set[group]:

        vote, conf = kNN(train_set, data, k=200)

        if group == vote:

            correct += 1

        else:

            print(conf)    # These are all the cases in which we are not confident if cancer is benign or malignant

        total += 1

        

print('accuracy:',  correct/total, 'confidence:', conf)