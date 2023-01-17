# Import the related library

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.utils import shuffle

from sklearn.svm import SVC

import pandas as pd

import numpy as np
# Load the data and transfer the data into index format by mapping

spice_2_id = dict()

id_2_spice = dict()



def load(file_name='../input/Iris.csv'):

    # Load data

    df = pd.read_csv(file_name)

    x = df.get_values().T[1:5].T

    y = df.get_values().T[-1].T



    # Build mapping and transfer

    counter = 0

    for name in y:

        if name not in spice_2_id:

            spice_2_id[name] = counter

            counter += 1

    for i in range(len(y)):

        y[i] = spice_2_id[y[i]]

    y = np.asarray(y, dtype=np.float)

    

    # Shuffle, split and return

    x, y = shuffle(x, y)

    return train_test_split(x, np.reshape(y, [-1, 1]), test_size=0.065)
# Train the model and show accuracy

if __name__ == '__main__':

    train_x, test_x, train_y, test_y = load()



    # SVM model

    clf = SVC()

    clf.fit(train_x, np.reshape(train_y, [-1]))

    y_ = clf.predict(test_x)

    print('<< SVM >>')

    print('tag    : ', np.reshape(test_y, [-1]))

    print('predict: ', y_)

    print('acc    : ', np.sum(np.equal(np.reshape(test_y, [-1]), y_)) / len(y_))



    # Random forest model

    clf = RandomForestClassifier()

    clf.fit(train_x, np.reshape(train_y, [-1]))

    y_ = clf.predict(test_x)

    print('<< RF >>')

    print('tag    : ', np.reshape(test_y, [-1]))

    print('predict: ', y_)

    print('acc    : ', np.sum(np.equal(np.reshape(test_y, [-1]), y_)) / len(y_))