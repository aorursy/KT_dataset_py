# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        #print(os.path.join(dirname, filename))

        pass



# Any results you write to the current directory are saved as output.
def shuffle_data(X, y):

    permuted_index = np.random.permutation(len(X))

    X = X[permuted_index]

    y = y[permuted_index]

    return X, y



def create_Dataset(path):

    imdb_path = os.path.join(path, 'aclImdb')

    

    train_texts = []

    test_texts = []

    train_labels = []

    test_labels = []

    

    for folder in ['train', 'test']:

        for subfolder in ['pos', 'neg']:

            com_path = os.path.join(imdb_path, folder, subfolder)

            for filename in sorted(os.listdir(com_path)):

                if filename.endswith('.txt'):

                    with open(os.path.join(com_path, filename), encoding='utf8') as f:

                        if folder == 'train': train_texts.append(f.read())

                        else: test_texts.append(f.read())

                        if subfolder == 'pos': label = 1

                        else: label = 0

                        if folder == 'train': train_labels.append(label)

                        else: test_labels.append(label)

    print('a')

    train_texts = np.array(train_texts)

    train_labels = np.array(train_labels)

    test_texts = np.array(test_texts)

    test_labels = np.array(test_labels)

    

    train_texts, train_labels = shuffle_data(train_texts, train_labels)

    test_texts, test_labels = shuffle_data(test_texts, test_labels)

    

    output = np.column_stack((train_texts.flatten(), train_labels.flatten()))

    #np.savetxt('train_output.csv',output,delimiter=',')

    train_data = pd.DataFrame(output)

    train_data.to_csv('train_data.csv', index=False)

    

    output = np.column_stack((test_texts.flatten(), test_labels.flatten()))

    test_data = pd.DataFrame(output)

    test_data.to_csv('test_data.csv', index=False)

    return train_texts, train_labels, test_texts, test_labels



path = './../input/imdb-movie-reviews-dataset/aclimdb/'

train_X, train_y, test_X, test_y = create_Dataset(path)