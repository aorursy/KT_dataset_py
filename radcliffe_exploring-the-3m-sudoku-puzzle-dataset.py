import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
filename = '/kaggle/input/3-million-sudoku-puzzles-with-ratings/sudoku-3m.csv'

df = pd.read_csv(filename)

df.head()
def view_grid(puzzle_string):

    return pd.DataFrame(np.array(list(puzzle_string.replace('.', ' '))).reshape((9, 9)))



view_grid(df.puzzle[0])
view_grid(df.solution[0])
df.groupby('clues').size().reset_index(name='count')
df.difficulty.hist()
pd.DataFrame(data = {'Value': [

    np.min(df.difficulty),

    np.max(df.difficulty),

    np.mean(df.difficulty),

    100 * np.mean(df.difficulty == 0),

    100 * np.mean(df.difficulty < 4)

]}, index = [

    'Min',

    'Max',

    'Mean',

    'Percent = 0',

    'Percent < 4'

])
from random import shuffle

import keras

import sklearn





class DataGenerator(keras.utils.Sequence):



    def __init__(self, list_IDs, batch_size=32, nrows=None, shuffle=True, transform=True):

        self.batch_size = batch_size

        self.list_IDs = list_IDs

        self.shuffle = shuffle

        self.transform = transform

        self.load_data(nrows=nrows)

        self.on_epoch_end()

    

    def __len__(self):

        'Denotes the number of batches per epoch'

        return int(np.floor(len(self.list_IDs) / self.batch_size))



    def __getitem__(self, index):

        'Generate one batch of data'

        # Generate indexes of the batch

        indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]



        # Find list of IDs

        list_IDs_temp = [self.list_IDs[k] for k in indexes]



        # Generate data

        X, y, clues, difficulty = self.data_generation(list_IDs_temp)



        return X, y, clues, difficulty

    

    def load_data(self, nrows=None):

        df = pd.read_csv(filename, nrows=nrows)

        string_to_array = lambda s: np.array(list(map(int, s.replace('.', '0')))).reshape((9, 9))

        self.puzzles = np.stack(df['puzzle'].apply(string_to_array))

        self.solutions = np.stack(df['solution'].apply(string_to_array))

        self.clues = np.array(df['clues'])

        self.difficulty = np.array(df['difficulty'])



    def on_epoch_end(self):

        'Updates indexes after each epoch'

        self.indexes = np.arange(len(self.list_IDs))

        if self.shuffle == True:

            np.random.shuffle(self.indexes)

    

    def data_generation(self, list_IDs_temp):

        'Generates data containing batch_size samples'

        # Initialization

        puzzles = self.puzzles[list_IDs_temp]

        solutions = self.solutions[list_IDs_temp]

        clues = self.clues[list_IDs_temp]

        difficulty = self.difficulty[list_IDs_temp]

        if self.transform:

            self.transform_grids(puzzles, solutions)

        

        return puzzles, solutions, clues, difficulty

    

    def transform_grids(self, X, y):

    

        def axis_permutation():

            x = [0, 1, 2]

            y = [3, 4, 5]

            z = [6, 7, 8]

            p = [x, y, z]

            shuffle(x)

            shuffle(y)

            shuffle(z)

            shuffle(p)

            return p[0] + p[1] + p[2]

        

        def relabel_cells(X, y):

            p = list(range(1, 10))

            shuffle(p)

            X = sum(p[i - 1] * (X == i) for i in range(1, 10))

            y = sum(p[i - 1] * (y == i) for i in range(1, 10))

            return X, y

        

        p = axis_permutation()

        q = axis_permutation()

        X = X[:, p][:, :, q]

        y = y[:, p][:, :, q]

        if np.random.rand() > 0.5:

            X = X.transpose((0, 2, 1))

            y = y.transpose((0, 2, 1))

        

        X, y = relabel_cells(X, y)

        return X, y
from sklearn.model_selection import train_test_split

ids = list(range(1000))

train_ids, test_ids = train_test_split(ids, test_size=0.33)



training_generator = DataGenerator(train_ids, batch_size=32, nrows=1000)

for X, y, clues, difficulty in training_generator:

    print(*clues)