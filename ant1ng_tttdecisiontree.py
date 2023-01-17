import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn import tree, preprocessing # DT library

from sklearn.metrics import accuracy_score

import pickle # DT serialization
!ls -l '/kaggle/input/tttsolved'
import glob, os

os.chdir('/kaggle/input/tttsolved/')

sizes = {}

for file in glob.glob("*RemotenessOptimized.csv"):

    sizes[file] = os.path.getsize(file) / (10 ** 6)

sizes = pd.DataFrame.from_dict(sizes, orient='index')

sizes.rename(columns={0 : 'Size (in mb)'}, inplace=True)

sizes = sizes.sort_index()

sizes
df3 = pd.read_csv('/kaggle/input/tttsolved/TTT3AndRemotenessOptimized.csv')

df4 = pd.read_csv('/kaggle/input/tttsolved/TTT4AndRemotenessOptimized.csv')
!ls -l
def split_key(df):

    le = preprocessing.LabelEncoder()

    le.fit(['X', 'O', ' '])

    key_len = len(df['key'].values[0]) 

    for i in range(key_len):

        df[i] = [key[i] if len(key) > i else ' ' for key in df['key']]

        df[i] = le.transform(df[i])
def key_len(df):

    def len_key(key):

        return len(key) - key.count(' ')

    def white_key(key):

        return key.count(' ')

    df['len'] = df['key'].apply(len_key)

    df['white'] = df['key'].apply(white_key)
import math

def categorize_groups(df):

    length = len(df['key'].values[0])

    side_length = math.sqrt(length)

    groups = [[i, i + 1, i + side_length, i + 1 + side_length] for i in range(int(length - 1 - side_length))]

    for i, group in zip(range(len(groups)), groups):

        df['group'+str(i)] = sum([df[entry] for entry in group])
import math

def primitive(key):

    side = int(math.sqrt(len(key)))

    board = [list(key[i:i+side]) for i in range(0, len(key), side)]

    if len(board[-1]) < side: board[-1] += [' ' for _ in range(side - len(board[-1]))]

    try:

        # Horizontals

        for row in board:

            if len(set(row)) == 1 and row[0] != ' ': return 2



        # Verticals

        for col_num in range(len(board[0])):

            col = [row[col_num] for row in board]

            if len(set(col)) == 1 and col[0] != ' ': return 2



    # Diagonals

    

        diag1 = [board[i][i] for i in range(len(board[0]))]

        diag2 = [board[i][len(board[0])-1-i] for i in range(len(board[0]))]

        if len(set(diag1)) == 1 and diag1[0] != ' ': return 2

        if len(set(diag2)) == 1 and diag2[0] != ' ': return 2

    except:

        print(key)

        print(board)

        

    if ' ' not in key:

        return 1

    return 0    
def value_encoding(df):

    ve = preprocessing.LabelEncoder()

    ve.fit(['Lose', 'Win', 'Tie'])

    df['value'] = ve.transform(df['value'])
def feature_engineering(dfs):

    for df in dfs:

        split_key(df)

        categorize_groups(df)

        df['primitive'] = df['key'].apply(primitive)

        key_len(df)

        value_encoding(df)

feature_engineering([df3, df4])
def classify(df, drop, value):

    y = df[value].values

    X = df.drop(drop, axis=1).values

    clf = tree.DecisionTreeClassifier(criterion='entropy')

    return clf.fit(X, y)



def find_size(name='temp'):

    os.chdir('/kaggle/working')

    return os.path.getsize(name) / (10 ** 6)



def write_classifier(clf, name='temp'):

    os.chdir('/kaggle/working')

    pickle_out = open(name, "wb")

    pickle.dump(clf, pickle_out)

    pickle_out.close()
value_classifier = []

write_classifier(classify(df3, ['remoteness', 'len', 'white', 'key', 'value'], 'value'))

value_classifier.append(find_size())

write_classifier(classify(df4, ['remoteness', 'len', 'white', 'key', 'value'], 'value'))

value_classifier.append(find_size())

sizes['Value classifier'] = value_classifier

sizes
remote_classifier = []

write_classifier(classify(df3, ['remoteness', 'key', 'value'], 'remoteness'))

remote_classifier.append(find_size())

write_classifier(classify(df4, ['remoteness', 'key', 'value'], 'remoteness'))

remote_classifier.append(find_size())

sizes['Remoteness classifier'] = remote_classifier

sizes
remoteness_value_classifier = []

write_classifier(classify(df3, ['remoteness', 'key'], 'remoteness'))

remoteness_value_classifier.append(find_size())

write_classifier(classify(df4, ['remoteness', 'key'], 'remoteness'))

remoteness_value_classifier.append(find_size())

sizes['Remoteness classifier (with value feature)'] = remoteness_value_classifier

sizes
value_remoteness_classifier = []

write_classifier(classify(df3, ['len', 'white', 'key', 'value'], 'value'))

value_remoteness_classifier.append(find_size())

write_classifier(classify(df4, ['len', 'white', 'key', 'value'], 'value'))

value_remoteness_classifier.append(find_size())

sizes['Value classifier (with remoteness feature)'] = value_remoteness_classifier

sizes
sizes['Value then remote'] = sizes['Value classifier'] + sizes['Remoteness classifier (with value feature)']

sizes['Remote then value'] = sizes['Remoteness classifier'] + sizes['Value classifier (with remoteness feature)']
sizes
# Idea of using multiclass-multilabel classifier instead of two separate classifiers

clf = classify(df4, ['remoteness', 'len', 'white', 'key', 'value'], ['value', 'remoteness'])

write_classifier(clf)

find_size()