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

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import seaborn as sns

import matplotlib.pyplot as plt

from scipy import stats

from tqdm import tqdm



from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split

#Dependencies

import keras

from keras.models import Sequential

from keras.layers import Dense



import matplotlib.pyplot as plt
def position_parser(position_string):

    

    piece_map = {'K':[1,0,0,0,0,0,0,0,0,0,0,0],

                 'Q':[0,1,0,0,0,0,0,0,0,0,0,0],

                 'R':[0,0,1,0,0,0,0,0,0,0,0,0],

                 'B':[0,0,0,1,0,0,0,0,0,0,0,0],

                 'N':[0,0,0,0,1,0,0,0,0,0,0,0],

                 'P':[0,0,0,0,0,1,0,0,0,0,0,0],

                 'k':[0,0,0,0,0,0,1,0,0,0,0,0],

                 'q':[0,0,0,0,0,0,0,1,0,0,0,0],

                 'r':[0,0,0,0,0,0,0,0,1,0,0,0],

                 'b':[0,0,0,0,0,0,0,0,0,1,0,0],

                 'n':[0,0,0,0,0,0,0,0,0,0,1,0],

                 'p':[0,0,0,0,0,0,0,0,0,0,0,1]}

    

    position_array = []

    

    ps = position_string.replace('/','')

    

    

    for char in ps:

        position_array += 12 * int(char) * [0] if char.isdigit() else piece_map[char]

    

    #print("position_parser =>  position_array: {}".format(asizeof.asizeof(position_array)))

    

    return position_array
def fen_to_binary_vector(fen):

    

    #counter += 1

    #clear_output(wait=True)

    #print(str(counter)+"\n")

    

    fen_infos = fen.split()

    

    pieces_ = 0

    turn_ = 1

    castling_rights_ = 2

    en_passant_ = 3

    half_moves_ = 4

    moves_ = 5

    

    binary_vector = []

    

    binary_vector += ( [1 if fen_infos[turn_] == 'w' else 0]

                        + [1 if 'K' in fen_infos[castling_rights_] else 0]

                        + [1 if 'Q' in fen_infos[castling_rights_] else 0]

                        + [1 if 'k' in fen_infos[castling_rights_] else 0]

                        + [1 if 'q' in fen_infos[castling_rights_] else 0]

                        + position_parser(fen_infos[pieces_])

                     )

    

    #print("fen_to_binary_vector =>  binary_vector: {}".format(asizeof.asizeof(binary_vector)))

    #clear_output(wait=True)

    

    return binary_vector
fen_to_eval = pd.read_csv("/kaggle/input/fen-to-stockfish-evaluation/fen_to_stockfish_evaluation.csv",names=["fen",'eval'])
ax = sns.distplot(fen_to_eval["eval"])

ax.set(xlim=(-50, 50))
ax = sns.distplot(fen_to_eval["eval"])

ax.set(xlim=(-20, 20))
ax = sns.distplot(fen_to_eval["eval"])

ax.set(xlim=(-5, 5))
bound_eval = 4
bound_sample = fen_to_eval[fen_to_eval["eval"].abs() < bound_eval]
len(bound_sample)
bound_sample = bound_sample.sample(frac = 1/6, random_state=1)
data = pd.concat([fen_to_eval[fen_to_eval["eval"].abs() > bound_eval],bound_sample])
ax = sns.distplot(data["eval"])

ax.set(xlim=(-20, 20))
len(data)
del fen_to_eval, bound_sample
sample = data.sample(1000000,random_state=1)
sample.reset_index(drop=True,inplace=True)
y = np.array(sample["eval"])
scaler = MinMaxScaler()

scaler.fit(y.reshape(-1, 1))

y = scaler.transform(y.reshape(-1, 1))
X = np.zeros((len(sample), 773))
ax = sns.distplot(y)

ax.set(xlim=(0, 1))
for index, row in tqdm(sample.iterrows()):

    X[index,:] = np.array(fen_to_binary_vector(row["fen"]))
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.1)
del X

del y
model = Sequential()

model.add(Dense(800, input_dim=773, activation='relu'))

model.add(Dense(500, activation='relu'))

model.add(Dense(500, activation='relu'))

model.add(Dense(1, activation='sigmoid'))
model.compile(loss='mean_squared_error', optimizer='adam')
history = model.fit(X_train, y_train, epochs=50, batch_size=64)


plt.plot(history.history['loss'])

plt.title('Model loss over epochs')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.show()
model_json = model.to_json()

with open("model.json", "w") as json_file:

    json_file.write(model_json)