# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
os.system('apt-get install p7zip')
!pip install pyunpack

!pip install patool
from pyunpack.cli import Archive



directory = '/kaggle/working/'

Archive('/kaggle/input/kkbox-music-recommendation-challenge/train.csv.7z').extractall(directory)

Archive('/kaggle/input/kkbox-music-recommendation-challenge/test.csv.7z').extractall(directory)
train = pd.read_csv('/kaggle/working/train.csv')

test = pd.read_csv('/kaggle/working/test.csv')
validation = train.sample(frac=0.1, random_state=2020)

train = train.drop(validation.index, axis=0)



train = train.reset_index(drop=True)

validation = validation.reset_index(drop=True)
from scipy.sparse import csr_matrix
def dataframe_to_matrix(df):

    users = df.msno.unique()

    songs = df.song_id.unique()



    user2idx = {user:idx for idx, user in enumerate(users)}

    song2idx = {song:idx for idx, song in enumerate(songs)}

    

    rows = np.array(df.msno.map(user2idx))

    cols = np.array(df.song_id.map(song2idx))

    data = np.array(df.target)

    

    matrix = csr_matrix((data, (rows, cols)))

    

    return user2idx, song2idx, matrix
user2idx, song2idx, matrix = dataframe_to_matrix(train)
from sklearn.decomposition import TruncatedSVD
def matrix_factorization(matrix, n_components=100, n_iter=5):

    svd = TruncatedSVD(n_components=n_components, n_iter=n_iter, random_state=2020)

    U = svd.fit_transform(matrix)

    S = np.diag(svd.singular_values_)

    Vt = svd.components_.T

    

    return svd, U, S, Vt
svd, U, S, Vt = matrix_factorization(matrix, n_components=200, n_iter=10)
def predict(user, item, S, threshold):

    user = user.reshape((1, -1))

    item = item.reshape((-1, 1))

    rating = np.dot(np.dot(user, S), item)

    return rating, int(rating >= threshold)
def calculate_accuracy(prediction, real):

    return np.mean(np.equal(prediction, real))
def item_to_idx(item, dictionary):

    return dictionary[item] if item in dictionary else -1
from tqdm import tqdm
scores = dict()



real = np.array(validation.target)

thresholds = np.arange(0, 50, 10)



for threshold in tqdm(thresholds):

    prediction = list()

    for user, song in zip(validation.msno, validation.song_id):

        user_idx, song_idx = item_to_idx(user, user2idx), item_to_idx(song, song2idx)

        if user_idx == -1 or song_idx == -1:

            p = 0

        else:

            _, p = predict(U[user_idx], Vt[song_idx], S, threshold)

        prediction.append(p)

        

    accuracy = calculate_accuracy(prediction, real)

    scores[threshold] = accuracy

scores
import matplotlib.pyplot as plt
plt.figure(figsize=(16, 9))

plt.plot(list(scores.keys()), list(scores.values()))

plt.title('Validation Accuracy by Threshold')

plt.xlabel('Threshold')

plt.ylabel('Validation Accuracy')

plt.show()
thresholds = np.arange(0, 50, 10)

metrics = [(100, 5, 

             np.array([0.5163769990050723, 

                       0.6042369825765647, 

                       0.6067568336898266, 

                       0.6048428854531801, 

                       0.6009512810711604])

           ),

           (100, 10, 

             np.array([0.5163024471969876, 

                       0.6042681587872183,

                       0.6069086482808353,

                       0.6049811451699917,

                       0.6010759859137748])

           ),

           (200, 5, 

             np.array([0.5181066009526366,

                       0.6068937379192184,

                       0.6096548657931906,

                       0.6073640920538617,

                       0.6031593700778863])

           ),

           (200, 10, 

             np.array([0.5180930460784393,

                       0.6067500562527279,

                       0.6097795706358049,

                       0.6075877474781156,

                       0.6033626931908445])

           ),

          ]



plt.figure(figsize=(16, 9))

for c, i, score in metrics:

    plt.plot(thresholds, score, label='n_component={}, n_iter={}'.format(c, i))

plt.legend()

plt.xlabel('Threshold')

plt.ylabel('Accuracy')

plt.title('Accuracy by Threshold. (Hyperparameter Adjusting)')

plt.savefig('/kaggle/working/Hyperparameter_Comparison.png')

plt.show()
n_components = 200

n_iter = 10

threshold = 20
train = pd.concat([train, validation], axis=0, ignore_index=True, sort=False).reset_index(drop=True)
user2idx, song2idx, matrix = dataframe_to_matrix(train)
svd, U, S, Vt = matrix_factorization(matrix, n_components=n_components, n_iter=n_iter)
real = np.array(validation.target)

prediction = list()

for user, song in tqdm(zip(test.msno, test.song_id)):

    user_idx, song_idx = item_to_idx(user, user2idx), item_to_idx(song, song2idx)

    if user_idx == -1 or song_idx == -1:

        p = 0

    else:

        _, p = predict(U[user_idx], Vt[song_idx], S, threshold)

    prediction.append(p)
submission = pd.DataFrame(columns=['id', 'target'])

submission['id'] = test.id

submission['target'] = prediction

submission.head()

submission.to_csv('/kaggle/working/submission.csv', index=False)