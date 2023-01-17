import numpy as np

import pandas as pd

from tqdm import tqdm

import matplotlib.pyplot as plt

%pylab inline
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
from sklearn.preprocessing import OneHotEncoder

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.metrics import mean_squared_error
fresh_data = pd.read_csv('/kaggle/input/pubg-finish-placement-prediction/train_V2.csv')

fresh_data.head() #буду считать, что от Id, groupId, matchId не зависит
fresh_data.shape #думаю это оч много информации для обучения! Для обучения возьму только 500000 максимум, иначе обучение долгое
data = fresh_data.head(500000)

del fresh_data
from scipy.stats import pearsonr

pearsonr(data.maxPlace, data.numGroups) 
plt.hist(data.winPlacePerc) #+- сбалансированная выборка
data = data.drop(['Id', 'groupId', 'matchId', 'maxPlace'], axis=1) #потому что мне они не нужны!
data.iloc[:3, 2:3] # вот как можно к отдельным столбцам. Сначала по index, затем по columns!!!
encoder = OneHotEncoder()

encoder.fit(np.array(data.matchType).reshape(-1, 1))
X = np.hstack((np.array(data.drop(['matchType', 'winPlacePerc'], axis=1)),

               encoder.transform(np.array(data.matchType).reshape(-1, 1)).toarray()))

y = np.array(data.winPlacePerc)
model = RandomForestRegressor(n_estimators=100, max_depth=10)

model.fit(X, y)
test_fresh_data = pd.read_csv('/kaggle/input/pubg-finish-placement-prediction/test_V2.csv')

test_fresh_data.head(10)
X_test = np.hstack((np.array(test_fresh_data.drop(['Id', 'groupId', 'matchId', 'maxPlace', 'matchType'], axis=1)),

                    encoder.transform(np.array(test_fresh_data.matchType).reshape(-1, 1)).toarray()))
answers = model.predict(X_test)

for i in range(len(answers)):

    answers[i] = max(answers[i], 0.0)

    answers[i] = min(answers[i], 1.0)

answ = pd.DataFrame()

answ['Id'] = test_fresh_data['Id']

answ['winPlacePerc'] = answers
answ.to_csv('submission.csv', index=False)