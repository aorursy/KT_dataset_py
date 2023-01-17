# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from datetime import datetime

from datetime import date

import plotly.graph_objects as go

from plotly.subplots import make_subplots



from sklearn.neural_network import MLPRegressor

from sklearn.model_selection import train_test_split

from sklearn import preprocessing



from sklearn.metrics import mean_squared_error



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv("../input/nba2k20-player-dataset/nba2k20-full.csv",parse_dates=True)

data
data['weight'] = [float(data['weight'][i].split()[3]) for i in range(len(data))]

data['height'] = [float(data['height'][i].split()[-1]) for i in range(len(data))]

data['salary'] = [int(data['salary'][i].split('$')[1]) for i in range(len(data))]



data['b_day'] = data['b_day'].apply(lambda x: datetime.strptime(x, '%m/%d/%y').date())

data['age'] = (datetime.today().date() - data['b_day']).astype('<m8[Y]').astype('int64')



data['draft_round'] = data['draft_round'].apply(lambda x: 0 if x=='Undrafted' else int(x)) 



data['team'] = data['team'].fillna('No team')
for column in ['weight', 'height']:

    upper_lim = data[column].quantile(.95)

    lower_lim = data[column].quantile(.05)

    data.loc[(data[column] > upper_lim),column] = upper_lim

    data.loc[(data[column] < lower_lim),column] = lower_lim

for column in ['age', 'rating']:

    upper_lim = data[column].quantile(.95)

    lower_lim = data[column].quantile(.05)

    data.loc[(data[column] > upper_lim),column] = int(upper_lim)

    data.loc[(data[column] < lower_lim),column] = int(lower_lim)
data['position'] = data['position'].apply(lambda x: 'F-C' if x=='C-F' else x)

data['position'] = data['position'].apply(lambda x: 'F-G' if x=='G-F' else x)



for column in ['team', 'country', 'position', 'draft_round']:

    encoded_columns = pd.get_dummies(data[column])

    data = data.join(encoded_columns).drop(column, axis=1)
data = data.drop(['college', 'full_name', 'b_day', 'jersey', 'draft_peak'], axis=1)
data
y, X = data['salary'], data.drop('salary', axis=1)

X = preprocessing.normalize(X)



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)



regr = MLPRegressor(random_state=0, 

                    hidden_layer_sizes = (64, 64, 64, 120),

                    alpha=0.001,

                    solver='lbfgs',

                    learning_rate='invscaling', learning_rate_init=1e-5,

                    max_iter=10000).fit(X_train, y_train)

y_predict = regr.predict(X_test)



print('Mean squared error: ', np.sqrt(mean_squared_error(y_test, y_predict)))

print('Score: ', regr.score(X_test, y_test))

import plotly.graph_objects as go



fig = go.Figure()

fig.add_trace(go.Scatter(x=list(range(len(y_predict))), y=y_predict,

                         mode='lines',

                         name='Prediction'))

fig.add_trace(go.Scatter(x=list(range(len(y_test))), y=y_test,

                         mode='lines',

                         name='True value'))



fig.show()
