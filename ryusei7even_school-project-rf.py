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
import pandas as pd

from sklearn.decomposition import PCA

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder



def generate_data(location_str,year):

    '''

    Read data from any year & any location

    :param location_str: name of location (string)

    :param year: year (int)

    :return: DataFrame which has been merged

    '''



    df = pd.DataFrame()

    for i in range(1,13):

        if year == 2014:

            if i<5: continue

        if i>=10:

            path = '../input/schoolweather/' + location_str + '-station_realtime-' + str(year) + '-' + str(i) + '.csv'

            data = pd.read_csv(path, header=None, low_memory=False).drop([0])



            df=df.append(data)

            continue

        path='../input/schoolweather/'+location_str+'-station_realtime-'+str(year)+'-0'+str(i)+'.csv'

        data = pd.read_csv(path, header=None, low_memory=False).drop([0])

        # print('path'+str(i))



        df=df.append(data)



    return df
data = generate_data('', 2015)
def preprocessing_data(data):

    '''

    preprocessing

    :param data: data from csv file

    :return: X_train, X_test, y_train, y_test

    '''



    x = data[list(range(23))]

    y = data[list(range(23))]



    x = x.iloc[1:, 8:23]

    ori_y=y.iloc[2:, 5]

    # print(ori_y)



    # LabelEncoder

    fit_y = LabelEncoder().fit_transform(ori_y.astype(str))



    # for i,j in zip(ori_y, fit_y):

    #     print(i,' ==> ',j)



#     print(type(fit_y))

    x = x.iloc[1:, 8:23]

    transfer = PCA(n_components=0.82)

    transfer.fit_transform(x)

    # transfer.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(x, fit_y, test_size=0.2, random_state=666)



    return X_train, X_test, y_train, y_test
X_train, X_test, y_train, y_test = preprocessing_data(data)
from sklearn.linear_model import SGDClassifier

import numpy as np

import pandas as pd

import plotly.offline as py

import plotly.graph_objs as go
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(n_estimators=8)

model.fit(X_train, y_train)
score = model.score(X_test, y_test)

print('Accuracy:\n' + str(score))



depth = np.arange(1, 11, 1)

acc_list = []
for dp in depth:

    model = RandomForestRegressor(max_depth=dp)

    model.fit(X_train, y_train)

    acc = model.score(X_test, y_test)

    acc_list.append(acc)

    print(dp, 'Accuracy: %.3f%%' % (100 * acc))
def data_visualization(depth, acc_list):

    '''

    data_visualization using plotly

    :param depth: depth of the random forest

    :param acc_list: accuracy list

    :return: graph

    '''



    trace = go.Scatter(

        x=depth,

        y=acc_list,

#         mode='markers',

        line_color='rgb(0,176,246)',

        marker = dict(

        size = 6,

        color = np.random.randn(500),

        colorscale = 'Viridis',

        showscale = True

    ))

    graph = py.iplot([trace])

    return graph
data_visualization(depth, acc_list)
import plotly.express as px

def data_visualization_bar(depth, acc_list):

    '''

    data_visualization using plotly

    :param depth: depth of the random forest

    :param acc_list: accuracy list

    :return: fig

    '''



    temp = {

        'Accuracy':acc_list,

       'Depth':np.arange(1,11)

    }

    data = pd.DataFrame(temp)

    fig = px.bar(data, x='Depth', y='Accuracy', color='Accuracy', height=500)

    return fig
data_visualization_bar(depth, acc_list)