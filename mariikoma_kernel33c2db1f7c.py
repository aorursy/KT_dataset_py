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

import matplotlib.pyplot as plt

import numpy as np

from sklearn import datasets, linear_model

from sklearn.metrics import mean_squared_error, r2_score
import csv

from datetime import datetime

regrConfirmados = linear_model.LinearRegression()

regrMortos = linear_model.LinearRegression()

with open ('/kaggle/input/covid19-global-forecasting-week-2/train.csv') as csvfile:

    #print (csvfile.read())

    data = csv.reader(csvfile, delimiter=',')

    data_list = list(data)

    numeroDias = []

    confirmados = []

    mortos = []

    for row in data_list[1:]:

        datafim = datetime.strptime(row[3], '%Y-%m-%d')

        datainicio = datetime.strptime('2020-01-22', '%Y-%m-%d')

        dias = abs((datafim - datainicio).days)

        numeroDias.append (dias)

        confirmados.append (float(row[4]))

        mortos.append (float(row[5]))

        #row.append (str (dias))

        #print (', '.join(row))

    res = np.reshape(numeroDias,(-1, 1))

    regrConfirmados.fit(res,confirmados)

    regrMortos.fit(res,mortos)

    

with open ('/kaggle/input/covid19-global-forecasting-week-2/test.csv') as testfile:

    data = csv.reader(testfile, delimiter=',')

    data_list = list(data)

    numeroDias = []

    ids = []

    for row in data_list[1:]:

        datafim = datetime.strptime(row[3], '%Y-%m-%d')

        datainicio = datetime.strptime('2020-01-22', '%Y-%m-%d')

        dias = abs((datafim - datainicio).days)

        numeroDias.append (dias)

        ids.append(row[0])

        

    res = np.reshape(numeroDias,(-1, 1))

    predictConf = regrConfirmados.predict(res)

    predictMortos = regrMortos.predict(res)

    

    submission = ['ForecastId,ConfirmedCases,Fatalities\n']

    for index, id in enumerate(ids):

        strRow = ','.join([str(id), str(predictConf[index]), str(predictMortos[index])])

        submission.append(strRow + '\n')

        

    with open("submission.csv", "w", newline="") as f:

        f.writelines(submission)