# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import math
import random

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv('../input/mexico-road-accidents-during-2019/incidentes-viales-c5-2019.csv', encoding = "ISO-8859-1")
data['Date'] = data[u'fecha_creacion'].apply(lambda x: str(x)) + '-' + data[u'hora_creacion'].apply(lambda x: str(x)) #создает колонку vr-data(где-то сохраняется случай)
accidents = data.groupby(['Date']).size()#группирует
print(accidents.mean())#выводит среднее
df = accidents.to_frame() #переводит в дата-фрейм
N_test = 10000
elements = 200
# mean array of samples

means = [0] * N_test
# sample generation
for i in range (N_test):
     rows = np.random.choice(df.index.values , elements)
     sampled_df = df.loc[rows]
     means[i] = sampled_df.mean()
rows = np.random.choice(df.index. values , 200)
sampled_df = df.loc[rows]
est_sigma_mean = sampled_df.std()/ math.sqrt(200)
print('Direct estimation of SE from one sample of 200 elements:', est_sigma_mean[0])
print('Estimation of the SE by simulating 10000 samples of 200 elements:', np.array(means).std())
def meanBootstrap(X, numberb):
    x = [0]* numberb
    for i in range (numberb):
        sample = [X[j]
            for j in np.random.randint(len(X), size=len(X))]
        x[i] = np.mean(sample)
    return x
m = meanBootstrap(accidents, 500)
print('Mean estimate:', np.mean(m))

m = accidents.mean()
se = accidents.std() / math.sqrt(len(accidents))
ci = [m - se * 1.96, m + se * 1.96]
print('Confidence interval:', ci)

m = meanBootstrap(accidents , 10000)
sample_mean = np.mean(m)
sample_se = np.std(m)
print('Mean estimate:', sample_mean)
print('SE of the estimate:', sample_se)
ci = [np.percentile(m, 2.5), np. percentile(m, 97.5)]
print('Confidence interval:', ci)
