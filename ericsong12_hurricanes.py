# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import csv as csv

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
atlantic = pd.read_csv('../input/hurricane-database/atlantic.csv',header = 0)
pacific = pd.read_csv('../input/hurricane-database/pacific.csv',header = 0)

def data_clean(data):
    #data['Latitude'] = data['Latitude'].fillna(data['Latitude'].median())
    data = data.drop(['Name','Event'],axis=1)
    return data
atlantic_data = data_clean(atlantic)
pacific_data = data_clean(pacific)
X = pacific_data['Date']
y = pacific_data['Latitude']
#X_test = test_data.drop(["ID"],axis=1)

def pred_with_svm():
    kernel_svm = svm.SVC(gamma=.1)
    kernel_svm.fit(X,y)
    score_kernel_svm = kernel_svm.score(X,y)
    #print(score_kernel_svm）
#final.to_csv('hurricanes.csv',index=False)

# Any results you write to the current directory are saved as output.
