import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from numpy.linalg import norm
from scipy.spatial import distance
import sklearn

santa = pd.read_csv("../input/cities.csv")
CityID = santa.iloc[:, 0]
XY = santa.iloc[:, 1:]
path = santa.iloc[0]
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
santa.head()
santa.shape
def is_prime(num):
    if num > 1:
        for i in np.arange(2, np.sqrt(num+1)) :
            if num % i == 0:
                return 0
        
        return 1
    
    return 0
prime_cities = santa['CityId'].apply(is_prime)
santa['Prime'] = prime_cities
santa.head()
def total_distance(dfcity,path):
    prev_city = path[0]
    total_distance = 0
    step_num = 1
    for city_num in path[1:]:
        next_city = city_num
        total_distance = total_distance + \
            np.sqrt(pow((dfcity.X[city_num] - dfcity.X[prev_city]),2) + pow((dfcity.Y[city_num] - dfcity.Y[prev_city]),2)) * \
            (1+ 0.1*((step_num % 10 == 0)*int(not(prime_cities[prev_city]))))
        prev_city = next_city
        step_num = step_num + 1
    return total_distance

no_path = list(santa.CityId[:].append(pd.Series([0])))
print('Total distance with no path is '+ "{:,}".format(total_distance(santa,no_path)))
sorted_cities = list(santa.iloc[1:,].sort_values(['X','Y'])['CityId'])
sorted_cities = [0] + sorted_cities + [0]
print('Total distance with the sorted city path is '+ "{:,}".format(total_distance(santa,sorted_cities)))
submission = pd.DataFrame(sorted_cities)
submission.columns = ['Path']
submission.head()
submission.to_csv('sample_submission.csv', index=False)
y_pred = classifier.predict(X_test)  
from sklearn.metrics import classification_report, confusion_matrix  
print(confusion_matrix(y_test, y_pred))  
print(classification_report(y_test, y_pred))