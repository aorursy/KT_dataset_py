import numpy as np

import scipy

import matplotlib.pyplot as plt

#seed(1)

anomalies = []



# multiply and add by random numbers to get some real values

data = np.random.randn(50000)  * 20 + 20



# Function to Detection Outlier on one-dimentional datasets.

def find_anomalies(data):

    # Set upper and lower limit to 3 standard deviation

    random_data_std = scipy.std(data)

    random_data_mean = scipy.mean(data)

    anomaly_cut_off = random_data_std * 3

    

    lower_limit  = random_data_mean - anomaly_cut_off 

    upper_limit = random_data_mean + anomaly_cut_off

    print(lower_limit)

    print (upper_limit)

    # Generate outliers

    for outlier in data:

        if outlier > upper_limit or outlier < lower_limit:

            anomalies.append(outlier)

    return anomalies



find_anomalies(data)
import seaborn as sns

import matplotlib.pyplot as plt



sns.boxplot(data=data)
from sklearn.cluster import DBSCAN

import random 

random.seed(1)

random_data = np.random.randn(50000,2)  * 20 + 20



outlier_detection = DBSCAN(min_samples = 2, eps = 3)

clusters = outlier_detection.fit_predict(random_data)

list(clusters).count(-1) 



#Sklearn labels noisy points as -1

from sklearn.ensemble import IsolationForest

import numpy as np

np.random.seed(1)

random_data = np.random.randn(50000,2)  * 20 + 20



clf = IsolationForest( behaviour = 'new', max_samples=100, random_state = 1, contamination= 'auto')

preds = clf.fit_predict(random_data)

preds