import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

d=pd.read_csv('/kaggle/input/customer-segmentation-tutorial-in-python/Mall_Customers.csv')
d.info()
d.head()
#describing the data

d.describe()
!pip install pycaret
# Importing anomaly detection module.

from pycaret.anomaly import *
# Initializing the setup function used for pre-processing.

setup_anomaly_data = setup(data=d)
# Instantiating Isolation Forest model.

iforest = create_model('iforest')
# Plotting the data using Isolation Forest model.

plot_model(iforest)
# Generating the predictions using Isolation Forest trained model.

iforest_predictions = predict_model(iforest, data = d)

print(iforest_predictions)
# Checking anomaly rows. Label = 1 is the anomaly data.

iforest_anomaly_rows = iforest_predictions[iforest_predictions['Label'] == 1]

print(iforest_anomaly_rows.head())
# Checking the number of anomaly rows returned by Isolaton Forest.

print(iforest_anomaly_rows.shape) 
print(iforest_anomaly_rows.head())
# Instantiating KNN model.

knn = create_model('knn')
# Plotting the data using KNN model.

plot_model(knn)
# Generating the predictions using KNN trained model.

knn_predictions = predict_model(knn, data = d)

print(knn_predictions)
knn_anomaly_rows = knn_predictions[knn_predictions['Label'] == 1]
# Checking the number of anomaly rows returned by KNN model.

knn_anomaly_rows.head()
knn_anomaly_rows.shape 
# Instantiating Cluster model.

cluster = create_model('cluster')



# Plotting the data using Cluster model.

plot_model(cluster)
# Generating the predictions using Cluster trained model.

cluster_predictions = predict_model(cluster, data = d)

print(cluster_predictions)

# Checking cluster anomaly rows. Predictions with Label = 1 are anomalies.

cluster_anomaly_rows = cluster_predictions[cluster_predictions['Label'] == 1]
# Checking the number of anomaly rows returned by Cluster model

print(cluster_anomaly_rows.head())

cluster_anomaly_rows.shape