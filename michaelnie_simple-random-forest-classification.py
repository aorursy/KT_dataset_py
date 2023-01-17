import numpy as np

import pandas as pd

import cv2

import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier

from IPython.display import display

%matplotlib inline

data = pd.read_csv('../input/train.csv')

display(data.head(n=3))
ex_row = data.iloc[[10]].values.ravel()

ex_img = np.array(ex_row[1:].reshape(28, 28), dtype="float32")

plt.imshow(ex_img, cmap = 'Greys')

plt.axis('off')

plt.show()
display(ex_row[0])
data_matrix = data.as_matrix()

X = data_matrix[:, 1:]

y = data_matrix[:, 0]
rfc = RandomForestClassifier(n_estimators=10).fit(X, y)
test_data = pd.read_csv('../input/test.csv').as_matrix()

display(test_data.shape)
predicted_numbers = rfc.predict(test_data)

display(predicted_numbers)
results_data = pd.DataFrame({'ImageId': range(1, len(predicted_numbers)+1), 'Label': predicted_numbers})

results_data.to_csv('results.csv', sep=',', index=False)
