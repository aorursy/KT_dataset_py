import numpy as np

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn import linear_model, metrics

from sklearn.ensemble import RandomForestClassifier

import matplotlib.pyplot as plt

%matplotlib inline



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        file_path = os.path.join(dirname, filename)

print(file_path)
heart = pd.read_csv(file_path) 

y = heart.iloc[:,13]

X = heart.iloc[:,:13]
train_data, test_data, train_labels, test_labels = train_test_split(X, y, test_size = 0.3)
RF = RandomForestClassifier(n_estimators=200, max_depth=25, random_state=1)

RF.fit(train_data, train_labels)
predictions = RF.predict(test_data)
metrics.accuracy_score(test_labels, predictions)
round (metrics.mean_squared_error(test_labels, predictions), 3)
round(metrics.mean_absolute_error(test_labels, predictions), 3)
# Get numerical feature importances

importances = list(RF.feature_importances_)
feature_list = list(heart.columns)

# List of tuples with variable and importance

feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]

# Sort the feature importances by most important first

feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)

[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances];
# Set the style

plt.style.use('fivethirtyeight')

# list of x locations for plotting

x_values = list(range(len(importances)))

# Make a bar chart

plt.bar(x_values, importances, orientation = 'vertical')

# Tick labels for x axis

plt.xticks(x_values, feature_list, rotation='vertical')

# Axis labels and title

plt.ylabel('Importance'); plt.xlabel('Variable'); plt.title('Variable Importances');