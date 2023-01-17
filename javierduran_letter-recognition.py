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
train_csv = pd.read_csv("/kaggle/input/digit-recognizer/train.csv")

train_csv.head()
y = train_csv['label']

X = train_csv.drop('label', axis = 1)

X.info()
import matplotlib.pyplot as plt



def displayImage(n):

    imageMatrix = np.asanyarray(X.iloc[n]).reshape(28,28)

    plt.imshow(imageMatrix, cmap = 'gray')

    plt.show()

    
displayImage(37458)
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 42, test_size = 0.12, stratify = y)
from sklearn.ensemble import RandomForestClassifier



clf = RandomForestClassifier(n_estimators = 330)

clf.fit(X_train, y_train)
test_csv = pd.read_csv("/kaggle/input/digit-recognizer/test.csv")

#X_test = test_csv
from sklearn.metrics import accuracy_score

y_pred = clf.predict(X_test)

print(accuracy_score(y_test, y_pred))
#from sklearn.linear_model import SGDClassifier

#sgc = SGDClassifier()

#sgc.fit(X_train, y_train)
#sgc_pred = sgc.predict(X_test)

#print(accuracy_score(y_test, sgc_pred))
true_values_df = pd.DataFrame(y_test)

true_values_df['Predicted Values'] = y_pred

incorrect_values = true_values_df[true_values_df['label']!=true_values_df['Predicted Values']]

incorrect_values.head(55)
incorrect_values['label'].value_counts()
from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier(hidden_layer_sizes = (150,), beta_2 = 0.99, beta_1 = 0.86)

mlp.fit(X_train, y_train)
mlp_pred = mlp.predict(X_test)

print(accuracy_score(y_test, mlp_pred))
true_values_df = pd.DataFrame(y_test)

true_values_df['Predicted Values'] = mlp_pred

incorrect_values = true_values_df[true_values_df['label']!=true_values_df['Predicted Values']]

incorrect_values.head(55)
incorrect_values['label'].value_counts()
final_pred = mlp.predict(test_csv)

final_pred_df = pd.DataFrame(final_pred)

final_pred_df['ImageId'] = final_pred_df.index

final_pred_df.columns = ['Label', 'ImageId']

final_pred_df.head()
final_predictions = final_pred_df[['ImageId', 'Label']]

final_predictions.head()

final_predictions.to_csv('final_predictions_csv', index = False)