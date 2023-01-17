from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.metrics import plot_confusion_matrix
import numpy as np
import matplotlib.pyplot as plt

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data_train = pd.read_csv('../input/main-tenis/play_tennis_train.csv')
data_test = pd.read_csv('../input/main-tenis/play_tennis_test.csv')

data_train.head()
le = preprocessing.LabelEncoder() 

data_train_df = pd.DataFrame(data_train)
data_train_df_encoded = data_train_df.apply(le.fit_transform)

print(data_train_df_encoded.head())

data_test_df = pd.DataFrame(data_test)
data_test_df_encoded = data_test_df.apply(le.fit_transform)
x_train = data_train_df_encoded.drop(['play'],axis=1)
y_train = data_train_df_encoded['play']

x_test = data_test_df_encoded.drop(['play'],axis=1)
y_test = data_test_df_encoded['play']
model = GaussianNB()
nbtrain = model.fit(x_train, y_train)

y_pred = nbtrain.predict(x_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))