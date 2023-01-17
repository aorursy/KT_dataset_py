import matplotlib.pyplot as plt
from sklearn.svm import SVC
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import decomposition
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics
df = pd.read_csv("../input/CSV_datasetsix_vowel_dataset_with_class.csv")
df.head()
pix=[]
for i in range(784):
    pix.append('pixel'+str(i))
features=pix
X = df.loc[:, features].values
y = df.loc[:,'class'].values

X_train, X_test, y_train, y_test = train_test_split(
 X, y, test_size = 0.25, random_state = 100)
y_train=y_train.ravel()
y_test=y_test.ravel()

svm_model = SVC(kernel = 'poly', C = 1, gamma=2).fit(X_train, y_train)
svm_predictions = svm_model.predict(X_test)

accuracy = svm_model.score(X_test, y_test)
print('Accuracy: ',accuracy*100)