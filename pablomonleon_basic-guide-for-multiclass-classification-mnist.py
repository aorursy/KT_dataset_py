import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings("ignore")

import matplotlib.image as mpimg

import matplotlib



from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

import itertools
train = pd.read_csv("/home/pablo/Escritorio/Postgrado_UB/Problemas Kaggle/Plantillas/CLASIFICACIÓN MULTICLASS- MNIST/datos/train.csv")

test = pd.read_csv("/home/pablo/Escritorio/Postgrado_UB/Problemas Kaggle/Plantillas/CLASIFICACIÓN MULTICLASS- MNIST/datos/test.csv")
y_train=train['label']

x_train=train.drop('label',axis=1)
y_train.value_counts()

sns.countplot(y_train)



print('missing values: ', x_train.isnull().sum().values)
x_train=x_train/255

test = test / 255
some_digit = np.array(x_train.iloc[6])



some_digit_image = some_digit.reshape(28, 28)



plt.imshow(some_digit_image, cmap = matplotlib.cm.binary,

interpolation="nearest")

plt.axis("off")

plt.show()
from sklearn.preprocessing import LabelEncoder

enc = LabelEncoder()

label_encoded = enc.fit_transform(train['label'])
X_train, X_val, Y_train, Y_val = train_test_split(x_train, y_train, test_size = 0.1, random_state=1235)
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import confusion_matrix,accuracy_score,classification_report



rfc = RandomForestClassifier()



rfc.fit(X_train,Y_train)



predictions  = rfc.predict(X_val)

accuracy     = accuracy_score(Y_val,predictions)

print('accuracy: ',accuracy)
lgbm = LGBMClassifier(n_estimators=500, random_state=3)

lgbm = lgbm.fit(X_train, Y_train)
predictions_lgbm  = lgbm.predict(X_val)

accuracy_lgbm     = accuracy_score(Y_val,predictions_lgbm)

print('accuracy_lgbm: ',accuracy_lgbm)