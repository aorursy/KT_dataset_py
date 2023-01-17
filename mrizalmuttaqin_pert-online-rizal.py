# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.metrics import plot_confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
data_train = pd.read_csv("/kaggle/input/main-tenis/play_tennis_train.csv")
data_test = pd.read_csv("/kaggle/input/main-tenis/play_tennis_test.csv")
le = preprocessing.LabelEncoder() # instance LabelEncoder dg object le
# mengubah data teks dari file train & test menjadi data numeric
# encode data train
data_train_df = pd.DataFrame(data_train)
data_train_df_encoded = data_train_df.apply(le.fit_transform)

print(data_train_df_encoded.head())

#encode data test
data_test_df = pd.DataFrame(data_test)
data_test_df_encoded = data_test_df.apply(le.fit_transform)
# x_train adalah data fitur (outlock, temp, humid, wind) 
# semua kolom diambil kecuali kolom play (isinya output class) 
x_train = data_train_df_encoded.drop(['play'],axis=1)
y_train = data_train_df_encoded['play']
# x adalah data fitur, sedangkan y adalah data output class
# x bentuknya matriks multidimensi (2D karena > 1 kolom)
# skalar, vector, matriks, dan tensor
# skalar = 1 atau 2 atau 3 (haya satuan, tidak punya arah/dimensi)
# vector = [1,2,3,4,5] (1 dimensi)
# matriks = [[1,2],[2,3],[3,4]] (>= 2D)
# tensor = >= 3D
x_test = data_test_df_encoded.drop(['play'],axis=1)
y_test = data_test_df_encoded['play']
model = GaussianNB()
nbtrain = model.fit(x_train, y_train)

y_pred = nbtrain.predict(x_test)
print("Accuracy : ", metrics.accuracy_score(y_test, y_pred))
disp = plot_confusion_matrix(nbtrain, x_test, y_test,
                                 display_labels=['No','Yes'],
                                 cmap=plt.cm.Blues)
disp.ax_.set_title('Confusion Matrix')

print('Confusion Matrix')
print(disp.confusion_matrix)

plt.show()
# confusion_matrix(y_test, y_pred, labels=[0, 1])