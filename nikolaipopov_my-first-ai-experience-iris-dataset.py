import pandas as pd

from mpl_toolkits.mplot3d import Axes3D

import matplotlib.pyplot as plt

import numpy as np

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

#Визуализируем набор данных в 3D. Правда, используем только три измерения тестовых данных,

#что может показаться нелогичным

#3D data visualization using only first three dimensions of X

fig = plt.figure()

ax = fig.add_subplot(111, projection='3d')

df = pd.read_csv('../input/Iris.csv')

#data preparation

#Подготовка данных

ix1 = df.loc[0:50,'SepalLengthCm']

iy1 = df.loc[0:50,'SepalWidthCm']

iz1 = df.loc[0:50,'PetalLengthCm']

ix2 = df.loc[51:100,'SepalLengthCm']

iy2 = df.loc[51:100,'SepalWidthCm']

iz2 = df.loc[51:100,'PetalLengthCm']

#plotting data

#Выводим в трехмерном пространстве

ax.scatter(ix1, iy1, iz1, c='r', marker='^')

ax.scatter(ix2, iy2, iz2, c='b', marker='o')



plt.show()

X = df.ix[0:150, 1:5]

y = df.ix[0:150, 5]



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

#It may be useful to display sizes of training and testing data sets after splitting

#Может быть полезным вывести размер тренировочного и тестового массивов

#после разделения первоначальной выборки



#print(np.size(X_train,0))

#print(np.size(X_test,0))

#print(np.size(y_train))

#print(np.size(y_test))

sc = StandardScaler()

sc.fit(X_train)

X_train_std = sc.transform(X_train)

X_test_std = sc.transform(X_test)



from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(C = 10, random_state = 1)

lr.fit(X_train_std, y_train)

y_predict = lr.predict(X_test_std)

res = y_test==y_predict

acc_1 = np.count_nonzero(res, 0)

acc_2 = np.size(res,0)

acc = acc_1/acc_2



#Calculating accuracy coeff

#Рассчитываем коэффициент точности прогноза, который получился равным 0.98 (49 из 50)

print('Number of true predictions: ', acc_1)

print('overall size of test dataset: ',acc_2)

print('accuracy: ',acc)

#print(y_predict)

print(res==True)