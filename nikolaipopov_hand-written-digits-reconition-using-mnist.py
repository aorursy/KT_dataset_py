import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier

#read the data into DataFrame
#считываем данные в датафрейм
df = pd.read_csv('../input/train.csv')

#display the last five strings to make sure of correct result
#Выводим последние 5 строк массива
#print(df.tail())
print(np.size(df, 0))
y = df.iloc[0:np.size(df, 0)-1 , 0]
X = df.iloc[0:np.size(df, 0)-1, 1:785]
print(X.tail())

#splitting the training dataset, displaying DataFrame sizes
#Разобьем тренировочный массив, выведем для проверки размеры датафреймов
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
print(np.size(X_train, 0))
print(np.size(X_test, 0))
print(np.size(y_train, 0))
print(y_train==1)

#Convert data sets into arrays
#Преобразуем в массивы
X_train = X_train.as_matrix()
X_test = X_test.as_matrix()
y_test = y_test.as_matrix()
y_train = y_train.as_matrix()

#Displaying training data as pictures
#Выводим тренировочные данные в графическом виде
fig, ax = plt.subplots(nrows = 2, ncols = 5, sharex = True, sharey = True)
ax = ax.flatten()
for i in range(10):
    img = X_train[y_train == i][0].reshape(28, 28)
    ax[i].imshow(img, cmap ='Greys', interpolation = 'nearest')
ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
plt.show()

#initializing and fitting perceptron
#Инициализируем и обучаем нейросеть
clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(50,), random_state=1)
clf.fit(X_train, y_train)
#Применяем настроенную нейросеть для распознавания образов (тестовая выборка)
#applying perceptron on test data set
y_predict = clf.predict(X_test)
res = y_predict == y_test
df2 = pd.DataFrame(y_predict)
#df2.to_clipboard()

#Calculating accuracy coeff
#Рассчитываем коэффициент точности прогноза, который получился равным 0.88
acc_1 = np.count_nonzero(res, 0)
acc_2 = np.size(res,0)
acc = acc_1/acc_2
print('Number of true predictions: ', acc_1)
print('overall size of test dataset: ',acc_2)
print('accuracy: ',acc)
