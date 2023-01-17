from tensorflow.keras.datasets import cifar10

import matplotlib.pyplot as plt

%matplotlib inline 

(X_train, y_train), (X_test, y_test) = cifar10.load_data()

classes=['самолет', 'автомобиль', 'птица', 'кот', 'олень', 'собака', 'лягушка', 'лошадь', 'корабль', 'грузовик']