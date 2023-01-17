import numpy as np 
import matplotlib.pyplot as plt
import os
import cv2
import random
import tensorflow as tf
from tensorflow import keras
DATADIR = "/kaggle/input/train-set/train" #шлях до данних
CATEGORIES = ['empty', 'tuc', 'redbull_yellow'] #розділити на категорії
whole_data = []
# функція, щоб створити загальний масив данних - з зображеннями та відповідними до них мітками (0-empty, 1-tuc, 2-redbull-yellow)
def create_whole_data():
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        class_num = CATEGORIES.index(category)

        for img in os.listdir(path):
            img_array = cv2.imread(os.path.join(path, img), cv2.COLOR_BGR2RGB)
            whole_data.append([img_array, class_num])

            
create_whole_data()
# перемішати данні
random.shuffle(whole_data)
#створити окремі масиви для міток та зображень
X = []
y = []
for features, label in whole_data:
    X.append(features)
    y.append(label)

X = np.array(X)
y = np.array(y)
print(X.shape) #1500 зображень, розміром 198*198 пікселей, по 3 числа, що відбовідають за колір
print(y) # масив з мітками від 0 до 2 включно
# к-сть зображень і міток
print(len(X)) 
print(len(y)) 
from sklearn.model_selection import train_test_split
# розділити данні на тренувальні і тестові, де для тесту 25 % данних
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1, stratify=y)
# для навчання 1125 зображень, для тесту 375
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
# одне з зображень для прикладу, значення пікселей від 0 до 255
plt.figure()
plt.imshow(X_train[0])
plt.colorbar()
plt.grid(False)
plt.show()
# маштабування , щоб значення піклей було від 0 до 1
X_train = X_train / 255.0 
X_test = X_test / 255.0
# перші 25 зображень та назва їх категорій
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(X_train[i])
    plt.xlabel(CATEGORIES[y_train[i]], color='r')
plt.show()
# налаштування шарів (layers)

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(192,192,3)), #перетворює формат зображення в одномірний масив
    keras.layers.Dense(128, activation='relu'), #використання передавальної функції
    keras.layers.Dense(3, activation='softmax') #розділяє оцінку належності до кожної категорії на 3, оцінки в сумі дають 1
])

model.compile(optimizer='adam', #оптимізатор, показує яким чином оновлюється модель
              loss='sparse_categorical_crossentropy', #функція втрат - вимірює точність моделі під час навчання
              metrics=['accuracy']) # використання метрики "точності" - кількість правильно класифікованих зображень
model.fit(X_train, y_train, epochs=10) #тренування моделі на тренувальних данних
test_loss, test_acc = model.evaluate(X_test,  y_test, verbose=2)
#точність моделі на тестовому наборі данних
print('Accuracy on testing data: ', test_acc)
predictions = model.predict(X_test)
#передбачення для тестового набору
# для прикладу виведено передбачення належності першого зображення до трьох катрегорій,
# вибрано індекс з найбільшим числом передбачення - тобто ту категорію, яку передбачила модель
# та виведено реальну категорію для даного зображення
print('Prediction for the first element: ', predictions[0])
print('Predicted label for the first element: ', np.argmax(predictions[0]))
print('Actual label for the first element: ', y_test[1])
# функція для виводу зображень з передбаченою категорією, передбачуваним відсотком належності 
# до цієї категорії а також реальною категорією в ()
def plot_image(i, predictions_array, true_label, img):
  predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(img)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'

  plt.xlabel("{} {:2.0f}% ({})".format(CATEGORIES[predicted_label],
                                100*np.max(predictions_array),
                                CATEGORIES[true_label]),
                                color=color)
# для прикладу виведено 3 зображення
for i in range(11, 14):
    plt.figure(figsize=(6,3))
    plot_image(i, predictions, y_test, X_test)
    plt.show()

