!python --version
import tensorflow as tf # Машинное обучение
import keras # API для TensorFlow
import numpy as np # Научные вычисления
import pandas as pd # Обработка и анализ данных
pkgs = {
    'Package': ['TensorFlow', 'Keras', 'NumPy', 'Pandas'],
    'Version': [i.__version__ for i in [tf, keras, np, pd]]}

df_pkgs = pd.DataFrame(data = pkgs) # Версии используемых библиотек
df_pkgs.head(None).style.hide_index() # Отображение первых N строк или все если указать None
# Набор данных
xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype = float) # X
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype = float) # Y
# Модель с последовательными слоями в нейронной сети
model = keras.Sequential([
    # Полносвязный слой
    keras.layers.Dense(
        units = 1, # Количество нейронов
        input_shape = [1] # Входные данные
    )
])
# Сборка модели
model.compile(
    optimizer = 'sgd', # Оптимизатор
    loss = 'mean_squared_error' # Функция потерь
)
# Обучение модели
history = model.fit(
    xs, # X
    ys, # Y
    epochs = 500, # Количество эпох
    verbose = False # Отключение вывода
)
print(model.predict([10.0])) # Предсказание модели