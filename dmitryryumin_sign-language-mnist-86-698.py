!python --version
import tensorflow as tf # Машинное обучение
import keras # API для TensorFlow
import numpy as np # Научные вычисления
import matplotlib as mpl # Визуализация графиков
import matplotlib.pyplot as plt # MATLAB-подобный способ построения графиков
import pandas as pd # Обработка и анализ данных
import seaborn as sns # Визуализация графиков (надстройка над matplotlib)
import os # Взаимодействие с файловой системой

# Визуализация в виде обычного изображения
%matplotlib inline
pkgs = {
    'Название пакета': ['TensorFlow', 'Keras', 'NumPy', 'Matplotlib', 'Pandas', 'Seaborn'],
    'Версия': [i.__version__ for i in [tf, keras, np, mpl, pd, sns]]}

df_pkgs = pd.DataFrame(data = pkgs) # Версии используемых библиотек
df_pkgs.head(None).style.hide_index() # Отображение первых N строк или все если указать None
# Максимальное количество отображаемых элементов
pd.set_option('display.max_columns', 15) # Столбцы
pd.set_option('display.max_rows', 10) # Строки
np.random.seed(1) # Генерация псевдослучайных чисел
# Загрузка обучающей выборки
df_train = pd.read_csv(
    os.path.join('../input/sign-language-mnist/sign_mnist_train', 'sign_mnist_train.csv')
)
df_train = df_train.iloc[np.random.permutation(len(df_train))] # Случайное перетасовывание обучающей выборки
# Загрузка валидационной выборки
df_test = pd.read_csv(
    os.path.join('../input/sign-language-mnist/sign_mnist_test', 'sign_mnist_test.csv')
)
# Обучающая выборка
for i in range(0,len(df_train.label.values)):
    if df_train.label[i] > 8:
        df_train.label[i] -= 1

# Валидационная выборка
for i in range(0,len(df_test.label.values)):
    if df_test.label[i] > 8:
        df_test.label[i] -= 1
df_train.head(5) # Отображение первых N строк или все если указать None
df_train.shape
train_size = df_train.shape[0] # Количество изображений жестов

val_percent = 10 # Размер валидационной выборки

# Количество изображений в выборках
val_size = int(df_train.shape[0] * (1 / val_percent)) # Валидационная
train_size -= val_size # Обучающая

# Изменение размера обучающей выборки
X_train = np.asarray(df_train.iloc[:train_size, 1:]).reshape([train_size, 28, 28, 1])
y_train = np.asarray(df_train.iloc[:train_size, 0]).reshape([train_size, 1])

# Изменение размера вылидационной выборки
X_val = np.asarray(df_train.iloc[train_size:, 1:]).reshape([val_size, 28, 28, 1])
y_val = np.asarray(df_train.iloc[train_size:, 0]).reshape([val_size, 1])
X_train.shape
X_val.shape
# Изменение размера тестовой выборки
X_test = np.asarray(df_test.iloc[:, 1:]).reshape([-1, 28, 28, 1])
y_test = np.asarray(df_test.iloc[:, 0]).reshape([-1, 1])
X_test.shape
# Выборки
X_train = X_train / 255 # Обучающая
X_val = X_val / 255 # Валидационная
X_test = X_test / 255 # Тестовая
def _countplot(df, sampling, pad):
    """
    График подсчета количества элементов в каждом классе

    (pandas.core.frame.DataFrame, str, int) -> None

    Аргументы:
        df - Данные
        sampling - Выборка (train, val)
        pad - Отступ от ряда до его значения

    Возвращает: None
    """

    # Создание новой фигуры
    plt.figure(
        figsize = (18, 11) # Размер фигуры в дюймах
    )

    ax = plt.axes() # Добавление осей к текущей фигуре
    ax.xaxis.tick_bottom() # Перемещение меток в нижнюю часть

    # Количество элементов в каждом классе
    ax = sns.countplot(df.label, label = 'Количество')

    # Метки классов
    if sampling == 'train':
        iloc = df.iloc[train_size:, :]
        title = 'обучающей'
    elif sampling == 'val':
        iloc = df.iloc[:train_size, :]
        title = 'тестовой'
    
    counts = iloc.groupby(df.label)['label'].count().index.tolist()

    i = 0

    for p in ax.patches:
        height = p.get_height()

        ax.text(
            p.get_x() + p.get_width() / 2.0, # X позиция размещения текста
            height + pad, # Y позиция размещения текста
            df.label.value_counts()[counts[i]], # Текст
            ha = 'center', # Выравнивание
            fontdict = {
                'fontsize': 14, # Размер заголовка
                'color': '#000000' # Цвет заголовка
            },
        )

        i += 1

    # Изменение внешнего вида меток
    ax.tick_params(
        axis = 'x', # Ось
        direction = 'out', # Расположение линий меток
        length = 10, # Длина линий меток
        width = 1, # Ширина линий меток 
        color = '#000000', # Цвет линий меток
        pad = 5, # Расстояние между линиями меток и метками
        labelsize = 14, # Размер метки
        labelcolor = '#000000', # Цвет метки
        bottom = True,  # Рисование линий меток
    )
    ax.tick_params(
        axis = 'y', # Ось
        direction = 'out', # Расположение линий меток
        length = 10, # Длина линий меток
        width = 1, # Ширина линий меток 
        color = '#000000', # Цвет линий меток
        pad = 5, # Расстояние между линиями меток и метками
        labelsize = 14, # Размер метки
        labelcolor = '#000000', # Цвет метки
        left = True  # Рисование линий меток
    )

    # Заголовок осей
    ax.set_title(
        label = 'Количество элементов в каждом классе ' + title + ' выборки', # Заголовок
        fontdict = {
            'fontsize': 18, # Размер заголовка
            'color': '#000000' # Цвет заголовка
        },
        pad = 20 # Отступ заголовка от вершины осей
    )

    # Изменение внешнего вида меток данных
    ax.set_xlabel(
        'Метки',
        fontsize = 14, # Размер метки
        fontdict = {
            'color': '#000000' # Цвет метки
        },
        labelpad = 10 # Отступ

    );
    ax.set_ylabel(
        'Количество',
        fontsize = 14, # Размер метки
        fontdict = {
            'color': '#000000' # Цвет метки
        },
        labelpad = 10 # Отступ
    );
    
    plt.show() # Отображение фигуры
_countplot(df_train, 'train', 15)
_countplot(df_test, 'val', 5)
rows = 4 # Количество строк
cols = 6 # Количество столбцов

pic_index = 0 # Счетчик изображений

fig = plt.gcf() # Создание фигуры
fig.set_size_inches(cols * 2.4, rows * 2.4) # Установка размера фигуры в дюймах

idx_show = [] # Метки которые уже были показаны

cnt = 0 # Счетчик

for i in range(0, len(X_train)):
    # Изображение с меткой не отображалась
    if y_train[i][0] not in idx_show:
        cnt += 1
        
        sp = plt.subplot(rows, cols, cnt)
        sp.axis('Off') # Отключение осей

        plt.imshow(X_train[i].reshape([28, 28]), cmap = 'gray') 

        label = y_train[i][0] if y_train[i][0] <= 8 else y_train[i][0] + 1 # Буквенная метка
        
        plt.title('{} ({})'.format(y_train[i][0], chr(label + 65)), y = -0.15, fontsize = 14, color = '#000000')
        
        idx_show.append(y_train[i][0]) # Добавление метки 
        
        if cnt == rows * cols:
            break

plt.suptitle(
    'Отображение изображений из обучающей выборки', # Заголовок
    fontsize = 20, # Размер заголовка
    fontdict = {
        'color': '#000000' # Цвет заголовка
    },
    y = 1.04 # Отступ заголовка от вершины осей
)

plt.tight_layout(pad = 0, w_pad = 0, h_pad = 1.0) # Установка расстояния между осями

plt.show() # Отображение фигуры
# Модель с последовательными слоями в нейронной сети
model = keras.Sequential([
    
    # Сверточный слой
    keras.layers.Conv2D(
        16, # Количество фильтров
        (3, 3), # Размер свертки
        activation = tf.nn.relu, # Функция активации
        input_shape = (28, 28, 1) # Размер входных данных
    ),
    keras.layers.MaxPooling2D(2, 2), # Уменьшение размерности
    
    # Сверточный слой
    keras.layers.Conv2D(
        32, # Количество фильтров
        (3, 3), # Размер свертки
        activation = tf.nn.relu # Функция активации
    ),
    keras.layers.MaxPooling2D(2,2), # Уменьшение размерности
    
    keras.layers.Flatten(), # Преобразование массива пикселей в вектор пикселей
    
    # Полносвязный скрытый слой
    keras.layers.Dense(
        units = 16, # Количество нейронов
        activation = tf.nn.relu # Функция активации
    ),
    # Полносвязный слой
    keras.layers.Dense(
        24, # Количество нейронов = количество классов
        activation = tf.nn.softmax # Функция активации
    )
])
model.compile(
    optimizer = keras.optimizers.Adam(learning_rate = 0.001), # Оптимизатор
    loss = 'sparse_categorical_crossentropy', # Функция потерь
    metrics = ['accuracy'] # Метрика оценивания
)

model.summary() # Визуализация модели
class ModelCallback(keras.callbacks.Callback):
    """
    Обратный вызов
    """
    
    def on_epoch_end(self, epoch, logs = {}):
        """
        Вызывается в конце каждой эпохи
        """
        
        # Сравнение точности на текущей эпохе
        if(logs.get('accuracy') >= 1.0):
              self.model.stop_training = True # Остановка обучения

callbacks = ModelCallback() # Обратный вызов
history = model.fit(
    X_train, # Обучающая выборка
    y_train, # Метки
    batch_size = 32, # Размер выборки
    epochs = 15, # Количество эпох
    validation_data = [
        X_val, # Валидационная выборка
        y_val # Метки
    ],
    callbacks = [callbacks] # Обратный вызов функции
)
eval_test = model.evaluate(
    X_test, # Тестовый набор данных
    y_test  # Метки классов
)

eval_test_d = {
    'test_loss': [eval_test[0]],
    'test_accuracy, %': [eval_test[1] * 100]}

df_eval_test = pd.DataFrame(data = eval_test_d).round(3).astype(str)

df_eval_test.head(None).style.hide_index() # Отображение первых N строк или все если указать None