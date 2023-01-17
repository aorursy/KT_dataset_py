import os

import numpy as np
import pandas as pd
import tensorflow as tf

import sklearn.metrics as metrics
import matplotlib.pyplot as plt
# Используем keras из tensorflow

keras = tf.keras
layers = keras.layers
image = keras.preprocessing.image

# Некоторые настройки tensorflow нужно делать сразу

# tf.debugging.set_log_device_placement(True)
# Обзор исходных данных

inp_dir = '/kaggle/input'

print("    Содержимое inp_dir:")
print(os.listdir(inp_dir))
print()

hcd_dir = os.path.join(inp_dir, 'histopathologic-cancer-detection')

print("    Содержимое hcd_dir:")
print(os.listdir(hcd_dir))
print()

trn_dir = os.path.join(hcd_dir, 'train')
tst_dir = os.path.join(hcd_dir, 'test')

trn_files = os.listdir(trn_dir)
tst_files = os.listdir(tst_dir)

trn_files.sort()
tst_files.sort()

print("len(trn_files):", len(trn_files))
print("len(tst_files):", len(tst_files))
print()

print("    Имена файлов:")
print(trn_files[:2])
print(tst_files[:2])
trn_df = pd.read_csv(os.path.join(hcd_dir, 'train_labels.csv'))
print("len(trn_df):", len(trn_df))
trn_df.head()
# Посмотрим на данные

img0 = image.load_img(os.path.join(trn_dir, trn_files[0]))

print("type(img0):", type(img0))
print("size:", img0.size)
print("mode:", img0.mode)

img0
'''
# Выборочно проверим по 100 файлов

for k in np.random.choice(len(trn_files), 100):
    fn = os.path.join(trn_dir, trn_files[k])
    img = image.load_img(fn)
    assert img.size == (96, 96) and img.mode == "RGB"

for k in np.random.choice(len(tst_files), 100):
    fn = os.path.join(tst_dir, tst_files[k])
    img = image.load_img(fn)
    assert img.size == (96, 96) and img.mode == "RGB"
'''
None
# Сколько нужно RAM

arr0 = image.img_to_array(img0)

print("type(arr0):", type(arr0))
print("shape:", arr0.shape)
print("dtype:", arr0.dtype)
print("size:", arr0.size)

trn_mem = 4 * arr0.size * len(trn_files)
tst_mem = 4 * arr0.size * len(tst_files)

print("Memory for all train samples in GB:", trn_mem/2**30)
print("Memory for all test samples in GB: ", tst_mem/2**30)
"""
# Так как нам выделено всего 16 GB RAM,
# для обработки всех данных нужно потрудиться,
# например, использовать генератор 

class TestSequence(keras.utils.Sequence):
    def init(self, ???):
        ???
    
    def __len__(self):
        ???

    def __getitem__(self, idx):
        ???
"""
None
# Для простоты ограничимся 8GB

def make_trn_xy(prefix, df, num):
    assert 4*27648*num < 8*2**30
    np.random.seed(2020)  # Для воспроизводимости выборки
    kf_box = np.random.choice(len(df), num, replace=False)
    x_all = np.empty((num,96,96,3), dtype=np.float32)
    y_all = np.empty((num,), dtype=np.float32)
    for k0, kf in enumerate(kf_box):
        fn = df['id'].values[kf] + '.tif'
        x_all[k0] = make_array(os.path.join(prefix, fn))
        y_all[k0] = df.iloc[kf]['label']
    return x_all, y_all

def make_tst_x(prefix, fns):
    num = len(fns)
    x = np.empty((num,96,96,3), dtype=np.float32)
    for k0, fn in enumerate(fns):
        x[k0] = make_array(os.path.join(prefix, fn))
    return x

def make_array(filename):
    img = image.load_img(filename)
    x = image.img_to_array(img)
    assert x.shape == (96, 96, 3)
    x = x / 255.0
    return x
# Модель
# https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html

model = keras.Sequential()

model.add(layers.Conv2D(32, (3, 3), input_shape=(96, 96, 3)))
model.add(layers.Activation('relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

model.add(layers.Conv2D(32, (3, 3)))
model.add(layers.Activation('relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

model.add(layers.Conv2D(64, (3, 3)))
model.add(layers.Activation('relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

model.add(layers.Flatten())  # this converts our 3D feature maps to 1D feature vectors
model.add(layers.Dense(64))
model.add(layers.Activation('relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1))
model.add(layers.Activation('sigmoid'))

model.compile(
    loss='binary_crossentropy',
    optimizer=keras.optimizers.Adam(),
    metrics=['AUC']
)
# Создаем данные в RAM (дорогая операция)

n_t, n_v = 32000, 8000
x_all, y_all = make_trn_xy(trn_dir, trn_df, n_t+n_v)
x_a = x_all[:n_t]
y_a = y_all[:n_t]
x_b = x_all[n_t:]
y_b = y_all[n_t:]
'''
# Если нужно гарантированно вычислять на определенном устройстве

with tf.device('/CPU:0'):
    h2 = model.fit(
        x_a,
        y_a,
        epochs=20,
    )
'''
None
h = model.fit(
    x_a,
    y_a,
    batch_size=64,
    epochs=8,
    validation_data=(x_b,y_b),
)
z0 = h.epoch
z1 = h.history['AUC']
z2 = h.history['val_AUC']

plt.plot(z0, z1, 'ob')
plt.plot(z0, z2, 'or')

for zz in zip(z0, z1, z2):
    print("%2d  %5.3f  %5.3f" % zz)
# Напечатаем

loss, auc = model.evaluate(
    x_b,
    y_b
)
# print("loss: %5.3f" % loss)
print("AUC:  %5.3f" % auc)
# Построим ROC

p_b = model.predict(x_b)

fpr, tpr, thr = metrics.roc_curve(y_b, p_b)

plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr)
plt.xlabel('FPR')

None
# Confusion matrix

y_pred = np.round(p_b)
cm = metrics.confusion_matrix(y_b, y_pred)
tn, fp, fn, tp = cm.ravel()
print("TN:", tn)
print("FP:", fp)
print("FN:", fn)
print("TP:", tp)
cm
# Функция создания файла предсказаний

def make_pred_file(prefix, fns, m, output):
    with open(output, 'w') as fout:
        print('id,label', file=fout)
        k0 = 0
        k0_report = 0
        while k0 < len(fns):
            if k0 >= k0_report:
                print("Start:", k0)
                k0_report += 4096
            k1 = k0 + 64
            b = fns[k0:k1]
            x = make_tst_x(prefix, b)
            p = m.predict(x)
            for fn, v in zip(b, p):
                ident = fn[:-4]
                label = 0 if v < 0.5 else 1
                print("%s,%d" % (ident,label), file=fout)
            k0 = k1
# Создаем файл для загрузки

make_pred_file(tst_dir, tst_files, model, "tmp.csv")
tmp_df = pd.read_csv('tmp.csv')
tmp_df.head()
# Результаты
# Private score: 0.7880
# Public score:  0.8050
