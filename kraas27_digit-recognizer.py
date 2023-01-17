import cv2

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline
train = np.loadtxt('../input/train.csv', delimiter=',', skiprows=1)

# test = np.loadtxt('./data/digit/test.csv', delimiter=',', skiprows=1)
# сохраняем разметку в отдельную переменную

train_label = train[:, 0]

# приводим размерность к удобному для обаботки виду

train_img = np.resize(train[:, 1:], (train.shape[0], 28, 28))

# test_img = np.resize(test, (test.shape[0], 28, 28))
train_img.shape
fig = plt.figure(figsize=(20, 10))

for i, img in enumerate(train_img[0:5], 1):

    subplot = fig.add_subplot(1, 7, i)

    plt.imshow(img, cmap='gray');

    subplot.set_title('%s' % train_label[i - 1]);
train_sobel_x = np.zeros_like(train_img)

train_sobel_y = np.zeros_like(train_img)

for i in range(len(train_img)):

    train_sobel_x[i] = cv2.Sobel(train_img[i], cv2.CV_64F, dx=1, dy=0, ksize=3)

    train_sobel_y[i] = cv2.Sobel(train_img[i], cv2.CV_64F, dx=0, dy=1, ksize=3)
# test_sobel_x = np.zeros_like(test_img)

# test_sobel_y = np.zeros_like(test_img)

# for i in range(len(test_img)):

#     test_sobel_x[i] = cv2.Sobel(test_img[i], cv2.CV_64F, dx=1, dy=0, ksize=3)

#     test_sobel_y[i] = cv2.Sobel(test_img[i], cv2.CV_64F, dx=0, dy=1, ksize=3)
train_g, train_theta = cv2.cartToPolar(train_sobel_x, train_sobel_y)
# test_g, test_theta = cv2.cartToPolar(test_sobel_x, test_sobel_y)
fig = plt.figure(figsize=(20, 10))

for i, img in enumerate(train_g[:5], 1):

    subplot = fig.add_subplot(1, 7, i)

    plt.imshow(img, cmap='gray');

    subplot.set_title('%s' % train_label[i - 1]);

    subplot = fig.add_subplot(3, 7, i)

    plt.hist(train_theta[i - 1].flatten(),

             bins=16, weights=train_g[i - 1].flatten())
# Гистограммы вычисляются с учетом длины вектора градиента

train_hist = np.zeros((len(train_img), 16))

for i in range(len(train_img)):

    hist, borders = np.histogram(train_theta[i],

                                 bins=16,

                                 range=(0., 2. * np.pi),

                                 weights=train_g[i])

    train_hist[i] = hist
# test_hist = np.zeros((len(test_img), 16))

# for i in range(len(test_img)):

#     hist, borders = np.histogram(test_theta[i],

#                                  bins=16,

#                                  range=(0., 2. * np.pi),

#                                  weights=test_g[i])

#     test_hist[i] = hist
# По умолчанию используется L2 норма

train_hist = train_hist / np.linalg.norm(train_hist, axis=1)[:, None]
# test_hist = test_hist / np.linalg.norm(test_hist, axis=1)[:, None]
train_hist.shape
train.shape#, test.shape
data_for_svd = train[:, 1:]

data_for_svd.shape
data_mean = np.mean(data_for_svd, axis=0)

data_for_svd -= data_mean
cov_matrix = np.dot(data_for_svd.T, data_for_svd) / data_for_svd.shape[0]
U, S, _ = np.linalg.svd(cov_matrix)
S_thr = 0.83

S_cumsum = 0

for i in range(S.shape[0]):

    S_cumsum += S[i]/np.sum(S)

    if S_cumsum >= S_thr:

        n_comp = i+1

        print ('n_comp:', n_comp, '\t', 'cumsum:', S_cumsum)

        break
data_reduced = np.dot(data_for_svd, U[:, :n_comp])

data_reduced.shape
train_data_svd = data_reduced[:42000]

test_data_svd = data_reduced[42000:]

train_data_svd.shape, test_data_svd.shape
train_data = np.hstack((train_hist, train_data_svd))

# test_data = np.hstack((test_hist, test_data_svd))

train_data.shape#, test_data.shape
(h, w) = train_img.shape[1:]

(cX, cY) = (int(w * 0.5), int(h * 0.5))

        

segments = [(0, w, 0, cY), 

            (0, w, cY, h),

            (0, cX, 0, h),

            (cX, w, 0, h)]
# посмотрим правильно ли разделились сигменты

fig = plt.figure(figsize=(16, 4))

for num, i in enumerate(segments, 1):

    subplot = fig.add_subplot(1, 4, num)

    plt.imshow(train_img[1, i[0]:i[1], i[2]:i[3]], cmap='gray')
for i in segments:

    train_img_s = train_img[:, i[0]:i[1], i[2]:i[3]]

#     test_img_s = test_img[:, i[0]:i[1], i[2]:i[3]]

    train_sobel_x = np.zeros_like(train_img_s)

    train_sobel_y = np.zeros_like(train_img_s)

    for i in range(len(train_img_s)):

        train_sobel_x[i] = cv2.Sobel(train_img_s[i], cv2.CV_64F, dx=1, dy=0, ksize=3)

        train_sobel_y[i] = cv2.Sobel(train_img_s[i], cv2.CV_64F, dx=0, dy=1, ksize=3)

#     test_sobel_x = np.zeros_like(test_img_s)

#     test_sobel_y = np.zeros_like(test_img_s)

#     for i in range(len(test_img_s)):

#         test_sobel_x[i] = cv2.Sobel(test_img_s[i], cv2.CV_64F, dx=1, dy=0, ksize=3)

#         test_sobel_y[i] = cv2.Sobel(test_img_s[i], cv2.CV_64F, dx=0, dy=1, ksize=3)



    train_g, train_theta = cv2.cartToPolar(train_sobel_x, train_sobel_y)

#     test_g, test_theta = cv2.cartToPolar(test_sobel_x, test_sobel_y)





    train_hist = np.zeros((len(train_img_s), 16))

    for i in range(len(train_img_s)):

        hist, borders = np.histogram(train_theta[i],

                                     bins=16,

                                     range=(0., 2. * np.pi),

                                     weights=train_g[i])

        train_hist[i] = hist



#     test_hist = np.zeros((len(test_img_s), 16))

#     for i in range(len(test_img_s)):

#         hist, borders = np.histogram(test_theta[i],

#                                      bins=16,

#                                      range=(0., 2. * np.pi),

#                                      weights=test_g[i])

#         test_hist[i] = hist



    train_hist_part = train_hist / np.linalg.norm(train_hist, axis=1)[:, None]

#     test_hist_part = test_hist / np.linalg.norm(test_hist, axis=1)[:, None]



    train_data = np.hstack((train_data, train_hist_part))

#     test_data = np.hstack((test_data, test_hist_part))
train_data.shape#, test_data.shape
from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(train_data, train_label, test_size=0.2)
X_train.shape, X_val.shape, y_train.shape, y_val.shape
xgb = XGBClassifier(colsample_bytree=1, gamma=0, max_depth=5, reg_alpha=0.6, reg_lambda=0.1, 

                    subsample=0.9, n_jobs=-1, n_estimators=500, learning_rate=0.05)
xgb.fit(X_train, y_train)
y_pred = xgb.predict(X_val)
from sklearn.metrics import accuracy_score

print('Accuracy: %s' % accuracy_score(y_val, y_pred))
from sklearn.metrics import classification_report

print(classification_report(y_val, y_pred))
from sklearn.metrics import confusion_matrix

print(confusion_matrix(y_val, y_pred))