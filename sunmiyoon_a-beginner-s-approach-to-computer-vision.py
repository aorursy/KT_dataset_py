import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os
print(os.listdir("../input"))
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
X_train = train.iloc[:, 1:]
Y_train = train.iloc[:, 0]
X_test = test
plt.imshow(train.iloc[1, 1:].values.reshape(28, 28), cmap='gray')
plt.title('label: {}'.format(train.iloc[1, 0]))
train['label'].hist(bins=20)
# examining the pixel values
# These images are not actually black and white. They are gray scale (0-255).
plt.hist(train.ix[:, 1:].iloc[1])
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
train_images, vali_images, train_labels, vali_labels = train_test_split(X_train, Y_train, train_size=0.8, random_state=0)
print("shape of train images: {}".format(train_images.shape))
print("shape of validation images: {}".format(vali_images.shape))
forest = RandomForestClassifier(n_estimators=100, random_state=5)
forest.fit(train_images, train_labels)
print('accuracy of training set: {}'.format(forest.score(train_images, train_labels)))
print('accuracy of validation set: {}'.format(forest.score(vali_images, vali_labels)))
cross_val_score(forest, X_train, Y_train)
submission = pd.DataFrame()
submission['Label'] = forest.predict(X_test)
submission.index += 1
submission.index.name = 'ImageId'
plt.imshow(test.iloc[0, :].values.reshape(28, 28), cmap='gray')
plt.title('label: {}'.format(train.iloc[0, 0]))
submission.head(2)
submission.to_csv('./submission.csv')