import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
d0 = pd.read_csv("/kaggle/input/mnist-dataset/train.csv")
print(d0.head(5))
l = d0['label']
d= d0.drop('label',axis=1)
print(d)
print(d0.shape)
print(l.shape)
# display or plot a number
data = np.matrix(d)
img = data[785].reshape(28,28)
plt.imshow(img,cmap='gray')
plt.show()
# display or plot a number
data = np.matrix(d)
img = data[1].reshape(28,28)
plt.imshow(img,cmap='gray')
plt.show()
# display or plot a number
data = np.matrix(d)
img = data[2].reshape(28,28)
plt.imshow(img,cmap='gray')
plt.show()
# display or plot a number
data = np.matrix(d)
img = data[3].reshape(28,28)
plt.imshow(img,cmap='gray')
plt.show()
# display or plot a number
data = np.matrix(d)
img = data[6].reshape(28,28)
plt.imshow(img,cmap='gray')
plt.show()