import numpy as np

import pandas as pd

import PIL as pil
data = pd.read_csv("/kaggle/input/digit-recognizer/train.csv").values
X = data[:,1:]

y = data[:,0]
def draw(x, y):

    label = y

    pic_data = x.reshape((28,28)).astype(np.uint8)

    img = pil.Image.fromarray(pic_data)

    print(label)

    return img

draw(X[0], y[0])
N = 42000
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt



X_embedded = TSNE(n_components=4, n_jobs=-1, init="pca").fit_transform(X[:N])

for d in range(10):

    mask = (y[:N] == d)

    digits_X_emb = X_embedded[mask]

    plt.scatter(digits_X_emb[:,0], digits_X_emb[:,1], s=10, label=d)

plt.legend()

plt.show()
from sklearn.svm import SVC



svm = SVC()

svm.fit(X[:N], y[:N])

test = pd.read_csv("/kaggle/input/digit-recognizer/test.csv").values

res = svm.predict(test)
output = pd.read_csv("/kaggle/input/digit-recognizer/sample_submission.csv")

output["Label"] = res

output.to_csv('submission.csv', index=False)

!head -n5 submission.csv