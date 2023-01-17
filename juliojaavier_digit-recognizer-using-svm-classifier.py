import pandas as pd
import numpy  as np
from matplotlib import pyplot as plt
from sklearn    import svm, metrics
import warnings
warnings.filterwarnings("ignore")
%matplotlib inline
train   = pd.read_csv("../input/train.csv")
target  = train['label']
X_train = train[train.columns[1:]]
imgmatrix = X_train[:].as_matrix()
plt.figure(figsize=(15,10))
plt.suptitle("Training Digits", fontsize="x-large")
for k in range(150):
    plt.subplot(10,15,1+k)
    image = imgmatrix[k].reshape((28,28))
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
plt.savefig('digits.png')
classifier = svm.SVC(kernel='rbf', gamma=0.001)
classifier.fit(imgmatrix[0:500], target[0:500])
predicted = classifier.predict(imgmatrix[0:600])
accuracy  = target[0:600] == predicted
print('Accuracy: %i percent'%sum(accuracy))
plt.figure(figsize=(16,3))
plt.plot(target[0:600].index, target[0:600], color='red', alpha=.5)
plt.plot(target[0:600].index, predicted, color='blue', alpha=.5)
plt.savefig('accuracy.png')
len(target)
