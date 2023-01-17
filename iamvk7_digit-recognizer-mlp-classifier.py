# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
data = pd.read_csv('../input/train.csv')
data.head()
# Any results you write to the current directory are saved as output.
labels = data['label']
del data['label']
data.head()
y = labels.as_matrix()
X = data.as_matrix()
labels.head()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
print(len(X_train))
print(len(y_train))
print(len(X_test))
print(len(y_test))
import matplotlib.pyplot as plt
import matplotlib.cm as cm
def display_image(img):
    one_image = img.reshape(28,28)
    #print(one_image)
    plt.axis('off')
    plt.imshow(one_image, cmap=cm.binary)
    plt.show()
print(len(X_train[0]))
#print(X_train[0])
display_image(X_train[0])
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(500, 500,500), random_state=1)
clf.fit(X_train, y_train)
pred = clf.predict(X_test)
print(accuracy_score(pred,y_test))
test_data = pd.read_csv('../input/test.csv')
test_data.head()
X_res = test_data.as_matrix()
print(X_res[:5])
predict_res = clf.predict(X_res)
print(predict_res[:5])
imid = [i for i in range(1,len(predict_res)+1)]
result = pd.DataFrame({'ImageId':pd.Series(imid),'Label':pd.Series(predict_res)})
result.to_csv('result.csv',index=False)
result.head(20)
