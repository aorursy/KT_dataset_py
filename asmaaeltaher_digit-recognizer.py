import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# visualization
import seaborn as sns
import matplotlib.pyplot as plt, matplotlib.image as mpimg
%matplotlib inline
train = pd.read_csv('../input/train.csv')
imgs = train.drop(['label'], axis=1)
labels = train['label']
test = pd.read_csv('../input/test.csv')
i=50
img= imgs.iloc[i].as_matrix()
img= img.reshape((28,28))
plt.imshow(img, cmap='gray')
plt.title(labels.iloc[i])
plt.hist(imgs.iloc[i])
imgs[imgs>0] = 1
test[test>0] = 1
i=1
img= imgs.iloc[i].as_matrix()
img= img.reshape((28,28))
plt.imshow(img, cmap='gray')
plt.title(labels.iloc[i])
plt.hist(imgs.iloc[i])
train_x, test_x, train_y, test_y= train_test_split(imgs, labels, test_size=0.20)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC

from sklearn.metrics import accuracy_score
model_1 = SVC()
model_1.fit(train_x, train_y)
preds_1 = model_1.predict(test_x)
print(accuracy_score(preds_1,test_y))
model_2 = KNeighborsClassifier(n_neighbors=3)
model_2.fit(train_x, train_y)
preds_2 = model_2.predict(test_x)
print(accuracy_score(preds_2,test_y))
model_1 = SVC()
model_1.fit(imgs, labels)
preds_1 = model_1.predict(test)
output_1 = pd.DataFrame({"ImageId": list(range(1,len(preds_1)+1)),
                        "Label": preds_1})
output_1.to_csv( 'Digit_Recognizer_preds_1.csv',index=False , header=True )


model_2.fit(imgs, labels)
preds_2 = model_2.predict(test)
output_2 = pd.DataFrame({"ImageId": list(range(1,len(preds_2)+1)),
                         "Label": preds_2})
output_2.to_csv( 'Digit_Recognizer_preds_2.csv', index=False, header=True )
