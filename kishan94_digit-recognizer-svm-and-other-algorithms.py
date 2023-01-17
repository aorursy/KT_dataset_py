import pandas as pd

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

%matplotlib inline



from sklearn.tree import DecisionTreeClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier , GradientBoostingClassifier
labled_images = pd.read_csv('../input/train.csv')

images = labled_images.iloc[0:10000,1:]

labels = labled_images.iloc[0:10000,:1]
train_images, test_images,train_labels, test_labels = train_test_split(images, labels, train_size=0.8, random_state=0)
i=1

img = train_images.iloc[i].as_matrix()

img = img.reshape((28,28))

plt.imshow(img,cmap='binary')

plt.title(train_labels.iloc[i,0])
test_images = pd.read_csv('../input/test.csv')

test_images[test_images>0]=1

train_images[train_images>0]=1

model = RandomForestClassifier(n_estimators=100)
model = SVC(kernel='linear',C=0.4)
model = GradientBoostingClassifier()
model = KNeighborsClassifier(n_neighbors = 5)
model = LogisticRegression()
model.fit(train_images,train_labels)

print (model.score(test_images,test_labels))
result = model.predict(test_images)
submit = pd.DataFrame(result)

submit.columns=['Label']

submit.index += 1

submit.index.name = 'ImageId'





submit.to_csv('svm_digit.csv' , header=True)