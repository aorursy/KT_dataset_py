import pandas as pd
import matplotlib.pyplot as plt, matplotlib.image as mpimg
from sklearn.model_selection import train_test_split

from sklearn.ensemble import GradientBoostingClassifier

%matplotlib inline
labeled_images = pd.read_csv('../input/train.csv')

images = labeled_images.iloc[0:,1:]
labels = labeled_images.iloc[0:,:1]
train_images, test_images,train_labels, test_labels = train_test_split(images, labels, train_size=0.8, random_state=2)
i=2
img=train_images.iloc[i].as_matrix()
img=img.reshape((28,28))
plt.imshow(img,cmap='gray')
plt.title(train_labels.iloc[i,0])
plt.hist(train_images.iloc[i])
#score= 0.905 5000
#test_images=test_images/255.0
#train_images=train_images/255.0

#score= 0.914 5000rows
#score= 0.9421428571428572 todo
test_images[test_images>0]=1
train_images[train_images>0]=1


img=train_images.iloc[i].as_matrix().reshape((28,28))
plt.imshow(img,cmap='binary')
plt.title(train_labels.iloc[i])
plt.hist(train_images.iloc[i])
gbc=GradientBoostingClassifier()
gbc.fit(train_images, train_labels.values.ravel())
gbc.score(test_images,test_labels)
test_data=pd.read_csv('../input/test.csv')
test_data[test_data>0]=1
results=gbc.predict(test_data)
results.size
df = pd.DataFrame(results)
df.index+=1
df.index.name='ImageId'
df.columns=['Label']
df.to_csv('results.csv', header=True)
import os
os.listdir("./")