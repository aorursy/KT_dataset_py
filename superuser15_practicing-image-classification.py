import pandas as pd
import matplotlib.pyplot as plt, matplotlib.image as mpimg
from sklearn.model_selection import train_test_split
from sklearn import svm
%matplotlib inline
labeled_images = pd.read_csv('../input/train.csv')

images = labeled_images.iloc[0:5000,1:]
labels = labeled_images.iloc[0:5000,:1]
train_images, test_images,train_labels, test_labels = train_test_split(images, labels, train_size=0.8, random_state=0)
labeled_images.head()
print(labels.shape)
i=1
img=train_images.iloc[i].as_matrix()

img=img.reshape((28,28))
plt.imshow(img,cmap='gray')
plt.title(train_labels.iloc[i,0])
#checking img
#print(type(img))
#print(img)
img1 = train_images.iloc[1] #getting the first row from the data frame train_images and storing it in a series img1
img1 = img1.as_matrix()  #converting the 1-d array to matrix
img1 = img1.reshape((28,28)) #now reshaping the matrix to (28 x 28) matrix

#plt.imshow() function displays the image on the axes and cmap is "colormap" 
#If None, default to rc image.cmap value. cmap is ignored if X is 3-D, directly specifying RGB(A) values

plt.imshow(img1,cmap = 'gray')
#(train_labels.iloc[1,0]) parameter inside below function is getting the label of the selected image which in our case is 1 
#and the value of the label is 6. 

plt.title(train_labels.iloc[i,0])  #giving title to the our image i.e. 6
plt.hist(train_images.iloc[i]) 
#train_labels.values.ravel()

clf = svm.SVC(gamma = 'auto')
clf.fit(train_images, train_labels.values.ravel())
score = clf.score(test_images,test_labels)
print(score * 100)

test_images[test_images>0]=1
train_images[train_images>0]=1
img=train_images.iloc[i].as_matrix().reshape((28,28))
plt.imshow(img,cmap='binary')
plt.title(train_labels.iloc[i])
#print(type(test_images)) # pandas dataframe
t = test_images.iloc[1]
i=1
for i in t:
    print(i)
plt.hist(train_images.iloc[i])
clf = svm.SVC(gamma = 'auto')
clf.fit(train_images, train_labels.values.ravel())
score2 = clf.score(test_images,test_labels)
print(score2)
test_data=pd.read_csv('../input/test.csv')
test_data[test_data>0]=1
results=clf.predict(test_data[0:5000])

df = pd.DataFrame(results)
print(df.head())
df.index.name='ImageId'
df.index+=1
df.columns=['Label']

df.to_csv('results.csv', header=True)
