import numpy as np
from skimage.io import imread, imsave
from skimage.transform import resize
fpath = '../input/pokemon-images-and-types/images/images/abomasnow.png'
img = imread(fpath)
print(img.shape)
print(img)
import matplotlib.pyplot as plt
plt.imshow(img)
plt.show()

fpath = '../input/pokemon-images-and-types/images/images/aegislash-blade.png'
img = imread(fpath)
plt.imshow(img)
plt.show()
img[:,:,1] = 0
img[:,:,2] = 0
plt.imshow(img)
plt.show()

fpath = '../input/pokemon-images-and-types/images/images/aegislash-blade.png'
img = imread(fpath)
img[:,:,0] = 0 # show the green value
img[:,:,2] = 0
plt.imshow(img)
plt.show()

fpath = '../input/pokemon-images-and-types/images/images/aegislash-blade.png'
img = imread(fpath)
img[:,:,0] = 0 #show the blue value
img[:,:,1] = 0
plt.imshow(img)
plt.show()

print(img.shape)
imgresized = resize(img, (256, 256))
print(img.shape)
print(imgresized.shape)


plt.imshow(imgresized)
plt.show()
#building training dataset

from skimage.io import imread
import numpy as np
from skimage.transform import resize
import matplotlib.pyplot as plt
import glob

dataset = []
label = []

fpath = '../input/saudogphonedemo/dataset/train/dog/*.jpg'
allpath = glob.glob(fpath)
#print(allpath)
for p in allpath:
    img = imread(p)
    imgresize = resize(img, (64, 64))
    imgflatten = imgresize.flatten()
    dataset.append(imgflatten)
    label.append('dog')
    #plt.imshow(imgresize)
    #plt.show()
#print(imgresize.shape)

fpath = '../input/saudogphonedemo/dataset/train/phone/*.jpg'
allpath = glob.glob(fpath)
#print(allpath)
for p in allpath:
    img = imread(p)
    imgresize = resize(img, (64, 64))
    imgflatten = imgresize.flatten()
    dataset.append(imgflatten)
    label.append('phone')

#print(dataset)
X = np.array(dataset)
print(X)
print(X.shape)
print(label)
print(len(label))
print(64*64*3)
def NN(img, X, label):
    qa = img
    a = X
    G = label
    minusa = a - qa
    sq = minusa*minusa
    sqsum = np.sum(sq, axis=1)
    res = np.argmin(np.sqrt(sqsum))
    index = res
    return G[index]

print('dog images.....')
testimgpath = '../input/saudogphonedemo/dataset/test/dog/*.jpg'
allimgpath = glob.glob(testimgpath)
for p in allimgpath:
    img = imread(p)
    imgresize = resize(img, (64, 64))
    imgflatten = imgresize.flatten()

    print(p, NN(imgflatten, X, label))

    
print('phone images.....')
testimgpath = '../input/saudogphonedemo/dataset/test/phone/*.jpg'
allimgpath = glob.glob(testimgpath)
for p in allimgpath:
    img = imread(p)
    imgresize = resize(img, (64, 64))
    imgflatten = imgresize.flatten()

    print(p, NN(imgflatten, X, label))
    
    

#building training dataset

from skimage.io import imread
import numpy as np
from skimage.transform import resize
import matplotlib.pyplot as plt
import glob

dataset = []
label = []

fpath = '../input/saudogphonedemo/dataset/train/dog/*.jpg'
allpath = glob.glob(fpath)
#print(allpath)
for p in allpath:
    img = imread(p)
    imgresize = resize(img, (64, 64))
    imgflatten = imgresize.flatten()
    dataset.append(imgflatten)
    label.append('dog')
    #plt.imshow(imgresize)
    #plt.show()
#print(imgresize.shape)

fpath = '../input/saudogphonedemo/dataset/train/phone/*.jpg'
allpath = glob.glob(fpath)
#print(allpath)
for p in allpath:
    img = imread(p)
    imgresize = resize(img, (64, 64))
    imgflatten = imgresize.flatten()
    dataset.append(imgflatten)
    label.append('phone')

#print(dataset)
X = np.array(dataset)
print(X)
print(X.shape)
print(label)
print(len(label))
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
model = KNeighborsClassifier(n_neighbors = 5) #declare
model.fit(X, label)
model = DecisionTreeClassifier()
model.fit(X, label)
dogimgpath = '../input/saudogphonedemo/dataset/test/dog/OIP_352.jpg'
img = imread(dogimgpath)
imgresize = resize(img, (64, 64))
imgflatten = imgresize.flatten()
result = model.predict(imgflatten.reshape(1, -1))
print(result)
testimgpath = '../input/saudogphonedemo/dataset/test/dog/*.jpg'
allimgpath = glob.glob(testimgpath)

for p in allimgpath:
    img = imread(p)
    imgresize = resize(img, (64, 64))
    imgflatten = imgresize.flatten()
    print(model.predict(imgflatten.reshape(1, -1)))
testimgpath = '../input/saudogphonedemo/dataset/test/dog/*.jpg'
allimgpath = glob.glob(testimgpath)
testarray = []

for p in allimgpath:
    img = imread(p)
    imgresize = resize(img, (64, 64))
    imgflatten = imgresize.flatten()
    testarray.append(imgflatten)
test = np.array(testarray)
print(model.predict(test))
from sklearn.model_selection import cross_val_score

for neighbor in range(1, 10, 2):
    print(neighbor)
    model = KNeighborsClassifier(n_neighbors = neighbor)
    scores = cross_val_score(model, X, label, cv=5)
    print(scores)
    print(scores.mean())
from sklearn.model_selection import cross_val_score

for neighbor in range(1, 10, 2):
    for pnum in [1, 2]:
        print(neighbor, pnum)
        model = KNeighborsClassifier(n_neighbors = neighbor, p = pnum)
        scores = cross_val_score(model, trainX, label, cv=5)
# print(scores)
        print(scores.mean())


from sklearn.svm import LinearSVC

for Cp in [0.1, 1, 10]:
    for lossp in ['hinge', 'squared_hinge']:
        model = LinearSVC(C = Cp, loss = lossp, max_iter = 100)
        scores = cross_val_score(model, trainX, label, cv = 5)
        print(Cp, lossp, ':', scores.mean())
from sklearn.svm import SVC

for Cp in [1]:
    for gammap in [0.01, 0.05]:
        model = SVC(C = Cp, gamma = gammap)
        scores = cross_val_score(model, trainX, label, cv=5)
        print(Cp, gammap, ':', scores.mean())

from sklearn.model_selection import GridSearchCV
param_grid = {'n_neighbors':[1, 3]}

model = GridSearchCV(KNeighborsClassifier(), param_grid, cv=3)
model.fit(X, label)

print(model.best_params_)

para_grid = {'C':[1, 5], 'gamma':[0.0001, 0.0005, 0.001, 0.005]}
model = GridSearchCV(SVC(), para_grid, cv=2)
model.fit(trainX, label)
print(model.best_params_)
print(model.best_params_)
from sklearn.model_selection import cross_val_score

model = KNeighborsClassifier()
scores = cross_val_score(model, trainX, label, cv=5)
print(scores.mean())


# model = KNeighborsClassifier(n_neighbors=1)

from sklearn.model_selection import GridSearchCV

# param_grid = [
#   {'C': [1, 10], 'kernel': ['linear']},
#   {'C': [1, 10], 'gamma': [0.1, 0.01], 'kernel': ['rbf']},
# ]

param_grid = {'n_neighbors':[1, 3]}

clf = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5)
clf.fit(trainX, label)

print(clf.best_params_)
# model = RandomForestClassifier(n_estimators=50, min_samples_split=5, oob_score=True)
# # model = SVC(gamma='scale')
# # model = LogisticRegression()
# model.fit(trainX, label)
# print(model.oob_score_)

# print(trainX)
# print(label)

 
fnames = './dataset/test/cat/*.jpg'
fnameall = glob.glob(fnames)

groundtruth = []
prediction = []

for fname in fnameall:
    img = imread(fname)
    imgre = imresize(img, (64, 64))
#     plt.imshow(imgre)
#     plt.show()

    imgflatten = imgre.flatten()
    imgflatten = imgflatten/255.0
    category = model.predict([imgflatten])
    prediction.extend(category)
    groundtruth.append('cat')
#     print(category)

fnames = './dataset/test/dog/*.jpg'
fnameall = glob.glob(fnames)

for fname in fnameall:
    img = imread(fname)
    imgre = imresize(img, (64, 64))
#     plt.imshow(imgre)
#     plt.show()

    imgflatten = imgre.flatten()
    imgflatten = imgflatten/255.0

    category = model.predict([imgflatten])
    prediction.extend(category)
    groundtruth.append('dog')


print("groundtruth")
print(groundtruth)
print('prediction')
print(prediction)

from sklearn.metrics import accuracy_score

acc = accuracy_score(groundtruth, prediction)
print(acc)

from sklearn.model_selection import cross_val_score

model = KNeighborsClassifier()
scores = cross_val_score(model, trainX, label, cv=5)
print(scores.mean())


# model = KNeighborsClassifier(n_neighbors=1)

from sklearn.model_selection import GridSearchCV

# param_grid = [
#   {'C': [1, 10], 'kernel': ['linear']},
#   {'C': [1, 10], 'gamma': [0.1, 0.01], 'kernel': ['rbf']},
# ]

param_grid = {'n_neighbors':[1, 3]}

clf = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5)
clf.fit(trainX, label)

print(clf.best_params_)
# model = RandomForestClassifier(n_estimators=50, min_samples_split=5, oob_score=True)
# # model = SVC(gamma='scale')
# # model = LogisticRegression()
# model.fit(trainX, label)
# print(model.oob_score_)

# print(trainX)
# print(label)
%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets.samples_generator import make_blobs
x, y = make_blobs(n_samples=300, centers = 5, cluster_std=1.0, random_state=0)
plt.scatter(x[:,0], x[:,1], c=y, cmap='viridis')
plt.show()
print(y)
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=10)
kmeans.fit(x)
y_pred = kmeans.predict(x)
plt.scatter(x[:,0], x[:,1], c=y_pred, cmap='viridis')
centers = kmeans.cluster_centers_
plt.scatter(centers[:,0], centers[:,1], c='black', s=100)
plt.show()
loss = []
for i in range(1, 11, 1):
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(x)
    loss.append(kmeans.inertia_)
print(loss)
plt.plot(range(1,11,1), loss)
plt.title('Elbow method')
plt.xlabel("number of clusters")
plt.ylabel('loss fuction')
plt.show()
kmeans = KMeans(n_clusters=5)
kmeans.fit(x)
y_pred = kmeans.predict(x)
plt.scatter(x[:,0], x[:,1], c=y_pred, cmap='viridis')
centers = kmeans.cluster_centers_
plt.scatter(centers[:,0], centers[:,1], c='black', s=100)
plt.show()

