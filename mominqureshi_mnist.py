import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

train = pd.read_csv('../input/digit-recognizer/train.csv')
test = pd.read_csv('../input/digit-recognizer/test.csv')
X = train.drop(['label'], 1).values
Y = train['label'].values

X = X / 255.0

test_x = test.values
test_x = test_x / 255.0

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=0)

from sklearn.neural_network import MLPClassifier
classifier = MLPClassifier(solver='adam', alpha=1e-5, max_iter=50,
                           hidden_layer_sizes=50, random_state=1)

classifier.fit(X, Y)
print("FFNN")
print("Training set score: %f" % classifier.score(x_train, y_train))
print("Test set score: %f" % classifier.score(x_test, y_test))
res = classifier.predict(test_x)

from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier()

classifier.fit(X, Y)
print("KNN")
print("Training set score: %f" % classifier.score(x_train, y_train))
print("Test set score: %f" % classifier.score(x_test, y_test))
res = classifier.predict(test_x)

from sklearn.svm import SVC
from skimage.feature import hog
ppc = 7
hog_images = []
hog_features = []
for image in X:
    image = image.reshape(28, 28)
    fd = hog(image, orientations=8, pixels_per_cell=(ppc, ppc), cells_per_block=(4, 4), block_norm='L2')
    hog_features.append(fd)

hog_features = np.array(hog_features)

hog_images_test = []
hog_features_test = []
for image in test_x:
    image = image.reshape(28, 28)
    fd = hog(image, orientations=8, pixels_per_cell=(ppc, ppc), cells_per_block=(4, 4), block_norm='L2')
    hog_features_test.append(fd)

hog_features_test = np.array(hog_features_test)

print(X.shape)

classifier = SVC(verbose=True)
classifier.fit(hog_features, Y)
print("Training set score: %f" % classifier.score(hog_features, Y))
res = classifier.predict(hog_features_test)
import pandas as pd
df = pd.DataFrame(data=res, columns=['Label'])
df.index += 1
df.to_csv("submission.csv", index_label='ImageId')