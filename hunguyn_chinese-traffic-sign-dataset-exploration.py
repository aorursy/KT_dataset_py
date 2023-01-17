import pandas as pd
import os

import IPython.display as display
from PIL import Image
from PIL.ImageDraw import Draw
from skimage.feature import hog
from skimage import color
from skimage.transform import resize
from sklearn import svm
from sklearn.metrics import classification_report,accuracy_score, confusion_matrix
import numpy as np
data_dir = '../input/gtsrb-german-traffic-sign'
df = pd.read_csv(os.path.join(data_dir, 'Train.csv'))
df.head(5)
for index, row in df.sample(frac=1)[:5].iterrows():
    image = Image.open(os.path.join(data_dir, row.Path))
    draw = Draw(image)
#     draw.rectangle([row.Roi.x1, row.Roi.y1, row.Roi.x2, row.Roi.y2], outline='#00FF00', width=3)
    display.display(image)
    print('category:', row.ClassId)
def view_image(row):
    image = Image.open(os.path.join(data_dir, row.Path))
    draw = Draw(image)
#     draw.rectangle([row.Roi.x1, row.Roi.y1, row.Roi.x2, row.Roi.y2], outline='#00FF00', width=3)
    display.display(image)
    print('category:', row.ClassId)

data = df.loc[(df['ClassId'] == 33) | (df['ClassId'] == 34)]
data
unk = df.loc[(df['ClassId'] != 33) & (df['ClassId'] != 34)].sample(500)
unk.head(5)
data = data.append(unk)
len(data)
view_image(data.iloc[1000])
view_image(data.iloc[1300])
view_image(data.iloc[10])
from sklearn.model_selection import train_test_split
#train test split
train, test = train_test_split(data, test_size=0.2, random_state=42, shuffle=True)
len(test)
import cv2

x_train = [cv2.imread(os.path.join(data_dir, path), cv2.IMREAD_GRAYSCALE) for path in train['Path']]
x_train[0]
y_train = train['ClassId'].copy()

y_train[(y_train != 33) & (y_train != 34)] = 0
ppc = 32
hog_images = []
hog_features = []
for i in range(len(x_train)):
    image = x_train[i]
    image = resize(image, (128, 128), anti_aliasing=True)
    fd= hog(image, orientations=4, pixels_per_cell=(ppc,ppc),cells_per_block=(2, 2),block_norm= 'L2')
#     hog_images.append(hog_image)
    hog_features.append(fd)
clf = svm.SVC()
# hog_features = np.array(hog_features)
# train_frame = np.hstack((hog_features,y_train))
# np.random.shuffle(train_frame)
clf.fit(hog_features, y_train)
y_pred = clf.predict(hog_features)
print(classification_report(y_train, y_pred))
x_test = [cv2.imread(os.path.join(data_dir, path), cv2.IMREAD_GRAYSCALE) for path in test['Path']]
y_test = test['ClassId'].copy()
y_test[(y_test != 33) & (y_test != 34)] = 0
ppc = 32
test_features = []
for i in range(len(x_test)):
    image = x_test[i]
    image = resize(image, (128, 128), anti_aliasing=True)
    fd= hog(image, orientations=4, pixels_per_cell=(ppc,ppc),cells_per_block=(2, 2),block_norm= 'L2')
#     hog_images.append(hog_image)
    test_features.append(fd)
y_pred = clf.predict(test_features)
print(classification_report(y_test, y_pred))
confusion_matrix(y_test, y_pred)
np.where(y_test != y_pred)
view_image(unk.iloc[291])
view_image(unk.iloc[53])