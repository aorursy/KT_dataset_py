%matplotlib inline
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import skimage.feature as skimage
from skimage.filters import threshold_otsu

dataset = pd.read_csv("../input/train.csv")
target = dataset[[0]].values.ravel()
train = dataset.iloc[:,1:].values
test = pd.read_csv("../input/test.csv").values

target = target.astype(np.uint8)
train = np.array(train).reshape((-1, 1, 28, 28)).astype(np.uint8)
test = np.array(test).reshape((-1, 1, 28, 28)).astype(np.uint8)

image = train[555][0]
#imagenew = skimage.corner_fast(image, n=12, threshold=0.15)
thresh = threshold_otsu(image)
imagenew = image > thresh

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

ax1.axis('off')
ax1.imshow(image, cmap=plt.cm.gray)
ax1.set_title('Input image')
ax1.set_adjustable('box-forced')

# Rescale histogram for better display
#hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 0.02))

ax2.axis('off')
ax2.imshow(imagenew, cmap=plt.cm.gray)
ax2.set_title('New image')
ax1.set_adjustable('box-forced')
plt.show()

#rf = RandomForestClassifier(n_estimators=100)
#rf.fit(train, target)
#pred = rf.predict(test)

#np.savetxt('submission_rand_forest.csv', np.c_[range(1,len(test)+1),pred], delimiter=',', header = 'ImageId,Label', comments = '', fmt='%d')

