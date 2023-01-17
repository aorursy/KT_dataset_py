import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statistics import mean
%matplotlib inline
import cv2
import os
location_train = '../input/ds-14/train images/train images/'
train_images_name = os.listdir(location_train)
len(train_images_name)
image_data = cv2.imread(location_train+train_images_name[0])
image_data.shape
train_data = pd.read_csv('../input/ds-14/train.csv')
train_data.head()
plt.figure(figsize= (15,6))
sns.countplot(train_data.subspecies)
H,W = [],[]
for i,filename in enumerate(train_data.file):
    oriimg = cv2.imread(location_train+filename)
    height, width, depth = oriimg.shape
    H.append(height)
    W.append(width)
print('maximum Image Height: ',max(H))
print('maximum Image Width: ',max(W))
print('Average Image Height: ',mean(H))
print('Average Image Width: ',mean(W))
H2= int(mean(H))
W2= int(mean(W))
oriimg = cv2.imread(location_train+train_data.file[0])
IMG = cv2.resize(oriimg,(W2,H2),interpolation=cv2.INTER_CUBIC).flatten()

for i,filename in enumerate(train_data.file):
    if i != 0:
        oriimg = cv2.imread(location_train+filename)
        newimg = cv2.resize(oriimg,(W2,H2),interpolation=cv2.INTER_CUBIC).flatten()
        IMG = np.vstack((IMG,newimg))
    
IMG.shape
IMG.dtype
Ytrain = train_data.subspecies
from sklearn.decomposition import PCA
pca = PCA(.95,random_state=42)
pca.fit(IMG)
PCA_of_IMG=pca.transform(IMG)
print(PCA_of_IMG.shape,Ytrain.shape)
type(PCA_of_IMG)
test_data = pd.read_csv('../input/ds-14/test.csv')
test_data.head()
location_test = '../input/ds-14/test images/test images/'
oriimg = cv2.imread(location_test+test_data.file[0])
IMG_test = cv2.resize(oriimg,(W2,H2),interpolation=cv2.INTER_CUBIC).flatten()

for i,filename in enumerate(test_data.file):
    if i != 0:
        oriimg = cv2.imread(location_test+filename)
        newimg = cv2.resize(oriimg,(W2,H2),interpolation=cv2.INTER_CUBIC).flatten()
        IMG_test = np.vstack((IMG_test,newimg))
test_img=pca.transform(IMG_test)
xtrain = PCA_of_IMG
ytrain = Ytrain
xtest = test_img
#Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start=49, stop=51, num=3)]

# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(24, 26, num=3)]
max_depth.append(None)

# Minimum number of samples required to split a node
min_samples_split = [ 2]

#criterion = ["gini", "entropy"]
criterion = [ "entropy"]
#max_features = ["auto", "sqrt", "log2"]
random_grid1 = {"n_estimators": n_estimators,
                "criterion": criterion,
                "max_depth": max_depth,
                #"max_features": max_features,
                "min_samples_split": min_samples_split}
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
clf_rf = RandomForestClassifier(n_estimators=500,random_state=100)
rf_random = RandomizedSearchCV(
    estimator=clf_rf, param_distributions=random_grid1, n_iter=5000, cv=5, verbose=1, random_state=100, n_jobs=4)

# Fit the random search model
rf_random.fit(xtrain, ytrain)
clf_rf_rand = rf_random.best_estimator_

clf_rf_rand.fit(xtrain, ytrain)
pred_rf_rand = clf_rf_rand.predict(xtest)
pred = pred_rf_rand
pred
len(pred)
test_data.file
submission=pd.DataFrame({'file':test_data.file,'subspecies':pred})

submission.set_index(submission.columns[0],inplace= True)

# Write code here
submission.to_csv('Submission_RF_ver_1.csv')
