import pandas as pd
import cv2

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
# Read data from csv
train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')
# Separate target and features
train_target = train_df['label']
train_features = train_df.drop(labels='label', axis='columns')

# Normalize each pixel from [0,1]
train_features = round(train_features / 255, 2)
test_df = round(test_df / 255, 2)
# Perform PCA
pca = PCA(n_components=25)
pca_features = pd.DataFrame(pca.fit_transform(train_features))

plt.bar(range(pca.n_components_), pca.explained_variance_ratio_)
# Split train and test
Xtrain, Xtest, Ytrain, Ytest = train_test_split(pca_features, train_target, test_size=0.2)

# Fit data with multiple classifiers, print scores for user
rfc = RandomForestClassifier(n_estimators=125, n_jobs=4, verbose=1)
svc = SVC(gamma='scale')
knn = KNeighborsClassifier(weights='distance', n_jobs=4)

rfc.fit(Xtrain, Ytrain)
svc.fit(Xtrain, Ytrain)
knn.fit(Xtrain, Ytrain)

print('RFC: '+ str(rfc.score(Xtest, Ytest)))
print('SVC: ' + str(svc.score(Xtest, Ytest)))
print('KNN: ' + str(knn.score(Xtest, Ytest)))
# Make predictions using SVC since it performed the best
test_pca = pd.DataFrame(pca.transform(test_df))
predictions = svc.predict(test_pca)

# Write predictions to output csv
pred_df = pd.DataFrame({'ImageId': test_pca.index + 1,
                        'Label': predictions})
pred_df.to_csv('predictions.csv', index=False)
print("Done writing to csv")