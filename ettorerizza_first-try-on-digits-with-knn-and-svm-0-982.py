# from scipi

import joblib 



import matplotlib as mpl

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd



%matplotlib inline

plt.rcParams['figure.figsize'] = [10, 10]



#  Data Modelling Libraries



from sklearn import preprocessing

from sklearn.decomposition import PCA

from sklearn.ensemble import (VotingClassifier)

from sklearn.metrics import (accuracy_score, classification,

                             classification_report, confusion_matrix)



from sklearn.model_selection import train_test_split



from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

train_df = pd.read_csv("../input/train.csv",header=0)

submit_test_df = pd.read_csv("../input/test.csv",header=0)
train_data = train_df.values



X_train, X_test, y_train, y_test = train_test_split(train_data[:, 1:],

                                                    train_data[:, 0],

                                                    test_size=0.2,

                                                    random_state=42,

                                                    shuffle=False)



# ces versions dataframe sont créées juste pour éviter des bugs lors de certains concaténations

X_train_df = pd.DataFrame(data=X_train)

y_train_df = pd.Series(data=y_train)



X_test_df = pd.DataFrame(data=X_test)

y_test_df = pd.Series(data=y_test)



digits_images_df = X_train_df



digits_labels_df = y_train_df



digits_images = X_train



digits_labels = y_train
def plot_digit(data):

    image = data.reshape(28, 28)

    plt.imshow(image, cmap = mpl.cm.binary,

               interpolation="nearest")

    plt.axis("off")
from scipy.ndimage.interpolation import shift

def shift_digit(digit_array, dx, dy, new=0):

    return shift(digit_array.reshape(28, 28), [dy, dx], cval=new).reshape(784)







some_digit = X_train[0]

some_digit_image = some_digit.reshape(28, 28)



plot_digit(shift_digit(some_digit, 5, 1, new=100))
from scipy.ndimage.interpolation import shift



def shift_digit(digit_array, dx, dy, new=0):

    return shift(digit_array.reshape(28, 28), [dy, dx], cval=new).reshape(784)



X_train_expanded = [X_train]

y_train_expanded = [y_train]

for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):

    shifted_images = np.apply_along_axis(shift_digit, axis=1, arr=X_train, dx=dx, dy=dy)

    X_train_expanded.append(shifted_images)

    y_train_expanded.append(y_train)



X_train_expanded = np.concatenate(X_train_expanded)

y_train_expanded = np.concatenate(y_train_expanded)

X_train_expanded.shape, y_train_expanded.shape
print("Train", X_train.shape)

print("Test", X_test.shape)

print("Train Expanded", X_train_expanded.shape)

print("Train Expanded Ratio", X_train_expanded.shape[0] / X_train.shape[0])
pca = PCA(n_components=50, random_state=42)



X_train_pca = pca.fit_transform(X_train_expanded)



X_test_pca = pca.transform(X_test)
knn_best = KNeighborsClassifier(algorithm='auto',

                                leaf_size=30,

                                metric='minkowski',

                                metric_params=None,

                                n_jobs=None,

                                n_neighbors=5,

                                p=2,

                                weights='uniform')







knn_best.fit(X_train_pca, y_train_expanded)



y_knn_pred = knn_best.predict(X_test_pca)



accuracy_score(y_test, y_knn_pred)
svm_best = SVC(C=10, cache_size=200, class_weight=None, coef0=0.0,

    decision_function_shape='ovr', degree=3, gamma=0.01, kernel='poly',

    max_iter=-1, probability=True, random_state=42, shrinking=True, tol=0.001,

    verbose=True)
svm_best.fit(X_train_pca, y_train_expanded)



y_svm_pred = svm_best.predict(X_test_pca)



accuracy_score(y_test, y_svm_pred)
VotingPredictor = VotingClassifier(estimators=[('KNN', knn_best),

                                               ('SVM', svm_best)],

                                   voting='soft',

                                   n_jobs=-1)



VotingPredictor = VotingPredictor.fit(X_train_pca, y_train_expanded)



voting_y_pred = VotingPredictor.predict(X_test_pca)



#print(scores)



print("Accuracy of Voting:", accuracy_score(y_test, voting_y_pred))
# save the model to disk

joblib.dump(VotingPredictor, "voting.sav")



# load the model from disk

#loaded_model = joblib.load(filename)
submission = pca.transform(submit_test_df.values)



pred_voting = VotingPredictor.predict(submission)
sam = pd.read_csv("../sample_submission.csv")



def write_prediction(prediction, name):

    ImageId = np.array(sam['ImageId']).astype(int)

    solution = pd.DataFrame(prediction, ImageId, columns = ['Label'])

    solution.to_csv(name, index_label = ['ImageId'])

    

write_prediction(pred_voting, "samdigit_voting.csv")