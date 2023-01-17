# Library

import numpy as np # linear algebra

import pandas as pd # data processing

import matplotlib as mpl # data visualization

import matplotlib.pyplot as plt # data visualization

from sklearn.linear_model import LogisticRegression # logistic regression

from sklearn.linear_model import RidgeClassifier # ridge classifier

from sklearn.ensemble import RandomForestClassifier # random forest

from sklearn.ensemble import VotingClassifier # Ensemble model

from sklearn.model_selection import StratifiedKFold # Stratified K-Fold for cross-validation

from sklearn.model_selection import KFold # Stratified K-Fold for cross-validation

from sklearn.model_selection import GridSearchCV # to perform Grid search cross-validation 

from sklearn.model_selection import cross_val_score # to calculate cross-validation score

from scipy.ndimage.interpolation import shift # shift image for data augmentation

from sklearn import metrics

import random

import time

from datetime import datetime

train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")
train.shape, test.shape

# There are 42k training data and 28k test data
# Get sense in target value



train['label'].value_counts()

# Each label has more than 3.7k samples

# Smallest sample is digit 5

# Biggest samples is digit 1

# We will make sure do sratified cross-validation since # of samples per label different
# Define Functions



# Series to Array

def series_to_array(series):

    data = np.array(series)

    return data



# Plot Digit

def plot_digit(series):

    data = series_to_array(series)

    image = data.reshape(28, 28)

    plt.imshow(image, cmap = mpl.cm.binary,

               interpolation="nearest")

    plt.axis("off")

    

# Plot Multiple Digits

def plot_digits(instances, images_per_row=10, **options):

    size = 28

    images_per_row = min(len(instances), images_per_row)

    images = [instance.reshape(size,size) for instance in instances]

    n_rows = (len(instances) - 1) // images_per_row + 1

    row_images = []

    n_empty = n_rows * images_per_row - len(instances)

    images.append(np.zeros((size, size * n_empty)))

    for row in range(n_rows):

        rimages = images[row * images_per_row : (row + 1) * images_per_row]

        row_images.append(np.concatenate(rimages, axis=1))

    image = np.concatenate(row_images, axis=0)

    plt.imshow(image, cmap = mpl.cm.binary, **options)

    plt.axis("off")
# Plot one random digit

plot_digit(train.iloc[37694,1:])
# Plot multiple random digits

plt.figure(figsize=(9,9))

example_images = np.r_[test[:12000:600], test[13000:30600:600], test[30600:60000:590]]

plot_digits(example_images, images_per_row=10)

plt.show()
# Prepare Train and Test dataset

X_train = pd.DataFrame(train.iloc[:, 1:])

y_train = pd.DataFrame(train.iloc[:, 0]) # temporarily convert to DataFrame for randomized process

X_test = test



X_train.shape, y_train.shape, X_test.shape
# Randomized Train dataset

random.seed(51)

shuffle_index = np.arange(42000)

random.shuffle(shuffle_index)

X_train, y_train = X_train.iloc[shuffle_index], y_train.iloc[shuffle_index]



y_train = y_train.iloc[:,0] # Revert back to series



# Dataset ready to process
# Stratified k-fold being used in order to get stratified split

skfolds = StratifiedKFold(n_splits=2, random_state=41)



# K-fold cross-validation

kfolds = KFold(n_splits=2, random_state=41)



# For simplicity in measuring performance

def cv_acc(model, X=X_train, y=y_train):

    acc = cross_val_score(model, X, y_train, cv=3, scoring = "accuracy")

    return (acc)
# Multiclass Classifier

# I will be using three Multiclass Classifier i.e. (1) Logistic Regression, (2) Ridge Classifier, and (3) Random Forest

# Define Multiclass Classifier



logistic_multi = LogisticRegression(multi_class='multinomial', solver = 'saga', penalty = 'l1') # define as multiclass classifier

ridge_multi = RidgeClassifier(alpha = 0.7)

rf_multi = RandomForestClassifier(n_estimators = 10)
# Do Cross Validation for MultiClass Classifier

score_log_multi = cv_acc(logistic_multi , X_train)

score_ridge_multi = cv_acc(ridge_multi , X_train)

score_rf_multi = cv_acc(rf_multi , X_train)

print("Finished!\n", datetime.now(), )
# Print Accuracy estimated using CV

print("Logistic MultiClass Accuracy: {:.4f} (with STD {:.4f})\n".format(score_log_multi.mean(), score_log_multi.std()))

print("Ridge MultiClass Accuracy: {:.4f} (with STD {:.4f})\n".format(score_ridge_multi.mean(), score_ridge_multi.std()))

print("Random Forest MultiClass Accuracy: {:.4f} (with STD {:.4f})\n".format(score_rf_multi.mean(), score_rf_multi.std()))



# Random Forest & Logistic Multiclass give > 90% accuracy

# Let's compare it with Ensemble Binary Classifier which consist of multiple Random Forest binary classifier
# Ensemble Binary Classifier



# Let's create target for binary classification

y_train_0 = (y_train == 0)

y_train_1 = (y_train == 1)

y_train_2 = (y_train == 2)

y_train_3 = (y_train == 3)

y_train_4 = (y_train == 4)

y_train_5 = (y_train == 5)

y_train_6 = (y_train == 6)

y_train_7 = (y_train == 7)

y_train_8 = (y_train == 8)

y_train_9 = (y_train == 9) # Now we have separated target for each digit



# Lets define Binary Classifier for each digit

rf_0 = RandomForestClassifier(n_estimators = 10)

rf_1 = RandomForestClassifier(n_estimators = 10)

rf_2 = RandomForestClassifier(n_estimators = 10)

rf_3 = RandomForestClassifier(n_estimators = 10)

rf_4 = RandomForestClassifier(n_estimators = 10)

rf_5 = RandomForestClassifier(n_estimators = 10)

rf_6 = RandomForestClassifier(n_estimators = 10)

rf_7 = RandomForestClassifier(n_estimators = 10)

rf_8 = RandomForestClassifier(n_estimators = 10)

rf_9 = RandomForestClassifier(n_estimators = 10)
# Estimate accuracy of each model using CV

score_rf_0 = cv_acc(rf_0 , X_train, y_train_0)

score_rf_1 = cv_acc(rf_1 , X_train, y_train_1)

score_rf_2 = cv_acc(rf_2 , X_train, y_train_2)

score_rf_3 = cv_acc(rf_3 , X_train, y_train_3)

score_rf_4 = cv_acc(rf_4 , X_train, y_train_4)

score_rf_5 = cv_acc(rf_5 , X_train, y_train_5)

score_rf_6 = cv_acc(rf_6 , X_train, y_train_6)

score_rf_7 = cv_acc(rf_7 , X_train, y_train_7)

score_rf_8 = cv_acc(rf_8 , X_train, y_train_8)

score_rf_9 = cv_acc(rf_9 , X_train, y_train_9)

# Print Accuracy estimated using CV

print("Random Forest 0: {:.4f} (with STD {:.4f})\n".format(score_rf_0.mean(), score_rf_0.std()))

print("Random Forest 1: {:.4f} (with STD {:.4f})\n".format(score_rf_1.mean(), score_rf_1.std()))

print("Random Forest 2: {:.4f} (with STD {:.4f})\n".format(score_rf_2.mean(), score_rf_2.std()))

print("Random Forest 3: {:.4f} (with STD {:.4f})\n".format(score_rf_3.mean(), score_rf_3.std()))

print("Random Forest 4: {:.4f} (with STD {:.4f})\n".format(score_rf_4.mean(), score_rf_4.std()))

print("Random Forest 5: {:.4f} (with STD {:.4f})\n".format(score_rf_5.mean(), score_rf_5.std()))

print("Random Forest 6: {:.4f} (with STD {:.4f})\n".format(score_rf_6.mean(), score_rf_6.std()))

print("Random Forest 7: {:.4f} (with STD {:.4f})\n".format(score_rf_7.mean(), score_rf_7.std()))

print("Random Forest 8: {:.4f} (with STD {:.4f})\n".format(score_rf_8.mean(), score_rf_8.std()))

print("Random Forest 9: {:.4f} (with STD {:.4f})\n".format(score_rf_9.mean(), score_rf_9.std()))



# Great! Estimators have roughly 93% accuracy
# Fit binary classifier

rf_0.fit(X_train, y_train_0)

rf_1.fit(X_train, y_train_1)

rf_2.fit(X_train, y_train_2)

rf_3.fit(X_train, y_train_3)

rf_4.fit(X_train, y_train_4)

rf_5.fit(X_train, y_train_5)

rf_6.fit(X_train, y_train_6)

rf_7.fit(X_train, y_train_7)

rf_8.fit(X_train, y_train_8)

rf_9.fit(X_train, y_train_9)
# Vote Classifier

# Using vote classifier as method to combine binary classifier

# Soft Voting will be used as we want the average class probabilities from each estimator



binary_classifiers = [('rf_0', rf_0), ('rf_1', rf_1), ('rf_2', rf_2), ('rf_3', rf_3), ('rf_4', rf_4), 

                      ('rf_5', rf_5), ('rf_6', rf_6), ('rf_7', rf_7), ('rf_8', rf_8), ('rf_9', rf_9)]



voting_cl = VotingClassifier( estimators = binary_classifiers, voting = 'soft')

# Estimate accuracy of Voting Classifier

score_voting = cv_acc(voting_cl , X_train, y_train)

# Print estimated accuracy of Voting Classifier

print("Voting Classifier: {:.4f} (with STD {:.4f})\n".format(score_voting.mean(), score_voting.std()))



# Significant improvement! We got estimation of 96.3% accuracy!
# Lets use this Voting Classifier for submission



voting_cl.fit(X_train, y_train)

y_predicted = voting_cl.predict(X_test)
# Submission

submission1 = pd.read_csv("../input/sample_submission.csv")

submission1.iloc[:,1] = (y_predicted)

submission1.to_csv("submission1.csv", index=False)
def plot_digit(series):

    data = series_to_array(series)

    image = data.reshape(28, 28)

    plt.imshow(image, cmap = mpl.cm.binary,

               interpolation="nearest")

    plt.axis("off")



def shift_image(series, dx, dy):

    data = series_to_array(series)

    image = data.reshape((28, 28))

    shifted_image = shift(image, [dy, dx], cval=0, mode="constant")

    return shifted_image.reshape([-1])

image = X_train.iloc[1050]

shifted_image_down = shift_image(image, 0, 5)

shifted_image_left = shift_image(image, -5, 0)



plt.figure(figsize=(12,3))

plt.subplot(131)

plt.title("Original", fontsize=14)

plt.imshow(series_to_array(image).reshape(28, 28), interpolation="nearest", cmap="Greys")

plt.subplot(132)

plt.title("Shifted down", fontsize=14)

plt.imshow(series_to_array(shifted_image_down).reshape(28, 28), interpolation="nearest", cmap="Greys")

plt.subplot(133)

plt.title("Shifted left", fontsize=14)

plt.imshow(series_to_array(shifted_image_left).reshape(28, 28), interpolation="nearest", cmap="Greys")

plt.show()
# Create augmented data



def shift_image2(image, dx, dy):

    image = image.reshape((28, 28))

    shifted_image = shift(image, [dy, dx], cval=0, mode="constant")

    return shifted_image.reshape([-1])



X_train_array = X_train.values # Convert DataFrame into Array



X_train_augmented = [ images for images in X_train_array]

y_train_augmented = [ labels for labels in y_train]



# We will create shifted left and right images by 1 pixel

for dx, dy in ((1, 0), (-1, 0)):

    for images, labels in zip(X_train_array, y_train):

        X_train_augmented.append(shift_image2(images, dx, dy))

        y_train_augmented.append(labels)



X_train_augmented = np.array(X_train_augmented)

y_train_augmented = np.array(y_train_augmented)



shuffle_idx = np.random.permutation(len(X_train_augmented))

X_train_augmented = X_train_augmented[shuffle_idx]

y_train_augmented = y_train_augmented[shuffle_idx]
X_train_augmented.shape, y_train_augmented.shape

# We got 126000 samples!

# Lets try to estimate accuracy using augmented dataset
# Let's create target for binary classification

y_train_0_agm = (y_train_augmented == 0)

y_train_1_agm = (y_train_augmented == 1)

y_train_2_agm = (y_train_augmented == 2)

y_train_3_agm = (y_train_augmented == 3)

y_train_4_agm = (y_train_augmented == 4)

y_train_5_agm = (y_train_augmented == 5)

y_train_6_agm = (y_train_augmented == 6)

y_train_7_agm = (y_train_augmented == 7)

y_train_8_agm = (y_train_augmented == 8)

y_train_9_agm = (y_train_augmented == 9) # Now we have separated target for each digit



# Create Binary Classifier

rf_0_agm = RandomForestClassifier(n_estimators = 10)

rf_1_agm = RandomForestClassifier(n_estimators = 10)

rf_2_agm = RandomForestClassifier(n_estimators = 10)

rf_3_agm = RandomForestClassifier(n_estimators = 10)

rf_4_agm = RandomForestClassifier(n_estimators = 10)

rf_5_agm = RandomForestClassifier(n_estimators = 10)

rf_6_agm = RandomForestClassifier(n_estimators = 10)

rf_7_agm = RandomForestClassifier(n_estimators = 10)

rf_8_agm = RandomForestClassifier(n_estimators = 10)

rf_9_agm = RandomForestClassifier(n_estimators = 10)



# Fit Binary Classifier

print("Started! We are now building binary classifiers!\n", datetime.now(), )

rf_0_agm.fit(X_train_augmented, y_train_0_agm)

rf_1_agm.fit(X_train_augmented, y_train_1_agm)

rf_2_agm.fit(X_train_augmented, y_train_2_agm)

rf_3_agm.fit(X_train_augmented, y_train_3_agm)

rf_4_agm.fit(X_train_augmented, y_train_4_agm)

rf_5_agm.fit(X_train_augmented, y_train_5_agm)

rf_6_agm.fit(X_train_augmented, y_train_6_agm)

rf_7_agm.fit(X_train_augmented, y_train_7_agm)

rf_8_agm.fit(X_train_augmented, y_train_8_agm)

rf_9_agm.fit(X_train_augmented, y_train_9_agm)

print("Finished! binary classifiers created!\n", datetime.now(), )
# Vote Classifier

binary_classifiers_2 = [('rf_0_agm', rf_0_agm), ('rf_1_agm', rf_1_agm), ('rf_2_agm', rf_2_agm), ('rf_3_agm', rf_3_agm), ('rf_4_agm', rf_4_agm), 

                      ('rf_5_agm', rf_5_agm), ('rf_6_agm', rf_6_agm), ('rf_7_agm', rf_7_agm), ('rf_8_agm', rf_8_agm), ('rf_9_agm', rf_9_agm)]



voting_cl_agm= VotingClassifier( estimators = binary_classifiers_2, voting = 'soft')
# Estimate accuracy of Voting Classifier

score_voting_agm = cross_val_score(voting_cl_agm, X_train_augmented, y_train_augmented, cv=2, scoring = "accuracy")

print("Finished! cross-validation complete!\n", datetime.now(), )
# Print estimated accuracy of Voting Classifier

print("Voting Classifier - Augmented: {:.4f} (with STD {:.4f})\n".format(score_voting_agm.mean(), score_voting_agm.std()))



# We got 96.74% Accuracy! Our model slightly improved with augmented data.
# Fit Voting Classifier with augmented data

voting_cl_agm.fit(X_train_augmented, y_train_augmented)

y_predicted_2 = voting_cl_agm.predict(X_test)
# Submission 2

submission2 = pd.read_csv("../input/sample_submission.csv")

submission2.iloc[:,1] = (y_predicted_2)

submission2.to_csv("submission2.csv", index=False)
# Turns out shifting our images to the left and right works to improve accuracy

# We will try to create more shifted left and right by 2 pixel



# We will create shifted left and right images by 2 pixel



X_train_augmented_2 = [ images for images in X_train_array]

y_train_augmented_2 = [ labels for labels in y_train]



for dx, dy in ((2, 0), (-2, 0), (1,0), (-1,0)):

    for images, labels in zip(X_train_array, y_train):

        X_train_augmented_2.append(shift_image2(images, dx, dy))

        y_train_augmented_2.append(labels)



X_train_augmented_2 = np.array(X_train_augmented_2)

y_train_augmented_2 = np.array(y_train_augmented_2)



shuffle_idx = np.random.permutation(len(X_train_augmented_2))

X_train_augmented_2 = X_train_augmented_2[shuffle_idx]

y_train_augmented_2 = y_train_augmented_2[shuffle_idx]



X_train_augmented_2.shape, y_train_augmented_2.shape

# Now we have 210000 rows data
# Let's create target for binary classification

y_train_0_agm_2 = (y_train_augmented_2 == 0)

y_train_1_agm_2 = (y_train_augmented_2 == 1)

y_train_2_agm_2 = (y_train_augmented_2 == 2)

y_train_3_agm_2 = (y_train_augmented_2 == 3)

y_train_4_agm_2 = (y_train_augmented_2 == 4)

y_train_5_agm_2 = (y_train_augmented_2 == 5)

y_train_6_agm_2 = (y_train_augmented_2 == 6)

y_train_7_agm_2 = (y_train_augmented_2 == 7)

y_train_8_agm_2 = (y_train_augmented_2 == 8)

y_train_9_agm_2 = (y_train_augmented_2 == 9) # Now we have separated target for each digit



# Create Binary Classifier

rf_0_agm_2 = RandomForestClassifier(n_estimators = 10)

rf_1_agm_2 = RandomForestClassifier(n_estimators = 10)

rf_2_agm_2 = RandomForestClassifier(n_estimators = 10)

rf_3_agm_2 = RandomForestClassifier(n_estimators = 10)

rf_4_agm_2 = RandomForestClassifier(n_estimators = 10)

rf_5_agm_2 = RandomForestClassifier(n_estimators = 10)

rf_6_agm_2 = RandomForestClassifier(n_estimators = 10)

rf_7_agm_2 = RandomForestClassifier(n_estimators = 10)

rf_8_agm_2 = RandomForestClassifier(n_estimators = 10)

rf_9_agm_2 = RandomForestClassifier(n_estimators = 10)



# Fit Binary Classifier

print("Started! We are now building binary classifiers!\n", datetime.now(), )

rf_0_agm_2.fit(X_train_augmented_2, y_train_0_agm_2)

rf_1_agm_2.fit(X_train_augmented_2, y_train_1_agm_2)

rf_2_agm_2.fit(X_train_augmented_2, y_train_2_agm_2)

rf_3_agm_2.fit(X_train_augmented_2, y_train_3_agm_2)

rf_4_agm_2.fit(X_train_augmented_2, y_train_4_agm_2)

rf_5_agm_2.fit(X_train_augmented_2, y_train_5_agm_2)

rf_6_agm_2.fit(X_train_augmented_2, y_train_6_agm_2)

rf_7_agm_2.fit(X_train_augmented_2, y_train_7_agm_2)

rf_8_agm_2.fit(X_train_augmented_2, y_train_8_agm_2)

rf_9_agm_2.fit(X_train_augmented_2, y_train_9_agm_2)

print("Finished! binary classifiers created!\n", datetime.now(), )
# Vote Classifier

binary_classifiers_3 = [('rf_0_agm_2', rf_0_agm_2), ('rf_1_agm_2', rf_1_agm_2), ('rf_2_agm_2', rf_2_agm_2), ('rf_3_agm_2', rf_3_agm_2), ('rf_4_agm_2', rf_4_agm_2), 

                      ('rf_5_agm_2', rf_5_agm_2), ('rf_6_agm_2', rf_6_agm_2), ('rf_7_agm_2', rf_7_agm_2), ('rf_8_agm_2', rf_8_agm_2), ('rf_9_agm_2', rf_9_agm_2)]



voting_cl_agm_2= VotingClassifier( estimators = binary_classifiers_3, voting = 'soft')



# Estimate accuracy of Voting Classifier

print("Started! cross-validation start!\n", datetime.now(), )

score_voting_agm_2 = cross_val_score(voting_cl_agm_2, X_train_augmented_2, y_train_augmented_2, cv=2, scoring = "accuracy")

print("Finished! cross-validation complete!\n", datetime.now(), )
# Print estimated accuracy of Voting Classifier

print("Voting Classifier - Augmented 2: {:.4f} (with STD {:.4f})\n".format(score_voting_agm_2.mean(), score_voting_agm_2.std()))



# We got 96.78% Accuracy! 0.02% improvement. Turns out at this level, additional augmented data not much improve our performance
# Fit Voting Classifier with augmented data - 2

voting_cl_agm_2.fit(X_train_augmented_2, y_train_augmented_2)

y_predicted_3 = voting_cl_agm_2.predict(X_test)



# Submission 3

submission3 = pd.read_csv("../input/sample_submission.csv")

submission3.iloc[:,1] = (y_predicted_3)

submission3.to_csv("submission3.csv", index=False)