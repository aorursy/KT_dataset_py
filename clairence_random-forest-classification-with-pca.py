# Import dependencies

import numpy as np

import pandas as pd

import os

import random

import time



# ML imports

from sklearn.model_selection import KFold, train_test_split, cross_val_score, cross_val_predict, GridSearchCV 

from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, classification_report

from sklearn.ensemble import RandomForestClassifier

from sklearn.decomposition import PCA, IncrementalPCA



# Visualizations

import matplotlib.pyplot as plt  # static plotting

import seaborn as sns  # pretty plotting, including heat map

%matplotlib inline
# Initialize process time list for code timing across entire program

process_time = []
# Import train and test datasets



# The training data set, (train.csv), has 785 columns. The first column, called "label", is the 

# digit that was drawn by the user. The rest of the columns contain the pixel-values of the associated image.

sample_submission = pd.read_csv("../input/digit-recognizer/sample_submission.csv")

test = pd.read_csv("../input/digit-recognizer/test.csv")

train = pd.read_csv("../input/digit-recognizer/train.csv")
# 42,000 images

# Each image has 784 features (each image is 28 x 28 pixels) and 1 label feature (785 total)

# Each feature represents one pixel's intensity from 0 (white) to 255 (black)

train.shape
# 28,000 images, contains 784 features and no label

test.shape
# Inspect class balances for train set, seems relatively balanced

print(train['label'].value_counts(ascending=False))

print('--------------------')

print(train['label'].value_counts(normalize=True))
# Plot total distribution of labels in Kaggle train set

mn_plt_total = sns.countplot(train['label'], palette="muted").set_title('Total Digit Distribution')
# Save the labels to a Pandas series target

y = train['label']



# Drop the label feature

X = train.drop("label",axis=1)
# Confirm that all 256 values between the min-max of 0-255 exist in the train set

len(np.unique(X))
# View images

images_to_plot = 9

random_indices = random.sample(range(42000), images_to_plot)



sample_images = X.loc[random_indices, :]

sample_labels = y.loc[random_indices]
# Plot examples

plt.clf()

plt.style.use('seaborn-muted')



fig, axes = plt.subplots(3,3, 

                         figsize=(5,5),

                         sharex=True, sharey=True,

                         subplot_kw=dict(adjustable='box', aspect='equal')) #https://stackoverflow.com/q/44703433/1870832



for i in range(images_to_plot):

    

    # axes (subplot) objects are stored in 2d array, accessed with axes[row,col]

    subplot_row = i//3 

    subplot_col = i%3  

    ax = axes[subplot_row, subplot_col]



    # plot image on subplot

    plottable_image = np.reshape(sample_images.iloc[i,:].values, (28,28))

    ax.imshow(plottable_image, cmap='gray_r')

    

    ax.set_title('Digit Label: {}'.format(sample_labels.iloc[i]))

    ax.set_xbound([0,28])



plt.tight_layout()

plt.show()
# Assign classifier

# Use default criterion (gini)

cls_rf = RandomForestClassifier(random_state=1, n_jobs=-1, n_estimators=90)
# Split train and test set 90/10

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.20, random_state=1)
# Check split

print(X_train.shape)

print(y_train.shape)

print(X_test.shape)

print(y_test.shape)
# Evaluate distribution of labels across train set

mn_plt_trn = sns.countplot(y_train, palette="muted").set_title('Train Digit')
# Evaluate distribution of labels across test set

mn_plt_trn = sns.countplot(y_test, palette="muted").set_title('Test Digit')
# Fit model and run on test set with timing

start_time = time.time()

cls_rf.fit(X_train, y_train)

print('Accuracy: ', cls_rf.score(X_test, y_test))



# Timing

elapsed_time = time.time() - start_time

print('Formatted time: ', time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))

print('Time in seconds: ', elapsed_time)

process_time.append(elapsed_time)
# cls_rf = RandomForestClassifier()



param_grid = {

    'bootstrap': [True],

    'n_estimators': [100,150,200,250,300,400]

}



grid_search_rf = GridSearchCV(estimator=cls_rf, param_grid=param_grid, cv=3, 

                              return_train_score=True, n_jobs=-1, verbose=2)



grid_search_rf.fit(X_train, y_train)
# Display parameter recommendations

print('Test set score: ', grid_search_rf.score(X_test, y_test))

print('Best parameters: ', grid_search_rf.best_params_)

print('Best cross-validation score: ', grid_search_rf.best_score_)
# Adjust model parameters, not using max depth

cls_rf = RandomForestClassifier(random_state=1, n_jobs=-1, n_estimators=300)



# Original

# cls_rf = RandomForestClassifier(random_state=1, n_jobs=-1, n_estimators=90)
# Fit the tuned parameters with timing

start_time = time.time()

cls_rf.fit(X_train, y_train)



# Results with tuning

print('Accuracy: ', cls_rf.score(X_test, y_test))



# Timing

elapsed_time = time.time() - start_time

print('Formatted time: ', time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))

print('Time in seconds: ', elapsed_time)

process_time.append(elapsed_time)
# F1 score

pred = cls_rf.predict(X_test)

print('F1 accuracy: ', f1_score(pred, y_test, average='macro'))
# Print classification report

target_names = ['class 0', 'class 1', 'class 2', 'class 3', 'class 4', 'class 5', 'class 6', 'class 7',

                   'class 8', 'class 9']

print(classification_report(y_test, pred, target_names=target_names))
# Plot confusion matrix of actual versus predicted labels

rf_cm = confusion_matrix(y_test, pred)

rf_cm_plt=sns.heatmap(rf_cm.T, square=True, annot=True, fmt='d', cbar=False, cmap="Blues")

plt.xlabel('Actual label')

plt.ylabel('Predicted label')

plt.title("Valid");
# Submission code

testData = pd.read_csv("test.csv")

start_time = time.time()

pred = cls_rf.predict(testData)

elapsed_time = time.time() - start_time

print('Formatted time: ', time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))

print('Time in seconds: ', elapsed_time)

process_time.append(elapsed_time)



# # Create Dataframe

# data = pred

# df_1 = pd.DataFrame(pred)

# df_1['ImageID'] = df_1.index + 1

# df_1.columns = ['Label', 'ImageID']

# submission = df_1[['ImageID', 'Label']]



# # Output to csv

# submission.to_csv('Boetticher_RF_predictions.csv',header=True, index=False)
# Combine training and test set for PCA

x = np.concatenate((X_train, X_test), axis=0)
# Fit model using PCA, generating principal components that represent 95 percent of the variability in 

# the explanatory features

start_time = time.time()

pca = PCA(.95)

pca.fit(x)

totimages = pca.transform(x)

# pca.n_components_



# Timing

elapsed_time = time.time() - start_time

print('Formatted time: ', time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))

print('Time in seconds: ', elapsed_time)

process_time.append(elapsed_time)
# Output number of principal components explaining 95% of variability in the explanatory features: 154

print('Principal components count: ', pca.n_components_)
# Explained variance ratio for 154 PCs

# 10% of the train dataset's variance lies along the first PC

# 7% along the second

# 6% along the third, etc.

pca.explained_variance_ratio_
#Explained variance plot

plt.plot(np.cumsum(pca.explained_variance_ratio_))

plt.xlabel('number of components')

plt.ylabel('cumulative explained variance')

plt.show()
# Reform the training and testing images

# Training: up to 33,600

# Testing: 33,600 to 42,000

trainimages = totimages[0:33600, :]

testimages = totimages[33600:42000, :]
# Convert to integers

trainimages = trainimages.astype(int)

testimages = testimages.astype(int)
# Time component identification

start_time = time.time()



n_batches = 100

inc_pca = IncrementalPCA(n_components=154)

for X_batch in np.array_split(X_train, n_batches):

    inc_pca.partial_fit(X_batch)



X_reduced = inc_pca.transform(X_train)



# Timing

elapsed_time = time.time() - start_time

print('Formatted time: ', time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))

print('Time in seconds: ', elapsed_time)

process_time.append(elapsed_time)
# Explained variance ratio for 154 PCs with incremental PCA

# 10% of the train dataset's variance lies along the first PC

# 7% along the second

# 6% along the third, etc.

# Not drastically different than PCA and took longer

inc_pca.explained_variance_ratio_
# Fit the model

start_time = time.time()

cls_rf2 = RandomForestClassifier(random_state=1, n_jobs=-1, n_estimators=300)

cls_rf2.fit(trainimages, y_train.values.ravel())

print('Accuracy: ', cls_rf2.score(testimages, y_test))



# Timing

elapsed_time = time.time() - start_time

print('Formatted time: ', time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))

print('Time in seconds: ', elapsed_time)

process_time.append(elapsed_time)
# Predict a second time with principal components

pred2 = cls_rf2.predict(testimages)

print('F1 accuracy: ', f1_score(pred2, y_test, average='macro'))
# Plot confusion matrix of actual versus predicted labels

rf_cm = confusion_matrix(y_test, pred2)

rf_cm_plt=sns.heatmap(rf_cm.T, square=True, annot=True, fmt='d', cbar=False, cmap="Blues")

plt.xlabel('Actual label')

plt.ylabel('Predicted label')

plt.title("Valid");
# Submission code with timing

x_test = pd.read_csv("test.csv")

start_time = time.time()

pred_PCA = cls_rf2.predict(pca.transform(x_test))

elapsed_time = time.time() - start_time

print('Formatted time: ', time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))

print('Time in seconds: ', elapsed_time)

process_time.append(elapsed_time)



# # Create Dataframe

# data = pred_PCA

# df_1 = pd.DataFrame(pred_PCA)

# df_1['ImageID'] = df_1.index + 1

# df_1.columns = ['Label', 'ImageID']

# submission = df_1[['ImageID', 'Label']]



# # Output to csv

# submission.to_csv('Boetticher_RF__PCA_predictions_2.csv',header=True, index=False)
# Final process_time addition for entire program

# Find sum of elements in process_time list 

total = 0



# Iterate over each element in process_time list and add them in variable total 

for ele in range(0, len(process_time)): 

    total = total + process_time[ele] 

    

# Conversion

def convert(seconds): 

    min, sec = divmod(seconds, 60) 

    hour, min = divmod(min, 60) 

    return "%d:%02d:%02d" % (hour, min, sec) 

      

# Time output

n = total

convert(n)



# Results

print("Total time of model fitting and evalation for study (in seconds): ", total) 

print('Formatted time: ', time.strftime("%H:%M:%S", time.gmtime(total)))