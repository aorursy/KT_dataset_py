import os
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
#Reading the data

#Note that the data is in a flat format, where one row represents a whole image, with the label.
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
#Separating the digit label form the train data, and removing it from the original dataframe
train_y = train['label']
train_X = train.drop(columns=['label'])
del(train) #because we need to save some space
#Converting the 1-Dimensional image data into 2D data (will be required for visualization purposes)
train_X = np.array(train_X).reshape(len(train_X), -1)
test = np.array(test).reshape(len(test), -1)
#Initialize a scikit-learn MLP CLassifier with custom parameters
clf = MLPClassifier(alpha=0.00001, momentum=0.9, beta_1=0.4, beta_2=0.8, max_iter=500, verbose=True,tol=0.00001, hidden_layer_sizes=(784, 150, 10,))
#Starting the training phase
#The scores can be improved (or reduced)based on the parameters of the classifier
clf.fit(X=train_X, y=train_y)
#Getting the predicted labels for each image in the test set
predicted = clf.predict(test)
#Creating the final submission-ready dataframe with ImageId column, and the predicted results.
image_ids = range(1, len(test)+1)
submission_df = pd.DataFrame({'ImageId': image_ids, 'Label': predicted})
#Saving the final dataframe as csv, which can be submitted now.
submission_df.to_csv(path_or_buf='submission.csv', index=False)
