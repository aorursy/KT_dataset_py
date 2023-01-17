# Import libraries



import numpy as np

import pandas as pd

from sklearn.preprocessing import StandardScaler

from sklearn import svm

from sklearn.neural_network import MLPClassifier
# Import data



train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')
# Split training data into features and labels dataframes



train_features = train_df.drop(['label'],1)

train_labels = train_df['label']
# Convert to np array and convert int to float



X_train_array = np.array(train_features)

X_train_array = X_train_array.astype('float')



y_train_array = np.array(train_labels)

y_train_array = y_train_array.astype('float')





X_test_array = np.array(test_df)

X_test_array = X_test_array.astype('float')

# Scale the data



scaler = StandardScaler().fit(X_train_array)

X_train = scaler.transform(X_train_array)

X_test = scaler.transform(X_test_array)



# The labels are not scaled



y_train = y_train_array
# Choose classifier



#clf = svm.SVC(kernel='linear')



clf = MLPClassifier(solver='lbfgs', hidden_layer_sizes=(600), random_state=1)
# Check size of test dataset



print(len(X_test))
# Fit classifier on training dataset



clf.fit(X_train, y_train)
# Predict using the test dataset



predictions = clf.predict(X_test)



predictions = predictions.astype('int')



print(predictions)
# Create ImageId column for submission



ImageId = []

for i in range(1, 28001):

    ImageId.append(i)
# Create dataframe for submission



submission = pd.DataFrame({

    'ImageId': ImageId,

    'Label': predictions

})



print(submission)
# Submit



submission.to_csv('kaggle_mnist.csv', index=False)