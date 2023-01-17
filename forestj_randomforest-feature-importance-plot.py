%matplotlib inline
import pandas as pd
import numpy as np
import sklearn.ensemble
from  sklearn import cross_validation
import time # timing

# The competition datafiles are in the directory ../input
# Read competition data files:
t_start = time.time()
test = pd.read_csv('../input/test.csv')
train= pd.read_csv('../input/train.csv')

# Time to load data
t_load = time.time()
print('load time:'+str( t_load-t_start))
# Split test and validation sets
X_train,X_valid,y_train,y_valid = cross_validation.train_test_split(train[train.keys()[1:]],train['label'],test_size=0.2)

# Train Random Forest
rfc = sklearn.ensemble.RandomForestClassifier(n_estimators=100)
rfc.fit(X_train,y_train)
t_train=time.time()
print('train time:%g' % (t_train-t_load))

# Check the score
print(rfc.score(X_valid,y_valid))
# Predict the validation set and the test set and output to submission
y_valid_pred = rfc.predict(X_valid)
X_pred = rfc.predict(test)
filename = 'submission.csv'
with open(filename,'w') as f:
    f.write('ImageId,Label\n')
    for i in range(len(test)):
        f.write(str(i+1)+','+str(X_pred[i])+'\n')
# Plot several misclassified cases
# Define display function, copied form Koba Khitalishvili
import matplotlib.pyplot as plt
import matplotlib.cm as cm
def display(img):
    plt.imshow(np.array(img).reshape(28,28),cmap=cm.binary)

# Find the mispredicted cases
error_case,= np.where(y_valid!=y_valid_pred)
correct_case = np.where(y_valid==y_valid_pred)

# Print three cases
i=int(len(error_case)/2)
plt.figure()
display(X_valid.values[error_case[i],:])
# Print true label and predicted label
print('True label:'+str(y_valid.values[error_case[i]])+' predicted:'+str(y_valid_pred[error_case[i]]))
i=30
print(i)
plt.figure()
display(X_valid.values[error_case[i],:])
# Print true label and predicted label
print('True label:'+str(y_valid.values[error_case[i]])+' predicted:'+str(y_valid_pred[error_case[i]]))
# Show the feature importance given by Random Forest
# I think it is the most interesting part
plt.figure()
plt.imshow(rfc.feature_importances_.reshape(28,28),interpolation="nearest")