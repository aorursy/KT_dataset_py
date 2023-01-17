# Importing packages:

import numpy as np

import pandas as pd



from sklearn.preprocessing import MaxAbsScaler

from sklearn.metrics import accuracy_score,f1_score,log_loss,roc_auc_score



from sklearn.model_selection import train_test_split,ShuffleSplit,cross_val_score,GridSearchCV



from sklearn import svm

from sklearn.linear_model import LogisticRegression,SGDClassifier



from os.path import join as opj

from matplotlib import pyplot as plt



from matplotlib.colors import Normalize



# Reading the traning data set json file to a pandas dataframe

train=pd.read_json('../input/train.json')



# Lets take a look at the first 5 rows of the dataset

train.head(5)
# Replace the 'na's with numpy.nan

train.inc_angle.replace({'na':np.nan}, inplace=True)



# Drop the rows that has NaN value for inc_angle

train.drop(train[train['inc_angle'].isnull()].index,inplace=True)
X_HH_train=np.array([np.array(band).astype(np.float32) for band in train.band_1])

X_HV_train=np.array([np.array(band).astype(np.float32) for band in train.band_2])

X_angle_train=np.array([[np.array(angle).astype(np.float32) for angle in train.inc_angle]]).T

y_train=train.is_iceberg.values.astype(np.float32)

X_train=np.concatenate((X_HH_train,X_HV_train,X_angle_train), axis=1)

# Now, we have 75*75 numerical features for band_1, 75*75 numerical features for band_2, and 1  feature for angle 

X_train.shape
scaler = MaxAbsScaler()

X_train_maxabs = scaler.fit_transform(X_train)
# Create the SVM instance using Radial Basis Function (rbf) kernel

clf = svm.SVC(kernel='rbf',probability=False)

# Set the range of hyper-parameter we wanna use to tune our SVM classifier

C_range = [0.1,1,10,50,100]

gamma_range = [0.00001,0.0001,0.001,0.01,0.1]

param_grid_SVM = dict(gamma=gamma_range, C=C_range)

# set the gridsearch using 3-fold cross validation and 'ROC Area Under the Curve' as the cross validation score. 

grid = GridSearchCV(clf, param_grid=param_grid_SVM, cv=3,scoring='roc_auc')

grid.fit(X_train_maxabs, y_train)

print("The best parameters are %s with a score of %0.2f" % (grid.best_params_, grid.best_score_))
class MidpointNormalize(Normalize):



    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):

        self.midpoint = midpoint

        Normalize.__init__(self, vmin, vmax, clip)



    def __call__(self, value, clip=None):

        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]

        return np.ma.masked_array(np.interp(value, x, y))



    

plt.figure(figsize=(8, 6))

plt.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)

scores = grid.cv_results_['mean_test_score'].reshape(len(C_range),len(gamma_range))

plt.imshow(scores, interpolation='nearest', cmap=plt.cm.hot,norm=MidpointNormalize(vmin=0.5, midpoint=0.95))

plt.xlabel('gamma')

plt.ylabel('C')

plt.colorbar()

plt.xticks(np.arange(len(gamma_range)), gamma_range, rotation=45)

plt.yticks(np.arange(len(C_range)), C_range)

plt.title('Validation ROC_AUC score')

plt.show()