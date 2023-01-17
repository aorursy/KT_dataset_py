import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import json
from pandas.io.json import json_normalize    
import os

from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import cohen_kappa_score
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
# Show the contents of the input directory
print(os.listdir("../input"))
train = pd.read_csv('../input/train/train.csv')
test = pd.read_csv('../input/test/test.csv')

# Define routines to read in the Training,Test sentiment score and magnitude;
# 0,0 is returned if there is no file found.
# Note that when used the argument fn will be one row of a dataframe
# in shich case fn['PetID'] is the PetId.

def readFile(fn):
    file = '../input/train_sentiment/'+fn['PetID']+'.json'
    if os.path.exists(file):
        with open(file) as data_file:    
            data = json.load(data_file)  

        df = json_normalize(data)
        mag = df['documentSentiment.magnitude'].values[0]
        score = df['documentSentiment.score'].values[0]
        return pd.Series([mag,score],index=['mag','score']) 
    else:
        return pd.Series([0,0],index=['mag','score'])
    
def readTestFile(fn):
    file = '../input/test_sentiment/'+fn['PetID']+'.json'
    if os.path.exists(file):
        with open(file) as data_file:    
            data = json.load(data_file)  

        df = json_normalize(data)
        mag = df['documentSentiment.magnitude'].values[0]
        score = df['documentSentiment.score'].values[0]
        return pd.Series([mag,score],index=['mag','score']) 
    else:
        return pd.Series([0,0],index=['mag','score'])
# Here the routines above are applied to each row of the dataframes.
# This is done using panadas' `apply()` with a small "anonymous function" defined with a python `lambda`.
# Note that just `train` could be used inplace of `train[['PetID']]`,
# this would make it clearer that x is a row of the dataframe and not the PetID value.

train[['SentMagnitude', 'SentScore']] = train[['PetID']].apply(lambda x: readFile(x), axis=1)
test[['SentMagnitude', 'SentScore']] = test[['PetID']].apply(lambda x: readTestFile(x), axis=1)

# Setup the training X, y, and test X
train_X = train.drop(['Name', 'Description', 'RescuerID', 'PetID', 'AdoptionSpeed'], axis=1)
train_y = train['AdoptionSpeed']
test_X = test.drop(['Name', 'Description', 'RescuerID', 'PetID'], axis=1)

# Define what will be the final predicted train and test values
train_meta = np.zeros(train_y.shape)
test_meta = np.zeros(test_X.shape[0])

# Choose and initialize a model.
clf = RandomForestClassifier(bootstrap=True, criterion = 'gini', max_depth=80,
                             max_features='auto', min_samples_leaf=5,
                             min_samples_split=5, n_estimators=200)

# Divide the training data into k-folds, k=4 here.
splits = list(StratifiedKFold(n_splits=4, shuffle=True, random_state=1812).split(train_X, train_y))

# Loop over the folds and fit the model to the fold's training data.
# Then evaluate that model on i) the validation data of that fold, 
# and ii) on all of the test data.
for idx, (train_idx, valid_idx) in enumerate(splits):
        # The training and validation sets for this fold
        X_train = train_X.iloc[train_idx]
        y_train = train_y[train_idx]
        X_val = train_X.iloc[valid_idx]
        y_val = train_y[valid_idx]
        
        # Fit the model
        clf.fit(X_train, y_train)
        
        # Look at the validation kappa and accuracy with classes right from the model
        y_pred = clf.predict(X_val)
        print("Fold {}: accuracy = {:.1f}%, kappa = {:.4f}  (no boundary adjustment)".format(idx,
                                100.0*accuracy_score(y_val, y_pred),     
                                cohen_kappa_score(y_val, y_pred, weights='quadratic')))
        #
        # Assign real-valued classes in addition to the integer classes of y_pred.
        # Start with the predicted probabilities by class
        y_probs = clf.predict_proba(X_val)
        # and get the class values (use a copy incase we change values)
        class_vals = clf.classes_.copy()
        # Change the ordinal weight of class 0 to be -1 as suggested by the plot in discussion:
        # https://www.kaggle.com/c/petfinder-adoption-prediction/discussion/76265
        # Does mot make much difference, though.
        class_vals[0] = -1
        # Create the float class values as the probability-weighted class
        # Here a python "list comprehension" is used rather than a loop.
        y_floats = [sum(y_probs[ix]*class_vals) for ix in range(len(y_probs[:,0]))]
        #   
        # Save these y_float values instead of the y_pred integers;
        ##train_meta[valid_idx] = y_pred.reshape(-1)
        train_meta[valid_idx] = y_floats
        # the predictions for just this validation fold are saved in the train_meta array;
        # looping over all folds will provide one prediction for each training sample.

        # Now use this fold's same model to generate Test predictions.
        ##y_test = clf.predict(test_X)
        # Instead of integer classes, get the predicted probabilites
        test_probs = clf.predict_proba(test_X)
        # and turn these into float class values.
        # Unlike the validation case, we get a test prediction from every fold,
        # so those float predictions are averaged. python list comprehension is used again.
        ##test_meta += y_test.reshape(-1) / len(splits)
        test_meta += np.array([sum(test_probs[ix]*class_vals) for
                               ix in range(len(test_probs[:,0]))]) / len(splits)
# Next two routines are a way to map float regression values to ordinal classes
# by making use of the known distribution of the training classes.

# In the following, y_pred is a floating value, e.g., the output of a regression to the class.
# Many sklearn _classifiers_ can also provide probabilities of the classes which
# can be turned into a floating value as the probability-weighted class, e.g.,:
#       y_probs = clf.predict_proba(X_val)
#       # The class values; use a copy incase we want to modify the values
#       class_vals = clf.classes_.copy()
#       y_floats = [sum(y_probs[ix]*class_vals) for ix in range(len(y_probs[:,0]))]


def get_class_bounds(y, y_pred, N=5, class0_fraction=-1):
    """
    Find boundary values for y_pred to match the known y class percentiles.
    Returns N-1 boundaries in y_pred values that separate y_pred
    into N classes (0, 1, 2, ..., N-1) with same percentiles as y has.
    Can adjust the fraction in Class 0 by the given factor (>=0), if desired. 
    """
    ysort = np.sort(y)
    predsort = np.sort(y_pred)
    bounds = []
    for ibound in range(N-1):
        iy = len(ysort[ysort <= ibound])
        # adjust the number of class 0 predictions?
        if (ibound == 0) and (class0_fraction >= 0.0) :
            iy = int(class0_fraction * iy)
        bounds.append(predsort[iy])
    return bounds

def assign_class(y_pred, boundaries):
    """
    Given class boundaries in y_pred units, output integer class values
    """
    y_classes = np.zeros(len(y_pred))
    for iclass, bound in enumerate(boundaries):
        y_classes[y_pred >= bound] = iclass + 1
    return y_classes.astype(int)
# Look at the histogram of the predicted float class values.
plt.hist(train_meta, bins=50)
plt.title("Training: meta float values")
plt.xlabel("Training y float values")
plt.show()
# This cell calculates and plots the kappa (and MSE) vs the class0 fraction adjustment.
# Note that MSE prefers (lower MSE) a class0 fraction near/at 0,
# whereas kappa prefers (higher kappa) a fraction near 1.
# Then the class0 fraction that gives best training kappa is selected.

# Save values of kappa, MSE, and accuracy vs the class0 fraction
kappas = []
mses = []
accurs = []
# fractions to try... (could go larger than 1 if desired.)
cl0fracs = np.array(np.arange(0.01,1.001,0.01))
for cl0frac in cl0fracs:
    boundaries = get_class_bounds(train_y, train_meta, class0_fraction=cl0frac)
    train_meta_ints = assign_class(train_meta, boundaries)
    kappa = cohen_kappa_score(train['AdoptionSpeed'], train_meta_ints, weights='quadratic')
    kappas.append(kappa)
    mse = mean_squared_error(train['AdoptionSpeed'], train_meta_ints)
    mses.append(mse)
    accur = accuracy_score(train['AdoptionSpeed'], train_meta_ints)
    accurs.append(accur)
    
# Use the class0 fraction that gives the highest training kappa
ifmax = np.array(kappas).argmax()
cl0frac = cl0fracs[ifmax]

print("Best kappa for class0 fraction = {:.4f}".format(cl0frac))
# Plots to show the kappa, MSE, and Accuracy vs class0 fraction

plt.plot(cl0fracs, kappas)
# indicate the highest-kappa point
plt.plot([cl0frac],[kappas[ifmax]],marker="o",color="green")
plt.title("Training: kappa vs class0_fraction")
plt.xlabel("class0_fraction")
plt.ylabel("kappa")
plt.show()

plt.plot(cl0fracs, mses)
plt.title("Training: MSE vs class0_fraction")
plt.xlabel("class0_fraction")
plt.ylabel("MSE")
plt.show()

plt.plot(cl0fracs, accurs)
plt.title("Training: Accuracy vs class0_fraction")
plt.xlabel("class0_fraction")
plt.ylabel("Accuracy")
plt.show()
# Can skip the class0_fraction adjustment and plotting cells above;
# can delete those two cells and just uncomment this line:
##cl0frac = 1.0

print("Using class0_fraction = {:.4f}, gives boundaries:".format(cl0frac))
boundaries = get_class_bounds(train_y, train_meta, class0_fraction=cl0frac)
print(boundaries)

train_meta_ints = assign_class(train_meta, boundaries)
kappa = cohen_kappa_score(train_y, train_meta_ints, weights='quadratic')

print("Adjusted boundaries give:")
print("kappa = {:.4f}  (with accuracy = {:.1f}%)".format(kappa,
                                100.0*accuracy_score(train_y, train_meta_ints)))
# Confusion Matrix
con_mat = confusion_matrix(train_y, train_meta_ints)

# Look at the number that are on the diagonal (exact agreement)
diag = 0.0
for id in range(5):
    diag += con_mat[id,id]
print("\nConfusion matrix - Columns are prediced 0, predicted 1, etc.\n")
print(con_mat)
print("")
print("\n{2:.2f}% = {0}/{1} are on the diagonal (= accuracy)".format(
        int(diag), con_mat.sum(), 100.0*diag/con_mat.sum()))
plt.hist(train_meta_ints, bins=40, color='blue')
plt.hist(train_y, bins=20, bottom=0.0, alpha=0.2)
plt.title("Train: Boundary-based Predictions")
plt.show()
plt.hist(test_meta, bins=50)
plt.title("Test: meta float values")
plt.show()
# Map the test values to integers using the training boundaries
test_meta_ints = assign_class(test_meta, boundaries)
plt.hist(test_meta_ints.astype(int), bins=50)
plt.title("Test: Boundary-based Predictions")
plt.show()
sub = pd.read_csv('../input/test/sample_submission.csv')
sub['AdoptionSpeed'] = test_meta_ints
sub['AdoptionSpeed'] = sub['AdoptionSpeed'].astype(int)
sub.to_csv("submission.csv", index=False)

!head -5 submission.csv
!echo ...
!tail -5 submission.csv

