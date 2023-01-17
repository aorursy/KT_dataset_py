import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import matplotlib as mpl

import sys, os, scipy, sklearn

import sklearn.metrics, sklearn.preprocessing, sklearn.model_selection, sklearn.tree, sklearn.linear_model, sklearn.cluster
mpl.rcParams['font.size'] = 14

pd.options.display.max_columns = 1000
data_folder = './'

data_files = os.listdir(data_folder)

display('Course files:',

        data_files)

for file_name in data_files:

    if '.csv' in file_name:

        globals()[file_name.replace('.csv','')] = pd.read_csv(data_folder+file_name, 

                                                              ).reset_index(drop=True)

        print(file_name)

        display(globals()[file_name.replace('.csv','')].head(), globals()[file_name.replace('.csv','')].shape)
import os

print(os.listdir("../input"))
auto = pd.read_csv('../input/automobile/auto.csv')

df = auto
X = df[['displ', 'hp', 'weight', 'accel', 'size', 'origin']]

y = df['mpg']
OneHotEncoder = sklearn.preprocessing.OneHotEncoder()

OneHotEncodings = OneHotEncoder.fit_transform(df[['origin']]).toarray()

OneHotEncodings = pd.DataFrame(OneHotEncodings,

                               columns = ['origin_'+header for header in OneHotEncoder.categories_[0]])



X = X.drop(columns = 'origin').reset_index(drop=True)

X = pd.concat((X,OneHotEncodings),axis=1)
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X,y)

print(X_train.shape,y_train.shape)
# Import train_test_split from sklearn.model_selection

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeRegressor



# Set SEED for reproducibility

SEED = 1



# Split the data into 70% train and 30% test

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=SEED)



# Instantiate a DecisionTreeRegressor dt

dt = DecisionTreeRegressor(min_samples_leaf=0.26, max_depth=4, random_state=SEED)
from sklearn.model_selection import cross_val_score



# Compute the array containing the 10-folds CV MSEs

MSE_CV_scores = - cross_val_score(dt, X_train, y_train, scoring = 'neg_mean_squared_error', cv=10,  n_jobs=-1)



# Compute the 10-folds CV RMSE

import numpy as np

RMSE_CV = np.sqrt(MSE_CV_scores.mean())



# Print RMSE_CV

print('CV RMSE: {:.2f}'.format(RMSE_CV))
# Import mean_squared_error from sklearn.metrics as MSE

from sklearn.metrics import mean_squared_error as MSE



# Fit dt to the training set

dt.fit(X_train, y_train)



# Predict the labels of the training set

y_pred_train = dt.predict(X_train)



# Evaluate the training set RMSE of dt

RMSE_train = (MSE(y_train, y_pred_train))**(0.5)



# Print RMSE_train

print('Train RMSE: {:.2f}'.format(RMSE_train))
df = pd.read_csv('../input/indian-liver-patient-preprocessed/indian_liver_patient_preprocessed.csv')

df.head()



X = df.drop(columns = ['Liver_disease'])

y = df['Liver_disease']



X_train, X_test,  y_train, y_test = sklearn.model_selection.train_test_split(X,y)
from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier as KNN



# Set seed for reproducibility

SEED=1



# Instantiate lr

lr = LogisticRegression(random_state=SEED)



# Instantiate knn

knn = KNN(n_neighbors=27)



# Instantiate dt

dt = DecisionTreeClassifier(min_samples_leaf=0.13, random_state=SEED)



# Define the list classifiers

classifiers = [('Logistic Regression', lr), ('K Nearest Neighbours', knn), ('Classification Tree', dt)]
from sklearn.metrics import accuracy_score



# Iterate over the pre-defined list of classifiers

for clf_name, clf in classifiers:    

 

    # Fit clf to the training set

    clf.fit(X_train, y_train)    

   

    # Predict y_pred

    y_pred = clf.predict(X_test)

    

    # Calculate accuracy

    accuracy = accuracy_score(y_test, y_pred) 

   

    # Evaluate clf's accuracy on the test set

    print('{:s} : {:.3f}'.format(clf_name, accuracy))
# Import VotingClassifier from sklearn.ensemble

from sklearn.ensemble import VotingClassifier



# Instantiate a VotingClassifier vc

vc = VotingClassifier(estimators=classifiers)     



# Fit vc to the training set

vc.fit(X_train,y_train)   



# Evaluate the test set predictions

y_pred = vc.predict(X_test)



# Calculate accuracy score

accuracy = accuracy_score(y_test, y_pred)

print('Voting Classifier: {:.3f}'.format(accuracy))
# Import DecisionTreeClassifier

from sklearn.tree import DecisionTreeClassifier



# Import BaggingClassifier

from sklearn.ensemble import BaggingClassifier



# Instantiate dt

dt = DecisionTreeClassifier(random_state=1)



# Instantiate bc

bc = BaggingClassifier(base_estimator=dt, n_estimators=50, random_state=1)
# Fit bc to the training set

bc.fit(X_train, y_train)



# Predict test set labels

y_pred = bc.predict(X_test)



# Evaluate acc_test

acc_test = accuracy_score(y_test, y_pred)

print('Test set accuracy of bc: {:.2f}'.format(acc_test)) 
# Import DecisionTreeClassifier

from sklearn.tree import DecisionTreeClassifier



# Import BaggingClassifier

from sklearn.ensemble import BaggingClassifier



# Instantiate dt

dt = DecisionTreeClassifier(min_samples_leaf=8, random_state=1)



# Instantiate bc

bc = BaggingClassifier(base_estimator=dt, 

            n_estimators=50,

            oob_score=True,

            random_state=1)
# Fit bc to the training set 

bc.fit(X_train, y_train)



# Predict test set labels

y_pred = bc.predict(X_test)



# Evaluate test set accuracy

acc_test = accuracy_score(y_test, y_pred)



# Evaluate OOB accuracy

acc_oob = bc.oob_score_



# Print acc_test and acc_oob

print('Test set accuracy: {:.3f}, OOB accuracy: {:.3f}'.format(acc_test, acc_oob))
# Import RandomForestRegressor

from sklearn.ensemble import RandomForestRegressor



# Instantiate rf

rf = RandomForestRegressor(n_estimators=25,

            random_state=2)

            

# Fit rf to the training set    

rf.fit(X_train, y_train) 
# Import mean_squared_error as MSE

from sklearn.metrics import mean_squared_error as MSE



# Predict the test set labels

y_pred = rf.predict(X_test)



# Evaluate the test set RMSE

rmse_test = MSE(y_test,y_pred)**0.5



# Print rmse_test

print('Test set RMSE of rf: {:.2f}'.format(rmse_test))
# Create a pd.Series of features importances

importances = pd.Series(data=rf.feature_importances_,

                        index= X_train.columns)



# Sort importances

importances_sorted = importances.sort_values()



# Draw a horizontal barplot of importances_sorted

importances_sorted.plot(kind='barh', color='lightgreen')

plt.title('Features Importances')

plt.show()