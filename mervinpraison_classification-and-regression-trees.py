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
wbc = pd.read_csv('../input/ninechapter-breastcancer/breastCancer.csv')

df = wbc
label_encoder = sklearn.preprocessing.LabelEncoder()

label_encoder.fit(df['diagnosis'])
X= df[['radius_mean', 'concave points_mean']]

y = label_encoder.transform(df['diagnosis'])



X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X,y)
SEED = 1
# Import DecisionTreeClassifier from sklearn.tree

from sklearn.tree import DecisionTreeClassifier



# Instantiate a DecisionTreeClassifier 'dt' with a maximum depth of 6

dt = DecisionTreeClassifier(max_depth=6, random_state=SEED)



# Fit dt to the training set

dt.fit(X_train, y_train)



# Predict test set labels

y_pred = dt.predict(X_test)

print(y_pred[0:5])
# Import accuracy_score

from sklearn.metrics import accuracy_score



# Predict test set labels

y_pred = dt.predict(X_test)



# Compute test set accuracy  

acc = accuracy_score(y_test, y_pred)

print("Test set accuracy: {:.2f}".format(acc))
import mlxtend.plotting
def plot_labeled_decision_regions(X_test, y_test, clfs):

    

    for clf in clfs:



        mlxtend.plotting.plot_decision_regions(np.array(X_test), np.array(y_test), clf=clf, legend=2)

        

        plt.ylim((0,0.2))



        # Adding axes annotations

        plt.xlabel(X_test.columns[0])

        plt.ylabel(X_test.columns[1])

        plt.title(str(clf).split('(')[0])

        plt.show()
# Import LogisticRegression from sklearn.linear_model

from sklearn.linear_model import  LogisticRegression



# Instatiate logreg

logreg = LogisticRegression(random_state=1, solver='lbfgs')



# Fit logreg to the training set

logreg.fit(X_train, y_train)



# Define a list called clfs containing the two classifiers logreg and dt

clfs = [logreg, dt]



# Review the decision regions of the two classifiers

plot_labeled_decision_regions(X_test, y_test, clfs)
# Import DecisionTreeClassifier from sklearn.tree

from sklearn.tree import DecisionTreeClassifier



# Instantiate dt_entropy, set 'entropy' as the information criterion

dt_entropy = DecisionTreeClassifier(max_depth=8, criterion='entropy', random_state=1)

dt_gini = DecisionTreeClassifier(max_depth=8, criterion='gini', random_state=1)





# Fit dt_entropy to the training set

dt_entropy.fit(X_train, y_train)

dt_gini.fit(X_train,y_train)
# Import accuracy_score from sklearn.metrics

from sklearn.metrics import accuracy_score



# Use dt_entropy to predict test set labels

y_pred = dt_entropy.predict(X_test)



# Evaluate accuracy_entropy

accuracy_entropy = accuracy_score(y_test, y_pred)



y_pred = dt_gini.predict(X_test)



accuracy_gini = accuracy_score(y_test, y_pred)



# Print accuracy_entropy

print('Accuracy achieved by using entropy: ', accuracy_entropy)



# Print accuracy_gini

print('Accuracy achieved by using the gini index: ', accuracy_gini)
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
# Import DecisionTreeRegressor from sklearn.tree

from sklearn.tree import DecisionTreeRegressor



# Instantiate dt

dt = DecisionTreeRegressor(max_depth=8,

             min_samples_leaf=0.13,

            random_state=3)

lr = sklearn.linear_model.LinearRegression()



# Fit dt to the training set

dt.fit(X_train, y_train)

lr.fit(X_train,y_train)

# Import mean_squared_error from sklearn.metrics as MSE

from sklearn.metrics import mean_squared_error as MSE



# Compute y_pred

y_pred = dt.predict(X_test)



# Compute mse_dt

mse_dt = MSE(y_test, y_pred)



# Compute rmse_dt

import numpy as np

rmse_dt = np.sqrt(mse_dt)



# Print rmse_dt

print("Test set RMSE of dt: {:.2f}".format(rmse_dt))
# Predict test set labels 

y_pred_lr = lr.predict(X_test)



# Compute mse_lr

mse_lr = MSE(y_test, y_pred_lr)



# Compute rmse_lr

import numpy as np

rmse_lr = np.sqrt(mse_lr)



# Print rmse_lr

print('Linear Regression test set RMSE: {:.2f}'.format(rmse_lr))



# Print rmse_dt

print('Regression Tree test set RMSE: {:.2f}'.format(rmse_dt))