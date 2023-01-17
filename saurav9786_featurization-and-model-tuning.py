# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

import warnings

warnings.filterwarnings('ignore')



from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#Read the dataset



df_pre = pd.read_csv("/kaggle/input/wine-quality/winequalityN.csv")



Shape=df_pre.shape



print("Rows:",Shape[0])

print("Columns:",Shape[1])



df_pre.head(10)
#Info about the dataset

df_pre.info()
dff=df_pre
dff.describe().T
#Checking or the null values



dff.isna().any()
#Null value count in the dataset



dff.isna().sum()
#Replacing the null values with the median



dff = dff.fillna(dff.median())

#Handling the categorical values

dff = pd.get_dummies(data = dff, columns = ['type'] , prefix = ['type'] , drop_first = True)

# Lets check for highly correlated variables



cor= dff.corr()

cor.loc[:,:] = np.tril(cor,k=-1)  # reference:https://www.geeksforgeeks.org/numpy-tril-python/

cor=cor.stack()

cor[(cor > 0.55) | (cor< -0.55)]
cor=dff.corr()

cor
#Correlation plot

plt.figure(figsize=(10,8))

sns.heatmap(cor,annot=True,linewidths=.5,center=0,cbar=False,cmap="YlGnBu")

plt.show()
Target_Imb=dff["quality"].value_counts(normalize=True)

Target_Imb
#Combine 7&8 together; combine 3 and 4 with 5 so that we have only 3 levels and a more balanced Y variable

dff['quality'] = dff['quality'].replace(8,7)

dff['quality'] = dff['quality'].replace(3,5)

dff['quality'] = dff['quality'].replace(4,5)

dff['quality'].value_counts()
# splitting data into training and test set for independent attributes

from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test =train_test_split(dff.drop('quality',axis=1), dff['quality'], test_size=0.30,

                                                   random_state=22)

X_train.shape,X_test.shape
from sklearn.tree import DecisionTreeClassifier

from sklearn import metrics

from sklearn.metrics import accuracy_score,f1_score,recall_score,precision_score, confusion_matrix

model_entropy=DecisionTreeClassifier(criterion='entropy')

model_entropy.fit(X_train, y_train)

model_entropy.score(X_train, y_train)

model_entropy.score(X_test, y_test)
clf_pruned = DecisionTreeClassifier(criterion = "entropy", random_state = 100, max_depth=5, min_samples_leaf=5)

clf_pruned.fit(X_train, y_train)

preds_pruned = clf_pruned.predict(X_test)

preds_pruned_train = clf_pruned.predict(X_train)

print(accuracy_score(y_test,preds_pruned))

print(accuracy_score(y_train,preds_pruned_train))
acc_DT = accuracy_score(y_test, preds_pruned)

#Store the accuracy results for each model in a dataframe for final comparison

resultsDf = pd.DataFrame({'Method':['Decision Tree'], 'accuracy': acc_DT})

resultsDf = resultsDf[['Method', 'accuracy']]

resultsDf
from sklearn.ensemble import RandomForestClassifier

rfcl = RandomForestClassifier(n_estimators = 50)

rfcl = rfcl.fit(X_train, y_train)

pred_RF = rfcl.predict(X_test)

acc_RF = accuracy_score(y_test, pred_RF)

tempResultsDf = pd.DataFrame({'Method':['Random Forest'], 'accuracy': [acc_RF]})

resultsDf = pd.concat([resultsDf, tempResultsDf])

resultsDf = resultsDf[['Method', 'accuracy']]

resultsDf



acc_RF=acc_RF*100
from sklearn.ensemble import AdaBoostClassifier

abcl = AdaBoostClassifier( n_estimators= 100, learning_rate=0.1, random_state=22)

abcl = abcl.fit(X_train, y_train)

pred_AB =abcl.predict(X_test)

acc_AB = accuracy_score(y_test, pred_AB)

tempResultsDf = pd.DataFrame({'Method':['Adaboost'], 'accuracy': [acc_AB]})

resultsDf = pd.concat([resultsDf, tempResultsDf])

resultsDf = resultsDf[['Method', 'accuracy']]

resultsDf
from sklearn.ensemble import GradientBoostingClassifier

gbcl = GradientBoostingClassifier(n_estimators = 50, learning_rate = 0.1, random_state=22)

gbcl = gbcl.fit(X_train, y_train)

pred_GB =gbcl.predict(X_test)

acc_GB = accuracy_score(y_test, pred_GB)

tempResultsDf = pd.DataFrame({'Method':['Gradient Boost'], 'accuracy': [acc_GB]})

resultsDf = pd.concat([resultsDf, tempResultsDf])

resultsDf = resultsDf[['Method', 'accuracy']]

resultsDf
from sklearn.ensemble import BaggingClassifier



bgcl = BaggingClassifier(n_estimators=50, max_samples= .7, bootstrap=True, oob_score=True, random_state=22)

bgcl = bgcl.fit(X_train, y_train)

pred_BG =bgcl.predict(X_test)

acc_BG = accuracy_score(y_test, pred_BG)

tempResultsDf = pd.DataFrame({'Method':['Bagging'], 'accuracy': [acc_BG]})

resultsDf = pd.concat([resultsDf, tempResultsDf])

resultsDf = resultsDf[['Method', 'accuracy']]

resultsDf
from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

num_folds = 20

seed = 100

kfold = KFold(n_splits=num_folds, random_state=seed)



y = dff['quality']

X = dff.loc[:, dff.columns != 'quality']



results = cross_val_score(rfcl,X, y, cv=kfold)



Kfold_CV=np.around(np.mean(abs(results*100)))



for i in range(num_folds):

    print("Kfold",i,":",results[i]*100,"%\n")



print("Mean:",Kfold_CV,"%")



print("\nStandard Deviation:",results.std())





print("\n\nRandom Forest Accuracy",acc_RF,"%\n\n")



improvement=Kfold_CV-acc_RF

print("Accuracy improvement:",np.around(improvement),"%")



from sklearn.model_selection import LeaveOneOut

from sklearn.model_selection import cross_val_score

scores = cross_val_score(rfcl, X_train, y_train, cv=LeaveOneOut())

scores



print("Mean accuracy:",scores.mean()*100,"%")



print("\nRandom Forest Accuracy",acc_RF,"%")



print("\nStandard Deviation:",scores.std())



Leave_One_Out=np.mean(abs(scores.mean()*100))

improvement=Leave_One_Out-acc_RF

print("\nAccuracy improvement:",np.around(improvement),"%")
from sklearn.model_selection  import StratifiedKFold, cross_val_score



k = 20



stratified_kfold = StratifiedKFold(n_splits = k, random_state = 55)

results = cross_val_score(rfcl, X, y, cv = stratified_kfold)



strat_CV=np.around(np.mean(abs(results*100)))



for i in range(k):

    print("Kfold",i,":",results[i]*100,"%\n")



print("\n\nMean:",strat_CV,"%")



print("\nStandard Deviation:",results.std())



print("\n\nRandom Forest Accuracy",acc_RF,"%\n\n")



improvement=strat_CV-acc_RF

print("Accuracy improvement:",np.around(improvement),"%")
# Number of iterations for bootstrapping

bootstrap_iteration = 75

accuracy = []



from sklearn.utils import resample

from sklearn.metrics import accuracy_score



for i in range(bootstrap_iteration):

    X_, y_ = resample(X_train, y_train)

    rfcl.fit(X_, y_)

    y_pred = rfcl.predict(X_test)

    

    acc = accuracy_score(y_pred, y_test)

    accuracy.append(acc)

    

accuracy = np.array(accuracy)



print('Standard deviation: ', accuracy.std())



Boot=np.around(accuracy.mean()*100)



print("\n\nMean:",Boot,"%")



print("\nStandard Deviation:",accuracy.std())



print("\n\nRandom Forest Accuracy",acc_RF,"%\n\n")



improvement=Boot-acc_RF

print("Accuracy improvement:",np.around(improvement),"%")
#pretty print



from pprint import pprint

from sklearn.ensemble import RandomForestRegressor



rf = RandomForestRegressor(random_state = 1)



print('Parameters currently in use:\n')

pprint(rf.get_params())
import numpy as np

print(np.linspace(start = 5, stop = 10, num = 2))
from sklearn.model_selection import RandomizedSearchCV



# Number of trees in random forest

n_estimators = [int(x) for x in np.linspace(start = 10 , stop = 15, num = 2)]   # returns evenly spaced 10 numbers

# Number of features to consider at every split

max_features = ['auto', 'sqrt']

# Maximum number of levels in tree

max_depth = [int(x) for x in np.linspace(5, 10, num = 2)]  # returns evenly spaced numbers can be changed to any

max_depth.append(None)

# Minimum number of samples required to split a node

min_samples_split = [2, 5, 10]

# Minimum number of samples required at each leaf node

min_samples_leaf = [1, 2, 4]

# Method of selecting samples for training each tree

bootstrap = [True, False]



# Create the random grid

random_grid = {'n_estimators': n_estimators,

               'max_features': max_features,

               'max_depth': max_depth,

               'min_samples_split': min_samples_split,

               'min_samples_leaf': min_samples_leaf,

               'bootstrap': bootstrap}



pprint(random_grid)



rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid,

                              n_iter = 5, scoring='neg_mean_absolute_error', 

                              cv = 3, verbose=2, random_state=42, n_jobs=-1,

                              return_train_score=True)



# Fit the random search model

rf_random.fit(X_train, y_train);

rf_random.best_params_
from sklearn.model_selection import GridSearchCV



param_grid = {

    'bootstrap': [True],

    'max_depth': [5,6],

    'max_features': [2, 3],

    'min_samples_leaf': [3, 4],

    'min_samples_split': [5,10],

    'n_estimators': [5,6,7]

}    



rf = RandomForestRegressor(random_state = 1)



grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 

                          cv = 3, n_jobs = 1, verbose = 0, return_train_score=True)



grid_search.fit(X_train, y_train);



grid_search.best_params_

best_grid = grid_search.best_estimator_



Grid_search_cv=np.around(best_grid.score(X_test, y_test)*100)

print("Grid sarch CV Score:",Grid_search_cv,"%")



print("\n\nRandom Forest Accuracy",acc_RF,"%\n\n")



improvement=Grid_search_cv-acc_RF

print("Accuracy improvement:",np.around(improvement),"%")



print("Randon Forest Accuracy:", acc_RF,"%")



print("\nK Fold:", Kfold_CV,"%")



print("\nLeave one Out:", Leave_One_Out,"%")



print("\nStratified CV:", strat_CV,"%")



print("\nBootstrapping:", Boot,"%")



print("\nRandom sarch CV:",rf_random,"%")



print("\Grid sarch CV Score:",grid_search,"%")


