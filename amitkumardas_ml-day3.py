# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Code for data holdout ...

from sklearn.model_selection import train_test_split



data = pd.read_csv("../input/autompg/auto-mpg.csv")

print(data.shape)

data_train, data_test = train_test_split(data, test_size = 0.3, random_state = 123) 

# The parameter test_size sets the ratio of test data to the input data to 0.3 or 30% and random_state sets the seed for random number generator

print(data_train.shape)

print(data_test.shape)
# Code for  K-fold cross-validation ...



from sklearn.model_selection import KFold

kf = KFold(n_splits=10)

for train_index, test_index  in kf.split(data):

    data_train = data.iloc[train_index]

    data_test = data.iloc[test_index]

    print(data_train.shape)

    print(data_test.shape)
# Code for Bootstrap sampling ...



from sklearn.utils import resample

X = data.iloc[:,0:9]

resample(X, n_samples=200, random_state=0) # Generates a sample of size 200 (as mentioned in parameter n_samples) with repetition
# Code for model training and evaluation (Classification) ...



from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix



data = pd.read_csv("../input/apndcts/apndcts.csv")



predictors = data.iloc[:,0:7] # Segregating the predictors

target = data.iloc[:,7] # Segregating the target/class

predictors_train, predictors_test, target_train, target_test = train_test_split(predictors, target, test_size = 0.3, random_state = 123) # Holdout of data

dtree_entropy = DecisionTreeClassifier(criterion = "entropy", random_state = 100, max_depth=3, min_samples_leaf=5) #Model is initialized

# Finally the model is trained

model = dtree_entropy.fit(predictors_train, target_train)

prediction = model.predict(predictors_test)



acc_score = 0



acc_score = accuracy_score(target_test, prediction, normalize = True)

print(acc_score)

conf_mat = confusion_matrix(target_test, prediction)

print(conf_mat)
# Code for model training and evaluation (Classification) ...



import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error, r2_score

from sklearn import linear_model

from sklearn.impute import SimpleImputer



data = pd.read_csv("../input/autompg/auto-mpg.csv")

data = data.dropna(axis=0) #Remove all rows where value of any column is ‘NaN’

#data.replace('?',0, inplace=True)

predictors = data.iloc[:,1:7] # Seggretating the predictor variables ...

target = data.iloc[:,0]  # Seggretating the target / class variable (mpg) ...

predictors_train, predictors_test, target_train, target_test = train_test_split(predictors, target, test_size = 0.3, random_state = 123)



lm = linear_model.LinearRegression()



# First train model / classifier with the input dataset (training data part of it)

model = lm.fit(predictors_train, target_train)

# Make prediction using the trained model

prediction = model.predict(predictors_test)



msq = mean_squared_error(target_test, prediction)

print("Mean squared error:",msq)



r2s = r2_score(target_test, prediction)

print("r2 score : ", r2s)
import pandas as pd

import numpy as np

from sklearn.cluster import KMeans

from sklearn.metrics.cluster import v_measure_score

from sklearn.metrics import silhouette_score



data = pd.read_csv("../input/spine-data/spine.csv")

data_woc = data.iloc[:,0:12] # Stripping out the class variable from the data set ...

data_class = data.iloc[:,12] # Segregating the target /class variable ...

f1 = data_woc['pelvic_incidence'].values

f2 = data_woc['pelvic_radius'].values

f3 = data_woc['thoracic_slope'].values

X = np.array(list(zip(f1, f2, f3)))

kmeans = KMeans(n_clusters = 2, random_state = 123)

model = kmeans.fit(X)

cluster_labels = kmeans.predict(X)

v_measure_score(cluster_labels, data_class)



sil = silhouette_score(X, cluster_labels, metric = 'euclidean',sample_size = len(data)) # “X” is a feature matrix 

                                        # for the feature subset selected for clustering and “data” is the data set

print(sil)
# Bagging based Ensemble learning ...



import pandas as pd

import numpy as np

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import MinMaxScaler

from sklearn import model_selection

from sklearn.ensemble import BaggingClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn import preprocessing



data = pd.read_csv("../input/wdbc-data/wdbc.csv")

le = preprocessing.LabelEncoder()

le.fit(data['class'])

data['class'] = le.transform(data['class'])



# Convert the DataFrame object into NumPy array otherwise you will not be able to impute

values = data.values



# Now impute it

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

imputedData = imputer.fit_transform(values)



# Normalize the ranges of the features to a uniform range, say 0 - 1 ...

scaler = MinMaxScaler(feature_range=(0, 1))

normalizedData = scaler.fit_transform(imputedData)



# Segregate the features from the labels

X = normalizedData[:,0:9]

Y = normalizedData[:,9]



kfold = model_selection.KFold(n_splits=10, random_state=7)



# Bagged Decision Trees for Classification - necessary dependencies

cart = DecisionTreeClassifier()

num_trees = 100

model = BaggingClassifier(base_estimator=cart, n_estimators=num_trees, random_state=7)

results = model_selection.cross_val_score(model, X, Y, cv=kfold, error_score='raise')

print(results.mean())
# AdaBoost Classification



import pandas as pd

import numpy as np

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import MinMaxScaler

from sklearn import model_selection

from sklearn.ensemble import AdaBoostClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn import preprocessing



data = pd.read_csv("../input/wdbc-data/wdbc.csv")

le = preprocessing.LabelEncoder()

le.fit(data['class'])

data['class'] = le.transform(data['class'])



# Convert the DataFrame object into NumPy array otherwise you will not be able to impute

values = data.values



# Now impute it

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

imputedData = imputer.fit_transform(values)



# Normalize the ranges of the features to a uniform range, say 0 - 1 ...

scaler = MinMaxScaler(feature_range=(0, 1))

normalizedData = scaler.fit_transform(imputedData)



# Segregate the features from the labels

X = normalizedData[:,0:9]

Y = normalizedData[:,9]



kfold = model_selection.KFold(n_splits=10, random_state=7)



seed = 7

num_trees = 70

kfold = model_selection.KFold(n_splits=10, random_state=seed)

model = AdaBoostClassifier(n_estimators=num_trees, random_state=seed)

#results = model_selection.cross_val_score(model, X, Y, cv=kfold, error_score=np.nan)

print(results.mean())
# Voting Ensemble for Classification



import pandas as pd

import numpy as np

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import MinMaxScaler

from sklearn import model_selection

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.svm import SVC

from sklearn.ensemble import VotingClassifier

from sklearn import preprocessing

from sklearn.model_selection import cross_val_score



data = pd.read_csv("../input/wdbc-data/wdbc.csv")

le = preprocessing.LabelEncoder()

le.fit(data['class'])

data['class'] = le.transform(data['class'])



# Convert the DataFrame object into NumPy array otherwise you will not be able to impute

values = data.values



# Now impute it

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

imputedData = imputer.fit_transform(values)



# Normalize the ranges of the features to a uniform range, say 0 - 1 ...

scaler = MinMaxScaler(feature_range=(0, 1))

normalizedData = scaler.fit_transform(imputedData)



# Segregate the features from the labels

X = normalizedData[:,0:9]

Y = normalizedData[:,9]



kfold = model_selection.KFold(n_splits=10, random_state=7)



# create the sub models

estimators = []

model1 = LogisticRegression()

estimators.append(('logistic', model1))

model2 = DecisionTreeClassifier()

estimators.append(('cart', model2))

model3 = SVC()

estimators.append(('svm', model3))

# create the ensemble model

ensemble = VotingClassifier(estimators)



print((cross_val_score(ensemble, X, Y, cv=kfold)))

#results = cross_val_score(ensemble, X, Y, cv=kfold)

#print(results.mean())
import pandas as pd

from sklearn.model_selection import train_test_split

import numpy as np

from sklearn.model_selection import GridSearchCV

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import VotingClassifier



#read in the dataset

df = pd.read_csv("../input/diabetes/diabetes.csv")



#split data into inputs and targets

X = df.drop(columns = ['diabetes'])

y = df['diabetes']



#split data into train and test sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)



#create new a knn model

knn = KNeighborsClassifier()

#create a dictionary of all values we want to test for n_neighbors

params_knn = {'n_neighbors': np.arange(1, 25)}

#use gridsearch to test all values for n_neighbors

knn_gs = GridSearchCV(knn, params_knn, cv=5)

#fit model to training data

knn_gs.fit(X_train, y_train)

#save best model

knn_best = knn_gs.best_estimator_

#check best n_neigbors value

print(knn_gs.best_params_)





#create a new logistic regression model

log_reg = LogisticRegression()

#fit the model to the training data

log_reg.fit(X_train, y_train)

print('log_reg: {}'.format(log_reg.score(X_test, y_test)))





#create a new random forest classifier

rf = RandomForestClassifier(max_depth=2, random_state=0)

#create a dictionary of all values we want to test for n_estimators

#params_rf = {'n_neighbors': [50, 100, 200]}

#use gridsearch to test all values for n_estimators

#rf_gs = GridSearchCV(rf, params_rf, cv=5)

#fit model to training data

rf.fit(X_train, y_train)

#save best model

#rf_best = rf.best_estimator_

#check best n_estimators value

#print(rf.best_params_)





#create a dictionary of our models

estimators=[('knn', knn_best), ('rf', rf), ('log_reg', log_reg)]

#create our voting classifier, inputting our models

ensemble = VotingClassifier(estimators, voting='hard')



#fit model to training data

ensemble.fit(X_train, y_train)

#test our model on the test data

ensemble.score(X_test, y_test)