# organize imports

from keras.models import Sequential

from keras.layers.core import Dense, Activation, Dropout

from sklearn.model_selection import train_test_split
import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

import warnings

warnings.filterwarnings('ignore')
# seed for reproducing same results

seed = 9

np.random.seed(seed)
# load pima indians dataset

df = pd.read_csv('/kaggle/input/parkinsons-data-set/parkinsons.data')

data = df.copy()
data.head()
def draw_missing_data_table(df):

    total = df.isnull().sum().sort_values(ascending=False)

    percent = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)

    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

    return missing_data



draw_missing_data_table(data)
predictors = data.drop(['name'], axis = 1)



predictors = predictors.drop(['status'], axis = 1)

X = predictors

Y = data['status']
# from imblearn.under_sampling import RandomUnderSampler

# rus = RandomUnderSampler(random_state=0, replacement=True)

# X_resampled, y_resampled = rus.fit_resample(X,Y)

# len(X_resampled)
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = .25, random_state = 7)



from sklearn.preprocessing import StandardScaler  

scaler = StandardScaler()

scaler.fit(X_train)

X_train = scaler.transform(X_train)

X_test = scaler.transform(X_test)
data.describe(include='all')
sns.pairplot(data, hue='status')
data.hist(figsize=(28,28))
data.hist(bins=50, figsize=(28,28))

plt.figure(figsize=(20,15))

sns.heatmap(data.corr() ,annot=True)

plt.show()
data.plot(kind='box', subplots=True, layout=(5,5), sharex=False, sharey=False, figsize=(28, 28))

plt.show()
dataset = data

all_cols = list(dataset.columns.values)

all_cols.remove('name')

all_cols
bins = np.linspace(-10, 10, 30)

plt.figure(figsize=(15,30))

for i in range(1, 22):

    plt.subplot(14, 2, i)

    col = all_cols[i]

    plt.title(all_cols[i])

    plt.hist(data[col][data.status == 1], alpha=0.5, label='x')

    plt.hist(data[col][data.status == 0], alpha=0.5, label='y')

    



plt.legend(loc='upper right')

plt.show()
from sklearn import metrics

from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier()

model.fit(X_train, Y_train)

y_pred = model.predict(X_test)

# summarize the fit of the model

print("KNeighborsClassifier: ")

print(metrics.accuracy_score(Y_test, y_pred))
from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier(hidden_layer_sizes=(22,18,12),max_iter=1500)

mlp.fit(X_train,Y_train)

y_pred = mlp.predict(X_test)

print("MLPClassifier: ")

print(metrics.accuracy_score(Y_test, y_pred))

from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score

model = XGBClassifier()

model.fit(X_train, Y_train)

y_pred = model.predict(X_test)

print("XGBClassifier: ")

print(metrics.accuracy_score(Y_test, y_pred))
classifier = Sequential()

#First Hidden Layer

classifier.add(Dense(16, activation='relu', kernel_initializer='random_normal', input_dim=22))

#Second  Hidden Layer

classifier.add(Dense(8, activation='relu', kernel_initializer='random_normal'))

#Output Layer

classifier.add(Dense(1, activation='sigmoid', kernel_initializer='random_normal'))

#Compiling the neural network

classifier.compile(optimizer ='adam',loss='binary_crossentropy', metrics =['accuracy'])

#Fitting the data to the training dataset

classifier.fit(X_train,Y_train, batch_size=10, epochs=100)
# evaluate the model

tscores = classifier.evaluate(X_test, Y_test)

print("Test Accuracy: %.2f%%" %(tscores[1]*100))
trscores=classifier.evaluate(X_train, Y_train)

print("Train Accuracy: %.2f%%" %(trscores[1]*100))
y_pred=classifier.predict(X_test)


classifier.save_weights("model0.h5")

print("Saved model to disk")
model_json = classifier.to_json()

with open("model0.json", "w") as json_file:

    json_file.write(model_json)
from keras.models import model_from_json



json_file = open('./model0.json', 'r')

loaded_model_json = json_file.read()

json_file.close()

loaded_model = model_from_json(loaded_model_json)

# load weights into new model

loaded_model.load_weights("./model0.h5")

print("Loaded model from disk")
# evaluate loaded model on test data

loaded_model.compile(optimizer ='adam',loss='binary_crossentropy', metrics =['accuracy'])

score = loaded_model.evaluate(X_test, Y_test, verbose=0)

print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))


from xgboost import Booster



model._Booster.save_model('model.bin')



def load_xgb_model():

    _m = XGBClassifier()

    _b = Booster()

    _b.load_model('model.bin')

    _m._Booster = _b

    return _m



model = load_xgb_model()
model.fit(X_train, Y_train)
y_pred = model.predict(X_test)
model.score(X_test , Y_test) 
dataset = data
dataset['status'].value_counts(normalize=True)*100
# from pandas.plotting import scatter_matrix



# attributes = list(data.columns.values)

# scatter_matrix(data[attributes], figsize=(500.0, 500.0))
from sklearn import datasets

from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import GaussianNB

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import VotingClassifier





clf1 = LogisticRegression(random_state=1)

clf2 = RandomForestClassifier(n_estimators=50, random_state=1)

clf3 = GaussianNB()



eclf = VotingClassifier(

    estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)],

    voting='hard')



for clf, label in zip([clf1, clf2, clf3, eclf], ['Logistic Regression', 'Random Forest', 'naive Bayes', 'Ensemble']):

    scores = cross_val_score(clf, X, Y, scoring='accuracy', cv=5)

    print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))

    

    
eclf.fit(X_train,Y_train)

y_pred=eclf.predict(X_test)

eclf.score(X_test , Y_test) 
from sklearn import datasets

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from itertools import product

from sklearn.ensemble import VotingClassifier



# Loading some example data



# Training classifiers

clf1 = DecisionTreeClassifier(max_depth=4)

clf2 = KNeighborsClassifier(n_neighbors=7)

clf3 = SVC(kernel='rbf', probability=True)

eclf = VotingClassifier(estimators=[('dt', clf1), ('knn', clf2), ('svc', clf3)],

                        voting='soft', weights=[4, 5, 3])



clf1 = clf1.fit(X_train, Y_train)

clf2 = clf2.fit(X_train, Y_train)

clf3 = clf3.fit(X_train, Y_train)

eclf = eclf.fit(X_train, Y_train)
eclf.score(X_test, Y_test)
from sklearn.model_selection import cross_val_score

from sklearn.datasets import make_blobs

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.tree import DecisionTreeClassifier





clf = DecisionTreeClassifier(max_depth=None, min_samples_split=2,

    random_state=0)

clf.fit(X_train,Y_train)

y_pred=clf.predict(X_test)

clf.score(X_test , Y_test) 
clf = RandomForestClassifier(n_estimators=10, max_depth=None,

    min_samples_split=2, random_state=0)

scores = cross_val_score(clf, X, Y, cv=5)

scores.mean()

clf.fit(X_train,Y_train)

y_pred=clf.predict(X_test)

clf.score(X_test , Y_test) 
clf = ExtraTreesClassifier(n_estimators=100, max_depth=10,

    min_samples_split=2, random_state=0)

clf.fit(X_train,Y_train)

y_pred=clf.predict(X_test)

clf.score(X_test , Y_test) 


# Voting Ensemble for Classification

import pandas

from sklearn import model_selection

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.svm import SVC

from sklearn.ensemble import VotingClassifier



seed = 7

kfold = model_selection.KFold(n_splits=10, random_state=seed)

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
ensemble.fit(X_train,Y_train)

y_pred=ensemble.predict(X_test)

ensemble.score(X_test , Y_test) 
# Stochastic Gradient Boosting Classification

import pandas

from sklearn import model_selection

from sklearn.ensemble import GradientBoostingClassifier

# url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"

# names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']

# dataframe = pandas.read_csv(url, names=names)

# array = dataframe.values

# X = array[:,0:8]

# Y = array[:,8]

# seed = 7

num_trees = 100

kfold = model_selection.KFold(n_splits=10, random_state=seed)

model = GradientBoostingClassifier(n_estimators=num_trees, random_state=seed)

results = model_selection.cross_val_score(model, X, Y, cv=kfold)

print(results.mean())
results
clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,max_depth=1, random_state=0).fit(X_train, Y_train)

clf.score(X_test, Y_test)
clf = RandomForestClassifier(n_estimators=10, max_depth=10,

    min_samples_split=2, random_state=0)

clf.fit(X_train,Y_train)

y_pred=clf.predict(X_test)

clf.score(X_test , Y_test) 


import itertools

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import matplotlib.gridspec as gridspec



from sklearn import datasets



from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB 

from sklearn.ensemble import RandomForestClassifier

from mlxtend.classifier import StackingClassifier



from sklearn.model_selection import cross_val_score, train_test_split



from mlxtend.plotting import plot_learning_curves

from mlxtend.plotting import plot_decision_regions
from sklearn.model_selection import RandomizedSearchCV,GridSearchCV

from scipy.stats import randint



est = RandomForestClassifier(n_jobs=-1)

rf_p_dist={'max_depth':[1,2,3,4,5,6,7,8,9,10],

           'n_estimators':[1,2,3,4,5,10,11,12,13,100,200,300,400,500],

              'max_features':[1,2,3,4,5],

               'criterion':['gini','entropy'],

               'bootstrap':[True,False],

               'min_samples_leaf':[1,2,3,4,5]

         

              

              }



rs=GridSearchCV(estimator=est,param_grid=rf_p_dist)

y = np.array(Y_train)



x = np.array(X_train)

# iris = datasets.load_iris()

# X, y = iris.data[:, 1:3], iris.target





from sklearn.model_selection import GridSearchCV

from sklearn.neighbors import KNeighborsClassifier

#create new a knn model

knn = KNeighborsClassifier()

#create a dictionary of all values we want to test for n_neighbors

params_knn = {'n_neighbors': np.arange(1, 2)}

#use gridsearch to test all values for n_neighbors

knn_gs = GridSearchCV(knn, params_knn, cv=5)





from sklearn.ensemble import AdaBoostClassifier



# abcl = AdaBoostClassifier( n_estimators= 50)

# clf2 = RandomForestClassifier(random_state=1)

# clf3 = abcl



clf2 = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,max_depth=1, random_state=0)

clf3 = RandomForestClassifier(n_estimators=10, max_depth=None,min_samples_split=2, random_state=0)



lr = LogisticRegression()

sclf = StackingClassifier(classifiers=[clf2, clf3], 

                          meta_classifier=lr)
label = ['KNN', 'Random Forest', 'Naive Bayes', 'Stacking Classifier']

clf_list = [clf1, clf2, clf3, sclf]

    

fig = plt.figure(figsize=(10,8))

gs = gridspec.GridSpec(2, 2)

grid = itertools.product([0,1],repeat=2)



clf_cv_mean = []

clf_cv_std = []

for clf, label, grd in zip(clf_list, label, grid):

        

    scores = cross_val_score(clf, x, y, cv=3, scoring='accuracy')

    print("Accuracy: %.2f (+/- %.2f) [%s]" %(scores.mean(), scores.std(), label))

    clf_cv_mean.append(scores.mean())

    clf_cv_std.append(scores.std())

        

    clf.fit(x, y)



plt.show()
y_pred = clf.predict(X_test)
clf.score(X_test , Y_test) 
#plot learning curves

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

    

plt.figure()

plot_learning_curves(X_train, Y_train, X_test, Y_test, sclf, print_model=False, style='ggplot')

plt.show()
import itertools

import numpy as np



import seaborn as sns

import matplotlib.pyplot as plt

import matplotlib.gridspec as gridspec



from sklearn import datasets



from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import LogisticRegression



from sklearn.ensemble import AdaBoostClassifier

from sklearn.model_selection import cross_val_score, train_test_split



from mlxtend.plotting import plot_learning_curves

from mlxtend.plotting import plot_decision_regions
    

#XOR dataset

#X = np.random.randn(200, 2)

#y = np.array(map(int,np.logical_xor(X[:, 0] > 0, X[:, 1] > 0)))

    

clf = DecisionTreeClassifier(criterion='entropy', max_depth=1)



num_est = [1, 2, 3, 10]

label = ['AdaBoost (n_est=1)', 'AdaBoost (n_est=2)', 'AdaBoost (n_est=3)', 'AdaBoost (n_est=10)']
fig = plt.figure(figsize=(10, 8))

gs = gridspec.GridSpec(2, 2)

grid = itertools.product([0,1],repeat=2)



for n_est, label, grd in zip(num_est, label, grid):     

    boosting = AdaBoostClassifier(base_estimator=clf, n_estimators=n_est)   

    boosting.fit(X, Y)



plt.show()


#Ensemble Size

num_est = map(int, np.linspace(1,100,20))

bg_clf_cv_mean = []

bg_clf_cv_std = []

for n_est in num_est:

    ada_clf = AdaBoostClassifier(base_estimator=clf, n_estimators=n_est)

    scores = cross_val_score(ada_clf, X, Y, cv=3, scoring='accuracy')

    bg_clf_cv_mean.append(scores.mean())

    bg_clf_cv_std.append(scores.std())


%matplotlib inline



import itertools

import numpy as np



import seaborn as sns

import matplotlib.pyplot as plt

import matplotlib.gridspec as gridspec



from sklearn import datasets



from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier



from sklearn.ensemble import BaggingClassifier

from sklearn.model_selection import cross_val_score, train_test_split



from mlxtend.plotting import plot_learning_curves

from mlxtend.plotting import plot_decision_regions



np.random.seed(0)


    

clf1 = DecisionTreeClassifier(criterion='entropy', max_depth=1)

clf2 = KNeighborsClassifier(n_neighbors=1)    



bagging1 = BaggingClassifier(base_estimator=clf1, n_estimators=10, max_samples=0.8, max_features=0.8)

bagging2 = BaggingClassifier(base_estimator=clf2, n_estimators=10, max_samples=0.8, max_features=0.8)


label = ['Decision Tree', 'K-NN', 'Bagging Tree', 'Bagging K-NN']

clf_list = [clf1, clf2, bagging1, bagging2]



fig = plt.figure(figsize=(10, 8))

gs = gridspec.GridSpec(2, 2)

grid = itertools.product([0,1],repeat=2)



for clf, label, grd in zip(clf_list, label, grid):        

    scores = cross_val_score(clf, X_train, Y_train, cv=3, scoring='accuracy')

    print("Accuracy: %.2f (+/- %.2f) [%s]" %(scores.mean(), scores.std(), label))

        

    clf.fit(X_train, Y_train)





plt.show()
clf.score(X_test,Y_test)
from sklearn.ensemble import RandomForestClassifier

rfcl = RandomForestClassifier(n_estimators = 70)

rfcl = rfcl.fit(X_train, Y_train)

y_pred = rfcl.predict(X_test)

rfcl.score(X_test , Y_test)
rfcl.score(X_test , Y_test)
from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier()

dt.fit(X_train, Y_train)

y_pred = dt.predict(X_test)
from sklearn.metrics import roc_curve, auc

false_positive_rate, true_positive_rate, thresholds = roc_curve(Y_test, y_pred)

roc_auc = auc(false_positive_rate, true_positive_rate)

roc_auc
max_depths = np.linspace(1, 32, 32, endpoint=True)

train_results = []

test_results = []

for max_depth in max_depths:

   dt = DecisionTreeClassifier(max_depth=max_depth)

   dt.fit(X_train, Y_train)

   train_pred = dt.predict(X_train)

   false_positive_rate, true_positive_rate, thresholds = roc_curve(Y_train, train_pred)

   roc_auc = auc(false_positive_rate, true_positive_rate)

   # Add auc score to previous train results

   train_results.append(roc_auc)

   y_pred = dt.predict(X_test)

   false_positive_rate, true_positive_rate, thresholds = roc_curve(Y_test, y_pred)

   roc_auc = auc(false_positive_rate, true_positive_rate)

   # Add auc score to previous test results

   test_results.append(roc_auc)

from matplotlib.legend_handler import HandlerLine2D

line1, = plt.plot(max_depths, train_results, 'b', label='Train AUC')

line2, = plt.plot(max_depths, test_results, 'r', label='Test AUC')

plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})

plt.ylabel('AUC score')

plt.xlabel('Tree depth')

plt.show()
min_samples_splits = np.linspace(0.1, 1.0, 10, endpoint=True)

train_results = []

test_results = []

for min_samples_split in min_samples_splits:

   dt = DecisionTreeClassifier(min_samples_split=min_samples_split)

   dt.fit(X_train, Y_train)

   train_pred = dt.predict(X_train)

   false_positive_rate, true_positive_rate, thresholds =    roc_curve(Y_train, train_pred)

   roc_auc = auc(false_positive_rate, true_positive_rate)

   train_results.append(roc_auc)

   y_pred = dt.predict(X_test)

   false_positive_rate, true_positive_rate, thresholds = roc_curve(Y_test, y_pred)

   roc_auc = auc(false_positive_rate, true_positive_rate)

   test_results.append(roc_auc)

from matplotlib.legend_handler import HandlerLine2D

line1, = plt.plot(min_samples_splits, train_results, 'b', label='Train AUC')

line2, = plt.plot(min_samples_splits, test_results, 'r', label='Test AUC')

plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})

plt.ylabel('AUC score')

plt.xlabel('min samples split')

plt.show()

min_samples_leafs = np.linspace(0.1, 0.5, 5, endpoint=True)

train_results = []

test_results = []

for min_samples_leaf in min_samples_leafs:

   dt = DecisionTreeClassifier(min_samples_leaf=min_samples_leaf)

   dt.fit(X_train, Y_train)

   train_pred = dt.predict(X_train)

   false_positive_rate, true_positive_rate, thresholds = roc_curve(Y_train, train_pred)

   roc_auc = auc(false_positive_rate, true_positive_rate)

   train_results.append(roc_auc)

   y_pred = dt.predict(X_test)

   false_positive_rate, true_positive_rate, thresholds = roc_curve(Y_test, y_pred)

   roc_auc = auc(false_positive_rate, true_positive_rate)

   test_results.append(roc_auc)

from matplotlib.legend_handler import HandlerLine2D

line1, = plt.plot(min_samples_leafs, train_results, 'b', label='Train AUC')

line2, = plt.plot(min_samples_leafs, test_results, 'r', label='Test AUC')

plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})

plt.ylabel('AUC score')

plt.xlabel('min samples leaf')

plt.show()
max_features = list(range(1,X_train.shape[1]))

train_results = []

test_results = []

for max_feature in max_features:

   dt = DecisionTreeClassifier(max_features=max_feature)

   dt.fit(X_train, Y_train)

   train_pred = dt.predict(X_train)

   false_positive_rate, true_positive_rate, thresholds = roc_curve(Y_train, train_pred)

   roc_auc = auc(false_positive_rate, true_positive_rate)

   train_results.append(roc_auc)

   y_pred = dt.predict(X_test)

   false_positive_rate, true_positive_rate, thresholds = roc_curve(Y_test, y_pred)

   roc_auc = auc(false_positive_rate, true_positive_rate)

   test_results.append(roc_auc)

from matplotlib.legend_handler import HandlerLine2D

line1, = plt.plot(max_features, train_results, 'b', label='Train AUC')

line2, = plt.plot(max_features, test_results, 'r', label='Test AUC')

plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})

plt.ylabel('AUC score')

plt.xlabel('max features')

plt.show()
max_features
#Count mis-classified one

count_misclassified = (Y_test != y_pred).sum()

print('Misclassified samples: {}'.format(count_misclassified))
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB as gnb

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, confusion_matrix

from sklearn import tree

from os import system
from sklearn.ensemble import RandomForestClassifier

DecisionTreeClassifier(class_weight=None, criterion='entropy', max_depth=20,

            max_features=None, max_leaf_nodes=None,

            min_impurity_split=1e-07, min_samples_leaf=6,

            min_samples_split=1, min_weight_fraction_leaf=0.0,

            presort=False, random_state=0, splitter='best')

rfcl = RandomForestClassifier(n_estimators = 50)

rfcl = rfcl.fit(X_train, Y_train)

y_pred = rfcl.predict(X_test)

rfcl.score(X_test , Y_test)
from sklearn.ensemble import GradientBoostingClassifier

gbcl = GradientBoostingClassifier(n_estimators = 50, learning_rate = 0.05)

gbcl = gbcl.fit(X_train,Y_train)

y_pred = gbcl.predict(X_test)

gbcl.score(X_test , Y_test)
# Fitting classifier to the Training set

from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(max_depth=5, random_state=0)

classifier.fit(X_train, Y_train)
from sklearn.ensemble import AdaBoostClassifier

abcl = AdaBoostClassifier( n_estimators= 50)

abcl = abcl.fit(X_train,Y_train)

y_pred = abcl.predict(X_test)

abcl.score(X_test , Y_test)
print('Class 0', round(dataset['status'].value_counts()[0]/len(df) * 100,2), '% of the dataset-----', round(dataset['status'].value_counts()[0]))

print('Class 1', round(dataset['status'].value_counts()[1]/len(df) * 100,2), '% of the dataset-----', round(dataset['status'].value_counts()[1]))
sns.countplot('status',data=dataset)

plt.title('New Distributin Dataset')

plt.show()
# Fitting classifier to the Training set

from sklearn.svm import SVC

classifier = SVC(kernel = "rbf", random_state = 9)

classifier.fit(X_train, Y_train)
# Predicting the Test set results

y_pred = classifier.predict(X_test)

# Making the Confusion Matrix

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(Y_test, y_pred)

cm
classifier.score(X_train, Y_train)