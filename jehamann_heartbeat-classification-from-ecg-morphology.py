import warnings

warnings.filterwarnings('ignore')

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.linear_model import SGDClassifier

from sklearn.multiclass import OneVsOneClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import LinearSVC, SVC

from sklearn.model_selection import cross_val_predict

from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

from sklearn.preprocessing import StandardScaler

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import VotingClassifier



import seaborn as sn



hbeat_signals = pd.read_csv("../input/mitbih-arrhythmia-database-de-chazal-class-labels/DS1_signals.csv", header=None)

hbeat_labels = pd.read_csv("../input/mitbih-arrhythmia-database-de-chazal-class-labels//DS1_labels.csv", header=None)



print("+"*50)

print("Signals Info:")

print("+"*50)

print(hbeat_signals.info())

print("+"*50)

print("Labels Info:")

print("+"*50)

print(hbeat_labels.info())

print("+"*50)
hbeat_signals.head()
# Collect data of different hheartbeats in different lists

#class 0

cl_0_idx = hbeat_labels[hbeat_labels[0] == 0].index.values

cl_N = hbeat_signals.iloc[cl_0_idx]

#class 1

cl_1_idx = hbeat_labels[hbeat_labels[0] == 1].index.values

cl_S = hbeat_signals.iloc[cl_1_idx]

#class 2

cl_2_idx = hbeat_labels[hbeat_labels[0] == 2].index.values

cl_V = hbeat_signals.iloc[cl_2_idx]

#class 3

cl_3_idx = hbeat_labels[hbeat_labels[0] == 3].index.values

cl_F = hbeat_signals.iloc[cl_3_idx]



# make plots for the different hbeat classes

plt.subplot(221)

for n in range(3):

    cl_N.iloc[n].plot(title='Class N (0)', figsize=(10,8))

plt.subplot(222)

for n in range(3):

    cl_S.iloc[n].plot(title='Class S (1)')

plt.subplot(223)

for n in range(3):

    cl_V.iloc[n].plot(title='Class V (2)')

plt.subplot(224)

for n in range(3):

    cl_F.iloc[n].plot(title='Class F (3)')

#check if missing data

print("Column\tNr of NaN's")

print('+'*50)

for col in hbeat_signals.columns:

    if hbeat_signals[col].isnull().sum() > 0:

        print(col, hbeat_signals[col].isnull().sum()) 

joined_data = hbeat_signals.join(hbeat_labels, rsuffix="_signals", lsuffix="_labels")



#rename columns

joined_data.columns = [i for i in range(180)]+['class']
#get correlaction matrix

corr_matrix = joined_data.corr()
print('+'*50)

print('Top 10 high positively correlated features')

print('+'*50)

print(corr_matrix['class'].sort_values(ascending=False).head(10))

print('+'*50)

print('Top 10 high negatively correlated features')

print('+'*50)

print(corr_matrix['class'].sort_values().head(10))
%matplotlib inline



from pandas.plotting import scatter_matrix



#Take features with the larges correlations

features = [79,80,78,77]

scatter_matrix(joined_data[features], figsize=(20,15), c =joined_data['class'], alpha=0.5);
print('-'*20)

print('Class\t %')

print('-'*20)

print(joined_data['class'].value_counts()/len(joined_data))

joined_data.hist('class');

print('-'*20)
print("class\t%")

joined_data['class'].value_counts()/len(joined_data)
from sklearn.model_selection import StratifiedShuffleSplit



split = StratifiedShuffleSplit(n_splits=1, test_size=0.2,random_state=42)



for train_index, test_index in split.split(joined_data, joined_data['class']):

    strat_train_set = joined_data.loc[train_index]

    strat_test_set = joined_data.loc[test_index]    
print("class\t%")

strat_train_set['class'].value_counts()/len(strat_train_set)
def compare_conf_matrix_scores(models, X, y):

    """

    This function compares predictive scores and confusion matrices fro different ML algorithms

    """

    

    for i, model in enumerate(models):



        # perform Kfold cross-validation returning prediction scores of each test fold.

        labels_train_pred = cross_val_predict(model, X, y, cv=5)

        print('+'*50)

        print('Model {} Confusion matrix'.format(i+1))

        print('+'*50)

        print(confusion_matrix(y, labels_train_pred))

        print('+'*50)



        prec_score = precision_score(y, labels_train_pred, average='macro')

        rec_score = recall_score(y, labels_train_pred, average='macro')

        f1_sc = f1_score(y, labels_train_pred, average='macro')

        print('Precision score: {}\nRecall Score: {}\nf1 score: {}'.format(prec_score,rec_score, f1_sc))

    print('+'*50)

    

#produce labels and features sets for the training stage

strat_features_train = strat_train_set.drop('class', 1)

strat_labels_train = strat_train_set['class']
#initiate ML the classifiers



# one versus one clasifier

ova_clf = OneVsOneClassifier(SGDClassifier(random_state=42, n_jobs=-1))



#random forest

forest_clf = RandomForestClassifier(random_state=42, n_jobs=-1)



#Support vector machines

svm_clf = LinearSVC(random_state=42)

svc = SVC(decision_function_shape='ovo', random_state=42, max_iter=1000)



warnings.filterwarnings('ignore')



compare_conf_matrix_scores([ova_clf, forest_clf, svm_clf, svc], strat_features_train, strat_labels_train)
#initialize standardscaler instance

scaler = StandardScaler()



#standarized data, i.e,  substract mean and devides by variance

std_features = scaler.fit_transform(strat_features_train)
# make plots for the different hbeat classes (standarized)

fig = plt.figure(figsize=(10,8))

plt.subplot(221)

plt.plot(figsize=(10,8))

x = np.linspace(0,179,180)

for n in range(3):

    plt.title('Class N (0)')

    plt.plot(x, std_features[cl_0_idx[n]])

plt.subplot(222)

for n in range(3):

    plt.title('Class S (1)')

    plt.plot(x, std_features[cl_1_idx[n]])

plt.subplot(223)

for n in range(3):

    plt.title('Class V (2)')

    plt.plot(x, std_features[cl_2_idx[n]])

plt.subplot(224)

for n in range(3):

    plt.title('Class F (3)')

    plt.plot(x, std_features[cl_3_idx[n]])
fig= plt.figure(figsize=(25,15))

for n in range(9):

    plt.subplot(3,3,n+1)

    scatter = plt.scatter(std_features[:,(n+1)*10],std_features[:,-1*(n+1)*5], alpha=0.5, c=strat_labels_train)

    plt.xlabel('Feat. {}'.format((n+1)*10))

    plt.ylabel('Feat. {}'.format(180-1*(n+1)*5))

plt.rc('font', size=20)

plt.rc('legend', fontsize=20)

#plt.rc('axes', labelsize=40)

#plt.legend(*scatter.legend_elements(), loc="best", title="Classes");
warnings.filterwarnings('ignore')



compare_conf_matrix_scores([ova_clf, forest_clf, svm_clf, svc], std_features, strat_labels_train)
# K nearest neighbors

knn_clf = KNeighborsClassifier(n_jobs=-1)



#Gaussian Naive Bayes

gnb_clf = GaussianNB()



#Stochastic gradient classifier

sgd_clf = SGDClassifier(n_jobs=-1,random_state=42)



compare_conf_matrix_scores([knn_clf, gnb_clf, sgd_clf], std_features, strat_labels_train)
#parameter grid

forest_param_grid = {'n_estimators': [50,100,200,300], 'max_depth':[2,4,8]}

knn_param_grid = {'n_neighbors':[2,4,8,10], 'weights':['uniform', 'distance']}



warnings.filterwarnings('ignore')



#initialize classifiers

forest = RandomForestClassifier(random_state=42, n_jobs=-1)

knn = KNeighborsClassifier(n_jobs=-1)



#initialize grid search

forest_grid_search = GridSearchCV(forest, forest_param_grid, cv=5, scoring="f1_macro")

knn_grid_search = GridSearchCV(knn, knn_param_grid, cv=5, scoring="f1_macro")



#fit classifiers using gridsearch

forest_grid_search.fit(std_features, strat_labels_train)

knn_grid_search.fit(std_features, strat_labels_train)



print("+"*50)

print("Model\t\tBest params\t\tBest score")

print("-"*50)

print("Random Forest\t\t", forest_grid_search.best_params_, forest_grid_search.best_score_)

print("-"*50)

print("KNN\t\t", knn_grid_search.best_params_, knn_grid_search.best_score_)

print("+"*50)
forest_param_grid = {'n_estimators': [158,160,162], 'max_depth':[83,80,87]}

forest_grid_search = GridSearchCV(forest, forest_param_grid, cv=5, scoring="f1_macro")

forest_grid_search.fit(std_features, strat_labels_train)



print("+"*50)

print('Model\t\tBest params\t\tBest score')

print("-"*50)

print("Random Forest\t\t", forest_grid_search.best_params_, forest_grid_search.best_score_)

print("+"*50)
#parameter grid

svc_param_grid = {'C':[10], 'gamma':[0.1,1,10]}



warnings.filterwarnings('ignore')



#initialize classifiers

svc = SVC(kernel='rbf',decision_function_shape='ovo',random_state=42, max_iter = 500)



#initialize grid search

svc_grid_search = GridSearchCV(svc, svc_param_grid, cv=3, scoring="f1_macro")



#fit classifiers using gridsearch

svc_grid_search.fit(std_features, strat_labels_train)



print("+"*50)

print('Model\t\tBest params\t\tBest score')

print("-"*50)

print("SVC\t\t", svc_grid_search.best_params_, svc_grid_search.best_score_)

print("+"*50)

best_forest = forest_grid_search.best_estimator_

best_knn = knn_grid_search.best_estimator_

best_svc = svc_grid_search.best_estimator_



compare_conf_matrix_scores([best_forest, best_knn, best_svc], std_features, strat_labels_train)
#init scaler

scaler = StandardScaler()



#fit scaler to train data

scaler.fit(strat_features_train)



#produce labels and features sets for the test stage

strat_features_test = strat_test_set.drop('class', 1)

strat_labels_test = strat_test_set['class']



#transform the test data

std_features_test = scaler.transform(strat_features_test)



#predict values for the test data

forest_pred = best_forest.predict(std_features_test)

knn_pred = best_knn.predict(std_features_test)

svc_pred = best_svc.predict(std_features_test)



#determine f1 score

forest_f1 = f1_score(strat_labels_test, forest_pred, average='macro')

knn_f1 = f1_score(strat_labels_test, knn_pred, average='macro')

svc_f1 = f1_score(strat_labels_test, svc_pred, average='macro')



#determine confusion matrix

print('+'*50)

print('Random Forest Confusion matrix (f1 score: {})'.format(forest_f1))

print('+'*50)

print(confusion_matrix(strat_labels_test, forest_pred))

print('+'*50)

print('KNN Confusion matrix (f1 score: {})'.format(knn_f1))

print('+'*50)

print(confusion_matrix(strat_labels_test, knn_pred))

print('+'*50)

print('SVC Confusion matrix (f1 score: {})'.format(svc_f1))

print('+'*50)

print(confusion_matrix(strat_labels_test, svc_pred))

#initialize ensemble

ensemble=VotingClassifier(estimators=[('Random Forest', best_forest), ('KNN', best_knn), ('SVC', best_svc)], voting='hard')



#fit ensemble

ensemble.fit(std_features,strat_labels_train)



compare_conf_matrix_scores([ensemble], std_features, strat_labels_train)
#predict values for the test data

ensemble_pred = ensemble.predict(std_features_test)



#determine f1 score

ensemble_f1 = f1_score(strat_labels_test, ensemble_pred, average='macro')



#determine confusion matrix

print('+'*50)

print('Ensemble Confusion matrix (f1 score: {})'.format(ensemble_f1))

print('+'*50)

print(confusion_matrix(strat_labels_test, ensemble_pred))

print('+'*50)
import numpy as np

import matplotlib.pyplot as plt



h = 0.155 # step size in the mesh



# we create an instance Classifier and fit the data. We are picking features 70 and 145.

X, y = std_features[:,[70,145]], strat_labels_train



#initialize classifiers

clf1 = RandomForestClassifier(max_depth=83,n_estimators=158, n_jobs=-1,random_state=42)

clf2 = KNeighborsClassifier(n_jobs=-1, n_neighbors=4,weights='distance')

clf3 = SVC(C=10,decision_function_shape='ovo', gamma=0.1, kernel='rbf', max_iter=500, random_state=42)

clf4 = ensemble=VotingClassifier(estimators=[('Random Forest', clf1), ('KNN', clf2), ('SVC', clf3)], voting='hard')

#fit classifiers

clf1.fit(X,y)

clf2.fit(X,y)

clf3.fit(X,y)

clf4.fit(X,y)



# Plot the decision boundary. For that, we will assign a color to each point in the mesh [x_min, x_max]x[y_min, y_max].

x_min, x_max = X[:, 0].min(), X[:, 0].max()

y_min, y_max = X[:, 1].min(), X[:, 1].max()

xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))



#fig titles

tt =['Random Forest (depth=83)', 'KNN (k=4)','Kernel (RBF) SVM', 'Hard Voting']



fig= plt.figure(figsize=(20,15))



for idx, clf in enumerate([clf1, clf2, clf3, clf4]):



    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    Z = Z.reshape(xx.shape)

    

    #plot decision boundary

    plt.subplot(2,2,idx+1)

    plt.pcolormesh(xx, yy, Z, alpha=0.2)



    # Plot training points

    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.5)

    plt.xlim(xx.min(), xx.max())

    plt.ylim(yy.min(), yy.max())

    plt.xlabel('Feat. 70')

    plt.ylabel('Feat. 145')

    plt.title(tt[idx])

#    plt.legend(*scatter.legend_elements(), loc="best", title="Classes");    
import tensorflow as tf

from tensorflow import keras

from tensorflow.keras.utils import to_categorical

from tensorflow.keras.optimizers import SGD



#define parameters

batch_size = len(std_features)//300



#build model

model = keras.Sequential([

    keras.layers.Dense(200, activation='relu', input_shape=(180,)),

    keras.layers.Dense(100, activation='relu'),

    keras.layers.Dense(4, activation='softmax')

])



#transform the test data

strat_features_test = strat_test_set.drop('class', 1)

strat_labels_test = strat_test_set['class']



#standardize 

std_features_test = scaler.transform(strat_features_test)



#change labels to categorical, requaried to use 'categorical_crossentropy'

categorical_labels_train = to_categorical(strat_labels_train, num_classes=None)

categorical_labels_test = to_categorical(strat_labels_test, num_classes=None)



#stochastic gradient descent optimizer

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)



#compile model

model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])



#fit model and get scores

model.fit(std_features, categorical_labels_train,epochs=20,batch_size=batch_size)



score = model.evaluate(std_features_test, categorical_labels_test, batch_size=batch_size)
# check metrics on the test data

y_pred1 = model.predict(std_features_test)

y_pred = np.argmax(y_pred1, axis=1)



f1_sc = f1_score(strat_labels_test, y_pred , average="macro")

conf_mat = confusion_matrix(strat_labels_test, y_pred)



# Print f1, precision, and recall scores

print('+'*50)

print('Neural Network Confusion matrix (f1 score: {})'.format(f1_sc))

print('+'*50)

print(conf_mat)

print('+'*50)
#best knn model n_neighbors = 4, weights='distance', n_jobs=-1

#best random forest max_depth=83, n_estimators=158, n_jobs=-1

best_knn, best_forest, best_svc