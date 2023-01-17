#import all packages

import numpy as np 

import pandas as pd 



import os

print(os.listdir("../input"))





import seaborn as sns

import scikitplot as skplt

import matplotlib.pyplot as plt

import matplotlib.gridspec as gridspec



from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import RobustScaler

from imblearn.under_sampling import RandomUnderSampler



from imblearn.under_sampling import InstanceHardnessThreshold

from imblearn.over_sampling import RandomOverSampler

from imblearn.over_sampling import SMOTE

from imblearn.under_sampling import RandomUnderSampler

from imblearn.pipeline import make_pipeline

from imblearn.metrics import specificity_score





from sklearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.model_selection import StratifiedShuffleSplit



from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import train_test_split



from sklearn.metrics import confusion_matrix

from sklearn.metrics import recall_score, roc_auc_score, roc_curve, accuracy_score, precision_score, make_scorer





import keras

from keras import backend as K

from keras.models import Sequential

from keras.layers import Activation, Dropout

from keras.layers.core import Dense

from keras.optimizers import Adamax

from keras.metrics import categorical_crossentropy
#read data and take first look at it

df = pd.read_csv('../input/creditcard.csv')

df.head()
#get information provided by pandas

df.info()
df.describe()
# graph to show imbalance of the dataset



sns.countplot(df["Class"])

plt.title("Fraud==1 vs. Non fraud==0")

plt.show()
#plot "Amount" column



fig, ax = plt.subplots(1,3, figsize=(20, 8))



df['Amount'].plot(ax=ax[0])

ax[0].set_title("Amount per Transaction")

ax[0].set_xlabel("Transaction Number")

ax[0].set_ylabel("Amount in Dollar")



df['Amount'].plot.hist(ax=ax[1], bins=200, color="r")

ax[1].set_title("Distribution of Amounts")

ax[1].set_xlabel("Amount in Dollar")



df['Amount'].plot.hist(ax=ax[2], bins=200, color="g")

ax[2].set_title("Distribution of Amounts closup")

ax[2].set_xlabel("Amount in Dollar")

ax[2].set_ylim([0,50])

plt.show()
#plot "Time" distribution



df["Time"].plot.hist(bins=50)

plt.title("Distribution of time since first transaction")

plt.xlabel("Time in seconds")

plt.axvline(x=12500, color='r')

plt.axvline(x=12500+86400, color='r')

plt.show()
#check the distributions of the pca-features for fraud and non fraud and compare them



pca_features = df.iloc[0:-1,0:28].columns

plt.figure(figsize=(35,30*4))

grid = gridspec.GridSpec(28, 1)

for i, feat in enumerate(df[pca_features]):

    ax = plt.subplot(grid[i])

    sns.distplot(df[feat][df.Class == 1], bins=200)

    sns.distplot(df[feat][df.Class == 0], bins=200)

    ax.set_xlabel('')

    ax.set_title('Feature : ' + str(feat))

plt.show()
# remove features where the distributions are for both classes are too similar

df = df.drop(['V28','V27','V26','V25','V24','V23','V22','V20','V15','V13','V8'], axis =1)
# converting "Time" to a time period of one day from 0 to 86400 seconds



second_day_indices = df[df["Time"]>86400].index

df.loc[second_day_indices, "Time"] = df.loc[second_day_indices, "Time"] - 86400



print("Minimum time is now {}".format(df["Time"].min()))

print("Maximum time is now {}".format(df["Time"].max()))
# scale "Time" and "Drop", delete original columns

robust_scaler = RobustScaler()



df["time_scaled"] = robust_scaler.fit_transform(df['Time'].values.reshape(-1,1))

df["amount_scaled"] = robust_scaler.fit_transform(df["Amount"].values.reshape(-1,1))

df.drop(["Time", "Amount"], inplace=True, axis=1)

df.head()
# create feature and target dataset

X = df.drop(["Class"], axis=1)

y = df["Class"]
rus = RandomUnderSampler(random_state=42)

X_ramdom_undersampled, y_random_undersampled = rus.fit_resample(X, y)





sns.countplot(y_random_undersampled)

plt.title("Distribution of Class for random undersampling")

plt.show()
iht = InstanceHardnessThreshold(random_state=42, estimator=LogisticRegression(

                                    solver='liblinear', multi_class='auto'))

X_iht_undersampled, y_iht_undersampled = iht.fit_resample(X, y)





sns.countplot(y_iht_undersampled)

plt.title("Distribution of Class for IHT undersampling")

plt.show()
ros = RandomOverSampler(random_state=42)

X_ramdom_oversampled, y_random_oversampled = ros.fit_resample(X, y)





sns.countplot(y_random_oversampled)

plt.title("Distribution of Class for random oversampling")

plt.show()
smote = SMOTE(random_state=42)

X_smote_oversampled, y_smote_oversampled = smote.fit_resample(X, y)





sns.countplot(y_smote_oversampled)

plt.title("Distribution of Class for smote oversampling")

plt.show()
# splitting in training and test set

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
# make a first prediction with RandomForestClassifier

clf_rf = RandomForestClassifier(n_estimators=10)

clf_rf.fit(X_train, y_train)

prediction = clf_rf.predict(X_test)
# print roc curve

y_true = y_test

y_probas = clf_rf.predict_proba(X_test)

skplt.metrics.plot_roc(y_true, y_probas)

plt.show()
#roc-auc-score

roc_auc_score(y_true,y_probas[:,1])
#print confusion matrix

skplt.metrics.plot_confusion_matrix(y_true, prediction)

plt.show()
#accuracy score

accuracy_score(y_test, prediction)
#recall score

recall_score(y_test.values, prediction)
# precision-recall curve

skplt.metrics.plot_precision_recall(y_test, y_probas)

plt.show()
# specificity score

specificity_score(y_test.values, prediction)
# build custom scorer for finding best hyperparameters

def custom_score(y_true, y_pred):

    

    conf_matrix = confusion_matrix(y_true, y_pred)

    #define measures

    recall = 0.75 * recall_score(y_true, y_pred) 

    specificy = 0.25 * conf_matrix[0,0]/conf_matrix[0,:].sum() 

    #punish low recall scores

    if recall < 0.75:

        recall -= 0.2

    return recall + specificy 

    

#initialize make_scorer

optimized_score = make_scorer(custom_score)
clf_rf = RandomForestClassifier(n_estimators=10)

rf_params = {"max_depth" : [5,7,10], 'criterion':['gini','entropy']}



clf_lr = LogisticRegression(solver='liblinear')

lr_params = {"C" : [0.001,0.01,0.1,1,10,100], "warm_start" :[True, False]}



clf_gb = GradientBoostingClassifier()

gb_params = {"learning_rate" : [0.001,0.01,0.1], 'criterion' : ['friedman_mse', 'mse']}



clf_svc = SVC(gamma='scale', probability=True)

svc_params = {'kernel' : ['linear', 'poly', 'rbf'], "C" : [0.001,0.01,0.1,1,10,100]}
# splitting in training and test set

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)



X_train = X_train.values

y_train = y_train.values

    

def find_best_estimator(estimator, params):

    gridsearch_cv = GridSearchCV(estimator, param_grid=params, cv=5, iid=False, scoring=optimized_score)



    sss = StratifiedShuffleSplit(n_splits = 5, test_size = 0.2, random_state=42)



    rus = RandomUnderSampler()

    



    recall_list = []

    specificity_list = []

    i=0

    print("")

    print( type(estimator).__name__)

    

    for train_index, test_index in sss.split(X_train, y_train):

        pipeline = make_pipeline(rus, gridsearch_cv)

        model = pipeline.fit(X_train[train_index], y_train[train_index])

        best_estimator = gridsearch_cv.best_estimator_

        prediction = best_estimator.predict(X_train[test_index])



        recall_list.append(recall_score(y_train[test_index], prediction))

        specificity_list.append(specificity_score(y_train[test_index], prediction))

        i=i+1

        

        print("Iteration {} out of 5 is finished".format(i))

      

    print("")

    print("recall on X_train split in train and test: {}".format(np.mean(recall_list)))

    print("Specificy on X_train split in train and test: {}".format(np.mean(specificity_list)))

    return best_estimator  
best_est_gb = find_best_estimator(clf_gb, gb_params)

best_est_lr = find_best_estimator(clf_lr, lr_params)

best_est_rf = find_best_estimator(clf_rf, rf_params)

best_est_svc = find_best_estimator(clf_svc, svc_params)
#change fond size for all plots

plt.rcParams.update({'font.size': 16})
# function to display all the metrics on the estimators

def evaluation_report(estimator):



    prediction = estimator.predict(X_test)

    prediction_proba = estimator.predict_proba(X_test)

       

    fig, ax = plt.subplots(1,3, figsize=(30, 8))



    fig.suptitle('Evaluation report for '+type(estimator).__name__, fontsize=16)



    skplt.metrics.plot_precision_recall(y_test, prediction_proba, ax=ax[0])

    ax[0].set_title("Precision-recall-curve")



    skplt.metrics.plot_roc(y_test, prediction_proba, ax=ax[1])

    ax[1].set_title("ROC-curve")



    skplt.metrics.plot_confusion_matrix(y_test, prediction, ax=ax[2])

    ax[2].set_title("Confusion-matrix")

    plt.show()

    print('The recall is {}'.format(recall_score(y_test.values, prediction)))

    print('The specificity is {}'.format(specificity_score(y_test.values, prediction)))

    print('The accuracy is {}'.format(accuracy_score(y_test, prediction)))

    print('The AUC-score is {}'.format(roc_auc_score(y_test,prediction_proba[:,1])))
evaluation_report(best_est_svc)
evaluation_report(best_est_lr)
evaluation_report(best_est_gb)
evaluation_report(best_est_rf)
# oversampling model with SMOTE

X_smote_oversampled, y_smote_oversampled = SMOTE().fit_resample(X_train, y_train)

n_inputs = X_smote_oversampled.shape[1]



oversample_model = Sequential([

    Dense(n_inputs, input_shape=(n_inputs, ), activation='relu'),

    Dense(32, activation='relu'),

    Dropout(0.5),

    Dense(64, activation='relu'),

    Dropout(0.5),

    Dense(32, activation='relu'),

    Dropout(0.5),    

    Dense(2, activation='softmax')

])



oversample_model.summary()



# compile model

oversample_model.compile(Adamax(lr=0.01), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# fit model

oversample_model.fit(X_smote_oversampled, y_smote_oversampled, validation_split=0.2, batch_size=25, epochs=15, shuffle=True, verbose=2)
# create same model with RandomUnderSampler

X_rus_undersampled, y_rus_undersampled = RandomUnderSampler().fit_resample(X_train, y_train)

n_inputs = X_rus_undersampled.shape[1]



undersample_model = Sequential([

    Dense(n_inputs, input_shape=(n_inputs, ), activation='relu'),

    Dense(32, activation='relu'),

    Dropout(0.5),

    Dense(64, activation='relu'),

    Dropout(0.5),

    Dense(32, activation='relu'),

    Dropout(0.5),    

    Dense(2, activation='softmax')

])
# compile model

undersample_model.compile(Adamax(lr=0.01), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# fit model

undersample_model.fit(X_rus_undersampled, y_rus_undersampled, validation_split=0.2, batch_size=25, epochs=15, shuffle=True, verbose=2)
# the prediction and probas in the evaluation report have to be adjusted slightly

def evaluation_report_keras(estimator):



    prediction = estimator.predict_classes(X_test, batch_size=200, verbose=0)

    prediction_proba = estimator.predict(X_test, batch_size=200, verbose=0)

       

    fig, ax = plt.subplots(1,3, figsize=(30, 8))



    

    skplt.metrics.plot_precision_recall(y_test, prediction_proba, ax=ax[0])

    ax[0].set_title("Precision-recall-curve")



    skplt.metrics.plot_roc(y_test, prediction_proba, ax=ax[1])

    ax[1].set_title("ROC-curve")



    skplt.metrics.plot_confusion_matrix(y_test, prediction, ax=ax[2])

    ax[2].set_title("Confusion-matrix")

    plt.show()

    print('The recall is {}'.format(recall_score(y_test.values, prediction)))

    print('The specificity is {}'.format(specificity_score(y_test.values, prediction)))

    print('The accuracy is {}'.format(accuracy_score(y_test, prediction)))

    print('The AUC-score is {}'.format(roc_auc_score(y_test,prediction_proba[:,1])))
# show evaluation report SMOTE-Keras-model

evaluation_report_keras(oversample_model)
# show evaluation report RandomUnderSampler-Keras-model

evaluation_report_keras(undersample_model)