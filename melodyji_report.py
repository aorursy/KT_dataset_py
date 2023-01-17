import numpy as np

import pandas as pd

creditcard = pd.read_csv("../input/creditcardfraud/creditcard.csv")

creditcard.info()
creditcard.head()
import matplotlib.pyplot as plt

count=pd.value_counts(creditcard['Class'],sort=True).sort_index()

count
ax=count.plot(kind='bar')

plt.title('Fraud Distribution')

plt.xlabel('Class')

plt.ylabel('Count')

plt.show()
from sklearn.preprocessing import StandardScaler, MinMaxScaler



creditcard['normAmount']  = creditcard['Amount'] 

cols_to_norm = ['Amount']

cols_new = ['normAmount']

creditcard[cols_new] = StandardScaler().fit_transform(creditcard[cols_to_norm])

creditcard = creditcard.drop(['Time', 'Amount'], axis = 1)

creditcard.head()
# get the cound from previous result. i.e. 492 for the fraudCount

fraudCount = count[1]

normalCount = count[0]



# locate the indices for the records

fraudIndex = creditcard[creditcard.Class == 1].index

normalIndex = creditcard[creditcard.Class == 0].index



# generate random normal records same number with fraud records

randomNormalIndex = np.random.choice(normalIndex, fraudCount, replace = False)



# merge them together

underSampleIndex = np.concatenate([np.array(fraudIndex), np.array(randomNormalIndex)])

underSampleData = creditcard.iloc[underSampleIndex,:]



X_U = underSampleData.iloc[:, underSampleData.columns != 'Class']

Y_U = underSampleData.iloc[:, underSampleData.columns == 'Class']

Y_U
from sklearn.model_selection import train_test_split



X = creditcard.iloc[:, creditcard.columns != 'Class']

Y = creditcard.iloc[:, creditcard.columns == 'Class']



# train/test for the original dataset

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state = 0)



# train/test for undersampled dataset

X_U_train, X_U_test, Y_U_train, Y_U_test = train_test_split(X_U, Y_U, test_size = 0.3, random_state = 0)
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import KFold, cross_val_score

from sklearn.metrics import confusion_matrix,recall_score,classification_report
# this method iterate the k (for KFold cross valiation) and the C parameter (for the logistic regression)

# in order to get the best C

def Kfold_recall_scores(x_train_data,y_train_data):

    recallResult = pd.DataFrame(columns = ['k','c','MeanRecall'])

    # try k for 3, 4, 5, 6

    for k in range(3,7):

        fold = KFold(k,shuffle=False) 

        c_params = [0.01,0.02,0.1,0.2,1,2,10,20,100,1000]

        for c in c_params:

            recalls = []

            for trainIndices, testIndices in fold.split(x_train_data):

                lrmodel = LogisticRegression(C = c, penalty='l2', max_iter=1000)

                lrmodel.fit(x_train_data.iloc[trainIndices,:],y_train_data.iloc[trainIndices,:].values.ravel())

                lrpred=lrmodel.predict(x_train_data.iloc[testIndices,:])

                recall = recall_score(y_train_data.iloc[testIndices,:],lrpred)

                recalls.append(recall)

            df = pd.DataFrame([[k, c, np.mean(recalls)]], columns = ['k','c','MeanRecall'])

            recallResult = recallResult.append(df, ignore_index=True)      

    return recallResult
recallResult = Kfold_recall_scores(X_U_train,Y_U_train)

recallResult
import itertools

# this method plots the confusion matrix

def plot_cm(cm, classes,

                          title='Confusion matrix',

                          cmap=plt.cm.Blues):

    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=0)

    plt.yticks(tick_marks, classes)



    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, cm[i, j],

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black")



    plt.tight_layout()

    plt.ylabel('True/False')

    plt.xlabel('Predict')
# in order to reuse the code, this method passes the X and Y and display the plot

def predict_and_plot(lr, X, Y, label = ''):

    y_pred = lr.predict(X.values)

    cm = confusion_matrix(Y,y_pred)

    np.set_printoptions(precision=2)



    # Recall = TP/(TP+FN)

    print("Recall in " + label + " test dataset: ", cm[1,1]/(cm[1,0]+cm[1,1]))

    # Precision = TP/(TP+FP)

    print("Precision  in " + label + " test dataset: ", cm[1,1]/(cm[0,1]+cm[1,1]))



    class_names = [0,1]

    plt.figure()

    plot_cm(cm, classes=class_names,title='Confusion Matrix')

    plt.show()
lr = LogisticRegression(C = 10, penalty = 'l2', max_iter=1000)

lr.fit(X_U_train,Y_U_train.values.ravel())
predict_and_plot(lr, X_U_test, Y_U_test, "undersampling")
predict_and_plot(lr, X_test, Y_test, "original")
from imblearn.over_sampling import SMOTE

from sklearn.ensemble import RandomForestClassifier

X = creditcard.iloc[:, creditcard.columns != 'Class']

Y = creditcard.iloc[:, creditcard.columns == 'Class']



# train/test for the original dataset

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state = 0)



oversampler=SMOTE(random_state=0)

X_O_train,Y_O_train=oversampler.fit_sample(X_train,Y_train['Class'])
lr = LogisticRegression(C = 10, penalty = 'l2', max_iter=1000)

lr.fit(X_O_train,Y_O_train.values.ravel())
predict_and_plot(lr, X_test, Y_test, "oversampling")
# rewrite the method to include the threshold parameter

def predict_and_plot_with_threshold(lr, X, Y, threshold, label = ''):

    y_pred = lr.predict_proba(X.values)

    

    # udpate the threshold here

    y_pred = y_pred[:,1] > threshold

    cm = confusion_matrix(Y,y_pred)

    np.set_printoptions(precision=2)

    

    print ("---------------threshold " + str(threshold) + "-------------------------------")

    # Recall = TP/(TP+FN)

    print("Recall in " + label + " test dataset: ", cm[1,1]/(cm[1,0]+cm[1,1]))

    # Precision = TP/(TP+FP)

    print("Precision in " + label + " test dataset: ", cm[1,1]/(cm[0,1]+cm[1,1]))



    class_names = [0,1]

    plt.figure()

    plot_cm(cm, classes=class_names,title='Threshold: ' + str(threshold))

    plt.show()
lr = LogisticRegression(C = 10, penalty = 'l2')

lr.fit(X_O_train,Y_O_train.values.ravel())

thresholds = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]

for t in thresholds:

    predict_and_plot_with_threshold(lr, X_test, Y_test, t, "original")
import time

start = time.time()

rf = RandomForestClassifier(n_estimators=100)

# log how long for training

rf.fit(X_O_train, Y_O_train.values.ravel())

end = time.time()

elapsed = end - start

print("training time: " + time.strftime("%H:%M:%S", time.gmtime(elapsed)))

rf_pred = rf.predict(X_test)

predict_and_plot(rf, X_test, Y_test, "original")
from sklearn.svm import SVC

svm=SVC(kernel='rbf', C=10)

start = time.time()

svm.fit(X_O_train, Y_O_train.values.ravel())

end = time.time()

elapsed = end - start

print("training time: " + time.strftime("%H:%M:%S", time.gmtime(elapsed)))

predict_and_plot(svm, X_test, Y_test, "original")
from sklearn.ensemble import GradientBoostingClassifier



gb = GradientBoostingClassifier(learning_rate=1, max_depth=3, n_estimators=100)

start = time.time()

gb.fit(X_O_train, Y_O_train.values.ravel())

end = time.time()

elapsed = end - start

print("training time: " + time.strftime("%H:%M:%S", time.gmtime(elapsed)))

predict_and_plot(svm, X_test, Y_test, "original")
from sklearn.ensemble import RandomForestClassifier, VotingClassifier

es = VotingClassifier(estimators=[('lr', lr), ('rf', rf)], voting='soft')

start = time.time()

es = es.fit(X_O_train, Y_O_train)

end = time.time()

elapsed = end - start

print("training time: " + time.strftime("%H:%M:%S", time.gmtime(elapsed)))

predict_and_plot(es, X_test, Y_test, "original")
from sklearn.ensemble import RandomForestClassifier, VotingClassifier

es = VotingClassifier(estimators=[('lr', lr), ('rf', rf)], voting='hard')

start = time.time()

es = es.fit(X_O_train, Y_O_train)

end = time.time()

elapsed = end - start

print("training time: " + time.strftime("%H:%M:%S", time.gmtime(elapsed)))

predict_and_plot(es, X_test, Y_test, "original")
import joblib

# save the models

joblib.dump(rf, 'rf_model.pkl')

joblib.dump(es, 'es_model.pkl')
models = {}

for model in ['rf', 'es']:

    models[model] = joblib.load('{}_model.pkl'.format(model))
predict_and_plot(models['rf'], X_test, Y_test, "original")
predict_and_plot(models['es'], X_test, Y_test, "original")