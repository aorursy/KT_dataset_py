# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df=pd.read_csv("/kaggle/input/creditcardfraud/creditcard.csv")
df.head()
x=sns.countplot(df['Class'])
from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import RobustScaler

sk=StandardScaler()

rs=RobustScaler()

df['Time']=sk.fit_transform(df['Time'].values.reshape(-1,1))

df['Amount']=rs.fit_transform(df['Amount'].values.reshape(-1,1))


df = df.sample(frac=1)



# amount of fraud classes 492 rows.

fraud_df = df.loc[df['Class'] == 1]

non_fraud_df = df.loc[df['Class'] == 0][:492]



normal_distributed_df = pd.concat([fraud_df, non_fraud_df])



# Shuffle dataframe rows

new_df = normal_distributed_df.sample(frac=1, random_state=42)



new_df.head()
x=sns.countplot(new_df['Class'])
X=new_df.iloc[:,:-1].values

X.shape
Y=new_df.iloc[:,-1].values

Y.shape
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.10, random_state=42)
from sklearn.svm import SVC

svc = SVC(kernel='linear')

svc.fit(x_train,y_train)

pred_svc =svc.predict(x_test)
from sklearn import metrics



# Model Accuracy: how often is the classifier correct?

print("Accuracy:",metrics.accuracy_score(y_test, pred_svc))
from sklearn.metrics import classification_report,accuracy_score

print(classification_report(y_test,pred_svc))
from sklearn.metrics import confusion_matrix

confusion_matrix(y_test, pred_svc)
print("Precision:",metrics.precision_score(y_test, pred_svc))



# Model Recall: what percentage of positive tuples are labelled as such?

print("Recall:",metrics.recall_score(y_test, pred_svc))
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5)

knn.fit(x_train,y_train)

pred_knn=knn.predict(x_test)
from sklearn import metrics



# Model Accuracy: how often is the classifier correct?

print("Accuracy:",metrics.accuracy_score(y_test, pred_knn))
from sklearn.metrics import classification_report,accuracy_score

print(classification_report(y_test,pred_knn))
from sklearn.metrics import confusion_matrix

confusion_matrix(y_test, pred_knn)
print("Precision:",metrics.precision_score(y_test, pred_knn))

# Model Recall: what percentage of positive tuples are labelled as such?

print("Recall:",metrics.recall_score(y_test, pred_knn))
from sklearn import naive_bayes

NB = naive_bayes.GaussianNB()

NB.fit(x_train,y_train)

pred_nb=NB.predict(x_test)
from sklearn import metrics



# Model Accuracy: how often is the classifier correct?

print("Accuracy:",metrics.accuracy_score(y_test, pred_nb))
from sklearn.metrics import confusion_matrix

confusion_matrix(y_test, pred_nb)
from sklearn.metrics import classification_report,accuracy_score

print(classification_report(y_test,pred_nb))
print("Precision:",metrics.precision_score(y_test, pred_nb))

# Model Recall: what percentage of positive tuples are labelled as such?

print("Recall:",metrics.recall_score(y_test, pred_nb))
from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators=250)

rfc.fit(x_train, y_train)

pred_rfc = rfc.predict(x_test)
from sklearn import metrics



# Model Accuracy: how often is the classifier correct?

print("Accuracy:",metrics.accuracy_score(y_test, pred_rfc))
from sklearn import metrics



# Model Accuracy: how often is the classifier correct?

print("Accuracy:",metrics.accuracy_score(y_test, pred_rfc))
from sklearn.metrics import confusion_matrix

confusion_matrix(y_test, pred_rfc)
from sklearn.metrics import classification_report,accuracy_score

print(classification_report(y_test,pred_rfc))
print("Precision:",metrics.precision_score(y_test, pred_rfc))

# Model Recall: what percentage of positive tuples are labelled as such?

print("Recall:",metrics.recall_score(y_test, pred_rfc))
# fit model no training data

from xgboost import XGBClassifier

model = XGBClassifier()

model.fit(x_train, y_train)
# make predictions for test data

y_predxg = model.predict(x_test)
accuracy = accuracy_score(y_test, y_predxg)

print("Accuracy: %.2f%%" % (accuracy * 100.0))
from sklearn.model_selection import train_test_split

x_train, x_val, y_train, y_val  = train_test_split(x_train, y_train, test_size = 0.10,random_state=42)
from keras.models import Sequential

from keras.layers import Dense,Dropout

from keras.activations import relu,softmax

from keras.regularizers import l2
model = Sequential()

model.add(Dense(16, input_dim=30,kernel_regularizer=l2(0.01), activation='relu'))

model.add(Dense(32, kernel_regularizer=l2(0.01),activation='relu'))

model.add(Dense(48, kernel_regularizer=l2(0.01),activation='relu'))

# model.add(Dense(64, kernel_regularizer=l2(0.01),activation='relu',))

# model.add(Dense(128, kernel_regularizer=l2(0.01),activation='relu',))

model.add(Dropout(0.4))

model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
from keras.callbacks import ModelCheckpoint

checkpointer=ModelCheckpoint(filepath='Convolutional.hdf5',verbose=1,save_best_only=True)

history = model.fit(x_train, y_train, epochs=50, batch_size=16,validation_data=(x_val,y_val))
score=model.evaluate(x_test,y_test,verbose=1)               #evaluates the model

accuracy=100*score[1]                                       

print('Test accuracy is %.4f%%' % accuracy)
score=model.evaluate(x_train,y_train,verbose=1)               #evaluates the model

accuracy=100*score[1]                                       

print('Test accuracy is %.4f%%' % accuracy)
plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()
# summarize history for loss

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()
from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

classifiers = {

    "LogisiticRegression": LogisticRegression(),

    "KNearest": KNeighborsClassifier(),

    "Support Vector Classifier": SVC(),

    "DecisionTreeClassifier": DecisionTreeClassifier()

}
from sklearn.model_selection import cross_val_score





for key, classifier in classifiers.items():

    classifier.fit(x_train, y_train)

    training_score = cross_val_score(classifier, x_train, y_train, cv=5)

    print("Classifiers: ", classifier.__class__.__name__, "Has a training score of", round(training_score.mean(), 2) * 100, "% accuracy score")
from sklearn.model_selection import GridSearchCV





# Logistic Regression 

log_reg_params = {"penalty": ['l1', 'l2'], 'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}







grid_log_reg = GridSearchCV(LogisticRegression(), log_reg_params)

grid_log_reg.fit(x_train, y_train)

# We automatically get the logistic regression with the best parameters.

log_reg = grid_log_reg.best_estimator_



knears_params = {"n_neighbors": list(range(2,5,1)), 'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']}



grid_knears = GridSearchCV(KNeighborsClassifier(), knears_params)

grid_knears.fit(x_train, y_train)

# KNears best estimator

knears_neighbors = grid_knears.best_estimator_



# Support Vector Classifier

svc_params = {'C': [0.5, 0.7, 0.9, 1], 'kernel': ['rbf', 'poly', 'sigmoid', 'linear']}

grid_svc = GridSearchCV(SVC(), svc_params)

grid_svc.fit(x_train, y_train)



# SVC best estimator

svc = grid_svc.best_estimator_



# DecisionTree Classifier

tree_params = {"criterion": ["gini", "entropy"], "max_depth": list(range(2,4,1)), 

              "min_samples_leaf": list(range(5,7,1))}

grid_tree = GridSearchCV(DecisionTreeClassifier(), tree_params)

grid_tree.fit(x_train, y_train)



# tree best estimator

tree_clf = grid_tree.best_estimator_
log_reg_score = cross_val_score(log_reg, x_train, y_train, cv=5)

print('Logistic Regression Cross Validation Score: ', round(log_reg_score.mean() * 100, 2).astype(str) + '%')





knears_score = cross_val_score(knears_neighbors, x_train, y_train, cv=5)

print('Knears Neighbors Cross Validation Score', round(knears_score.mean() * 100, 2).astype(str) + '%')



svc_score = cross_val_score(svc, x_train, y_train, cv=5)

print('Support Vector Classifier Cross Validation Score', round(svc_score.mean() * 100, 2).astype(str) + '%')



tree_score = cross_val_score(tree_clf, x_train, y_train, cv=5)

print('DecisionTree Classifier Cross Validation Score', round(tree_score.mean() * 100, 2).astype(str) + '%')
from sklearn.model_selection import ShuffleSplit

from sklearn.model_selection import learning_curve



def plot_learning_curve(estimator1, estimator2, estimator3, estimator4, X, y, ylim=None, cv=None,

                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):

    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2, figsize=(20,14), sharey=True)

    if ylim is not None:

        plt.ylim(*ylim)

    # First Estimator

    train_sizes, train_scores, test_scores = learning_curve(

        estimator1, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)

    train_scores_mean = np.mean(train_scores, axis=1)

    train_scores_std = np.std(train_scores, axis=1)

    test_scores_mean = np.mean(test_scores, axis=1)

    test_scores_std = np.std(test_scores, axis=1)

    ax1.fill_between(train_sizes, train_scores_mean - train_scores_std,

                     train_scores_mean + train_scores_std, alpha=0.1,

                     color="#ff9124")

    ax1.fill_between(train_sizes, test_scores_mean - test_scores_std,

                     test_scores_mean + test_scores_std, alpha=0.1, color="#2492ff")

    ax1.plot(train_sizes, train_scores_mean, 'o-', color="#ff9124",

             label="Training score")

    ax1.plot(train_sizes, test_scores_mean, 'o-', color="#2492ff",

             label="Cross-validation score")

    ax1.set_title("Logistic Regression Learning Curve", fontsize=14)

    ax1.set_xlabel('Training size (m)')

    ax1.set_ylabel('Score')

    ax1.grid(True)

    ax1.legend(loc="best")

    

    # Second Estimator 

    train_sizes, train_scores, test_scores = learning_curve(

        estimator2, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)

    train_scores_mean = np.mean(train_scores, axis=1)

    train_scores_std = np.std(train_scores, axis=1)

    test_scores_mean = np.mean(test_scores, axis=1)

    test_scores_std = np.std(test_scores, axis=1)

    ax2.fill_between(train_sizes, train_scores_mean - train_scores_std,

                     train_scores_mean + train_scores_std, alpha=0.1,

                     color="#ff9124")

    ax2.fill_between(train_sizes, test_scores_mean - test_scores_std,

                     test_scores_mean + test_scores_std, alpha=0.1, color="#2492ff")

    ax2.plot(train_sizes, train_scores_mean, 'o-', color="#ff9124",

             label="Training score")

    ax2.plot(train_sizes, test_scores_mean, 'o-', color="#2492ff",

             label="Cross-validation score")

    ax2.set_title("Knears Neighbors Learning Curve", fontsize=14)

    ax2.set_xlabel('Training size (m)')

    ax2.set_ylabel('Score')

    ax2.grid(True)

    ax2.legend(loc="best")

    

    # Third Estimator

    train_sizes, train_scores, test_scores = learning_curve(

        estimator3, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)

    train_scores_mean = np.mean(train_scores, axis=1)

    train_scores_std = np.std(train_scores, axis=1)

    test_scores_mean = np.mean(test_scores, axis=1)

    test_scores_std = np.std(test_scores, axis=1)

    ax3.fill_between(train_sizes, train_scores_mean - train_scores_std,

                     train_scores_mean + train_scores_std, alpha=0.1,

                     color="#ff9124")

    ax3.fill_between(train_sizes, test_scores_mean - test_scores_std,

                     test_scores_mean + test_scores_std, alpha=0.1, color="#2492ff")

    ax3.plot(train_sizes, train_scores_mean, 'o-', color="#ff9124",

             label="Training score")

    ax3.plot(train_sizes, test_scores_mean, 'o-', color="#2492ff",

             label="Cross-validation score")

    ax3.set_title("Support Vector Classifier \n Learning Curve", fontsize=14)

    ax3.set_xlabel('Training size (m)')

    ax3.set_ylabel('Score')

    ax3.grid(True)

    ax3.legend(loc="best")

    

    # Fourth Estimator

    train_sizes, train_scores, test_scores = learning_curve(

        estimator4, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)

    train_scores_mean = np.mean(train_scores, axis=1)

    train_scores_std = np.std(train_scores, axis=1)

    test_scores_mean = np.mean(test_scores, axis=1)

    test_scores_std = np.std(test_scores, axis=1)

    ax4.fill_between(train_sizes, train_scores_mean - train_scores_std,

                     train_scores_mean + train_scores_std, alpha=0.1,

                     color="#ff9124")

    ax4.fill_between(train_sizes, test_scores_mean - test_scores_std,

                     test_scores_mean + test_scores_std, alpha=0.1, color="#2492ff")

    ax4.plot(train_sizes, train_scores_mean, 'o-', color="#ff9124",

             label="Training score")

    ax4.plot(train_sizes, test_scores_mean, 'o-', color="#2492ff",

             label="Cross-validation score")

    ax4.set_title("Decision Tree Classifier \n Learning Curve", fontsize=14)

    ax4.set_xlabel('Training size (m)')

    ax4.set_ylabel('Score')

    ax4.grid(True)

    ax4.legend(loc="best")

    return plt
cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=42)

plot_learning_curve(log_reg, knears_neighbors, svc, tree_clf, x_train, y_train, (0.87, 1.01), cv=cv, n_jobs=4)
from sklearn.metrics import roc_curve

from sklearn.model_selection import cross_val_predict

# Create a DataFrame with all the scores and the classifiers names.



log_reg_pred = cross_val_predict(log_reg, x_train, y_train, cv=5,

                             method="decision_function")



knears_pred = cross_val_predict(knears_neighbors, x_train, y_train, cv=5)



svc_pred = cross_val_predict(svc, x_train, y_train, cv=5,

                             method="decision_function")



tree_pred = cross_val_predict(tree_clf, x_train, y_train, cv=5)
from sklearn.metrics import roc_auc_score



print('Logistic Regression: ', roc_auc_score(y_train, log_reg_pred))

print('KNears Neighbors: ', roc_auc_score(y_train, knears_pred))

print('Support Vector Classifier: ', roc_auc_score(y_train, svc_pred))

print('Decision Tree Classifier: ', roc_auc_score(y_train, tree_pred))