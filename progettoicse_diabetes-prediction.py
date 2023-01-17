import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from pandas.plotting import scatter_matrix

import seaborn as sns

import matplotlib.pyplot as plt

from matplotlib import pylab

%matplotlib inline



from sklearn import model_selection

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score

from sklearn.metrics import average_precision_score

from sklearn.metrics import roc_curve

from sklearn.metrics import roc_auc_score

from sklearn.metrics import precision_recall_curve



from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import ShuffleSplit

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import SGDClassifier

from keras.models import Sequential

from keras.layers import Conv2D, MaxPooling2D

from keras.layers import Activation, Dropout, Flatten, Dense

import keras

from keras.models import Sequential

from keras.layers import Dense

from inspect import signature

from sklearn.cluster import KMeans



import warnings

warnings.filterwarnings('ignore')



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
diab_df = pd.read_csv("../input/pima-indians-diabetes-database/diabetes.csv")

diab_df.head(20)
plot = diab_df.hist(figsize = (20,20))
# Outcome Dictionary

diab_df["Diabetes"] = diab_df.pop("Outcome")
sns.set(style = "darkgrid")

sns.countplot(x = "Diabetes", data = diab_df, palette = "Blues")
#Relation between Age and Outcome



plt.rcParams['figure.figsize'] = (8, 8)

sns.swarmplot(diab_df['Diabetes'], diab_df['Age'], palette = 'Blues', size = 10)

plt.title('Relation of Age and Target', fontsize = 20, fontweight = 30)

plt.show()
pd.crosstab(diab_df['Age'], diab_df['Diabetes']).plot(kind = "bar", figsize = (20,5))
pd.crosstab(diab_df['Pregnancies'], diab_df['Diabetes']).plot(kind = "bar", figsize = (20,5), color = ['#99A6BB','#AA4510' ])
countNoDisease = len(diab_df[diab_df['Diabetes'] == 0])

countHaveDisease = len(diab_df[diab_df['Diabetes'] == 1])

print("Percentage of Patients not having diabetes: {:.2f}%".format((countNoDisease / (len(diab_df['Diabetes'])) * 100)))

print("Percentage of Patients having diabetes: {:.2f}%".format((countHaveDisease / (len(diab_df['Diabetes'])) * 100)))
#get correlations of each features in dataset

corrmat = diab_df.corr()

top_corr_features = corrmat.index

plt.figure(figsize = (20,20))



#plot heat map

sns.heatmap(diab_df[top_corr_features].corr(), annot = True, cmap = "Blues")

#Correlation with output variable

cor_diag = abs(corrmat["Diabetes"])



#Selecting highly correlated features

relevant_features = cor_diag[cor_diag > 0.15]

relevant_features
X = diab_df.drop(['BloodPressure','SkinThickness','Insulin','Diabetes'], 1)

Y = diab_df['Diabetes']
#standardizing the input feature



sc = StandardScaler()

X = sc.fit_transform(X)
#creating training set and test set



X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)
def plt_conf_matrix(pred):

  # Confusion Matrix

  cm = confusion_matrix(Y_test, pred)

  plt.figure(figsize = (5, 4))

  sns.heatmap(cm, xticklabels = ['Positive','Negative'], yticklabels = ['Positive','Negative'], annot = True, fmt = 'd', cmap="Blues")

  plt.title('Confusion Matrix')

  plt.ylabel('Actual Values')

  plt.xlabel('Predicted Values')

  plt.show()

  

def plt_var_dev(scores):

  #Variance & Dev. standard

  print('\nVariance: {}'.format(round(np.var(scores) * 100, 2)))

  print('Dev. standard: {}\n'.format(round(np.std(scores) * 100, 2)))

  data = {'variance': np.var(scores), 'standard dev': np.std(scores)}

  names = list(data.keys())

  values = list(data.values())

  fig,axs = plt.subplots(1, 1, figsize = (5, 3), sharey = True)

  axs.bar(names, values)

  plt.show()



def plt_auc(model):

  #AUC

  probs = model.predict_proba(X_test)

  # keep probabilities for the positive outcome only

  probs = probs[:, 1]

  auc = roc_auc_score(Y_test, probs)

  print('\nAUC: %.2f\n' % auc)

  # calculate roc curve

  fpr, tpr, thresholds = roc_curve(Y_test, probs)

  plt.figure(figsize = (5, 4))

  # plot no skill

  plt.plot([0, 1], [0, 1], linestyle = '--')

  # plot the roc curve for the model

  plt.plot(fpr, tpr, marker = '.')

  plt.xlabel('FP RATE')

  plt.ylabel('TP RATE')

  plt.show()

  

def plt_prec_rec(pred):

  #Precision-Recall Curve

  average_precision = average_precision_score(Y_test, pred)

  precision, recall, _ = precision_recall_curve(Y_test, pred)

  print('\nAP = {0:0.2f}\n'.format(average_precision))

  step_kwargs = ({'step': 'post'}

                 if 'step' in signature(plt.fill_between).parameters

                 else {})

  plt.figure(figsize = (5, 4))

  plt.step(recall, precision, color = 'b', alpha = 0.2, where = 'post')

  plt.fill_between(recall, precision, alpha = 0.2, color = 'b', **step_kwargs)

  plt.xlabel('Recall')

  plt.ylabel('Precision')

  plt.ylim([0.0, 1.05])

  plt.xlim([0.0, 1.0])

  plt.title('Precision-Recall curve')
scores = []

models_name = []
lr = LogisticRegression()

lr.fit(X_train,Y_train)

y_pred_lr = lr.predict(X_test)

score_lr = round(accuracy_score(y_pred_lr,Y_test) * 100, 2)



print("The accuracy score achieved using Logistic Regression is: " + str(score_lr) + " %")

print ('\nClasification report:\n', classification_report(Y_test, y_pred_lr))



plt_conf_matrix(y_pred_lr)

plt_auc(lr)

plt_prec_rec(y_pred_lr)
cv = ShuffleSplit(n_splits = 5, test_size = 0.15, random_state = 0)

scores_lr = cross_val_score(lr, X, diab_df.Diabetes, cv = cv)

score_lr = round(scores_lr.mean() * 100, 2)



print("The accuracy score achieved using LR & Cross-Validation technique is: " + str(score_lr) + " %")



scores.append(score_lr)

models_name.append('LGR')



plt_var_dev(scores_lr)
k_range = range(1,10)

error = []



for k in k_range:

    knn = KNeighborsClassifier(n_neighbors = k)

    knn.fit(X_train,Y_train)

    y_pred_knn = knn.predict(X_test)

    score_knn = round(accuracy_score(y_pred_knn,Y_test) * 100, 2)

    error.append(np.mean(y_pred_knn != Y_test))

    

plt.figure(figsize=(4,4))

plt.plot(range(1, 10), error)

plt.title('Error Rate K Value')  

plt.xlabel('K Value')  

plt.ylabel('Mean Error')

plt.show()
knn = KNeighborsClassifier(n_neighbors = 8)

knn.fit(X_train,Y_train)

y_pred_knn = knn.predict(X_test)

score_knn = round(accuracy_score(y_pred_knn, Y_test) * 100, 2)



print("The accuracy score achieved using KNN is: " + str(score_knn) + " %")

print ('\nClasification report:\n', classification_report(Y_test, y_pred_knn))



plt_conf_matrix(y_pred_knn)

plt_auc(knn)

plt_prec_rec(y_pred_knn)
cv = ShuffleSplit(n_splits = 5, test_size = 0.15, random_state = 0)

scores_knn = cross_val_score(knn, X, diab_df.Diabetes, cv = cv)

score_knn = round(scores_knn.mean() * 100, 2)



print("The accuracy score achieved using KNN & Cross-Validation technique is: " + str(score_knn) + " %")



scores.append(score_knn)

models_name.append('KNN')



plt_var_dev(scores_knn)
nb = GaussianNB()

nb.fit(X_train,Y_train)

y_pred_nb = nb.predict(X_test)

score_nb = round(accuracy_score(y_pred_nb,Y_test) * 100, 2)



print("The accuracy score achieved using Naive Bayes is: " + str(score_nb) + " %")

print ('\nClasification report:\n', classification_report(Y_test, y_pred_nb))



plt_conf_matrix(y_pred_nb)

plt_auc(nb)

plt_prec_rec(y_pred_nb)
cv = ShuffleSplit(n_splits = 5, test_size = 0.15, random_state = 0)

scores_nb = cross_val_score(nb, X, diab_df.Diabetes, cv = cv)

score_nb = round(scores_nb.mean() * 100, 2)



print("The accuracy score achieved using Gaussian Naive Bayes & Cross-Validation technique is: " + str(score_nb) + " %")



scores.append(score_nb)

models_name.append('GNB')



plt_var_dev(scores_nb)
dt = DecisionTreeClassifier()

dt.fit(X_train, Y_train)

y_pred_dt = dt.predict(X_test)

score_dt = round(accuracy_score(y_pred_dt, Y_test) * 100, 2)



print("The accuracy score achieved using Decision Tree is: " + str(score_dt) + " %")

print ('\nClasification report:\n', classification_report(Y_test, y_pred_dt))



scores.append(score_dt)

models_name.append('DT')



plt_conf_matrix(y_pred_dt)

plt_auc(dt)

plt_prec_rec(y_pred_dt)
sv = SVC(kernel = 'sigmoid')

sv.fit(X_train, Y_train)

y_pred_svm = sv.predict(X_test)

score_svm = round(accuracy_score(y_pred_svm, Y_test) * 100, 2)



print("The accuracy score achieved using Linear SVM is: " + str(score_svm) + " %")

print ('\nClasification report:\n', classification_report(Y_test, y_pred_svm))



plt_conf_matrix(y_pred_svm)

plt_prec_rec(y_pred_svm)
cv = ShuffleSplit(n_splits = 5, test_size = 0.15, random_state = 0)

scores_svm = cross_val_score(sv, X, diab_df.Diabetes, cv = cv)

score_svm = round(scores_svm.mean() * 100, 2)



print("The accuracy score achieved using SVM & Cross-Validation technique is: " + str(score_svm) + " %")



scores.append(score_svm)

models_name.append('SVM')



plt_var_dev(scores_svm)
rf = RandomForestClassifier(n_estimators = 100, bootstrap = True)

rf.fit(X_train, Y_train)

y_pred_rf = rf.predict(X_test)

score_rf = round(accuracy_score(y_pred_rf, Y_test) * 100, 2)



print("The accuracy score achieved using Random Forest is: " + str(score_rf) + " %")

print ('\nClasification report:\n', classification_report(Y_test, y_pred_rf))





scores.append(score_rf)

models_name.append('RF')



plt_conf_matrix(y_pred_rf)

plt_auc(rf)

plt_prec_rec(y_pred_rf)
classifier = Sequential()

# First Hidden Layer

classifier.add(Dense(4, activation = 'relu', kernel_initializer = 'random_normal', input_dim = 5))

# Second  Hidden Layer

classifier.add(Dense(4, activation = 'relu', kernel_initializer = 'random_normal'))

# Output Layer

classifier.add(Dense(1, activation = 'sigmoid', kernel_initializer = 'random_normal'))
# Compiling the neural network

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
# Fitting the data to the training dataset

history = classifier.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs = 100, batch_size = 16, verbose = 2)
# Model accuracy

plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.title('Model Accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'])

plt.show()
# Model Losss

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('Model Loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'])

plt.show()
y_pred = classifier.predict(X_test)

y_pred = (y_pred > 0.5)
cm = confusion_matrix(Y_test, y_pred)

true_pos = np.diag(cm)

false_pos = np.sum(cm, axis = 0) - true_pos

false_neg = np.sum(cm, axis = 1) - true_pos

score_nn = round(np.sum(true_pos)/(np.sum(true_pos) + np.sum(false_pos)) * 100, 2)



print("\nThe accuracy score achieved using Neural Network is: " + str(score_nn) + " %")

print ('\nClasification report:\n', classification_report(Y_test, y_pred))





plt_conf_matrix(y_pred)

plt_prec_rec(y_pred)



scores.append(score_nn)

models_name.append('NN')
feature = ['Pregnancies',	'Glucose', 'BloodPressure',	'SkinThickness',	'Insulin',	'BMI',	'DiabetesPedigreeFunction',	'Age','Diabetes']

feature_dummied = ['Pregnancies',	'Glucose',	'BMI',	'DiabetesPedigreeFunction',	'Age']

data_dummies = pd.get_dummies(diab_df, columns = feature_dummied)

data_dummies.head()

X =  data_dummies.drop(["Diabetes"], axis=1)

Y_df = pd.get_dummies(data_dummies['Diabetes'], columns=['Diabetes'])

Y_df = Y_df.drop([0], axis = 1)
from scipy.spatial.distance import cdist



distortions = []

K = range(1, 10)

for k in K:

    km = KMeans(n_clusters = k).fit(X)

    km.fit(X)

    distortions.append(sum(np.min(cdist(X, km.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0])



# Plot the elbow

plt.figure(figsize = (5,4))

plt.plot(K, distortions, 'bx-')

plt.xlabel('k')

plt.ylabel('Distortion')

plt.title('The Elbow Method showing the optimal k')

plt.show()
km = KMeans(n_clusters = 2, init = 'k-means++')

y_kmeans = km.fit_predict(X)



score_km = round(accuracy_score(y_kmeans, Y_df) * 100, 2)



scores.append(score_km)

models_name.append('KM')
print("The accuracy score achieved using K-Means is: " + str(score_km) + " %")

print ('\nClasification report:\n',classification_report(Y_df, y_kmeans))



cm = confusion_matrix(Y_df, y_kmeans)

plt.figure(figsize = (5, 4))

sns.heatmap(cm, xticklabels = ['Positive','Negative'], yticklabels = ['Positive','Negative'], annot = True, fmt = 'd', cmap = 'Blues')

plt.title('Confusion Matrix')

plt.ylabel('Actual Values')

plt.xlabel('Predicted Values')

plt.show()



average_precision = average_precision_score(Y_df, y_kmeans)

precision, recall, _ = precision_recall_curve(Y_df, y_kmeans)



print('\nAP = {0:0.2f}\n'.format(average_precision))



# In matplotlib < 1.5, plt.fill_between does not have a 'step' argument

step_kwargs = ({'step': 'post'}

               if 'step' in signature(plt.fill_between).parameters

               else {})

plt.figure(figsize = (5,4))

plt.step(recall, precision, color='b', alpha=0.2,

         where='post')

plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)



plt.xlabel('Recall')

plt.ylabel('Precision')

plt.ylim([0.0, 1.05])

plt.xlim([0.0, 1.0])

plt.title('Precision-Recall curve')

plt.show()
sns.set_style("whitegrid")

plt.figure(figsize = (10, 6))

plt.yticks(np.arange(0, 100, 10))

plt.ylabel("Accuracy (%)")

plt.xlabel("Algorithms")

ax = sns.barplot(x = models_name, y = scores, palette = "Blues")

total = len(diab_df)



i=0

for p in ax.patches:

    height = p.get_height()

    ax.text(p.get_x()+p.get_width()/2, height + 0.1,

       height,ha="center")

    i += 1