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

from sklearn.metrics import roc_auc_score

from sklearn.metrics import roc_curve

from sklearn.metrics import average_precision_score

from sklearn.metrics import precision_recall_curve

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

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

#from sklearn.cluster import KMeans



import warnings

warnings.filterwarnings('ignore')



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
breast_df = pd.read_csv("../input/breast-cancer-wisconsin-data/data.csv")

breast_df.head(20)
# Target Dictionary

target_dict = { 'B': 0, 'M': 1}
#breast_df.diagnosis.unique()
df = breast_df[['id']]



df['Radius Mean'] = breast_df[['radius_mean']]



df['Texture Mean'] = breast_df[['texture_mean']]



df['Perimeter Mean'] = breast_df[['perimeter_mean']]



df['Area Mean'] = breast_df[['area_mean']]



df['Smoothness Mean'] = breast_df[['smoothness_mean']]



df['Compactness Mean'] = breast_df[['compactness_mean']]



df['Concavity Mean'] = breast_df[['concavity_mean']]



df['Concave Points Mean'] = breast_df[['concave points_mean']]



df['Symmetry Mean'] = breast_df[['symmetry_mean']]



df['Fractal Dimension Mean'] = breast_df[['fractal_dimension_mean']]



#-------------------------------------------------------------



df['Radius SE'] = breast_df[['radius_se']]



df['Texture SE'] = breast_df[['texture_se']]



df['Perimeter SE'] = breast_df[['perimeter_se']]



df['Area SE'] = breast_df[['area_se']]



df['Smoothness SE'] = breast_df[['smoothness_se']]



df['Compactness SE'] = breast_df[['compactness_se']]



df['Concavity SE'] = breast_df[['concavity_se']]



df['Concave Points SE'] = breast_df[['concave points_se']]



df['Symmetry SE'] = breast_df[['symmetry_se']]



df['Fractal Dimension SE'] = breast_df[['fractal_dimension_se']]



#-------------------------------------------------------------



df['Radius Worst'] = breast_df[['radius_worst']]



df['Texture Worst'] = breast_df[['texture_worst']]



df['Perimeter Worst'] = breast_df[['perimeter_worst']]



df['Area Worst'] = breast_df[['area_worst']]



df['Smoothness Worst'] = breast_df[['smoothness_worst']]



df['Compactness Worst'] = breast_df[['compactness_worst']]



df['Concavity Worst'] = breast_df[['concavity_worst']]



df['Concave Points Worst'] = breast_df[['concave points_worst']]



df['Symmetry Worst'] = breast_df[['symmetry_worst']]



df['Fractal Dimension Worst'] = breast_df[['fractal_dimension_worst']]



df['Diagnosis'] = breast_df['diagnosis'].apply(lambda x:target_dict[x])



#map({'M':1,'B':0})



# target as Heart Disease

#df['Breast Cancer'] = breast_df['diagnosis'].apply(lambda x:target_dict[x])

df.head()
sns.set(style = "darkgrid")

sns.countplot(x = "Diagnosis", data = df, palette = "bwr")
countNoDisease = len(df[df['Diagnosis'] == 0])

countHaveDisease = len(df[df['Diagnosis'] == 1])

print("Percentage of Patients having Benignant Breast Cancer: {:.2f}%".format((countNoDisease / (len(df['Diagnosis'])) * 100)))

print("Percentage of Patients having Malignant Breast Cancer: {:.2f}%".format((countHaveDisease / (len(df['Diagnosis'])) * 100)))
features_mean=list(df.columns[1:11])

# split dataframe into two based on diagnosis

dfM=df[df['Diagnosis'] ==1]

dfB=df[df['Diagnosis'] ==0]


plt.rcParams.update({'font.size': 8})

fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(8,10))

axes = axes.ravel()

for idx,ax in enumerate(axes):

    ax.figure

    binwidth= (max(df[features_mean[idx]]) - min(df[features_mean[idx]]))/50

    

    ax.hist([dfM[features_mean[idx]],dfB[features_mean[idx]]], bins=np.arange(min(df[features_mean[idx]]), max(df[features_mean[idx]]) + binwidth, binwidth) , alpha=0.5,stacked = True,  label=['M','B'],color=['r','g'])

    ax.legend(loc='upper right')

    ax.set_title(features_mean[idx])

plt.tight_layout()

plt.show()
#get correlations of each features in dataset

corrmat = df.corr()

top_corr_features = corrmat.index

plt.figure(figsize = (20,20))



#plot heat map

sns.heatmap(df[top_corr_features].corr(), annot = True, cmap = "RdYlGn")
#Correlation with output variable

cor_diag = abs(corrmat["Diagnosis"])



#Selecting highly correlated features

relevant_features = cor_diag[cor_diag > 0.40]

relevant_features
X = df.drop(['Fractal Dimension Mean','Texture SE','Smoothness SE','Compactness SE','Concavity SE','Symmetry SE','Fractal Dimension SE','Radius Worst','Perimeter Worst','Area Worst','Diagnosis','id'], 1)

Y = df['Diagnosis']
sc = StandardScaler()

X = sc.fit_transform(X)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.30, random_state = 0)
def plt_conf_matrix(pred):

  # Confusion Matrix

  cm = confusion_matrix(Y_test, pred)

  plt.figure(figsize = (5, 4))

  sns.heatmap(cm, xticklabels = ['Positive','Negative'], yticklabels = ['Positive','Negative'], annot = True, fmt = 'd')

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
k_range = range(1,10)

scores_list = []



for k in k_range:

    knn = KNeighborsClassifier(n_neighbors = k)

    knn.fit(X_train,Y_train)

    y_pred_knn = knn.predict(X_test)

    score_knn = round(accuracy_score(y_pred_knn,Y_test) * 100, 2)

    scores_list.append(score_knn)



print("The accuracy score achieved using KNN is: " + str(max(scores_list)) + " %")



scores.append(max(scores_list))

models_name.append('KNN')



plt_conf_matrix(y_pred_knn)

plt_auc(knn)

plt_prec_rec(y_pred_knn)
rf = RandomForestClassifier(n_estimators = 100, bootstrap = True)

rf.fit(X_train, Y_train)

y_pred_rf = rf.predict(X_test)

score_rf = round(accuracy_score(y_pred_rf, Y_test) * 100, 2)



print("The accuracy score achieved using Random Forest is: " + str(score_rf) + " %")



scores.append(score_rf)

models_name.append('RF')



plt_conf_matrix(y_pred_rf)

plt_auc(rf)

plt_prec_rec(y_pred_rf)
lr = LogisticRegression()

lr.fit(X_train,Y_train)

y_pred_lr = lr.predict(X_test)

score_lr = round(accuracy_score(y_pred_lr,Y_test) * 100, 2)



print("The accuracy score achieved using Logistic Regression is: " + str(score_lr) + " %")

print ('\nClasification report:\n', classification_report(Y_test, y_pred_lr))



scores.append(score_lr)

models_name.append('LGR')



plt_conf_matrix(y_pred_lr)

plt_auc(lr)

plt_prec_rec(y_pred_lr)
nb = GaussianNB()

nb.fit(X_train,Y_train)

y_pred_nb = nb.predict(X_test)

score_nb = round(accuracy_score(y_pred_nb,Y_test) * 100, 2)



print("The accuracy score achieved using Naive Bayes is: " + str(score_nb) + " %")



scores.append(score_nb)

models_name.append('GNB')



plt_conf_matrix(y_pred_nb)

plt_auc(nb)

plt_prec_rec(y_pred_nb)
sv = SVC(kernel = 'sigmoid')

sv.fit(X_train, Y_train)

y_pred_svm = sv.predict(X_test)

score_svm = round(accuracy_score(y_pred_svm, Y_test) * 100, 2)



print("The accuracy score achieved using Linear SVM is: " + str(score_svm) + " %")



scores.append(score_svm)

models_name.append('SVM')



plt_conf_matrix(y_pred_svm)

#plt_auc(sv)

plt_prec_rec(y_pred_svm)
dt = DecisionTreeClassifier()

dt.fit(X_train, Y_train)

y_pred_dt = dt.predict(X_test)

score_dt = round(accuracy_score(y_pred_dt, Y_test) * 100, 2)



print("The accuracy score achieved using Decision Tree is: " + str(score_dt) + " %")



scores.append(score_dt)

models_name.append('DT')



plt_conf_matrix(y_pred_dt)

plt_auc(dt)

plt_prec_rec(y_pred_dt)
classifier = Sequential()

# First Hidden Layer

classifier.add(Dense(4, activation = 'relu', kernel_initializer = 'random_normal', input_dim = 20))

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
# Confusion Matrix

cm = confusion_matrix(Y_test, y_pred)

plt.figure(figsize = (5,4))

sns.heatmap(cm, xticklabels = ['Positive','Negative'], yticklabels = ['Positive','Negative'], annot = True, fmt = 'd', cmap = "BuGn")

plt.title('Confusion Matrix')

plt.ylabel('Actual Values')

plt.xlabel('Predicted Values')

plt.show()
true_pos = np.diag(cm)

false_pos = np.sum(cm, axis = 0) - true_pos

false_neg = np.sum(cm, axis = 1) - true_pos

score_nn = round(np.sum(true_pos)/(np.sum(true_pos) + np.sum(false_pos)) * 100, 2)



print("The accuracy score achieved using Neural Network is: " + str(score_nn) + " %")



plt_prec_rec(y_pred)



scores.append(score_nn)

models_name.append('NN')


sns.set_style("whitegrid")

plt.figure(figsize = (8, 5))

plt.yticks(np.arange(0, 100, 10))

plt.ylabel("Accuracy %")

plt.xlabel("Algorithms")

sns.barplot(x = scores, y = models_name, palette = "BuGn_r")

plt.show()
feature = ['Radius Mean','Texture Mean','Perimeter Mean','Area Mean','Smoothness Mean','Compactness Mean','Concavity Mean','Concave Points Mean','Symmetry Mean','Fractal Dimension Mean','Radius SE','Texture SE','Perimeter SE','Smothness SE','Compactness SE','Concavity SE','Concave Points SE','Symmetry SE','Fractal Dimension SE','Radius Worst','Texture Worst','Perimeter Worst','Area Worst','Smoothness Worst','Compactness Worst','Concavity Worst','Concave Points Worst','Symmetry Worst','Fractal Dimension Worst']

feature_dummied = ['Radius Mean','Texture Mean','Perimeter Mean','Area Mean','Compactness Mean','Concavity Mean','Concave Points Mean','Radius SE','Perimeter SE','Concave Points SE','Area SE','Radius Worst','Texture Worst','Perimeter Worst','Area Worst','Smoothness Worst','Compactness Worst','Concavity Worst','Concave Points Worst','Symmetry Worst']

data_dummies = pd.get_dummies(df, columns = feature_dummied)

data_dummies.head()

X =  data_dummies.drop(["Diagnosis"], axis=1)

Y_df = pd.get_dummies(data_dummies['Diagnosis'], columns=['Diagnosis'])

Y_df = Y_df.drop([0], axis = 1)