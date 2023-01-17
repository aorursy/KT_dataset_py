import pandas as pd

from pandas.plotting import scatter_matrix

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

from matplotlib import pylab

%matplotlib inline



from sklearn import model_selection

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score

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

#from sklearn.cluster import KMeans



import warnings

warnings.filterwarnings('ignore')
heart_df = pd.read_csv("../input/heart-disease-uci/heart.csv")

heart_df.head()
# Sex Dictionary

sex_dict = { 0: 'Female',1: 'Male' }



# Chest Pain Dictionary

chest_pain_dict = { 0:'Typical Angina', 1:'Atypical Angina', 2:'Non-Anginal Pain', 3:'Asymptomatic'}



# Blood Sugar  Dictionary

fbs_dict = { 0: 'No Blood Sugar', 1: 'Blood Sugar'}



# Rest ECG

restecg_dict = { 0:'Normal restecg', 1:'ST-T wave abnormality restecg', 2:'ventricular hypertrophy restecg'}



# Exercise Induced Angina

exang_dict = { 0:'Exang No', 1:'Exang Yes'}



# Slope of the peak exercise ST segment 

slope_dict = { 0:'Upsloping', 1:'flat', 2:'Downsloping'}



# Number of Major Vessels (0-3) Colored by Flourosopy

ca_dict = { 0:'Major vessel 0', 1:'Major vessel 1', 2:'Major vessel 2', 3:'Major vessel 3', 4:'Major vessel 4'}



# Thalassemia Dictionary

thal_dict = { 0: 'None', 1: 'Normal', 2:'Fixed Defect',3:'Reversable Defect'}



# Target Dictionary

target_dict = { 0: 'Not Present', 1: 'Present'}
df = heart_df[['age']]



# trestbps as Resting Blood Pressure

df['Resting Blood Pressure'] = heart_df['trestbps']



# chol as Serum Cholestoral

df['Serum Cholestoral'] = heart_df['chol']



# thalach as Max. Heart Rate

df['Max. Heart Rate'] = heart_df['thalach']



# old peak ST Depression

df['ST Depression'] = heart_df['oldpeak']



# sex as Sex

df['Sex'] = heart_df['sex'].apply(lambda x:sex_dict[x])



# thal as Thalassemia

df['Thalassemia'] = heart_df['thal'].apply(lambda x:thal_dict[x])



# fbs as Thalassemia

df['Fasting Blood Sugar'] = heart_df['fbs'].apply(lambda x:fbs_dict[x])



# cp as Chest Pain

df['Chest Pain'] = heart_df['cp'].apply(lambda x:chest_pain_dict[x])



# target as Heart Disease

df['Heart Disease'] = heart_df['target'].apply(lambda x:target_dict[x])
df.head()
sns.set(style = "darkgrid")

sns.countplot(x = "Heart Disease", data = df, palette = "bwr")
countNoDisease = len(df[df['Heart Disease'] == 'Not Present'])

countHaveDisease = len(df[df['Heart Disease'] == 'Present'])

print("Percentage of Patients Haven't Heart Disease: {:.2f}%".format((countNoDisease / (len(df['Heart Disease'])) * 100)))

print("Percentage of Patients Have Heart Disease: {:.2f}%".format((countHaveDisease / (len(df['Heart Disease'])) * 100)))
pd.crosstab(df['Sex'], df['Heart Disease']).plot(kind = 'bar', figsize = (13,5), color = ['#1CA53B','#AA1111' ])
pd.crosstab(df['age'], df['Heart Disease']).plot(kind = "bar", figsize = (20,6))
pd.crosstab(df['ST Depression'], df['Heart Disease']).plot(kind = "bar", figsize = (15,6), color = ['#DAF7A6','#FF5733' ])
pd.crosstab(df['Fasting Blood Sugar'], df['Heart Disease']).plot(kind = "bar", figsize = (15,6), color = ['#FFC300','#581845' ])
pd.crosstab(df['Chest Pain'], df['Heart Disease']).plot(kind = "bar", figsize = (15,6), color = ['#11A5AA','#AA1190' ])
pd.crosstab(heart_df.thal, df['Heart Disease']).plot(kind = "bar", figsize = (15,6), color = ['#99A6BB','#AA4510' ])
#get correlations of each features in dataset

corrmat = heart_df.corr()

top_corr_features = corrmat.index

plt.figure(figsize = (20,20))



#plot heat map

sns.heatmap(heart_df[top_corr_features].corr(), annot = True, cmap = "RdYlGn")
#Correlation with output variable

cor_target = abs(corrmat["target"])



#Selecting highly correlated features

relevant_features = cor_target[cor_target > 0.10]

relevant_features
X = heart_df.drop(['fbs','chol','target'], 1)

Y = heart_df['target']
#standardizing the input feature



sc = StandardScaler()

X = sc.fit_transform(X)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)
scores = []

models_name = []
sgd = SGDClassifier(loss="hinge", penalty="l2", max_iter=5)

sgd.fit(X_train, Y_train)

y_pred_sgd = sgd.predict(X_test)

score_sgd = round(accuracy_score(y_pred_sgd,Y_test) * 100, 2)



print("The accuracy score achieved using Stochastic Gradient Discent is: " + str(score_sgd) + " %")



scores.append(score_sgd)

models_name.append('SGD')
lr = LogisticRegression()

lr.fit(X_train,Y_train)

y_pred_lr = lr.predict(X_test)

score_lr = round(accuracy_score(y_pred_lr,Y_test) * 100, 2)



print("The accuracy score achieved using Logistic Regression is: " + str(score_lr) + " %")



scores.append(score_lr)

models_name.append('LGR')
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
nb = GaussianNB()

nb.fit(X_train,Y_train)

y_pred_nb = nb.predict(X_test)

score_nb = round(accuracy_score(y_pred_nb,Y_test) * 100, 2)



print("The accuracy score achieved using Naive Bayes is: " + str(score_nb) + " %")



scores.append(score_nb)

models_name.append('GNB')
dt = DecisionTreeClassifier()

dt.fit(X_train, Y_train)

y_pred_dt = dt.predict(X_test)

score_dt = round(accuracy_score(y_pred_dt, Y_test) * 100, 2)



print("The accuracy score achieved using Decision Tree is: " + str(score_dt) + " %")



scores.append(score_dt)

models_name.append('DT')
sv = SVC(kernel = 'sigmoid')

sv.fit(X_train, Y_train)

y_pred_svm = sv.predict(X_test)

score_svm = round(accuracy_score(y_pred_svm, Y_test) * 100, 2)



print("The accuracy score achieved using Linear SVM is: " + str(score_svm) + " %")



scores.append(score_svm)

models_name.append('SVM')
rf = RandomForestClassifier(n_estimators = 100, bootstrap = True)

rf.fit(X_train, Y_train)

y_pred_rf = rf.predict(X_test)

score_rf = round(accuracy_score(y_pred_rf, Y_test) * 100, 2)



print("The accuracy score achieved using Random Forest is: " + str(score_rf) + " %")



scores.append(score_rf)

models_name.append('RF')
classifier = Sequential()

# First Hidden Layer

classifier.add(Dense(4, activation = 'relu', kernel_initializer = 'random_normal', input_dim = 11))

# Second  Hidden Layer

classifier.add(Dense(4, activation = 'relu', kernel_initializer = 'random_normal'))

# Output Layer

classifier.add(Dense(1, activation = 'sigmoid', kernel_initializer = 'random_normal'))
# Compiling the neural network

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
# Fitting the data to the training dataset

history = classifier.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs = 100, batch_size = 16, verbose = 2)
# Model accuracy

plt.plot(history.history['acc'])

plt.plot(history.history['val_acc'])

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

sns.heatmap(cm, xticklabels = ['Positive','Negative'], yticklabels = ['Positive','Negative'], annot = True, fmt = 'd')

plt.title('Confusion Matrix')

plt.ylabel('Actual Values')

plt.xlabel('Predicted Values')

plt.show()
true_pos = np.diag(cm)

false_pos = np.sum(cm, axis = 0) - true_pos

false_neg = np.sum(cm, axis = 1) - true_pos

score_nn = round(np.sum(true_pos)/(np.sum(true_pos) + np.sum(false_pos)) * 100, 2)



print("The accuracy score achieved using Neural Network is: " + str(score_nn) + " %")



scores.append(score_nn)

models_name.append('NN')
print(scores)



colors = ["purple", "green", "orange", "magenta", "red", "yellow", "grey", "blue"]

sns.set_style("whitegrid")

plt.figure(figsize = (8, 5))

plt.yticks(np.arange(0, 100, 10))

plt.ylabel("Accuracy %")

plt.xlabel("Algorithms")

sns.barplot(x = scores, y = models_name, palette = colors)

plt.show()