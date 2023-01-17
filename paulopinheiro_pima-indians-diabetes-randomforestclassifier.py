import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from plotly import tools

import plotly.plotly as py

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.figure_factory as ff

from IPython.display import HTML, Image



df = pd.read_csv('../input/diabetes.csv')
df.head(10)
df.describe()
f, ax = plt.subplots(1, 2, figsize = (15, 7))

f.suptitle("Diabetes?", fontsize = 18.)

_ = df.Outcome.value_counts().plot.bar(ax = ax[0], rot = 0, color = (sns.color_palette()[0], sns.color_palette()[2])).set(xticklabels = ["No", "Yes"])

_ = df.Outcome.value_counts().plot.pie(labels = ("No", "Yes"), autopct = "%.2f%%", label = "", fontsize = 13., ax = ax[1],\

colors = (sns.color_palette()[0], sns.color_palette()[2]), wedgeprops = {"linewidth": 1.5, "edgecolor": "#F7F7F7"}), ax[1].texts[1].set_color("#F7F7F7"), ax[1].texts[3].set_color("#F7F7F7")
fig, ax = plt.subplots(4,2, figsize=(25,25))

sns.distplot(df.Age, bins = 20, ax=ax[0,0]) 

sns.distplot(df.Pregnancies, bins = 20, ax=ax[0,1]) 

sns.distplot(df.Glucose, bins = 20, ax=ax[1,0]) 

sns.distplot(df.BloodPressure, bins = 20, ax=ax[1,1]) 

sns.distplot(df.SkinThickness, bins = 20, ax=ax[2,0])

sns.distplot(df.Insulin, bins = 20, ax=ax[2,1])

sns.distplot(df.DiabetesPedigreeFunction, bins = 20, ax=ax[3,0]) 

sns.distplot(df.BMI, bins = 20, ax=ax[3,1]) 
sns.regplot(x='BMI', y= 'Insulin', data=df)
sns.pairplot(data=df,hue='Outcome')
fig,ax = plt.subplots(nrows=4, ncols=2, figsize=(18,18))

plt.suptitle('Violin Plots',fontsize=24)

sns.violinplot(x="Pregnancies", data=df,ax=ax[0,0],palette='Set3')

sns.violinplot(x="Glucose", data=df,ax=ax[0,1],palette='Set3')

sns.violinplot (x ='BloodPressure', data=df, ax=ax[1,0], palette='Set3')

sns.violinplot(x='SkinThickness', data=df, ax=ax[1,1],palette='Set3')

sns.violinplot(x='Insulin', data=df, ax=ax[2,0], palette='Set3')

sns.violinplot(x='BMI', data=df, ax=ax[2,1],palette='Set3')

sns.violinplot(x='DiabetesPedigreeFunction', data=df, ax=ax[3,0],palette='Set3')

sns.violinplot(x='Age', data=df, ax=ax[3,1],palette='Set3')

plt.show()
corr=df.corr()



sns.set(font_scale=1.15)

plt.figure(figsize=(14, 10))



sns.heatmap(corr, vmax=.8, linewidths=0.01,

            square=True,annot=True,cmap='YlGnBu',linecolor="black")

plt.title('Correlation between features');
from sklearn.metrics import accuracy_score

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import GradientBoostingClassifier



X = df.iloc[:, :-1]

y = df.iloc[:, -1]





X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
#Model

DT = DecisionTreeClassifier(max_depth=3)



#fiting the model

DT.fit(X_train, y_train)



#prediction

y_pred = DT.predict(X_test)



#Accuracy

print("Accuracy", DT.score(X_test, y_test)*100)



#Plot the confusion matrix

sns.set(font_scale=1.5)

cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm, annot=True, fmt='g')

plt.ylabel('True label')

plt.xlabel('Predicted label')

plt.tight_layout()

plt.show()

print('accuracy   ', (cm[0, 0] + cm[1, 1])/ (cm[0, 1] + cm[0, 0] + cm[1, 1] + cm[1, 0])*100)

print('specificity', cm[0, 0] / (cm[0, 1] + cm[0, 0])*100)

print('sensitivity', cm[1, 1] / (cm[1, 1] + cm[1, 0])*100)
from sklearn.tree import export_graphviz

import pydot



feature_list = X.columns.values

# Save the tree as a png image

export_graphviz(DT, out_file = 'diabetes.dot', feature_names = feature_list, rounded = True, precision = 1, filled = True, class_names=['negative','positive'])

(graph, ) = pydot.graph_from_dot_file('diabetes.dot')

graph.write_png('diabetes.png');

Image('diabetes.png')
feature_import = pd.DataFrame(data=DT.feature_importances_, index=feature_list, columns=['values'])

feature_import.sort_values(['values'], ascending=False, inplace=True)

feature_import.transpose()
from sklearn.ensemble import RandomForestClassifier



#Model

RFC = RandomForestClassifier(n_estimators=500, bootstrap=True)



#fiting the model

RFC.fit(X_train, y_train)



#prediction

y_pred = RFC.predict(X_test)



#Accuracy

print("Accuracy", RFC.score(X_test, y_test)*100)



#Plot the confusion matrix

sns.set(font_scale=1.5)

cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm, annot=True, fmt='g')

plt.ylabel('True label')

plt.xlabel('Predicted label')

plt.tight_layout()

plt.show()

print('accuracy   ', (cm[0, 0] + cm[1, 1])/ (cm[0, 1] + cm[0, 0] + cm[1, 1] + cm[1, 0])*100)

print('specificity', cm[0, 0] / (cm[0, 1] + cm[0, 0])*100)

print('sensitivity', cm[1, 1] / (cm[1, 1] + cm[1, 0])*100)
# Save the tree as a png image

export_graphviz(RFC.estimators_[0], out_file = 'diabetes_RFC.dot', feature_names = feature_list, rounded = True, precision = 1, filled = True, class_names=['negative','positive'])

(graph, ) = pydot.graph_from_dot_file('diabetes_RFC.dot')

graph.write_png('diabetes_RFC.png');

Image('diabetes_RFC.png')
feature_import = pd.DataFrame(data=RFC.feature_importances_, index=feature_list, columns=['values'])

feature_import.sort_values(['values'], ascending=False, inplace=True)

feature_import.transpose()
from sklearn.ensemble import GradientBoostingClassifier



gbrt = GradientBoostingClassifier(n_estimators=200)

gbrt.fit(X_train, y_train)



y_pred = gbrt.predict(X_test)

#Accuracy

print("Accuracy", gbrt.score(X_test, y_test)*100)



#Plot the confusion matrix

sns.set(font_scale=1.5)

cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm, annot=True, fmt='g')

plt.ylabel('True label')

plt.xlabel('Predicted label')

plt.tight_layout()

plt.show()

print('accuracy   ', (cm[0, 0] + cm[1, 1])/ (cm[0, 1] + cm[0, 0] + cm[1, 1] + cm[1, 0])*100)

print('specificity', cm[0, 0] / (cm[0, 1] + cm[0, 0])*100)

print('sensitivity', cm[1, 1] / (cm[1, 1] + cm[1, 0])*100)
# Save the tree as a png image

export_graphviz(gbrt.estimators_[1, 0], out_file = 'diabetes_GB.dot', feature_names = feature_list, rounded = True, precision = 1, filled = True, class_names=['negative','positive'])

(graph, ) = pydot.graph_from_dot_file('diabetes_GB.dot')

graph.write_png('diabetes_GB.png');

Image('diabetes_GB.png')
feature_import = pd.DataFrame(data=gbrt.feature_importances_, index=feature_list, columns=['values'])

feature_import.sort_values(['values'], ascending=False, inplace=True)

feature_import.transpose()
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train_std = sc.fit_transform(X_train)

X_test_std = sc.transform(X_test)
from sklearn.svm import SVC

svc = SVC(kernel='linear')

svc.fit(X_train_std, y_train)



y_pred = svc.predict(X_test_std)

#Accuracy

print("Accuracy", svc.score(X_test_std, y_test)*100)



#Plot the confusion matrix

sns.set(font_scale=1.5)

cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm, annot=True, fmt='g')

plt.ylabel('True label')

plt.xlabel('Predicted label')

plt.tight_layout()

plt.show()

print('accuracy   ', (cm[0, 0] + cm[1, 1])/ (cm[0, 1] + cm[0, 0] + cm[1, 1] + cm[1, 0])*100)

print('specificity', cm[0, 0] / (cm[0, 1] + cm[0, 0])*100)

print('sensitivity', cm[1, 1] / (cm[1, 1] + cm[1, 0])*100)
from sklearn.svm import SVC

svc = SVC(kernel='poly', degree=3)

svc.fit(X_train_std, y_train)



y_pred = svc.predict(X_test_std)

#Accuracy

print("Accuracy", svc.score(X_test_std, y_test)*100)



#Plot the confusion matrix

sns.set(font_scale=1.5)

cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm, annot=True, fmt='g')

plt.ylabel('True label')

plt.xlabel('Predicted label')

plt.tight_layout()

plt.show()

print('accuracy   ', (cm[0, 0] + cm[1, 1])/ (cm[0, 1] + cm[0, 0] + cm[1, 1] + cm[1, 0])*100)

print('specificity', cm[0, 0] / (cm[0, 1] + cm[0, 0])*100)

print('sensitivity', cm[1, 1] / (cm[1, 1] + cm[1, 0])*100)
from sklearn.svm import SVC

svc = SVC(kernel='rbf')

svc.fit(X_train_std, y_train)



y_pred = svc.predict(X_test_std)

#Accuracy

print("Accuracy", svc.score(X_test_std, y_test)*100)



#Plot the confusion matrix

sns.set(font_scale=1.5)

cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm, annot=True, fmt='g')

plt.ylabel('True label')

plt.xlabel('Predicted label')

plt.tight_layout()

plt.show()

print('accuracy   ', (cm[0, 0] + cm[1, 1])/ (cm[0, 1] + cm[0, 0] + cm[1, 1] + cm[1, 0])*100)

print('specificity', cm[0, 0] / (cm[0, 1] + cm[0, 0])*100)

print('sensitivity', cm[1, 1] / (cm[1, 1] + cm[1, 0])*100)
from sklearn.svm import SVC

svc = SVC(kernel='sigmoid')

svc.fit(X_train_std, y_train)



y_pred = svc.predict(X_test_std)

#Accuracy

print("Accuracy", svc.score(X_test_std, y_test)*100)



#Plot the confusion matrix

sns.set(font_scale=1.5)

cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm, annot=True, fmt='g')

plt.ylabel('True label')

plt.xlabel('Predicted label')

plt.tight_layout()

plt.show()

print('accuracy   ', (cm[0, 0] + cm[1, 1])/ (cm[0, 1] + cm[0, 0] + cm[1, 1] + cm[1, 0])*100)

print('specificity', cm[0, 0] / (cm[0, 1] + cm[0, 0])*100)

print('sensitivity', cm[1, 1] / (cm[1, 1] + cm[1, 0])*100)
from keras import models

from keras import layers

from keras import optimizers



def CreateModel(dropout = 0.0): 

    model = models.Sequential()

    model.add(layers.Dense(128, activation='relu', input_shape=(X_train.values.shape[1],)))

    model.add(layers.Dropout(dropout))

    model.add(layers.Dense(64, activation='relu'))

    model.add(layers.Dropout(dropout))

    model.add(layers.Dense(32, activation='relu'))

    model.add(layers.Dropout(dropout))

    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model
model = CreateModel()

history_nostd = model.fit(X_train.values, y_train.values, epochs=30, batch_size=16, validation_data=(X_test.values,y_test.values))
plt.plot(history_nostd.history['acc'],'bo', label='Trainning acc')

plt.plot(history_nostd.history['val_acc'],'b', label='Validation acc')

plt.legend()
y_pred = model.predict(X_test.values)

plt.scatter(y_pred, y_test)

plt.ylabel('True Outcome')

plt.xlabel('DNN Output')

plt.tight_layout()
y_pred = (y_pred > 0.5)



#Accuracy

results=model.evaluate(X_test, y_test)

print(results)



#Plot the confusion matrix

sns.set(font_scale=1.5)

cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm, annot=True, fmt='g')

plt.ylabel('True label')

plt.xlabel('Predicted label')

plt.tight_layout()

plt.show()

print('accuracy   ', (cm[0, 0] + cm[1, 1])/ (cm[0, 1] + cm[0, 0] + cm[1, 1] + cm[1, 0])*100)

print('specificity', cm[0, 0] / (cm[0, 1] + cm[0, 0])*100)

print('sensitivity', cm[1, 1] / (cm[1, 1] + cm[1, 0])*100)
model = CreateModel()

history = model.fit(X_train_std, y_train.values, epochs=30, batch_size=16, validation_data=(X_test_std,y_test.values))
plt.plot(history.history['acc'],'bo', label='Trainning acc')

plt.plot(history.history['val_acc'],'b', label='Validation acc')

plt.legend()
y_pred = model.predict(X_test_std)

plt.scatter(y_pred, y_test)

plt.ylabel('True Outcome')

plt.xlabel('DNN Output')

plt.tight_layout()
y_pred = (y_pred > 0.5)



#Accuracy

results=model.evaluate(X_test_std, y_test)

print(results)



#Plot the confusion matrix

sns.set(font_scale=1.5)

cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm, annot=True, fmt='g')

plt.ylabel('True label')

plt.xlabel('Predicted label')

plt.tight_layout()

plt.show()

print('accuracy   ', (cm[0, 0] + cm[1, 1])/ (cm[0, 1] + cm[0, 0] + cm[1, 1] + cm[1, 0])*100)

print('specificity', cm[0, 0] / (cm[0, 1] + cm[0, 0])*100)

print('sensitivity', cm[1, 1] / (cm[1, 1] + cm[1, 0])*100)
model = CreateModel(dropout=0.5)

history = model.fit(X_train_std, y_train.values, epochs=50, batch_size=64, validation_data=(X_test_std,y_test.values))
plt.plot(history.history['acc'],'bo', label='Trainning acc')

plt.plot(history.history['val_acc'],'b', label='Validation acc')

plt.legend()
y_pred = model.predict(X_test_std)

plt.scatter(y_pred, y_test)

plt.ylabel('True Outcome')

plt.xlabel('DNN Output')

plt.tight_layout()
y_pred = (y_pred > 0.5)



#Accuracy

results=model.evaluate(X_test_std, y_test)

print(results)



#Plot the confusion matrix

sns.set(font_scale=1.5)

cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm, annot=True, fmt='g')

plt.ylabel('True label')

plt.xlabel('Predicted label')

plt.tight_layout()

plt.show()

print('accuracy   ', (cm[0, 0] + cm[1, 1])/ (cm[0, 1] + cm[0, 0] + cm[1, 1] + cm[1, 0])*100)

print('specificity', cm[0, 0] / (cm[0, 1] + cm[0, 0])*100)

print('sensitivity', cm[1, 1] / (cm[1, 1] + cm[1, 0])*100)
from sklearn.metrics import roc_curve

from sklearn.metrics import auc

y_pred_keras = model.predict(X_test_std).ravel()

fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_test, y_pred_keras)

auc_keras = auc(fpr_keras, tpr_keras)
plt.figure(1)

plt.plot([0, 1], [0, 1], 'k--')

plt.plot(fpr_keras, tpr_keras, label='DNN (area = {:.3f})'.format(auc_keras))

plt.xlabel('False positive rate')

plt.ylabel('True positive rate')

plt.title('ROC curve')

plt.legend(loc='best')

k=0

for i,j in zip(fpr_keras,tpr_keras):

    value = "{:.{}f}".format( thresholds_keras[k], 2 ) 

    if (k%4==0) :

        plt.annotate(value,xy=(i,j), fontsize=10)

    k=k+1

plt.show()

optimal_idx = np.argmin(np.sqrt(np.square(1-tpr_keras)+np.square(fpr_keras)))

optimal_threshold = thresholds_keras[optimal_idx]

print(optimal_threshold)

print(optimal_idx)

plt.plot(np.sqrt(np.square(1-tpr_keras)+np.square(fpr_keras)))
y_pred = model.predict(X_test_std)

plt.scatter(y_pred, y_test)

plt.ylabel('True Outcome')

plt.xlabel('DNN Output')

plt.tight_layout()
y_pred = (y_pred >= optimal_threshold)



#Accuracy

from sklearn.metrics import accuracy_score



results =  accuracy_score(y_test, y_pred)

print(results)



#Plot the confusion matrix

sns.set(font_scale=1.5)

cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm, annot=True, fmt='g')

plt.ylabel('True label')

plt.xlabel('Predicted label')

plt.tight_layout()

plt.show()

print('accuracy   ', (cm[0, 0] + cm[1, 1])/ (cm[0, 1] + cm[0, 0] + cm[1, 1] + cm[1, 0])*100)

print('specificity', cm[0, 0] / (cm[0, 1] + cm[0, 0])*100)

print('sensitivity', cm[1, 1] / (cm[1, 1] + cm[1, 0])*100)
plt.figure(1)

plt.plot([0, 1], [0, 1], 'k--')

plt.plot(1-fpr_keras, tpr_keras, '-bo')

plt.xlabel('Specificity (true negative rate)')

plt.ylabel('Sensitiviy (true positive rate)')

plt.legend(loc='best')

k=0

for i,j in zip(1-fpr_keras,tpr_keras):

    value = "{:.{}f}".format( thresholds_keras[k], 2 ) 

    if (k%3==0) :

        plt.annotate(value,xy=(i,j), fontsize=10)

    k=k+1

plt.show()