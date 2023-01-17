#importing

import pandas as pd

import numpy as np



import seaborn as sns #visualisation

import matplotlib.pyplot as plt #visualisation

%matplotlib inline 

sns.set(color_codes=True)

import os

import missingno as msno

import statistics as st



import plotly.graph_objects as go

from plotly.subplots import make_subplots

fig = go.Figure(data=go.Bar(y=[2, 3, 1]))





from plotly import tools 

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.figure_factory as ff
data=pd.read_csv('../input/diabetes-dataset/health care diabetes.csv')
data_raw=data.copy()
# To display the top 5 rows

data.head(5)
# Total number of rows and columns

data.shape
# Checking the data type

data.dtypes
data.count() 
data.isnull().any()
data.describe()
#summarize information ion called dataset

data.info()
Positive = data[data['Outcome']==1]

Positive.head(5)
# Matrix Visualization of Handled Values

msno.matrix(data)
# Make figure with subplots

fig = make_subplots(rows=1, cols=2, specs=[[{"type": "bar"},

                                            {"type": "surface"}]])



# Add bar traces to subplot (1, 1)

fig.add_trace(go.Bar(y=[2, 1, 3]), row=1, col=1)

fig.add_trace(go.Bar(y=[3, 2, 1]), row=1, col=1)

fig.add_trace(go.Bar(y=[2.5, 2.5, 3.5]), row=1, col=1)



# Add surface trace to subplot (1, 2)

# Read data from a csv

z_data = pd.read_csv("../input/diabetes-dataset/health care diabetes.csv")

fig.add_surface(z=z_data)



# Hide legend

fig.update_layout(

    showlegend=False,

    title_text="Data_Set Visualization",

    height=500,

    width=900,

)



fig.show()
fig, ax = plt.subplots(4,2, figsize=(16,16))

sns.distplot(data.Pregnancies, bins = 20, ax=ax[0,1]) 



sns.distplot(data.Glucose, bins = 20, ax=ax[1,0]) 



sns.distplot(data.BloodPressure, bins = 20, ax=ax[1,1])



sns.distplot(data.SkinThickness, bins = 20, ax=ax[2,0])



sns.distplot(data.Insulin, bins = 20, ax=ax[2,1])



sns.distplot(data.BMI, bins = 20, ax=ax[3,1])



sns.distplot(data.DiabetesPedigreeFunction, bins = 20, ax=ax[3,0]) 



sns.distplot(data.Age, bins = 20, ax=ax[0,0]) 
# Predictor Variable Analysis 
data['Pregnancies'].value_counts().head(7)
plt.hist(data['Pregnancies'])
data['Glucose'].value_counts().head(7)
plt.hist(data['Glucose'])
data['BloodPressure'].value_counts().head(7)
plt.hist(data['BloodPressure'])
data['SkinThickness'].value_counts().head(7)
plt.hist(data['SkinThickness'])
data['Insulin'].value_counts().head(7)
plt.hist(data['Insulin'])
data['BMI'].value_counts().head(7)
plt.hist(data['BMI'])
data.describe().transpose()
sns.regplot(x='Age', y= 'Pregnancies', data=data)
plt.hist(Positive['BMI'],histtype='stepfilled',bins=20)
Positive['BMI'].value_counts().head(7)
plt.hist(Positive['Glucose'],histtype='stepfilled',bins=20)
Positive['Glucose'].value_counts().head(7)
plt.hist(Positive['BloodPressure'],histtype='stepfilled',bins=20)
Positive['BloodPressure'].value_counts().head(7)
plt.hist(Positive['SkinThickness'],histtype='stepfilled',bins=20)
Positive['SkinThickness'].value_counts().head(7)
plt.hist(Positive['Insulin'],histtype='stepfilled',bins=20)
Positive['Insulin'].value_counts().head(7)
# Initial Data Analysis

f, ax = plt.subplots(1, 2, figsize = (15, 7))

f.suptitle("Diabetic or Non-Diabetic", fontsize = 18.)

_ = data.Outcome.value_counts().plot.bar(ax = ax[0], rot = 0, color = (sns.color_palette()[0], sns.color_palette()[2])).set(xticklabels = ["Non Diabetic", "Diabetic"])

_ = data.Outcome.value_counts().plot.pie(labels = ("Non Diabetic", "Diabetic"), autopct = "%.2f%%", label = "", fontsize = 13., ax = ax[1],\

colors = (sns.color_palette()[0], sns.color_palette()[2]), wedgeprops = {"linewidth": 1.5, "edgecolor": "#F7F7F7"}), ax[1].texts[1].set_color("#F7F7F7"), ax[1].texts[3].set_color("#F7F7F7")
Pregnancies = Positive['Pregnancies']

BloodPressure = Positive['BloodPressure']

Glucose = Positive['Glucose']

SkinThickness = Positive['SkinThickness']

Insulin = Positive['Insulin']

BMI = Positive['BMI']
plt.scatter(BloodPressure, Glucose, color=['b'])

plt.xlabel('BloodPressure')

plt.ylabel('Glucose')

plt.title('BloodPressure & Glucose')

plt.show()
Glucose_BP=sns.scatterplot(x= "Glucose" ,y= "BloodPressure",

              hue="Outcome",

              data=data);
B =sns.scatterplot(x= "BMI" ,y= "Insulin",

              hue="Outcome",

              data=data);
S =sns.scatterplot(x= "SkinThickness" ,y= "Insulin",

              hue="Outcome",

              data=data);
S =sns.scatterplot(x= "BMI" ,y= "Pregnancies",

              hue="Outcome",

              data=data);
sns.pairplot(data=data,hue='Outcome')
### correlation matrix

corr=data.corr()



sns.set(font_scale=1.15)

plt.figure(figsize=(14, 10))



sns.heatmap(corr, vmax=.8, linewidths=0.01,

            square=True,annot=True,cmap='YlGnBu',linecolor="black")

plt.title('Correlation between Predictors');
from sklearn.metrics import accuracy_score

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import GradientBoostingClassifier



X = data.iloc[:, :-1]

y = data.iloc[:, -1]





X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
#Model

LR = LogisticRegression()



#fiting the model

LR.fit(X_train, y_train)



#prediction

y_pred = LR.predict(X_test)



#Accuracy

print("Accuracy ", LR.score(X_test, y_test)*100)



#Plot the confusion matrix

sns.set(font_scale=1.5)

cm = confusion_matrix(y_pred, y_test)

sns.heatmap(cm, annot=True, fmt='g')

plt.show()
#Model

DT = DecisionTreeClassifier()



#fiting the model

DT.fit(X_train, y_train)



#prediction

y_pred = DT.predict(X_test)



#Accuracy

print("Accuracy ", DT.score(X_test, y_test)*100)



#Plot the confusion matrix

sns.set(font_scale=1.5)

cm = confusion_matrix(y_pred, y_test)

sns.heatmap(cm, annot=True, fmt='g')

plt.show()
#Applying Random Forest

from sklearn.ensemble import RandomForestClassifier

RF = RandomForestClassifier(n_estimators=11)

RF.fit(X_train,y_train)





#prediction

y_pred = RF.predict(X_test)



#Accuracy

print("Accuracy ", RF.score(X_test, y_test)*100)



#Plot the confusion matrix

sns.set(font_scale=1.5)

cm = confusion_matrix(y_pred, y_test)

sns.heatmap(cm, annot=True, fmt='g')

plt.show()
#Support Vector Classifier



from sklearn.svm import SVC 

VC = SVC(kernel='rbf',

           gamma='auto')

VC.fit(X_train,y_train)



VC.score(X_test,y_test)



#prediction

y_pred = VC.predict(X_test)



#Accuracy

print("Accuracy ", VC.score(X_test, y_test)*100)



#Plot the confusion matrix

sns.set(font_scale=1.5)

cm = confusion_matrix(y_pred, y_test)

sns.heatmap(cm, annot=True, fmt='g')

plt.show()
#Applying K-NN

from sklearn.neighbors import KNeighborsClassifier

KNN_Model = KNeighborsClassifier(n_neighbors=7,

                             metric='minkowski',

                             p = 2)

KNN_Model.fit(X_train,y_train)



#prediction

y_pred = KNN_Model.predict(X_test)



#Accuracy

print("Accuracy ", KNN_Model.score(X_test, y_test)*100)



#Plot the confusion matrix

sns.set(font_scale=1.5)

cm = confusion_matrix(y_pred, y_test)

sns.heatmap(cm, annot=True, fmt='g')

plt.show()
#Preparing ROC Curve (Receiver Operating Characteristics Curve)

from sklearn.metrics import roc_curve

from sklearn.metrics import roc_auc_score



# predict probabilities

probs = LR.predict_proba(X)

# keep probabilities for the positive outcome only

probs = probs[:, 1]

# calculate AUC

auc = roc_auc_score(y, probs)

print('AUC: %.3f' % auc)

# calculate roc curve

fpr, tpr, thresholds = roc_curve(y, probs)

# plot no skill

plt.plot([0, 1], [0, 1], linestyle='--')

# plot the roc curve for the model

plt.plot(fpr, tpr, marker='.')
#Preparing ROC Curve (Receiver Operating Characteristics Curve)

from sklearn.metrics import roc_curve

from sklearn.metrics import roc_auc_score



# predict probabilities

probs = KNN_Model.predict_proba(X)

# keep probabilities for the positive outcome only

probs = probs[:, 1]

# calculate AUC

auc = roc_auc_score(y, probs)

print('AUC: %.3f' % auc)

# calculate roc curve

fpr, tpr, thresholds = roc_curve(y, probs)

# plot no skill

plt.plot([0, 1], [0, 1], linestyle='--')

# plot the roc curve for the model

plt.plot(fpr, tpr, marker='.')
#Precision Recall Curve for KNN



from sklearn.metrics import precision_recall_curve

from sklearn.metrics import f1_score

from sklearn.metrics import auc

from sklearn.metrics import average_precision_score

# predict probabilities

probs = KNN_Model.predict_proba(X)

# keep probabilities for the positive outcome only

probs = probs[:, 1]

# predict class values

yhat = KNN_Model.predict(X)

# calculate precision-recall curve

precision, recall, thresholds = precision_recall_curve(y, probs)

# calculate F1 score

f1 = f1_score(y, yhat)

# calculate precision-recall AUC

auc = auc(recall, precision)

# calculate average precision score

ap = average_precision_score(y, probs)

print('f1=%.3f auc=%.3f ap=%.3f' % (f1, auc, ap))

# plot no skill

plt.plot([0, 1], [0.5, 0.5], linestyle='--')

# plot the precision-recall curve for the model

plt.plot(recall, precision, marker='.')
#Preparing ROC Curve (Receiver Operating Characteristics Curve)

from sklearn.metrics import roc_curve

from sklearn.metrics import roc_auc_score



# predict probabilities

probs = DT.predict_proba(X)

# keep probabilities for the positive outcome only

probs = probs[:, 1]

# calculate AUC

auc = roc_auc_score(y, probs)

print('AUC: %.3f' % auc)

# calculate roc curve

fpr, tpr, thresholds = roc_curve(y, probs)

# plot no skill

plt.plot([0, 1], [0, 1], linestyle='--')

# plot the roc curve for the model

plt.plot(fpr, tpr, marker='.')
#Precision Recall Curve for Decision Tree Classifier



from sklearn.metrics import precision_recall_curve

from sklearn.metrics import f1_score

from sklearn.metrics import auc

from sklearn.metrics import average_precision_score

# predict probabilities

probs = DT.predict_proba(X)

# keep probabilities for the positive outcome only

probs = probs[:, 1]

# predict class values

yhat = DT.predict(X)

# calculate precision-recall curve

precision, recall, thresholds = precision_recall_curve(y, probs)

# calculate F1 score

f1 = f1_score(y, yhat)

# calculate precision-recall AUC

auc = auc(recall, precision)

# calculate average precision score

ap = average_precision_score(y, probs)

print('f1=%.3f auc=%.3f ap=%.3f' % (f1, auc, ap))

# plot no skill

plt.plot([0, 1], [0.5, 0.5], linestyle='--')

# plot the precision-recall curve for the model

plt.plot(recall, precision, marker='.')
#Preparing ROC Curve (Receiver Operating Characteristics Curve)

from sklearn.metrics import roc_curve

from sklearn.metrics import roc_auc_score



# predict probabilities

probs = RF.predict_proba(X)

# keep probabilities for the positive outcome only

probs = probs[:, 1]

# calculate AUC

auc = roc_auc_score(y, probs)

print('AUC: %.3f' % auc)

# calculate roc curve

fpr, tpr, thresholds = roc_curve(y, probs)

# plot no skill

plt.plot([0, 1], [0, 1], linestyle='--')

# plot the roc curve for the model

plt.plot(fpr, tpr, marker='.')
#Precision Recall Curve for Random Forest



from sklearn.metrics import precision_recall_curve

from sklearn.metrics import f1_score

from sklearn.metrics import auc

from sklearn.metrics import average_precision_score

# predict probabilities

probs = RF.predict_proba(X)

# keep probabilities for the positive outcome only

probs = probs[:, 1]

# predict class values

yhat = RF.predict(X)

# calculate precision-recall curve

precision, recall, thresholds = precision_recall_curve(y, probs)

# calculate F1 score

f1 = f1_score(y, yhat)

# calculate precision-recall AUC

auc = auc(recall, precision)

# calculate average precision score

ap = average_precision_score(y, probs)

print('f1=%.3f auc=%.3f ap=%.3f' % (f1, auc, ap))

# plot no skill

plt.plot([0, 1], [0.5, 0.5], linestyle='--')

# plot the precision-recall curve for the model

plt.plot(recall, precision, marker='.')