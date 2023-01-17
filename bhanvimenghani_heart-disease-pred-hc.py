import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
data = pd.read_csv('../input/heart-disease-uci/heart.csv')
data.describe
data.head()
data.columns
# target is the output and the final conclusion if a person has heart disease or not 

data.target.value_counts()
!pip install pandas-profiling
from pandas_profiling import ProfileReport

prof = ProfileReport(data)

prof.to_file(output_file='output.html')
prof
!pip install sweetviz
import sweetviz
report = sweetviz.analyze(data)
#display the report

report.show_html('Heart_EDA.html')
ax = sns.countplot(x="cp",hue="sex", data=data)

plt.title('Heart Disease count According To Chest Pain Type')

plt.xlabel('Chest Pain Type')

plt.ylabel('Count ')
fig_dims = (6,6)

fig, ax = plt.subplots(figsize=fig_dims)

ax = sns.barplot(x="target",y="trestbps", hue="sex", data=data)

plt.title('Heart Disease Frequency According To resting blood pressure  ')

plt.xlabel('Target : Disease or not ')

plt.ylabel('trst beats per second ')
sns.catplot(x="target", y="trestbps", data=data, kind="box")
fig_dims = (6,6)

fig, ax = plt.subplots(figsize=fig_dims)

fig = sns.violinplot(x=data['target'], y=data['chol'])
data.fbs.value_counts()
sns.countplot(data=data, x="fbs", hue="target")

plt.title('Heart Disease Frequency According To FBS')

plt.xlabel('FBS - (Fasting Blood Sugar > 120 mg/dl) (1 = true; 0 = false)')

plt.xticks(rotation = 0)

plt.legend(["Haven't Disease", "Have Disease"])

plt.ylabel('Frequency of Disease or Not')

plt.show()
sequential_colors = sns.color_palette("PuRd", 2)

sns.palplot(sequential_colors)
sns.set_palette(sequential_colors)
sns.countplot( x=data['restecg'], hue=data['target'])
220-80

sns.boxplot(y=data['thalach'], x=data['target'])
import plotly.express as px
fig = px.scatter(data, x="thalach", y="age", color="target")

fig.add_shape(

        # Line Vertical

        dict(

            type="line",

            x0=140,

            y0=0,

            x1=140,

            y1=80,

            line=dict(

                color="RoyalBlue",

                width=3

            )

))

fig.add_shape(

        # Line Vertical

        dict(

            type="line",

            x0=190,

            y0=0,

            x1=190,

            y1=80,

            line=dict(

                color="RoyalBlue",

                width=3

            )

))
sns.set_palette("Paired")
sns.violinplot(x=data["exang"], y=data["target"])
sequential_colors = sns.color_palette("summer", 2)

sns.palplot(sequential_colors)

sns.set_palette(sequential_colors)
sns.barplot(x=data["target"], y=data["oldpeak"])
a = pd.get_dummies(data['cp'], prefix = "cp")

b = pd.get_dummies(data['thal'], prefix = "thal")

c = pd.get_dummies(data['slope'], prefix = "slope")

d = pd.get_dummies(data['sex'], prefix = "sex")
updated_clms = [data, a,b,c,d]

data = pd.concat(updated_clms, axis=1)

data.head()
data = data.drop(columns = ['cp','thal', 'slope', 'sex'])
from sklearn.preprocessing import StandardScaler

standardScaler = StandardScaler()

columns_to_scale = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']

data[columns_to_scale] = standardScaler.fit_transform(data[columns_to_scale])
data.head()
y = data.target

y
# Just for demonstration purposes we will be implementing with all features

X_important_features = data[['cp_1', 'cp_2','cp_3','thal_0','thal_1','thal_2','thal_3','slope_0','slope_1','slope_2']]
# 

X = data.drop(['target'], axis = 1)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2,random_state=0)

#transpose matrices

X_train = X_train.T

y_train = y_train.T

X_test = X_test.T

y_test = y_test.T
from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()

nb.fit(X_train.T, y_train.T)
accuracies = {}

acc = nb.score(X_test.T,y_test.T)*100

accuracies['Naive Bayes'] = acc

print("Accuracy of Naive Bayes: {:.2f}%".format(acc))
y_pred=nb.predict(X_test.T)

y_pred
precisions={}

recalls={}

f1_scores={}
from sklearn.metrics import confusion_matrix

from sklearn.metrics import roc_auc_score
cf_matrix = confusion_matrix(y_test.T, y_pred)

print(cf_matrix)
sns.heatmap(cf_matrix/np.sum(cf_matrix), annot=True, 

            fmt='.2%', cmap='Blues')
import scikitplot as skplt

import matplotlib.pyplot as plt



y_pred_proba = nb.predict_proba(X_test.T)

y_true = y_test.T

skplt.metrics.plot_roc_curve(y_true, y_pred_proba)

plt.show()
#recall

from sklearn.metrics import recall_score

recall = recall_score(y_test.T, y_pred)

print('Recall: %.3f' % recall)



recalls['Naive Bayes'] = recall
# precision

from sklearn.metrics import precision_score

precision = precision_score(y_test.T, y_pred)

print('Precision: %.3f' % precision)



precisions['Naive Bayes'] = precision
# fi score

f1 = 2*((precision*recall)/(precision+recall))

f1_scores['Naive Bayes'] = f1

print(f1)
import pickle

filename = 'naive_bayes.h5'

pickle.dump(nb, open(filename, 'wb'))
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 2)  # n_neighbors means k

knn.fit(X_train.T, y_train.T)

prediction = knn.predict(X_test.T)

print("{} NN Score: {:.2f}%".format(2, knn.score(X_test.T, y_test.T)*100))
knn = KNeighborsClassifier(n_neighbors = 18)  # n_neighbors means k

knn.fit(X_train.T, y_train.T)

prediction = knn.predict(X_test.T)

print("{} NN Score: {:.2f}%".format(2, knn.score(X_test.T, y_test.T)*100))
acc = knn.score(X_test.T,y_test.T)*100

accuracies['KNN'] = acc

print("Accuracy of KNN: {:.2f}%".format(acc))
y_pred=knn.predict(X_test.T)

y_pred
cf_matrix = confusion_matrix(y_test.T, y_pred)



sns.heatmap(cf_matrix/np.sum(cf_matrix), annot=True, 

            fmt='.2%', cmap='Blues')
import scikitplot as skplt

import matplotlib.pyplot as plt



y_pred_proba = knn.predict_proba(X_test.T)

y_true = y_test.T

skplt.metrics.plot_roc_curve(y_true, y_pred_proba)

plt.show()
from sklearn.metrics import recall_score

recall = recall_score(y_test.T, y_pred)

print('Recall: %.3f' % recall)

recalls['KNN'] = recall
# precision

from sklearn.metrics import precision_score

precision = precision_score(y_test.T, y_pred)

print('Precision: %.3f' % precision)

precisions['KNN'] = precision
# fi score

f1 =2*((precision*recall)/(precision+recall))

print("F1 Score ",f1)

f1_scores['KNN'] = f1
import pickle

filename = 'knn_model.sav'

pickle.dump(knn, open(filename, 'wb'))
from sklearn.tree import DecisionTreeClassifier



# Define model. Specify a number for random_state to ensure same results each run

dt = DecisionTreeClassifier(random_state=1)



# Fit model

dt.fit(X_train.T, y_train.T)
acc = dt.score(X_test.T,y_test.T)*100

accuracies['Decision Tree'] = acc

print("Accuracy of Decision Tree: {:.2f}%".format(acc))
y_pred=dt.predict(X_test.T)

y_pred
cf_matrix = confusion_matrix(y_test.T, y_pred)



sns.heatmap(cf_matrix/np.sum(cf_matrix), annot=True, 

            fmt='.2%', cmap='Blues')
import scikitplot as skplt

import matplotlib.pyplot as plt



y_pred_proba = dt.predict_proba(X_test.T)

y_true = y_test.T

skplt.metrics.plot_roc_curve(y_true, y_pred_proba)

plt.show()
from sklearn.metrics import recall_score

recall = recall_score(y_test, y_pred)

print('Recall: %.3f' % recall)

recalls['Decision Tree'] = recall
# precision

from sklearn.metrics import precision_score

precision = precision_score(y_test, y_pred, labels=[1,2], average='micro')

print('Precision: %.3f' % precision)

precisions['Decision Tree'] = precision
# fi score

f1 = 2*((precision*recall)/(precision+recall))

print("F1 Score ",f1)

f1_scores['Decision Tree']= f1
import pickle

filename = 'dt_model.sav'

pickle.dump(dt, open(filename, 'wb'))
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_val_score
rf = RandomForestClassifier(n_estimators=1,random_state=1)

rf.fit(X_test.T,y_test.T)
acc = rf.score(X_test.T,y_test.T)*100

accuracies['Random Forest'] = acc

print("Accuracy of Random Forest: {:.2f}%".format(acc))
y_pred=rf.predict(X_test.T)

y_pred
cf_matrix = confusion_matrix(y_test.T, y_pred)



sns.heatmap(cf_matrix/np.sum(cf_matrix), annot=True, 

            fmt='.2%', cmap='Blues')
import scikitplot as skplt

import matplotlib.pyplot as plt



y_pred_proba = rf.predict_proba(X_test.T)

y_true = y_test.T

skplt.metrics.plot_roc_curve(y_true, y_pred_proba)

plt.show()
from sklearn.metrics import recall_score

recall = recall_score(y_test, y_pred)

print('Recall: %.3f' % recall)

recalls['Random Forest'] = recall
# precision

from sklearn.metrics import precision_score

precision = precision_score(y_test, y_pred, labels=[1,2], average='micro')

print('Precision: %.3f' % precision)

precisions['Random Forest'] = precision
#f1 score

f1 = 2*((precision*recall)/(precision+recall))

print("F1 Score ",f1)

f1_scores['Random Forest']= f1
import pickle

filename = 'rf_model.sav'

pickle.dump(rf, open(filename, 'wb'))
f1_scores
y = accuracies.values()

x = accuracies.keys()

plt.bar(x,y)
recalls

precisions

f1_scores
y = recalls.values()

x = recalls.keys()

plt.bar(x,y)
y = precisions.values()

x = precisions.keys()

plt.bar(x,y)
y = f1_scores.values()

x = f1_scores.keys()

plt.bar(x,y)