import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import numpy as np

import pandas as pd

import seaborn as sns

import warnings

warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt

import os
df0 = pd.read_csv("/kaggle/input/predict-the-disease/Training.csv")

print("Dataset with rows {} and columns {}".format(df0.shape[0],df0.shape[1]))

df0.head()
df1 = pd.read_csv("/kaggle/input/predict-the-disease/Testing.csv")

print("Dataset with rows {} and columns {}".format(df1.shape[0],df1.shape[1]))

df1.head()
df0.describe()
df0.isnull().sum()
# Training data

df0.prognosis.value_counts()
# Testing data

df1.prognosis.value_counts()
df0.prognosis.value_counts(normalize=True)
df1.prognosis.value_counts(normalize=True)
plt.figure(figsize=(18,5))

sns.countplot('prognosis',data=df0)

plt.show()
X= df0.drop(["prognosis"],axis=1)

y = df0['prognosis']
X1= df1.drop(["prognosis"],axis=1)

y1 = df1['prognosis']
plt.figure(figsize=(20,20))

sns.heatmap(X.corr(),annot=True)
# initializing the pca

from sklearn import decomposition

pca = decomposition.PCA()
# configuring the parameteres

# the number of components = 2

pca.n_components = 2

pca_data = pca.fit_transform(X)



# pca_reduced will contain the 2-d projects of simple data

print("shape of pca_reduced.shape = ", pca_data.shape)
# attaching the label for each 2-d data point 

pca_data = np.vstack((pca_data.T, y)).T



# creating a new data fram which help us in ploting the result data

pca_df = pd.DataFrame(data=pca_data, columns=("1st_principal", "2nd_principal", "prognosis"))

sns.FacetGrid(pca_df, hue="prognosis", size=6).map(plt.scatter, '1st_principal', '2nd_principal').add_legend()

plt.show()
# PCA for dimensionality redcution (non-visualization)



pca.n_components = 132

pca_data = pca.fit_transform(X)



percentage_var_explained = pca.explained_variance_ / np.sum(pca.explained_variance_);



cum_var_explained = np.cumsum(percentage_var_explained)



# Plot the PCA spectrum

plt.figure(1, figsize=(6, 4))



plt.clf()

plt.plot(cum_var_explained, linewidth=2)

plt.axis('tight')

plt.grid()

plt.xlabel('n_components')

plt.ylabel('Cumulative_explained_variance')

plt.show()



# data prepararion

from wordcloud import WordCloud 

x2011 = df0.prognosis

plt.subplots(figsize=(8,8))

wordcloud = WordCloud(

                          background_color='white',

                          width=800,

                          height=800

                         ).generate(" ".join(x2011))

plt.title('Diseases',size=30)

plt.imshow(wordcloud)

plt.axis('off')

plt.show()
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

from sklearn.model_selection import cross_val_score
train_X, val_X, train_y, val_y = train_test_split(X, y, test_size=0.33, random_state=42)
rfc_mod = RandomForestClassifier(random_state=42, class_weight='balanced').fit(train_X, train_y)
y_pred_rfc = rfc_mod.predict(val_X)

y_pred_rfc
y_pred_rfc1 = rfc_mod.predict(X1)

y_pred_rfc1
print("Accuracy Score:", accuracy_score(y_pred_rfc, val_y))

print('cross validation:',cross_val_score(rfc_mod, X, y, cv=3).mean())

print("F1 Score :",f1_score(y_pred_rfc,val_y,average = "weighted"))

print('Report:\n',classification_report(val_y, y_pred_rfc))

print('Confusion Matrix: \n',confusion_matrix(val_y, y_pred_rfc))
print("Accuracy Score:", accuracy_score(y_pred_rfc1, y1))

print('cross validation:',cross_val_score(rfc_mod, X, y, cv=3).mean())

print("F1 Score :",f1_score(y_pred_rfc,val_y,average = "weighted"))

print('Report:\n',classification_report(val_y, y_pred_rfc))

print('Confusion Matrix: \n',confusion_matrix(val_y, y_pred_rfc))
importances=rfc_mod.feature_importances_

feature_importances=pd.Series(importances, index=train_X.columns).sort_values(ascending=False)

plt.figure(figsize=(10,7))

sns.barplot(x=feature_importances[0:20], y=feature_importances.index[0:20])

plt.title('Feature Importance RFC Model',size=20)

plt.ylabel("Features")

plt.show()
from sklearn.feature_selection import RFE
rfe = RFE(rfc_mod, 20)

rfe.fit(train_X,train_y)
train_X.columns[rfe.support_]
colm = train_X.columns[rfe.support_]
rfc_mod.fit(train_X[colm],train_y)
y_pred_rfc2 = rfc_mod.predict(val_X[colm])

y_pred_rfc2
print("Accuracy Score:", accuracy_score(y_pred_rfc2, val_y))

print("F1 Score :",f1_score(y_pred_rfc2,val_y,average = "weighted"))

print('Report:\n',classification_report(val_y, y_pred_rfc2))

print('Confusion Matrix: \n',confusion_matrix(val_y, y_pred_rfc2))
dectre_mod = DecisionTreeClassifier(random_state=42, class_weight='balanced').fit(train_X,train_y)
y_pred_dectre = dectre_mod.predict(val_X)

y_pred_dectre
y_pred_dectre1 = dectre_mod.predict(X1)

y_pred_dectre1
print("Accuracy Score:", accuracy_score(y_pred_dectre, val_y))

print("F1 Score :",f1_score(y_pred_dectre,val_y,average = "weighted"))

print('Report:\n',classification_report(val_y, y_pred_dectre))

print('Confusion Matrix: \n',confusion_matrix(val_y, y_pred_dectre))
print("Accuracy Score:", accuracy_score(y_pred_dectre1, y1))

print("F1 Score :",f1_score(y_pred_dectre1,y1,average = "weighted"))

print('Report:\n',classification_report(y1, y_pred_dectre1))

print('Confusion Matrix: \n',confusion_matrix(y1, y_pred_dectre1))
importances=dectre_mod.feature_importances_

feature_importances=pd.Series(importances, index=train_X.columns).sort_values(ascending=False)

plt.figure(figsize=(10,7))

sns.barplot(x=feature_importances[0:20], y=feature_importances.index[0:20])

plt.title('Feature Importance Decision Tree Model',size=20)

plt.ylabel("Features")

plt.show()
rfe = RFE(dectre_mod, 20)

rfe.fit(train_X,train_y)
train_X.columns[rfe.support_]
col = train_X.columns[rfe.support_]
dectre_mod.fit(train_X[col],train_y)
y_pred_dectre2 = dectre_mod.predict(val_X[col])

y_pred_dectre2
print("Accuracy Score:", accuracy_score(y_pred_dectre2, val_y))

print("F1 Score :",f1_score(y_pred_dectre2,val_y,average = "weighted"))

print('Report:\n',classification_report(val_y, y_pred_dectre2))

print('Confusion Matrix: \n',confusion_matrix(val_y, y_pred_dectre2))
logreg_mod = LogisticRegression(random_state=42, solver='lbfgs', class_weight='balanced').fit(train_X,train_y)
y_pred_logreg = logreg_mod.predict(val_X)

y_pred_logreg
y_pred_logreg1 = logreg_mod.predict(X1)

y_pred_logreg1
print("Accuracy Score:", accuracy_score(y_pred_logreg, val_y))

print("F1 Score :",f1_score(y_pred_logreg,val_y,average = "weighted"))

print('Report:\n',classification_report(val_y, y_pred_logreg))

print('Confusion Matrix: \n',confusion_matrix(val_y, y_pred_logreg))
print("Accuracy Score:", accuracy_score(y_pred_logreg1, y1))

print("F1 Score :",f1_score(y_pred_logreg1,y1,average = "weighted"))

print('Report:\n',classification_report(y1, y_pred_logreg1))

print('Confusion Matrix: \n',confusion_matrix(y1, y_pred_logreg1))
rfe = RFE(logreg_mod, 20)

rfe.fit(train_X,train_y)
train_X.columns[rfe.support_]
cols = train_X.columns[rfe.support_]
logreg_mod.fit(train_X[cols],train_y)
y_pred_logreg2 = logreg_mod.predict(val_X[cols])

y_pred_logreg2
for i in y_pred_logreg2:

    print(i)
print("Accuracy Score:", accuracy_score(y_pred_logreg2, val_y))

print("F1 Score :",f1_score(y_pred_logreg2,val_y,average = "weighted"))

print('Report:\n',classification_report(val_y, y_pred_logreg2))

print('Confusion Matrix: \n',confusion_matrix(val_y, y_pred_logreg2))
y_pred_logreg3 = logreg_mod.predict(X1[cols])

y_pred_logreg3
for i in y_pred_logreg3:

    print(i)
print("Accuracy Score:", accuracy_score(y_pred_logreg3, y1))

print("F1 Score :",f1_score(y_pred_logreg3,y1,average = "weighted"))

print('Report:\n',classification_report(y1, y_pred_logreg3))

print('Confusion Matrix: \n',confusion_matrix(y1, y_pred_logreg3))