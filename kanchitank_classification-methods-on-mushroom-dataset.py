import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns # plotting graphs for visualizations

import matplotlib.pyplot as plt # plotting graphs for visualizations

import os

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, auc, roc_curve

from sklearn.tree import DecisionTreeClassifier, export_graphviz

import graphviz



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df = pd.read_csv("/kaggle/input/mushroom-classification/mushrooms.csv")
df.head()
df.info()
df.describe()
print("Dataset shape:",df.shape)
df["class"].unique()
count = df.iloc[:,0].value_counts()

plt.figure(figsize=(8,7))

sns.barplot(count.index, count.values, alpha=0.8, palette="prism")

plt.ylabel('Count', fontsize=12)

plt.xlabel('Class', fontsize=12)

plt.title('Number of poisonous/edible mushrooms')

plt.show()
labelencoder=LabelEncoder()

for column in df.columns:

    df[column] = labelencoder.fit_transform(df[column])
df.head()
df=df.drop(["veil-type"],axis=1)
df_div = pd.melt(df, "class", var_name="Characteristics")

fig, ax = plt.subplots(figsize=(16,6))

p = sns.violinplot(ax = ax, x="Characteristics", y="value", hue="class", split = True, data=df_div, inner = 'quartile', palette = 'Set1')

df_no_class = df.drop(["class"],axis = 1)

p.set_xticklabels(rotation = 90, labels = list(df_no_class.columns));
plt.figure(figsize=(14,12))

sns.heatmap(df.corr(),linewidths=.1,cmap="Purples", annot=True)

plt.yticks(rotation=0);
df[['class', 'gill-color']].groupby(['gill-color'], as_index=False).mean().sort_values(by='class', ascending=False)
new_var = df[['class', 'gill-color']]

new_var = new_var[new_var['gill-color']<=3.5]

sns.factorplot('class', col='gill-color', data=new_var, kind='count', size=4.5, aspect=.8, col_wrap=4);
new_var=df[['class', 'gill-color']]

new_var=new_var[new_var['gill-color']>3.5]



sns.factorplot('class', col='gill-color', data=new_var, kind='count', size=4.5, aspect=.8, col_wrap=4);
X = df.drop(['class'], axis=1)  

y = df["class"]



X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.1)   
from sklearn.tree import DecisionTreeClassifier



dt = DecisionTreeClassifier()

dt.fit(X_train, y_train)
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'



dot_data = export_graphviz(dt, out_file=None, 

                         feature_names=X.columns,  

                         filled=True, rounded=True,  

                         special_characters=True)  

graph = graphviz.Source(dot_data)  

graph 
features_list = X.columns.values

feature_importance = dt.feature_importances_

sorted_idx = np.argsort(feature_importance)



plt.figure(figsize=(8,7))

plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx], align='center', color ="red")

plt.yticks(range(len(sorted_idx)), features_list[sorted_idx])

plt.xlabel('Importance')

plt.title('Feature importances')

plt.draw()

plt.show()
y_pred_dt = dt.predict(X_test)
print("Decision Tree Classifier report: \n\n", classification_report(y_test, y_pred_dt))
print("Test Accuracy: {}%".format(round(dt.score(X_test, y_test)*100, 2)))
cm = confusion_matrix(y_test, y_pred_dt)



x_axis_labels = ["Edible", "Poisonous"]

y_axis_labels = ["Edible", "Poisonous"]



f, ax = plt.subplots(figsize =(7,7))

sns.heatmap(cm, annot = True, linewidths=0.2, linecolor="black", fmt = ".0f", ax=ax, cmap="Purples", xticklabels=x_axis_labels, yticklabels=y_axis_labels)

plt.xlabel("PREDICTED LABEL")

plt.ylabel("TRUE LABEL")

plt.title('Confusion Matrix for Decision Tree Classifier')

plt.show()
from sklearn.linear_model import LogisticRegression



lr = LogisticRegression(solver="lbfgs", max_iter=500)

lr.fit(X_train, y_train)



print("Test Accuracy: {}%".format(round(lr.score(X_test, y_test)*100,2)))
y_pred_lr = lr.predict(X_test)

print("Logistic Regression Classifier report: \n\n", classification_report(y_test, y_pred_lr))
cm = confusion_matrix(y_test, y_pred_lr)



x_axis_labels = ["Edible", "Poisonous"]

y_axis_labels = ["Edible", "Poisonous"]



f, ax = plt.subplots(figsize =(7,7))

sns.heatmap(cm, annot = True, linewidths=0.2, linecolor="black", fmt = ".0f", ax=ax, cmap="Purples", xticklabels=x_axis_labels, yticklabels=y_axis_labels)

plt.xlabel("PREDICTED LABEL")

plt.ylabel("TRUE LABEL")

plt.title('Confusion Matrix for Logistic Regression Classifier')

plt.show()
from sklearn.neighbors import KNeighborsClassifier



best_Kvalue = 0

best_score = 0



for i in range(1,10):

    knn = KNeighborsClassifier(n_neighbors=i)

    knn.fit(X_train, y_train)

    if knn.score(X_test, y_test) > best_score:

        best_score = knn.score(X_train, y_train)

        best_Kvalue = i

        

print("Best KNN Value: {}".format(best_Kvalue))

print("Test Accuracy: {}%".format(round(best_score*100,2)))
y_pred_knn = knn.predict(X_test)

print("KNN Classifier report: \n\n", classification_report(y_test, y_pred_knn))
cm = confusion_matrix(y_test, y_pred_knn)



x_axis_labels = ["Edible", "Poisonous"]

y_axis_labels = ["Edible", "Poisonous"]



f, ax = plt.subplots(figsize =(7,7))

sns.heatmap(cm, annot = True, linewidths=0.2, linecolor="black", fmt = ".0f", ax=ax, cmap="Purples", xticklabels=x_axis_labels, yticklabels=y_axis_labels)

plt.xlabel("PREDICTED LABEL")

plt.ylabel("TRUE LABEL")

plt.title('Confusion Matrix for KNN Classifier')

plt.show()
from sklearn.svm import SVC



svm = SVC(random_state=42, gamma="auto")

svm.fit(X_train, y_train)



print("Test Accuracy: {}%".format(round(svm.score(X_test, y_test)*100, 2)))
y_pred_svm = svm.predict(X_test)

print("SVM Classifier report: \n\n", classification_report(y_test, y_pred_svm))
cm = confusion_matrix(y_test, y_pred_svm)



x_axis_labels = ["Edible", "Poisonous"]

y_axis_labels = ["Edible", "Poisonous"]



f, ax = plt.subplots(figsize =(7,7))

sns.heatmap(cm, annot = True, linewidths=0.2, linecolor="black", fmt = ".0f", ax=ax, cmap="Purples", xticklabels=x_axis_labels, yticklabels=y_axis_labels)

plt.xlabel("PREDICTED LABEL")

plt.ylabel("TRUE LABEL")

plt.title('Confusion Matrix for SVM Classifier')

plt.show()
from sklearn.naive_bayes import GaussianNB



nb = GaussianNB()

nb.fit(X_train, y_train)



print("Test Accuracy: {}%".format(round(nb.score(X_test, y_test)*100, 2)))
y_pred_nb = nb.predict(X_test)

print("Naive Bayes Classifier report: \n\n", classification_report(y_test, y_pred_nb))
cm = confusion_matrix(y_test, y_pred_nb)



x_axis_labels = ["Edible", "Poisonous"]

y_axis_labels = ["Edible", "Poisonous"]



f, ax = plt.subplots(figsize =(7,7))

sns.heatmap(cm, annot = True, linewidths=0.2, linecolor="black", fmt = ".0f", ax=ax, cmap="Purples", xticklabels=x_axis_labels, yticklabels=y_axis_labels)

plt.xlabel("PREDICTED LABEL")

plt.ylabel("TRUE LABEL")

plt.title('Confusion Matrix for Naive Bayes Classifier')

plt.show()
from sklearn.ensemble import RandomForestClassifier



rf = RandomForestClassifier(n_estimators=100, random_state=42)

rf.fit(X_train, y_train)



print("Test Accuracy: {}%".format(round(rf.score(X_test, y_test)*100, 2)))
y_pred_rf = rf.predict(X_test)

print("Random Forest Classifier report: \n\n", classification_report(y_test, y_pred_rf))
cm = confusion_matrix(y_test, y_pred_rf)



x_axis_labels = ["Edible", "Poisonous"]

y_axis_labels = ["Edible", "Poisonous"]



f, ax = plt.subplots(figsize =(7,7))

sns.heatmap(cm, annot = True, linewidths=0.2, linecolor="black", fmt = ".0f", ax=ax, cmap="Purples", xticklabels=x_axis_labels, yticklabels=y_axis_labels)

plt.xlabel("PREDICTED LABEL")

plt.ylabel("TRUE LABEL")

plt.title('Confusion Matrix for Random Forest Classifier');

plt.show()
preds = dt.predict(X_test)



print(preds[:36])

print(y_test[:36].values)



# 0 - Edible

# 1 - Poisonous