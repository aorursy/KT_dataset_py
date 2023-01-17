import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt



import missingno as msno



from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV

from sklearn.svm import SVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB



from sklearn.metrics import plot_confusion_matrix

from sklearn.metrics import roc_auc_score, roc_curve

from sklearn.metrics import recall_score

from sklearn.metrics import precision_score

from sklearn.metrics import f1_score

from sklearn.metrics import accuracy_score



%matplotlib inline
df = pd.read_csv("../input/mushroom-classification/mushrooms.csv")

df.head()
def display_all(df):

    with pd.option_context("display.max_rows", 100, "display.max_columns", 100):

        display(df)
display_all(df.head())
df.info()
df.describe().T
msno.matrix(df)

plt.show()
sns.countplot(x="class",data=df)

plt.show()
for i in df.columns:

    print(i, df[i].unique())

    print("----------------")
catog_feat = df.drop("class",axis=1).columns.tolist()

fig, axes = plt.subplots(11,2, figsize=(20,75))

for variable, subplot in zip(catog_feat, axes.flatten()):

    sns.countplot(x=variable, data=df, ax=subplot,hue="class")
used_data = ["odor", "bruises", "gill-spacing", "gill-size", "gill-color","stalk-shape", "stalk-surface-below-ring", "stalk-color-above-ring",

             "ring-type","spore-print-color", "population", "habitat"]

X = df[used_data]

X_dum = pd.get_dummies(X, drop_first=True)

X_dum
y =  df["class"].copy()

y = [1 if i == "p" else 0 for i in y]

y = np.array(y)

y.shape
X_train, X_test, y_train, y_test = train_test_split(X_dum, y, test_size=0.3, random_state=42, stratify=y, shuffle=True)

display(X_train.head())

display(y_train)
def feature_importance(df, model):

    return pd.DataFrame({"columns":df.columns, "importances":model.feature_importances_}

                       ).sort_values("importances", ascending=False)
def plot_feature_importances(x):

    return x.plot('columns', 'importances', 'barh', figsize=(12,7), legend=False)
def print_metrics(target, predictions):    

    acc_score = accuracy_score(target, predictions)

    pre_score = precision_score(target, predictions)

    rec_score = recall_score(target, predictions)

    f1 = f1_score(target, predictions)

    auc = roc_auc_score(target, predictions)



    print("Accuracy:  {0:.3f}".format(acc_score))

    print("Precision: {0:.3f}".format(pre_score))

    print("Recall:    {0:.3f}".format(rec_score))

    print("F1 Score:  {0:.3f}".format(f1))

    print("Auc Score: {0:.3f}".format(auc))
def confusion_matrices(model):

    plot_confusion_matrix(model, X_train,y_train, values_format="2d", cmap="Blues")

    plt.title("Confusion matrix for train set")

    plot_confusion_matrix(model, X_test,y_test, values_format="2d", cmap="Blues")

    plt.title("Confusion matrix for test set")

    plt.show()
def plot_roc_curve(fpr, tpr, model_name):

    x_t = np.linspace(0,1,num=3)

    y_t = np.linspace(0,1,num=3)

    

    plt.figure(figsize=(10,7))

    plt.plot(x_t, y_t, 'r--')

    plt.plot(fpr,tpr)

    plt.xlabel("False Positive Rate")

    plt.ylabel("True Positive Rate")

    plt.title("Roc Curve For " + model_name)

    

    plt.show()
rf = RandomForestClassifier()

rf.fit(X_train,y_train)

y_pred = rf.predict(X_test)

print_metrics(y_test, y_pred)
fi = feature_importance(X_train, rf)[0:5]

display(fi)

plot_feature_importances(fi);
confusion_matrices(rf)

fpr, tpr, threshold = roc_curve(y_test, y_pred)

plot_roc_curve(fpr, tpr, "Random Forest Classifier")
gn = GaussianNB()

gn.fit(X_train,y_train)

y_pred_gn = gn.predict(X_test)

print_metrics(y_test, y_pred_gn)
confusion_matrices(gn)

fpr, tpr, threshold = roc_curve(y_test, y_pred_gn)

plot_roc_curve(fpr, tpr, "Naive Bayes Classifier")
knn = KNeighborsClassifier(n_neighbors=5)

knn.fit(X_train, y_train)

y_pred_knn = knn.predict(X_test)

print_metrics(y_test, y_pred_knn)
confusion_matrices(knn)

fpr, tpr, threshold = roc_curve(y_test, y_pred_knn)

plot_roc_curve(fpr, tpr, "K Neighbors Classifier")
svc = SVC()

svc.fit(X_train, y_train)

y_pred_svc = svc.predict(X_test)

print_metrics(y_test, y_pred_svc)
confusion_matrices(svc)

fpr, tpr, threshold = roc_curve(y_test, y_pred_svc)

plot_roc_curve(fpr, tpr, "Support Vector Classifier")