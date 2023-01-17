%matplotlib inline

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt



col_names=['BI_RADS', 'age', 'shape', 'margin', 'density','severity']

file=pd.read_csv("../input/mammographic_masses.data.txt")

file.head(5)
file=pd.read_csv("../input/mammographic_masses.data.txt", na_values='?', names=col_names, usecols=range(1,6))

file.describe(include=("all"))
print("Number of NULL values per feature:")

file.isnull().sum()
file.dropna(inplace=True)

file.shape

file. describe(include=("all"))
fig, axes = plt.subplots(1,4, sharey=False, figsize=(18,4))

ax1, ax2, ax3, ax4 = axes.flatten()



ax1.hist(file['age'], bins=10, color="lightslategray")

ax2.hist(file['shape'], bins=4, color="steelblue")

ax3.hist(file['margin'], bins=5, color="mediumslateblue")

ax4.hist(file['density'], bins=4, color="darkslategray")

ax1.set_xlabel('AGE', fontsize="large")

ax2.set_xlabel('SHAPE', fontsize="large")

ax3.set_xlabel('MARGIN', fontsize="large")

ax4.set_xlabel('DENSITY', fontsize="large")

ax1.set_ylabel("AMOUNT", fontsize="large")



plt.suptitle('COMPARISON of DISTRIBUTIONS', ha='center', fontsize='x-large')

plt.show()
feature_names=['age', 'shape', 'margin', 'density']

features=file[['age', 'shape', 'margin', 'density']].values

classes=file['severity'].values
from sklearn.metrics import accuracy_score

from sklearn.preprocessing import StandardScaler

scaler=StandardScaler()

features_scaled=scaler.fit_transform(features)

print("Scaled features:")

features_scaled
from sklearn.model_selection import train_test_split

train_f, test_f, train_c, test_c=train_test_split(features_scaled, classes, test_size=0.2, random_state=0)
from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score



dicti={SVC(kernel="rbf", C=1, gamma=1000, probability=True):"svc",

    LogisticRegression(solver="liblinear", random_state=0):"lr",

    KNeighborsClassifier(n_neighbors=10):'knn',

    RandomForestClassifier(max_depth=3, n_estimators=100, random_state=0):'rfc',

    DecisionTreeClassifier(random_state=0):'dtc'}

for model in dicti:

    model.fit(train_f, train_c)

    pred_c=model.predict(test_f)

    accc=accuracy_score(test_c, pred_c)

    print("Accuracy score for ", dicti[model], " is ", accc.round(2))
from sklearn.model_selection import cross_val_score

for model in dicti:

    score=cross_val_score(model,features_scaled,classes, cv=10)

    print("Accuracy score for ", dicti[model], "with cros. val. is ",'{:3.2f}'.format(score.mean()))
from sklearn.neighbors import KNeighborsClassifier

for n in range(1,11):

    model=KNeighborsClassifier(n_neighbors=n)

    model.fit(train_f, train_c)

    pred_c=model.predict(test_f)

    acc=accuracy_score(test_c, pred_c)

    print(n,"neighbor(s):")

    print("Accuracy score for KNN is :", acc.round(2))

    score=cross_val_score(model,train_f,train_c, cv=10)

    print("Accuracy score for KNN with cros. val. is ",'{:3.2f}'.format(score.mean()),"\n")
from sklearn.naive_bayes import MultinomialNB

from sklearn.preprocessing import MinMaxScaler

scaler=MinMaxScaler()

MNB=MultinomialNB()

features_minmax=scaler.fit_transform(features)

cv_scores=cross_val_score(MNB,features_minmax, classes, cv=10)

print("Accuracy score for Multinomial Naive Bayes with cross. val. is ",'{:3.2f}'.format(score.mean()))
from sklearn.metrics import roc_curve, auc

dicti={SVC(kernel="rbf", C=1, gamma=1000, probability=True):"svc",

    LogisticRegression(solver="liblinear", random_state=0):"lr",

    KNeighborsClassifier(n_neighbors=10):'knc',

    RandomForestClassifier(max_depth=3, n_estimators=100):'rfc',

    DecisionTreeClassifier():'dtc'}

for model in dicti:

    model.fit(train_f,train_c)

    prob=model.predict_proba(test_f)

    fpr, tpr, thresholds=roc_curve(test_c, prob[:,1])

    roc_auc=auc(fpr, tpr)

    plt.plot(fpr, tpr, lw=3, label=dicti[model]+' AUC = %0.2f' % roc_auc)

    plt.legend(loc='lower right')

plt.plot([0, 1], [0, 1],'r--')

plt.title('Receiver Operating Characteristic', fontsize=15)



plt.xlim([-0.02, 1.02])

plt.ylim([-0.02, 1.02])

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')

plt.rcParams["figure.figsize"] = (10,5)

plt.show()
from sklearn.decomposition import PCA

pca=PCA()

train_f_pca=pca.fit_transform(train_f)

test_f_pca=pca.transform(test_f)



df = pd.DataFrame({'Variance Explained':pca.explained_variance_ratio_,

             'Principal Components':['PC1','PC2', 'PC3', 'PC4']})

sns.barplot(x='Principal Components',y="Variance Explained", data=df, color="b")

plt.title("Variance Explained by Principal Components\n", fontsize=20, color="b")

plt.show()
print("Explained variance per component:")

pca.explained_variance_ratio_.tolist()
from matplotlib.colors import ListedColormap

pca2 = PCA(2)  # project from 4 to 2 dimensions

train_f_pca2=pca2.fit_transform(train_f)

test_f_pca2=pca2.transform(test_f)

plt.scatter(train_f_pca2[:, 0], train_f_pca2[:, 1], c=train_c, edgecolor='k',s=50, alpha=0.7, cmap=ListedColormap(('g','r')))

plt.xlabel('component 1')

plt.ylabel('component 2')

plt.title("Visualization of Train Data\n with two components\n", color="r", fontsize=15)

plt.colorbar(label='benign'+" "*15+'malignant')

plt.show()
plt.scatter(test_f_pca2[:, 0], test_f_pca2[:, 1], c=test_c, edgecolor='black',s=50, alpha=0.7, cmap=ListedColormap(('g','r')))

plt.xlabel('component 1')

plt.ylabel('component 2')

plt.title('Visualization of Test Data\n with two components')

plt.colorbar(label='beign'+" "*15+'malign')

plt.title("Visualization of Test Data\n with two components\n", color="r", fontsize=15)

plt.show()
from sklearn.model_selection import train_test_split

train_f, test_f, train_c, test_c=train_test_split(features_scaled, classes, test_size=0.2, random_state=0)



classifier =LogisticRegression(solver="liblinear", random_state=0)

classifier.fit(train_f_pca2, train_c)

pred_c = classifier.predict(test_f_pca2)

plt.subplot(2,1,1)



#Train set boundary

X_set, y_set = train_f_pca2, train_c

X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),\

                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))

plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),

                 alpha = 0.6, cmap = ListedColormap(('green', 'red')))

plt.xlim(X1.min(), X1.max())

plt.ylim(X2.min(), X2.max())

for i, j in enumerate(np.unique(y_set)):

    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],s=10,

                c = ListedColormap(('green', 'red'))(i), label = j)

plt.title('Logistic Rgression\nBoundary Line with PCA (Train Set)')

plt.xlabel('Component 1')

plt.ylabel('Component 2')

plt.legend()

       

plt.subplot(2,1,2)

#Test set boundary

X_set, y_set = test_f_pca2, test_c

X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),\

                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))

plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),

                 alpha = 0.6, cmap = ListedColormap(('green', 'red')))

plt.xlim(X1.min(), X1.max())

plt.ylim(X2.min(), X2.max())

for i, j in enumerate(np.unique(y_set)):

    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],s=10, c = ListedColormap(('green', 'red'))(i), label = j)

plt.title('Boundary Line with PCA (Test Set)')

plt.xlabel('Component 1')

plt.ylabel('Component 2')

plt.legend()

plt.tight_layout()

plt.show()
from sklearn.model_selection import train_test_split

train_f, test_f, train_c, test_c=train_test_split(features_scaled, classes, test_size=0.2, random_state=0)



classifier = RandomForestClassifier(max_depth=3, n_estimators=100)

classifier.fit(train_f_pca2, train_c)

pred_c = classifier.predict(test_f_pca2)

   

plt.subplot(2,1,1)

#Train set boundary

X_set, y_set = train_f_pca2, train_c

X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),\

                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))

plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),

                 alpha = 0.6, cmap = ListedColormap(('green', 'red')))

plt.xlim(X1.min(), X1.max())

plt.ylim(X2.min(), X2.max())

for i, j in enumerate(np.unique(y_set)):

    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],s=10,

                c = ListedColormap(('green', 'red'))(i), label = j)

plt.title('Randon Forest Classifier\nBoundary Line with PCA (Train Set)')

plt.xlabel('Component 1')

plt.ylabel('Component 2')

plt.legend()

       

plt.subplot(2,1,2)

#Test set boundary

X_set, y_set = test_f_pca2, test_c

X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),\

                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))

plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),

                 alpha = 0.6, cmap = ListedColormap(('green', 'red')))

plt.xlim(X1.min(), X1.max())

plt.ylim(X2.min(), X2.max())

for i, j in enumerate(np.unique(y_set)):

    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],s=10, c = ListedColormap(('green', 'red'))(i), label = j)

plt.title('Boundary Line with PCA (Test Set)')

plt.xlabel('Component 1')

plt.ylabel('Component 2')

plt.legend()

plt.tight_layout()

plt.show()   