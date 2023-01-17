import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn import preprocessing

from sklearn.linear_model import LogisticRegression

from sklearn.decomposition import PCA, KernelPCA

from sklearn.metrics import classification_report

from sklearn.model_selection import cross_val_score, train_test_split

from matplotlib import pyplot as plt
# Import the data and look at the first few rows

dat = pd.read_csv("../input/data.csv")

dat.head()
# The id column is unimportant and the last column has NaN values --> drop

dat = dat.drop(["id","Unnamed: 32"],axis=1)

# separate the features X and the class labels y

X = preprocessing.scale(dat.drop("diagnosis",axis=1))

y = dat["diagnosis"]

# plot a 2D representation of the features for each class using PCA

X_M = X[y=="M"]

X_B = X[y=="B"]

pca = PCA(n_components=2).fit(X)

plt.scatter(*(pca.transform(X_M).transpose()))

plt.scatter(*(pca.transform(X_B).transpose()))
clf = LogisticRegression(random_state=0, solver='lbfgs',multi_class='multinomial',max_iter=100)

cross_val_score(clf, X, y, scoring='recall_macro', cv=5)
# partition the data into test and train data

training_fraction = 0.8

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.50, random_state=0, stratify=y)

y_pred = clf.fit(X_train,y_train).predict(X_test)

print(classification_report(y_test,y_pred))