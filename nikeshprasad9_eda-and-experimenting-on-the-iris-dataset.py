import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import warnings



warnings.filterwarnings("ignore")

%matplotlib inline

plt.rcParams.update({'font.size': 12})    # It just updates the fontsize to be used in matplotlib plots to 12 which is 10 by default.
data = pd.read_csv('../input/Iris.csv', index_col="Id")

data.head()  # Checking the first 5 samples from the imported data.
print("Number of samples in our dataset: ", data.shape[0])

print("Number of columns in our dataset: ", data.shape[1])
data.dtypes
data.isnull().any()
data.info()
classes = data["Species"].unique()

print("We have {} classes which are {}".format(len(classes), classes))
print(data["Species"].value_counts())



sns.countplot(data["Species"])

plt.xlabel("Species", labelpad=10, fontsize=14)

plt.ylabel("Counts", labelpad=10, fontsize=14)

plt.show()
data.hist(figsize=(18,8), edgecolor="black")

plt.show()
fig = plt.figure(figsize=(18,6))



ax1 = fig.add_subplot(121)

sns.boxplot(data=data, orient="h", ax=ax1)

ax1.set_title("Box and Whisker plot for different features")



ax2 = fig.add_subplot(122)

sns.kdeplot(data["SepalLengthCm"], shade=True, shade_lowest=True, ax=ax2)

sns.kdeplot(data["SepalWidthCm"], shade=True, shade_lowest=True, ax=ax2)

sns.kdeplot(data["PetalLengthCm"], shade=True, shade_lowest=True, ax=ax2)

sns.kdeplot(data["PetalWidthCm"], shade=True, shade_lowest=True, ax=ax2)

ax2.set_title("Kernel Density Estimation plot for different features")



ax2.legend(fontsize=12)



plt.show()
fig, ax = plt.subplots(1,4, figsize=(18,6))



setosa = data[data["Species"]=="Iris-setosa"]

versicolor = data[data["Species"]=="Iris-versicolor"]

virginica = data[data["Species"]=="Iris-virginica"]



sns.kdeplot(setosa["PetalWidthCm"], label="setosa", shade=True, ax=ax[0])

sns.kdeplot(versicolor["PetalWidthCm"], label="versicolor", shade=True, ax=ax[0])

sns.kdeplot(virginica["PetalWidthCm"], label="virginica", shade=True, ax=ax[0])

ax[0].set_title("Species vs PetalWidthCm")

ax[0].legend()



sns.kdeplot(setosa["PetalLengthCm"], label="setosa", shade=True, ax=ax[1])

sns.kdeplot(versicolor["PetalLengthCm"], label="versicolor", shade=True, ax=ax[1])

sns.kdeplot(virginica["PetalLengthCm"], label="virginica", shade=True, ax=ax[1])

ax[1].set_title("Species vs PetalLengthCm")

ax[1].legend()



sns.kdeplot(setosa["SepalWidthCm"], label="setosa", shade=True, ax=ax[2])

sns.kdeplot(versicolor["SepalWidthCm"], label="versicolor", shade=True, ax=ax[2])

sns.kdeplot(virginica["SepalWidthCm"], label="virginica", shade=True, ax=ax[2])

ax[2].set_title("Species vs SepalWidthCm")

ax[2].legend()



sns.kdeplot(setosa["SepalLengthCm"], label="setosa", shade=True, ax=ax[3])

sns.kdeplot(versicolor["SepalLengthCm"], label="versicolor", shade=True, ax=ax[3])

sns.kdeplot(virginica["SepalLengthCm"], label="virginica", shade=True, ax=ax[3])

ax[3].set_title("Species vs SepalLengthCm")

ax[3].legend()



plt.show()
fig = plt.figure(figsize=(18,6))

ax1 = fig.add_subplot(141)

sns.swarmplot(x="Species", y="PetalWidthCm", data=data, ax=ax1)

ax2 = fig.add_subplot(142)

sns.swarmplot(x="Species", y="PetalLengthCm", data=data, ax=ax2)

ax3 = fig.add_subplot(143)

sns.swarmplot(x="Species", y="SepalWidthCm", data=data, ax=ax3)

ax4 = fig.add_subplot(144)

sns.swarmplot(x="Species", y="SepalLengthCm", data=data, ax=ax4)



plt.show()
sns.pairplot(data, x_vars=["SepalWidthCm", "PetalLengthCm", "PetalWidthCm"], y_vars="SepalLengthCm", kind="reg", height=3, aspect=1.2)

sns.pairplot(data, x_vars=["SepalWidthCm", "PetalLengthCm", "PetalWidthCm"], y_vars="SepalLengthCm", hue="Species", kind="scatter", height=3, aspect=1.2)

plt.show()
sns.pairplot(data, x_vars=["SepalLengthCm", "PetalLengthCm", "PetalWidthCm"], y_vars="SepalWidthCm", kind="reg", height=3, aspect=1.2)

sns.pairplot(data, x_vars=["SepalLengthCm", "PetalLengthCm", "PetalWidthCm"], y_vars="SepalWidthCm", hue="Species", kind="scatter", height=3, aspect=1.2)

plt.show()
sns.pairplot(data, x_vars=["SepalLengthCm", "SepalWidthCm", "PetalWidthCm"], y_vars="PetalLengthCm", kind="reg", height=3, aspect=1.2)

sns.pairplot(data, x_vars=["SepalLengthCm", "SepalWidthCm", "PetalWidthCm"], y_vars="PetalLengthCm", hue="Species", kind="scatter", height=3, aspect=1.2)

plt.show()
sns.pairplot(data, x_vars=["SepalLengthCm", "SepalWidthCm", "PetalLengthCm"], y_vars="PetalWidthCm", kind="reg", height=3, aspect=1.2)

sns.pairplot(data, x_vars=["SepalLengthCm", "SepalWidthCm", "PetalLengthCm"], y_vars="PetalWidthCm", hue="Species", kind="scatter", height=3, aspect=1.2)

plt.show()
sns.heatmap(data.corr(), annot=True)

plt.show()
X = data.iloc[:, :-1]    # Here we select all the columns except the last one.

y = data.iloc[:, -1]    # Here, selecting the last column as the output column.

print('Shape of X : ', X.shape)

print('Shape of y : ', y.shape)
from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier



models = []

models.append(("LR", LogisticRegression()))

models.append(("KNN", KNeighborsClassifier()))

models.append(("SVC_linear", SVC(kernel="linear", gamma='auto')))

models.append(("SVC_rbf", SVC(kernel="rbf", gamma='auto')))

models.append(("DT", DecisionTreeClassifier()))
from sklearn.model_selection import KFold



seed = 22

scoring = 'accuracy'



kfold = KFold(n_splits=10, shuffle=True, random_state=seed)
from sklearn.model_selection import cross_validate



results = []

model_name = []



for name, model in models:

    scores = cross_validate(model, X, y, cv=kfold, scoring=scoring)

    test_score = scores["test_score"]

    results.append(test_score.mean())

    model_name.append(name)

    msg = "{:^12s}: {:.3f} ±({:.2f})".format(name, test_score.mean(), test_score.std())

    print(msg)
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=seed)  #Splitting the data into train and test set.

print('No. of training examples : ', X_train.shape[0])

print('No. of test examples : ', X_test.shape[0])
from sklearn.model_selection import GridSearchCV



C = np.arange(0.1, 2, 0.1)  # We will try different values of C between 0.1 and 1.9

kernel = ["linear", "rbf"]

param_grid = {"C":C, "kernel":kernel}



gs = GridSearchCV(estimator=SVC(), param_grid=param_grid, cv=kfold, verbose=True)

gs.fit(X_train, y_train)



print("Best Score: ", gs.best_score_)

print(gs.best_estimator_)
from sklearn.metrics import classification_report



model = gs.best_estimator_



print(" GridSearchCV best estimator ".center(50, "="))



print("Training set classification report".center(50, "_"))

print(classification_report(y_train, model.predict(X_train)))



print("Test set classification report".center(50, "_"))

print(classification_report(y_test, model.predict(X_test)))
# Storing the value of C and kernel to use in other SVC() models when trying something different.



C = gs.best_estimator_.C

kernel = gs.best_estimator_.kernel
data_copy = data.copy()  # Making a copy of the dataframe

data_copy["pwidth_plength"] = data_copy["PetalWidthCm"] * data_copy["PetalLengthCm"]  # Adding a new feature to the dataframe as described above

data_copy.drop(columns=["PetalWidthCm", "PetalLengthCm"], inplace=True)  # Droping both the columns

data_copy.head()
y_copy = data_copy["Species"]

X_copy = data_copy.drop(columns="Species")

X_copy_train, X_copy_test, y_copy_train, y_copy_test = train_test_split(X_copy, y_copy, test_size=0.2, random_state=seed)
svc = SVC(C=C, kernel=kernel)

svc.fit(X_copy_train, y_copy_train)



print(" Combined PetalWidthCm and PetalLengthCm ".center(50, "="))



print("Training set classification report".center(50, "_"))

print(classification_report(y_copy_train, svc.predict(X_copy_train)))



print("Test set classification report".center(50, "_"))

print(classification_report(y_copy_test, svc.predict(X_copy_test)))
sns.boxplot(data["SepalWidthCm"])

plt.show()
Q1, Q3 = data["SepalWidthCm"].quantile(q=[0.25, 0.75])  # We use the pandas.Series.quantile function to get the value of Q1 and Q3

IQR = Q3 - Q1  # Calculating the Inter Quartile range(IQR)



data_wo_outliers = data_copy[(data_copy["SepalWidthCm"] > (Q1 - IQR*1.5)) & (data_copy["SepalWidthCm"] < (Q3 + IQR*1.5))]

print(data_wo_outliers.shape)

print("{} samples were outliers".format(150-data_wo_outliers.shape[0]))
y_wo_outliers = data_wo_outliers["Species"]

X_wo_outliers = data_wo_outliers.drop(columns="Species")



X_copy_train, X_copy_test, y_copy_train, y_copy_test = train_test_split(X_wo_outliers, y_wo_outliers, test_size=0.2, random_state=seed)
svc = SVC(C=C, kernel=kernel)

svc.fit(X_copy_train, y_copy_train)



print(" Removed Outliers ".center(50, "="))



print("Training set classification report".center(50, "_"))

print(classification_report(y_copy_train, svc.predict(X_copy_train)))



print("Test set classification report".center(50, "_"))

print(classification_report(y_copy_test, svc.predict(X_copy_test)))