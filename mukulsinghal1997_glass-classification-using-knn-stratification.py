import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import scipy.stats
import warnings

warnings.filterwarnings("ignore")
np.random.seed(43)
plt.style.use("ggplot")
df = pd.read_csv("../input/glass/glass.csv")
df.head()
df.describe().T
from pandas.plotting import parallel_coordinates
plt.figure(figsize=(10,8))

parallel_coordinates(df, "Type")
import missingno as msno
msno.bar(df)
sns.countplot(df["Type"])
print(df.groupby("Type").min())

print("------------------------------------------------------------------")

print(df.groupby("Type").max())
col = df.columns
fig, ax = plt.subplots(3,3, figsize=(10,10))

axes_all = [axes for axes_row in ax for axes in axes_row]

for i, c in enumerate(df[col]):

    if c  == "Type":

        break

    else:

        sns.boxplot(df[c], data = df, ax = axes_all[i])
#Detecting the no of outliers present in our data



def outlier_func(data, col):

    

    Q1 = df.quantile(q = 0.25, axis = 0)

    Q3 = df.quantile(q = 0.70, axis = 0)

    IQR = Q3-Q1

    

    min_val = Q1 - 1.5*IQR

    max_val = Q3 + 1.5*IQR

    

    df1 = df[df[col] <=  min_val[col]].shape[0]

    df2 = df[df[col] >=  max_val[col]].shape[0]

    

    print(f"There are {df1 + df2} total number of outliers in which {df1} datapoints are below or equals to the Q1 Deviation {Q1[col]} and {df2} are above or equal to the {Q3[col]}\n")

    print(f"The IQR of '{col}' is: {IQR[col]}")

    

    print(f"The Q1 Deviation of '{col}' is: {Q1[col]}")

    print(f"The Q3 Deviation of '{col}' is: {Q3[col]} \n")

    

    print("The min value is:", min(data[col]))

    print("The max value is:", max(data[col]), "\n")

    

    print("The skewness is: ",scipy.stats.skew(data[col]))

    print("The Kurtosis is: ",scipy.stats.kurtosis(data[col]))

    

    

    

    #Also returning the visual representation of the outlier

    

    plt.figure(figsize=(8,6))

    sns.distplot(data[col], color = 'g')

    plt.axvline(df[col].mean(), linestyle = '--', color = 'k')

    plt.axvline(df[col].median(), linestyle = '--', color = 'orange')

    

    plt.axvspan(xmin = Q1[col], xmax=data[col].min(), alpha = 0.15, color = 'r')

    plt.axvspan(xmin = Q3[col], xmax=data[col].max(), alpha = 0.15, color = 'r')

    

    plt.legend(["Mean", "Median","Outlier Bound"])
outlier_func(df, "RI")
df["Ba"].value_counts()
df.drop(labels = 'Ba', axis = 1, inplace = True) #Lets remove the 'Ba' from our datset as most of the values are 0
df
corrmat = df.corr()
plt.figure(figsize=(10,8))

sns.heatmap(corrmat, annot = True, cmap = 'Blues')
df.drop(labels = "RI", axis = 1, inplace = True)
plt.figure(figsize=(8,6))

sns.heatmap(df.corr(), annot = True, cmap = 'Blues')
from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.neighbors import KNeighborsClassifier
X = df.iloc[:, :7]
y = df.iloc[:, 7]
X_train, X_test, y_train, y_test = train_test_split(X, y , test_size = 0.2, random_state = 0)
print(X_train.shape)

print(X_test.shape)

print(y_train.shape)

print(y_test.shape)
test_score = []

train_score = []
for i in range(1,16):

    knn = KNeighborsClassifier(i)

    knn.fit(X_train, y_train)

    

    train_score.append(knn.score(X_train, y_train))

    test_score.append(knn.score(X_test, y_test))
plt.plot(range(1,16), train_score)

plt.plot(range(1,16), test_score)
#lets try to standardize the data and then try

#We will also on additional parameter to 

from sklearn.preprocessing import StandardScaler
se = StandardScaler()
X_col = X.columns
X_std = se.fit_transform(X)
X_std = pd.DataFrame(X_std, columns = X_col)
X_std.mean()
X_std.std()
X_train, X_test, y_train, y_test = train_test_split(X_std, y , test_size = 0.2, random_state = 0)
print(X_train.shape)

print(X_test.shape)

print(y_train.shape)

print(y_test.shape)
test_score = []

train_score = []
for i in range(1,16):

    knn = KNeighborsClassifier(i)

    knn.fit(X_train, y_train)

    

    train_score.append(knn.score(X_train, y_train))

    test_score.append(knn.score(X_test, y_test))
plt.plot(range(1,16), train_score)

plt.plot(range(1,16), test_score, linestyle = '--', marker = '*')
knn = KNeighborsClassifier(4)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
from sklearn.metrics import accuracy_score, classification_report
print(accuracy_score(y_test, y_pred) * 100)
sns.countplot(y_pred)
X_train, X_test, y_train, y_test = train_test_split(X_std, y , test_size = 0.2, random_state = 0, stratify = y)
print(X_train.shape)

print(X_test.shape)

print(y_train.shape)

print(y_test.shape)
neighbors = [x for x in range(1,16)]
cross_score = []

for k in neighbors:

    KNN = KNeighborsClassifier(n_neighbors = k)

    scores = cross_val_score(KNN, X, y, cv = 10, scoring = 'accuracy')

    cross_score.append(scores.mean())
MSE = [1-x for x in cross_score]
plt.plot(neighbors, MSE)

plt.xlabel("K Neighbors")

plt.ylabel("Error")

plt.show()
#So based on this, our best value for k is 3

KNN_Model = KNeighborsClassifier(n_neighbors = 3)

KNN_Model_Fit = KNN_Model.fit(X_train, y_train)

KNN_Model_Predict = KNN_Model_Fit.predict(X_test)
print(round(accuracy_score(y_test, KNN_Model_Predict) * 100),"%")
print(classification_report(y_test, KNN_Model_Predict))
sns.countplot(y_pred)
#Lets Try to Standardize the data after splitting the data and see if our accuracy improved or not

#By normalizing or standardizing the data after splitting, means we can avoid the issue of data leakage
X_train_n, X_test_n, y_train_n, y_test_n = train_test_split(X, y, test_size = 0.2, random_state =0, stratify=y)
print(X_train.shape)

print(X_test.shape)

print(y_train.shape)

print(y_test.shape)
#Now try to standardize the data

se_new = StandardScaler()
X_train_std = se_new.fit_transform(X_train_n)

X_test_std = se_new.fit_transform(X_test_n)
#So based on this, our best value for k is 3

KNN_Model = KNeighborsClassifier(n_neighbors = 3)

KNN_Model_Fit = KNN_Model.fit(X_train_std, y_train)

KNN_Model_Predict = KNN_Model_Fit.predict(X_test_std)
print(round(accuracy_score(y_test, KNN_Model_Predict) * 100),"%")
print(classification_report(y_test, KNN_Model_Predict))
sns.countplot(y_pred)