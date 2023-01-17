import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

df = pd.read_csv("../input/swiss-banknote-conterfeit-detection/banknotes.csv")
df.head()
df.describe()
df.info()
sns.heatmap(df.isnull())

plt.title("Missing values?", fontsize = 18)

plt.show()
# Pairwise relationships depending on counterfeit

sns.pairplot(df, hue = "conterfeit")

plt.show()
sns.heatmap(df.corr(), annot = True, cmap="RdBu")

plt.title("Pairwise correlation of the columns", fontsize = 18)

plt.show()
# Shuffle the dataset

df = df.reindex(np.random.permutation(df.index))



X = df.drop(columns = "conterfeit")

y = df["conterfeit"]



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)



from sklearn.preprocessing import StandardScaler

st = StandardScaler()

X_train = st.fit_transform(X_train)

from sklearn.linear_model import LogisticRegression

model = LogisticRegression()

model.fit(X_train,y_train)



pred = model.predict(st.transform(X_test))



from sklearn.metrics import classification_report, confusion_matrix, accuracy_score



class_report = classification_report(y_test, pred)

conf_matrix = confusion_matrix(y_test,pred)

acc = accuracy_score(y_test,pred)



print("Classification report:\n\n", class_report)

print("Confusion Matrix\n",conf_matrix)

print("\nAccuracy\n",acc)



results = []

results.append(("LogisticRegression",class_report, conf_matrix, acc))
from sklearn.ensemble import RandomForestClassifier



rfc = RandomForestClassifier()



rfc.fit(X_train, y_train)



pred = rfc.predict(st.transform(X_test))



class_report = classification_report(y_test, pred)

conf_matrix = confusion_matrix(y_test,pred)

acc = accuracy_score(y_test,pred)



print("Classification report:\n\n", class_report)

print("Confusion Matrix\n",conf_matrix)

print("\nAccuracy\n",acc)



results.append(("RandomForestClassifier",class_report, conf_matrix, acc))
from sklearn.tree import DecisionTreeClassifier



dtc = DecisionTreeClassifier()



dtc.fit(X_train, y_train)



pred = dtc.predict(st.transform(X_test))



class_report = classification_report(y_test, pred)

conf_matrix = confusion_matrix(y_test,pred)

acc = accuracy_score(y_test,pred)



print("Classification report:\n\n", class_report)

print("Confusion Matrix\n",conf_matrix)

print("\nAccuracy\n",acc)



results.append(("DecisionTreeClassifier",class_report, conf_matrix, acc))
import tensorflow.keras

from tensorflow.keras import Sequential

from tensorflow.keras.layers import Dense, Activation



model = Sequential()

model.add(Dense(6))

model.add(Dense(10))

model.add(Dense(10))

model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train,y_train.values, epochs = 50, verbose = 0)
pred = model.predict(st.transform(X_test))

pred = [int(round(t)) for t in pred.reshape(1,-1)[0]]



class_report = classification_report(y_test, pred)

conf_matrix = confusion_matrix(y_test,pred)

acc = accuracy_score(y_test,pred)



print("Classification report:\n\n", class_report)

print("Confusion Matrix\n",conf_matrix)

print("\nAccuracy\n",acc)



results.append(("Neural Network",class_report, conf_matrix, acc))
from sklearn.svm import SVC

svc = SVC()



svc.fit(X_train, y_train)



pred = svc.predict(st.transform(X_test))



class_report = classification_report(y_test, pred)

conf_matrix = confusion_matrix(y_test,pred)

acc = accuracy_score(y_test,pred)



print("Classification report:\n\n", class_report)

print("Confusion Matrix\n",conf_matrix)

print("\nAccuracy\n",acc)



results.append(("SVC",class_report, conf_matrix, acc))
from sklearn.decomposition import TruncatedSVD

svd = TruncatedSVD(n_components = 2, random_state = 0)



transf = svd.fit_transform(X)



plt.scatter(x = transf[:,0], y = transf[:,1])

plt.title("Dataset after transformation with SVD", fontsize = 18)

plt.show()
from sklearn.cluster import KMeans



km = KMeans(n_clusters = 2)

c = km.fit_predict(transf)



plt.scatter(x = transf[:,0], y = transf[:,1], c = c)

plt.title("Clustering with Kmeans after SVD", fontsize = 18)

plt.show()
plt.scatter(x = transf[:,0], y = transf[:,1], c = y)

plt.title("Original labels after SVD", fontsize = 18)

plt.show()
from sklearn.decomposition import PCA

pca = PCA(n_components = 2, random_state = 0)



transf = pca.fit_transform(X)



plt.scatter(x = transf[:,0], y = transf[:,1])

plt.title("Dataset after transformation with PCA", fontsize = 18)

plt.show()
km = KMeans(n_clusters = 2)

c = km.fit_predict(transf)



plt.scatter(x = transf[:,0], y = transf[:,1], c = c)

plt.title("Clustering with Kmeans after PCA", fontsize = 18)

plt.show()
plt.scatter(x = transf[:,0], y = transf[:,1], c = y)

plt.title("Original labels after PCA", fontsize = 18)

plt.show()
labels  = []

height = []

for i in range(len(results)):

    labels.append(results[i][0])

    height.append(results[i][-1])

    

plt.figure(figsize = (12,6))    

ax = sns.barplot(labels,height)

ax.set_xticklabels(labels, fontsize = 18, rotation = 90)

plt.title("Comparison of the models", fontsize = 18)

plt.ylabel("Prediction accuracy")

plt.show()