# Set-up libraries
import os
import pandas as pd
import seaborn as sns
sns.set()
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
# Check input data source
for dirname, _, filenames in os.walk('../input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
# Read-in data
df = pd.read_csv('../input/iris/Iris.csv')
# Look at some details
df.info()
# Look at some records
df.head()
# Check for missing values
df.isna().sum()
# Check for duplicate values
df.duplicated().sum()
# Look at breakdown of label
df['Species'].value_counts()
sns.countplot(df['Species'])
# Explore data visually
sns.pairplot(df, hue='Species')
# Summarise
df.describe()
# Get the features for input
X = df.drop('Species', axis=1)
# Build and fit models
wcss_scores = []
iterations = list(range(1,10))

for k in iterations:
    model = KMeans(n_clusters=k)
    model.fit(X)
    wcss_scores.append(model.inertia_)
# Plot performances
plt.figure(figsize=(12,6))
sns.lineplot(x=iterations, y=wcss_scores)
# Compare ground truth and cluster labels visually

plt.figure(figsize=(27,27))

plt.subplot(3,2,1)
plt.title('Ground truth: Petal',fontsize=22)
plt.xlabel('Petal Width')
plt.xlabel('Petal Length')
plt.scatter(df.PetalLengthCm[df.Species == "Iris-setosa"],
            df.PetalWidthCm[df.Species == "Iris-setosa"])
plt.scatter(df.PetalLengthCm[df.Species == "Iris-versicolor"],
            df.PetalWidthCm[df.Species == "Iris-versicolor"])
plt.scatter(df.PetalLengthCm[df.Species == "Iris-virginica"],
            df.PetalWidthCm[df.Species == "Iris-virginica"])

plt.subplot(3,2,2)
plt.title('Ground truth: Sepal',fontsize=22)
plt.xlabel('Sepal Width')
plt.xlabel('Sepal Length')
plt.scatter(df.SepalLengthCm[df.Species == "Iris-setosa"],
            df.SepalWidthCm[df.Species == "Iris-setosa"])
plt.scatter(df.SepalLengthCm[df.Species == "Iris-versicolor"],
            df.SepalWidthCm[df.Species == "Iris-versicolor"])
plt.scatter(df.SepalLengthCm[df.Species == "Iris-virginica"],
            df.SepalWidthCm[df.Species == "Iris-virginica"])

plt.subplot(3,2,3)
plt.title('K = 3: Petal',fontsize=22)
plt.xlabel('Petal Width')
plt.xlabel('Petal Length')
model = KMeans(n_clusters=3)
X['labels'] = model.fit_predict(X)
plt.scatter(X.PetalLengthCm[X.labels == 0], X.PetalWidthCm[X.labels == 0])
plt.scatter(X.PetalLengthCm[X.labels == 1], X.PetalWidthCm[X.labels == 1])
plt.scatter(X.PetalLengthCm[X.labels == 2], X.PetalWidthCm[X.labels == 2])

plt.subplot(3,2,4)
plt.title('K = 3: Sepal',fontsize=22)
plt.xlabel('Sepal Width')
plt.xlabel('Sepal Length')
model = KMeans(n_clusters=3)
X['labels'] = model.fit_predict(X)
plt.scatter(X.SepalLengthCm[X.labels == 0], X.SepalWidthCm[X.labels == 0])
plt.scatter(X.SepalLengthCm[X.labels == 1], X.SepalWidthCm[X.labels == 1])
plt.scatter(X.SepalLengthCm[X.labels == 2], X.SepalWidthCm[X.labels == 2])

plt.subplot(3,2,5)
plt.title('K = 2: Petal',fontsize=22)
plt.xlabel('Petal Width')
plt.xlabel('Petal Length')
model = KMeans(n_clusters=2)
X['labels'] = model.fit_predict(X)
plt.scatter(X.PetalLengthCm[X.labels == 0], X.PetalWidthCm[X.labels == 0])
plt.scatter(X.PetalLengthCm[X.labels == 1], X.PetalWidthCm[X.labels == 1])

plt.subplot(3,2,6)
plt.title('K = 2: Sepal',fontsize=22)
plt.xlabel('Sepal Width')
plt.xlabel('Sepal Length')
model = KMeans(n_clusters=2)
X['labels'] = model.fit_predict(X)
plt.scatter(X.SepalLengthCm[X.labels == 0], X.SepalWidthCm[X.labels == 0])
plt.scatter(X.SepalLengthCm[X.labels == 1], X.SepalWidthCm[X.labels == 1])
