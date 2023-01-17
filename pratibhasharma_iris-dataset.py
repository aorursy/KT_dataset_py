#Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import scipy
import sklearn
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import seaborn as sns
#Loading in the data
df = pd.read_csv("../input/Iris.csv")
df.shape
df.info()
df.Species.unique()
df.isnull().sum()
df.groupby('Species').count()
df.where(df['Species']=="Iris-setosa").head()
df[df['SepalLengthCm']>7]
#Relationships btwn two quantitative variables
sns.FacetGrid(df, hue="Species", size=7) \
.map(plt.scatter, "SepalLengthCm", "PetalLengthCm") \
.add_legend()
plt.show()
sns.FacetGrid(df, hue="Species", size=7) \
.map(plt.scatter, "SepalLengthCm", "SepalWidthCm") \
.add_legend()
plt.show()
df.plot(kind='box',subplots='True',layout=(2,3),figsize=(7,7))
sns.boxplot(x='Species',y='PetalLengthCm',data=df)
plt.show()
#Insert jitter=True so that the data points remain scattered and not piled into a verticle line.
#Assign ax to each axis, so that each plot is ontop of the previous axis. 
ax = sns.boxplot(x='Species',y='PetalLengthCm',data=df)
ax = sns.stripplot(x='Species',y='PetalLengthCm',data = df, jitter=True)
plt.show()
#Make the scatter points more visible
ax = sns.boxplot(x='Species',y='PetalLengthCm',data=df)
ax = sns.stripplot(x='Species',y='PetalLengthCm',data = df, jitter=True)
boxtwo = ax.artists[2]
boxtwo.set_facecolor('pink')
boxthree = ax.artists[1]
boxthree.set_facecolor('red')
ax.artists[0].set_facecolor('green')
plt.show()
df.hist(figsize=(20,20))
#Multivariate Plot
pd.plotting.scatter_matrix(df,figsize=(10,10))
plt.figure()
sns.violinplot(data=df,x="Species",y="PetalLengthCm")

#Using seaborn's pairplot to look at bivariate relations between each pair of variables
sns.pairplot(data=df, hue="Species")
#Correlation Heatmap
plt.figure(figsize=(10,10))
sns.heatmap(df.corr(),annot=True)

#Multivariate data visualization using RadViz from pandas
#Based on a spring tension minimization algorithm 
from pandas.plotting import radviz
plt.figure(figsize=(7,7))
radviz(df.drop("Id",axis=1),"Species")

# Seperating the data into dependent and independent variables
X = df.iloc[:, :-1].values # All rows and all columns except the last ## Feature variables
y = df.iloc[:, -1].values # Only the last column ## Target variable
# Splitting the data into training and test data
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=0)
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=8)
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
print(classification_report(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))
print("accuracy is",accuracy_score(y_test,y_pred))
from sklearn.neighbors import RadiusNeighborsClassifier
model = RadiusNeighborsClassifier(radius=8)
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
print(classification_report(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))
print("accuracy is",accuracy_score(y_test,y_pred))
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print('accuracy is',accuracy_score(y_pred,y_test))
