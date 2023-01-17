#Importing required packages.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
#Loading dataset
df = pd.read_csv('../input/red-wine-quality-cortez-et-al-2009/winequality-red.csv')
#Let's check how the data is distributed
df.head()
#df.tail()
#df.describe()
#df.shape
#df.columns
#Information about the data columns
df.info()
#Checking Null values on the dataset
df.isnull().sum()
#Plotting Histograms
sns.distplot(df['pH'],kde=False,hist=True,bins=11,hist_kws=dict(edgecolor="k", linewidth=1))
plt.title("Histogram 1")
plt.ylabel("Frequency")
plt.show()
sns.distplot(df['fixed acidity'],kde=False,hist=True,bins=11,hist_kws=dict(edgecolor="k", linewidth=1))
plt.title("Histogram 2")
plt.ylabel("Frequency")
plt.show()
sns.distplot(df['alcohol'],kde=False,hist=True,bins=11,hist_kws=dict(edgecolor="k", linewidth=1))
plt.title("Histogram 3")
plt.ylabel("Frequency")
plt.show()
sns.distplot(df['free sulfur dioxide'],kde=False,hist=True,bins=11,hist_kws=dict(edgecolor="k", linewidth=1))
plt.title("Histogram 4")
plt.ylabel("Frequency")
plt.show()
sns.distplot(df['density'],kde=False,hist=True,bins=11,hist_kws=dict(edgecolor="k", linewidth=1))
plt.title("Histogram 5")
plt.ylabel("Frequency")
plt.show()
# for i in df:
#     plt.figure()
#     df.boxplot([i])
#     plt.show()
# Plotting Barplots
sns.boxplot(y='fixed acidity', data = df)
#sns.boxplot('fixed acidity', data = df)
plt.title("Boxplot 1")
plt.show()
sns.boxplot(y='volatile acidity', data = df)
#sns.boxplot('volatile acidity', data = df)
plt.title("Boxplot 2")
plt.show()
sns.boxplot(y='citric acid', data = df)
#sns.boxplot('citric acid', data = df)
plt.title("Boxplot 3")
plt.show()
sns.boxplot(y='residual sugar', data = df)
#sns.boxplot('residual sugar', data = df)
plt.title("Boxplot 4")
plt.show()
sns.boxplot(y='chlorides', data = df)
#sns.boxplot('chlorides', data = df)
plt.title("Boxplot 5")
plt.show()
sns.boxplot(y='free sulfur dioxide', data = df)
#sns.boxplot('free sulfur dioxide', data = df)
plt.title("Boxplot 6")
plt.show()
sns.boxplot(y='total sulfur dioxide', data = df)
# sns.boxplot('total sulfur dioxide', data = df)
plt.title("Boxplot 7")
plt.show()
sns.boxplot(y='density', data = df)
#sns.boxplot('density', data = df)
plt.title("Boxplot 8")
plt.show()
sns.boxplot(y='pH', data = df)
#sns.boxplot('pH', data = df)
plt.title("Boxplot 9")
plt.show()
sns.boxplot(y='sulphates', data = df)
#sns.boxplot('sulphates', data = df)
plt.title("Boxplot 10")
plt.show()
sns.boxplot(y='alcohol', data = df)
#sns.boxplot('alcohol', data = df)
plt.title("Boxplot 11")
plt.show()
sns.boxplot(y='quality',data = df)
#sns.boxplot('quality',data = df)
plt.title("Boxplot 12")
plt.show()
# Plotting Scatter Plots
sns.scatterplot(x='fixed acidity', y='volatile acidity', data=df)
plt.title("Scatter Plot 1")
plt.show()
sns.scatterplot(x='citric acid', y='residual sugar', data=df)
plt.title("Scatter Plot 2")
plt.show()
sns.scatterplot(x='chlorides', y='free sulfur dioxide', data=df)
plt.title("Scatter Plot 3")
plt.show()
sns.scatterplot(x='total sulfur dioxide', y='density', data=df)
plt.title("Scatter Plot 4")
plt.show()
sns.scatterplot(x='pH', y='sulphates', data=df)
plt.title("Scatter Plot 5")
plt.show()
sns.scatterplot(x='alcohol', y='quality', data=df)
plt.title("Scatter Plot 6")
plt.show()
# Create arrays for the features and the response variable
y = df["quality"].values
X = df.drop(["quality"],axis=1).values
# Train and Test splitting of data 
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.1,random_state=38)
# Import KNeighborsClassifier from sklearn.neighbors
from sklearn.neighbors import KNeighborsClassifier 

# Create a k-NN classifier with 7 neighbors: knn
knn=KNeighborsClassifier(n_neighbors=20)

# Fit the classifier to the data
knn.fit(X_train,y_train)
# Predict the labels for the training data X_test
pred = knn.predict(X_test)
#Compute accuracy on the testing set
from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,pred)
accuracy
#Displaying the accuracy of the Model
print('Accuracy : ',100*accuracy)
# Import classification_report,confusion_matrix from sklearn.metrics
from sklearn.metrics import classification_report,confusion_matrix

#Confusion matrix for the KNeighborsClassifier 
print(confusion_matrix(y_test,pred))
# Classification report for the KNeighborsClassifier 
print(classification_report(y_test,pred))