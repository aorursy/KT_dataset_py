#Import pandas for Dataset Manupilation and matplotlib and seaborn for Visualization

import pandas as pd  

import matplotlib.pyplot as plt 

import seaborn as sns 

import numpy as np

from matplotlib.colors import ListedColormap
#Import functions for Model, Dataset Splitting and Evaluation

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder

from sklearn.neighbors import KNeighborsClassifier

from sklearn import metrics
df=pd.read_csv("../input/iris/Iris.csv") #Read the Dataset CSV File to a dataframe object
df.shape # To view the shape of our dataset (150 rows and 6 columns)
df.head()
df.info() #Information about the Dataframe
df.describe() # Further Statistical Information about the dataset
df.plot(kind="scatter", x="SepalLengthCm", y="SepalWidthCm") # Plot the data points (x-Sepal Length and y-Sepal Width)

plt.show()
# Plot Sepal Length vs Sepal Width Species Wise

sns.FacetGrid(df, hue="Species", height=5).map(plt.scatter, "SepalLengthCm", "SepalWidthCm").add_legend() 

plt.show()
# Display distribution of data points of each class in each attribute

plt.figure(figsize=(15,10))

plt.subplot(2,2,1)

sns.stripplot(x = 'Species', y = 'SepalLengthCm', data = df, jitter = True)

plt.subplot(2,2,2)

sns.stripplot(x = 'Species', y = 'SepalWidthCm', data = df, jitter = True)

plt.subplot(2,2,3)

sns.stripplot(x = 'Species', y = 'PetalLengthCm', data = df, jitter = True)

plt.subplot(2,2,4)

sns.stripplot(x = 'Species', y = 'PetalWidthCm', data = df, jitter = True)
#Drop the Id Column and fit Label Encoder on the Species column to encode the categorical values 

df = df.drop(['Id'], axis = 1)

Encoder=LabelEncoder()

df['Species']=Encoder.fit_transform(df['Species'])
sns.pairplot(data=df,kind='scatter',diag_kind='kde') #Shows relationships among all pairs of features
# This is a box plot of the data distribution of each class for each feature. The boxplot is used to identify outliers as well as data dsitribution

# Bottom black horizontal line of blue box plot is minimum value

# First black horizontal line of rectangle shape of blue box plot is First quartile or 25%

# Second black horizontal line of rectangle shape of blue box plot is Second quartile or 50% or median.

# Third black horizontal line of rectangle shape of blue box plot is third quartile or 75%

# Top black horizontal line of rectangle shape of blue box plot is maximum value.

# Small diamond shape of blue box plot is outlier data or erroneous data.

plt.figure(figsize=(15,10))

plt.subplot(2,2,1)

sns.boxplot(x = 'Species', y = 'SepalLengthCm', data = df)

sns.stripplot(x = 'Species', y = 'SepalLengthCm', data = df,jitter=True)

plt.subplot(2,2,2)

sns.boxplot(x = 'Species', y = 'SepalWidthCm', data = df)

sns.stripplot(x = 'Species', y = 'SepalWidthCm', data = df,jitter=True)

plt.subplot(2,2,3)

sns.boxplot(x = 'Species', y = 'PetalLengthCm', data = df)

sns.stripplot(x = 'Species', y = 'PetalLengthCm', data = df,jitter=True)

plt.subplot(2,2,4)

sns.boxplot(x = 'Species', y = 'PetalWidthCm', data = df)

sns.stripplot(x = 'Species', y = 'PetalWidthCm', data = df,jitter=True)
#A combination of the box plot and strip plot can be represented as a violin plot

# The white dot in the middle is the median value and the thick black bar in the centre represents the interquartile range. 

# The thin black line extended from it represents the upper (max) and lower (min) adjacent values in the data.

plt.figure(figsize=(15,10))

plt.subplot(2,2,1)

sns.violinplot(x = 'Species', y = 'SepalLengthCm', data = df, size=6)

plt.subplot(2,2,2)

sns.violinplot(x = 'Species', y = 'SepalWidthCm', data = df, size=6)

plt.subplot(2,2,3)

sns.violinplot(x = 'Species', y = 'PetalLengthCm', data = df, size=6)

plt.subplot(2,2,4)

sns.violinplot(x = 'Species', y = 'PetalWidthCm', data = df, size=6)
# Create X attributes and Y labels from dataframe object

X = df[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']].values

y = df['Species'].values
corr=df.corr() #Correlation Matrix
# Display the correlation matrix using a heatmap

ax = sns.heatmap(

    corr, 

    vmin=-1, vmax=1, center=0,

    cmap=sns.diverging_palette(20, 220, n=200),

    square=True

)

ax.set_xticklabels(

    ax.get_xticklabels(),

    rotation=45,

    horizontalalignment='right'

);
# Exactly –1. A perfect downhill (negative) linear relationship



# –0.70. A strong downhill (negative) linear relationship



# –0.50. A moderate downhill (negative) relationship



# –0.25. A weak downhill (negative) linear relationship



# 0. No linear relationship





# +0.25. A weak uphill (positive) linear relationship



# +0.50. A moderate uphill (positive) relationship



# +0.70. A strong uphill (positive) linear relationship



# Exactly +1. A perfect uphill (positive) linear relationship
# Create the training and test sets using 0.2 as test size (i.e 80% of data for training rest 20% for model testing)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
#For choosing the best k value for our KNearestNeighbor Classifier, Let us run the model for different k values (i.e Between 1 and 26) and plot the scores

kvals=range(1,26)

scores={}

scores_list=[]

for k in kvals:

    knn=KNeighborsClassifier(n_neighbors=k)

    knn.fit(X_train,y_train)

    y_pred=knn.predict(X_test)

    scores[k]=metrics.accuracy_score(y_test,y_pred)

    scores_list.append(metrics.accuracy_score(y_test,y_pred))
plt.plot(kvals,scores_list)

plt.xlabel("Value of K")

plt.ylabel("Testing Accuracy")
#Here we can see that there were a few initial peak and drops in the accuracy, but after a K value of 5 it increased to 100% and stayed
# Set the final model's K value to 6 and fit the training data and run predictions on test

Finalknn=KNeighborsClassifier(n_neighbors=6)

Finalknn.fit(X_train,y_train)

y_pred=knn.predict(X_test)
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred}) #View first 25 predictions against actual values

df1 = df.head(25)

print(df1)
# Get the confusion Matrix of the Model

cnf_matrix = metrics.confusion_matrix(y_test, y_pred)

cnf_matrix
# Plot the Confusion Matrix as a HeatMap

class_names=[0,1,2] # Name  of classes

fig, ax = plt.subplots()

tick_marks = np.arange(len(class_names))

plt.xticks(tick_marks, class_names)

plt.yticks(tick_marks, class_names)

# create heatmap

sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')

ax.xaxis.set_label_position("top")

plt.tight_layout()

plt.title('Confusion matrix', y=1.1)

plt.ylabel('Actual label')

plt.xlabel('Predicted label')
print(metrics.classification_report(y, Finalknn.predict(X))) # Displays a comprehensive Report of the KNN Model fitted on the whole Dataset
# Now let us plot the classifier boundaries as plot

# Run the model for first two features alone

plotknn=KNeighborsClassifier(n_neighbors=6)

plotknn.fit(X[:,:2],y)

h=0.2

#Assign the boundary values

x_min, x_max = X[:,0].min() - .5, X[:,0].max() + .5

y_min, y_max = X[:,1].min() - .5, X[:,1].max() + .5
# Create the grid

xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
# Run the model 

Z = plotknn.predict(np.c_[xx.ravel(), yy.ravel()])
#Set datapoints color and Boundary region colors and plot

cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA','#00AAFF'])

cmap_bold = ListedColormap(['#FF0000', '#00FF00','#006699'])

Z = Z.reshape(xx.shape)

plt.figure(1, figsize=(4, 3))

plt.set_cmap(plt.cm.Paired)

plt.pcolormesh(xx, yy, Z,cmap=cmap_light)



plt.scatter(X[:,0], X[:,1],c=y,cmap=cmap_bold )

plt.xlabel('Sepal length')

plt.ylabel('Sepal width')



plt.xlim(xx.min(), xx.max())

plt.ylim(yy.min(), yy.max())

plt.xticks(())

plt.yticks(())



plt.show()