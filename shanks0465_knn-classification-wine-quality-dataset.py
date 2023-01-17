#Import pandas for dataset manupilation and matplotlib and seaborn for visualization

import pandas as pd  

import matplotlib.pyplot as plt 

import seaborn as sns 

import numpy as np
#Import functions for Model, Dataset Splitting and Evaluation

from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier

from sklearn import metrics
#Read the CSV File into a DataFrame Object 

df=pd.read_csv('/kaggle/input/red-wine-quality-cortez-et-al-2009/winequality-red.csv') #Read the Dataset CSV File to a dataframe object
df.shape # To view the shape of our dataset (1599 rows and 12 columns)
df.head() #Display first 5 rows of the Dataset
df.tail() #Display last 5 rows of the Dataset
#Display detailed information such as count, max,min,etc. 

df.describe()
df.isnull().any() #To check if any column has null values. Column returns False if no null values
# Create X attributes and Y labels from dataframe object

X = df[['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates','alcohol']].values

y = df['quality'].values
#Get Correlation Matrix of the dataset attributes

corr=df.corr()
#Display the correlation matrix as a heatmap

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

print("""

# Exactly –1. A perfect downhill (negative) linear relationship



# –0.70. A strong downhill (negative) linear relationship



# –0.50. A moderate downhill (negative) relationship



# –0.25. A weak downhill (negative) linear relationship



# 0. No linear relationship





# +0.25. A weak uphill (positive) linear relationship



# +0.50. A moderate uphill (positive) relationship



# +0.70. A strong uphill (positive) linear relationship



# Exactly +1. A perfect uphill (positive) linear relationship

""")
df.plot(kind="scatter", x="alcohol", y="volatile acidity") # Plot the data points (x-Sepal Length and y-Sepal Width)

plt.show()
# Plot Sepal Length vs Sepal Width Species Wise

sns.FacetGrid(df, hue="quality", height=5).map(plt.scatter, "alcohol", "volatile acidity").add_legend() 

plt.show()
# Display distribution of data points of each class in each attribute

plt.figure(figsize=(15,10))

plt.subplot(3,3,1)

sns.stripplot(x = 'quality', y = 'fixed acidity', data = df, jitter = True)

plt.subplot(3,3,2)

sns.stripplot(x = 'quality', y = 'volatile acidity', data = df, jitter = True)

plt.subplot(3,3,3)

sns.stripplot(x = 'quality', y = 'citric acid', data = df, jitter = True)

plt.subplot(3,3,4)

sns.stripplot(x = 'quality', y = 'residual sugar', data = df, jitter = True)

plt.subplot(3,3,5)

sns.stripplot(x = 'quality', y = 'free sulfur dioxide', data = df, jitter = True)

plt.subplot(3,3,6)

sns.stripplot(x = 'quality', y = 'total sulfur dioxide', data = df, jitter = True)

plt.subplot(3,3,7)

sns.stripplot(x = 'quality', y = 'density', data = df, jitter = True)

plt.subplot(3,3,8)

sns.stripplot(x = 'quality', y = 'sulphates', data = df, jitter = True)

plt.subplot(3,3,9)

sns.stripplot(x = 'quality', y = 'alcohol', data = df, jitter = True)
sns.pairplot(data=df,kind='scatter',diag_kind='kde') #Shows relationships among all pairs of features
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
#Here we can see that there were a few initial peak and drops in the accuracy, but after a K value of 21 it baselined and saw a slight slope drop 
# Set the final model's K value to 22 and fit the training data and run predictions on test

Finalknn=KNeighborsClassifier(n_neighbors=22)

Finalknn.fit(X_train,y_train)

y_pred=knn.predict(X_test)
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred}) #View first 25 predictions against actual values

df1 = df.head(25)

print(df1)
# Get the confusion Matrix of the Model

cnf_matrix = metrics.confusion_matrix(y_test, y_pred)

cnf_matrix
# Plot the Confusion Matrix as a HeatMap

class_names=[3,4,5,6,7,8] # Name  of classes

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
print(metrics.classification_report(y, Finalknn.predict(X))) # Displays a comprehensive Report of the KNN Model