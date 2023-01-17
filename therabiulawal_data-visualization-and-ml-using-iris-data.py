from IPython.display import Image

Image(url='https://www.tensorflow.org/images/iris_three_species.jpg')
import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score

%matplotlib inline

names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']

df = pd.read_csv("../input/Iris.csv") 

df.drop("Id", axis=1, inplace=True)

df.columns = names
df.head(5)
print(df)
df.describe()
df.info()
df.keys()
df.shape
# Let's count example of each species in dataset. You'll see 3 species ['versicolor', 'virginica', 'setosa'] 

# and 50 examples for each.

df['species'].value_counts()
plt.figure(figsize=(14,8))

ax = df[df.species=='Iris-setosa'].plot.scatter(x='sepal_length', y='sepal_width', label='Setosa')

df[df.species=='Iris-versicolor'].plot.scatter(x='sepal_length', y='sepal_width', color='red', label='Versicolor', ax=ax)

df[df.species=='Iris-virginica'].plot.scatter(x='sepal_length', y='sepal_width', color='orange', label='Virginica', ax=ax)

ax.set_xlabel("Sepal Length")

ax.set_ylabel("Sepal Width")

ax.set_title("Relationship between Sepal Length and Width")

plt.show()
df.hist(edgecolor='black', linewidth=1)

plt.show()
df.plot(kind = "density", figsize=(10,8))


sns.jointplot(x = 'sepal_length', y = 'sepal_width', data = df)
#seaborn scatterplot by species on sepal_length vs sepal_width

g = sns.FacetGrid(df, hue='species', size=5) 

g = g.map(plt.scatter, 'sepal_length', 'sepal_width').add_legend()
# seaborn scatterplot by species on petal_length vs petal_width. Remember? We did it first using Pandas plot.

g = sns.FacetGrid(df, hue='species', size=5) 

g = g.map(plt.scatter, 'petal_length', 'petal_width').add_legend()
# We did some seaborn plotting. Let's try another cool pandas plotting. Yes, BOXPLOT! 

# It will create boxes represeting data values



# Seaborn boxplot for on each features split out by specis. Try this plot again just changing figsize.

df.boxplot(by = 'species', figsize = (16,8))
# The violinplot shows density of the length and width in the species

# Denser regions of the data are fatter, and sparser thiner in a violin plot

plt.figure(figsize=(14,10))

plt.subplot(2,2,1)

sns.violinplot(x='species', y='sepal_length', data=df, size=5)

plt.subplot(2,2,2)

sns.violinplot(x='species', y='sepal_width', data=df, size=5)

plt.subplot(2,2,3)

sns.violinplot(x='species', y='petal_length', data=df, size=5)

plt.subplot(2,2,4)

sns.violinplot(x='species', y='petal_width', data=df, size=5)
sns.pairplot(data = df, hue = 'species', size = 3)
# One more sophisticated technique pandas has available is called Andrews Curves

# Andrews Curves involve using attributes of samples as coefficients for Fourier series

# and then plotting these

from pandas.tools.plotting import andrews_curves

andrews_curves(df, 'species')
# Another multivariate visualization technique pandas has is parallel_coordinates

# Parallel coordinates plots each feature on a separate column & then draws lines

# connecting the features for each data sample

from pandas.tools.plotting import parallel_coordinates

parallel_coordinates(df, 'species')
# A final multivariate visualization technique pandas has is radviz

# Which puts each feature as a point on a 2D plane, and then simulates

# having each sample attached to those points through a spring weighted

# by the relative value for that feature

from pandas.tools.plotting import radviz

radviz(df, 'species')
plt.figure(figsize=(7,4)) 

#draws  heatmap with input as the correlation matrix calculted by(df.corr())

sns.heatmap(df.corr(),cbar = True, square = True, annot=True, fmt='.2f',annot_kws={'size': 15},cmap='Dark2') 

plt.show()
#training and test data spliting

X = df.iloc[:, 0:4]

y = df.iloc[:, -1]



#buliding model and importing algorithm

from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)



#Check shape of train and test dataset

print('There are {} samples in the training set and {} samples in the test set'.format(X_train.shape[0],X_test.shape[0]))
X_train.head()
y_train.head()
from sklearn.preprocessing import StandardScaler



scaler = StandardScaler()



scaler.fit(X_train)

X_train = scaler.transform(X_train)

X_test = scaler.transform(X_test)

print('Standardized features\n')

print(str(X_train[:4]))
from sklearn.tree import DecisionTreeClassifier



#Instantiate the model

model = DecisionTreeClassifier()



#Fitting the model

model.fit(X_train, y_train)



#Predict on test data

y_pred = model.predict(X_test)



#Evaluate accuracy

print('The accuracy of the Decision Tree is', accuracy_score(y_pred, y_test))

from sklearn.linear_model import LogisticRegression

model = LogisticRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print('The accuracy of the Logistic Regression is', accuracy_score(y_pred, y_test))
from sklearn.neighbors import KNeighborsClassifier



knn = KNeighborsClassifier(n_neighbors=8)

knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

from sklearn.metrics import accuracy_score

print ('The accuracy of the KNN is', accuracy_score(y_pred, y_test))
result = []

x = np.arange(1,11)

for i in list(range(1,11)):

    model = KNeighborsClassifier(n_neighbors=i)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print('The accuracy of the KNN {} is {}'.format( i, accuracy_score(y_pred, y_test)))

    result.append(accuracy_score(y_pred, y_test))

plt.figure(figsize = (10,6))

plt.plot(x, result)

plt.xticks(x)

plt.title('Plotting accuracy of K =  1 to 10')

plt.xlabel('K')

plt.ylabel('Accuray')
from sklearn.svm import LinearSVC



model = LinearSVC(C=10)

model.fit(X_train,y_train)

y_pred = model.predict(X_test)

print('The accuracy of the Support Vector Machine is', accuracy_score(y_pred, y_test))
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(max_depth=2)

model.fit(X_train,y_train)

y_pred = model.predict(X_test)

print('The accuracy of the Random Forest is', accuracy_score(y_pred, y_test))