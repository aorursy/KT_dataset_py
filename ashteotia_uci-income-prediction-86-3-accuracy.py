import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

%matplotlib inline
print('Imported')
path = '../input/adult.csv'
data = pd.read_csv(path, na_values=['?']);
data.shape
data.info()
data.head(5)
#fill missing values
data['workclass'] = data['workclass'].fillna(data['workclass'].mode()[0])
data['occupation'] = data['occupation'].fillna(data['occupation'].mode()[0])
data['native.country'] = data['native.country'].fillna(data['native.country'].mode()[0])

data.head()
data.info()
data['income'].value_counts()
sns.set_style('whitegrid');
sns.pairplot(data, hue = 'income', size = 10)
plt.show()
sns.set_style('whitegrid');
sns.FacetGrid(data, hue = 'income', size = 7)\
   .map(plt.scatter, 'age', 'workclass')\
   .add_legend();
plt.title('Age vs Workclass');
plt.show();

sns.set_style('whitegrid');
sns.FacetGrid(data, hue = 'income', size = 7)\
   .map(plt.scatter, 'age', 'hours.per.week')\
   .add_legend();
plt.title('Age vs Hours_per_week');
plt.show();

sns.set_style('whitegrid');
sns.FacetGrid(data, hue = 'income', size = 7)\
   .map(plt.scatter, 'age', 'fnlwgt')\
   .add_legend();
plt.title('Age vs fnlwgt');
plt.show();

sns.set_style('whitegrid');
sns.FacetGrid(data, hue = 'income', size = 7)\
   .map(plt.scatter, 'age', 'education.num')\
   .add_legend();
plt.title('Age vs education.num');
plt.show();

sns.set_style('whitegrid');
sns.FacetGrid(data, hue = 'income', size = 7)\
   .map(plt.scatter, 'age', 'capital.gain')\
   .add_legend();
plt.title('Age vs capital.gain');
plt.show();

sns.set_style('whitegrid');
sns.FacetGrid(data, hue = 'income', size = 7)\
   .map(plt.scatter, 'age', 'capital.loss')\
   .add_legend();
plt.title('Age vs Capital Loss');
plt.show();

sns.set_style('whitegrid');
sns.FacetGrid(data, hue = 'income', size = 7)\
   .map(plt.scatter, 'capital.loss', 'capital.gain')\
   .add_legend();
plt.title('Capital Loss vs Capital Gain');
plt.show();

# We already know from data.info() that our dataset has TWO variables types: int64, Object
# So variable with datatype Object are categorical variables/fetaures
categorical_var = data.select_dtypes(include=['object']).columns
print(categorical_var)
print(len(categorical_var))
# Before tranforming the fetures we need to seprate them as: variables and target varibles
# here 'income' is our target variable we need to seperate from our main dataset beofore transformation
X = data.iloc[:,0:-1]
y = data.iloc[:,-1]
X.head()
y.value_counts()
# we can use pd.dummies for handling categorical data
X = pd.get_dummies(X)
X.head()
y.value_counts()
data['income'].value_counts()
from sklearn.cross_validation import train_test_split
from sklearn import cross_validation
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import cross_val_score
# split the data set into train and test
X_1, X_test, y_1, y_test = cross_validation.train_test_split(X, y, test_size=0.3, random_state=0)

# split the train data set into cross validation train and cross validation test
X_tr, X_cv, y_tr, y_cv = cross_validation.train_test_split(X_1, y_1, test_size=0.2)


for i in range(1,30,2):
    # instantiate learning model (k = 30)
    knn = KNeighborsClassifier(n_neighbors=i)

    # fitting the model on crossvalidation train
    knn.fit(X_tr, y_tr)

    # predict the response on the crossvalidation train
    pred = knn.predict(X_cv)

    # evaluate CV accuracy
    acc = accuracy_score(y_cv, pred, normalize=True) * float(100)
    print('\nCV accuracy for k = %d is %d%%' % (i, acc))
    
k = 17
knn = KNeighborsClassifier(k)
knn.fit(X_tr,y_tr)
pred = knn.predict(X_test)
knnacc = accuracy_score(y_test, pred, normalize=True) * float(100)
print('\n****Test accuracy for k =',k,' is %d%%' % (knnacc))
#Using Gradienboosting
from sklearn.ensemble import GradientBoostingClassifier
gbc = GradientBoostingClassifier().fit(X_tr,y_tr)

pred = gbc.predict(X_test)

gbcacc = gbc.score(X_test, y_test)

print('GBC: ', gbcacc * 100, '%')
# using Decision tree 
from sklearn import tree
dtc = tree.DecisionTreeClassifier(criterion='gini') # for classification, here you can change the algorithm as gini or entropy (information gain) by default it is gini  
# model = tree.DecisionTreeRegressor() for regression
# Train the model using the training sets and check score
dtc.fit(X_tr,y_tr)

pred = dtc.predict(X_test)

dtcacc = dtc.score(X_test, y_test)

print('Decision Tree classifier:', dtcacc * 100,'%')

# Random forest
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(max_depth=12, random_state=0)
rfc.fit(X_tr, y_tr)

pred = rfc.predict(X_test)

rfcacc = rfc.score(X_test, y_test)

print('Random Forest:', rfcacc * 100,'%')

accuracyScore = [knnacc, gbcacc * 100, dtcacc * 100, rfcacc * 100]
algoName = ['KNN', 'GBC', 'DT', 'RF']
plt.scatter(algoName, accuracyScore)
plt.grid()
plt.title('Algorithm Accuracy Comparision')
plt.xlabel('Algorithm')
plt.ylabel('Score in %')
plt.show()