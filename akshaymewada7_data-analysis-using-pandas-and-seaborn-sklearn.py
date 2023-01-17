# Importing Modules

import pandas as pd 

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

% matplotlib inline
# Collecting data from csv and creating DataFrame

breast_cancer = pd.read_csv('../input/data.csv')

breast_cancer.head()
breast_cancer.info()
# Checking is there any null value

breast_cancer.isnull().sum()
#Describing DataFrame

breast_cancer.describe()


radius = breast_cancer[['radius_mean','radius_se','radius_worst','diagnosis']]

sns.pairplot(radius, hue='diagnosis')
texture = breast_cancer[['texture_mean','texture_se','texture_worst','diagnosis']]

sns.pairplot(texture, hue='diagnosis')
perimeter = breast_cancer[['perimeter_mean','perimeter_se','perimeter_worst','diagnosis']]

sns.pairplot(perimeter, hue='diagnosis')
area = breast_cancer[['area_mean','area_se','area_worst','diagnosis']]

sns.pairplot(area, hue='diagnosis')
smoothness = breast_cancer[['smoothness_mean','smoothness_se','smoothness_worst','diagnosis']]

sns.pairplot(smoothness, hue='diagnosis')
compactness = breast_cancer[['compactness_mean','compactness_se','compactness_worst','diagnosis']]

sns.pairplot(compactness, hue='diagnosis')
concavity = breast_cancer[['concavity_mean','concavity_se','concavity_worst','diagnosis']]

sns.pairplot(concavity, hue='diagnosis')
concave_points = breast_cancer[['concave points_mean','concave points_se','concave points_worst','diagnosis']]

sns.pairplot(concave_points, hue='diagnosis')
symmetry = breast_cancer[['symmetry_mean','symmetry_se','symmetry_worst','diagnosis']]

sns.pairplot(symmetry, hue='diagnosis')
fractal_dimension = breast_cancer[['fractal_dimension_mean','fractal_dimension_se','fractal_dimension_worst','diagnosis']]

sns.pairplot(fractal_dimension, hue='diagnosis')
# Calculating Numbers of Labels of Class 'M'  and 'B'

# Plotting lables

Label = breast_cancer['diagnosis']

a =pd.DataFrame(Label.value_counts())

print(a)

a.plot(kind='barh')
# Collecting Training Data from DataFrame

# Dropping Unwanted columns

train =  breast_cancer.drop(['id','diagnosis','Unnamed: 32'],axis=1)


plt.figure(figsize=(5,5))

sns.heatmap(data=train.ix[:,0:10].corr())
plt.figure(figsize=(5,5))

sns.heatmap(data=train.ix[:,11:20].corr())
plt.figure(figsize=(5,5))

sns.heatmap(data=train.ix[:,21:30].corr())
from sklearn.preprocessing import StandardScaler,LabelEncoder



# StandardScaler to scale train Dataset

s = StandardScaler()

s.fit(train)

Train = s.transform(train)



# Label Encoder to Encode Labels

lb = LabelEncoder()

lb.fit(Label)

Target = lb.transform(Label)
from sklearn.cross_validation import train_test_split

X_train,X_test,Y_train,Y_test = train_test_split(Train,Target,test_size = 0.2)
# importing the model for prediction



from sklearn.naive_bayes import GaussianNB

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC
# creating list of tuple wth model and its name  

models = []

models.append(('GNB',GaussianNB()))

models.append(('KNN',KNeighborsClassifier()))

models.append(('DT',DecisionTreeClassifier()))

models.append(('RF',RandomForestClassifier()))

models.append(('LG',LogisticRegression()))

models.append(('svc',SVC()))
# imorting cross Validation for calcuting score

from sklearn.cross_validation import cross_val_score



acc = []   # list for collecting Accuracy of all model

names = []    # List of model name



for name, model in models:

    

    acc_of_model = cross_val_score(model, X_train, Y_train, cv=30, scoring='accuracy')

    

    # appending Accuray of different model to acc List

    acc.append(acc_of_model)

    

    # appending name of models

    names.append(name)

    

    # printing Output 

    Out = "%s: %f" % (name, acc_of_model.mean())

    print(Out)
# Compare Algorithms Accuracy with each other on same Dataset

fig = plt.figure()

fig.suptitle('Algorithm Comparison')

ax = fig.add_subplot(111)

plt.boxplot(acc)

ax.set_xticklabels(names)

plt.show()