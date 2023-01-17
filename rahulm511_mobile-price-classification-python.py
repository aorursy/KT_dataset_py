import numpy as np

import pandas as pd

import matplotlib.pyplot as plt 

import seaborn as sns  #Plotting Lib
pd.options.display.max_columns = None

intrain = pd.read_csv('/kaggle/input/mobile-price-classification/train.csv')

intrain.head()
print(intrain.describe())
#Bar Plot

data = intrain['price_range'].value_counts() 

points = data.index 

frequency = data.values 

plt.bar(points, frequency, width = 0.4)
#Relation between the Price and Ram

sns.boxplot(intrain['price_range'],intrain['ram'])

plt.show()
#Relation between the Price and Ram

sns.boxplot(intrain['touch_screen'],intrain['price_range'])

plt.show()
intrain.corr()
corr=intrain.corr()

fig = plt.figure(figsize=(12,10))

r = sns.heatmap(corr, cmap='Blues')

r.set_title("Correlation ")
#Check the missing values in each column

intrain.isnull().sum()
X = intrain.iloc[:, :-1].values  # All input variable 

y = intrain.iloc[:,20].values    #Output Variable

print(intrain.columns)

print(y)

#Split the Data into Train and Test Sets

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# Create the Decision Tree model using our X_Train Dataset

from sklearn import tree

clf = tree.DecisionTreeClassifier()

clf = clf.fit(X_train, y_train)
#Visualize the Decision Tree which is created

clf1 = tree.DecisionTreeClassifier(max_depth=2) #Limiting the Depth to 2 for better visualization of Tree

clf1 = clf1.fit(X_train, y_train)

from sklearn.tree import export_graphviz

from sklearn.externals.six import StringIO  

from IPython.display import Image  

from sklearn.tree import export_graphviz

# Export as dot file

export_graphviz(clf1, out_file='tree.dot', 

                feature_names = ['battery_power', 'blue', 'clock_speed', 'dual_sim', 'fc', 'four_g',

       'int_memory', 'm_dep', 'mobile_wt', 'n_cores', 'pc', 'px_height',

       'px_width', 'ram', 'sc_h', 'sc_w', 'talk_time', 'three_g',

       'touch_screen', 'wifi'],

                class_names = ['0','1','2','3'],

                rounded = True, proportion = False, 

                precision = 2, filled = True)



# Convert to png

from subprocess import call

call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png', '-Gdpi=600'])



# Display in python

import matplotlib.pyplot as plt

plt.figure(figsize = (14, 18))

plt.imshow(plt.imread('tree.png'))

plt.axis('off');

plt.show();
#Test the model on X_test data

y_pred_test = clf.predict(X_test)
from sklearn.metrics import confusion_matrix

print(confusion_matrix(y_test, y_pred_test))

from sklearn.metrics import classification_report

print(classification_report(y_test,y_pred_test))
#Now Lets read the test dataset provide us and try to predict the values for it

intest = pd.read_csv('/kaggle/input/mobile-price-classification/test.csv')

Xtest = intest.iloc[:, 1:].values

Pred= clf.predict(Xtest)

intest['Pred'] = Pred

intest.head()
#Build the Random Forest Model for same data

from sklearn.ensemble import RandomForestClassifier

rfclf = RandomForestClassifier(n_estimators=100)

rfclf.fit(X_train, y_train) 
#Using feature importance find the most important features which decide the price ranges

fig = plt.figure(figsize=(8,6))

points = intrain.columns[0:20] 

frequency = rfclf.feature_importances_ 

plt.barh(points, frequency)
#Test the model on X_test data

y_predrf_test = rfclf.predict(X_test)
from sklearn.metrics import confusion_matrix

print(confusion_matrix(y_test, y_predrf_test))

from sklearn.metrics import classification_report

print(classification_report(y_test,y_predrf_test))