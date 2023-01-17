import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import time

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV

from sklearn.preprocessing import Imputer

from sklearn.metrics import confusion_matrix

from subprocess import check_output

from sklearn import tree

from IPython.display import Image as PImage

from subprocess import check_call

from PIL import Image, ImageDraw, ImageFont

import re

import matplotlib.pyplot as plt



print(check_output(["ls", "../input"]).decode("utf8"))
train = pd.read_csv("../input/train.csv")

testset = pd.read_csv("../input/test.csv")
train['Sex'][train.Sex == 'female'] = 1

train['Sex'][train.Sex == 'male'] = 0

train.head(5)
imp = Imputer(missing_values='NaN', strategy='mean', axis=0)

columns = ['Pclass','Sex','Age','Fare','Parch','SibSp']

for col in columns:

    train[col] = imp.fit_transform(train[col].reshape(-1,1))
X = train[columns]

y = train.Survived



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
clf = DecisionTreeClassifier(max_depth = 3)

clf.fit(X_train,y_train)

print('Accuracy using the defualt gini impurity criterion...',clf.score(X_test,y_test))



clf = DecisionTreeClassifier(max_depth = 3, criterion = "entropy")

clf.fit(X_train,y_train)

print('Accuracy using the entropy criterion...',clf.score(X_test,y_test))
t = time.time()

clf = DecisionTreeClassifier(max_depth = 3, splitter = 'best')

clf.fit(X_train,y_train)

print('Best Split running time...',time.time() - t)

print('Best Split accuracy...',clf.score(X_test,y_test))



t = time.time()

clf = DecisionTreeClassifier(max_depth = 3, splitter = 'random')

clf.fit(X_train,y_train)

print('Random Split running time...',time.time() - t)

print('Random Split accuracy...',clf.score(X_test,y_test))
clf = DecisionTreeClassifier(max_depth = 3)

clf.fit(X_train,y_train)



with open("tree1.dot", 'w') as f:

     f = tree.export_graphviz(clf,

                              out_file=f,

                              max_depth = 5,

                              impurity = False,

                              feature_names = X_test.columns.values,

                              class_names = ['No', 'Yes'],

                              rounded = True,

                              filled= True )

        

#Convert .dot to .png to allow display in web notebook

check_call(['dot','-Tpng','tree1.dot','-o','tree1.png'])



# Annotating chart with PIL

img = Image.open("tree1.png")

draw = ImageDraw.Draw(img)

img.save('sample-out.png')

PImage("sample-out.png")
clf = DecisionTreeClassifier(max_depth = 3 ,splitter = 'random')

clf.fit(X_train,y_train)



with open("tree1.dot", 'w') as f:

     f = tree.export_graphviz(clf,

                              out_file=f,

                              max_depth = 5,

                              impurity = False,

                              feature_names = X_test.columns.values,

                              class_names = ['No', 'Yes'],

                              rounded = True,

                              filled= True )

        

#Convert .dot to .png to allow display in web notebook

check_call(['dot','-Tpng','tree1.dot','-o','tree1.png'])



# Annotating chart with PIL

img = Image.open("tree1.png")

draw = ImageDraw.Draw(img)

img.save('sample-out.png')

PImage("sample-out.png")
test_score = []

train_score = []

max_features = range(len(columns)-1)

for feat in max_features:

    clf = DecisionTreeClassifier(max_features = feat + 1)

    clf.fit(X_train,y_train)

    train_score.append(clf.score(X_train,y_train))

    test_score.append(clf.score(X_test,y_test))



plt.figure(figsize = (8,8))

plt.plot(max_features,train_score)

plt.plot(max_features, test_score)

plt.xlabel('Max Features')

plt.ylabel('Accuracy')

plt.legend(['Training set','Test set'])
test_score = []

train_score = []

max_features = range(len(columns)-1)

for feat in max_features:

    clf = DecisionTreeClassifier(max_features = feat + 1, max_depth = 5)

    clf.fit(X_train,y_train)

    train_score.append(clf.score(X_train,y_train))

    test_score.append(clf.score(X_test,y_test))

    

plt.figure(figsize = (8,8))   

plt.plot(max_features,train_score)

plt.plot(max_features, test_score)

plt.xlabel('Max Features')

plt.ylabel('Accuracy')

plt.legend(['Training set','Test set'])
test_score = []

train_score = []

for depth in range(20):

    clf = DecisionTreeClassifier(max_depth = depth + 1)

    clf.fit(X_train,y_train)

    train_score.append(clf.score(X_train,y_train))

    test_score.append(clf.score(X_test,y_test))



plt.figure(figsize = (8,8))

plt.plot(range(20),train_score)

plt.plot(range(20), test_score)

plt.xlabel('Tree Depth')

plt.ylabel('Accuracy')

plt.legend(['Training set','Test set'])
clf = DecisionTreeClassifier(max_depth = 6)

clf.fit(X_train,y_train)



with open("tree1.dot", 'w') as f:

     f = tree.export_graphviz(clf,

                              out_file=f,

                              max_depth = 5,

                              impurity = False,

                              feature_names = X_test.columns.values,

                              class_names = ['No', 'Yes'],

                              rounded = True,

                              filled= True )

        

#Convert .dot to .png to allow display in web notebook

check_call(['dot','-Tpng','tree1.dot','-o','tree1.png'])



# Annotating chart with PIL

img = Image.open("tree1.png")

draw = ImageDraw.Draw(img)

img.save('sample-out.png')

PImage("sample-out.png")
clf = DecisionTreeClassifier(max_depth = 3)

clf.fit(X_train,y_train)



with open("tree1.dot", 'w') as f:

     f = tree.export_graphviz(clf,

                              out_file=f,

                              max_depth = 5,

                              impurity = False,

                              feature_names = X_test.columns.values,

                              class_names = ['No', 'Yes'],

                              rounded = True,

                              filled= True )

        

#Convert .dot to .png to allow display in web notebook

check_call(['dot','-Tpng','tree1.dot','-o','tree1.png'])



# Annotating chart with PIL

img = Image.open("tree1.png")

draw = ImageDraw.Draw(img)

img.save('sample-out.png')

PImage("sample-out.png")
plt.barh(range(len(columns)),clf.feature_importances_)

plt.yticks(range(len(columns)),columns)

plt.xlabel('Feature Importance')
test_score = []

train_score = []

min_sample_split = np.arange(5,100,5)

for split in min_sample_split:

    clf = DecisionTreeClassifier(min_samples_split = split)

    clf.fit(X_train,y_train)

    train_score.append(clf.score(X_train,y_train))

    test_score.append(clf.score(X_test,y_test))

    

plt.figure(figsize = (8,8))   

plt.plot(min_sample_split,train_score)

plt.plot(min_sample_split, test_score)

plt.xlabel('Min Sample Split')

plt.ylabel('Accuracy')

plt.legend(['Training set','Test set'])
clf = DecisionTreeClassifier(min_samples_split = 5)

clf.fit(X_train,y_train)



with open("tree1.dot", 'w') as f:

     f = tree.export_graphviz(clf,

                              out_file=f,

                              max_depth = 5,

                              impurity = False,

                              feature_names = X_test.columns.values,

                              class_names = ['No', 'Yes'],

                              rounded = True,

                              filled= True )

        

#Convert .dot to .png to allow display in web notebook

check_call(['dot','-Tpng','tree1.dot','-o','tree1.png'])



# Annotating chart with PIL

img = Image.open("tree1.png")

draw = ImageDraw.Draw(img)

img.save('sample-out.png')

PImage("sample-out.png")
clf = DecisionTreeClassifier(min_samples_split = 80)

clf.fit(X_train,y_train)



with open("tree1.dot", 'w') as f:

     f = tree.export_graphviz(clf,

                              out_file=f,

                              max_depth = 5,

                              impurity = False,

                              feature_names = X_test.columns.values,

                              class_names = ['No', 'Yes'],

                              rounded = True,

                              filled= True )

        

#Convert .dot to .png to allow display in web notebook

check_call(['dot','-Tpng','tree1.dot','-o','tree1.png'])



# Annotating chart with PIL

img = Image.open("tree1.png")

draw = ImageDraw.Draw(img)

img.save('sample-out.png')

PImage("sample-out.png")
test_score = []

train_score = []

min_sample_leaf = np.arange(5,100,5)

for leaf in min_sample_leaf:

    clf = DecisionTreeClassifier(min_samples_leaf = leaf)

    clf.fit(X_train,y_train)

    train_score.append(clf.score(X_train,y_train))

    test_score.append(clf.score(X_test,y_test))



plt.figure(figsize = (8,8))

plt.plot(min_sample_split,train_score)

plt.plot(min_sample_split, test_score)

plt.xlabel('Min Sample Leaf')

plt.ylabel('Accuracy')

plt.legend(['Training set','Test set'])
clf = DecisionTreeClassifier(min_samples_leaf = 5)

clf.fit(X_train,y_train)



with open("tree1.dot", 'w') as f:

     f = tree.export_graphviz(clf,

                              out_file=f,

                              max_depth = 5,

                              impurity = False,

                              feature_names = X_test.columns.values,

                              class_names = ['No', 'Yes'],

                              rounded = True,

                              filled= True )

        

#Convert .dot to .png to allow display in web notebook

check_call(['dot','-Tpng','tree1.dot','-o','tree1.png'])



# Annotating chart with PIL

img = Image.open("tree1.png")

draw = ImageDraw.Draw(img)

img.save('sample-out.png')

PImage("sample-out.png")
clf = DecisionTreeClassifier(min_samples_leaf = 45)

clf.fit(X_train,y_train)



with open("tree1.dot", 'w') as f:

     f = tree.export_graphviz(clf,

                              out_file=f,

                              max_depth = 5,

                              impurity = False,

                              feature_names = X_test.columns.values,

                              class_names = ['No', 'Yes'],

                              rounded = True,

                              filled= True )

        

#Convert .dot to .png to allow display in web notebook

check_call(['dot','-Tpng','tree1.dot','-o','tree1.png'])



# Annotating chart with PIL

img = Image.open("tree1.png")

draw = ImageDraw.Draw(img)

img.save('sample-out.png')

PImage("sample-out.png")
test_score = []

train_score = []

max_leaf_nodes  = np.arange(5,100,5)

for leaf in max_leaf_nodes :

    clf = DecisionTreeClassifier(max_leaf_nodes  = leaf)

    clf.fit(X_train,y_train)

    train_score.append(clf.score(X_train,y_train))

    test_score.append(clf.score(X_test,y_test))

    

plt.figure(figsize = (8,8))

plt.plot(min_sample_split,train_score)

plt.plot(min_sample_split, test_score)

plt.xlabel('Min Leaf Nodes')

plt.ylabel('Accuracy')

plt.legend(['Training set','Test set'])
clf = DecisionTreeClassifier(max_leaf_nodes = 10)

clf.fit(X_train,y_train)



with open("tree1.dot", 'w') as f:

     f = tree.export_graphviz(clf,

                              out_file=f,

                              max_depth = 5,

                              impurity = False,

                              feature_names = X_test.columns.values,

                              class_names = ['No', 'Yes'],

                              rounded = True,

                              filled= True )

        

#Convert .dot to .png to allow display in web notebook

check_call(['dot','-Tpng','tree1.dot','-o','tree1.png'])



# Annotating chart with PIL

img = Image.open("tree1.png")

draw = ImageDraw.Draw(img)

img.save('sample-out.png')

PImage("sample-out.png")
clf = DecisionTreeClassifier(max_depth = 3)

clf.fit(X_train,y_train)

print('Class Weight is normal...')

print(confusion_matrix(y_test,clf.predict(X_test)))



clf = DecisionTreeClassifier(max_depth = 3, class_weight = 'balanced')

clf.fit(X_train,y_train)

print('Class weight is balanced to compensate for class imbalance...')

print(confusion_matrix(y_test,clf.predict(X_test)))
clf = DecisionTreeClassifier(max_depth = 3)

t = time.time()

clf.fit(X_train,y_train)

print('Without presot accuracy', clf.score(X_test,y_test))

print('Without presort runtime...',time.time() - t)



clf = DecisionTreeClassifier(max_depth = 3, presort = True)

t = time.time()

clf.fit(X_train,y_train)

print('With presot accuracy', clf.score(X_test,y_test))

print('With Presort runtime...',time.time() - t)