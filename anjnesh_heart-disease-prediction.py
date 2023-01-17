import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import re



train = pd.read_csv("../input/heart.csv")

# loading data from heart.csv
train.dropna()

#  deletes rows with empty cells
# Percentage of men and women in our dataset



label = 'Men','Women'

sizes = [0.6831, 0.3168]

colors = ['gold','lightskyblue']

explode = (0.1, 0)  # explode 1st slice

 

# Plot

plt.pie(sizes, explode=explode, labels=label, colors=colors,

        autopct='%1.1f%%', shadow=True, startangle=140)

plt.title('% of men and women in our dataset\n\n\n')

plt.axis('equal')

plt.show()
# Plot of men at risk vs women at risk



label = 'risk of Men of having heartattack','risk of women having heartattack'

sizes = [(0.449), (0.75)]

colors = ['gold','lightskyblue']

explode = (0.08, 0)  # explode 1st slice

 

# Plot

plt.pie(sizes, explode=explode, labels=label, colors=colors,

        autopct='%1.1f%%', shadow=True, startangle=0)

 

plt.axis('equal')

plt.show()
# percentage of men at risk



slices_hours = [207-93, 93]

activities = ['% Men not at risk', '% Men at risk']

colors = ['r', 'g','b','y']

plt.pie(slices_hours, labels=activities, colors=colors, startangle=90, autopct='%.1f%%')

plt.title('ANALYSIS OF POSITIVE HEART ATTACK IN MEN OUT OF TOTAL MEN')

plt.show()
# percentage of women at risk



slices_hours = [96-72, 72]

activities = ['% Women not at risk', '% Women at risk']

colors = ['y','b']

plt.pie(slices_hours, labels=activities, colors=colors, startangle=0, autopct='%.1f%%')

plt.title('ANALYSIS OF POSITIVE HEART ATTACK IN WOMEN OUT OF TOTAL WOMEN')

plt.show()
# age distribution of people affected



import seaborn as sns



# seaborn histogram

sns.distplot(train['age'], hist=True, kde=False, 

             bins=int(180/5), color = 'blue',

             hist_kws={'edgecolor':'black'})

# Add labels

plt.title('distribution of people count vs age')

plt.xlabel('age(in years)')

plt.ylabel('number of person')

plt.show()

sns.distplot(train['age'])
# distribution of cholesterol level in affected people



sns.distplot(train['chol'], hist=True, kde=False, 

             bins=int(200/5), color = 'blue',

             hist_kws={'edgecolor':'black'})



plt.title('distribution serum cholesterol in mg/dl')

plt.xlabel('serum cholesterol in mg/dl')

plt.ylabel('count')

plt.show()

sns.distplot(train['chol'])
# max heart rate distribution 



sns.distplot(train['thalach'], hist=True, kde=False, 

             bins=int(200/5), color = 'blue',

             hist_kws={'edgecolor':'black'})



plt.title('distribution of maximum heart rate achieved')

plt.xlabel('maximum heart rate achieved')

plt.ylabel('count')

plt.show()

sns.distplot(train['thalach'])
# resting heart rate distribution



sns.distplot(train['trestbps'], hist=True, kde=False, 

             bins=int(100/5), color = 'blue',

             hist_kws={'edgecolor':'red'})



plt.title('distribution of resting blood pressure (in mm Hg on admission to the hospital)')

plt.xlabel('resting blood pressure (in mm Hg on admission to the hospital)')

plt.ylabel('count')

plt.show()

sns.distplot(train['trestbps'])
#  IMPORTING REQUIRED LIBRARIES





from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB
# SPLITTING DATA



from sklearn.model_selection import train_test_split

train1 = train.copy()

feature_df = train1[["age","sex","cp","trestbps","thalach","chol","restecg","exang","oldpeak","slope","ca","thal"]]

x = np.asarray(feature_df)

y = np.asarray(train["target"].astype('int'))

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 5)
def train_model(x_train, y_train, x_test, y_test, classifier, **kwargs):

    

    

    # instantiate model

    model = classifier(**kwargs)

    

    # train model

    model.fit(x_train,y_train)

    y_pred = model.predict(x_test)

    from sklearn.metrics import confusion_matrix

    print("Confusion matrix")

    print(confusion_matrix(y_test,y_pred))



    

    # check accuracy and print out the results

    fit_accuracy = model.score(x_train, y_train)

    test_accuracy = model.score(x_test, y_test)

    

    print(f"Train accuracy: {fit_accuracy:0.2%}")

    print(f"Test accuracy: {test_accuracy:0.2%}")

    

    return model
# KNN





model = train_model(x_train, y_train, x_test, y_test, KNeighborsClassifier)
# Seek optimal 'n_neighbours' parameter





for i in range(1,10):

    print("n_neigbors = "+str(i))

    train_model(x_train, y_train, x_test, y_test, KNeighborsClassifier, n_neighbors=i)
# OPTIMUM VALUE OF N NEIGHBOURS = 7
# Decision Tree

model = train_model(x_train, y_train, x_test, y_test, DecisionTreeClassifier, random_state=2606)



# Check optimal 'max_depth' parameter





for i in range(1,8):

    print("max_depth = "+str(i))

    train_model(x_train, y_train, x_test, y_test, DecisionTreeClassifier, max_depth=i, random_state=2606)
# With max_depth set as 3, the score went to almost 80%. Decision Tree outperforms KNN.
# LOGISTIC REGRESSION





model = train_model(x_train, y_train, x_test, y_test, LogisticRegression)
#Gaussian Naive Bayes





model = train_model(x_train, y_train, x_test, y_test, GaussianNB)
# Support Vector Machines





model = train_model(x_train, y_train, x_test, y_test, SVC)
# Support Vector Machine Linear





model = train_model(x_train, y_train, x_test, y_test, SVC, C=0.05, kernel='linear')
# Random Forests





model = train_model(x_train, y_train, x_test, y_test, RandomForestClassifier, random_state=2606)
# USING ANN Multilayer Perceptron Classifier



from sklearn.metrics import make_scorer, accuracy_score

from sklearn.model_selection import GridSearchCV

from sklearn.neural_network import MLPClassifier



ann_clf = MLPClassifier()



parameters = {'solver': ['lbfgs'], 'alpha':[1e-4], 'hidden_layer_sizes':(9,14,14,2),'random_state':[1]}



acc_scorer = make_scorer(accuracy_score)

grid_obj = GridSearchCV(ann_clf, parameters, scoring=acc_scorer)

grid_obj = grid_obj.fit(x_train, y_train)

ann_clf= grid_obj.best_estimator_









ann_clf.fit(x_train,y_train)

y_pred_ann = ann_clf.predict(x_test)

from sklearn.metrics import confusion_matrix

cm_ann = confusion_matrix(y_test, y_pred_ann)

cm_ann

ann_result = accuracy_score(y_test, y_pred_ann)

ann_result