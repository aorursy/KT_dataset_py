import os
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier

from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
#print(train.shape)
#print(test.shape)
train.head()
gen_pivot = train.pivot_table(index = "Sex", values = "Survived")
print(gen_pivot)
class_pivot = train.pivot_table(index = "Pclass", values = "Survived")
print(class_pivot)
'''
class_pivot.plot.bar()
plt.show()
'''
age_pivot = train.pivot_table(index = "Age", values = "Survived")

survived = train[train["Survived"] == 1]
died = train[train["Survived"] == 0]

survived["Age"].plot.hist(alpha = 0.5, color = "blue", bins=50)
died["Age"].plot.hist(alpha = 0.5, color = "red", bins = 50)

plt.legend(['survived', 'died'])
plt.show()
def process_age(df, cut_points, label_names):
    df["Age"] = df["Age"].fillna(-1)
    df["Age_categories"] = pd.cut(df["Age"], cut_points, labels = label_names)
cut_points = [-1,0, 5, 12, 18, 35, 60, 100]
label_names = ["Missing", 'Infant', "Child", 'Teenager', "Young Adult", 'Adult', 'Senior']
process_age(train, cut_points, label_names)
process_age(test, cut_points, label_names)
#print(list(train))

agecat_pivot = train.pivot_table(index = "Age_categories", values = "Survived")
agecat_pivot.plot.bar()
plt.show()
def create_dummies(df, column_name):
    dummies = pd.get_dummies(df[column_name], prefix = column_name)
    df = pd.concat([df, dummies], axis=1)
    return df
train = create_dummies(train, "Sex")
train = create_dummies(train, "Age_categories")
train = create_dummies(train, "Pclass")

test = create_dummies(test, "Sex")
test = create_dummies(test, "Age_categories")
test = create_dummies(test, "Pclass")
cols = ['Sex_female', 'Sex_male', 'Age_categories_Missing', 
          'Age_categories_Infant', 'Age_categories_Child', 'Age_categories_Teenager', 
          'Age_categories_Young Adult', 'Age_categories_Adult', 'Age_categories_Senior', 
          'Pclass_1', 'Pclass_2', 'Pclass_3']

all_x = train[cols]
all_y = train['Survived']
from sklearn.model_selection import train_test_split
train_x, test_x, train_y, test_y = train_test_split(all_x, all_y, test_size=0.2, random_state=37)
#train function
def train_clf(clf, X, y):
    clf.fit(X,y)
    
#predict function
def predict(clf, X):
    return(clf.predict(X))
# Initialize the models
A = LogisticRegression()
B = MultinomialNB(alpha=1.0,fit_prior=True)
C = DecisionTreeClassifier(random_state=42)
D = AdaBoostClassifier(n_estimators=100) 
E = KNeighborsClassifier(n_neighbors=1)
F = RandomForestClassifier(n_estimators=10, max_depth=None, min_samples_split=2, random_state=0)
# loop to call function for each model
clf = [A,B,C,D,E,F]
objects = ('LogRegression','Multi-NBayes', 'D-Trees', 'AdaBoost', 'K-NNeighbors', 'RandFor')
objs = ('LR', 'M-NB','DT','AdaB','KNN','RF')
score_f1 = [0,0,0,0,0,0]
score_accuracy = [0,0,0,0,0,0]

print("classifier\t F1_Score \t Accuracy-Score")
for a in range(len(clf)):
    train_clf(clf[a], train_x, train_y)
    y_pred = predict(clf[a],test_x)
    score_f1[a] = f1_score(test_y, y_pred) 
    score_accuracy[a] = accuracy_score(y_pred, test_y)
    print(objects[a],"\t", score_f1[a], "\t", score_accuracy[a])
#ploating data for F1 Score
y_pos = np.arange(len(objs))
y_val = [ x for x in score_f1]
plt.bar(y_pos,y_val, align='center', alpha=0.7)
plt.xticks(y_pos, objs)
plt.ylabel('F1 Score')
plt.title('Accuracy of Models')
plt.show()
#ploating data for Accuracy Score
y_pos = np.arange(len(objs))
y_val = [ x for x in score_accuracy]
plt.bar(y_pos,y_val, align='center', alpha=0.7)
plt.xticks(y_pos, objs)
plt.ylabel('Accuracy Score')
plt.title('Accuracy of Models')
plt.show()
