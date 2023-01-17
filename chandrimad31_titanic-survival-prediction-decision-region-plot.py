import pandas as pd

import matplotlib.pyplot as plt

df = pd.read_csv('/kaggle/input/titanic/train.csv')

plt.style.use('ggplot')

fig = plt.figure(figsize = (8,5))

df.Survived.value_counts(normalize = True).plot(kind='bar', color= ['red','blue'], alpha = 0.5, rot = 1)

plt.title('Survived and Deceased')

plt.show()
plt.scatter(df.Survived, df.Age, color='maroon', alpha = 0.09)

plt.title('Relation between Survival and Age')

plt.show()
df.Pclass.value_counts(normalize = True).plot(kind='bar', color=['orange','red','green'], alpha = 0.6, rot = 1)

plt.title('Class Distribution')

plt.show()
for i in [1,2,3]:

    df.Age[df.Pclass == i].plot(kind = 'kde', alpha = 1.0)

plt.title("Class vs Age")

plt.legend(("1st", "2nd", "3rd"))

plt.show()
df.Embarked.value_counts(normalize = True).plot(kind='bar', color=['orange','red','green'], alpha = 0.6, rot = 1)

plt.title('Places where embarked')

plt.show()
m_color = '#F8BA00'

df.Survived[df.Sex == 'male'].value_counts(normalize = True).plot(kind='bar', alpha = 0.6, color = m_color, rot = 1)

plt.title('Male Survived')

plt.show()

f_color = '#FA0000'

df.Survived[df.Sex == 'female'].value_counts(normalize = True).plot(kind='bar', alpha = 0.6, color = f_color, rot = 1)

plt.title('Female Survived')

plt.show()

df.Sex[df.Survived == 1].value_counts(normalize = True).plot(kind='bar', alpha = 0.6, color = [f_color, m_color], rot = 1)

plt.title('Gender of Survived')

plt.show()
for i in [1,2,3]:

    df.Survived[df.Pclass == i].plot(kind = 'kde', alpha = 0.9, rot = 1)

plt.title("Class vs Survived")

plt.legend(("1st", "2nd", "3rd"))

plt.show()
plt.subplot2grid((4,4),(0,0), rowspan = 2, colspan = 2)

df.Survived[(df.Sex == 'male') & (df.Pclass == 1)].value_counts(normalize = True).plot(kind='bar', alpha = 0.5, color = m_color, rot = 1)

plt.title('Rich Male Survived')

plt.show()

plt.subplot2grid((4,4),(0,1), rowspan = 2, colspan = 2)

df.Survived[(df.Sex == 'male') & (df.Pclass == 3)].value_counts(normalize = True).plot(kind='bar', alpha = 0.5, color = m_color, rot = 1)

plt.title('Poor Male Survived')

plt.show()

plt.subplot2grid((4,4),(1,0), rowspan = 2, colspan = 2)

df.Survived[(df.Sex == 'female') & (df.Pclass == 1)].value_counts(normalize = True).plot(kind='bar', alpha = 0.5, color = f_color, rot = 1)

plt.title('Rich Female Survived')

plt.show()

plt.subplot2grid((4,4),(1,1), rowspan = 2, colspan = 2)

df.Survived[(df.Sex == 'female') & (df.Pclass == 3)].value_counts(normalize = True).plot(kind='bar', alpha = 0.5, color = f_color, rot = 1)

plt.title('Poor Female Survived')

plt.show()
def clean_data(data):

    data["Fare"] = data["Fare"].fillna(data["Fare"].dropna().median())

    data["Age"] = data["Age"].fillna(data["Age"].dropna().median())

    

    data.loc[data["Sex"] == "male", "Sex"] = 0

    data.loc[data["Sex"] == "female", "Sex"] = 1

    

    data["Embarked"] = data["Embarked"].fillna("S")

    data.loc[data["Embarked"] == "S", "Embarked"] = 0

    data.loc[data["Embarked"] == "C", "Embarked"] = 1

    data.loc[data["Embarked"] == "Q", "Embarked"] = 2
import warnings

warnings.filterwarnings("ignore")
# Check score with simple Logistic Regression Model

import pandas as pd

from sklearn import linear_model

train = pd.read_csv("/kaggle/input/titanic/train.csv")

clean_data(train)

target = train['Survived'].values

features = train[['Pclass', 'Age', 'Fare', 'Embarked', 'Sex', 'SibSp', 'Parch']].values

classifier = linear_model.LogisticRegression()

classifier_ = classifier.fit(features, target)

print(classifier_.score(features, target))
# Check score with Logistic Regression Model with Polynomial Degree = 2

from sklearn import linear_model, preprocessing

poly = preprocessing.PolynomialFeatures(degree=2)

poly_features = poly.fit_transform(features)

classifier_ = classifier.fit(poly_features, target)

print(classifier_.score(poly_features, target))
# Check score with Decision Tree Model

import pandas as pd

from sklearn import tree

train = pd.read_csv("/kaggle/input/titanic/train.csv")

clean_data(train)

target = train["Survived"].values

features = train[["Pclass", "Age", "Fare", "Embarked", "Sex", "SibSp", "Parch"]].values

decision_tree = tree.DecisionTreeClassifier(random_state = 42)

decision_tree_ = decision_tree.fit(features, target)

print(decision_tree_.score(features, target)) 
# Making the Decision Tree more generalized to reduce overfitting

from sklearn import model_selection

generalized_tree = tree.DecisionTreeClassifier(

                    random_state = 1,

                    max_depth = 7,

                    min_samples_split = 2)

generalized_tree_ = generalized_tree.fit(features, target)

scores = model_selection.cross_val_score(generalized_tree, features, target, scoring = 'accuracy', cv = 50)

print(scores)

print(scores.mean())
import graphviz

from sklearn.tree import DecisionTreeClassifier, export_graphviz

from sklearn.preprocessing import StandardScaler as scaler



data = export_graphviz(DecisionTreeClassifier(max_depth=3).fit(features, target), out_file=None, 

                       feature_names = ['Pclass', 'Age', 'Fare', 'Embarked', 'Sex', 'SibSp', 'Parch'],

                       class_names = ['Survived (0)', 'Survived (1)'], 

                       filled = True, rounded = True, special_characters = True)

# we have intentionally kept max_depth short here to accommodate the entire visual-tree

graph = graphviz.Source(data)

graph
#Perform Grid Search to tune hyperparameters of the Random Forest model

from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import RandomForestClassifier

forest = RandomForestClassifier(random_state = 1)

n_estimators = [1740, 1742, 1745, 1750]

max_depth = [6, 7, 8]

min_samples_split = [4, 5, 6]

min_samples_leaf = [4, 5, 6] 

oob_score = ['True']



hyperF = dict(n_estimators = n_estimators, max_depth = max_depth, min_samples_split = min_samples_split, min_samples_leaf = min_samples_leaf, oob_score = oob_score)



gridF = GridSearchCV(forest, hyperF, verbose = 1, n_jobs = 4)

bestF = gridF.fit(features, target)
#print(bestF)
# Check score with Random Forest Model having the best hyperparameters

from sklearn.ensemble import RandomForestClassifier

train = pd.read_csv("/kaggle/input/titanic/train.csv")

clean_data(train)

target = train["Survived"].values

features = train[["Pclass", "Age", "Fare", "Embarked", "Sex", "SibSp", "Parch"]].values

r_forest = RandomForestClassifier(criterion='gini',bootstrap=True,

                                    n_estimators=1745,

                                    max_depth=7,

                                    min_samples_split=6,

                                    min_samples_leaf=6,

                                    max_features='auto',

                                    oob_score=True,

                                    random_state=123,

                                    n_jobs=-1,

                                    verbose=0)

rf_clf = r_forest.fit(features, target)

print(rf_clf.score(features, target)) 
rf_clf.oob_score_
import numpy as np

import matplotlib.pyplot as plt

import matplotlib.gridspec as gridspec

import itertools

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from mlxtend.classifier import EnsembleVoteClassifier

from mlxtend.plotting import plot_decision_regions



value = 1.50

width = 0.75



clf1 = LogisticRegression(random_state=0)

clf2 = RandomForestClassifier(random_state=0)

clf3 = DecisionTreeClassifier(random_state=0) 

eclf = EnsembleVoteClassifier(clfs=[clf1, clf2, clf3], weights=[1, 1, 1], voting='soft')



X_list = train[["Pclass", "Age", "Fare", "Embarked", "Sex", "SibSp", "Parch"]]

X = np.asarray(X_list, dtype=np.float32)

y_list = train["Survived"]

y = np.asarray(y_list, dtype=np.int32)



# Plotting Decision Regions

gs = gridspec.GridSpec(2, 2)

fig = plt.figure(figsize=(10, 8))



labels = ['Logistic Regression',

          'Random Forest',

          'Decision Tree',

          'Ensemble']



for clf, lab, grd in zip([clf1, clf2, clf3, eclf],

                         labels,

                         itertools.product([0, 1],

                         repeat=2)):

    clf.fit(X, y)

    ax = plt.subplot(gs[grd[0], grd[1]])

    fig = plot_decision_regions(X=X, y=y, clf=clf, 

                                filler_feature_values={2: value, 3: value, 4: value, 5: value, 6: value}, 

                                filler_feature_ranges={2: width, 3: width, 4: width, 5: width, 6: width},

                                legend=2)

    plt.title(lab)



plt.show()
import pandas as pd

from sklearn import tree

test = pd.read_csv("/kaggle/input/titanic/test.csv")

clean_data(test)

prediction = rf_clf.predict(test[["Pclass", "Age", "Fare", "Embarked", "Sex", "SibSp", "Parch"]])

output = pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': prediction})

output.to_csv('titanic_submission.csv', index=False)

print("Submission successful")