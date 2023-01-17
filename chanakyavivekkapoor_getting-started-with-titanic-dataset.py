# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

import warnings

warnings.filterwarnings('always')

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train = pd.read_csv('/kaggle/input/titanic/train.csv')

test = pd.read_csv('/kaggle/input/titanic/test.csv')

test_ID = test['PassengerId'] ## Saving the IDs to be used while submitting the notebook 
train.head()
print("The number of rows in the dataset are: ", train.shape[0])

print("The number of columns which can be used to predict wheteher a person survived is: ", train.shape[1] - 1)
train.info()
# Outlier detection 

from collections import Counter



def detect_outliers_using_Tukey_Method(table, n, features):

    """

    Takes a dataframe table of features and returns a list of the indices

    corresponding to the observations containing more than n outliers according

    to the Tukey method.

    """

    outlier_indices = []

    

    # iterate over each column

    for col in features:

        # 1st quartile (25%)

        Q1 = np.percentile(table[col], 25)

        # 3rd quartile (75%)

        Q3 = np.percentile(table[col],75)

        # Interquartile range (IQR)

        IQR = Q3 - Q1

        

        # outlier step

        outlier_step = 1.5 * IQR

        

        # Determine a list of indices of outliers for feature col

        outlier_list_col = table[(table[col] < Q1 - outlier_step) | (table[col] > Q3 + outlier_step )].index

        

        # append the found outlier indices for col to the list of outlier indices 

        outlier_indices.extend(outlier_list_col)

        

    # select observations containing more than 2 outliers

    outlier_indices = Counter(outlier_indices)        

    multiple_outliers = list( k for k, v in outlier_indices.items() if v > n )

    

    return multiple_outliers   



# detect outliers from Age, SibSp , Parch and Fare as these are the continuous variables

Outliers_to_drop = detect_outliers_using_Tukey_Method(train ,2, ["Age","SibSp","Parch","Fare"])
train.loc[Outliers_to_drop]
train.drop(Outliers_to_drop, axis = 0).reset_index(drop=True, inplace = True)
train.isnull().sum()
from math import pi



from bokeh.io import output_notebook

from bokeh.io import show

from bokeh.plotting import figure

from bokeh.transform import cumsum

from bokeh.palettes import Spectral6

from bokeh.models import ColumnDataSource

from bokeh.layouts import gridplot
train['Survived'].value_counts()

survived_and_not_survived = {'Survived' : train['Survived'].value_counts()[1], 'Not Survived' : train['Survived'].value_counts()[0]}
data = pd.Series(survived_and_not_survived).reset_index(name = 'value').rename(columns = {'index':'Survived'})

data['angle'] = data['value']/data['value'].sum() * 2 * pi

data['color'] = ['skyblue', 'salmon']
output_notebook()
p = figure(plot_height=300, plot_width = 300, title="Pie Chart", toolbar_location=None,

           tools="hover", tooltips="@Survived: @value", x_range=(-0.5, 1.0))



p.wedge(x=0, y=1, radius=0.4,

        start_angle=cumsum('angle', include_zero=True), end_angle=cumsum('angle'),

        line_color="white", fill_color = 'color', legend_field='Survived', source=data)

p.legend.location = "top_right"

p.legend.label_text_font_size = '5pt'
unique = ['Survived', 'Not Survived']

top = [train['Survived'].value_counts()[1], train['Survived'].value_counts()[0]]

source = ColumnDataSource(data=dict(Survived = unique, counts = top, color = Spectral6))
p1 = figure(

    x_range = unique,

    plot_height=300,

    plot_width = 300,

    x_axis_label = 'Survived',

    y_axis_label = 'Count(Survived)',

    title = 'Count of Survived and Not Survived',

    tools="hover", tooltips="@Survived: @counts"

)



p1.vbar(

    x = 'Survived',

    top = 'counts',

    bottom = 0,

    width = 0.9,

    source = source,

    color = 'color'

)
show(gridplot([[p,p1]]))
types = {}

for i in train.columns:

    types[i] = train[i].unique()

print(types)
train.groupby(['Sex', 'Survived'])['Survived'].count()
sex_vs_survived = train.groupby(['Sex', 'Survived'])['Survived'].count().to_list()
unique = ['Female', 'Male']

top = [sex_vs_survived[1], sex_vs_survived[3]]

source = ColumnDataSource(data = dict(Survived = unique, counts = top, color = Spectral6))

       

p2 = figure(

    x_range = unique,

    plot_height = 400,

    plot_width = 400,

    x_axis_label = 'Sex',

    y_axis_label = 'Count(Survived)',

    title = 'Sex vs Survived',

    tools="hover", tooltips="@Survived: @counts"

)



p2.vbar(

    x = 'Survived',

    top = 'counts',

    bottom = 0,

    width = 0.9,

    source = source,

    color = 'color'

)
unique = ['Female', 'Male']

condition = ['Survived', 'Died']

colors = ["#c9d9d3", "#718dbf"]



data = {'Sex' : unique,

        'Survived' : [sex_vs_survived[1], sex_vs_survived[3]],

        'Died'   : [sex_vs_survived[0], sex_vs_survived[2]],

        }



p3 = figure(x_range = unique, plot_height = 400, plot_width = 400, title = "Sex vs Survival",

           toolbar_location=None, tools="")



p3.vbar_stack(condition, x ='Sex', width = 0.9, color = colors, source = data,

             legend_label = condition)
show(gridplot([[p3, p2]]))
pd.crosstab(train.Pclass, train.Survived, margins = True)
Pclass_Values = {}

for i in train['Pclass'].value_counts().index:

    Pclass_Values[i] = train['Pclass'].value_counts()[i]

print(Pclass_Values)
PClass = list(Pclass_Values.keys())

top = list(Pclass_Values.values())

source = ColumnDataSource(data = dict(Classes = PClass, counts = top, color = Spectral6))

       

p4 = figure(

    plot_height = 400,

    plot_width = 400,

    x_axis_label = 'Classes',

    y_axis_label = 'Count(Classes)',

    title = 'No of passengers in each class',

    tools="hover", tooltips="@Classes: @counts"

)



p4.vbar(

    x = 'Classes',

    top = 'counts',

    bottom = 0,

    width = 0.9,

    source = source,

    color = 'color'

)
pclass_vs_survived = train.groupby(['Pclass', 'Survived'])['Survived'].count().to_list()
pclass_vs_survived
unique = [1, 2, 3]

condition = ['Survived', 'Died']

colors = ["#718dbf", "#c9d9d3"]



data = {'Classes' : unique,

        'Survived' : [pclass_vs_survived[1], pclass_vs_survived[3], pclass_vs_survived[5]],

        'Died'   : [pclass_vs_survived[0], pclass_vs_survived[2], pclass_vs_survived[4]],

        }



p5 = figure(plot_height = 400, plot_width = 400, title = "Pclass vs Survival",

           )



p5.vbar_stack(condition, x ='Classes', width = 0.9, color = colors, source = data,

             legend_label = condition)
show(gridplot([[p4, p5]]))
pd.crosstab([train.Sex, train.Survived], train.Pclass, margins=True)
sns.factorplot('Pclass', 'Survived', hue = 'Sex', data = train)

plt.show()
train.Fare.isnull().sum()
hist, edges = np.histogram(train['Fare'], density=True, bins = 70)

p6 = figure(

    x_axis_label = 'Fare',

    title = 'Distplot'

)



p6.quad(

    bottom = 0,

    top = hist,

    left = edges[:-1],

    right = edges[1:],

    line_color = 'white'

)





show(p6)

print("The skewness of the feature is: ", train['Fare'].skew())
train["Fare"] = train["Fare"].map(lambda i: np.log(i + 1))
hist, edges = np.histogram(train['Fare'], density=True, bins = 70)

p7 = figure(

    x_axis_label = 'Fare',

    title = 'Distplot'

)



p7.quad(

    bottom = 0,

    top = hist,

    left = edges[:-1],

    right = edges[1:],

    line_color = 'white'

)





show(p7)

print("The skewness of the feature is: ", train['Fare'].skew())
train['Embarked'].value_counts()
train.groupby(['Embarked', 'Survived'])['Survived'].count()
embarked_vs_survived = train.groupby(['Embarked', 'Survived'])['Survived'].count().to_list()
unique = [1, 2, 3]

condition = ['Survived', 'Died']

colors = ["green", "red"]



data = {'Classes' : unique,

        'Survived' : [embarked_vs_survived[1], embarked_vs_survived[3], embarked_vs_survived[5]],

        'Died'   : [embarked_vs_survived[0], embarked_vs_survived[2], embarked_vs_survived[4]],

        }



p8 = figure(plot_height = 400, plot_width = 400, title = "Embarked vs Survival",)



p8.vbar_stack(condition, x ='Classes', width = 0.9, color = colors, source = data,

             legend_label = condition)

show(p8)
train['Age'].isnull().sum()
train_copy = train

train_copy.Age.dropna(inplace = True)
train_copy.Age.isnull().sum()
train

hist, edges = np.histogram(train_copy['Age'], density = True, bins = 30)

p8 = figure(

    plot_height = 500,

    plot_width = 500,

    x_axis_label = 'Age',

    title = 'Distplot'

)



p8.quad(

    bottom = 0,

    top = hist,

    left = edges[:-1],

    right = edges[1:],

    line_color = 'white'

)





show(p8)
g = sns.FacetGrid(train_copy, col='Survived')

g = g.map(sns.distplot, "Age")
print("The skewness for the first graph above is", train_copy["Age"][(train_copy["Survived"] == 0)].skew())

print("The skewness for the second graph above is", train_copy["Age"][(train_copy["Survived"] == 1)].skew())
dataset = pd.concat([train, test], axis = 0)
print("Train dataset shape: ", train.shape)

print("Test dataset shape: ", test.shape)

print("Combined dataset shape", dataset.shape)
dataset.isnull().sum()
sns.factorplot(x = "Sex", y = "Age", data = dataset, kind = "box")
sns.factorplot(x = "Sex", y = "Age", hue = "Pclass", data = dataset, kind = "box")
sns.factorplot(x = "Parch", y = "Age", data = dataset, kind = "box")
sns.factorplot(y = "Age", x = "SibSp", data = dataset, kind="box")
dataset['Sex'] = dataset['Sex'].map({'male' : 0, 'female' : 1})
sns.heatmap(dataset[["Age", "Sex", "SibSp", "Parch", "Pclass"]].corr(),cmap = "BrBG",annot = True)
null_index = dataset['Age'][dataset['Age'].isnull()].index
null_index
for i in null_index:

    age_med = dataset["Age"].median()

    age_pred = dataset["Age"][((dataset['SibSp'] == dataset.iloc[i]["SibSp"]) & (dataset['Parch'] == dataset.iloc[i]["Parch"]) & (dataset['Pclass'] == dataset.iloc[i]["Pclass"]))].median()

    dataset['Age'].iloc[i] = age_pred

    
dataset["Name"].head()
dataset_title = [i.split(",")[1].split(".")[0].strip() for i in dataset["Name"]]

dataset["Title"] = pd.Series(dataset_title)

dataset["Title"].head()
dataset['Title']
g = sns.countplot(x="Title",data=dataset)

g = plt.setp(g.get_xticklabels(), rotation=45)
dataset["Title"] = dataset["Title"].replace(['Lady', 'the Countess','Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

# First replace the 5 uncommon titles with rare.

dataset["Title"] = dataset["Title"].map({"Master":0, "Miss":1, "Ms" : 1 , "Mme":1, "Mlle":1, "Mrs":1, "Mr":2, "Rare":3})

# Map each of the titles to a number.

dataset["Title"] = dataset["Title"].astype(int)
g = sns.countplot(dataset["Title"])

g = g.set_xticklabels(["Master","Miss/Ms/Mme/Mlle/Mrs","Mr","Rare"])
g = sns.factorplot(x="Title",y="Survived",data=dataset,kind="bar")

g = g.set_xticklabels(["Master","Miss-Mrs","Mr","Rare"])

g = g.set_ylabels("survival probability")
dataset.drop('Name', axis = 1, inplace = True)
dataset['FamSize'] = dataset["SibSp"] + dataset["Parch"] + 1
g = sns.factorplot(x = "FamSize",y = "Survived",data = dataset)

g = g.set_ylabels("Survival Probability")
dataset['Single'] = dataset['FamSize'].map(lambda s: 1 if s == 1 else 0)

dataset['SmallFam'] = dataset['FamSize'].map(lambda s: 1 if  s == 2  else 0)

dataset['MedFam'] = dataset['FamSize'].map(lambda s: 1 if 3 <= s <= 4 else 0)

dataset['LargeFam'] = dataset['FamSize'].map(lambda s: 1 if s >= 5 else 0)
g = sns.factorplot(x = "Single", y = "Survived", data = dataset, kind = "bar")

g = sns.factorplot(x = "SmallFam", y = "Survived", data = dataset, kind = "bar")

g = sns.factorplot(x = "MedFam", y = "Survived", data = dataset, kind = "bar")

g = sns.factorplot(x = "LargeFam", y = "Survived", data = dataset, kind = "bar")
dataset = pd.get_dummies(dataset, columns = ["Title"])

dataset = pd.get_dummies(dataset, columns = ["Embarked"], prefix="Em")
dataset.head()
dataset.columns
dataset.drop(["PassengerId", "Ticket", "Fare", "Cabin"], axis = 1, inplace = True)
dataset.columns
dataset
dataset.shape
dataset.isnull().sum()
dataset['Age'].fillna(dataset['Age'].mean(), inplace = True)
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.neural_network import MLPClassifier

from sklearn.svm import SVC

from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve
len_train = train.shape[0]

train = dataset[:len_train]

test = dataset[len_train:]
X = train.drop('Survived', axis = 1)

y = train['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)
ss = StandardScaler()

X_train = ss.fit_transform(X_train)

X_test = ss.transform(X_test)
X_train.shape
X_test.shape
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.neural_network import MLPClassifier

from sklearn.svm import SVC

from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve
kfold = StratifiedKFold(n_splits=10)

random_state = 2

classifiers = []

classifiers.append(SVC(random_state=random_state))



classifiers.append(DecisionTreeClassifier(random_state=random_state))



classifiers.append(AdaBoostClassifier(DecisionTreeClassifier(random_state = random_state),random_state = random_state,learning_rate = 0.1))

classifiers.append(RandomForestClassifier(random_state = random_state))

classifiers.append(ExtraTreesClassifier(random_state = random_state))

classifiers.append(GradientBoostingClassifier(random_state = random_state))

classifiers.append(MLPClassifier(random_state = random_state))

classifiers.append(KNeighborsClassifier())

classifiers.append(LogisticRegression(random_state = random_state))

classifiers.append(LinearDiscriminantAnalysis())



cv_results = []

for classifier in classifiers :

    cv_results.append(cross_val_score(classifier, X_train, y = y_train, scoring = "accuracy", cv = kfold, n_jobs = 4))



cv_means = []

cv_std = []

for cv_result in cv_results:

    cv_means.append(cv_result.mean())

    cv_std.append(cv_result.std())



cv_res = pd.DataFrame({"CrossValMeans":cv_means,"CrossValerrors": cv_std,"Algorithm":["SVC","DecisionTree","AdaBoost",

"RandomForest","ExtraTrees","GradientBoosting","MultipleLayerPerceptron","KNeighboors","LogisticRegression","LinearDiscriminantAnalysis"]})



g = sns.barplot("CrossValMeans","Algorithm",data = cv_res, palette="Set3",orient = "h",**{'xerr':cv_std})

g.set_xlabel("Mean Accuracy")

g = g.set_title("Cross validation scores")
LR = LogisticRegression()

LR.fit(X_train, y_train)

predictions = LR.predict(X_test)

from sklearn.metrics import accuracy_score

print(accuracy_score(y_test, predictions))
from sklearn import svm, datasets

from sklearn.model_selection import GridSearchCV

parameters = {'kernel' : ('linear', 'rbf'), 'C' : [1, 10], 'gamma' : [0.001, 0.01, 0.1]}

svc = svm.SVC()

clf = GridSearchCV(svc, parameters)

clf.fit(X_train, y_train)
best_parameters = clf.best_params_
best_parameters
svc = SVC(kernel = best_parameters['kernel'], C = best_parameters['C'], gamma = best_parameters['gamma'] )
svc.fit(X_train, y_train)
predictions = svc.predict(X_test)
print(accuracy_score(y_test, predictions))
ABC = AdaBoostClassifier(DecisionTreeClassifier(random_state = 2), random_state = 2, learning_rate=0.1)
ABC.fit(X_train, y_train)
prediction_using_ABC = ABC.predict(X_test)
print(accuracy_score(y_test, prediction_using_ABC))
LDA = LinearDiscriminantAnalysis()

LDA.fit(X_train, y_train)
prediction_using_LDA = LDA.predict(X_test)
print(accuracy_score(y_test, prediction_using_LDA))
test.drop('Survived', axis = 1, inplace = True)
test.head()
test = ss.transform(test)
predictions = svc.predict(test)
test_Survived = pd.Series(predictions, name = "Survived")



results = pd.concat([test_ID, test_Survived],axis=1).astype(int)



results.to_csv("submission1.csv", index = False)