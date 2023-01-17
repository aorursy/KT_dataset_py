from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import plotly.plotly as py
from plotly import tools
import plotly.figure_factory as ff
import pandas as pd
import numpy as np 
import seaborn as sns
import random 
import warnings
import operator
import copy

from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import tree
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")
init_notebook_mode(connected=True)
%matplotlib inline
plt.style.use('ggplot')

# Original Data
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

# Copy for preparation
train_prep = copy.deepcopy(train)
test_prep = copy.deepcopy(test)
train_prep.head()
train_prep.describe(include="all")
# Looking for null-values
train_prep.isnull().sum()
train_prep.info()
sns.pairplot(pd.get_dummies(train_prep, columns=["Sex"], drop_first=True), hue="Survived")
train_prep.drop(columns=["PassengerId"], inplace=True)
test_prep.drop(columns=["PassengerId"], inplace=True)
fig = go.Figure()

groups = train_prep.groupby(["Survived"]).count().reset_index()

data = go.Pie(
    labels = ["Died", "Survived"],
    values = [groups.Pclass[0], groups.Pclass[1]],
    marker=dict(colors=['#ff7f0e', '#1f77b4'])
)

layout = go.Layout(
    title='Survivors'
)

fig = go.Figure(data=[data], layout=layout)
iplot(fig)
fig = go.Figure()

groups = train_prep.groupby(["Pclass"]).count().reset_index()

trace1 = go.Bar(
    x = ["1st", "2nd", "3rd"],
    y = [groups.Sex[0], groups.Sex[1], groups.Sex[2]],
)

data = [trace1]
layout = go.Layout(
    title='Booked class',
    xaxis=dict(
        title='Class',
        titlefont=dict(
            family='Courier New, monospace',
            size=18,
            color='#7f7f7f'
        )
    ),
    yaxis=dict(
        title='No. of people',
        titlefont=dict(
            family='Courier New, monospace',
            size=18,
            color='#7f7f7f'
        )
    )
)

fig = go.Figure(data=data, layout=layout)
iplot(fig)
fig = go.Figure()

groups = train_prep.groupby(["Pclass", "Survived"]).count().reset_index()

trace1 = go.Bar(
    x = ["1st", "2nd", "3rd"],
    y = [groups.Sex[1], groups.Sex[3], groups.Sex[5]],
    name = "Survived"
)

trace2 = go.Bar(
    x = ["1st", "2nd", "3rd"],
    y = [groups.Sex[0], groups.Sex[2], groups.Sex[4]],
    name = "Died"
)

data = [trace1, trace2]
layout = go.Layout(
    barmode='stack',
    title='Survivors/deaths of the different classes booked',
    xaxis=dict(
        title='Class',
        titlefont=dict(
            family='Courier New, monospace',
            size=18,
            color='#7f7f7f'
        )
    ),
    yaxis=dict(
        title='No. of people',
        titlefont=dict(
            family='Courier New, monospace',
            size=18,
            color='#7f7f7f'
        )
    )
)

fig = go.Figure(data=data, layout=layout)
iplot(fig)
train_prep = pd.get_dummies(train_prep, columns=["Pclass"])
test_prep = pd.get_dummies(test_prep, columns=["Pclass"])
train_prep["Name"].head(10)
train_prep["Title"] = train_prep["Name"].str.extract(', ([A-Za-z]+)\.', expand=False)
test_prep["Title"] = test_prep["Name"].str.extract(', ([A-Za-z]+)\.', expand=False)

train_prep.drop(columns=["Name"], inplace=True)
test_prep.drop(columns=["Name"], inplace=True)
groups = train_prep.groupby(["Sex", "Title"], as_index=False)["Survived"].count()
groups
train_prep[train_prep["Title"].isnull()]
test_prep[test_prep["Title"].isnull()]
train_prep["Title"] = train_prep["Title"].replace(["Miss", "Mlle"], "Ms")
test_prep["Title"] = test_prep["Title"].replace(["Miss", "Mlle"], "Ms")

train_prep["Title"] = train_prep["Title"].replace(["Mme"], "Mrs")
test_prep["Title"] = test_prep["Title"].replace(["Mme"], "Mrs")

train_prep["Title"] = train_prep["Title"].fillna("Ms")

groups = train_prep.groupby(["Sex", "Title"], as_index=False)["Survived"].count()
groups
#Group 1
train_prep["Title"] = train_prep["Title"].replace(["Ms", "Mrs", "Mr", "Sir", "Jonkheer", "Lady", "Don", "Dona"], "1")
#Group 2
train_prep["Title"] = train_prep["Title"].replace(["Dr", "Master"], "2")
#Group 3
train_prep["Title"] = train_prep["Title"].replace(["Major","Col", "Capt", "Rev"], "3")

#Group 1
test_prep["Title"] = test_prep["Title"].replace(["Ms", "Mrs", "Mr", "Sir", "Jonkheer", "Lady", "Don", "Dona"], "1")
#Group 2
test_prep["Title"] = test_prep["Title"].replace(["Dr", "Master"], "2")
#Group 3
test_prep["Title"] = test_prep["Title"].replace(["Major","Col", "Capt", "Rev"], "3")

fig = go.Figure()

groups = train_prep.groupby(["Survived", "Sex"]).count().reset_index()

# Survived
trace1 = go.Bar(
    x = ["female", "male"],
    y = groups[(groups["Survived"] == 1)].Embarked,
    name = "Survived"
)

# Died
trace2 = go.Bar(
    x = ["female", "male"],
    y = groups[(groups["Survived"] == 0)].Embarked,
    name = "Died"
)

data = [trace1, trace2]
layout = go.Layout(
    barmode='stack',
    title='Survivors',
    xaxis=dict(
        title='Sex',
        titlefont=dict(
            family='Courier New, monospace',
            size=18,
            color='#7f7f7f'
        )
    ),
    yaxis=dict(
        title='No. of people',
        titlefont=dict(
            family='Courier New, monospace',
            size=18,
            color='#7f7f7f'
        )
    )
)

fig = go.Figure(data=data, layout=layout)
iplot(fig)

train_prep = pd.get_dummies(train_prep, columns=["Sex"], drop_first=True)
test_prep = pd.get_dummies(test_prep, columns=["Sex"], drop_first=True)
train_prep_age_mean = train_prep["Age"].mean()
test_prep_age_mean = test_prep["Age"].mean()

train_prep["Age"] = train_prep["Age"].fillna(train_prep_age_mean)
test_prep["Age"] = test_prep["Age"].fillna(test_prep_age_mean)

bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, np.inf]
labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8']
train_prep['AgeGroup'] = pd.cut(train_prep["Age"], bins, labels=labels)
test_prep['AgeGroup'] = pd.cut(test_prep["Age"], bins, labels=labels)

data = [go.Histogram(x=train_prep["AgeGroup"], histnorm="probability")]
iplot(data)
showLegend = [True,False]

data = []
for i in range(0,len(pd.unique(train_prep['Survived']))):
    male = {
            "type": 'violin',
            "x": train_prep['Survived'][ (train_prep['Sex_male'] == 1) & (train_prep['Survived'] == pd.unique(train_prep['Survived'])[i]) ],
            "y": train_prep['Age'][ (train_prep['Sex_male'] == 1) & (train_prep['Survived'] == pd.unique(train_prep['Survived'])[i]) ],
            "name": 'male',
            "side": 'negative',
            "showlegend": showLegend[i],
            "line": {
                "color": '#1f77b4'
            }
        }
    data.append(male)
    female = {
            "type": 'violin',
            "x": train_prep['Survived'][ (train_prep['Sex_male'] == 0) & (train_prep['Survived'] == pd.unique(train_prep['Survived'])[i]) ],
            "y": train_prep['Age'][ (train_prep['Sex_male'] == 0) & (train_prep['Survived'] == pd.unique(train_prep['Survived'])[i]) ],
            "name": 'female',
            "side": 'positive',
            "showlegend": showLegend[i],
            "line": {
                "color": '#ff7f0e'
            }
        }
    data.append(female)
        

fig = {
    "data": data,
    "layout" : {
        "title": "Age distribution by sex and survival",
        "yaxis": {
            "zeroline": True,
        },
        "violingap": 0,
        "violinmode": "overlay"
    }
}


iplot(fig, validate = False)

train_prep.drop(columns="Age", inplace=True)
test_prep.drop(columns="Age", inplace=True)
family = train_prep
family["familymembers"] = train_prep["Parch"] + train_prep["SibSp"]

groups = family.groupby(["familymembers"]).count().reset_index()

fig = go.Figure()

data = go.Pie(
    values = groups["Survived"]
)

layout = go.Layout(
    title='Distribution of persons by number of family members'
)

fig = go.Figure(data=[data], layout=layout)
iplot(fig)
groups = family.groupby(["familymembers", "Survived"]).count().reset_index()

barplot = sns.barplot(x="familymembers", y="Fare", hue="Survived", data=groups)
barplot.set_title("Survivor/Dead by number of family members")
barplot.set_xlabel("No. of family members")
barplot.set_ylabel("No. of people")
plt.tight_layout()
train_prep.drop(columns=["familymembers"], inplace=True)
'''
How can i do the previous grapic in plotlywithout stacking the last two groups?

groups = family.groupby(["familymembers", "Survived"]).count().reset_index()

fig = go.Figure()

trace1 = go.Bar(
    y = groups.iloc[::2, :]["Age"],
    name = "Survived"
)

trace2 = go.Bar(
    x = ["1st", "2nd", "3rd"],
    y = [groups.Sex[0], groups.Sex[2], groups.Sex[4]],
    name = "Died"
)

data = [trace1, trace2]
layout = go.Layout(
    barmode='stack',
    title='Survivors/deaths of the different classes booked',
    xaxis=dict(
        title='Class',
        titlefont=dict(
            family='Courier New, monospace',
            size=18,
            color='#7f7f7f'
        )
    ),
    yaxis=dict(
        title='No. of people',
        titlefont=dict(
            family='Courier New, monospace',
            size=18,
            color='#7f7f7f'
        )
    )
)

fig = go.Figure(data=data, layout=layout)
iplot(fig)
'''
train_prep.drop(columns=["Ticket"], inplace=True)
test_prep.drop(columns=["Ticket"], inplace=True)
trace1 = go.Box(
    y=train_prep[train_prep["Survived"] == 1]["Fare"],
    name="Survived"
)

trace2 = go.Box(
    y=train_prep[train_prep["Survived"] == 0]["Fare"],
    name="Died"
)

data=[trace1, trace2]

iplot(data)
train_prep.loc[train_prep['Cabin'].notnull(), 'Cabin'] = 1
test_prep.loc[test_prep['Cabin'].notnull(), 'Cabin'] = 1

train_prep["Cabin"].fillna(0, inplace=True)
test_prep["Cabin"].fillna(0, inplace=True)
fig = go.Figure()

groups = train_prep.groupby(["Embarked"]).count().reset_index()

data = go.Pie(
    labels = ["Cherbourg", "Queenstown", "Southampton"],
    values = groups.Fare
)

layout = go.Layout(
    title='No. of Passenger embarked per Port'
)

fig = go.Figure(data=[data], layout=layout)
iplot(fig)
fig = go.Figure()
x = ["Cherbourg", "Queenstown", "Southampton"]

groups = train_prep.groupby(["Embarked", "Survived"]).count().reset_index()

trace1 = go.Bar(
    x = x,
    y = [groups.Fare[1], groups.Fare[3], groups.Fare[5]],
    name = "Survived"
)

trace2 = go.Bar(
    x = x,
    y = [groups.Fare[0], groups.Fare[2], groups.Fare[4]],
    name = "Died"
)



data = [trace1, trace2]
layout = go.Layout(
    barmode='stack',
    title='Survivors/deaths by embarked port',
    xaxis=dict(
        title='Class',
        titlefont=dict(
            family='Courier New, monospace',
            size=18,
            color='#7f7f7f'
        )
    ),
    yaxis=dict(
        title='No. of people',
        titlefont=dict(
            family='Courier New, monospace',
            size=18,
            color='#7f7f7f'
        )
    )
)

fig = go.Figure(data=data, layout=layout)
iplot(fig)
train_prep = pd.get_dummies(train_prep, columns=["Embarked"], drop_first=True)
test_prep = pd.get_dummies(test_prep, columns=["Embarked"], drop_first=True)
train_prep.info()
test_prep.info()
train_prep.head()
test_prep.head()
test_prep["Fare"] = test_prep["Fare"].fillna(test_prep["Fare"].mean())

scaler = MinMaxScaler()

train_prep[["SibSp", "Parch", "Fare", "Title"]] = scaler.fit_transform(train_prep[["SibSp", "Parch", "Fare", "Title"]])
test_prep[["SibSp", "Parch", "Fare", "Title"]] = scaler.fit_transform(test_prep[["SibSp", "Parch", "Fare", "Title"]])
X_train = train_prep.drop(columns=["Survived"])
y_train = train_prep["Survived"]
X_test = test_prep

names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",
         "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
         "Naive Bayes", "QDA"
]

classifiers = [
    KNeighborsClassifier(),
    SVC(kernel="linear"),
    SVC(kernel="rbf"),
    GaussianProcessClassifier(),
    tree.DecisionTreeClassifier(max_depth=3),
    RandomForestClassifier(max_depth=3),
    MLPClassifier(),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis()
]

results = {}
for name, clf in zip(names, classifiers):
    scores = cross_val_score(clf, X_train, y_train, cv=5)
    results[name] = scores
    
for name, scores in results.items():
    print("%20s | Accuracy: %0.2f%% (+/- %0.2f%%)" % (name, 100*scores.mean(), 100*scores.std() * 2))
nn = classifiers[3]
nn.fit(X_train, y_train)
predictions = nn.predict(X_test)

submission = pd.DataFrame({ 'PassengerId' : test["PassengerId"], 'Survived': predictions })
submission.to_csv('submission_gp.csv', index=False)
nn = classifiers[4]
nn.fit(X_train, y_train)
predictions = nn.predict(X_test)

submission = pd.DataFrame({ 'PassengerId' : test["PassengerId"], 'Survived': predictions })
submission.to_csv('submission_dt.csv', index=False)
nn = classifiers[5]
nn.fit(X_train, y_train)
predictions = nn.predict(X_test)

submission = pd.DataFrame({ 'PassengerId' : test["PassengerId"], 'Survived': predictions })
submission.to_csv('submission_rf.csv', index=False)
nn = classifiers[6]
nn.fit(X_train, y_train)
predictions = nn.predict(X_test)

submission = pd.DataFrame({ 'PassengerId' : test["PassengerId"], 'Survived': predictions })
submission.to_csv('submission_nn.csv', index=False)
nn = classifiers[7]
nn.fit(X_train, y_train)
predictions = nn.predict(X_test)

submission = pd.DataFrame({ 'PassengerId' : test["PassengerId"], 'Survived': predictions })
submission.to_csv('submission_ab.csv', index=False)