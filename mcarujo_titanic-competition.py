import warnings

warnings.filterwarnings("ignore")
!pip install nb_black -q
%load_ext nb_black
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import math

gender_submission = pd.read_csv("../input/titanic/gender_submission.csv")
test = pd.read_csv("../input/titanic/test.csv")
train = pd.read_csv("../input/titanic/train.csv")
train.info()
def plot_hists(df, labels):
    row = 1
    col = 1
    num_graphs = len(labels)
    rows = math.ceil(num_graphs / 3)
    fig = make_subplots(rows=rows, cols=3, subplot_titles=labels)

    index = []
    for row in range(1, rows + 1):
        for col in range(1, 4):
            index.append({"row": row, "col": col})

    graphs = []
    pos_g = 0
    for label in labels:
        local_data = df[label].value_counts()
        x = list(local_data.index)
        y = list(local_data)
        fig.add_trace(
            go.Bar(x=x, y=y, text=y, textposition="auto",),
            row=index[pos_g]["row"],
            col=index[pos_g]["col"],
        )
        pos_g = pos_g + 1

    fig.show()
with sns.axes_style("white"):
    table = train.corr()
    mask = np.zeros_like(table)
    mask[np.triu_indices_from(mask)] = True
    plt.figure(figsize=(18, 7))
    sns.heatmap(
        table,
        cmap="Reds",
        mask=mask,
        vmax=0.3,
        linewidths=0.5,
        annot=True,
        annot_kws={"size": 15},
    )
plot_hists(train, ["SibSp", "Survived", "Pclass", "Parch", "Embarked", "Sex"])
def serie_categorie(value, serie):
    if value <= serie.quantile(0.2):
        return 1
    elif value <= serie.quantile(0.4):
        return 2
    elif value <= serie.quantile(0.6):
        return 3
    elif value <= serie.quantile(0.8):
        return 4
    else:
        return 5
from sklearn.preprocessing import OneHotEncoder


def formating_df(df):
    aux = df.drop(["Name", "PassengerId", "Ticket"], axis=1)
    # Fare
    aux.Fare.fillna(0, inplace=True)
    aux["Fare"] = aux["Fare"].astype(int)
    aux["Fare"] = [serie_categorie(val, aux["Fare"]) for val in aux["Fare"]]

    # SibsSp, Parch and Cabin
    bin_fill = lambda a: 1 if a > 0 else 0
    aux.Cabin.fillna(0, inplace=True)
    cabin_fill = lambda a: 1 if a != 0 else 0  # 1 means cabin e 2 means not cabin
    aux.Cabin = [cabin_fill(line) for line in aux.Cabin]
    aux["SibSp"] = [bin_fill(line) for line in aux.SibSp]
    aux["Parch"] = [bin_fill(line) for line in aux.Parch]

    # Sex
    to_change = {"male": 0, "female": 1}
    aux.Sex = aux.Sex.map(to_change)
    to_change = {"S": 1, "C": 2, "Q": 3}
    aux.Embarked = aux.Embarked.map(to_change)

    # Age
    aux.fillna(0, inplace=True)
    aux["Age"] = [serie_categorie(val, aux["Age"]) for val in aux["Age"]]

    # New features don't divide cause 2/2=3/3=1 then you'll lose information
    aux["Age_Fare"] = aux["Fare"] * aux["Age"]
    aux["Fare_Pclass"] = aux["Fare"] * aux["Pclass"]
    aux["Age_Pclass"] = aux["Pclass"] * aux["Age"]

    return aux


train_fmt = formating_df(train)
test_fmt = formating_df(test)
plot_hists(train_fmt, train_fmt.columns)
with sns.axes_style("white"):
    table = train_fmt.corr()
    mask = np.zeros_like(table)
    mask[np.triu_indices_from(mask)] = True
    plt.figure(figsize=(18, 7))
    sns.heatmap(
        table,
        cmap="Reds",
        mask=mask,
        vmax=0.3,
        linewidths=0.5,
        annot=True,
        annot_kws={"size": 15},
    )
import plotly.graph_objects as go

fig = make_subplots(rows=1, cols=2, subplot_titles=("Age", "Fare"))

fig.add_trace(go.Box(y=train_fmt.Age, name="Age"), row=1, col=1)
fig.add_trace(go.Box(y=train_fmt.Fare, name="Fare"), row=1, col=2)

fig.update_layout(height=600, width=975, title_text="Age and Fare Boxplots")
fig.show()
sns.pairplot(
    train_fmt, kind="reg", y_vars="Survived", x_vars=train_fmt.columns,
)
print("Train shape:", train_fmt.shape)
print("Test shape:", test_fmt.shape)
# Formating all the datas
x_train = train_fmt.drop(["Survived"], axis=1)
y_train = train_fmt.Survived

x_test = test_fmt

enc = OneHotEncoder(handle_unknown="ignore")
enc.fit(x_train)

x_train = enc.transform(x_train)
x_test = enc.transform(x_test)
print("Train shape:", x_train.shape)
print("Test shape:", x_test.shape)
import time

from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

# Models
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb


models = [
    ("SVC", SVC()),  # I change my mind, this model is taking too much time
    ("SGDClassifier", SGDClassifier()),
    ("LGBMClassifier", lgb.LGBMClassifier()),
    ("MLPClassifier", MLPClassifier()),
    (
        "DecisionTreeClassifier",
        DecisionTreeClassifier(criterion="entropy", class_weight="balanced"),
    ),
    ("NearestCentroid", NearestCentroid()),
    ("KNeighborsClassifier", KNeighborsClassifier()),
    ("DummyClassifier", DummyClassifier(strategy="most_frequent")),
    ("LogisticRegression", LogisticRegression()),
    ("RidgeClassifier", RidgeClassifier()),
    ("RandomForestClassifier", RandomForestClassifier(n_estimators=20)),
]


def train_test_validation(model, name, X, Y):
    print(f"Starting {name}.")  # Debug
    ini = time.time()  # Start clock
    scores = cross_val_score(model, X, Y, cv=4)  # Cross-validation
    fim = time.time()  # Finish clock
    print(f"Finish {name}.")  # Debug
    return (
        name,
        scores.mean(),
        scores.std() / scores.mean(),
        scores.std(),
        scores.max(),
        scores.min(),
        fim - ini,
    )
%%time
results = [ train_test_validation(model[1], model[0], x_train, y_train) for model in models ] # Testing for all models
results = pd.DataFrame(results, columns=['Classifier', 'Mean', 'CV','Std','Max', 'Min', 'TimeSpend (s)']) # Making a data frame
results.sort_values('Mean', ascending=False)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

scaler = StandardScaler()


def train_test_model(name, model, X, Y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42
    )
    model.fit(X_train, y_train)
    predict = model.predict(X_test)
    accuracy = accuracy_score(y_test, predict) * 100
    return (name, accuracy)


results_test = [
    train_test_model(model[0], model[1], x_train, y_train) for model in models
]  # Testing for all models
results_test = pd.DataFrame(
    results_test, columns=["Model", "Accuracy"]
)  # Making a data frame
results_test.sort_values("Accuracy", ascending=False)
%%time
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform


random_grid ={'num_leaves': sp_randint(2, 100), 
             'min_child_samples': sp_randint(10, 1000), 
             'min_child_weight': [1e-9,1e-8,1e-7,1e-6,1e-5,1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7],
             'subsample': sp_uniform(loc=0.2, scale=0.8), 
             'colsample_bytree': sp_uniform(loc=0.2, scale=0.8),
             'reg_alpha': [0, 1e-1, 1, 2, 5, 7, 10, 50, 100],
             'reg_lambda': [0, 1e-1, 1, 5, 10, 20, 50, 100]}


lgb_model = lgb.LGBMClassifier()
lgb_random = RandomizedSearchCV(
    estimator=lgb_model,
    param_distributions=random_grid,
    n_iter=20000,
    cv=4,
    verbose=2,
    random_state=42,
    n_jobs=-1,
)

lgb_random.fit(x_train, y_train)
bp = dict(lgb_random.best_params_)
bp
lgb_tunned = lgb.LGBMClassifier(
    num_leaves=bp["num_leaves"],
    min_child_samples=bp["min_child_samples"],
    min_child_weight=bp["min_child_weight"],
    subsample=bp["subsample"],
    colsample_bytree=bp["colsample_bytree"],
    reg_alpha=bp["reg_alpha"],
    reg_lambda=bp["reg_lambda"],
)
import plotly.figure_factory as ff

columns = ["Classifier", "Mean", "CV", "Std", "Max", "Min", "TimeSpend (s)"]
cv_score = train_test_validation(lgb_tunned, "LGB Tunned", x_train, y_train)
fig = ff.create_table([columns, cv_score])
fig.show()
from sklearn.metrics import confusion_matrix

X_train_tunned, X_test_tunned, y_train_tunned, y_test_tunned = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

lgb_tunned.fit(X_train_tunned,y_train_tunned)

m_c = confusion_matrix(y_test_tunned, lgb_tunned.predict(X_test_tunned))
plt.figure(figsize=(5, 4))
sns.heatmap(m_c, annot=True, cmap="Reds", fmt="d").set(
    xlabel="Predict", ylabel="Real"
)

lgb_tunned.fit(x_train, y_train)
gender_submission["Survived"] = rf_tunned.predict(x_test)
gender_submission.to_csv("Random Forest" + "_submission.csv", index=False)