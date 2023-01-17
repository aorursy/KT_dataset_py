!pip install nb_black -q
%load_ext nb_black
import warnings



warnings.filterwarnings("ignore")
import numpy as np  # linear algebra

import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)



import os



for dirname, _, filenames in os.walk("/kaggle/input"):

    for filename in filenames:

        print(os.path.join(dirname, filename))
data = pd.read_csv("/kaggle/input/breast-cancer-wisconsin-data/data.csv")

data.drop(["Unnamed: 32", "id"], axis=1, inplace=True)
data.isna().sum()
import seaborn as sns

import matplotlib.pyplot as plt

import numpy as np



table = data

table["diagnosis_b"] = table.diagnosis.map({"M": 1.0, "B": 0.0})

table = table.corr()



with sns.axes_style("white"):

    mask = np.zeros_like(table)

    mask[np.triu_indices_from(mask)] = True

    plt.figure(figsize=(25, 25))

    sns.heatmap(

        round(table, 2),

        cmap="Reds",

        mask=mask,

        vmax=table.max().max(),

        vmin=table.min().min(),

        linewidths=0.5,

        annot=True,

        annot_kws={"size": 12},

    ).set_title("Correlation Matrix Breast Cancer Wisconsin Dataset")
sns.pairplot(data, hue='diagnosis_b', vars=['radius_mean',

'texture_mean',

'perimeter_mean',

'area_mean',

'smoothness_mean',

'compactness_mean',

'concavity_mean',

'concave points_mean',

'symmetry_mean',

'fractal_dimension_mean'])



import plotly.express as px



aux = table[["diagnosis_b"]].sort_values("diagnosis_b", ascending=False)

aux["columns"] = table.index

fig = px.bar(

    aux,

    x="columns",

    y="diagnosis_b",

    hover_data=["columns", "diagnosis_b"],

    color="diagnosis_b",

    height=600,

)

fig.show()
data.head()
import plotly.express as px





def plot_violin(columns, name):

    final = []

    for col in columns:

        aux = data[["diagnosis", col]]

        aux["type"] = col

        aux.columns = ["diagnosis", f"{name} values", "type"]

        final.append(aux)



    df = pd.concat(final)

    fig = px.violin(

        df,

        y=f"{name} values",

        x="type",

        color="diagnosis",

        box=True,

        points="all",

        hover_data=df.columns,

    )

    fig.update_layout(

        title_text=f"Values of {name} by the target (B,M)",

        xaxis_title="Diagnosis (0 = B = benign, 1 = M = malignant)",

    )

    fig.show()
col = "radius"

plot_violin([f"{col}_mean", f"{col}_se", f"{col}_worst"], col)
col = "texture"

plot_violin([f"{col}_mean", f"{col}_se", f"{col}_worst"], col)
col = "perimeter"

plot_violin([f"{col}_mean", f"{col}_se", f"{col}_worst"], col)
col = "area"

plot_violin([f"{col}_mean", f"{col}_se", f"{col}_worst"], col)
col = "smoothness"

plot_violin([f"{col}_mean", f"{col}_se", f"{col}_worst"], col)
col = "compactness"

plot_violin([f"{col}_mean", f"{col}_se", f"{col}_worst"], col)
col = "concavity"

plot_violin([f"{col}_mean", f"{col}_se", f"{col}_worst"], col)
col = "concave points"

plot_violin([f"{col}_mean", f"{col}_se", f"{col}_worst"], col)
col = "symmetry"

plot_violin([f"{col}_mean", f"{col}_se", f"{col}_worst"], col)
col = "fractal_dimension"

plot_violin([f"{col}_mean", f"{col}_se", f"{col}_worst"], col)
from sklearn.model_selection import train_test_split



x = data.drop(["diagnosis", "diagnosis_b"], axis=1)

y = data.diagnosis.values



train_x, test_x, train_y, test_y = train_test_split(x, y)
from sklearn.dummy import DummyClassifier



dc = DummyClassifier(strategy="most_frequent")

dc.fit(train_x, train_y)

dc.score(test_x, test_y)
from sklearn.ensemble import RandomForestClassifier

import random



SEED = 1234

random.seed(SEED)



rfc = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)

rfc.fit(train_x, train_y)

rfc.score(test_x, test_y)
from sklearn.feature_selection import SelectKBest, chi2



SEED = 1234

random.seed(SEED)



select_k = SelectKBest(chi2, k=5)



select_k.fit(train_x, train_y)

train_x_k = select_k.transform(train_x)

test_x_k = select_k.transform(test_x)



rfck = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)

rfck.fit(train_x_k, train_y)

rfck.score(test_x_k, test_y)
from sklearn.feature_selection import RFE



SEED = 12345

random.seed(SEED)



rfc_aux = RandomForestClassifier(

    n_estimators=100, max_depth=2, random_state=0, n_jobs=-1

)

rfc_rfe = RFE(estimator=rfc_aux, n_features_to_select=5, step=1,)



rfc_rfe.fit(train_x, train_y)

train_x_rfe = rfc_rfe.transform(train_x)

test_x_rfe = rfc_rfe.transform(test_x)



rfc_aux.fit(train_x_rfe, train_y)

rfc_aux.score(test_x_rfe, test_y)
train_x.columns[rfc_rfe.support_]
from sklearn.feature_selection import RFECV



SEED = 12345

random.seed(SEED)



rfc_aux_cv = RandomForestClassifier(n_estimators=10, max_depth=2, random_state=0)

rfc_rfecv = RFECV(estimator=rfc_aux_cv, cv=10, step=1, scoring="accuracy", n_jobs=-1)



rfc_rfecv.fit(train_x, train_y)

train_x_rfecv = rfc_rfecv.transform(train_x)

test_x_rfecv = rfc_rfecv.transform(test_x)



rfc_aux_cv.fit(train_x_rfecv, train_y)

rfc_aux_cv.score(test_x_rfecv, test_y)
train_x.columns[rfc_rfecv.support_]
from sklearn.metrics import confusion_matrix

from sklearn.decomposition import PCA

from sklearn.preprocessing import normalize



pca = PCA(n_components="mle")

pca_result = pca.fit(normalize(x, norm="max"))

train_x_pca = pca_result.transform(train_x)

test_x_pca = pca_result.transform(test_x)



pca_model = RandomForestClassifier(

    n_estimators=100, max_depth=2, random_state=0, n_jobs=-1

)

pca_model.fit(train_x_pca, train_y)





m_c = confusion_matrix(test_y, pca_model.predict(test_x_pca))

plt.figure(figsize=(5, 4))

sns.heatmap(m_c, annot=True, cmap="Reds", fmt="d").set(xlabel="Predict", ylabel="Real")

print(

    "Inicial number of features: ",

    test_x_pca.shape[1],

    " with the acurracy: ",

    pca_model.score(test_x_pca, test_y),

)
m_c = confusion_matrix(test_y, rfc.predict(test_x))

plt.figure(figsize=(5, 4))

sns.heatmap(m_c, annot=True, cmap="Reds", fmt="d").set(xlabel="Predict", ylabel="Real")

print(

    "Inicial number of features: ",

    train_x.shape[1],

    " with the acurracy: ",

    rfc.score(test_x, test_y),

)
m_c_k = confusion_matrix(test_y, rfck.predict(test_x_k))

plt.figure(figsize=(5, 4))

sns.heatmap(m_c_k, annot=True, cmap="Reds", fmt="d").set(

    xlabel="Predict", ylabel="Real"

)

print(

    "SelectKBest number of features: ",

    train_x_k.shape[1],

    " with the acurracy: ",

    rfck.score(test_x_k, test_y),

)
m_c_rfe = confusion_matrix(test_y, rfc_aux.predict(test_x_rfe))

plt.figure(figsize=(5, 4))

sns.heatmap(m_c_rfe, annot=True, cmap="Reds", fmt="d").set(

    xlabel="Predict", ylabel="Real"

)

print(

    "RFE number of features: ",

    train_x_rfe.shape[1],

    " with the acurracy: ",

    rfc_aux.score(test_x_rfe, test_y),

)
m_c_rfecv = confusion_matrix(test_y, rfc_aux_cv.predict(test_x_rfecv))

plt.figure(figsize=(5, 4))

sns.heatmap(m_c_rfecv, annot=True, cmap="Reds", fmt="d").set(

    xlabel="Predict", ylabel="Real"

)

print(

    "RFECV number of features: ",

    rfc_rfecv.n_features_,

    " with the acurracy: ",

    rfc_aux_cv.score(test_x_rfecv, test_y),

)
import plotly.graph_objects as go



pca = PCA(n_components=2)

pca_result = pca.fit_transform(normalize(x, norm="max"))

y_color = ["red" if el == "M" else "blue" for el in y]



fig = go.Figure(

    data=go.Scatter(

        x=pca_result[:, 0],

        y=pca_result[:, 1],

        mode="markers",

        marker={"color": y_color},

    )

)



fig.show()
from sklearn.manifold import TSNE

from sklearn.preprocessing import normalize

import plotly.graph_objects as go



tsne = TSNE(n_components=2)

tsne_result = tsne.fit_transform(normalize(x, norm="max"))



fig = go.Figure(

    data=go.Scatter(

        x=tsne_result[:, 0],

        y=tsne_result[:, 1],

        mode="markers",

        marker={"color": y_color},

    )

)



fig.show()
import time



from sklearn.metrics import accuracy_score

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import train_test_split



# Models

from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier

from sklearn.linear_model import SGDClassifier

from sklearn.neural_network import MLPClassifier

from sklearn.ensemble import RandomForestClassifier, VotingClassifier

from sklearn.neighbors import KNeighborsClassifier, NearestCentroid



models = [

    ("SVC", SVC()),

    ("RandomForestClassifier", RandomForestClassifier()),

    ("SGDClassifier", SGDClassifier()),

    ("MLPClassifier", MLPClassifier()),

    ("DecisionTreeClassifier", DecisionTreeClassifier()),

    ("NearestCentroid", NearestCentroid()),

    ("KNeighborsClassifier", KNeighborsClassifier()),

]





def train_test_validation(model, name, X, Y):

    print(f"Starting {name}.")  # Debug

    ini = time.time()  # Start clock

    scores = cross_val_score(model, X, Y, cv=4)  # Cross-validation

    fim = time.time()  # Finish clock

    print(f"Finish {name}.")  # Debug

    return (name, scores.mean(), scores.max(), scores.min(), fim - ini)
%%time

results = [ train_test_validation(model[1], model[0], x, y) for model in models ] # Testing for all models

results = pd.DataFrame(results, columns=['Classifier', 'Mean', 'Max', 'Min', 'TimeSpend (s)']) # Making a data frame
from plotly.subplots import make_subplots

import plotly.graph_objects as go



fig = make_subplots(rows=2, cols=1, shared_yaxes=True)

x_plot = results["Classifier"]

y_plot = round(results["Mean"] * 100, 2)

z_plot = round(results["TimeSpend (s)"], 2)



# Plots

fig.add_trace(go.Bar(x=x_plot, y=y_plot, text=y_plot, textposition="auto"), 1, 1)

fig.add_trace(go.Bar(x=x_plot, y=z_plot, text=z_plot, textposition="auto"), 2, 1)



fig.update_layout(height=800, width=1000, title_text="Traing Models Results")



# Update xaxis properties

fig.update_xaxes(title_text="Acurracy by Crossvalidation", row=1, col=1)

fig.update_xaxes(title_text="Time Spended by traing", row=2, col=1)



# Update yaxis properties

fig.update_yaxes(title_text="Accurracy in percent (%)", row=1, col=1)

fig.update_yaxes(title_text="Time in seconds (s)", row=2, col=1)





fig.show()
%%time

from sklearn.model_selection import RandomizedSearchCV

import numpy as np



# Number of trees in random forest

n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=25)]

# Number of features to consider at every split

max_features = ["auto", "sqrt"]

# Maximum number of levels in tree

max_depth = [int(x) for x in np.linspace(10, 110, num=25)]

max_depth.append(None)

# Minimum number of samples required to split a node

min_samples_split = [1, 2, 3, 5, 6, 7, 8, 9, 10]

# Minimum number of samples required at each leaf node

min_samples_leaf = [1, 2, 3, 5, 6, 7, 8, 9, 10]

# Method of selecting samples for training each tree

bootstrap = [True, False]

# Create the random grid

random_grid = {

    "n_estimators": n_estimators,

    "max_features": max_features,

    "max_depth": max_depth,

    "min_samples_split": min_samples_split,

    "min_samples_leaf": min_samples_leaf,

    "bootstrap": bootstrap,

}



# Use the random grid to search for best hyperparameters

# First create the base model to tune

rf = RandomForestClassifier()

# Random search of parameters, using 3 fold cross validation,

# search across 100 different combinations, and use all available cores

rf_random = RandomizedSearchCV(

    estimator=rf,

    param_distributions=random_grid,

    n_iter=300,

    cv=3,

    verbose=2,

    random_state=42,

    n_jobs=-1,

)

# Fit the random search model

rf_random.fit(x, y)
bp = dict(rf_random.best_params_)

bp
SEED = 1234

random.seed(SEED)



rf_tunned = RandomForestClassifier(

    n_estimators=bp["n_estimators"],

    min_samples_split=bp["min_samples_split"],

    min_samples_leaf=bp["min_samples_leaf"],

    max_features=bp["max_features"],

    max_depth=bp["max_depth"],

    bootstrap=bp["bootstrap"],

)
rf_tunned.fit(train_x, train_y)

rf_tunned.score(test_x, test_y)
m_c = confusion_matrix(test_y, rf_tunned.predict(test_x))

plt.figure(figsize=(5, 4))

sns.heatmap(m_c_rfe, annot=True, cmap="Reds", fmt="d").set(

    xlabel="Predict", ylabel="Real"

)