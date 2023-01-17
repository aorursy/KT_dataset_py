!pip install nb_black -q
%load_ext nb_black
import os

import numpy as np

import pandas as pd

import seaborn as sns

import plotly.express as px

import matplotlib.pyplot as plt



for dirname, _, filenames in os.walk("/kaggle/input"):

    for filename in filenames:

        print(os.path.join(dirname, filename))
data = pd.read_csv("/kaggle/input/appdata10/appdata10.csv", low_memory=True)

data["hour"] = data.hour.str.slice(1, 3)

data.info()
data.isna().sum()
print(data.shape)

print(len(data.user.unique()))
data.drop_duplicates("user", inplace=True)
table = data.corr()

with sns.axes_style("white"):

    mask = np.zeros_like(table)

    mask[np.triu_indices_from(mask)] = True

    plt.figure(figsize=(10, 10))

    sns.heatmap(

        round(table, 2),

        cmap="Reds",

        mask=mask,

        vmax=table.max().max(),

        vmin=table.min().min(),

        linewidths=0.5,

        annot=True,

        annot_kws={"size": 12},

    ).set_title("Correlation Matrix App behavior dataset")
aux = pd.DataFrame(

    data.groupby(["dayofweek", "enrolled"]).count()["user"]

).reset_index()

aux.enrolled = aux.enrolled.astype(str)

fig = px.bar(

    aux, y="user", x="dayofweek", color="enrolled", text="user", barmode="group"

)

fig.update_traces(texttemplate="%{text:.2s}", textposition="outside")

fig.update_layout(

    title_text="Part of the week Users use to enroll.",

    title_font_size=20,

    yaxis_title="Count",

    xaxis_title="Weekdays (0 = Sunday)",

)

fig.show()
aux = pd.DataFrame(data.groupby(["hour", "enrolled"]).count()["user"]).reset_index()

aux.enrolled = aux.enrolled.astype(str)

fig = px.bar(aux, y="user", x="hour", color="enrolled", text="user", barmode="group")

fig.update_traces(texttemplate="%{text:.2s}", textposition="outside")

fig.update_layout(

    title_text="Part of the day Users use to enroll.",

    title_font_size=20,

    yaxis_title="Count",

    xaxis_title="Hour of the day (24h format)",

)

fig.show()
fig = px.histogram(data, x="numscreens", marginal="box", color="enrolled", opacity=0.9,)

fig.update_layout(

    title_text="Histogram of number of screens (numscreens) accessed by users.",

    title_font_size=20,

    yaxis_title="Count",

    xaxis_title="Number of screnns accessed",

)

fig.show()
import plotly.graph_objects as go

from plotly.subplots import make_subplots



# Define color sets of paintings

labels = ["No", "Yes"]

bin_colors = ["red", "green"]

minigame = (

    data.minigame.value_counts().reset_index().sort_values("index").minigame.values

)

liked = data.liked.value_counts().reset_index().sort_values("index").liked.values

upf = (

    data.used_premium_feature.value_counts()

    .reset_index()

    .sort_values("index")

    .used_premium_feature.values

)

enrolled = (

    data.enrolled.value_counts().reset_index().sort_values("index").enrolled.values

)



# Create subplots, using 'domain' type for pie charts

specs = [

    [{"type": "domain"}, {"type": "domain"}],

    [{"type": "domain"}, {"type": "domain"}],

]



fig = make_subplots(

    rows=2,

    cols=2,

    specs=specs,

    subplot_titles=["Minigame", "Used Premium Feature", "Enrolled", "Liked"],

)



# Define pie charts

fig.add_trace(

    go.Pie(

        labels=labels,

        values=minigame,

        name="Did the minigame?",

        marker_colors=bin_colors,

    ),

    1,

    1,

)

fig.add_trace(

    go.Pie(

        labels=labels,

        values=liked,

        name="DId the user liked?",

        marker_colors=bin_colors,

    ),

    1,

    2,

)

fig.add_trace(

    go.Pie(

        labels=labels, values=upf, name="Was premium user?", marker_colors=bin_colors

    ),

    2,

    1,

)

fig.add_trace(

    go.Pie(

        labels=labels,

        values=enrolled,

        name="Enrolled in the game?",

        marker_colors=bin_colors,

    ),

    2,

    2,

)



# Tune layout and hover info

fig.update_traces(

    hole=0.3,

    textposition="inside",

    hoverinfo="name",

    textinfo="percent",

    textfont_size=15,

)

fig.update_layout(

    height=500,

    width=700,

    title_text="Binaries columns in pizza visualiation",

    title_font_size=20,

    showlegend=False,

)



fig = go.Figure(fig)

fig.show()
aux = data.dropna()

data["time_to_enroll"] = pd.to_datetime(aux.enrolled_date) - pd.to_datetime(

    aux.first_open

)
how_much_days = [str(i.days) for i in data["time_to_enroll"].dropna()]

fig = px.histogram(how_much_days, height=400)

fig.update_layout(

    title_text="Histogram of time to enroll in days (how much time to enroll our customers?)",

    title_font_size=20,

    xaxis_title="Time in days",

    yaxis_title="Count",

)



fig.show()

fig = px.histogram(how_much_days, height=400, log_y=True)

fig.update_layout(

    title_text="Histogram of time to enroll in days (how much time to enroll our customers?) - Log_y transformation",

    title_font_size=20,

    xaxis_title="Time in days",

    yaxis_title="Count in log scale",

)

fig.show()
import matplotlib.pyplot as plt

from wordcloud import WordCloud

from string import punctuation

import nltk



punctuation = [p for p in punctuation]

stopwords = nltk.corpus.stopwords.words("english")

stopwords = stopwords + punctuation + ["..."] + ["!!"]

token_punct = nltk.WordPunctTokenizer()

stemmer = nltk.RSLPStemmer()





def remove_punct(my_str):

    no_punct = ""

    for char in my_str:

        if char not in punctuation:

            no_punct = no_punct + char

    return no_punct





def tokenizer_column(serie):

    clear_col = list()

    for row in serie:

        new_line = list()

        line = token_punct.tokenize(remove_punct(row.lower()))

        for word in line:

            if word not in stopwords:  # stopwords

                new_line.append(stemmer.stem(word))

        clear_col.append(" ".join(new_line))

    return clear_col





def wordcloud(text, column_name, title):

    all_words = " ".join([text for text in text[column_name]])

    wordcloud = WordCloud(

        width=800, height=500, max_font_size=110, collocations=False

    ).generate(all_words)

    plt.figure(figsize=(24, 12))

    plt.imshow(wordcloud, interpolation="bilinear")

    plt.axis("off")

    plt.title(title)

    plt.show()





wordcloud(data, "screen_list", "screens")
result = []

[result.extend(el) for el in data.screen_list.str.split(",")]

result_set = set(result)

screen_list_dummies = pd.get_dummies(

    pd.Series(result), prefix_sep=",", columns=result_set

)

screen_list_dummies["enrolled"] = data.enrolled
df = (screen_list_dummies.mean() * 100).reset_index()

df.columns = ["screen", "acessed"]

fig = px.bar(

    df.sort_values("acessed", ascending=False)[1:],

    x="screen",

    y="acessed",

    color="acessed",

    height=600,

)

fig.update_layout(

    title_text="The most accessed screens in the App",

    title_font_size=20,

    xaxis_title="Screen's names",

    yaxis_title="Access by users in %",

    xaxis_tickangle=45,

)



fig.show()
table = screen_list_dummies.corr()

with sns.axes_style("white"):

    plt.figure(figsize=(12, 12))

    sns.heatmap(round(table, 4), cmap="Reds", vmax=1, vmin=0, linewidths=0,).set_title(

        "Correlation matrix between screen access"

    )
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.preprocessing import OneHotEncoder



enc = OneHotEncoder(handle_unknown="ignore")



data.screen_list = tokenizer_column(data.screen_list.str.replace(",", " "))

vectorizer = TfidfVectorizer(

    max_features=400,

    min_df=10,

    ngram_range=(1, 3),

    analyzer="word",

    stop_words="english",

)



X = np.concatenate(

    (

        ## NLP`

        vectorizer.fit_transform(data["screen_list"]).toarray(),

        ## OHE

        enc.fit_transform(data[["dayofweek", "hour", "age", "numscreens",]]).toarray(),

        ## BIN

        data[["minigame", "used_premium_feature", "liked",]].values,

    ),

    axis=1,

)



y = data.enrolled.values
X.shape
import time



from sklearn.metrics import accuracy_score

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import train_test_split



# Models

from sklearn.neural_network import MLPClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import BayesianRidge



models = [

    ("RandomForestClassifier", RandomForestClassifier()),

    ("MLPClassifier", MLPClassifier()),

    ("BayesianRidge", BayesianRidge()),

]





def train_test_validation(model, name, X, Y):

    print(f"Starting {name}.")  # Debug

    ini = time.time()  # Start clock

    scores = cross_val_score(model, X, Y, cv=4)  # Cross-validation

    fim = time.time()  # Finish clock

    print(f"Finish {name}.")  # Debug

    return (name, scores.mean(), scores.max(), scores.min(), fim - ini)
%%time

results = [ train_test_validation(model[1], model[0], X, y) for model in models ] # Testing for all models

results = pd.DataFrame(results, columns=['Classifier', 'Mean', 'Max', 'Min', 'TimeSpend (s)']) # Making a data frame

results
from sklearn.model_selection import RandomizedSearchCV



# Number of trees in random forest

n_estimators = [int(x) for x in np.linspace(start=1, stop=1000, num=100)]

# Number of features to consider at every split

max_features = ["auto", "sqrt"]

# Maximum number of levels in tree

max_depth = [int(x) for x in np.linspace(1, 1100, num=110)]

max_depth.append(None)

# Minimum number of samples required to split a node

min_samples_split = [2, 5, 10, 15, 20, 25, 30]

# Minimum number of samples required at each leaf node

min_samples_leaf = [1, 2, 4, 6, 8, 10]

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

rf = RandomForestClassifier(n_jobs=-1)

# Random search of parameters, using 3 fold cross validation,

# search across 100 different combinations, and use all available cores

rf_random = RandomizedSearchCV(

    estimator=rf,

    param_distributions=random_grid,

    n_iter=30,

    cv=4,

    verbose=10,

    random_state=42,

    n_jobs=-1,

)

# Fit the random search model

rf_random.fit(X[:10000], y[:10000])
p = rf_random.best_params_

print(p)

rfr = RandomForestClassifier(

    n_estimators=p["n_estimators"],

    max_features=p["max_features"],

    max_depth=p["max_depth"],

    min_samples_split=p["min_samples_split"],

    min_samples_leaf=p["min_samples_leaf"],

    bootstrap=p["bootstrap"],

    n_jobs=-1,

)



results = train_test_validation(rfr, "RandomForestRegressorT", X, y)

results = pd.DataFrame(results).T

results.columns = ["Classifier", "Mean", "Max", "Min", "TimeSpend (s)"]

results
%%time

from sklearn.manifold import TSNE





X_embedded = TSNE(n_components=2, n_jobs=-1).fit_transform(X[:10000])

X_embedded.shape
df = pd.DataFrame(X_embedded)

df.columns = ["Component 1", "Component 2"]

df["Target"] = y[:10000]



fig = px.scatter(df, x="Component 1", y="Component 2", color="Target")

fig.update_layout(

    title_text="Class visualization in 2D.",

    title_font_size=20,

    yaxis_title="Component 2",

    xaxis_title="Component 1",

)

fig.show()
%%time

X_embedded_3D = TSNE(n_components=3,n_jobs=-1).fit_transform(X[:10000])

X_embedded_3D.shape
df_3D = pd.DataFrame(X_embedded_3D)

df_3D.columns = ["Component 1", "Component 2", "Component 3"]

df_3D["Target"] = y[:10000]



fig = px.scatter_3d(

    df_3D, x="Component 1", y="Component 2", z="Component 3", color="Target"

)

fig.update_layout(

    title_text="Class visualization in 3D.", title_font_size=20,

)

fig.show()