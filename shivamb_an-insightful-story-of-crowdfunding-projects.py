## load required libraries 

from plotly.offline import iplot, init_notebook_mode

import plotly.graph_objs as go



from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.feature_extraction import DictVectorizer

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from pdpbox import pdp, get_dataset, info_plots

from eli5.sklearn import PermutationImportance

from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import accuracy_score

from sklearn.pipeline import make_pipeline

from sklearn.pipeline import FeatureUnion

from collections import Counter

import matplotlib.pyplot as plt 

import pandas as pd 

import numpy as np 

import warnings

import eli5

warnings.filterwarnings('ignore')

init_notebook_mode(connected=True)



def syllable_count(word):

    word = word.lower()

    vowels = "aeiouy"

    count = 0

    if word[0] in vowels:

        count += 1

    for index in range(1, len(word)):

        if word[index] in vowels and word[index - 1] not in vowels:

            count += 1

    if word.endswith("e"):

        count -= 1

    if count == 0:

        count += 1

    return count



projects = pd.read_csv("../input/ks-projects-201801.csv", parse_dates = ["launched", "deadline"])



print ("Total Projects: ", projects.shape[0], "\nTotal Features: ", projects.shape[1])

projects.head()
projects = projects.dropna()

projects = projects[projects["currency"] == "USD"]

projects = projects[projects["state"].isin(["failed", "successful"])]

projects = projects.drop(["backers", "ID", "currency", "country", "pledged", "usd pledged", "usd_pledged_real", "usd_goal_real"], axis = 1)
## feature engineering

projects["syllable_count"]   = projects["name"].apply(lambda x: syllable_count(x))

projects["launched_month"]   = projects["launched"].dt.month

projects["launched_week"]    = projects["launched"].dt.week

projects["launched_day"]     = projects["launched"].dt.weekday

projects["is_weekend"]       = projects["launched_day"].apply(lambda x: 1 if x > 4 else 0)

projects["num_words"]        = projects["name"].apply(lambda x: len(x.split()))

projects["num_chars"]        = projects["name"].apply(lambda x: len(x.replace(" ","")))

projects["duration"]         = projects["deadline"] - projects["launched"]

projects["duration"]         = projects["duration"].apply(lambda x: int(str(x).split()[0]))

projects["state"]            = projects["state"].apply(lambda x: 1 if x=="successful" else 0)



## label encoding the categorical features

projects = pd.concat([projects, pd.get_dummies(projects["main_category"])], axis = 1)

le = LabelEncoder()

for c in ["category", "main_category"]:

    projects[c] = le.fit_transform(projects[c])
## Generate Count Features related to Category and Main Category

t2 = projects.groupby("main_category").agg({"goal" : "mean", "category" : "sum"})

t1 = projects.groupby("category").agg({"goal" : "mean", "main_category" : "sum"})

t2 = t2.reset_index().rename(columns={"goal" : "mean_main_category_goal", "category" : "main_category_count"})

t1 = t1.reset_index().rename(columns={"goal" : "mean_category_goal", "main_category" : "category_count"})

projects = projects.merge(t1, on = "category")

projects = projects.merge(t2, on = "main_category")



projects["diff_mean_category_goal"] = projects["mean_category_goal"] - projects["goal"]

projects["diff_mean_category_goal"] = projects["mean_main_category_goal"] - projects["goal"]



projects = projects.drop(["launched", "deadline"], axis = 1)

projects[[c for c in projects.columns if c != "name"]].head()
## define predictors and label 

label = projects.state

features = [c for c in projects.columns if c not in ["state", "name"]]



## prepare training and testing dataset

X_train, X_test, y_train, y_test = train_test_split(projects[features], label, test_size = 0.025, random_state = 2)

X_train1, y_train1 = X_train, y_train

X_test1, y_test1 = X_test, y_test



## train a random forest classifier 

model1 = RandomForestClassifier(n_estimators=50, random_state=0).fit(X_train1, y_train1)

y_pred = model1.predict(X_test1)
feature_importances = pd.DataFrame(model1.feature_importances_, index = X_train.columns, columns=['importance'])

feature_importances = feature_importances.sort_values('importance', ascending=False)



colors = ["gray"] * 9 + ["green"]*6

trace1 = go.Bar(y = [x.title()+"  " for x in feature_importances.index[:15][::-1]], 

                x = feature_importances.importance[:15][::-1], 

                name="feature importance (relative)",

                marker=dict(color=colors, opacity=0.4), orientation = "h")



data = [trace1]



layout = go.Layout(

    margin=dict(l=400), width = 1000,

    xaxis=dict(range=(0.0,0.15)),

    title='Relative Feature Importance (Which Features are important to make predictions ?)',

    barmode='group',

    bargap=0.25

)

fig = go.Figure(data=data, layout=layout)

iplot(fig)



from IPython.display import display

from IPython.core.display import HTML



tag = "<div> Most Important (Relative) : "

for feat in feature_importances.index[:10]:

    tag += "<span><font color='green'>" +feat.title().replace("_","")+ "</font> &nbsp;|&nbsp; </span>" 

tag += "<br>Least Important (Relative) : "

for feat in feature_importances.index[-15:]:

    tag += "<span><font color='red'>" +feat.title().replace("_","")+ "</font> &nbsp;|&nbsp; </span>" 

tag += "</div>"

display(HTML(tag))
import itertools 

import networkx as nx 

G = nx.Graph()

for tree_in_forest in model1.estimators_:

    doc = {}

    for i,key in enumerate(X_test1.columns):

        doc[key] = tree_in_forest.feature_importances_[i]

    sorted_doc = sorted(doc.items(), key=lambda kv: kv[1], reverse = True)[:10]

    sorted_doc = [c for c in sorted_doc if c[0] != "diff_mean_category_goal"]

    for i, j in itertools.product(sorted_doc, sorted_doc):

        if i == j:

            continue

        if i[1] >= 0.08 or j[1] >= 0.08:

            if np.absolute(i[1] - j[1]) <= 0.05:

                G.add_edge(i[0], j[0])



k = dict(G.degree()).keys()

v = dict(G.degree()).values()



plt.figure(figsize=(18, 6))

ax = plt.subplot(111)

plt.bar(k, v, width=0.80, color='#c4c4c4')

plt.title("Degree Centrality of Features")

plt.ylabel("Degree Centrality", fontsize=16)

plt.xticks(rotation=15, fontsize=16)

plt.yticks(color="white")



ax.spines['right'].set_visible(False)

ax.spines['bottom'].set_visible(False)

ax.spines['top'].set_visible(False)

ax.spines['left'].set_visible(False)

ax.xaxis.set_ticks_position('bottom')



def simpleaxis(ax):

    ax.spines['top'].set_visible(False)

    ax.spines['right'].set_visible(False)

    ax.get_xaxis().tick_bottom()

    ax.get_yaxis().tick_left()





labels = {}    

for node in G.nodes():

    labels[node] = node

        

plt.axes([0.5, 0.4, 0.5, 0.5])

pos = nx.spring_layout(G)

plt.axis('off')

nx.draw_networkx_nodes(G, pos, node_color = "red", node_size=20)

nx.draw_networkx_labels(G, pos, labels, font_size=16, font_color='black')

nx.draw_networkx_edges(G, pos, alpha=0.4)

plt.show()
from eli5.sklearn import PermutationImportance

import eli5

perm = PermutationImportance(model1, random_state=1).fit(X_test, y_test)

pi_df = eli5.explain_weights_df(perm, feature_names = X_test.columns.tolist())

pi_df["color"] = pi_df["weight"].apply(lambda x : "green" if x > 0 else "red")



data = [

    go.Bar(

        orientation = "h",

        y = pi_df.feature[::-1],

        x = pi_df.weight[::-1],

        marker = dict(

            opacity = 0.5,

            color = pi_df.color[::-1]        ),

        error_x = dict( type='data', color="#9fa3a3",

            array=list(pi_df["std"][::-1]),

            visible=True),

        name = 'expenses'

    )

]





layout = go.Layout(title="Permutation Importance", height = 800, margin=dict(l=300))



annotations = []

for i, row in pi_df.iterrows():

    dict(y=row.feature, x=row.weight, text="d",

                                  font=dict(family='Arial', size=14,

                                  color='rgba(245, 246, 249, 1)'),

                                  showarrow=False,)

layout['annotations'] = annotations

fig = go.Figure(data=data, layout = layout)

iplot(fig, filename='base-bar')
imp_df = feature_importances.reset_index().rename(columns = {"index" : "feature"})

combined_df = imp_df.merge(pi_df, on="feature")



trace0 = go.Scatter(

    x = combined_df.importance,

    y = combined_df.weight,

    text = [v.title() if i < 16 else "" for i,v in enumerate(list(combined_df.feature)) ],

    mode='markers+text',

    textposition='top center',

    marker=dict(

        size = 10, color="red", opacity=0.5,

    ),

)



trace1 = go.Scatter(

    x=[0.034, 0.095],

    y=[0.008, 0.020],

    text=['Cluster of Features',

          'Highly Important Features'],

    mode='text',

)



data = [trace0]

layout = go.Layout(title = "Features : Relative Importance VS Permutation Importance", 

                   showlegend = False, yaxis=dict(title="Permutation Importance (Feature Weight)", showgrid=False),

                   xaxis=dict(title="Feature Importance (Relative)", showgrid=False))

#                       shapes = [{ 'type': 'circle', 'xref': 'x', 'yref': 'y',

#                                   'x0': 0.024, 'y0': 0.007, 'x1': 0.045, 'y1': 0.001,'opacity': 1.0,

#                                   'line': { 'color': 'rgba(50, 171, 96, 1)', 'dash': 'dot',}},

#                                { 'type': 'rect', 'x0': 0.065, 'y0': 0.019, 'x1': 0.12, 'y1': 0.0002,

#                                 'line': { 'color': 'rgba(128, 0, 128, 1)' , 'dash' : 'dot' }}])

fig = go.Figure(data = data, layout = layout)

iplot(fig, filename='bubblechart-size-ref')
def clean_name(x):

    words = x.lower().split()

    cln = [wrd for wrd in words if not wrd[0].isdigit()]

    return " ".join(cln)

projects["cleaned_name"] = projects["name"].apply(lambda x : clean_name(x))



## add text features : top 100

vec = TfidfVectorizer(max_features=100, ngram_range=(1, 2), lowercase=True, stop_words="english", min_df=6)

X = vec.fit_transform(projects['cleaned_name'].values)



## append to original dataframe

vectors_df = pd.DataFrame(X.toarray(), columns=["_"+xx for xx in vec.get_feature_names()])

projects1_df = pd.concat([projects[features], vectors_df], axis=1)



## train the model

X_train, X_test, y_train, y_test = train_test_split(projects1_df, label, test_size = 0.25, random_state = 2)

X_train2, y_train2 = X_train[:15000], y_train[:15000]

X_test2, y_test2 = X_test[:1000], y_test[:1000]

model2 = RandomForestClassifier(random_state=1).fit(X_train2, y_train2)

y_pred = model2.predict(X_test2)



####### Interpretation 



from plotly import tools



perm = PermutationImportance(model2, random_state=1).fit(X_test2, y_test2)

pi_df = eli5.explain_weights_df(perm, feature_names = X_test2.columns.tolist(), feature_filter=lambda x: x[0] == '_')

pi_df["feature"] = pi_df["feature"].apply(lambda x : x[1:])

highs = pi_df[pi_df.weight >= 0.001]

med = pi_df[(pi_df.weight > -0.0005) & (pi_df.weight < 0.001)]

lows = pi_df[pi_df.weight <= -0.0005]



trace1 = go.Bar(

        orientation = "h",

        y = highs.feature[::-1],

        x = highs.weight[::-1],

        marker = dict(opacity = 0.4, color = "green" ), error_x = dict(type='data', color="#9fa3a3", array=list(highs["std"][::-1]), visible=True))

trace2 = go.Bar(

        orientation = "h",

        y = med.feature[:15][::-1],

        x = med.weight[:15][::-1],

        marker = dict(opacity = 0.4, color = "gray"), error_x = dict(type='data', color="#9fa3a3", array=list(med["std"][:15][::-1]), visible=True))

trace3 = go.Bar(

        orientation = "h",

        y = lows.feature,

        x = lows.weight,

        marker = dict(opacity = 0.4, color = "red"), error_x = dict(type='data', color="#9fa3a3", array=list(lows["std"][::-1]), visible=True))



ttls = ["Positive Impact","", "Moderate + or - Impact" ,"", "Negative Impact"]

fig = tools.make_subplots(rows=1, cols=5, print_grid=False, subplot_titles = ttls)

fig.append_trace(trace1, 1, 1)

fig.append_trace(trace2, 1, 3)

fig.append_trace(trace3, 1, 5)



fig['layout'].update(showlegend=False, title='Impact of Words Used in Project Name - Permutation Importance')

iplot(fig, filename='simple-subplot-with-annotations')
def _plot_pdp(feature, pdp_color, fill_color):

    plot_params = {

        'title': feature.title() + ' - Partial Dependency Plot',

        'title_fontsize': 15,

        'subtitle': 'How changes in "%s" affects the model predictions' % feature.title(),

        'subtitle_fontsize': 12,

        'font_family': 'Calibri',

        'xticks_rotation': 0,

        'line_cmap': 'cool',

        'zero_color': '#a2a5a0',

        'zero_linewidth': 1.0,

        'pdp_linewidth': 2.0,

        'fill_alpha': 0.25,

        'markersize': 5.5,

        'pdp_hl_color': 'green',

        'pdp_color': pdp_color,

        'fill_color': fill_color,



    }

    pdp_goals = pdp.pdp_isolate(model=model1, dataset=X_test1, model_features=X_test1.columns, feature=feature)

    pdp.pdp_plot(pdp_goals, feature, plot_params = plot_params)

    plt.ylabel("Change in Model Predictions");

    plt.show();

    

cols_of_interest = ['num_words', 'num_chars', 'syllable_count',

                    'duration', 'launched_month', 'launched_day',

                    'category_count', 'main_category_count']



_plot_pdp(cols_of_interest[0], "#f442b3", "#efaad6")
_plot_pdp(cols_of_interest[1], "#902fe0", "#c4a1e0")
_plot_pdp(cols_of_interest[2], "#5dcc2a", "#9dce86")
features_to_plot = ['num_words', 'num_chars']

inter1 = pdp.pdp_interact(model1, X_test1, X_test1.columns, features_to_plot)



plot_params = {

    'title': 'PDP interactaction plot for NumWords and NumChars',

    'subtitle': 'More red indicates better region',

    'title_fontsize': 15,

    'subtitle_fontsize': 12,

    'contour_color':  'white',

    'font_family': 'Calibri',

    'cmap': 'rainbow',

    'inter_fill_alpha': 0.6,

    'inter_fontsize': 9,

}



pdp.pdp_interact_plot(pdp_interact_out=inter1, feature_names=features_to_plot, plot_type='contour', plot_params = plot_params)

plt.show()
_plot_pdp(cols_of_interest[3], "#ff0077", "#fcb3d5")
_plot_pdp(cols_of_interest[4], "#00ffbb", "#c2fcec")
_plot_pdp(cols_of_interest[5], "#ff9d00", "#f7d399")
_plot_pdp(cols_of_interest[7], "#0800ff", "#cac9f2")
pdp_category = pdp.pdp_isolate(model=model1, dataset=X_test1, model_features=X_test1.columns,

                             feature=['Art', 'Comics', 'Crafts', 'Dance', 'Design', 'Fashion', 'Film & Video', 

                                      'Food', 'Games', 'Journalism', 'Music', 'Photography', 'Publishing', 'Technology', 'Theater'])

fig, axes = pdp.pdp_plot(pdp_isolate_out = pdp_category, feature_name='Project Category', 

                         center=True, plot_lines=False, frac_to_plot=100, plot_pts_dist=False)
fig, axes, summary_df = info_plots.actual_plot_interact(

    model=model1, X=X_test1,

    features=['goal', ['Art', 'Comics', 'Crafts', 'Dance',

       'Design', 'Fashion', 'Film & Video', 'Food', 'Games', 'Journalism',

       'Music', 'Photography', 'Publishing', 'Technology', 'Theater']],

    feature_names=['goal', 'Category'])
preds = model1.predict(X_test1)

dict(Counter(preds))
import shap 

shap.initjs()

data_for_prediction = X_test.iloc[1]

explainer = shap.TreeExplainer(model1)

shap_values = explainer.shap_values(data_for_prediction)

shap.force_plot(explainer.expected_value[1], shap_values[1], data_for_prediction, plot_cmap=["#f04e4e","#6677f9"])
shap.initjs()

data_for_prediction = X_test.iloc[2]

explainer = shap.TreeExplainer(model1)

shap_values = explainer.shap_values(data_for_prediction)

shap.force_plot(explainer.expected_value[1], shap_values[1], data_for_prediction, plot_cmap=["#f2654f","#7d4ff1"])
shap.initjs()

data_for_prediction = X_test.iloc[3]

explainer = shap.TreeExplainer(model1)

shap_values = explainer.shap_values(data_for_prediction)

shap.force_plot(explainer.expected_value[1], shap_values[1], data_for_prediction, plot_cmap=["#f79d2e","#6677f9"])
shap.initjs()

data_for_prediction = X_test.iloc[7]

explainer = shap.TreeExplainer(model1)

shap_values = explainer.shap_values(data_for_prediction)

shap.force_plot(explainer.expected_value[1], shap_values[1], data_for_prediction, plot_cmap=["#f04e4e","#6677f9"])
shap.initjs()

data_for_prediction = X_test.iloc[10]

explainer = shap.TreeExplainer(model1)

shap_values = explainer.shap_values(data_for_prediction)

shap.force_plot(explainer.expected_value[1], shap_values[1], data_for_prediction, plot_cmap=["#f79d2e","#812df7"])
X_test_s = X_test.head(1000)

shap_values = explainer.shap_values(X_test_s)

shap.summary_plot(shap_values[1], X_test_s)