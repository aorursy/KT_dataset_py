# scientific computing libaries

import pandas as pd

import numpy as np



# data mining libaries

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import LabelEncoder

from sklearn.decomposition import PCA#, FastICA

from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, GridSearchCV, learning_curve

from sklearn import svm

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import roc_curve, auc, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score, roc_auc_score



from imblearn.pipeline import make_pipeline, Pipeline

from imblearn.over_sampling import SMOTE



#plot libaries

import plotly

import plotly.graph_objs as go

import plotly.figure_factory as ff

from plotly.offline import init_notebook_mode

init_notebook_mode(connected=True) # to show plots in notebook



# online plotly

#from plotly.plotly import plot, iplot

#plotly.tools.set_credentials_file(username='XXXXXXXXXXXXXXX', api_key='XXXXXXXXXXXXXXX')



# offline plotly

from plotly.offline import plot, iplot



# do not show any warnings

import warnings

warnings.filterwarnings('ignore')



SEED = 17 # specify seed for reproducable results

pd.set_option('display.max_columns', None) # prevents abbreviation (with '...') of columns in prints
RANDOM_FOREST_PARAMS = {

    'clf__max_depth': [25, 50, 75],

    'clf__max_features': ["sqrt"], # just sqrt is used because values of log2 and sqrt are very similar for our number of features (10-19) 

    'clf__criterion': ['gini', 'entropy'],

    'clf__n_estimators': [100, 300, 500, 1000]

}



DECISION_TREE_PARAMS = {

    'clf__max_depth': [25, 50, 75],

    'clf__max_features': ["sqrt"], # just sqrt is used because values of log2 and sqrt are very similar for our number of features (10-19)

    'clf__criterion': ['gini', 'entropy'],

    'clf__min_samples_split': [6, 10, 14],

}



LOGISTIC_REGRESSION_PARAMS = {

    'clf__solver': ['liblinear'],

    'clf__C': [0.1, 1, 10],

    'clf__penalty': ['l2', 'l1']

}



KNN_PARAMS = {

    'clf__n_neighbors': [5, 15, 25, 35, 45, 55, 65],

    'clf__weights': ['uniform', 'distance'],

    'clf__p': [1, 2, 10]

}



KNN_PARAMS_UNIFORM = {

    'clf__n_neighbors': [5, 15, 25, 35, 45, 55, 65],

    'clf__weights': ['uniform'],

    'clf__p': [1, 2, 10]

}



SVM_PARAMS = [

{

    'clf__kernel': ['linear'],

    'clf__C': [0.1, 1, 10],

}, 

{

    'clf__kernel': ['rbf'],

    'clf__C': [0.01, 0.1, 1, 10, 100],

    'clf__gamma': [0.01, 0.1, 1, 10, 100],

}]
# load the dataset

df = pd.read_csv('../input/bigml_59c28831336c6604c800002a.csv')



print("The dataset has %d rows and %d columns." % df.shape)
# check for null values in the dataset

print("There are " + ("some" if df.isnull().values.any() else "no")  + " null/missing values in the dataset.")
df.head(3)
def preprocess_data(df):

    pre_df = df.copy()

    

    # Replace the spaces in the column names with underscores

    pre_df.columns = [s.replace(" ", "_") for s in pre_df.columns]

    

    # convert string columns to integers

    pre_df["international_plan"] = pre_df["international_plan"].apply(lambda x: 0 if x=="no" else 1)

    pre_df["voice_mail_plan"] = pre_df["voice_mail_plan"].apply(lambda x: 0 if x=="no" else 1)

    pre_df = pre_df.drop(["phone_number"], axis=1)

    le = LabelEncoder()

    le.fit(pre_df['state'])

    pre_df['state'] = le.transform(pre_df['state'])

    

    return pre_df, le
pre_df, _ = preprocess_data(df)

pre_df.head(3)
pre_df.describe()
colors = plotly.colors.DEFAULT_PLOTLY_COLORS

churn_dict = {0: "no churn", 1: "churn"}
y = df["churn"].value_counts()



data = [go.Bar(x=[churn_dict[x] for x in y.index], y=y.values, marker = dict(color = colors[:len(y.index)]))]

layout = go.Layout(

    title='Churn distribution',

    autosize=False,

    width=400,

    height=400,

    yaxis=dict(

        title='#samples',

    ),

)

fig = go.Figure(data=data, layout=layout)

iplot(fig, filename='basic-bar15')
churn_perc = df["churn"].sum() * 100 / df["churn"].shape[0]

print("Churn percentage is %.3f%%." % churn_perc)
state_churn_df = df.groupby(["state", "churn"]).size().unstack()

trace1 = go.Bar(

    x=state_churn_df.index,

    y=state_churn_df[0],

    marker = dict(color = colors[0]),

    name='no churn'

)

trace2 = go.Bar(

    x=state_churn_df.index,

    y=state_churn_df[1],

    marker = dict(color = colors[1]),

    name='churn'

)

data = [trace1, trace2]

layout = go.Layout(

    title='Churn distribution per state',

    autosize=True,

    barmode='stack',

    margin=go.layout.Margin(l=50, r=50),

    xaxis=dict(

        title='state',

        tickangle=45

    ),

    yaxis=dict(

        title='#samples',

        automargin=True,

    ),

    legend=dict(

        x=0,

        y=1,

    ),

)

fig = go.Figure(data=data, layout=layout)

iplot(fig, filename='stacked-bar')
churn = pre_df[pre_df["churn"] == 1]

no_churn = pre_df[pre_df["churn"] == 0]
def create_churn_trace(col, visible=False):

    return go.Histogram(

        x=churn[col],

        name='churn',

        marker = dict(color = colors[1]),

        visible=visible,

    )



def create_no_churn_trace(col, visible=False):

    return go.Histogram(

        x=no_churn[col],

        name='no churn',

        marker = dict(color = colors[0]),

        visible = visible,

    )



features_not_for_hist = ["state", "phone_number", "churn"]

features_for_hist = [x for x in pre_df.columns if x not in features_not_for_hist]

active_idx = 0

traces_churn = [(create_churn_trace(col) if i != active_idx else create_churn_trace(col, visible=True)) for i, col in enumerate(features_for_hist)]

traces_no_churn = [(create_no_churn_trace(col) if i != active_idx else create_no_churn_trace(col, visible=True)) for i, col in enumerate(features_for_hist)]

data = traces_churn + traces_no_churn



n_features = len(features_for_hist)

steps = []

for i in range(n_features):

    step = dict(

        method = 'restyle',  

        args = ['visible', [False] * len(data)],

        label = features_for_hist[i],

    )

    step['args'][1][i] = True # Toggle i'th trace to "visible"

    step['args'][1][i + n_features] = True # Toggle i'th trace to "visible"

    steps.append(step)



sliders = [dict(

    active = active_idx,

    currentvalue = dict(

        prefix = "Feature: ", 

        xanchor= 'center',

    ),

    pad = {"t": 50},

    steps = steps,

)]



layout = dict(

    sliders=sliders,

    yaxis=dict(

        title='#samples',

        automargin=True,

    ),

)



fig = dict(data=data, layout=layout)



iplot(fig, filename='histogram_slider')
def create_box_churn_trace(col, visible=False):

    return go.Box(

        y=churn[col],

        name='churn',

        marker = dict(color = colors[1]),

        visible=visible,

    )



def create_box_no_churn_trace(col, visible=False):

    return go.Box(

        y=no_churn[col],

        name='no churn',

        marker = dict(color = colors[0]),

        visible = visible,

    )



features_not_for_hist = ["state", "phone_number", "churn"]

features_for_hist = [x for x in pre_df.columns if x not in features_not_for_hist]

# remove features with too less distinct values (e.g. binary features), because boxplot does not make any sense for them

features_for_box = [col for col in features_for_hist if len(churn[col].unique())>5]



active_idx = 0

box_traces_churn = [(create_box_churn_trace(col) if i != active_idx else create_box_churn_trace(col, visible=True)) for i, col in enumerate(features_for_box)]

box_traces_no_churn = [(create_box_no_churn_trace(col) if i != active_idx else create_box_no_churn_trace(col, visible=True)) for i, col in enumerate(features_for_box)]

data = box_traces_churn + box_traces_no_churn



n_features = len(features_for_box)

steps = []

for i in range(n_features):

    step = dict(

        method = 'restyle',  

        args = ['visible', [False] * len(data)],

        label = features_for_box[i],

    )

    step['args'][1][i] = True # Toggle i'th trace to "visible"

    step['args'][1][i + n_features] = True # Toggle i'th trace to "visible"

    steps.append(step)



sliders = [dict(

    active = active_idx,

    currentvalue = dict(

        prefix = "Feature: ", 

        xanchor= 'center',

    ),

    pad = {"t": 50},

    steps = steps,

    len=1,

)]



layout = dict(

    sliders=sliders,

    yaxis=dict(

        title='value',

        automargin=True,

    ),

    legend=dict(

        x=0,

        y=1,

    ),

)



fig = dict(data=data, layout=layout)



iplot(fig, filename='box_slider')
corr = pre_df.corr()

trace = go.Heatmap(z=corr.values.tolist(), x=corr.columns, y=corr.columns)

data=[trace]

layout = go.Layout(

    title='Heatmap of pairwise correlation of the columns',

    autosize=False,

    width=850,

    height=700,

    yaxis=go.layout.YAxis(automargin=True),

    xaxis=dict(tickangle=40),

    margin=go.layout.Margin(l=0, r=200, b=200, t=80)

)





fig = go.Figure(data=data, layout=layout)

iplot(fig, filename='labelled-heatmap1')
from scipy.cluster import hierarchy as hc

X = np.random.rand(10, 10)

names = pre_df.columns

inverse_correlation = 1 - abs(pre_df.corr())

fig = ff.create_dendrogram(inverse_correlation.values, orientation='left', labels=names, colorscale=colors, linkagefun=lambda x: hc.linkage(x, 'average'))

fig['layout'].update(dict(

    title="Dendogram of clustering the features according to correlation",

    width=800, 

    height=600,

    margin=go.layout.Margin(l=180, r=50),

    xaxis=dict(

        title='distance',

    ),

    yaxis=dict(

        title='features',

        automargin=True,

    ),

))

iplot(fig, filename='dendrogram_corr_clustering')
# save the duplicate features for later usage

duplicate_features = ["total_day_charge", "total_eve_charge", "total_night_charge", "total_intl_charge"]
# splitting the dataset into feature vectors and the target variable

df_y = pre_df["churn"]

df_X = pre_df.drop(["churn"], axis=1)
# normalize the dataset (note: for decision tree/random forest it would not be needed)

df_X_normed = (df_X - df_X.mean()) / df_X.std()
# calculate the principal components

pca = PCA(random_state=SEED)

df_X_pca = pca.fit_transform(df_X_normed)
tot = sum(pca.explained_variance_) # total explained variance of all principal components

var_exp = [(i / tot) * 100 for i in sorted(pca.explained_variance_, reverse=True)] # individual explained variance

cum_var_exp = np.cumsum(var_exp) # cumulative explained variance
trace_cum_var_exp = go.Bar(

    x=list(range(1, len(cum_var_exp) + 1)), 

    y=var_exp,

    name="individual explained variance",

)

trace_ind_var_exp = go.Scatter(

    x=list(range(1, len(cum_var_exp) + 1)),

    y=cum_var_exp,

    mode='lines+markers',

    name="cumulative explained variance",

    line=dict(

        shape='hv',

    ))

data = [trace_cum_var_exp, trace_ind_var_exp]

layout = go.Layout(

    title='Individual and Cumulative Explained Variance',

    autosize=True,

    yaxis=dict(

        title='percentage of explained variance',

    ),

    xaxis=dict(

        title="principal components",

        dtick=1,

    ),

    legend=dict(

        x=0,

        y=1,

    ),

)

fig = go.Figure(data=data, layout=layout)

iplot(fig, filename='basic-bar')
n_components = 10

df_X_reduced = np.dot(df_X_normed.values, pca.components_[:n_components,:].T)

df_X_reduced = pd.DataFrame(df_X_reduced, columns=["PC#%d" % (x + 1) for x in range(n_components)])
# prints the best grid search scores along with their parameters.

def print_best_grid_search_scores_with_params(grid_search, n=5):

    if not hasattr(grid_search, 'best_score_'):

        raise KeyError('grid_search is not fitted.')

    print("Best grid scores on validation set:")

    indexes = np.argsort(grid_search.cv_results_['mean_test_score'])[::-1][:n]

    means = grid_search.cv_results_['mean_test_score'][indexes]

    stds = grid_search.cv_results_['std_test_score'][indexes]

    params = np.array(grid_search.cv_results_['params'])[indexes]

    for mean, std, params in zip(means, stds, params):

        print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
def do_gridsearch_with_cv(clf, params, X_train, y_train, cv, smote=None):



    if smote is None:

        pipeline = Pipeline([('clf', clf)])

    else:

        pipeline = Pipeline([('sm', sm), ('clf', clf)])

        

    gs = GridSearchCV(pipeline, params, cv=kf, n_jobs=-1, scoring='f1', return_train_score=True)

    gs.fit(X_train, y_train)

    return gs



def score_on_test_set(clfs, datasets):

    scores = []

    for c, (X_test, y_test) in zip(clfs, datasets):

        scores.append(c.score(X_test, y_test))

    return scores
# split data into train and test set in proportion 4:1 for all differntly preprocessed datasets

X_train, X_test, y_train, y_test = train_test_split(df_X_normed, df_y, test_size=0.2, random_state=SEED)

X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(df_X_reduced, df_y, test_size=0.2, random_state=SEED)

cols_without_duplicate = [x for x in df_X_normed.columns if x not in duplicate_features]

X_train_red, X_test_red, y_train_red, y_test_red = train_test_split(df_X_normed[cols_without_duplicate], df_y, test_size=0.2, random_state=SEED)
print("Shape of the full train dataset:", X_train.shape)

print("Shape of the train dataset with reduced features", X_train_red.shape)

print("Shape of the transformed train dataset using the first 10 Principal Components", X_train_pca.shape)
sm = SMOTE(random_state=SEED)

kf = StratifiedKFold(n_splits=5, random_state=SEED)

clf_rf = RandomForestClassifier(random_state=SEED)

clf_balanced = RandomForestClassifier(random_state=SEED, class_weight="balanced")
%%time

gs_full = do_gridsearch_with_cv(clf_rf, RANDOM_FOREST_PARAMS, X_train, y_train, kf, smote=None)

gs_red = do_gridsearch_with_cv(clf_rf, RANDOM_FOREST_PARAMS, X_train_red, y_train_red, kf, smote=None)

gs_pca = do_gridsearch_with_cv(clf_rf, RANDOM_FOREST_PARAMS, X_train_pca, y_train_pca, kf, smote=None)

gss_raw = [gs_full, gs_red, gs_pca]
test_results_raw = score_on_test_set(gss_raw, [(X_test, y_test), (X_test_red, y_test_red), (X_test_pca, y_test_pca)])
%%time

gs_full_balanced = do_gridsearch_with_cv(clf_balanced, RANDOM_FOREST_PARAMS, X_train, y_train, kf, smote=None)

gs_red_balanced = do_gridsearch_with_cv(clf_balanced, RANDOM_FOREST_PARAMS, X_train_red, y_train_red, kf, smote=None)

gs_pca_balanced = do_gridsearch_with_cv(clf_balanced, RANDOM_FOREST_PARAMS, X_train_pca, y_train_pca, kf, smote=None)

gss_balanced_weights = [gs_full_balanced, gs_red_balanced, gs_pca_balanced]
test_results_balanced_weights = score_on_test_set(gss_balanced_weights, [(X_test, y_test), (X_test_red, y_test_red), (X_test_pca, y_test_pca)])
%%time

gs_full_smote = do_gridsearch_with_cv(clf_rf, RANDOM_FOREST_PARAMS, X_train, y_train, kf, smote=sm)

gs_red_smote = do_gridsearch_with_cv(clf_rf, RANDOM_FOREST_PARAMS, X_train_red, y_train_red, kf, smote=sm)

gs_pca_smote = do_gridsearch_with_cv(clf_rf, RANDOM_FOREST_PARAMS, X_train_pca, y_train_pca, kf, smote=sm)

gss_smote = [gs_full_smote, gs_red_smote, gs_pca_smote]
test_results_smote = score_on_test_set(gss_smote, [(X_test, y_test), (X_test_red, y_test_red), (X_test_pca, y_test_pca)])
dataset_strings = ["full dataset", "data set with reduced features", "dataset with first 10 principal components"]

method_strings = ["without any balancing", "using balanced class weights", "using SMOTE"]



result_strings = dict()

for ms, results in zip(method_strings, [test_results_raw, test_results_balanced_weights, test_results_smote]):

    for ds, res in zip(dataset_strings, results):

        string = "%.3f" % res + "     " + ds + " " + ms

        result_strings[string] = res

        2

result_strings = sorted(result_strings.items(), key=lambda kv: kv[1], reverse=True)

print("F1 score  dataset and method")

for k, _ in result_strings:

    print(k)
def get_color_with_opacity(color, opacity):

    return "rgba(" + color[4:-1] + ", %.2f)" % opacity



# partially based on https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html

def plot_learning_curve(estimator, title, X, y, cv=None, n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5)):

    """

    Generate a simple plot of the test and training learning curve.



    Parameters

    ----------

    estimator : object type that implements the "fit" and "predict" methods

        An object of that type which is cloned for each validation.



    title : string

        Title for the chart.



    X : array-like, shape (n_samples, n_features)

        Training vector, where n_samples is the number of samples and

        n_features is the number of features.



    y : array-like, shape (n_samples) or (n_samples, n_features), optional

        Target relative to X for classification or regression;

        None for unsupervised learning.



    cv : int, cross-validation generator or an iterable, optional

        Determines the cross-validation splitting strategy.

        Possible inputs for cv are:

          - None, to use the default 3-fold cross-validation,

          - integer, to specify the number of folds.

          - :term:`CV splitter`,

          - An iterable yielding (train, test) splits as arrays of indices.



        For integer/None inputs, if ``y`` is binary or multiclass,

        :class:`StratifiedKFold` used. If the estimator is not a classifier

        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.



        Refer :ref:`User Guide <cross_validation>` for the various

        cross-validators that can be used here.



    n_jobs : int or None, optional (default=None)

        Number of jobs to run in parallel.

        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.

        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`

        for more details.



    train_sizes : array-like, shape (n_ticks,), dtype float or int

        Relative or absolute numbers of training examples that will be used to

        generate the learning curve. If the dtype is float, it is regarded as a

        fraction of the maximum size of the training set (that is determined

        by the selected validation method), i.e. it has to be within (0, 1].

        Otherwise it is interpreted as absolute sizes of the training sets.

        Note that for classification the number of samples usually have to

        be big enough to contain at least one sample from each class.

        (default: np.linspace(0.1, 1.0, 5))

    """

    

    train_sizes, train_scores, test_scores = learning_curve(

        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, scoring="f1", random_state=SEED)

    train_scores_mean = np.mean(train_scores, axis=1)

    train_scores_std = np.std(train_scores, axis=1)

    test_scores_mean = np.mean(test_scores, axis=1)

    test_scores_std = np.std(test_scores, axis=1)

    

    trace1 = go.Scatter(

        x=train_sizes, 

        y=train_scores_mean - train_scores_std, 

        showlegend=False,

        mode="lines",

        name="",

        hoverlabel = dict(

            namelength=20

        ),

        line = dict(

            width = 0.1,

            color = get_color_with_opacity(colors[0], 0.4),

        ),

    )

    trace2 = go.Scatter(

        x=train_sizes, 

        y=train_scores_mean + train_scores_std, 

        showlegend=False,

        fill="tonexty",

        mode="lines",

        name="",

        hoverlabel = dict(

            namelength=20

        ),

        line = dict(

            width = 0.1,

            color = get_color_with_opacity(colors[0], 0.4),

        ),

    )

    trace3 = go.Scatter(

        x=train_sizes, 

        y=train_scores_mean, 

        showlegend=True,

        name="Train score",

        line = dict(

            color = colors[0],

        ),

    )

    

    trace4 = go.Scatter(

        x=train_sizes, 

        y=test_scores_mean - test_scores_std, 

        showlegend=False,

        mode="lines",

        name="",

        hoverlabel = dict(

            namelength=20

        ),

        line = dict(

            width = 0.1,

            color = get_color_with_opacity(colors[1], 0.4),

        ),

    )

    trace5 = go.Scatter(

        x=train_sizes, 

        y=test_scores_mean + test_scores_std, 

        showlegend=False,

        fill="tonexty",

        mode="lines",

        name="",

        hoverlabel = dict(

            namelength=20

        ),

        line = dict(

            width = 0.1,

            color = get_color_with_opacity(colors[1], 0.4),

        ),

    )

    trace6 = go.Scatter(

        x=train_sizes, 

        y=test_scores_mean, 

        showlegend=True,

        name="Test score",

        line = dict(

            color = colors[1],

        ),

    )

    

    data = [trace1, trace2, trace3, trace4, trace5, trace6]

    layout = go.Layout(

        title=title,

        autosize=True,

        yaxis=dict(

            title='F1 Score',

        ),

        xaxis=dict(

            title="#Training samples",

        ),

        legend=dict(

            x=0.8,

            y=0,

        ),

    )

    fig = go.Figure(data=data, layout=layout)

    return iplot(fig, filename=title)
def plot_feature_importance(feature_importance, title):

    trace1 = go.Bar(

        x=feature_importance[:, 0],

        y=feature_importance[:, 1],

        marker = dict(color = colors[0]),

        name='feature importance'

    )

    data = [trace1]

    layout = go.Layout(

        title=title,

        autosize=True,

        margin=go.layout.Margin(l=50, r=100, b=150),

        xaxis=dict(

            title='feature',

            tickangle=30

        ),

        yaxis=dict(

            title='feature importance',

            automargin=True,

        ),

    )

    fig = go.Figure(data=data, layout=layout)

    return iplot(fig, filename=title)
%%time

clf_lr = LogisticRegression(random_state=SEED)

gs_lr = do_gridsearch_with_cv(clf_lr, LOGISTIC_REGRESSION_PARAMS, X_train, y_train, kf, smote=sm)
print_best_grid_search_scores_with_params(gs_lr)
gs_lr_score = gs_lr.score(X_test, y_test)

y_pred_lr = gs_lr.predict(X_test)

cm_lr = confusion_matrix(y_test, y_pred_lr)

cm_lr = cm_lr.astype('float') / cm_lr.sum(axis=1)[:, np.newaxis] # normalize the confusion matrix
cm_df = pd.DataFrame(cm_lr.round(3), index=["true no churn", "true churn"], columns=["predicted no churn", "predicted churn"])

cm_df
plot_learning_curve(gs_lr.best_estimator_, "Learning Curve of Logistic Regression", X_train, y_train, cv=5)
%%time

clf_knn = KNeighborsClassifier()

gs_knn = do_gridsearch_with_cv(clf_knn, KNN_PARAMS, X_train, y_train, kf, smote=sm)
print_best_grid_search_scores_with_params(gs_knn)
gs_knn_score = gs_knn.score(X_test, y_test)

y_pred_knn = gs_knn.predict(X_test)

cm_knn = confusion_matrix(y_test, y_pred_knn)

cm_knn = cm_knn.astype('float') / cm_knn.sum(axis=1)[:, np.newaxis] # normalize the confusion matrix
cm_df = pd.DataFrame(cm_knn.round(3), index=["true no churn", "true churn"], columns=["predicted no churn", "predicted churn"])

cm_df
plot_learning_curve(gs_knn.best_estimator_, "Learning Curve of KNN", X_train, y_train, cv=5)
clf_knn_uni = KNeighborsClassifier()

gs_knn_uniform = do_gridsearch_with_cv(clf_knn_uni, KNN_PARAMS_UNIFORM, X_train, y_train, kf, smote=sm)
print_best_grid_search_scores_with_params(gs_knn_uniform, 1)
plot_learning_curve(gs_knn_uniform.best_estimator_, "Learning Curve of KNN with uniform weights", X_train, y_train, cv=5)
%%time

clf_svm = svm.SVC(random_state=SEED, probability=True)

gs_svm = do_gridsearch_with_cv(clf_svm, SVM_PARAMS, X_train, y_train, kf, smote=sm)
print_best_grid_search_scores_with_params(gs_svm)
gs_svm_score = gs_svm.score(X_test, y_test)

y_pred_svm = gs_svm.predict(X_test)

cm_svm = confusion_matrix(y_test, y_pred_svm)

cm_svm = cm_svm.astype('float') / cm_svm.sum(axis=1)[:, np.newaxis] # normalize the confusion matrix
pd.DataFrame(cm_svm.round(3), index=["true no churn", "true churn"], columns=["predicted no churn", "predicted churn"])
plot_learning_curve(gs_svm.best_estimator_, "Learning Curve of SVM", X_train, y_train, cv=5)
%%time

clf_dt = DecisionTreeClassifier(random_state=SEED)

gs_dt = do_gridsearch_with_cv(clf_dt, DECISION_TREE_PARAMS, X_train, y_train, kf, smote=sm)
print_best_grid_search_scores_with_params(gs_dt)
gs_dt_score = gs_dt.score(X_test, y_test)

y_pred_dt = gs_dt.predict(X_test)

cm_dt = confusion_matrix(y_test, y_pred_dt)

cm_dt = cm_dt.astype('float') / cm_dt.sum(axis=1)[:, np.newaxis] # normalize the confusion matrix
cm_df = pd.DataFrame(cm_dt.round(3), index=["true no churn", "true churn"], columns=["predicted no churn", "predicted churn"])

cm_df
feature_importance = np.array(sorted(zip(X_train.columns, gs_dt.best_estimator_.named_steps['clf'].feature_importances_), key=lambda x: x[1], reverse=True))

plot_feature_importance(feature_importance, "Feature importance in the decision tree")
plot_learning_curve(gs_dt.best_estimator_, "Learning Curve of the Decision Tree", X_train, y_train, cv=5)
%%time

clf_rf = RandomForestClassifier(random_state=SEED)

gs_rf = do_gridsearch_with_cv(clf_rf, RANDOM_FOREST_PARAMS, X_train, y_train, kf, smote=sm)
print_best_grid_search_scores_with_params(gs_rf)
gs_rf_score = gs_rf.score(X_test, y_test)

y_pred_rf = gs_rf.predict(X_test)

cm_rf = confusion_matrix(y_test, y_pred_rf)

cm_rf = cm_rf.astype('float') / cm_rf.sum(axis=1)[:, np.newaxis] # normalize the confusion matrix
cm_df = pd.DataFrame(cm_rf.round(3), index=["true no churn", "true churn"], columns=["predicted no churn", "predicted churn"])

cm_df
feature_importance_rf = np.array(sorted(zip(X_train.columns, gs_rf.best_estimator_.named_steps['clf'].feature_importances_), key=lambda x: x[1], reverse=True))

plot_feature_importance(feature_importance_rf, "Feature importance in the Random Forest")
plot_learning_curve(gs_dt.best_estimator_, "Learning Curve of the Random Forest", X_train, y_train, cv=5)
# code partially from https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html

def plot_roc_curve(classifiers, legend, title, X_test, y_test):

    trace1 = go.Scatter(

        x=[0, 1], 

        y=[0, 1], 

        showlegend=False,

        mode="lines",

        name="",

        line = dict(

            color = colors[0],

        ),

    )

    

    data = [trace1]

    aucs = []

    for clf, string, c in zip(classifiers, legend, colors[1:]):

        y_test_roc = np.array([([0, 1] if y else [1, 0]) for y in y_test])

        y_score = clf.predict_proba(X_test)

        

        # Compute ROC curve and ROC area for each class

        fpr = dict()

        tpr = dict()

        roc_auc = dict()

        for i in range(2):

            fpr[i], tpr[i], _ = roc_curve(y_test_roc[:, i], y_score[:, i])

            roc_auc[i] = auc(fpr[i], tpr[i])



        # Compute micro-average ROC curve and ROC area

        fpr["micro"], tpr["micro"], _ = roc_curve(y_test_roc.ravel(), y_score.ravel())

        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        aucs.append(roc_auc['micro'])



        trace = go.Scatter(

            x=fpr['micro'], 

            y=tpr['micro'], 

            showlegend=True,

            mode="lines",

            name=string + " (area = %0.2f)" % roc_auc['micro'],

            hoverlabel = dict(

                namelength=30

            ),

            line = dict(

                color = c,

            ),

        )

        data.append(trace)



    layout = go.Layout(

        title=title,

        autosize=False,

        width=550,

        height=550,

        yaxis=dict(

            title='True Positive Rate',

        ),

        xaxis=dict(

            title="False Positive Rate",

        ),

        legend=dict(

            x=0.4,

            y=0.06,

        ),

    )

    fig = go.Figure(data=data, layout=layout)

    return aucs, iplot(fig, filename=title)
classifiers = [gs_lr, gs_knn, gs_svm, gs_dt, gs_rf]

classifier_names = ["Logistic Regression", "KNN", "SVM", "Decision Tree", "Random Forest"]

auc_scores, roc_plot = plot_roc_curve(classifiers, classifier_names, "ROC curve", X_test, y_test)

roc_plot
accs = []

recalls = []

precision = []

results_table = pd.DataFrame(columns=["accuracy", "precision", "recall", "f1", "auc"])

for (i, clf), name, auc in zip(enumerate(classifiers), classifier_names, auc_scores):

    y_pred = clf.predict(X_test)

    row = []

    row.append(accuracy_score(y_test, y_pred))

    row.append(precision_score(y_test, y_pred))

    row.append(recall_score(y_test, y_pred))

    row.append(f1_score(y_test, y_pred))

    row.append(auc)

    row = ["%.3f" % r for r in row]

    results_table.loc[name] = row
results_table