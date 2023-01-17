%matplotlib inline

# data manipulation libraries

import pandas as pd

import numpy as np



from time import time



# Graphs libraries

import matplotlib

import matplotlib.pyplot as plt

import matplotlib.patches as patches

plt.style.use('seaborn-white')

import seaborn as sns



import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.tools as tls

import plotly.figure_factory as ff

from plotly import tools



# ML Libraries

from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, roc_curve, auc

from sklearn.preprocessing import MinMaxScaler, LabelEncoder

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import RandomizedSearchCV



import lime

import lime.lime_tabular

import shap



# Design libraries

from IPython.display import Markdown, display

import warnings

warnings.filterwarnings("ignore")
data = pd.read_csv('../input/database.csv', na_values=['Unknown', ' '])
# Save original data just in case

data_orig = data.copy()
# Keep only solved crimes

data = data[data['Crime Solved'] == 'Yes']



# Drop useless columns

cols_to_drop = ['Record ID', 'Agency Code', 'Perpetrator Ethnicity', 'Perpetrator Race', 'Perpetrator Age', 'Incident','Crime Solved']

data.drop(columns=cols_to_drop, inplace=True)



# Get numerical and categorical columns

Y_columns = ['Perpetrator Sex']

cat_columns = []

num_columns = []



for col in data.columns.values:

    if data[col].dtypes == 'int64':

        num_columns += [col]

    else:

        cat_columns += [col]



# Get median values 

median_val = pd.Series()

for col in num_columns:

    median_val[col] = data[col].median()



# Handle missing data

for col in data:

    if col in median_val.index.values:

        data[col] = data[col].fillna(median_val[col])

    else:

        val = data[col].value_counts().sort_values(ascending=False).index[0]

        data[col] = data[col].fillna(val)



# Correct the problem about the Victim Age

data['Victim Age'] = np.where(data['Victim Age'] == 998, np.median(data[data['Victim Age'] <= 100]['Victim Age']), data['Victim Age'])
data.head()
categorical_features = cat_columns

categorical_features_idx = [np.where(data.columns.values == col)[0][0] for col in categorical_features]



data_encoded = data.copy()



categorical_names = {}

encoders = {}



# Use Label Encoder for categorical columns (including target column)

for feature in categorical_features:

    le = LabelEncoder()

    le.fit(data_encoded[feature])

    

    data_encoded[feature] = le.transform(data_encoded[feature])

    

    categorical_names[feature] = le.classes_

    encoders[feature] = le





numerical_features = [c for c in data.columns.values if c not in categorical_features]



for feature in numerical_features:

    val = data_encoded[feature].values[:, np.newaxis]

    mms = MinMaxScaler().fit(val)

    data_encoded[feature] = mms.transform(val)

    encoders[feature] = mms

    

data_encoded = data_encoded.astype(float)
data_encoded.head()
def decode_dataset(data, encoders, numerical_features, categorical_features):

    df = data.copy()

    for feat in df.columns.values:

        if feat in numerical_features:

            df[feat] = encoders[feat].inverse_transform(np.array(df[feat]).reshape(-1, 1))

    for feat in categorical_features:

        df[feat] = encoders[feat].inverse_transform(df[feat].astype(int))

    return df
decode_dataset(data_encoded, encoders=encoders, numerical_features=numerical_features, categorical_features=categorical_features).head()
X = data_encoded.drop(columns='Perpetrator Sex', axis=1)

y = data_encoded['Perpetrator Sex']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
display(Markdown('#### Train dataset shape :'))

print(X_train.shape)

display(Markdown('#### Test dataset shape :'))

print(X_test.shape)
y_test.value_counts()
clf = RandomForestClassifier(random_state=4242)
param_dist = {"n_estimators":[10,100],

              "max_depth": [3, 10],

              "max_features": [2,len(X.columns)],

              "min_samples_split":  [2,20],

              "bootstrap": [True, False],

              "criterion": ["gini", "entropy"]}



n_iter_search = 20

random_search = RandomizedSearchCV(clf, param_distributions=param_dist, n_iter=n_iter_search, cv=5, verbose=8)



start = time()

# random_search.fit(X_train, y_train)

print("RandomizedSearchCV took %.2f seconds for %d candidates parameter settings." % ((time() - start), n_iter_search))

# print(random_search.cv_results_)
clf.fit(X_train, y_train)
def get_model_performance(X_test, y_true, y_pred, probs):

    accuracy = accuracy_score(y_true, y_pred)

    matrix = confusion_matrix(y_true, y_pred)

    f1 = f1_score(y_true, y_pred)

    preds = probs[:, 1]

    fpr, tpr, threshold = roc_curve(y_true, preds)

    roc_auc = auc(fpr, tpr)



    return accuracy, matrix, f1, fpr, tpr, roc_auc



def plot_model_performance(model, X_test, y_true):

    y_pred = model.predict(X_test)

    probs = model.predict_proba(X_test)

    accuracy, matrix, f1, fpr, tpr, roc_auc = get_model_performance(X_test, y_true, y_pred, probs)



    display(Markdown('#### Accuracy of the model :'))

    print(accuracy)

    display(Markdown('#### F1 score of the model :'))

    print(f1)



    fig = plt.figure(figsize=(15, 6))

    ax = fig.add_subplot(1, 2, 1)

    sns.heatmap(matrix, annot=True, cmap='Blues', fmt='g')

    plt.title('Confusion Matrix')



    ax = fig.add_subplot(1, 2, 2)

    lw = 2

    plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)

    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')

    plt.xlim([0.0, 1.0])

    plt.ylim([0.0, 1.05])

    plt.xlabel('False Positive Rate')

    plt.ylabel('True Positive Rate')

    plt.title('Receiver Operating Characteristic curve')

    plt.legend(loc="lower right")
plot_model_performance(clf, X_test, y_test)
clf.feature_importances_

feature_imp = pd.Series(clf.feature_importances_, index=X_train.columns)

feature_imp.sort_values(ascending=False, inplace=True)
def get_trace(series, color='#2980b9'):

    series.sort_values(ascending=True, inplace=True)

    x, y = series.values, series.index

    trace = go.Bar(x=x, y=y, marker=dict(color=color), opacity=0.9, orientation='h')

    return trace



trace = get_trace(feature_imp)

layout = go.Layout(barmode='group', title='Feature Importance', yaxis=go.layout.YAxis(automargin=True))

fig = go.Figure([trace], layout=layout)

py.iplot(fig)
categorical_features = ['Agency Name', 'Agency Type', 'City', 'State', 'Month',

                        'Crime Type', 'Victim Sex', 'Victim Race', 'Victim Ethnicity',

                        'Relationship', 'Weapon', 'Record Source']

categorical_features_idx = [X.columns.values.tolist().index(col) for col in categorical_features]

categorical_names_LIME = {}



for feature, idx in zip(categorical_features, categorical_features_idx):

    categorical_names_LIME[idx] = categorical_names[feature]

    

# Reverse the MinMaxScaler to get the original values back.

for feat in X.columns.values:

    if feat in numerical_features:

        X[feat] = encoders[feat].inverse_transform(np.array(X[feat]).reshape(-1, 1))
# Initiate the LIME explainer

explainer = lime.lime_tabular.LimeTabularExplainer(X.values,

                                                   feature_names=X.columns.values,

                                                   class_names=['Female', 'Male'],

                                                   categorical_features=categorical_features_idx, 

                                                   categorical_names=categorical_names_LIME)
def get_trace(series, color='#2980b9'):

    series.sort_values(ascending=True, inplace=True)

    x, y = series.values, series.index

    trace = go.Bar(x=x, y=y, marker=dict(color=color), opacity=0.9, orientation='h')

    return trace



trace = get_trace(feature_imp)

layout = go.Layout(barmode='group', title='Feature Importance', yaxis=go.layout.YAxis(automargin=True))

fig = go.Figure([trace], layout=layout)

py.iplot(fig)
def plot_lime_importance(exp):

    importance = exp.as_list()

    importance.reverse()



    colors, x, y = list(), list(), list()



    for feat, val in importance:

        if val < 0:

            colors.append('#3498db')

        else:

            colors.append('#e67e22')

        x.append(np.abs(val))

        y.append(feat)



    trace = go.Bar(x=x, y=y, marker=dict(color=colors)

                   , opacity=0.9, orientation='h')

    layout = go.Layout(barmode='group',

                       title='Importance for the current prediction',

                       yaxis=go.layout.YAxis(automargin=True))

    fig = go.Figure([trace], layout=layout)

    py.iplot(fig)

    

def plot_prediction_importance(data, idx, model):

    row = pd.DataFrame(data=[data.iloc[idx,:]], columns=data.columns.values)

 

    exp = explainer.explain_instance(row.values[0], model.predict_proba)



    display(row)

    # Plot function from LIME

    exp.show_in_notebook(show_table=True, show_all=False)

    # Custom plot function

    plot_lime_importance(exp)
plot_prediction_importance(X_test, 6, clf)
def compute_mean(dataset):

    return dataset.mean()



def compute_mean_abs(dataset):

    return dataset.abs().mean()



t0 = time()

print('start : %0.4fs'%((time() - t0)))

    

importance = pd.DataFrame()



# I will just take a sample of set because it's a long operation on all the dataset.

test_lime = X_test.iloc[0:1000,:]



for row in test_lime.values:

    exp = explainer.explain_instance(row, clf.predict_proba)

    feats = [x[0] for x in exp.as_list()]

    val = [x[1] for x in exp.as_list()]

    

    for i in range(0,len(feats)):

        for feat in test_lime.columns.values:

            if(feat in feats[i]):

                feats[i] = feat

    

    exp_df = pd.DataFrame(data=[val], columns=feats)

    importance = importance.append(exp_df)



print('end : %0.4fs'%((time() - t0)))
mean_imp = compute_mean(importance)

# Stock importance to compare it with other importance after

lime_importance = compute_mean_abs(importance)
trace = get_trace(mean_imp.abs())

layout = go.Layout(barmode='group', title='LIME : Mean Importance', yaxis=go.layout.YAxis(automargin=True))

fig = go.Figure([trace], layout=layout)

py.iplot(fig)



trace = get_trace(lime_importance)

layout = go.Layout(barmode='group', title='LIME : Mean absolute Importance', yaxis=go.layout.YAxis(automargin=True))

fig = go.Figure([trace], layout=layout)

py.iplot(fig)
t0 = time()

print('start : %0.4fs'%((time() - t0)))



# Create object that can calculate shap values

explainer = shap.TreeExplainer(clf)

print('explainer end : %0.4fs'%((time() - t0)))



# Calculate Shap values

test_shap = X_test.iloc[0:1000,:].copy()

# shap_values = explainer.shap_values(test)

shap_values = explainer.shap_values(test_shap)

print('shap values end : %0.4fs'%((time() - t0)))
def plot_shap_explain(data, idx, model, explainer, shap_values, categorical_features=None, encoders=None):

    row = data.iloc[idx,:] 

    display(pd.DataFrame([row.values], columns=row.index))

    proba = model.predict_proba([row])[0]

    display(Markdown("Probability of having an Income <= 50K : **%0.2f**"%proba[0]))

    display(Markdown("Probability of having an Income > 50K : **%0.2f**"%proba[1]))

    

    if categorical_features != None:

        for feature in categorical_features:

            row[feature] = encoders[feature].inverse_transform([int(row[feature])])[0]

    

    display(Markdown("#### Explaination based on the 0 label (Income <= 50K)"))

    display(shap.force_plot(explainer.expected_value[0], shap_values[0][idx,:], row))

    display(Markdown("#### Explaination based on the 1 label (Income > 50K)"))

    display(shap.force_plot(explainer.expected_value[1], shap_values[1][idx,:], row))



# Take a random index

random_index = np.random.randint(0, len(test_shap))



shap.initjs()

plot_shap_explain(test_shap, random_index, clf, explainer, shap_values, categorical_features, encoders)
shap.summary_plot(shap_values[1], test_shap, test_shap.columns.values, plot_type='bar')

shap.summary_plot(shap_values[1], test_shap, test_shap.columns.values, plot_type='dot')
shap_df = pd.DataFrame(data=shap_values[1], columns=test_shap.columns)

shap_importance = compute_mean_abs(shap_df)
def get_trace(series, color='#2980b9'):

    series.sort_values(ascending=True, inplace=True)

    x, y = series.values, series.index

    trace = go.Bar(x=x, y=y, marker=dict(color=color), opacity=0.9, orientation='h')

    return trace



trace1 = get_trace(feature_imp)

trace2 = get_trace(lime_importance)

trace3 = get_trace(shap_importance)



fig = tools.make_subplots(rows=1, cols=3, subplot_titles=('Feature importance', 'LIME','SHAP'))



fig.append_trace(trace1, 1, 1)

fig.append_trace(trace2, 1, 2)

fig.append_trace(trace3, 1, 3)

fig['layout'].update(title='Comparison of feature importance, LIME importance and SHAP importance',

                     showlegend=False,

                     yaxis=go.layout.YAxis(automargin=True),

                     barmode='group')

py.iplot(fig)