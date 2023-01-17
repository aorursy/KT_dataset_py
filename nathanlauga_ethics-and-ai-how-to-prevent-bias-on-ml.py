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



# Libraries to study

from aif360.datasets import StandardDataset

from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric

from aif360.algorithms.preprocessing import LFR, Reweighing

from aif360.algorithms.inprocessing import AdversarialDebiasing, PrejudiceRemover

from aif360.algorithms.postprocessing import CalibratedEqOddsPostprocessing, EqOddsPostprocessing, RejectOptionClassification



# ML libraries

from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, roc_curve, auc

from sklearn.preprocessing import MinMaxScaler, LabelEncoder

from sklearn.ensemble import RandomForestClassifier

import tensorflow as tf



# Design libraries

from IPython.display import Markdown, display

import warnings

warnings.filterwarnings("ignore")
data = pd.read_csv('../input/database.csv', na_values=['Unknown', ' '])
data.shape
data.head().T
data.columns
cols_to_drop = ['Record ID', 'Agency Code', 'Perpetrator Ethnicity']

data_orig = data.copy()

data.drop(columns=cols_to_drop, inplace=True)

cols_to_drop = []
def print_missing_values(data):

    data_null = pd.DataFrame(len(data) - data.notnull().sum(), columns = ['Count'])

    data_null = data_null[data_null['Count'] > 0].sort_values(by='Count', ascending=False)

    data_null = data_null/len(data)*100



    trace = go.Bar(x=data_null.index, y=data_null['Count'], marker=dict(color='#c0392b'),

              name = 'At least one missing value', opacity=0.9)

    layout = go.Layout(barmode='group', title='Column with missing values in the dataset', showlegend=True,

                   legend=dict(orientation="h"), yaxis=dict(title='Percentage of the dataset'))

    fig = go.Figure([trace], layout=layout)

    py.iplot(fig)
print('Number total of rows : '+str(data.shape[0]))

print_missing_values(data)
data['Crime Solved'].value_counts()
data_orig = data.copy()

# data = data_orig



data = data[data['Crime Solved'] == 'Yes']

cols_to_drop += ['Crime Solved']
data['Perpetrator Age category'] = np.where(data['Perpetrator Age'] > 64, 'Elder', np.where(data['Perpetrator Age'] < 25, 'Young', 'Adult'))

# data['Victim Age category'] = np.where(data['Victim Age'] > 64, 'Elder', np.where(data['Victim Age'] < 25, 'Young', 'Adult'))
Y_columns = ['Perpetrator Sex', 'Perpetrator Race', 'Perpetrator Age category']

ignore_columns = ['Crime Solved']

cat_columns = []

num_columns = []



for col in data.columns.values:

    if col in Y_columns+ignore_columns:

        continue

    elif data[col].dtypes == 'int64':

        num_columns += [col]

    else:

        cat_columns += [col]

median_val = pd.Series()

for col in num_columns:

    if col not in cols_to_drop:

        median_val[col] = data[col].median()
median_val
def handle_missing_values(data, median_val):

    df = data.copy()

    for col in df:

        if col in median_val.index.values:

            df[col] = df[col].fillna(median_val[col])

        else:

            df[col] = df[col].fillna("Missing value")

    

    return df
data = handle_missing_values(data, median_val)
def target_distribution(y_var, data):

    val = data[y_var]



    plt.style.use('seaborn-whitegrid')

    plt.rcParams.update({'font.size': 13})

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))



    cnt = val.value_counts().sort_values(ascending=True)

    labels = cnt.index.values



    sizes = cnt.values

    colors = sns.color_palette("PuBu", len(labels))



    #------------COUNT-----------------------

    ax1.barh(cnt.index.values, cnt.values, color=colors)

    ax1.set_title('Count plot of '+y_var)



    #------------PERCENTAGE-------------------

    ax2.pie(sizes, labels=labels, colors=colors,autopct='%1.0f%%', shadow=True, startangle=130)

    ax2.axis('equal')

    ax2.set_title('Distribution of '+y_var)

    plt.show()
var = 'Perpetrator Race'

target_distribution(y_var=var, data=data)
var = 'Perpetrator Sex'

target_distribution(y_var=var, data=data)
var = 'Perpetrator Age category'

target_distribution(y_var=var, data=data)
data['Frequency'] = 1

freq_target = data[['Perpetrator Sex', 'Perpetrator Race', 'Perpetrator Age category', 'Frequency']]

del data['Frequency']

freq_target = freq_target.groupby(by=['Perpetrator Sex', 'Perpetrator Race', 'Perpetrator Age category']).count() / len(data)

print(freq_target.sort_values(by='Frequency', ascending=False))
def plot_histo(data, col, Y_columns):

    df = data.copy()

    fig, axs = plt.subplots(1,2,figsize=(20,6))

    

    for i in range(0,2):

        cnt = []; y_col = Y_columns[i]

        Y_values = df[y_col].dropna().drop_duplicates().values

        for val in Y_values:

            cnt += [df[df[y_col] == val][col].values]

        bins = df[col].nunique()



        axs[i].hist(cnt, bins=bins, stacked=True)

        axs[i].legend(Y_values,loc='upper right')

        axs[i].set_title("Histogram of the "+col+" column by "+y_col)



    plt.show()
plot_histo(data, col='Year',Y_columns=Y_columns)
plot_histo(data, col='Incident',Y_columns=Y_columns)
cols_to_drop += ['Incident']
plot_histo(data, col='Victim Age',Y_columns=Y_columns)
data[data['Victim Age'] > 100]['Victim Age'].value_counts()
data['Victim Age'] = np.where(data['Victim Age'] == 998, np.median(data[data['Victim Age'] <= 100]['Victim Age']), data['Victim Age'])
plot_histo(data, col='Victim Age',Y_columns=Y_columns)
plot_histo(data, col='Victim Count',Y_columns=Y_columns)
plot_histo(data, col='Perpetrator Count',Y_columns=Y_columns)
cat_columns
def plot_bar(data, col, Y_columns, max_cat=10):

    df = data.copy()

    

    fig, axs = plt.subplots(1,2,figsize=(20,6))

    cat_val = df[col].value_counts()[0:max_cat].index.values

    df = df[df[col].isin(cat_val)]



    for i in range(0,2):

        y_col = Y_columns[i]

        Y_values = df[y_col].dropna().drop_duplicates().values

        for val in Y_values:

            cnt = df[df[y_col] == val][col].value_counts().sort_index()

            axs[i].barh(cnt.index.values, cnt.values)

        axs[i].legend(Y_values,loc='upper right')

        axs[i].set_title("Bar plot of the "+col+" column by "+y_col)



    plt.show()
plot_bar(data, col='Agency Name',Y_columns=Y_columns)
plot_bar(data, col='Agency Type',Y_columns=Y_columns)
plot_bar(data, col='City',Y_columns=Y_columns)
plot_bar(data, col='State',Y_columns=Y_columns)
plot_bar(data, col='Month',Y_columns=Y_columns, max_cat=12)
plot_bar(data, col='Crime Type',Y_columns=Y_columns)
plot_bar(data, col='Victim Sex',Y_columns=Y_columns)
plot_bar(data, col='Victim Race',Y_columns=Y_columns)
plot_bar(data, col='Victim Ethnicity',Y_columns=Y_columns)
plot_bar(data, col='Relationship', Y_columns=Y_columns)
plot_bar(data, col='Weapon', Y_columns=Y_columns)
plot_bar(data, col='Record Source', Y_columns=Y_columns)
data.drop(cols_to_drop, axis=1, inplace=True)
categorical_features = cat_columns + ['Perpetrator Sex', 'Perpetrator Race', 'Perpetrator Age category']

# categorical_features = categorical_features 

categorical_features_idx = [np.where(data.columns.values == col)[0][0] for col in categorical_features]



del cat_columns
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



del num_columns
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
data_perp_sex = data_encoded.drop(['Perpetrator Race','Perpetrator Age category','Perpetrator Age'], axis=1)
privileged_sex = np.where(categorical_names['Victim Sex'] == 'Male')[0]

privileged_race = np.where(categorical_names['Victim Race'] == 'White')[0]
data_orig_sex = StandardDataset(data_perp_sex, 

                               label_name='Perpetrator Sex', 

                               favorable_classes=[1], 

                               protected_attribute_names=['Victim Sex', 'Victim Race'], 

                               privileged_classes=[privileged_sex, privileged_race])
def meta_data(dataset):

    # print out some labels, names, etc.

    display(Markdown("#### Dataset shape"))

    print(dataset.features.shape)

    display(Markdown("#### Favorable and unfavorable labels"))

    print(dataset.favorable_label, dataset.unfavorable_label)

    display(Markdown("#### Protected attribute names"))

    print(dataset.protected_attribute_names)

    display(Markdown("#### Privileged and unprivileged protected attribute values"))

    print(dataset.privileged_protected_attributes, dataset.unprivileged_protected_attributes)

    display(Markdown("#### Dataset feature names"))

    print(dataset.feature_names)
meta_data(data_orig_sex)
np.random.seed(42)



data_orig_sex_train, data_orig_sex_test = data_orig_sex.split([0.7], shuffle=True)



display(Markdown("#### Train Dataset shape"))

print("Perpetrator Sex :",data_orig_sex_train.features.shape)

display(Markdown("#### Test Dataset shape"))

print("Perpetrator Sex :",data_orig_sex_test.features.shape)
# Train and save the models

rf_orig_sex = RandomForestClassifier().fit(data_orig_sex_train.features, 

                     data_orig_sex_train.labels.ravel(), 

                     sample_weight=data_orig_sex_train.instance_weights)
X_test_sex = data_orig_sex_test.features

y_test_sex = data_orig_sex_test.labels.ravel()
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
plot_model_performance(rf_orig_sex, data_orig_sex_test.features, y_test_sex)
# This DataFrame is created to stock differents models and fair metrics that we produce in this notebook

algo_metrics = pd.DataFrame(columns=['model', 'fair_metrics', 'prediction', 'probs'])



def add_to_df_algo_metrics(algo_metrics, model, fair_metrics, preds, probs, name):

    return algo_metrics.append(pd.DataFrame(data=[[model, fair_metrics, preds, probs]], columns=['model', 'fair_metrics', 'prediction', 'probs'], index=[name]))
def fair_metrics(dataset, pred, pred_is_dataset=False):

    if pred_is_dataset:

        dataset_pred = pred

    else:

        dataset_pred = dataset.copy()

        dataset_pred.labels = pred

    

    cols = ['statistical_parity_difference', 'equal_opportunity_difference', 'average_abs_odds_difference',  'disparate_impact', 'theil_index']

    obj_fairness = [[0,0,0,1,0]]

    

    fair_metrics = pd.DataFrame(data=obj_fairness, index=['objective'], columns=cols)

    

    for attr in dataset_pred.protected_attribute_names:

        idx = dataset_pred.protected_attribute_names.index(attr)

        privileged_groups =  [{attr:dataset_pred.privileged_protected_attributes[idx][0]}] 

        unprivileged_groups = [{attr:dataset_pred.unprivileged_protected_attributes[idx][0]}] 

        

        classified_metric = ClassificationMetric(dataset, 

                                                     dataset_pred,

                                                     unprivileged_groups=unprivileged_groups,

                                                     privileged_groups=privileged_groups)



        metric_pred = BinaryLabelDatasetMetric(dataset_pred,

                                                     unprivileged_groups=unprivileged_groups,

                                                     privileged_groups=privileged_groups)



        acc = classified_metric.accuracy()



        row = pd.DataFrame([[metric_pred.mean_difference(),

                                classified_metric.equal_opportunity_difference(),

                                classified_metric.average_abs_odds_difference(),

                                metric_pred.disparate_impact(),

                                classified_metric.theil_index()]],

                           columns  = cols,

                           index = [attr]

                          )

        fair_metrics = fair_metrics.append(row)    

    

    fair_metrics = fair_metrics.replace([-np.inf, np.inf], 2)

        

    return fair_metrics



def plot_fair_metrics(fair_metrics):

    fig, ax = plt.subplots(figsize=(20,4), ncols=5, nrows=1)



    plt.subplots_adjust(

        left    =  0.125, 

        bottom  =  0.1, 

        right   =  0.9, 

        top     =  0.9, 

        wspace  =  .5, 

        hspace  =  1.1

    )



    y_title_margin = 1.2



    plt.suptitle("Fairness metrics", y = 1.09, fontsize=20)

    sns.set(style="dark")



    cols = fair_metrics.columns.values

    obj = fair_metrics.loc['objective']

    size_rect = [0.2,0.2,0.2,0.4,0.25]

    rect = [-0.1,-0.1,-0.1,0.8,0]

    bottom = [-1,-1,-1,0,0]

    top = [1,1,1,2,1]

    bound = [[-0.1,0.1],[-0.1,0.1],[-0.1,0.1],[0.8,1.2],[0,0.25]]



    display(Markdown("### Check bias metrics :"))

    display(Markdown("A model can be considered bias if just one of these five metrics show that this model is biased."))

    for attr in fair_metrics.index[1:len(fair_metrics)].values:

        display(Markdown("#### For the %s attribute :"%attr))

        check = [bound[i][0] < fair_metrics.loc[attr][i] < bound[i][1] for i in range(0,5)]

        display(Markdown("With default thresholds, bias against unprivileged group detected in **%d** out of 5 metrics"%(5 - sum(check))))



    for i in range(0,5):

        plt.subplot(1, 5, i+1)

        ax = sns.barplot(x=fair_metrics.index[1:len(fair_metrics)], y=fair_metrics.iloc[1:len(fair_metrics)][cols[i]])

        

        for j in range(0,len(fair_metrics)-1):

            a, val = ax.patches[j], fair_metrics.iloc[j+1][cols[i]]

            marg = -0.2 if val < 0 else 0.1

            ax.text(a.get_x()+a.get_width()/5, a.get_y()+a.get_height()+marg, round(val, 3), fontsize=15,color='black')



        plt.ylim(bottom[i], top[i])

        plt.setp(ax.patches, linewidth=0)

        ax.add_patch(patches.Rectangle((-5,rect[i]), 10, size_rect[i], alpha=0.3, facecolor="green", linewidth=1, linestyle='solid'))

        plt.axhline(obj[i], color='black', alpha=0.3)

        plt.title(cols[i])

        ax.set_ylabel('')    

        ax.set_xlabel('')
def get_fair_metrics_and_plot(data, model, plot=True, model_aif=False):

    pred = model.predict(data).labels if model_aif else model.predict(data.features)

    # fair_metrics function available in the metrics.py file

    fair = fair_metrics(data, pred)



    if plot:

        # plot_fair_metrics function available in the visualisations.py file

        # The visualisation of this function is inspired by the dashboard on the demo of IBM aif360 

        plot_fair_metrics(fair)

        display(fair)

    

    return fair
display(Markdown('### Bias metrics for the Sex model'))

fair = get_fair_metrics_and_plot(data_orig_sex_test, rf_orig_sex)
data_orig_test = data_orig_sex_test

data_orig_train = data_orig_sex_train

rf = rf_orig_sex



probs = rf.predict_proba(data_orig_test.features)

preds = rf.predict(data_orig_test.features)

algo_metrics = add_to_df_algo_metrics(algo_metrics, rf, fair, preds, probs, 'Origin')
def get_attributes(data, selected_attr=None):

    unprivileged_groups = []

    privileged_groups = []

    if selected_attr == None:

        selected_attr = data.protected_attribute_names

    

    for attr in selected_attr:

            idx = data.protected_attribute_names.index(attr)

            privileged_groups.append({attr:data.privileged_protected_attributes[idx]}) 

            unprivileged_groups.append({attr:data.unprivileged_protected_attributes[idx]}) 



    return privileged_groups, unprivileged_groups
privileged_groups, unprivileged_groups = get_attributes(data_orig_train, selected_attr=['Victim Race'])

t0 = time()



LFR_model = LFR(unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups, k=1, verbose=0)

# LFR.fit(data_orig_train)

data_transf_train = LFR_model.fit_transform(data_orig_train)



# Train and save the model

rf_transf = RandomForestClassifier().fit(data_transf_train.features, 

                     data_transf_train.labels.ravel(), 

                     sample_weight=data_transf_train.instance_weights)



data_transf_test = LFR_model.transform(data_orig_test)

fair = get_fair_metrics_and_plot(data_transf_test, rf_transf, plot=False)

probs = rf_transf.predict_proba(data_orig_test.features)

preds = rf_transf.predict(data_orig_test.features)



algo_metrics = add_to_df_algo_metrics(algo_metrics, rf_transf, fair, preds, probs, 'LFR')

print('time elapsed : %.2fs'%(time()-t0))
privileged_groups, unprivileged_groups = get_attributes(data_orig_train, selected_attr=['Victim Race'])

t0 = time()



RW = Reweighing(unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)

# RW.fit(data_orig_train)

data_transf_train = RW.fit_transform(data_orig_train)



# Train and save the model

rf_transf = RandomForestClassifier().fit(data_transf_train.features, 

                     data_transf_train.labels.ravel(), 

                     sample_weight=data_transf_train.instance_weights)



data_transf_test = RW.transform(data_orig_test)

fair = get_fair_metrics_and_plot(data_orig_test, rf_transf, plot=False)

probs = rf_transf.predict_proba(data_orig_test.features)

preds = rf_transf.predict(data_orig_test.features)



algo_metrics = add_to_df_algo_metrics(algo_metrics, rf_transf, fair, preds, probs, 'Reweighing')

print('time elapsed : %.2fs'%(time()-t0))
privileged_groups, unprivileged_groups = get_attributes(data_orig_train, selected_attr=['Victim Race'])

t0 = time()



# sess.close()

# tf.reset_default_graph()

sess = tf.Session()



debiased_model = AdversarialDebiasing(privileged_groups = privileged_groups,

                          unprivileged_groups = unprivileged_groups,

                          scope_name='debiased_classifier',

                          num_epochs=10,

                          debias=True,

                          sess=sess)



debiased_model.fit(data_orig_train)



fair = get_fair_metrics_and_plot(data_orig_test, debiased_model, plot=False, model_aif=True)

data_pred = debiased_model.predict(data_orig_test)



algo_metrics = add_to_df_algo_metrics(algo_metrics, debiased_model, fair, data_pred.labels, data_pred.scores, 'AdvDebiasing')

print('time elapsed : %.2fs'%(time()-t0))
t0 = time()

debiased_model = PrejudiceRemover(sensitive_attr="Victim Race", eta = 25.0)

debiased_model.fit(data_orig_train)



fair = get_fair_metrics_and_plot(data_orig_test, debiased_model, plot=False, model_aif=True)

data_pred = debiased_model.predict(data_orig_test)



algo_metrics = add_to_df_algo_metrics(algo_metrics, debiased_model, fair, data_pred.labels, data_pred.scores, 'PrejudiceRemover')

print('time elapsed : %.2fs'%(time()-t0))
data_orig_test_pred = data_orig_test.copy(deepcopy=True)



# Prediction with the original RandomForest model

scores = np.zeros_like(data_orig_test.labels)

scores = rf.predict_proba(data_orig_test.features)[:,1].reshape(-1,1)

data_orig_test_pred.scores = scores



preds = np.zeros_like(data_orig_test.labels)

preds = rf.predict(data_orig_test.features).reshape(-1,1)

data_orig_test_pred.labels = preds



def format_probs(probs1):

    probs1 = np.array(probs1)

    probs0 = np.array(1-probs1)

    return np.concatenate((probs0, probs1), axis=1)
privileged_groups, unprivileged_groups = get_attributes(data_orig_train, selected_attr=['Victim Race'])

t0 = time()



cost_constraint = "fnr" # "fnr", "fpr", "weighted"



CPP = CalibratedEqOddsPostprocessing(privileged_groups = privileged_groups,

                                     unprivileged_groups = unprivileged_groups,

                                     cost_constraint=cost_constraint,

                                     seed=42)



CPP = CPP.fit(data_orig_test, data_orig_test_pred)

data_transf_test_pred = CPP.predict(data_orig_test_pred)



fair = fair_metrics(data_orig_test, data_orig_test_pred, pred_is_dataset=True)



algo_metrics = add_to_df_algo_metrics(algo_metrics, 

                                      CPP, 

                                      fair, 

                                      data_transf_test_pred.labels, 

                                      format_probs(data_transf_test_pred.scores), 

                                      'CalibratedEqOdds')

print('time elapsed : %.2fs'%(time()-t0))
privileged_groups, unprivileged_groups = get_attributes(data_orig_train, selected_attr=['Victim Race'])

t0 = time()



ROC = RejectOptionClassification(privileged_groups = privileged_groups,

                             unprivileged_groups = unprivileged_groups)



ROC = ROC.fit(data_orig_test, data_orig_test_pred)

data_transf_test_pred = ROC.predict(data_orig_test_pred)



fair = fair_metrics(data_orig_test, data_transf_test_pred, pred_is_dataset=True)



algo_metrics = add_to_df_algo_metrics(algo_metrics, 

                                      ROC, 

                                      fair, 

                                      data_transf_test_pred.labels, 

                                      format_probs(data_transf_test_pred.scores), 

                                      'RejectOption')

print('time elapsed : %.2fs'%(time()-t0))
def plot_fair_metrics_plotly(fair_metrics):

    bottom = [-1, -1, -1, 0, 0]

    max_valid = [0.1, 0.1, 0.1, 1.2, 0.25]

    min_valid = [-0.1, -0.1, -0.1, 0.8, 0]

    cols = fair_metrics.columns.values



    for i in range(0, 5):

        col = cols[i]



        x, y = (fair_metrics[col].values, fair_metrics.index)

        colors = []

        for v in x:

            color = '#e74c3c' if v < min_valid[i] or v > max_valid[i] else '#2ecc71'

            colors.append(color)



        trace = go.Bar(x=x, y=y, marker=dict(color=colors)

                       , opacity=0.9, orientation='h')



        layout = go.Layout(barmode='group',

                           title=col,

                           xaxis=dict(range=[bottom[i], bottom[i] + 2]),

                           yaxis=go.layout.YAxis(automargin=True),

                           shapes=[

                               {

                                   'type': 'line',

                                   'x0': min_valid[i],

                                   'y0': -1,

                                   'x1': min_valid[i],

                                   'y1': len(y),

                                   'line': {

                                       'color': 'rgb(0, 0, 0)',

                                       'width': 2,

                                   },

                               }, {

                                   'type': 'line',

                                   'x0': max_valid[i],

                                   'y0': -1,

                                   'x1': max_valid[i],

                                   'y1': len(y),

                                   'line': {

                                       'color': 'rgb(0, 0, 0)',

                                       'width': 2,

                                   },

                               }])

        fig = go.Figure([trace], layout=layout)

        py.iplot(fig)





def plot_score_fair_metrics(score):

    display(score.sort_values(['nb_valid', 'score'], ascending=[0, 1]))

    score.sort_values(['nb_valid', 'score'], ascending=[1, 0], inplace=True)



    gold, silver, bronze, other = ('#FFA400', '#bdc3c7', '#cd7f32', '#3498db')

    colors = [gold if i == 0 else silver if i == 1 else bronze if i == 2 else other for i in range(0, len(score))]

    colors = [c for c in reversed(colors)]



    x, y = (score['score'].values, score.index)



    trace = go.Bar(x=x, y=y, marker=dict(color=colors)

                   , opacity=0.9, orientation='h')

    layout = go.Layout(barmode='group',

                       title='Fairest algorithm',

                       yaxis=go.layout.YAxis(automargin=True))

    fig = go.Figure([trace], layout=layout)

    py.iplot(fig)

    



def score_fair_metrics(fair):

    objective = [0, 0, 0, 1, 0]

    max_valid = [0.1, 0.1, 0.1, 1.2, 0.25]

    min_valid = [-0.1, -0.1, -0.1, 0.8, 0]



    nb_valid = np.sum(((fair.values > min_valid) * (fair.values < max_valid)), axis=1)

    score = np.sum(np.abs(fair.values - objective), axis=1)

    score = np.array([score, nb_valid])



    score = pd.DataFrame(data=score.transpose(), columns=['score', 'nb_valid'], index=fair.index)

    return score





def score_all_attr(algo_metrics):

    attributes = algo_metrics.loc['Origin', 'fair_metrics'].index.values[1:]



    all_scores = np.zeros((len(algo_metrics), 2))

    for attr in attributes:

        df_metrics = pd.DataFrame(columns=algo_metrics.loc['Origin', 'fair_metrics'].columns.values)

        for fair in algo_metrics.loc[:, 'fair_metrics']:

            df_metrics = df_metrics.append(fair.loc[attr], ignore_index=True)

        all_scores = all_scores + score_fair_metrics(df_metrics).values



    final = pd.DataFrame(data=all_scores, columns=['score', 'nb_valid'], index=algo_metrics.index)

    return final

def compare_fair_metrics(algo_metrics, attr='Victim Race'):

    

    df_metrics = pd.DataFrame(columns=algo_metrics.loc['Origin','fair_metrics'].columns.values)

    for fair in algo_metrics.loc[:,'fair_metrics']:

        df_metrics = df_metrics.append(fair.loc[attr], ignore_index=True)



    df_metrics.index = algo_metrics.index.values

    df_metrics = df_metrics.replace([np.inf, -np.inf], np.NaN)

    

    display(df_metrics)

    plot_fair_metrics_plotly(df_metrics)

    score = score_fair_metrics(df_metrics)

    plot_score_fair_metrics(score.dropna())
compare_fair_metrics(algo_metrics)
def plot_compare_model_performance(algo_metrics, dataset):

    X_test = dataset.features

    y_true = dataset.labels

    perf_metrics = pd.DataFrame()



    models_name = algo_metrics.index.values



    fig = plt.figure(figsize=(7, 7))

    plt.title('ROC curve for differents models')

    lw = 2

    palette = sns.color_palette("Paired")



    for model_name, i in zip(models_name, range(0, len(models_name))):

        model = algo_metrics.loc[model_name, 'model']



        if model_name != 'AdvDebiasing':

            probs = algo_metrics.loc[model_name, 'probs']

            y_pred = algo_metrics.loc[model_name, 'prediction']

            accuracy, matrix, f1, fpr, tpr, roc_auc = get_model_performance(X_test, y_true, y_pred, probs)



            perf_metrics = perf_metrics.append(

                pd.DataFrame([[accuracy, f1]], columns=['Accuracy', 'F1 Score'], index=[model_name]))

            plt.plot(fpr, tpr, color=palette[i], lw=lw, label=str(model_name) + ' (area = %0.2f)' % roc_auc)



    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')

    plt.xlim([0.0, 1.0])

    plt.ylim([0.0, 1.05])

    plt.xlabel('False Positive Rate')

    plt.ylabel('True Positive Rate')

    plt.title('Receiver Operating Characteristic curve')

    plt.legend(loc="lower right")

    display(perf_metrics.sort_values(by=['Accuracy', 'F1 Score'], ascending=[False, False]))

    plt.show()
plot_compare_model_performance(algo_metrics, data_orig_test)
# I remove others perpetrator column

columns_to_drop = ['Perpetrator Sex','Perpetrator Age category','Perpetrator Age']

dataset_race = data_encoded.drop(columns_to_drop, axis=1)
dataset_race.head()
categorical_names['Victim Sex'], categorical_names['Victim Race'], categorical_names['Perpetrator Race'] 
dataset = dataset_race

label_name = 'Perpetrator Race'

protected_attribute_names=['Victim Sex', 'Victim Race']



favorable_classes = np.where(categorical_names['Perpetrator Race'] == 'White')[0]

privileged_sex = np.where(categorical_names['Victim Sex'] == 'Male')[0]

privileged_race = np.where(categorical_names['Victim Race'] == 'White')[0]
privileged_classes = [privileged_sex, privileged_race]
data_orig_rac = StandardDataset(dataset, 

                               label_name=label_name, 

                               favorable_classes=favorable_classes, 

                               protected_attribute_names=protected_attribute_names, 

                               privileged_classes=[privileged_sex, privileged_race])
meta_data(data_orig_rac)
np.random.seed(42)

data_orig_rac_train, data_orig_rac_test = data_orig_rac.split([0.7], shuffle=True)



display(Markdown("#### Train Dataset shape"))

print("Perpetrator Race : ",data_orig_rac_train.features.shape)

display(Markdown("#### Test Dataset shape"))

print("Perpetrator Race : ",data_orig_rac_test.features.shape)
# if you want to change to model be my guest 

model = RandomForestClassifier()

model = model.fit(data_orig_rac_train.features, data_orig_rac_train.labels.ravel())
X_test_rac = data_orig_rac_test.features

y_test_rac = data_orig_rac_test.labels.ravel()

plot_model_performance(model, data_orig_rac_test.features, y_test_rac)
# we need to create a new dataset with predictions from the model into labels attribute

dataset = data_orig_rac_test

dataset_pred = dataset.copy()

dataset_pred.labels = model.predict(data_orig_rac_test.features)
dataset_pred.protected_attribute_names
# I get the index of Male and Female classes

privileged_race   = np.where(categorical_names['Victim Race'] == 'White')[0]

unprivileged_race = np.where(categorical_names['Victim Race'] == 'Black')[0]



# I format variable like in the documentation of ClassificationMetric and BinaryLabelDatasetMetric

privileged_groups   = [{'Victim Race' : privileged_race}] 

unprivileged_groups = [{'Victim Race' : unprivileged_race}] 



# I create both classes

classified_metric_race = ClassificationMetric(dataset, 

                                         dataset_pred,

                                         unprivileged_groups=unprivileged_groups,

                                         privileged_groups=privileged_groups)



metric_pred_race = BinaryLabelDatasetMetric(dataset_pred,

                                         unprivileged_groups=unprivileged_groups,

                                         privileged_groups=privileged_groups)
# TODO : do the same as above but with Victim Sex attribute

privileged_sex   = None

unprivileged_sex = None 



privileged_groups   = [{'' : privileged_sex}] 

unprivileged_groups = [{'' : unprivileged_sex}] 



# Remove comments below

# classified_metric_sex = ClassificationMetric(dataset, 

#                                          dataset_pred,

#                                          unprivileged_groups=unprivileged_groups,

#                                          privileged_groups=privileged_groups)



# metric_pred_sex = BinaryLabelDatasetMetric(dataset_pred,

#                                          unprivileged_groups=unprivileged_groups,

#                                          privileged_groups=privileged_groups)
def is_fair(metric, objective=0, threshold=0.2):

    return abs(metric - objective) <= threshold
# This is an example for mean_difference metric and race attribute:

my_fair_metric = metric_pred_race.mean_difference()

print(my_fair_metric)



if is_fair(my_fair_metric):

    print('My metric is fair')

else:

    print('My metric is not fair')
# This is all the code for 4 others fair metrics and victim race attribute

classified_metric_race.equal_opportunity_difference()

classified_metric_race.average_abs_odds_difference()

metric_pred_race.disparate_impact()

classified_metric_race.theil_index()



# TODO : find out which metrics said that this is fair or not. 

#        remember that disparate_impact fair is at 1 and the others at 0

your_metric = 0
# TODO : do the same for the victim sex attribute 

#        metric_pred_sex and classified_metric_sex class





# This is a function you can use to get privileged_groups and unprivileged_groups informations

# that are used for processors 



def get_attributes(data, selected_attr=None):

    unprivileged_groups = []

    privileged_groups = []

    if selected_attr == None:

        selected_attr = data.protected_attribute_names

    

    for attr in selected_attr:

            idx = data.protected_attribute_names.index(attr)

            privileged_groups.append({attr:data.privileged_protected_attributes[idx]}) 

            unprivileged_groups.append({attr:data.unprivileged_protected_attributes[idx]}) 



    return privileged_groups, unprivileged_groups
# an example

get_attributes(dataset, selected_attr=['Victim Sex'])
# also to remember the dataset used for training is this one

dataset = data_orig_rac_train

# and the test one :

dataset_test = data_orig_rac_test
privileged_groups, unprivileged_groups = get_attributes(dataset, selected_attr=['Victim Race'])

t0 = time()



# Creation of the algorithm class

RW = Reweighing(unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)



# I transform my dataset

data_transf_train = RW.fit_transform(dataset)



# Train and save the model

rf_transf = RandomForestClassifier()

rf_transf = rf_transf.fit(data_transf_train.features, 

                          data_transf_train.labels.ravel(), 

                         sample_weight=data_transf_train.instance_weights) # sample_weight arguments is used thanks to Reweighing class



print('time elapsed : %.2fs'%(time()-t0))
# Now I can predict on the test dataset

data_transf_test = RW.transform(dataset_test)



rf_transf.predict(dataset_test.features)