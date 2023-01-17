from glob import glob



import statsmodels.api as sm



from plotly import graph_objs as go

import plotly.express as px

import plotly.figure_factory as ff



import pandas as pd

import numpy as np



import seaborn as sns

import matplotlib.pyplot as plt



from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold

from sklearn.preprocessing import LabelEncoder



import pydicom



import lightgbm as lgb

from lightgbm import LGBMRegressor



import warnings

warnings.filterwarnings('ignore')
train_df = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/train.csv')

test_df = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/test.csv')



train_df.head()
print('Number of training samples:', train_df.shape[0])

print('Number of test samples:', test_df.shape[0])
print('Number of NaN values in training set:', int(train_df.isna().sum().sum()))

print('Number of NaN values in test set:', int(test_df.isna().sum().sum()))
print('Number of unique patients:', len(train_df['Patient'].unique()))
# Thanks to Twinkle Khanna for the plot, couldn't figure out how to do it with Seaborn ;)

# https://www.kaggle.com/twinkle0705/your-starter-notebook-for-osic



new_df = train_df.groupby(

    [

        train_df.Patient,

        train_df.Age,train_df.Sex, 

        train_df.SmokingStatus

    ]

)['Patient'].count()



new_df.index = new_df.index.set_names(

    [

        'id',

        'Age',

        'Sex',

        'SmokingStatus'

    ]

)



new_df = new_df.reset_index()

new_df.rename(columns = {'Patient': 'freq'},inplace = True)



fig = px.bar(new_df, x='id',y ='freq',color='freq')

fig.update_layout(

    xaxis={'categoryorder':'total ascending'},

    title='Distribution of images for each patient'

)

fig.update_xaxes(showticklabels=False)

fig.show()
ax = sns.distplot(train_df['Weeks'], bins=100)

ax.set_title('Distribution of Weeks')
ax = sns.distplot(train_df['FVC'], bins=100)

ax.set_title('Distribution of the target variable (FVC)')
fig = px.histogram(

    train_df, 

    x='FVC',

    nbins =100

)



fig.update_traces(

    marker_color='rgb(158,202,225)', 

    marker_line_color='rgb(8,48,107)',

    marker_line_width=1.5, 

    opacity=0.6

)



fig.update_layout(

    title = 'Distribution of FVC'

)



fig.show()
fig = px.histogram(

    new_df, 

    x='Age',

    nbins = 42

)



fig.update_traces(

    marker_color='rgb(158,202,225)', 

    marker_line_color='rgb(8,48,107)',

    marker_line_width=1.5, 

    opacity=0.6

)



fig.update_layout(

    title = 'Distribution of Age'

)



fig.show()
temp = train_df.groupby('Sex').count()['Patient'].reset_index().sort_values(by='Patient',ascending=False)

temp.style.background_gradient(cmap='Reds')
fig = px.funnel_area(

    names=temp['Sex'].values,

    values=temp['Patient'].values,

)



fig.update_layout(

    title = 'Distribution of Sex'

)



fig.show()
temp = train_df.groupby('SmokingStatus').count()['Patient'].reset_index().sort_values(by='Patient',ascending=False)

temp.style.background_gradient(cmap='Reds')
fig = px.funnel_area(

    names=temp['SmokingStatus'].values,

    values=temp['Patient'].values,

)



fig.update_layout(

    title = 'Distribution of SmokingStatus'

)



fig.show()
fig = plt.figure(figsize=(10,6))

ax = sns.countplot(x="SmokingStatus", hue="Sex", data=train_df)

for p in ax.patches:

    '''

    https://www.kaggle.com/rohitsingh9990/panda-eda-better-visualization-simple-baseline

    '''

    height = p.get_height()

    ax.text(p.get_x()+p.get_width()/2,

                height +3,

                '{:1.2f}%'.format(100*height/len(train_df)),

                ha="center")



ax.set_title('Bivariate analysis: Distribution of SmokingStatus with respect to Sex')
fig, ax = plt.subplots(1, 1, figsize=(7, 7))



sns.distplot(

    train_df[train_df['Sex'] == 'Male']['Age'], 

    bins=30, 

    ax=ax, 

    kde_kws=

        {

            "color": "blue", 

            "label": "Male"

        },

    hist_kws=

        {

            "linewidth": 3,

            "color": "blue"

        }

)



sns.distplot(

    train_df[train_df['Sex'] == 'Female']['Age'], 

    bins=30, 

    ax=ax,

    kde_kws=

        {

            "label": "Female",

            "color": 'mediumturquoise'

        },

    hist_kws=

        {

            'linewidth': 3,

            'color': 'mediumturquoise',

        }

)



fig.suptitle('Distribution of Age w.r.t. sex for unique patients')
fig = px.histogram(

    train_df, 

    x='Age',

    color='Sex',

    color_discrete_map=

        {

            'Male':'blue',

            'Female':'mediumturquoise'

        },

    hover_data=train_df.columns

)



fig.update_layout(title='Distribution of Age w.r.t. sex for unique patients')



fig.update_traces(

    marker_line_color='black',

    marker_line_width=1.5, 

    opacity=0.85

)



fig.show()
# Plot overlapping distribution



fig, ax = plt.subplots(1, 1, figsize=(7, 7))



sns.distplot(

    train_df[train_df['SmokingStatus'] == 'Never smoked']['Age'], 

    bins=30, 

    ax=ax, 

    kde_kws=

        {

            "color": "khaki", 

            "label": "Never smoked"

        },

    hist_kws=

        {

            "linewidth": 3,

            "color": "khaki"

        }

)



sns.distplot(

    train_df[train_df['SmokingStatus'] == 'Currently smokes']['Age'], 

    bins=30, 

    ax=ax,

    kde_kws=

        {

            "label": "Currently smokes",

            "color": 'darksalmon'

        },

    hist_kws=

        {

            'linewidth': 3,

            'color': 'darksalmon',

        }

)



sns.distplot(

    train_df[train_df['SmokingStatus'] == 'Ex-smoker']['Age'], 

    bins=30, 

    ax=ax,

    kde_kws=

        {

            "label": "Ex-smoker",

            "color": 'teal'

        },

    hist_kws=

        {

            'linewidth': 3,

            'color': 'teal',

        }

)



fig.suptitle('Distribution of Age w.r.t. SmokingStatus for unique patients')
fig = px.histogram(

    train_df, 

    x='Age',

    color='SmokingStatus',

    color_discrete_map=

        {

            'Never smoked':'khaki',

            'Currently smokes':'darksalmon',

            'Ex-smoker': 'teal', 

        },

    hover_data=train_df.columns

)



fig.update_layout(title='Distribution of Age w.r.t. SmokingStatus for unique patients')



fig.update_traces(

    marker_line_color='black',

    marker_line_width=1.5, 

    opacity=0.85

)



fig.show()
fig, ax = plt.subplots(1, 1, figsize=(7, 7))



sns.distplot(

    train_df[train_df['Sex'] == 'Male']['FVC'], 

    bins=30, 

    ax=ax, 

    kde_kws=

        {

            "color": "blue", 

            "label": "Male"

        },

    hist_kws=

        {

            "linewidth": 3,

            "color": "blue"

        }

)



sns.distplot(

    train_df[train_df['Sex'] == 'Female']['FVC'], 

    bins=30, 

    ax=ax,

    kde_kws=

        {

            "label": "Female",

            "color": 'mediumturquoise'

        },

    hist_kws=

        {

            'linewidth': 3,

            'color': 'mediumturquoise',

        }

)



fig.suptitle('Distribution of FVC w.r.t. sex for unique patients')
fig = px.histogram(

    train_df, 

    x='FVC',

    color='Sex',

    color_discrete_map=

        {

            'Male':'blue',

            'Female':'mediumturquoise'

        },

    hover_data=train_df.columns

)



fig.update_layout(title='Distribution of FVC w.r.t. sex for unique patients')



fig.update_traces(

    marker_line_color='black',

    marker_line_width=1.5, 

    opacity=0.85

)



fig.show()
# Plot overlapping distribution



fig, ax = plt.subplots(1, 1, figsize=(7, 7))



sns.distplot(

    train_df[train_df['SmokingStatus'] == 'Never smoked']['FVC'], 

    bins=30, 

    ax=ax, 

    kde_kws=

        {

            "color": "khaki", 

            "label": "Never smoked"

        },

    hist_kws=

        {

            "linewidth": 3,

            "color": "khaki"

        }

)



sns.distplot(

    train_df[train_df['SmokingStatus'] == 'Currently smokes']['FVC'], 

    bins=30, 

    ax=ax,

    kde_kws=

        {

            "label": "Currently smokes",

            "color": 'darksalmon'

        },

    hist_kws=

        {

            'linewidth': 3,

            'color': 'darksalmon',

        }

)



sns.distplot(

    train_df[train_df['SmokingStatus'] == 'Ex-smoker']['FVC'], 

    bins=30, 

    ax=ax,

    kde_kws=

        {

            "label": "Ex-smoker",

            "color": 'teal'

        },

    hist_kws=

        {

            'linewidth': 3,

            'color': 'teal',

        }

)



fig.suptitle('Distribution of FVC w.r.t. SmokingStatus for unique patients')
fig = px.histogram(

    train_df, 

    x='FVC',

    color='SmokingStatus',

    color_discrete_map=

        {

            'Never smoked':'khaki',

            'Currently smokes':'darksalmon',

            'Ex-smoker': 'teal', 

        },

    hover_data=train_df.columns

)



fig.update_layout(title='Distribution of FVC w.r.t. SmokingStatus for unique patients')



fig.update_traces(

    marker_line_color='black',

    marker_line_width=1.5, 

    opacity=0.85

)



fig.show()
sns.jointplot(x="Weeks", y="FVC", data=train_df)
X = train_df['Weeks'].values

y = train_df['FVC'].values



results = sm.OLS(y, X).fit()

print(results.summary())
def plot_patient_level_weeks_vs_fvc(patient_id, ax):

    X = train_df[train_df['Patient'] == patient_id]['Weeks'].values

    y = train_df[train_df['Patient'] == patient_id]['FVC'].values

    

    ax.set_title(patient_id)

    ax = sns.regplot(X, y, ax=ax, ci=None, line_kws={'color':'red'})
f, axes = plt.subplots(1, 3, figsize=(15, 5))



patient_ids = train_df["Patient"].sample(n=3).values



for i in range(3):

    plot_patient_level_weeks_vs_fvc(patient_ids[i], axes[i])
sns.jointplot(x="Percent", y="FVC", data=train_df)
X = train_df['Percent'].values

y = train_df['FVC'].values



results = sm.OLS(y, X).fit()

print(results.summary())
def load_scan(path):

    slices = [pydicom.read_file(path + '/' + s) for s in os.listdir(path)]

    slices.sort(key = lambda x: int(x.InstanceNumber))

    

    try:

        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])

    except:

        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)

        

    for s in slices:

        s.SliceThickness = slice_thickness

        

    return slices
gkf = GroupKFold(n_splits=5)



X = train_df[['Patient', 'Weeks', 'Percent', 'Age', 'Sex', 'SmokingStatus']].values

y = train_df['FVC'].values

groups = train_df['Patient'].values



for i, (trn_, val_) in enumerate(gkf.split(X, y, groups)):

    train_df.loc[val_, 'fold'] = i



train_df['fold'] = train_df['fold'].astype(int)
p = []



for i in range(5):

    f = set(train_df[train_df['fold'] == i]['Patient'].unique())

    p.append(f)



print('Number of common patients across folds:', len(set.intersection(*p)))
# Check distribution of patients within a fold



fig, ax = plt.subplots(1, 5, figsize=(18, 3))



for i in range(5):

    sns.countplot(train_df[train_df['fold'] == i]['Patient'], ax=ax[i])

    ax[i].set_title(f'Fold {i+1}')

    ax[i].xaxis.set_ticklabels([])



fig.suptitle('Distribution of patients in folds')

fig.tight_layout(rect=[0, 0.03, 1, 0.95])
train_df.head()
train_df.to_csv('train_df.csv')
# Quick label encoding



le = LabelEncoder()

train_df['Sex'] = le.fit_transform(train_df['Sex'])

test_df['Sex'] = le.transform(test_df['Sex'])



train_df['SmokingStatus'] = le.fit_transform(train_df['SmokingStatus'])

test_df['SmokingStatus'] = le.transform(test_df['SmokingStatus'])
def metric(actual_fvc, predicted_fvc, confidence, return_values = False):

    """

        Calculates the modified Laplace Log Likelihood score for this competition.

        Credits: https://www.kaggle.com/rohanrao/osic-understanding-laplace-log-likelihood

    """

    sd_clipped = np.maximum(confidence, 70)

    delta = np.minimum(np.abs(actual_fvc - predicted_fvc), 1000)

    metric = - np.sqrt(2) * delta / sd_clipped - np.log(np.sqrt(2) * sd_clipped)



    if return_values:

        return metric

    else:

        return np.mean(metric)
param = {

        'objective':'regression',

        'metric': 'rmse',

        'boosting_type': 'gbdt',

        'learning_rate': 0.01,

        'max_depth':-1

}



target = 'FVC'
def train_lgb_model(features):    

    y_oof = np.zeros(train_df.shape[0])

    y_test = np.zeros((test_df.shape[0], 5))



    for f, (train_ind, val_ind) in enumerate(gkf.split(train_df, train_df, groups)):

        print(f'Training on all folds except {f}, validation on {f}')

        t_df, val_df = train_df.iloc[train_ind], train_df.iloc[val_ind]



        model = LGBMRegressor()

        model.fit(t_df[features], t_df[target])

        

        lgb.plot_importance(model, title=f'Feature importance - Fold {f+1}')



        y_oof[val_ind] = model.predict(val_df[features])

        y_test[:, f] = model.predict(test_df[features])

    

    return y_oof, y_test
features = ['Weeks', 'Percent', 'Age', 'Sex', 'SmokingStatus']

y_oof, y_test = train_lgb_model(features)
train_df['oof_preds'] = y_oof

test_df['target'] = y_test.mean(axis=1)



score = metric(train_df[target], train_df['oof_preds'], np.std(train_df['oof_preds']))



print('OOF log-Laplace likelihood score:', score)
# Lagging features



def generate_lagging_features(df):

    for i in [1, 2, 3]:

        df['lag_FVC_'+str(i)] = df.groupby(['Patient'])['FVC'].transform(lambda x: x.shift(i))

        df['lag_percent_'+str(i)] = df.groupby(['Patient'])['Percent'].transform(lambda x: x.shift(i))

    

    return df
# Running statistics



def generate_running_statistics(df):

    for i in [3, 5, 7]:

        df['rolling_FVC_mean_'+str(i)] = df.groupby(['Patient'])['FVC'].transform(lambda x: x.shift(1).rolling(i).mean())

        df['rolling_FVC_std_'+str(i)]  = df.groupby(['Patient'])['FVC'].transform(lambda x: x.shift(1).rolling(i).std())

        df['rolling_FVC_max_'+str(i)]  = df.groupby(['Patient'])['FVC'].transform(lambda x: x.shift(1).rolling(i).max())

        

        df['rolling_percent_mean_'+str(i)] = df.groupby(['Patient'])['Percent'].transform(lambda x: x.shift(1).rolling(i).mean())

        df['rolling_percent_std_'+str(i)]  = df.groupby(['Patient'])['Percent'].transform(lambda x: x.shift(1).rolling(i).std())

        df['rolling_percent_max_'+str(i)]  = df.groupby(['Patient'])['Percent'].transform(lambda x: x.shift(1).rolling(i).max())

    

    return df
train_df = generate_lagging_features(train_df)

train_df = generate_running_statistics(train_df)



test_df = generate_lagging_features(test_df)

test_df = generate_running_statistics(test_df)
features = [x for x in list(train_df.columns) if x not in ['Patient', 'fold', 'FVC', 'oof_preds']]

y_oof, y_test = train_lgb_model(features)
train_df['oof_preds'] = y_oof

test_df['target'] = y_test.mean(axis=1)



score = metric(train_df[target], train_df['oof_preds'], np.std(train_df['oof_preds']))



print('OOF log-Laplace likelihood score:', score)