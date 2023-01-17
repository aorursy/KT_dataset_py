%matplotlib inline
import os
import pandas as pd
from tqdm import tqdm
import datetime as dt
import numpy as np
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_palette(sns.color_palette('tab20', 20))
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.figure_factory as ff
from sklearn import metrics
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier
pd.set_option('display.max_columns', 999)
np.random.seed(1987)
plt.rcParams['figure.figsize'] = [16, 10]
plt.rcParams['font.size'] = 14
start = dt.datetime.now()
!cp ../input/surveykernelexternals/* .

data_dir = '../input/kaggle-survey-2018/'
META_DATA_PATH = '../input/meta-kaggle/'
questions = pd.read_csv(os.path.join(data_dir, 'SurveySchema.csv'))
responses = pd.read_csv(os.path.join(data_dir, 'multipleChoiceResponses.csv'),
                        low_memory=False)
responses['cnt'] = 1
target_order = pd.DataFrame({
    'Q26': ['Definitely not', 'Definitely yes', 'Maybe', 'Probably not', 'Probably yes'],
    'target': [0, 4, 2, 1, 3],
})
target_order = target_order.sort_values(by='target')
target_order['color'] = ['rgb(217,30,30)', 'rgb(242,143,56)', 'rgb(242,211,56)',
                         'rgb(10,136,186)', 'rgb(12,51,131)']
responses = responses.merge(target_order, how='left', on='Q26')
questions.shape
questions.head()
responses.shape
responses.head()

q26 = responses[1:].groupby(['Q26', 'target'])[['cnt']].count()
q26 = q26.sort_values(by='target').reset_index()
q26
data = [
    go.Bar(
        x=q26.Q26.values,
        y=q26.cnt.values,
        marker=dict(color=-q26['target'].values,
                    colorscale='Portland', showscale=False)
    ),
]
layout = go.Layout(
    title='Do you consider yourself to be a data scientist?',
    yaxis=dict(title='# of Respondents:', ticklen=5, gridwidth=2),
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='q26')
responses[['Q24', 'Q25', 'Q26']][:2]
q24_clean = pd.DataFrame({
    'Q24': ['I have never written code and I do not want to learn',
            '40+ years', '30-40 years', '20-30 years',
            'I have never written code but I want to learn', '10-20 years',
            '5-10 years', '3-5 years', '< 1 year', '1-2 years'],
    'q24_clean': ['None', '10+', '10+', '10+', 'None', '10+', '5-10', '3-5', '<1', '1-2'],
    'f24': [-1, 10, 10, 10, -1, 10, 7.5, 4, 0.5, 1.5]
})
enhanced_responses = responses.merge(q24_clean, on='Q24')
q24_cross = enhanced_responses[1:].groupby([
    'Q26', 'target', 'color', 'q24_clean', 'f24'])[['cnt']].count()\
    .reset_index().sort_values(by='f24')
q24_cross
data = []
for ds in target_order.Q26.values:
    df = q24_cross[q24_cross.Q26 == ds]
    data.append(
        go.Bar(x=df.q24_clean.values, y=df.cnt.values,
               marker=dict(color=df.color.values),
               name=ds)
    )
layout = go.Layout(
    title='Data Analysis Experience',
    xaxis=dict(title='How long have you been writing code to analyze data? (years)',
               ticklen=5, zeroline=False, gridwidth=2),
    yaxis=dict(title='Data Scientist Confidence', ticklen=5, gridwidth=2),
    showlegend=True
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='data-code')
q25_clean = pd.DataFrame({
    'Q25': ['I have never studied machine learning and I do not plan to',
            '20+ years', '10-15 years', '4-5 years', '5-10 years', '3-4 years',
            'I have never studied machine learning but plan to learn in the future',
            '2-3 years', '1-2 years', '< 1 year'],
    'q25_clean': ['None', '4+', '4+', '4+', '4+', '3-4', 'None', '2-3', '1-2', '<1'],
    'f25': [-1, 5, 5, 5, 5, 3.5, -1, 2.5, 1.5, 0.5]
})
enhanced_responses = responses.merge(q25_clean, on='Q25')
q26_cross = enhanced_responses[1:].groupby([
    'Q26', 'target', 'color', 'q25_clean', 'f25'])[['cnt']]\
    .count().reset_index().sort_values(by='f25')
q26_cross

data = []
for ds in target_order.Q26.values:
    df = q26_cross[q26_cross.Q26 == ds]
    data.append(
        go.Bar(x=df.q25_clean.values, y=df.cnt.values,
               marker=dict(color=df.color.values),
               name=ds)
    )
layout = go.Layout(
    title='Machine Learning Experience',
    xaxis=dict(title='Machine Learning Experience (years)',
               ticklen=5, zeroline=False, gridwidth=2),
    yaxis=dict(title='Data Scientist Confidence', ticklen=5, gridwidth=2),
    showlegend=True
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='mlexp')
q6_cnt = responses[1:].groupby(['Q6'])[['cnt']].count().reset_index()
q6_mean = responses[1:].groupby(['Q6'])[['target']].mean().reset_index()

q6_clean = pd.merge(q6_cnt, q6_mean, on='Q6')
q6_clean.columns = ['Q6', 'Q6_cnt', 'Q6_dsc']
q6_clean = q6_clean.sort_values(by='Q6_dsc', ascending=False)
q6_clean
enhanced_responses = responses.merge(q6_clean, on='Q6')

q6_cross = enhanced_responses[1:].groupby([
    'Q6', 'Q6_cnt', 'Q6_dsc', 'Q26', 'target', 'color'])[['cnt']] \
    .count().reset_index().sort_values(by='Q6_dsc')
q6_cross = q6_cross[q6_cross.Q6_cnt > 500]
q6_cross
data = []
for ds in target_order.Q26.values:
    df = q6_cross[q6_cross.Q26 == ds]
    data.append(
        go.Bar(x=df.Q6.values, y=df.cnt.values,
               marker=dict(color=df.color.values),
               name=ds)
    )
layout = go.Layout(
    title='An individual with data scientist job title is probably a data scientist :)',
    xaxis=dict(title='Title most similar to current role',
               ticklen=5, zeroline=False, gridwidth=2),
    yaxis=dict(title='Data Scientist Confidence', ticklen=5, gridwidth=2),
    showlegend=True
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='job-title')
('<1' > '2',
 '10-15' < '3-4',
 'Doctoral degree' < 'No formal education past high school')
def first_number(s):
    s = s.replace('+', '-')
    s = s.replace(' ', '')
    s = s.replace(',000', '')
    s = s.replace('%', '-')
    try:
        return int(s.split('-')[0])
    except Exception:
        return np.nan

# Q1 5 23860
# What is your gender? - Selected Choice
Q1 = pd.DataFrame(
    {'Q1': ['Female', 'Male', 'Prefer not to say', 'Prefer to self-describe']})
Q1['f1'] = [0, 1, 2, 2]
print(Q1)

# Q2 13 23860
# What is your age (# years)?
Q2 = pd.DataFrame(
    {'Q2': ['45-49', '30-34', '35-39', '22-24', '25-29',
            '18-21', '40-44', '55-59', '60-69',
            '50-54', '80+', '70-79']})
Q2['f2'] = Q2.Q2.apply(first_number)
print(Q2)

# Q4 8 23439
# What is the highest level of formal education that you have attained or plan to attain within the next 2 years?
Q4 = pd.DataFrame(
    {'Q4': ['Doctoral degree', 'Bachelor’s degree', 'Master’s degree', 'Professional degree',
            'Some college/university study without earning a bachelor’s degree',
            'I prefer not to answer',
            'No formal education past high school']})
Q4['f4'] = [5, 2, 3, 2, 1, np.nan, 0]
print(Q4)

# Q8 12 21102
# How many years of experience do you have in your current role?
Q8 = pd.DataFrame({'Q8': ['5-10', '0-1', '10-15', '3-4', '1-2',
                          '2-3', '15-20', '4-5', '20-25',
                          '25-30', '30 +']})
Q8['f8'] = Q8.Q8.apply(first_number)
print(Q8)

# Q9 20 20186
# What is your current yearly compensation (approximate $USD)?
Q9 = pd.DataFrame({'Q9': ['I do not wish to disclose my approximate yearly compensation',
                          '10-20,000', '0-10,000', '20-30,000', '125-150,000', '30-40,000',
                          '50-60,000', '100-125,000', '90-100,000', '70-80,000', '80-90,000',
                          '60-70,000', '400-500,000', '40-50,000', '150-200,000', '500,000+',
                          '300-400,000', '200-250,000', '250-300,000']})
Q9['f9'] = Q9.Q9.apply(first_number)
print(Q9)

# Q10 7 20670
# Does your current employer incorporate machine learning methods into their business?
Q10 = pd.DataFrame({'Q10': [
    'I do not know',
    'No (we do not use ML methods)',
    'We are exploring ML methods (and may one day put a model into production)',
    'We recently started using ML methods (i.e., models in production for less than 2 years)',
    'We have well established ML methods (i.e., models in production for more than 2 years)',
    'We use ML methods for generating insights (but do not put working models into production)'
]})
Q10['f10'] = [-1, 0, 1, 2, 3, 1]
print(Q10)

# Q12_MULTIPLE_CHOICE 7 19199
# What is the primary tool that you use at work or school to analyze data? (include text response) - Selected Choice
Q12_MULTIPLE_CHOICE = pd.DataFrame({'Q12_MULTIPLE_CHOICE': [
    'Cloud-based data software & APIs (AWS, GCP, Azure, etc.)',
    'Basic statistical software (Microsoft Excel, Google Sheets, etc.)',
    'Local or hosted development environments (RStudio, JupyterLab, etc.)',
    'Advanced statistical software (SPSS, SAS, etc.)',
    'Other',
    'Business intelligence software (Salesforce, Tableau, Spotfire, etc.)'
]})
Q12_MULTIPLE_CHOICE['f12_MULTIPLE_CHOICE'] = [4, 1, 3, 2, -1, 1]
print(Q12_MULTIPLE_CHOICE)

# Q23 7 18548
# Approximately what percent of your time at work or school is spent actively coding?
Q23 = pd.DataFrame({'Q23': [
    '0% of my time',
    '1% to 25% of my time',
    '75% to 99% of my time',
    '50% to 74% of my time',
    '25% to 49% of my time',
    '100% of my time']})
Q23['f23'] = Q23.Q23.apply(first_number)
print(Q23)

# Q24 11 18534
# How long have you been writing code to analyze data?
Q24 = pd.DataFrame({'Q24': [
    'I have never written code but I want to learn', '5-10 years', '3-5 years', '< 1 year',
    '1-2 years', '10-20 years', '20-30 years', '30-40 years',
    'I have never written code and I do not want to learn', '40+ years'
]})
Q24['f24'] = [0, 6, 4, 0.5, 1.5, 12, 24, 30, -1, 40]
print(Q24)

# Q25 11 18492
# For how many years have you used machine learning methods (at work or in school)?
Q25 = pd.DataFrame({'Q25': [
    'I have never studied machine learning but plan to learn in the future', '< 1 year',
    '4-5 years', '2-3 years', '1-2 years', '5-10 years', '3-4 years',
    'I have never studied machine learning and I do not plan to', '20+ years', '10-15 years'
]})
Q25['f25'] = [0, 0.5, 4, 2.5, 1.5, 6, 3.5, -1, 20, 12]
print(Q25)

# Q26 6 18481
# Do you consider yourself to be a data scientist?
Q26 = pd.DataFrame({
    'Q26': ['Definitely not', 'Definitely yes', 'Maybe', 'Probably not', 'Probably yes'],
    'target': [0, 4, 2, 1, 3],
})
Q26 = Q26.sort_values(by='target')
Q26['color'] = ['rgb(217,30,30)', 'rgb(242,143,56)', 'rgb(242,211,56)', 'rgb(10,136,186)',
                'rgb(12,51,131)']
print(Q26)

# Q40 7 15880
# Which better demonstrates expertise in data science: academic achievements or independent projects? - Your views:
Q40 = pd.DataFrame({'Q40': [
    'Independent projects are much less important than academic achievements',
    'Independent projects are slightly less important than academic achievements',
    'Independent projects are equally important as academic achievements',
    'Independent projects are slightly more important than academic achievements',
    'Independent projects are much more important than academic achievements',
    'No opinion; I do not know'
]})
Q40['f40'] = [0, 1, 2, 3, 4, np.nan]
print(Q40)

# Q41_Part_1 5 14937
# How do you perceive the importance of the following topics? - Fairness and bias in ML algorithms:
Q41_Part_1 = pd.DataFrame(
    {'Q41_Part_1': ['Very important', 'Slightly important', 'Not at all important',
                    'No opinion; I do not know']})
Q41_Part_1['f41_Part_1'] = [3, 2, 1, np.nan]
print(Q41_Part_1)

# Q41_Part_2 5 14937
# How do you perceive the importance of the following topics? - Being able to explain ML model outputs and/or predictions
Q41_Part_2 = pd.DataFrame({
    'Q41_Part_2': ['Very important', 'Slightly important', 'Not at all important',
                   'No opinion; I do not know']})
Q41_Part_2['f41_Part_2'] = [3, 2, 1, np.nan]
print(Q41_Part_2)

# Q41_Part_3 5 14937
# How do you perceive the importance of the following topics? - Reproducibility in data science
Q41_Part_3 = pd.DataFrame(
    {'Q41_Part_3': ['Very important', 'Slightly important', 'Not at all important',
                    'No opinion; I do not know']})
Q41_Part_3['f41_Part_3'] = [3, 2, 1, np.nan]
print(Q41_Part_3)

# Q43 12 13120
# Approximately what percent of your data projects involved exploring unfair bias in the dataset and/or algorithm?
Q43 = pd.DataFrame({'Q43': [
    '0-10', '20-30', '0', '10-20', '30-40', '60-70', '40-50', '90-100', '70-80', '50-60',
    '80-90']})
Q43['f43'] = Q43.Q43.apply(first_number)

# Q46 12 13290
# Approximately what percent of your data projects involve exploring model insights?
Q46 = pd.DataFrame({'Q46': [
    '10-20', '20-30', '0', '50-60', '0-10', '40-50', '90-100', '80-90', '30-40', '70-80',
    '60-70']})
Q46['f46'] = [15, 25, 0, 55, 5, 45, 95, 85, 35, 75, 65]
print(Q46)

# Q48 6 13369
# Do you consider ML models to be "black boxes" with outputs that are difficult or impossible to explain?
Q48 = pd.DataFrame({'Q48': [
    'I view ML models as "black boxes" but I am confident that experts are able to explain model outputs',
    'Yes, most ML models are "black boxes"',
    'I am confident that I can understand and explain the outputs of many but not all ML models',
    'I am confident that I can explain the outputs of most if not all ML models',
    'I do not know; I have no opinion on the matter'
]})
Q48['f48'] = [0, 1, 2, 3, np.nan]
print(Q48)

# Merge ordinal features
responses = pd.read_csv(
    os.path.join(data_dir, 'multipleChoiceResponses.csv'), low_memory=False)
enhanced_responses = responses[1:].copy()
enhanced_responses = enhanced_responses[
    [c for c in enhanced_responses.columns if '_OTHER_TEXT' not in c]].copy()

enhanced_responses = enhanced_responses.merge(Q1, on='Q1', how='left')
enhanced_responses = enhanced_responses.merge(Q2, on='Q2', how='left')
enhanced_responses = enhanced_responses.merge(Q4, on='Q4', how='left')
enhanced_responses = enhanced_responses.merge(Q8, on='Q8', how='left')
enhanced_responses = enhanced_responses.merge(Q9, on='Q9', how='left')
enhanced_responses = enhanced_responses.merge(Q10, on='Q10', how='left')
enhanced_responses = enhanced_responses.merge(Q12_MULTIPLE_CHOICE, on='Q12_MULTIPLE_CHOICE',
                                              how='left')
enhanced_responses = enhanced_responses.merge(Q23, on='Q23', how='left')
enhanced_responses = enhanced_responses.merge(Q24, on='Q24', how='left')
enhanced_responses = enhanced_responses.merge(Q25, on='Q25', how='left')
enhanced_responses = enhanced_responses.merge(Q26, on='Q26', how='left')
enhanced_responses = enhanced_responses.merge(Q40, on='Q40', how='left')
enhanced_responses = enhanced_responses.merge(Q41_Part_1, on='Q41_Part_1', how='left')
enhanced_responses = enhanced_responses.merge(Q41_Part_2, on='Q41_Part_2', how='left')
enhanced_responses = enhanced_responses.merge(Q41_Part_3, on='Q41_Part_3', how='left')
enhanced_responses = enhanced_responses.merge(Q43, on='Q43', how='left')
enhanced_responses = enhanced_responses.merge(Q46, on='Q46', how='left')
enhanced_responses = enhanced_responses.merge(Q48, on='Q48', how='left')

enhanced_responses = enhanced_responses[~enhanced_responses.target.isna()].copy()
print(enhanced_responses.shape)

ordinal_cols = ['Q1', 'Q2', 'Q4', 'Q8', 'Q9', 'Q10', 'Q12_MULTIPLE_CHOICE', 'Q23', 'Q24',
                'Q25', 'Q26', 'Q40', 'Q41_Part_1', 'Q41_Part_2', 'Q41_Part_3', 'Q43', 'Q46',
                'Q48']

percentage_cols = [c for c in enhanced_responses.columns if
                   c.startswith('Q34') or c.startswith('Q35')]
for col in percentage_cols:
    enhanced_responses['f' + col[1:]] = enhanced_responses[col].astype(np.float64)

enhanced_responses['cnt'] = 1
enhanced_responses['probably'] = 1 * (enhanced_responses['target'] > 2)
enhanced_responses['definitely'] = 1 * (enhanced_responses['Q26'] == 'Definitely yes')
print(len(ordinal_cols))
print(ordinal_cols)

print([c for c in enhanced_responses.columns if c.startswith('f')])
print(len([c for c in enhanced_responses.columns if c.startswith('f')]))
print(enhanced_responses.shape)
numeric_features = [c for c in enhanced_responses.columns if c.startswith('f')]
print(numeric_features)

corr = enhanced_responses[numeric_features + ['target']].corr()
corr_cols = np.abs(corr[['target']]).sort_values(by='target').index[-13:]
corr = enhanced_responses[corr_cols].corr()

responses[['Q' + c[1:] for c in corr_cols[:-1]]][:1].columns
responses[['Q' + c[1:] for c in corr_cols[:-1]]][:1].values
corr_labels = pd.DataFrame({
    'Q': ['Q9', 'Q43', 'Q35_Part_2', 'Q24', 'Q48', 'Q4', 'Q12_MULTIPLE_CHOICE', 'Q35_Part_3',
          'Q46', 'Q23', 'Q10', 'Q25'],
    'L': ['Salary', 'Unfair Bias%', 'Online courses%', 'Data Analytics Years', 'Black Box',
          'Education', 'Primary tool', 'Work%', 'Insights%', 'Coding%', 'ML Employer',
          'Machine Learning Years']
})

corr = enhanced_responses[corr_cols].corr(method='pearson').round(2)

xcols = list(corr_labels.L.values) + ['Data Science Confidence']
ycols = list(corr_labels.L.values) + ['Data Science Confidence']

layout = dict(
    title='Ordinal feature correlations',
    width=900,
    height=900,
    #     margin=go.Margin(l=200, r=50, b=50, t=250, pad=4),
    margin=go.layout.Margin(l=200, r=50, b=50, t=250, pad=4),
)
fig = ff.create_annotated_heatmap(
    z=corr.values,
    x=list(xcols),
    y=list(ycols),
    colorscale='Portland',
    #     reversescale=True,
    showscale=True,
    font_colors=['#efecee', '#3c3636'])
fig['layout'].update(layout)
py.iplot(fig, filename='OrdinalCorrelations')
aggr = enhanced_responses.groupby(['f25', 'target'])[['cnt']].sum().reset_index()
aggr.shape
aggr.head()
regr = LinearRegression()
regr.fit(enhanced_responses[['f25']], enhanced_responses.target)
xs = np.array([[x] for x in np.arange(-1, 21)])
ps = regr.predict(xs)
data = [
    go.Scatter(
        y=aggr.target.values,
        x=aggr.f25.values,
        mode='markers',
        marker=dict(
            sizemode='diameter',
            sizeref=1,
            size=np.sqrt(aggr.cnt.values),
            color=aggr.target.values,
            colorscale='Portland',
            reversescale=True,
            showscale=False),
        text=aggr['cnt'].values
    ),
    go.Scatter(
        x=xs[:, 0],
        y=ps,
        mode='lines',
        line=dict(color='black', width=3, dash='dash')
    )
]
layout = go.Layout(
    autosize=True,
    title='Even the strongest correlation is not really strong',
    hovermode='closest',
    xaxis=dict(title='Machine Learning Experience (years)', ticklen=5,
               showgrid=False, zeroline=False, showline=False, range=[-1.5, 6.5]),
    yaxis=dict(title='Data Science Confidence', showgrid=False,
               zeroline=False, ticklen=5, gridwidth=2,
               tickvals=np.arange(5),
               ticktext=['No Way', 'Unlikely', 'Maybe?', 'Probably', 'Hell Yeah!']),
    showlegend=False
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='scatter_countries')
country_mean = enhanced_responses.groupby('Q3')[['target', 'probably', 'definitely']].mean()
country_cnt = enhanced_responses.groupby('Q3')[['cnt']].sum()
country_stats = pd.merge(country_mean, country_cnt, left_index=True,
                         right_index=True).reset_index()
country_stats.sort_values(by='cnt')
country_stats.mean()
y = 'probably'
data = [dict(
    type='choropleth',
    locations=country_stats['Q3'],
    locationmode='country names',
    z=country_stats[y],
    text=country_stats['Q3'],
    colorscale='Portland',
    reversescale=True,
    marker=dict(line=dict(color='rgb(180,180,180)', width=0.5)),
    colorbar=dict(autotick=False, tickprefix='', title=y),
)]
layout = dict(
    title='A pretty (useless) map about Data Scientist Confidence',
    geo=dict(showframe=False, showcoastlines=True, projection=dict(type='Mercator'))
)
fig = dict(data=data, layout=layout)
py.iplot(fig, validate=False, filename='world-map')
col_stats = pd.DataFrame({'name': enhanced_responses.columns,
                          'nunique': enhanced_responses.nunique().values,
                          'count': enhanced_responses.count().values,
                          })
col_stats['Q'] = col_stats.name.apply(lambda s: s.split('_')[0])
col_stats = col_stats[col_stats.Q.str.startswith('Q')]
multi_question_columns = col_stats.groupby('Q')[['count']].count().reset_index()
multi_question_columns = multi_question_columns[multi_question_columns['count'] > 1]

col_stats.head(20)
col_stats.shape
multi_question_columns.shape
multi_question_columns
# Count of multiple choices
for q, c in multi_question_columns.values:
    if q not in ordinal_cols:
        qs = col_stats[col_stats.Q == q].name.values
        enhanced_responses['f{}_answers'.format(q[1:])] = enhanced_responses[qs].count(axis=1)

# Binary columns
binary_cols = col_stats[col_stats['nunique'] == 1]
for col in binary_cols.name.values:
    enhanced_responses['f{}_binary'.format(col[1:])] = 1 - (1 * enhanced_responses[col].isna())

# One Hot Encoded categorical columns
categorical_cols = col_stats[(col_stats['nunique'] > 1) & (col_stats['nunique'] < 100)]
categorical_cols = categorical_cols[~categorical_cols.Q.isin(ordinal_cols)]
for col in categorical_cols.name.values:
    df = pd.get_dummies(enhanced_responses[col].values)
    for i in range(df.shape[1]):
        enhanced_responses['f{}_v_{}'.format(col[1:], i)] = df.values[:, i]
features = [c for c in enhanced_responses.columns if c.startswith('f')]
'We have {} respondents and {} features.'.format(enhanced_responses.shape[0], len(features))
# XGB
fix = {'nthread': 3, 'booster': 'gbtree', 'silent': 1, 'eval_metric': 'auc',
       'objective': 'binary:logistic'}
config = dict(min_child_weight=10, eta=0.05, colsample_bytree=0.5, max_depth=6, subsample=0.8)
config.update(fix)

X = enhanced_responses[features].values
y = enhanced_responses['probably'].values
Xtr, Xv, ytr, yv = train_test_split(X, y, test_size=0.25, random_state=42)
print(Xtr.shape, Xv.shape, y.mean())

fs = ['f%i' % i for i in range(len(features))]
dtrain = xgb.DMatrix(Xtr, label=ytr, feature_names=fs)
dvalid = xgb.DMatrix(Xv, label=yv, feature_names=fs)

xgb_probably = xgb.train(config, dtrain, 500, [(dtrain, 'train'), (dvalid, 'valid')],
                         early_stopping_rounds=20, maximize=True, verbose_eval=50)
fpr_xgb_probably, tpr_xgb_probably, thresholds = metrics.roc_curve(yv, xgb_probably.predict(
    dvalid))
auc_xgb_probably = metrics.auc(fpr_xgb_probably, tpr_xgb_probably)
auc_xgb_probably
# Logistic Regression
X = enhanced_responses[features].values
y = enhanced_responses['probably'].values
X[np.isnan(X)] = -1
scaler = StandardScaler()
X = scaler.fit_transform(X)
Xtr, Xv, ytr, yv = train_test_split(X, y, test_size=0.25, random_state=42)

lr = LogisticRegression(solver='lbfgs')
lr.fit(Xtr, ytr)
fpr_lr_probably, tpr_lr_probably, _ = metrics.roc_curve(yv, lr.predict_proba(Xv)[:, 1])
auc_lr_probably = metrics.auc(fpr_lr_probably, tpr_lr_probably)
auc_lr_probably
# Random Forest
X = enhanced_responses[features].values
y = enhanced_responses['probably'].values
X[np.isnan(X)] = -999
Xtr, Xv, ytr, yv = train_test_split(X, y, test_size=0.25, random_state=42)
rf = RandomForestClassifier(
    n_estimators=500, max_depth=10, max_features=0.3,
    min_samples_split=5, n_jobs=3, verbose=1)
rf.fit(Xtr, ytr)
fpr_rf_probably, tpr_rf_probably, _ = metrics.roc_curve(yv, rf.predict_proba(Xv)[:, 1])
auc_rf_probably = metrics.auc(fpr_rf_probably, tpr_rf_probably)
auc_rf_probably
# XGB
X = enhanced_responses[features].values
y = enhanced_responses['definitely'].values
Xtr, Xv, ytr, yv = train_test_split(X, y, test_size=0.25, random_state=42)
print(Xtr.shape, Xv.shape, y.mean())
fs = ['f%i' % i for i in range(len(features))]
dtrain = xgb.DMatrix(Xtr, label=ytr, feature_names=fs)
dvalid = xgb.DMatrix(Xv, label=yv, feature_names=fs)
xgb_definitely = xgb.train(config, dtrain, 500, [(dtrain, 'train'), (dvalid, 'valid')],
                           early_stopping_rounds=20, maximize=True, verbose_eval=50)
fpr_xgb_definitely, tpr_xgb_definitely, _ = metrics.roc_curve(yv, xgb_definitely.predict(dvalid))
auc_xgb_definitely = metrics.auc(fpr_xgb_definitely, tpr_xgb_definitely)
auc_xgb_definitely
data = [
    go.Scatter(x=fpr_xgb_probably, y=tpr_xgb_probably, mode='lines',
               name='XGB probably', line=dict(width=4), opacity=0.8),
    go.Scatter(x=fpr_xgb_definitely, y=tpr_xgb_definitely, mode='lines',
               name='XGB definitely', line=dict(width=4), opacity=0.8),
    go.Scatter(x=fpr_lr_probably, y=tpr_lr_probably, mode='lines',
               name='LR probably', line=dict(width=4), opacity=0.8),
    go.Scatter(x=fpr_rf_probably, y=tpr_rf_probably, mode='lines',
               name='RF probably', line=dict(width=4), opacity=0.8),
]
layout = go.Layout(
    title='Similar classifier performance (AUC ~ 0.78)',
    xaxis=dict(title='FPR', ticklen=5, zeroline=False, gridwidth=2),
    yaxis=dict(title='TPR', ticklen=5, gridwidth=2),
    showlegend=True,
    height=700,
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='users')
competitions = pd.read_csv(os.path.join(META_DATA_PATH, 'Competitions.csv'))
competitions = competitions.rename(columns={'Id': 'CompetitionId'})
competitions = competitions[competitions.HostSegmentTitle != 'InClass']
competitions = competitions[competitions.EvaluationAlgorithmAbbreviation == 'AUC']
competitions = competitions[competitions.RewardType == 'USD']
competitions = competitions[competitions.FinalLeaderboardHasBeenVerified]
competitions = competitions[competitions.CanQualifyTiers]
competitions = competitions[competitions.RewardQuantity > 1000]
competitions = competitions[competitions.TotalTeams > 200]
competitions = competitions[['CompetitionId', 'Slug', 'Title', 'DeadlineDate',
                             'RewardQuantity', 'TotalTeams']]
competitions.shape
competitions.sort_values(by='TotalTeams', ascending=False)
submissions = pd.read_csv(os.path.join(META_DATA_PATH, 'Submissions.csv'),
                          usecols=['TeamId', 'IsAfterDeadline', 'PrivateScoreFullPrecision'],
                          low_memory=False)
submissions = submissions[~submissions.IsAfterDeadline]
submissions['PrivateScoreFullPrecision'] = submissions.PrivateScoreFullPrecision.astype(np.float64)
submissions = submissions[submissions.PrivateScoreFullPrecision < 0.999]
submissions = submissions[submissions.PrivateScoreFullPrecision > 0.5]
best_submissions = submissions.groupby('TeamId')[['PrivateScoreFullPrecision']].max().reset_index()

teams = pd.read_csv(os.path.join(META_DATA_PATH, 'Teams.csv'),
                   usecols=['Id', 'CompetitionId', 'PrivateLeaderboardRank'])
teams = teams.rename(columns={'Id': 'TeamId'})
team_results = teams.merge(best_submissions, on='TeamId')
best_competition_results = team_results.groupby('CompetitionId')[['PrivateScoreFullPrecision']].max().reset_index()
competitions = competitions.merge(best_competition_results, on='CompetitionId')
competitions = competitions.sort_values(by='PrivateScoreFullPrecision')
competitions
results = pd.DataFrame({
    'Title': list(competitions.Title.values) + ['Data Science Confidence'],
    'AUC': list(competitions.PrivateScoreFullPrecision.values) + [auc_xgb_probably]
})
results = results.sort_values(by='AUC')
results.index = np.arange(len(results))

data = [
    go.Bar(
        y=results['AUC'].values,
        x=results.index,
        marker=dict(
            color=['black' if title == 'Data Science Confidence' else '#00BBFF' for title in results.Title.values],
        ),
        text=results.Title.values,
    )
]
layout = go.Layout(
    autosize=True,
    title='Most kaggle competitions are easier than predicting Data Science Confidence :)',
    hovermode='closest',
    xaxis=dict(title='Kaggle competitions', ticklen=0, zeroline=False, gridwidth=0),
    yaxis=dict(title='AUC', ticklen=5, gridwidth=2),
    showlegend=False
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='KaggleCompetitionBenchmark')
question_desc = pd.DataFrame([
    ['Q1', 'Gender'],
    ['Q10', 'ML employer'],
    ['Q11', 'Work activity'],
    ['Q12', 'Primary tool'],
    ['Q13', 'IDE'],
    ['Q14', 'Hosted notebook'],
    ['Q15', 'Cloud computing service'],
    ['Q16', 'Programming languages'],
    ['Q17', 'Top language'],
    ['Q18', 'Recommended language'],
    ['Q19', 'ML framework'],
    ['Q2', 'Age'],
    ['Q20', 'Top ML library'],
    ['Q21', 'Visualization'],
    ['Q22', 'Top visualization'],
    ['Q23', 'Code%'],
    ['Q24', 'Analysis experience'],
    ['Q25', 'ML experience'],
    ['Q26', 'DS confidence'],
    ['Q27', 'Cloud computing product'],
    ['Q28', 'ML product'],
    ['Q29', 'RDB'],
    ['Q3', 'Country'],
    ['Q30', 'Big Data'],
    ['Q31', 'Data type'],
    ['Q32', 'Top data type'],
    ['Q33', 'Public datasets'],
    ['Q34', 'Cleaning%'],
    ['Q35', 'Self-taught%'],
    ['Q36', 'Online Platform'],
    ['Q37', 'Top online platform'],
    ['Q38', 'Media source'],
    ['Q39', 'MOOC'],
    ['Q4', 'Education'],
    ['Q40', 'Independent project'],
    ['Q41', 'Fairness'],
    ['Q42', 'Metrics'],
    ['Q43', 'Unfair bias%'],
    ['Q44', 'Unfair bias difficulty'],
    ['Q45', 'Model insights circumstances'],
    ['Q46', 'Model insights%'],
    ['Q47', 'Model insights methods'],
    ['Q48', 'Black box'],
    ['Q49', 'Reproducibility tools'],
    ['Q5', 'Major'],
    ['Q50', 'Reproducibility barriers'],
    ['Q6', 'Role'],
    ['Q7', 'Industry'],
    ['Q8', 'Experience'],
    ['Q9', 'Compensation'],
], columns=['Q', 'short_description'])
question_desc['QDesc'] = question_desc['Q'] + ' - ' + question_desc['short_description']
question_desc
feature_importance = xgb_probably.get_fscore()
f1 = pd.DataFrame({'f': list(feature_importance.keys()), 'xgb_imp': list(feature_importance.values())})
f2 = pd.DataFrame({'f': fs, 'feature': features})
feature_importance = pd.merge(f1, f2, how='outer', on='f')
rf_imp = f1 = pd.DataFrame({'feature': features, 'rf_imp': rf.feature_importances_})
feature_importance = pd.merge(feature_importance, rf_imp, how='outer', on='feature')
feature_importance = feature_importance.fillna(0)
feature_importance['xgb_imp'] = feature_importance.xgb_imp.values / feature_importance.xgb_imp.sum()
feature_importance['Q'] = feature_importance.feature.apply(lambda s: 'Q' + s.split('_')[0][1:])
feature_importance = feature_importance.merge(question_desc, on='Q')
feature_importance = feature_importance.sort_values(by='xgb_imp', ascending=False)

feature_importance.shape
feature_importance.head()
feature_importance.sum()
question_importance = feature_importance.groupby('QDesc').sum().reset_index()
question_importance['imp'] = (question_importance.xgb_imp + question_importance.rf_imp) / 2
question_importance = question_importance.sort_values(by='imp', ascending=True)
question_importance.index = np.arange(len(question_importance))
question_importance.head()
question_importance.shape
top25 = question_importance[-25:].copy()
data = [
    go.Bar(
        y=question_importance['QDesc'].values,
        x=question_importance['xgb_imp'].values,
        orientation='h',
        text=top25.QDesc.values,
        name='XGB'
    ),
    go.Bar(
        y=question_importance['QDesc'].values,
        x=question_importance['rf_imp'].values,
        orientation='h',
        text=question_importance.QDesc.values,
        name='RF'
    ),
]
layout = go.Layout(
    height=1000,
    autosize=True,
    title='Question Importance',
    barmode='stack',
    hovermode='closest',
    xaxis=dict(title='Relative Question Importance', ticklen=5, zeroline=False, gridwidth=2, domain=[0.2, 1]),
    yaxis=dict(title='', ticklen=5, gridwidth=2),
    showlegend=True
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='QuestionImportance')
def f2q(f):
    return 'Q' + f.split('_')[0][1:]


bfs_filename = '../input/surveykernelexternals/bfs_result_df.csv'
if not os.path.exists(bfs_filename):
    bfs_result = []

    for q_exclude in tqdm(feature_importance.Q.unique()):
        remaining_features = [f for f in features if f2q(f) != q_exclude]
        X = enhanced_responses[remaining_features].values
        y = enhanced_responses['probably'].values
        X[np.isnan(X)] = -999
        Xtr, Xv, ytr, yv = train_test_split(X, y, test_size=0.25, random_state=42)
        rf_bfs = RandomForestClassifier(n_estimators=500, max_depth=10, max_features=0.3,
                                        min_samples_split=5, n_jobs=3, verbose=0)
        _ = rf_bfs.fit(Xtr, ytr)
        fpr_rf_bfs, tpr_rf_bfs, _ = metrics.roc_curve(yv, rf_bfs.predict_proba(Xv)[:, 1])
        auc_rf_bfs = metrics.auc(fpr_rf_bfs, tpr_rf_bfs)
        bfs_result.append([q_exclude, len(remaining_features), auc_rf_bfs])

    bfs_result_df = pd.DataFrame(bfs_result, columns=['Q', 'n_features_bfs', 'auc_bfs'])
    bfs_result_df.sort_values(by='auc_bfs')
    bfs_result_df.to_csv('bfs_result_df.csv', index=False)
else:
    bfs_result_df = pd.read_csv(bfs_filename)

bfs_result_df_with_desc = bfs_result_df.merge(question_desc, on='Q')
bfs_result_df_with_desc = bfs_result_df_with_desc.sort_values(by='auc_bfs', ascending=False)
data = [
    go.Scatter(
        y = bfs_result_df_with_desc.auc_bfs.values,
        x = np.arange(len(bfs_result_df_with_desc)),
        mode='markers',
        marker=dict(
            sizemode = 'diameter',
            sizeref = 1,
            size= 10,
            color = bfs_result_df_with_desc.auc_bfs.values,
            colorscale='Portland',
            reversescale=True,
            showscale=False),
        text=bfs_result_df_with_desc['QDesc'].values,
        name='RF BFS'
    ),
    go.Scatter(
        x=np.arange(len(bfs_result_df_with_desc)),
        y=auc_rf_probably * np.ones(len(bfs_result_df_with_desc)),
        mode='lines',
        line=dict(color='black', width=3, dash='dash'),
        name='RF all'
    )
]
layout = go.Layout(
    autosize=True,
    title='BFS - removing each questions individually',
    hovermode='closest',
    xaxis= dict(title='Questions', ticklen=5,
                showgrid=False, zeroline=False, showline=False),
    yaxis=dict(title='Model Performance (AUC)', showgrid=False,
               zeroline=False, ticklen=5, gridwidth=2),
    showlegend=True,

)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='bfs')
bfs_result_df_with_desc.head()
ffs_filename = '../input/surveykernelexternals/ffs_result_df.csv'
if not os.path.exists(ffs_filename):
    ffs_result = []
    best_auc = 0.
    best_candidate =''
    selected_questions = []
    for n in range(10):
        possible_questions = np.setdiff1d(feature_importance.Q.unique(), selected_questions)
        for q_candidate in tqdm(possible_questions):
            current_features = [f for f in features if f2q(f) in selected_questions + [q_candidate]]
            X = enhanced_responses[current_features].values
            y = enhanced_responses['probably'].values
            X[np.isnan(X)] = -999
            Xtr, Xv, ytr, yv = train_test_split(X, y, test_size=0.25, random_state=42)
            rf_ffs = RandomForestClassifier(n_estimators=500, max_depth=10, max_features=0.3,
                                            min_samples_split=5, n_jobs=3, verbose=0)
            _ = rf_ffs.fit(Xtr, ytr)
            fpr_rf_ffs, tpr_rf_ffs, _ = metrics.roc_curve(yv, rf_ffs.predict_proba(Xv)[:, 1])
            auc_rf_ffs = metrics.auc(fpr_rf_ffs, tpr_rf_ffs)
            ffs_result.append([n, q_candidate, len(selected_questions),
                               len(current_features), auc_rf_ffs])
            if auc_rf_ffs > best_auc:
                best_auc = auc_rf_ffs
                best_candidate = q_candidate
        selected_questions.append(best_candidate)
    
    ffs_result_df = pd.DataFrame(ffs_result, columns=['n', 'Q', 'nsq', 'nsf', 'auc_ffs'])
    ffs_result_df.sort_values(by='auc_ffs')
    ffs_result_df.to_csv(ffs_filename, index=False)
else:
    ffs_result_df = pd.read_csv(ffs_filename)

ffs_result_df_with_desc = ffs_result_df.merge(question_desc, on='Q')
ffs_result_df.shape
ffs_result_df.head()
data = [
    go.Scatter(
        x = ffs_result_df_with_desc.n.values,
        y = ffs_result_df_with_desc.auc_ffs.values,
        mode='markers',
        marker=dict(
            sizemode = 'diameter',
            sizeref = 1,
            size= 10,
            color = ffs_result_df_with_desc.auc_ffs.values,
            colorscale='Portland',
            reversescale=True,
            showscale=False),
        text=ffs_result_df_with_desc['QDesc'].values,
        name='RF FFS'
    ),
    go.Scatter(
        x=np.arange(10),
        y=auc_rf_probably * np.ones(10),
        mode='lines',
        line=dict(color='black', width=3, dash='dash'),
        name='RF All'
    )
]
layout = go.Layout(
    autosize=True,
    title='You do not really need all the features',
    hovermode='closest',
    xaxis= dict(title='Number of questions', ticklen= 5,
                showgrid=True, zeroline=False, showline=False),
    yaxis=dict(title='Model Performance (AUC)', showgrid=True,
               zeroline=False, ticklen=5, gridwidth=2),
    showlegend=True,
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='ffs')
end = dt.datetime.now()
print('Latest run {}.\nTotal time {}s'.format(end, (end - start).seconds))