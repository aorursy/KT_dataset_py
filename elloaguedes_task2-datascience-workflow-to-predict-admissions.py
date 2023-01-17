import datetime
print("Last Update: ",datetime.datetime.now())
# Organizing imports
import numpy as np
import pandas as pd

# Matplotlib
import matplotlib
from matplotlib import pyplot as plt

# plotly packages
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.graph_objs import *
data = pd.read_excel('/kaggle/input/covid19/dataset.xlsx')
data.head()
print("Data Samples: ", len(data))
data['SARS-Cov-2 exam result'] = (data['SARS-Cov-2 exam result'] == 'positive').astype(bool)
covid_positive = data[data['SARS-Cov-2 exam result'] == True]
num_positive = len(covid_positive)
print(f'In the dataset, there are {num_positive} entries of patients infected with SARS-CoV-2')
print(f'This number corresponds to around {round((num_positive / len(data)) * 100)}% of the dataset')
regular_ward = data[data['Patient addmited to regular ward (1=yes, 0=no)'] == 1]
semi_intensive = data[data['Patient addmited to semi-intensive unit (1=yes, 0=no)'] == 1]
icu = data[data['Patient addmited to intensive care unit (1=yes, 0=no)'] == 1]

print("From all patients, regardless of Covid-19 test result: ")
print(f'{len(regular_ward)} patients were admitted in a regular ward.')
print(f'{len(semi_intensive)} patients were admitted in a semi intensive care unit.')
print(f'{len(icu)} patients were admitted in an intensive care unit.')
pos_regular_ward = covid_positive[covid_positive['Patient addmited to regular ward (1=yes, 0=no)'] == 1]
pos_semi_intensive = covid_positive[covid_positive['Patient addmited to semi-intensive unit (1=yes, 0=no)'] == 1]
pos_icu = covid_positive[covid_positive['Patient addmited to intensive care unit (1=yes, 0=no)'] == 1]

print("From Covid-19 patients: ")
print(f'{len(pos_regular_ward)} patients were admitted in a regular ward.')
print(f'{len(pos_semi_intensive)} patients were admitted in a semi intensive care unit.')
print(f'{len(pos_icu)} patients were admitted in an intensive care unit.')
pos_admitted = sum(map(len, [pos_regular_ward, pos_semi_intensive, pos_icu]))
total_admitted = sum(map(len, [regular_ward, semi_intensive, icu]))
proportion_admitted = round((pos_admitted / total_admitted) * 100)

print(f'About {proportion_admitted}% of the admitted patients have tested positive for the virus.')
data = covid_positive
print("Data Samples: ", len(data))
def multiclass_target(row):
    check = 0
    check += 1 if (row['Patient addmited to regular ward (1=yes, 0=no)'] == 1) else 0
    check += 2 if (row['Patient addmited to semi-intensive unit (1=yes, 0=no)'] == 1) else 0
    check += 3 if (row['Patient addmited to intensive care unit (1=yes, 0=no)'] == 1) else 0
    row['target'] = check
    return row

data = data.apply(multiclass_target, axis=1)
data.head()
data.drop([
    'Patient addmited to regular ward (1=yes, 0=no)',
    'Patient addmited to semi-intensive unit (1=yes, 0=no)',
    'Patient addmited to intensive care unit (1=yes, 0=no)'], axis=1, inplace=True)
data.head()
def plot_histogram(df, feature_name, title, params={}):
    layout = Layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        title = title
    )
    fig = px.histogram(df, x=feature_name, **params)
    fig['layout'].update(layout)
    fig.show()
def percentage_missing(row, total_samples = 5644):
    missing = row[0] / total_samples
    row['missing'] = missing
    return row

def get_missing_df(original_df):
    null_df = pd.DataFrame(original_df.copy().isnull().sum()).reset_index()
    null_df = null_df[null_df[0] != 0].reset_index(drop=True)
    null_df = null_df.apply(percentage_missing, total_samples=len(original_df), axis=1)
    null_df.columns = ['Exam name', 'Number of missing results', 'Percentage of missing results']
    null_df = null_df
    return null_df
missing_total = get_missing_df(data)
missing_total.T
plot_histogram(missing_total, 'Percentage of missing results', 'Missing data histogram', { 'nbins': 20 })
columns_to_remove = missing_total.loc[missing_total['Percentage of missing results'] >= 0.90]['Exam name']
len(columns_to_remove)
data.drop(columns_to_remove,axis=1,inplace=True)
data.shape
for exam in list(data.columns):
    if exam not in list(missing_total['Exam name']):
        print(exam)
data.dtypes
data['Influenza A'].unique()
infection_vars = [
    'Respiratory Syncytial Virus', 'Influenza A', 'Influenza B',
    'Parainfluenza 1', 'CoronavirusNL63', 'Rhinovirus/Enterovirus',
    'Coronavirus HKU1', 'Parainfluenza 3', 'Chlamydophila pneumoniae',
    'Adenovirus', 'Parainfluenza 4', 'Coronavirus229E', 'CoronavirusOC43',
    'Inf A H1N1 2009', 'Bordetella pertussis', 'Metapneumovirus',
    'Parainfluenza 2', 'Influenza B, rapid test', 'Influenza A, rapid test'
]
update_mapping = {
    'not_detected': 0,
    'negative': 0,
    'detected': 1,
    'positive': 1
}

for column in infection_vars:
    data[column] = data[column].map(update_mapping)
    data[column] = data[column].astype(float)
def quantile_classification(row):
    if row['Patient age quantile'] <= 5:
        return "Infant"
    elif 5 < row['Patient age quantile'] <= 10:
        return "Young Adult"
    elif 10 < row['Patient age quantile'] <= 15:
        return "Adult"
    else:
        return "Elder"
    
data['Age Group'] = data.apply(lambda row: quantile_classification(row), axis=1)
data.head()
plot_histogram(data, 'Patient age quantile', 'Patient age quantile histogram',{"color":"Age Group"})
plot_histogram(data, 'Age Group', 'Patient age group histogram',{"color":"Age Group"})
data["Patient age quantile"].corr(data['target'])
import plotly.graph_objects as go


df2 = data.groupby(["Age Group"])['target'].value_counts()
z = [[114,1,0,1],[157,4,0,0],[148,15,5,2],[87,16,3,5]] ## copied by hand from df2

fig = go.Figure(data=go.Heatmap(z=z,y=['Infant', 'Young Adult', 'Adult', 'Elder'],
                                x=['No Admission', 'Regular Ward', 'Semi-Intensive Ward','Intensive Ward'],
                                hoverongaps = False))
    
fig.show()
df2 = data.groupby(["target"])['target'].agg('count')
df2[0]/sum(df2)
plot_histogram(data, 'target', 'Target label balancing',{"color":"target","nbins":4})

data_distribution = pd.DataFrame(columns=['Data Column', 'Value'])

for column in data.columns:
    if column not in missing_total['Exam name'].unique() and column != 'Patient age quantile': continue
    for value in data[column].values:
        data_distribution = data_distribution.append({ 'Data Column': column, 'Value': value }, ignore_index=True)
fig = px.box(data_distribution, x='Data Column', y='Value')
fig.show()
