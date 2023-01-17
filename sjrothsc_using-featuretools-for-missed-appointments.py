import numpy as np
import pandas as pd
import featuretools as ft
print('Featuretools version {}'.format(ft.__version__))

# Data Wrangling
# After loading the data with pandas, we /have/ to fix typos in some column names
# but we change others as well to suit personal preference.
data = pd.read_csv("../input/KaggleV2-May-2016.csv", parse_dates=['AppointmentDay', 'ScheduledDay'])
data.index = data['AppointmentID']
data.rename(columns = {'Hipertension': 'hypertension',
                       'Handcap': 'handicap',
                       'PatientId': 'patient_id',
                       'AppointmentID': 'appointment_id',
                       'ScheduledDay': 'scheduled_time',
                       'AppointmentDay': 'appointment_day',
                       'Neighbourhood': 'neighborhood',
                       'No-show': 'no_show'}, inplace = True)
for column in data.columns:
    data.rename(columns = {column: column.lower()}, inplace = True)
data['appointment_day'] = data['appointment_day'] + pd.Timedelta('1d') - pd.Timedelta('1s')

data['no_show'] = data['no_show'].map({'No': False, 'Yes': True})

# Show the size of the data in a print statement
print('{} Appointments, {} Columns'.format(data.shape[0], data.shape[1]))
print('Appointments: {}'.format(data.shape[0]))
print('Schedule times: {}'.format(data.scheduled_time.nunique()))
print('Patients: {}'.format(data.patient_id.nunique()))
print('Neighborhoods: {}'.format(data.neighborhood.nunique()))
pd.options.display.max_columns=100 
pd.options.display.float_format = '{:.2f}'.format


data.head(3)
import featuretools.variable_types as vtypes
# This is all of the code from the notebook
# No need to run/read this cell if you're running everything else

# List the semantic type for each column
variable_types = {'gender': vtypes.Categorical,
                  'patient_id': vtypes.Categorical,
                  'age': vtypes.Ordinal,
                  'scholarship': vtypes.Boolean,
                  'hypertension': vtypes.Boolean,
                  'diabetes': vtypes.Boolean,
                  'alcoholism': vtypes.Boolean,
                  'handicap': vtypes.Boolean,
                  'no_show': vtypes.Boolean,
                  'sms_received': vtypes.Boolean}

# Use those variable types to make an EntitySet and Entity from that table
es = ft.EntitySet('Appointments')
es = es.entity_from_dataframe(entity_id="appointments",
                              dataframe=data,
                              index='appointment_id',
                              time_index='scheduled_time',
                              secondary_time_index={'appointment_day': ['no_show', 'sms_received']},
                              variable_types=variable_types)

# Add a patients entity with patient-specific variables
es.normalize_entity('appointments', 'patients', 'patient_id',
                    additional_variables=['scholarship',
                                          'hypertension',
                                          'diabetes',
                                          'alcoholism',
                                          'handicap'])

# Make locations, ages and genders
es.normalize_entity('appointments', 'locations', 'neighborhood',
                    make_time_index=False)
es.normalize_entity('appointments', 'ages', 'age',
                    make_time_index=False)
es.normalize_entity('appointments', 'genders', 'gender',
                    make_time_index=False)

# Take the index and the appointment time to use as a cutoff time
cutoff_times = es['appointments'].df[['appointment_id', 'scheduled_time', 'no_show']].sort_values(by='scheduled_time')

# Rename cutoff time columns to avoid confusion
cutoff_times.rename(columns = {'scheduled_time': 'cutoff_time', 
                               'no_show': 'label'},
                    inplace = True)

# Make feature matrix from entityset/cutoff time pair
fm_final, _ = ft.dfs(entityset=es,
                      target_entity='appointments',
                      agg_primitives=['count', 'percent_true'],
                      trans_primitives=['is_weekend', 'weekday', 'day', 'month', 'year'],
                      approximate='3h',
                      max_depth=3,
                      cutoff_time=cutoff_times[20000:],
                      verbose=False)

print('Features: {}, Rows: {}'.format(fm_final.shape[1], fm_final.shape[0]))
fm_final.tail(3)
# List the semantic type for each column

import featuretools.variable_types as vtypes
variable_types = {'gender': vtypes.Categorical,
                  'patient_id': vtypes.Categorical,
                  'age': vtypes.Ordinal,
                  'scholarship': vtypes.Boolean,
                  'hypertension': vtypes.Boolean,
                  'diabetes': vtypes.Boolean,
                  'alcoholism': vtypes.Boolean,
                  'handicap': vtypes.Boolean,
                  'no_show': vtypes.Boolean,
                  'sms_received': vtypes.Boolean}
# Make an entity named 'appointments' which stores dataset metadata with the dataframe
es = ft.EntitySet('Appointments')
es = es.entity_from_dataframe(entity_id="appointments",
                              dataframe=data,
                              index='appointment_id',
                              time_index='scheduled_time',
                              secondary_time_index={'appointment_day': ['no_show', 'sms_received']},
                              variable_types=variable_types)
es['appointments']
# Make a patients entity with patient-specific variables
es.normalize_entity('appointments', 'patients', 'patient_id',
                    additional_variables=['scholarship',
                                          'hypertension',
                                          'diabetes',
                                          'alcoholism',
                                          'handicap'])

# Make locations, ages and genders
es.normalize_entity('appointments', 'locations', 'neighborhood',
                    make_time_index=False)
es.normalize_entity('appointments', 'ages', 'age',
                    make_time_index=False)
es.normalize_entity('appointments', 'genders', 'gender',
                    make_time_index=False)

# Show the patients entity
es['patients'].df.head(2)
# Take the index and the appointment time to use as a cutoff time
cutoff_times = es['appointments'].df[['appointment_id', 'scheduled_time', 'no_show']].sort_values(by='scheduled_time')

# Rename columns to avoid confusion
cutoff_times.rename(columns = {'scheduled_time': 'cutoff_time', 
                               'no_show': 'label'},
                    inplace = True)
# Generate features using the constructed entityset
fm, features = ft.dfs(entityset=es,
                      target_entity='appointments',
                      agg_primitives=['count', 'percent_true'],
                      trans_primitives=['is_weekend', 'weekday', 'day', 'month', 'year'],
                      max_depth=3, 
                      approximate='3h',
                      cutoff_time=cutoff_times[20000:],
                      verbose=True)
fm.tail()
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve

X = fm.copy().fillna(0)
label = X.pop('label')
X = X.drop(['patient_id', 'neighborhood', 'gender'], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, label, test_size=0.30, shuffle=False)
clf = RandomForestClassifier(n_estimators=150)
clf.fit(X_train, y_train)
probs = clf.predict_proba(X_test)
print('AUC score of {:.2f}'.format(roc_auc_score(y_test, probs[:,1])))
feature_imps = [(imp, X.columns[i]) for i, imp in enumerate(clf.feature_importances_)]
feature_imps.sort()
feature_imps.reverse()
print('Random Forest Feature Importances:')
for i, f in enumerate(feature_imps[0:8]):
    print('{}: {} [{:.3f}]'.format(i + 1, f[1], f[0]/feature_imps[0][0]))

from bokeh.models import HoverTool
from bokeh.models.sources import ColumnDataSource
from bokeh.plotting import figure
from bokeh.io import show, output_notebook
from bokeh.layouts import gridplot
from sklearn.metrics import precision_score, recall_score, f1_score

def plot_roc_auc(y_test, probs, pos_label=1):
    fpr, tpr, thresholds = roc_curve(y_test, 
                                     probs[:, 1], 
                                     pos_label=pos_label)


    output_notebook()
    p = figure(height=400, width=400)
    p.line(x=fpr, y=tpr)
    p.title.text ='Receiver operating characteristic'
    p.xaxis.axis_label = 'False Positive Rate'
    p.yaxis.axis_label = 'True Positive Rate'

    p.line(x=fpr, y=fpr, color='red', line_dash='dashed')
    return(p)

def plot_f1(y_test, probs, nprecs):
    threshes = [x/1000. for x in range(50, nprecs)]
    precisions = [precision_score(y_test, probs[:,1] > t) for t in threshes]
    recalls = [recall_score(y_test, probs[:,1] > t) for t in threshes]
    fones = [f1_score(y_test, probs[:,1] > t) for t in threshes]
    
    output_notebook()
    p = figure(height=400, width=400)
    p.line(x=threshes, y=precisions, color='green', legend='precision')
    p.line(x=threshes, y=recalls, color='blue', legend='recall')
    p.line(x=threshes, y=fones, color='red', legend='f1')
    p.xaxis.axis_label = 'Threshold'
    p.title.text = 'Precision, Recall, and F1 by Threshold'
    return(p)

def plot_kfirst(ytest, probs, firstk=500):
    A = pd.DataFrame(probs)
    A['y_test'] = y_test.values
    krange = range(firstk)
    firstk = []
    for K in krange:
        a = A[1][:K]
        a = [1 for prob in a]
        b = A['y_test'][:K]
        firstk.append(precision_score(b, a))
    
    output_notebook()
    p = figure(height=400, width=400)
    p.step(x=krange, y=firstk)
    p.xaxis.axis_label = 'Predictions sorted by most likely'
    p.yaxis.axis_label = 'Precision'
    p.title.text = 'K-first'
    p.yaxis[0].formatter.use_scientific = False
    return p

p1 = plot_roc_auc(y_test, probs)
p2 = plot_f1(y_test, probs, 1000)
p3 = plot_kfirst(y_test, probs, 300)
show(gridplot([p1, p2, p3], ncols=1))
tmp = fm.groupby('neighborhood').apply(lambda df: df.tail(1))['locations.COUNT(appointments)'].sort_values().reset_index().reset_index()
hover = HoverTool(tooltips=[
    ("Count", "@{locations.COUNT(appointments)}"),
    ("Place", "@neighborhood"),
])
source = ColumnDataSource(tmp)
p4 = figure(width=400, 
           height=400,
           tools=[hover, 'box_zoom', 'reset', 'save'])
p4.scatter('index', 'locations.COUNT(appointments)', alpha=.7, source=source, color='teal')
p4.title.text = 'Appointments by Neighborhood'
p4.xaxis.axis_label = 'Neighborhoods (hover to view)'
p4.yaxis.axis_label = 'Count'

tmp = fm.groupby('neighborhood').apply(lambda df: df.tail(1))[['locations.COUNT(appointments)', 
                                                               'locations.PERCENT_TRUE(appointments.no_show)']].sort_values(
    by='locations.COUNT(appointments)').reset_index().reset_index()
hover = HoverTool(tooltips=[
    ("Prob", "@{locations.PERCENT_TRUE(appointments.no_show)}"),
    ("Place", "@neighborhood"),
])
source = ColumnDataSource(tmp)
p5 = figure(width=400, 
           height=400,
           tools=[hover, 'box_zoom', 'reset', 'save'])
p5.scatter('index', 'locations.PERCENT_TRUE(appointments.no_show)', alpha=.7, source=source, color='maroon')
p5.title.text = 'Probability of no-show by Neighborhood'
p5.xaxis.axis_label = 'Neighborhoods (hover to view)'
p5.yaxis.axis_label = 'Probability of no-show'

tmp = fm.tail(5000).groupby('age').apply(lambda df: df.tail(1))[['ages.COUNT(appointments)']].sort_values(
    by='ages.COUNT(appointments)').reset_index().reset_index()
hover = HoverTool(tooltips=[
    ("Count", "@{ages.COUNT(appointments)}"),
    ("Age", "@age"),
])
source = ColumnDataSource(tmp)
p6 = figure(width=400, 
           height=400,
           tools=[hover, 'box_zoom', 'reset', 'save'])
p6.scatter('age', 'ages.COUNT(appointments)', alpha=.7, source=source, color='magenta')
p6.title.text = 'Appointments by Age'
p6.xaxis.axis_label = 'Age'
p6.yaxis.axis_label = 'Count'

source = ColumnDataSource(X.tail(5000).groupby('age').apply(lambda x: x.tail(1)))

hover = HoverTool(tooltips=[
    ("Prob", "@{ages.PERCENT_TRUE(appointments.no_show)}"),
    ("Age", "@age"),
])

p7 = figure(title="Probability no-show by Age", 
           x_axis_label='Age', 
           y_axis_label='Probability of no-show',
           width=400,
           height=400,
           tools=[hover, 'box_zoom', 'reset', 'save']
)

p7.scatter('age', 'ages.PERCENT_TRUE(appointments.no_show)', 
          alpha=.7, 
          source=source)


show(gridplot([p4, p6, p5, p7], ncols=2))