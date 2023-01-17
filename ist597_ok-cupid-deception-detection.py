import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

import seaborn as sns # data visualization library  

import plotly.graph_objs as go

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected=True)

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

# Loading data

path_file = '../input/profiles.csv'



df = pd.read_csv(open(path_file, 'r'))



list_df = []

df = df[df.status != "unknown"]

# df.dropna()

# df = df.rename(columns = {'fit': 'fit_feature'})



df['status'] = df['status'].replace("seeing someone", "married")

df['status'] = df['status'].replace("available", "single")



df['status'] = df['status'].replace("single", 0)

df['status'] = df['status'].replace("married", 1)



# df_single = df[df['status'] == 0].sample(frac=0.04124)

# df_married = df[df['status'] == 1]



# df = pd.concat([df_single, df_married], axis=1)

df.head()
print('Percent of bisexual in Married', 100.0* df[(df.orientation == 'bisexual') & (df.status == 1)].shape[0]/float(df[(df.status==1)].shape[0]))

print('Percent of bisexual in Single', 100.0* df[(df.orientation == 'bisexual') & (df.status == 0)].shape[0]/float(df[(df.status==0)].shape[0]))

print('Percent of straight in Married', 100.0* df[(df.orientation == 'straight') & (df.status == 1)].shape[0]/float(df[(df.status==1)].shape[0]))

print('Percent of straight in Single', 100.0* df[(df.orientation == 'straight') & (df.status == 0)].shape[0]/float(df[(df.status==0)].shape[0]))

df[(df.status == 1)].orientation.value_counts()



print('Percentage of people who do Drugs Somtimes', 100.0* df[(df.drugs == 'sometimes') & (df.status == 1)].shape[0]/float(df[(df.status==1)].shape[0]))

# print(df[(df.status == 1)].drugs.value_counts())



# df[(df.status == 1)].job.value_counts()


target_count = df[(df.status==1)].orientation.value_counts()



pie_trace = go.Pie(labels=target_count.index, values=target_count.values)

layout = dict(title= "Married Orientation distribution", height=400, width=800)

fig = dict(data=[pie_trace], layout=layout)

iplot(fig)

target_count = df[(df.status==0)].orientation.value_counts()



pie_trace = go.Pie(labels=target_count.index, values=target_count.values)

layout = dict(title= "Single Orientation distribution", height=400, width=800)

fig = dict(data=[pie_trace], layout=layout)

iplot(fig)
target_count = df[(df.status==0)].drugs.value_counts()



pie_trace = go.Pie(labels=target_count.index, values=target_count.values)

layout = dict(title= "Single Drug Usage distribution", height=400, width=800)

fig = dict(data=[pie_trace], layout=layout)

iplot(fig)
target_count = df[(df.status==1)].drugs.value_counts()



pie_trace = go.Pie(labels=target_count.index, values=target_count.values)

layout = dict(title= "Married Drug Usage distribution", height=400, width=800)

fig = dict(data=[pie_trace], layout=layout)

iplot(fig)
target_count = df[(df.status==1)].body_type.value_counts()



pie_trace = go.Pie(labels=target_count.index, values=target_count.values)

layout = dict(title= "Married Body_type distribution", height=400, width=800)

fig = dict(data=[pie_trace], layout=layout)

iplot(fig)
target_count = df[(df.status==0)].body_type.value_counts()



pie_trace = go.Pie(labels=target_count.index, values=target_count.values)

layout = dict(title= "Single Body_type distribution", height=400, width=800)

fig = dict(data=[pie_trace], layout=layout)

iplot(fig)


for columns in df.columns:

    # skipping essay, last online time, language and status

    if columns.startswith('ethnicity') or columns.startswith('location') or columns.startswith('essay') or columns.startswith('last_online') or columns.startswith('speaks') or columns.startswith('status'):

        continue

    else:

        list_df.append(pd.get_dummies(df[columns], prefix=columns))

        print(columns, pd.get_dummies(df[columns]).shape[1])



df_np = np.asarray(list_df)



features = pd.concat(df_np, axis=1)



labels = df['status']

feature_list = list(features.columns)





target_count = labels.value_counts()



pie_trace = go.Pie(labels=target_count.index, values=target_count.values)

layout = dict(title= "Single to Married distribution", height=400, width=800)

fig = dict(data=[pie_trace], layout=layout)

iplot(fig)



train_features, test_features, train_labels, test_labels = train_test_split(features.as_matrix(), labels, test_size=0.10, random_state=42, stratify=labels)



print('Training Features Shape:', train_features.shape)

print('Training Labels Shape:', train_labels.shape)

print('Testing Features Shape:', test_features.shape)

print('Testing Labels Shape:', test_labels.shape)



rf = RandomForestClassifier(n_estimators=10000, random_state=42, verbose=1, max_depth=10)



rf.fit(train_features, train_labels)


y_pred = rf.predict(test_features)

from sklearn.metrics import classification_report

target_names = ['single','married']

print(classification_report(test_labels, y_pred, target_names=target_names))


importances = rf.feature_importances_

std = np.std([tree.feature_importances_ for tree in rf.estimators_], axis=0)



indices = np.argsort(importances)[::-1]

print("Feature ranking:")

sort_features = []

sorted_importances = []

for f in range(train_features.shape[1]):

    print(f + 1, feature_list[indices[f]], importances[indices[f]])

    sort_features.append(feature_list[indices[f]])

    sorted_importances.append(importances[indices[f]])



# plt.figure()

# plt.title("Feature importances")

# plt.bar( range(train_features.shape[1])[:10],importances[indices][:10],

#         color="g", xerr=std[indices][:10], align="center")

# plt.yticks(range(train_features.shape[1])[:10], sort_features)

# # plt.xlim([-1, train_features.shape[1]])

# plt.show()











fig, ax = plt.subplots()

y_pos = np.arange(20)

ax.barh(y_pos, sorted_importances[:20], align='center')

ax.set_yticks(y_pos)

ax.set_yticklabels(sort_features)

ax.invert_yaxis()  # labels read top-to-bottom

ax.set_xlabel('Performance')

ax.set_title('How fast do you want to go today?')









plt.title('Feature importance in RandomForest Classifier')

plt.xlabel('Relative importance')

plt.ylabel('feature') 

plt.show()



import sys

!{sys.executable} -m pip install pydot-ng 