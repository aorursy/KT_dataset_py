import numpy as np

import pandas as pd

import plotly_express as px

import plotly.graph_objects as go

import plotly.io as pio

from plotly.offline import plot, iplot,init_notebook_mode

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import confusion_matrix

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import StandardScaler



init_notebook_mode()

pio.templates.default = 'plotly_white'
df = pd.read_csv('/kaggle/input/heart-disease-uci/heart.csv')

df.head()
df.info()



#converting the columns into categorical variables, this would make it easier for us to visualize them later. 

df['target'] = df['target'].astype('category')

df['slope'] = df['slope'].astype('category')

df['fbs'] = df['fbs'].astype('category')

df['ca'] = df['ca'].astype('category')

df['thal'] = df['thal'].astype('category')

df['exang'] = df['exang'].astype('category')
df.describe()
target = df.target.value_counts(normalize = True)*100

trace1 = go.Bar(

    x = ['Has Disease','Does not have Disease'],

    y = target.values, 

    text = target.values, 

    textposition = 'auto',

    texttemplate = "%{y:.2f} %"

)

fig = go.Figure(data = [trace1])

fig.update_layout(title_text = '<b>Target Distribution</b>',

                 xaxis_title="Target",

                yaxis_title="Percentage")

fig.show()
traces = []

for sex,data in df.groupby('sex'):

    if sex == 1:

        name = 'Male'

    else:

        name = 'Female'

    target = data['target'].value_counts(normalize = True)*100

    trace = go.Bar(

        x = target.index,

        y = target.values,

        text = target.values,

        textposition = 'auto',

        name = name,

        texttemplate = "%{y:.2f} %"

    )

    traces.append(trace)

fig = go.Figure(data = traces)

fig.update_layout(title = '<b>Distribution of target based on sex</b>',

    xaxis_title="Target",

    yaxis_title="Percentage",

    legend_title="Sex"

)

iplot(fig)
traces = []

for target,data in df.groupby('target'):

    if target == 1:

        name = 'Has Disease'

    else:

        name = 'Does not have Disease'

    age = data['age'].value_counts()

    trace = go.Bar(

        x = age.index,

        y = age.values,

        name = name

    )

    traces.append(trace)

fig = go.Figure(data = traces)

fig.update_layout(title = 'Distribution of target based on age',

    xaxis_title="Age",

    yaxis_title="Counts",

    legend_title="Target",

    legend = dict(x = 0)

)

iplot(fig)
print(f"The average age of People without Heart Disease is {df[df['target'] == 0]['age'].mean()}")

print(f"The average age of People with Heart Disease is {df[df['target'] == 1]['age'].mean()}")
fig = px.scatter(df,x = 'age', y = 'thalach',trendline = 'ols', marginal_y = 'violin',color = 'target')

fig.update_traces(marker = dict(size = 8, ))

fig.update_layout(title = '<b>Distribution of heartrate based on age in people with and without heart disease</b>',

    xaxis_title="Age",

    yaxis_title="Heart Rate",

    legend_title="Target",

    legend = dict(x = 0)

)

iplot(fig)
fig = px.scatter(df,x = 'age', y = 'trestbps', color = 'target', trendline = 'ols', marginal_y = 'violin')

fig.update_traces(marker = dict(size = 10, ))

fig.update_layout(title = '<b>Distribution of Resting Blood Pressure based on age in people with and without heart attack</b>',

    xaxis_title="Age",

    yaxis_title="Blood Pressure",

    legend_title="Target",

    legend = dict(x = 0)   

)

iplot(fig)
traces = []

for slope,data in df.groupby('slope'):

    target = data['target'].value_counts(normalize = True)*100

    trace = go.Bar(

        x = target.index,

        y = target.values,

        text = target.values,

        textposition = 'auto',

        name = slope,

        texttemplate = "%{y:.2f} %"

    )

    traces.append(trace)

fig = go.Figure(data = traces)

fig.update_layout(title = 'Distribution of target based on Slope of The Peak Exercise ST Segment ',

    xaxis_title="Target",

    yaxis_title="Counts",

    legend_title="Slope"

)

iplot(fig)
traces = []

for fbs,data in df.groupby('fbs'):

    if fbs == 1:

        name = 'Fasting Blood Sugar > 120 mg/dl'

    else:

        name = 'Fasting Blood Sugar <= 120 mg/dl'

    target = data['target'].value_counts(normalize = True)*100

    trace = go.Bar(

        x = target.index,

        y = target.values,

        text = target.values,

        textposition = 'auto',

        name = name,

        texttemplate = "%{y:.2f} %"

    )

    traces.append(trace)

fig = go.Figure(data = traces)

fig.update_layout(title = 'Distribution of target based on Fasing Blood Sugar',

    xaxis_title="Target",

    yaxis_title="Counts",

    legend_title="Fasting Blood Sugar"

)

iplot(fig)
traces = []

for ca,data in df.groupby('ca'):

    target = data['target'].value_counts(normalize = True)

    trace = go.Bar(

        x = target.index,

        y = target.values,

        text = target.values,

        textposition = 'auto',

        name = ca,

        texttemplate = "%{y:.2f} %"

    )

    traces.append(trace)

fig = go.Figure(data = traces)

fig.update_layout(title = 'Distribution of target based on Number of Major Vessels',

    xaxis_title="Target",

    yaxis_title="Counts",

    legend_title="Number of Major Vessels"

)

iplot(fig)
traces = []

for exang,data in df.groupby('exang'):

    if exang == 1:

        name = 'Yes'

    else:

        name = 'No'

    target = data['target'].value_counts(normalize = True)*100

    trace = go.Bar(

        x = target.index,

        y = target.values,

        text = target.values,

        textposition = 'auto',

        name = name,

        texttemplate = "%{y:.2f} %"

    )

    traces.append(trace)

fig = go.Figure(data = traces)

fig.update_layout(title = 'Distribution of target based on Exercise Induced Angina',

    xaxis_title="Target",

    yaxis_title="Counts",

    legend_title="Exercise Induced Angina"

)

iplot(fig)
traces = []

for thal,data in df.groupby('thal'):

    target = data['target'].value_counts(normalize = True)*100

    trace = go.Bar(

        x = target.index,

        y = target.values,

        text = target.values,

        textposition = 'auto',

        name = thal,

        texttemplate = "%{y:.2f} %"

    )

    traces.append(trace)

fig = go.Figure(data = traces)

fig.update_layout(title = 'Distribution of target based on Thal',

    xaxis_title="Target",

    yaxis_title="Counts",

    legend_title="Thal"

)

iplot(fig)
df = pd.read_csv('/kaggle/input/heart-disease-uci/heart.csv')
df['age'] = pd.cut(df['age'],bins=[0,47,61,100],labels=['Adult','Aging','Old'])
categorical_features = ['age','cp', 'fbs', 'exang', 'slope', 'ca', 'thal']

for feature in categorical_features:

    encoder = LabelEncoder()

    df[feature] = encoder.fit_transform(df[feature])
continuous_features = ['trestbps', 'chol','restecg','thalach','oldpeak']

scaler = StandardScaler()

df[continuous_features] = scaler.fit_transform(df[continuous_features])
y = df.target.values

X = df.drop(['target'], axis = 1)
from sklearn.model_selection import StratifiedShuffleSplit

stratifiedSplit = StratifiedShuffleSplit(n_splits=1, test_size = 0.1, random_state = 0)

for train_idx, test_idx in stratifiedSplit.split(X, y):

    x_train, x_test = X.iloc[train_idx,], X.iloc[test_idx,]

    y_train, y_test = y[train_idx], y[test_idx]
log_reg = LogisticRegression(random_state=0,class_weight='balanced')

log_reg.fit(x_train, y_train)

from sklearn.metrics import accuracy_score

train_acc = accuracy_score(y_train, log_reg.predict(x_train))

test_acc = accuracy_score(y_test, log_reg.predict(x_test))

print('-'*25)

print('Training Accuracy is {:.2f}'.format(train_acc*100))

print('-'*25)

print('-'*25)

print('Testing Accuracy is {:.2f}'.format(test_acc*100))

print('-'*25)
conf = confusion_matrix(y_test, log_reg.predict(x_test))

fig = px.imshow(conf)

fig.update_layout(

title = 'Logistic Regression Confusion Matrix',

xaxis_title = 'Predicted Label',

yaxis_title = 'True Label'

)

iplot(fig)
max_acc = 0.0

neighbours = 0

for i in range(1,10):

    knn = KNeighborsClassifier(n_neighbors=i,p=1)

    knn.fit(x_train, y_train)

    test_acc = accuracy_score(y_test, knn.predict(x_test))

    if(test_acc>max_acc):

        max_acc = test_acc

        neighbours = i

knn = KNeighborsClassifier(n_neighbors=neighbours)

knn.fit(x_train, y_train)

train_acc = accuracy_score(y_train, knn.predict(x_train))

print('-'*25)

print('Training Accuracy is {:.2f} with {} neighbours'.format(train_acc*100, neighbours))

print('-'*25)

print('-'*25)

print('Maximum Testing Accuracy is {:.2f} with {} neighbours'.format(max_acc*100, neighbours))

print('-'*25)
conf = confusion_matrix(y_test, knn.predict(x_test))

fig = px.imshow(conf)

fig.update_layout(

title = 'KNN Classifier Confusion Matrix',

xaxis_title = 'Predicted Label',

yaxis_title = 'True Label'

)

iplot(fig)
rf = RandomForestClassifier(n_estimators=5,min_samples_split=15,random_state = 0, class_weight='balanced_subsample')

rf.fit(x_train, y_train)

test_acc = accuracy_score(y_test, rf.predict(x_test))

train_acc = accuracy_score(y_train, rf.predict(x_train))

print('-'*25)

print('Training Accuracy is {:.2f}'.format(train_acc*100))

print('-'*25)

print('-'*25)

print('Testing Accuracy is {:.2f}'.format(test_acc*100))

print('-'*25)
conf = confusion_matrix(y_test, rf.predict(x_test))

fig = px.imshow(conf)

fig.update_layout(

title = 'Random Forest Classifier Confusion Matrix',

xaxis_title = 'Predicted Label',

yaxis_title = 'True Label'

)

iplot(fig)