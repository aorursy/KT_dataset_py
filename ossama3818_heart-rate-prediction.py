# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import seaborn as sns
import plotly.graph_objs as py
import plotly.express as px
import matplotlib.pyplot as plt
%matplotlib inline

from plotly.offline import iplot
from sklearn.metrics import confusion_matrix, accuracy_score

import warnings
warnings.filterwarnings('ignore')
heart_df = pd.read_csv('/kaggle/input/heart-failure-clinical-data/heart_failure_clinical_records_dataset.csv')
heart_df.head()
#checking the null values
heart_df.isnull().sum()
#checking the shape of the dataset

print('Dataset has', heart_df.shape[0], 'rows and', heart_df.shape[1], 'columns.')
heart_df.info()
heart_df.describe()
#splitting the dataset on the based of death event feature

died = heart_df.loc[heart_df['DEATH_EVENT'] == 1, :]
alive = heart_df.loc[heart_df['DEATH_EVENT'] == 0, :]


print(died.shape)
print(alive.shape)
#boxplot

plot1 = py.Box(y = died['age'], 
             name = 'Dead',
             marker = dict(color = '#48484B'))

plot2 = py.Box(y = alive['age'], 
             name = 'Alive',
             marker = dict(color = '#4B9DA5'))


layout = py.Layout(title = 'Heart Failure Rate by Age Group',
                  xaxis = dict(title = 'Heart Failure'),
                  yaxis = dict(title = 'Age'))

fig = py.Figure(data = [plot1, plot2], layout = layout)
iplot(fig)
#splitting the dataset on the basis of anaemia feature

anaemic = heart_df.loc[heart_df['anaemia'] == 1, :]
non_anaemic = heart_df.loc[heart_df['anaemia'] == 0, :]


#calculating the percentage of patients

deaths_by_anaemia = anaemic['DEATH_EVENT'].value_counts(normalize = True).reset_index()
alive_by_anaemia = non_anaemic['DEATH_EVENT'].value_counts(normalize = True).reset_index()
#boxplot
plot1 = py.Bar(x = deaths_by_anaemia.index,
                y = deaths_by_anaemia.DEATH_EVENT,
                name = "Anaemic",
                marker = dict(color = '#48484B',
                             line=dict(color='rgb(0,0,0)',width=1.5)))


plot2 = py.Bar(x = alive_by_anaemia.index,
                y = alive_by_anaemia.DEATH_EVENT,
                name = "Non Anaemic",
                marker = dict(color = '#4B9DA5',
                             line=dict(color='rgb(0,0,0)',width=1.5)))


# Adding titles and labels

data = [plot1, plot2]
layout = py.Layout(title = 'Failure Rate by Anaemia',
                   xaxis = dict(title = 'Heart Failed'),
                   yaxis = dict(title = '% Patients'),
                   barmode = "group")
fig = py.Figure(data = data, layout = layout)
iplot(fig)

h1 = heart_df[(heart_df["DEATH_EVENT"] == 0) & (heart_df["diabetes"] == 0)]
h2 = heart_df[(heart_df["DEATH_EVENT"] == 0) & (heart_df["diabetes"] == 1)]
h3 = heart_df[(heart_df["DEATH_EVENT"] == 1) & (heart_df["diabetes"] == 0)]
h4 = heart_df[(heart_df["DEATH_EVENT"] == 1) & (heart_df["diabetes"] == 1)]

#labelling the pie graph
labels = ['No Diabetes - Survived','Diabetes - Survived', "No Diabetes -  Died", "Diabetes  - Died"]
values = [len(h1),len(h2),len(h3),len(h4)]

#giving colors to different sections of the graph
colors = ['#4B9DA5','#82DDE6','#808788','#48484B']

#plotting the pie graph
fig = py.Figure(data=[py.Pie(labels=labels, values=values)])
fig.update_layout(title_text="Relation OF DEATH_EVENT & DIABETES")
fig.update_traces(marker=dict(colors=colors, line=dict(color='#000000', width=1)))
fig.show()
plot = px.histogram(heart_df, x="creatinine_phosphokinase", color="DEATH_EVENT", marginal="violin", hover_data=heart_df.columns,
                   title ="Relation of CPK Enzyme & DEATH_EVENT", 
                   labels={"creatinine_phosphokinase": "CREATININE PHOSPHOKINASE"},
                   template="ygridoff",
                   color_discrete_map={"0": "Red", "1": "Green"})
plot.show()
#splitting the dataset

bp = heart_df.loc[heart_df['high_blood_pressure'] == 1, :]
normal = heart_df.loc[heart_df['high_blood_pressure'] == 0, :]

#calculating percentage

died_bp = bp['DEATH_EVENT'].value_counts(normalize = True).reset_index()
died_normal = normal['DEATH_EVENT'].value_counts(normalize = True).reset_index()

#structuring the data

plot1 = py.Bar(x = died_bp.index,
                y = died_bp.DEATH_EVENT,
                name = "High Blood Pressure",
                marker = dict(color = '#48484B',
                             line=dict(color='rgb(0,0,0)',width=1.5)))


plot2 = py.Bar(x = died_normal.index,
                y = died_normal.DEATH_EVENT,
                name = "Normal",
                marker = dict(color = '#4B9DA5',
                             line=dict(color='rgb(0,0,0)',width=1.5)))


#adjusting the layout
data = [plot1, plot2]
layout = py.Layout(title = 'Heart Failure Rate due to High Blood Pressure',
                   xaxis = dict(title = 'Heart Failed'),
                   yaxis = dict(title = '% Patients'),
                   barmode = "group")
fig = py.Figure(data = data, layout = layout)
iplot(fig)
plot = px.histogram(heart_df, x="serum_creatinine", color="DEATH_EVENT", marginal="violin", hover_data=heart_df.columns,
                   title ="Impact of SERUM CREATININE on Heart Failure", 
                   labels={"serum_creatinine": "SERUM CREATININE"},
                   template="gridon",
                   color_discrete_map={"0": "RebeccaPurple", "1": "MediumPurple"})
plot.show()
#structuring the data
h1 = heart_df[(heart_df["DEATH_EVENT"]==0) & (heart_df["sex"]==1)]
h2 = heart_df[(heart_df["DEATH_EVENT"]==1) & (heart_df["sex"]==1)]
h3 = heart_df[(heart_df["DEATH_EVENT"]==0) & (heart_df["sex"]==0)]
h4 = heart_df[(heart_df["DEATH_EVENT"]==1) & (heart_df["sex"]==0)]

#labeling the graph
labels = ['Male - Survived','Male - Died', "Female -  Survived", "Female - Died"]
values = [len(h1),len(h2),len(h3),len(h4)]

#giving colors to different sections of the graph
colors = ['#4B9DA5','#82DDE6','#808788','#48484B']

pie = py.Figure(data=[py.Pie(labels=labels, values=values)])
pie.update_layout(title_text=" Impact of Gender on Heart Failure")
pie.update_traces(marker=dict(colors=colors, line=dict(color='#000000', width=1)))
pie.show()
#Structuring the dataset

smoker = heart_df.loc[heart_df['smoking'] == 1, :]
not_smoker = heart_df.loc[heart_df['smoking'] == 0, :]

#calculating the percentage
failed_smoker = smoker['DEATH_EVENT'].value_counts(normalize = True).reset_index()
failed_smoker = not_smoker['DEATH_EVENT'].value_counts(normalize = True).reset_index()

#plotting the Bar plot

plot1 = py.Bar(x = failed_smoker.index,
                y = failed_smoker.DEATH_EVENT,
                name = "Smoker",
                marker = dict(color = '#48484B',
                             line=dict(color='rgb(0,0,0)',width=1.5)))


plot2 = py.Bar(x = failed_smoker.index,
                y = failed_smoker.DEATH_EVENT,
                name = "Non Smoker",
                marker = dict(color = '#4B9DA5',
                             line=dict(color='rgb(0,0,0)',width=1.5)))

# Add appropriate titles and labels
data = [plot1, plot2]
layout = py.Layout(title = 'Impact of Smoking on Heart Failure',
                   xaxis = dict(title = 'Heart Failed'),
                   yaxis = dict(title = '% Patients'),
                   barmode = "group")
plot = py.Figure(data = data, layout = layout)
iplot(plot)
plot, ax = plt.subplots(figsize = (14, 10))

sns.heatmap(heart_df.corr(), annot = True)
plt.show()
#splitting up the dataset

scale = heart_df.drop(columns = ['anaemia', 'diabetes', 'high_blood_pressure', 'sex', 'smoking', 'DEATH_EVENT'])
no_scale = heart_df[['anaemia', 'diabetes', 'high_blood_pressure', 'sex', 'smoking', 'DEATH_EVENT']]
#scaling the features
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(scale)
scaled = pd.DataFrame(scaler.transform(scale), columns = scale.columns)
scaled_df = pd.concat([scaled, no_scale], axis = 1)
scaled_df.head()
#splitting the dataset using train test split

from sklearn.model_selection import train_test_split

x = scaled_df.drop(columns = ['DEATH_EVENT'])
y = scaled_df['DEATH_EVENT']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, stratify = y, random_state = 36)
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()
lr.fit(x_train, y_train)
#predicting the values
y_pred_lr = lr.predict(x_test)
print(y_pred_lr)
accuracies = []
#building a confusion matrix

matrix_lr = confusion_matrix(y_test, y_pred_lr)
accuracy_lr = accuracy_score(y_test, y_pred_lr)

accuracies.append(accuracy_lr)

print('The confusion matrix is :- ', matrix_lr)
print('The accuracy of the logistic regression is :-', accuracy_lr,'%.')
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 6)
knn_model = knn.fit(x_train, y_train)
y_pred_knn = knn.predict(x_test)
print(y_pred_knn)
#building a confusion matrix

matrix_knn = confusion_matrix(y_test, y_pred_knn)
accuracy_knn = accuracy_score(y_test, y_pred_knn)

accuracies.append(accuracy_knn)

print('The confusion matrix is :- ', matrix_knn)
print('The accuracy of the KNearest Neighbour algorithm is:-', accuracy_knn,'%.')
from sklearn.svm import SVC

svc = SVC(random_state=0, kernel = 'rbf')
svc.fit(x_train, y_train)
#predicting the values

y_pred_svc = svc.predict(x_test)
print(y_pred_svc)
#making a confusion matrix and predicting accuracy

matrix_svc = confusion_matrix(y_test, y_pred_svc)
accuracy_svc = accuracy_score(y_test, y_pred_svc)

accuracies.append(accuracy_svc)

print('The confusion matrix is :- ', matrix_svc)
print('The accuracy of the Support Vector Machine algorithm is:-', accuracy_svc,'%.')
from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier(max_leaf_nodes = 3, random_state=0, criterion='entropy')
tree.fit(x_train, y_train)
y_pred_tree = tree.predict(x_test)
print(y_pred_tree)
#making a confusion matrix and predicting accuracy

matrix_tree = confusion_matrix(y_test, y_pred_tree)
accuracy_tree = accuracy_score(y_test, y_pred_tree)

accuracies.append(accuracy_tree)

print('The confusion matrix is :- ', matrix_tree)
print('The accuracy of the Support Vector Machine algorithm is:-', accuracy_tree,'%.')
accuracies
list = ["Logistic Regression", "K-NearestNeighbours","Support Vector Machines","Decision Tree Classifier"]
plt.rcParams['figure.figsize']=15,6 
sns.set_style("darkgrid")
ax = sns.barplot(x=list, y=accuracies, palette = "rocket", saturation =1.5)
plt.xlabel("Classification Models", fontsize = 20 )
plt.ylabel("%age of Accuracy", fontsize = 20)
plt.title("Accuracy of different Classification Models", fontsize = 20)
plt.xticks(fontsize = 12, horizontalalignment = 'center', rotation = 8)
plt.yticks(fontsize = 13)
for p in ax.patches:
    width, height = p.get_width(), p.get_height()
    x, y = p.get_xy() 
    ax.annotate(f'{height:.2%}', (x + width/2, y + height*1.02), ha='center', fontsize = 'x-large')
plt.show()
