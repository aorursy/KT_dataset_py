import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

import plotly.offline as py

from plotly.offline import init_notebook_mode, iplot

import plotly.graph_objs as go



import warnings

warnings.filterwarnings('ignore')



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# Reading the dataset into a dataframe and storing the original copy for later reference



df = pd.read_csv('../input/heart-failure-clinical-data/heart_failure_clinical_records_dataset.csv')

original = df.copy()
# Printing the shape of the dataset



print('Dataset has', df.shape[0], 'rows and', df.shape[1], 'columns')
# Printing the info of the dataset



df.info()
# Printing the head of the dataset



df.head()
# Printing the descriptive stats of the dataset



df.describe()
init_notebook_mode()
#Splitting up the dataset based on the death event feature



failed = df.loc[df['DEATH_EVENT'] == 1, :]

not_failed = df.loc[df['DEATH_EVENT'] == 0, :]
print(failed.shape)

print(not_failed.shape)
# Create two traces each with the target class label and plot boxplots by considering age



trace1 = go.Box(y = failed['age'], 

             name = 'Failed',

             marker = dict(color = 'black'))



trace2 = go.Box(y = not_failed['age'], 

             name = 'Not Failed',

             marker = dict(color = '#eb2862'))





layout = go.Layout(title = 'Failure Rate by Age Group',

                  xaxis = dict(title = 'Heart Failure'),

                  yaxis = dict(title = 'Age'))



fig = go.Figure(data = [trace1, trace2], layout = layout)

iplot(fig)
#Splitting up the dataset based on the 'anaemia' feature



anaemic = df.loc[df['anaemia'] == 1, :]

not_anaemic = df.loc[df['anaemia'] == 0, :]
# Calculate the percentage of patients in each of the class (Anaemic or Not)



failed_anaemia = anaemic['DEATH_EVENT'].value_counts(normalize = True).reset_index()

failed_not_anaemia = not_anaemic['DEATH_EVENT'].value_counts(normalize = True).reset_index()
# Create two traces of bar plots



trace1 = go.Bar(x = failed_anaemia.index,

                y = failed_anaemia.DEATH_EVENT,

                name = "Anaemic",

                marker = dict(color = '#eb2862',

                             line=dict(color='rgb(0,0,0)',width=1.5)))





trace2 = go.Bar(x = failed_not_anaemia.index,

                y = failed_not_anaemia.DEATH_EVENT,

                name = "Not Anaemic",

                marker = dict(color = '#615a5c',

                             line=dict(color='rgb(0,0,0)',width=1.5)))
# Add appropriate titles and labels



data = [trace1, trace2]

layout = go.Layout(title = 'Failure Rate by Anaemia',

                   xaxis = dict(title = 'Heart Failed'),

                   yaxis = dict(title = '% Patients'),

                   barmode = "group")

fig = go.Figure(data = data, layout = layout)

iplot(fig)
# Create two traces each with the target class label and plot boxplots by considering creatinine phosphokinase



trace1 = go.Box(y = failed['creatinine_phosphokinase'], 

             name = 'Failed',

             marker = dict(color = 'black'))



trace2 = go.Box(y = not_failed['creatinine_phosphokinase'], 

             name = 'Not Failed',

             marker = dict(color = '#eb2862'))





layout = go.Layout(title = 'Failure Rate by CPK Level',

                  xaxis = dict(title = 'Heart Failure'),

                  yaxis = dict(title = 'CPK'))



fig = go.Figure(data = [trace1, trace2], layout = layout)

iplot(fig)
#Splitting up the dataset based on the diabetes feature



diabetic = df.loc[df['diabetes'] == 1, :]

not_diabetic = df.loc[df['diabetes'] == 0, :]
# Calculate the percentage of patients in each of the class (Diabetic or Not)



failed_diabetic = diabetic['DEATH_EVENT'].value_counts(normalize = True).reset_index()

failed_not_diabetic = not_diabetic['DEATH_EVENT'].value_counts(normalize = True).reset_index()
# Create two traces of bar plots



trace1 = go.Bar(x = failed_diabetic.index,

                y = failed_diabetic.DEATH_EVENT,

                name = "Diabetic",

                marker = dict(color = '#eb2862',

                             line=dict(color='rgb(0,0,0)',width=1.5)))





trace2 = go.Bar(x = failed_not_diabetic.index,

                y = failed_not_diabetic.DEATH_EVENT,

                name = "Not Diabetic",

                marker = dict(color = '#615a5c',

                             line=dict(color='rgb(0,0,0)',width=1.5)))
# Add appropriate titles and labels



data = [trace1, trace2]

layout = go.Layout(title = 'Failure Rate by Diabetes',

                   xaxis = dict(title = 'Heart Failed'),

                   yaxis = dict(title = '% Patients'),

                   barmode = "group")

fig = go.Figure(data = data, layout = layout)

iplot(fig)
# Create two traces each with the target class label and plot boxplots by considering ejection fraction



trace1 = go.Box(y = failed['ejection_fraction'], 

             name = 'Failed',

             marker = dict(color = 'black'))



trace2 = go.Box(y = not_failed['ejection_fraction'], 

             name = 'Not Failed',

             marker = dict(color = '#eb2862'))





layout = go.Layout(title = 'Failure Rate by Ejection Fraction',

                  xaxis = dict(title = 'Heart Failure'),

                  yaxis = dict(title = 'Ejection Fraction'))



fig = go.Figure(data = [trace1, trace2], layout = layout)

iplot(fig)
#Splitting up the dataset based on the 'high_blood_pressure' feature



bp = df.loc[df['high_blood_pressure'] == 1, :]

normal = df.loc[df['high_blood_pressure'] == 0, :]
# Calculate the percentage of patients in each of the class (BP or Normal)



failed_bp = bp['DEATH_EVENT'].value_counts(normalize = True).reset_index()

failed_normal = normal['DEATH_EVENT'].value_counts(normalize = True).reset_index()
# Create two traces of bar plots



trace1 = go.Bar(x = failed_bp.index,

                y = failed_bp.DEATH_EVENT,

                name = "High Blood Pressure",

                marker = dict(color = '#eb2862',

                             line=dict(color='rgb(0,0,0)',width=1.5)))





trace2 = go.Bar(x = failed_normal.index,

                y = failed_normal.DEATH_EVENT,

                name = "Normal",

                marker = dict(color = '#615a5c',

                             line=dict(color='rgb(0,0,0)',width=1.5)))
# Add appropriate titles and labels



data = [trace1, trace2]

layout = go.Layout(title = 'Failure Rate by High BP',

                   xaxis = dict(title = 'Heart Failed'),

                   yaxis = dict(title = '% Patients'),

                   barmode = "group")

fig = go.Figure(data = data, layout = layout)

iplot(fig)
# Create two traces each with the target class label and plot boxplots by considering platelets



trace1 = go.Box(y = failed['platelets'], 

             name = 'Failed',

             marker = dict(color = 'black'))



trace2 = go.Box(y = not_failed['platelets'], 

             name = 'Not Failed',

             marker = dict(color = '#eb2862'))





layout = go.Layout(title = 'Failure Rate by Platelets Count',

                  xaxis = dict(title = 'Heart Failure'),

                  yaxis = dict(title = 'Platelets'))



fig = go.Figure(data = [trace1, trace2], layout = layout)

iplot(fig)
# Create two traces each with the target class label and plot boxplots by considering creatinine



trace1 = go.Box(y = failed['serum_creatinine'], 

             name = 'Failed',

             marker = dict(color = 'black'))



trace2 = go.Box(y = not_failed['serum_creatinine'], 

             name = 'Not Failed',

             marker = dict(color = '#eb2862'))





layout = go.Layout(title = 'Failure Rate by Serum Creatinine Level',

                  xaxis = dict(title = 'Heart Failure'),

                  yaxis = dict(title = 'Serum Creatinine'))



fig = go.Figure(data = [trace1, trace2], layout = layout)

iplot(fig)
# Create two traces each with the target class label and plot boxplots by considering sodium



trace1 = go.Box(y = failed['serum_sodium'], 

             name = 'Failed',

             marker = dict(color = 'black'))



trace2 = go.Box(y = not_failed['serum_sodium'], 

             name = 'Not Failed',

             marker = dict(color = '#eb2862'))





layout = go.Layout(title = 'Failure Rate by Serum sodium Level',

                  xaxis = dict(title = 'Heart Failure'),

                  yaxis = dict(title = 'Serum Sodium'))



fig = go.Figure(data = [trace1, trace2], layout = layout)

iplot(fig)
#Splitting up the dataset based on the 'sex' feature



men = df.loc[df['sex'] == 1, :]

women = df.loc[df['sex'] == 0, :]
# Calculate the percentage of patients in each of the gender



failed_men = men['DEATH_EVENT'].value_counts(normalize = True).reset_index()

failed_women = women['DEATH_EVENT'].value_counts(normalize = True).reset_index()
# Create two traces of bar plots



trace1 = go.Bar(x = failed_men.index,

                y = failed_men.DEATH_EVENT,

                name = "Men",

                marker = dict(color = '#eb2862',

                             line=dict(color='rgb(0,0,0)',width=1.5)))





trace2 = go.Bar(x = failed_women.index,

                y = failed_women.DEATH_EVENT,

                name = "Women",

                marker = dict(color = '#615a5c',

                             line=dict(color='rgb(0,0,0)',width=1.5)))
# Add appropriate titles and labels



data = [trace1, trace2]

layout = go.Layout(title = 'Failure Rate by Gender',

                   xaxis = dict(title = 'Heart Failed'),

                   yaxis = dict(title = '% Patients'),

                   barmode = "group")

fig = go.Figure(data = data, layout = layout)

iplot(fig)
#Splitting up the dataset based on the 'smoking' feature



smoking = df.loc[df['smoking'] == 1, :]

not_smoking = df.loc[df['smoking'] == 0, :]
# Calculate the percentage of patients in each of the class



failed_smoking = smoking['DEATH_EVENT'].value_counts(normalize = True).reset_index()

failed_no_smoking = not_smoking['DEATH_EVENT'].value_counts(normalize = True).reset_index()
# Create two traces of bar plots



trace1 = go.Bar(x = failed_smoking.index,

                y = failed_smoking.DEATH_EVENT,

                name = "Smoker",

                marker = dict(color = '#eb2862',

                             line=dict(color='rgb(0,0,0)',width=1.5)))





trace2 = go.Bar(x = failed_no_smoking.index,

                y = failed_no_smoking.DEATH_EVENT,

                name = "Non Smoker",

                marker = dict(color = '#615a5c',

                             line=dict(color='rgb(0,0,0)',width=1.5)))
# Add appropriate titles and labels



data = [trace1, trace2]

layout = go.Layout(title = 'Failure Rate by Smoking',

                   xaxis = dict(title = 'Heart Failed'),

                   yaxis = dict(title = '% Patients'),

                   barmode = "group")

fig = go.Figure(data = data, layout = layout)

iplot(fig)
fig, ax = plt.subplots(figsize = (14, 10))



sns.heatmap(df.corr(), annot = True, cmap = 'summer')

plt.show()
df.head()
scale = df.drop(columns = ['anaemia', 'diabetes', 'high_blood_pressure', 'sex', 'smoking', 'DEATH_EVENT'])

no_scale = df[['anaemia', 'diabetes', 'high_blood_pressure', 'sex', 'smoking', 'DEATH_EVENT']]
# Import StandardScaler for standardizing the features



from sklearn.preprocessing import StandardScaler
# Fitting the scaler on the training data and transform both the sets



sc = StandardScaler()



sc.fit(scale)



scaled = pd.DataFrame(sc.transform(scale), columns = scale.columns)
scaled.head(2)
no_scale.head(2)
scaled_df = pd.concat([scaled, no_scale], axis = 1)



scaled_df.head(3)
# Importing required libraries



from sklearn.model_selection import train_test_split
X = scaled_df.drop(columns = ['DEATH_EVENT'])

y = scaled_df['DEATH_EVENT']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, stratify = y, random_state = 36)
from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC

from sklearn.naive_bayes import GaussianNB
lr = LogisticRegression(solver = 'liblinear', random_state = 42)



lr_model = lr.fit(X_train, y_train)

print('Training Score:', lr_model.score(X_train, y_train))

print('Testing Score:', lr_model.score(X_test, y_test))
dt = DecisionTreeClassifier(random_state = 42)



dt_model = dt.fit(X_train, y_train)

print('Training Score:', dt_model.score(X_train, y_train))

print('Testing Score:', dt_model.score(X_test, y_test))
rf = RandomForestClassifier(random_state = 42)



rf_model = rf.fit(X_train, y_train)

print('Training Score:', rf_model.score(X_train, y_train))

print('Testing Score:', rf_model.score(X_test, y_test))
knn = KNeighborsClassifier()



knn_model = knn.fit(X_train, y_train)

print('Training Score:', knn_model.score(X_train, y_train))

print('Testing Score:', knn_model.score(X_test, y_test))
svm = SVC(kernel = 'linear')



svm_model = svm.fit(X_train, y_train)

print('Training Score:', svm_model.score(X_train, y_train))

print('Testing Score:', svm_model.score(X_test, y_test))
nb = GaussianNB()



nb_model = nb.fit(X_train, y_train)

print('Training Score:', nb_model.score(X_train, y_train))

print('Testing Score:', nb_model.score(X_test, y_test))
# Check the weights given to features in the LR model



imp = lr_model.coef_
imp
data = [0.86033582,  0.07359915, -0.88052799, -0.29623866,  0.35454231,

        -0.26178752, -1.57917285, -0.10954753,  0.16821838, -0.60792902,

        -0.69118234,  0.13141275]

cols = X_train.columns
# Plotting the coefficients of LR Model to observe visually



fig, ax = plt.subplots(figsize = (14, 7))



sns.barplot(x = cols, y = data, palette = 'winter')

plt.title('Coefficients of LR Model')

plt.xlabel('Feature')

plt.ylabel('Coefficient')

plt.xticks(rotation = 90)

plt.show()
features = ['age', 'ejection_fraction', 'time', 'serum_creatinine', 'serum_sodium']
lr = LogisticRegression(solver = 'liblinear', random_state = 42, C = 0.05, max_iter = 300)



lr_model = lr.fit(X_train[features], y_train)

print('Training Score:', lr_model.score(X_train[features], y_train))

print('Testing Score:', lr_model.score(X_test[features], y_test))