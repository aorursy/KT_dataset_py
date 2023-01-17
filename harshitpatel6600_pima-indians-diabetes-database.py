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
import matplotlib.pyplot as plt

import seaborn as sns

import plotly.graph_objects as go
df = pd.read_csv('/kaggle/input/pima-indians-diabetes-database/diabetes.csv')
df.head()
df.shape
columns = df.columns.tolist()

columns
df.isna().sum()
sns.pairplot(df)
corr_mat = df.corr()

corr_mat = corr_mat.to_numpy()

corr_mat = np.round(corr_mat,2)
import plotly.figure_factory as ff



z = corr_mat



fig = ff.create_annotated_heatmap(z, colorscale='Viridis', x=columns, y=columns, showscale=True)

fig['layout']['xaxis']['side'] = 'bottom'

fig.show()
positive = len(df.loc[df['Outcome'] == 1])

negative = len(df.loc[df['Outcome'] == 0])
print('Number of Positive result: ' + str(positive))

print('Number of Negative result: ' + str(negative))
colors = ['lightslategray',] * 2

colors[1] = 'crimson'

fig = go.Figure(go.Bar(x=['Positive', 'Negative'], 

                       y=[positive, negative],

                       text=[positive, negative],

                       textposition='auto', 

                       marker_color=colors))



fig.update_layout(title_text='Number of Positive and Negative cases')



fig.show()
from sklearn.model_selection import train_test_split
# Spilliting dataset into dependent and independent dataset



X = df.drop(['Outcome'], axis=1)

y = df['Outcome']
from sklearn.preprocessing import StandardScaler



scaler = StandardScaler()

sc_X = pd.DataFrame(scaler.fit_transform(X.copy()))
X_train, X_test, y_train, y_test = train_test_split(sc_X, y, test_size=0.33, random_state=1)
from sklearn.neighbors import KNeighborsClassifier





test_scores = []

train_scores = []



for i in range(1,15):



    knn = KNeighborsClassifier(i)

    knn.fit(X_train,y_train)

    

    train_scores.append(knn.score(X_train,y_train))

    test_scores.append(knn.score(X_test,y_test))
x = []

for i in range(1,15):

    x.append(i)
# Create traces

fig = go.Figure()



fig.add_trace(go.Scatter(x=x, y=train_scores,

                    mode='lines+markers',

                    name='train'))



fig.add_trace(go.Scatter(x=x, y=test_scores,

                    mode='lines+markers',

                    name='test'))



fig.update_layout(title='k vs score (train and test)',

                   xaxis_title='k',

                   yaxis_title='score')



fig.show()
print(f'Max train score: {max(train_scores)*100}% with k values: {train_scores.index(max(train_scores))+1}')



print(f'Max test score: {round(max(test_scores)*100,2)}% with k values: {test_scores.index(max(test_scores))+1}')
k = test_scores.index(max(test_scores))+1

knn = KNeighborsClassifier(n_neighbors=k)

knn.fit(X_train, y_train)



predictions = knn.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
print(classification_report(y_test, predictions))
accuracy = round(accuracy_score(y_test, predictions),4)

print(f"\033[94m Accuracy: {accuracy*100}%\033[00m")
cm = confusion_matrix(y_test, predictions)

print(f"\033[92m{cm}\033[00m")
import plotly.figure_factory as ff



l = []

l.append(cm[1])

l.append(cm[0])

z = l



fig = ff.create_annotated_heatmap(z, colorscale='darkmint')



# add title

fig.update_layout(title_text='<i><b>Confusion matrix</b></i>')



# add custom xaxis title

fig.add_annotation(dict(font=dict(color="black",size=14),

                        x=0.5,

                        y=-0.15,

                        showarrow=False,

                        text="Predicted value",

                        xref="paper",

                        yref="paper"))



# add custom yaxis title

fig.add_annotation(dict(font=dict(color="black",size=14),

                        x=-0.05,

                        y=0.5,

                        showarrow=False,

                        text="Actual value",

                        textangle=-90,

                        xref="paper",

                        yref="paper"))



# adjust margins to make room for yaxis title

fig.update_layout(margin=dict(t=50, l=80))



# add colorbar

fig.show()
from sklearn.model_selection import RandomizedSearchCV
param_grid = {'n_neighbors':np.arange(1,50)}

knn = KNeighborsClassifier()

knn_cv= RandomizedSearchCV(knn,param_grid,cv=5)

knn_cv.fit(X,y)



print("Best Score:" + str(knn_cv.best_score_))

print("Best Parameters: " + str(knn_cv.best_params_))
res = pd.DataFrame({'Actual': y_test, 'Predictions': predictions})

res