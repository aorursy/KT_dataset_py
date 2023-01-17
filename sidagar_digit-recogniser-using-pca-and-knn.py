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
import seaborn as sb

import plotly.express as px

import sklearn.neighbors as KNN

import plotly.graph_objects as go

import plotly.figure_factory as ff

from matplotlib import pyplot as plt

from sklearn.decomposition import PCA

from sklearn.metrics import make_scorer

from sklearn.model_selection import train_test_split
df_train = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')

df_train.describe()
df_train_x = df_train.drop('label',axis =1)

df_train_y = df_train[['label']]
pca = PCA().fit(df_train_x)

pca.explained_variance_ratio_
a = []

s = 0

a.append([0,(1-s)*100,'Percentage varience lost is :'+str((1-s)*100)+'%'])

for i in range(len(pca.explained_variance_ratio_)):

    s+=pca.explained_variance_ratio_[i]

    a.append([i+1,(1-s)*100,'Percentage varience lost is : '+

              str((((1-s)*100)//0.0001)/10000)+'%'])

arr = pd.DataFrame(a)

arr = arr.rename(columns = {0:'No of components used:', 

                            1:'Total varience lost (in percentage)'} )

px.line(data_frame = arr,x = 'No of components used:',

        y = 'Total varience lost (in percentage)',

        range_x = [0,784],range_y = [0,100],hover_name = 2,

        title = 'Graph depicting the loss in varience as we reduce the number of components.')
components = 300

pca = PCA(n_components = components).fit(df_train_x)

numpy_train_x = df_train_x.to_numpy()

pca_trans = pca.transform(numpy_train_x)

pca_invtrans = pca.inverse_transform(pca_trans)

row = 10

column = 7



for i in range(row):

    for j in range(column):

        if j ==0:

            a = numpy_train_x[0+(i*column)].reshape(28,28)

            a = np.pad(a, pad_width=1, mode='constant', constant_values=400)

            b = pca_invtrans[0+(i*column)].reshape(28,28)

            b = np.pad(b, pad_width=1, mode='constant', constant_values=450)

            stack = np.hstack((a,b))

        else:

            a = numpy_train_x[j+(i*column)].reshape(28,28)

            a = np.pad(a, pad_width=1, mode='constant', constant_values=400)

            b = pca_invtrans[j+(i*column)].reshape(28,28)

            b = np.pad(b, pad_width=1, mode='constant', constant_values=450)

            stack = np.hstack((stack,a))

            stack = np.hstack((stack,b))

    if i ==0:

        final = stack

    else:

        final = np.vstack((final,stack))

final = np.pad(final,pad_width=2, mode='constant', constant_values=500)

img = final
a = df_train_y['label'][0:row*column].to_numpy()

label = []

for i in a:

    label.append("The Label for the digit is: "+str(i))

final = []

border = ['Border']*604

final.append(border)

final.append(border)

for i in range(row):

    final.append(border)

    a = ['Border','Border']

    for j in range(column):

        for k in range(2):

            a.append('Border')

            for l in range(28):

                a.append(label[i*column+j])

            a.append('Border')

    a.append('Border')

    a.append('Border')

    for i in range(28):

        final.append(a)

    final.append(border)

final.append(border)

final.append(border)

label = final
fig = go.Figure(data = go.Heatmap(z = img,colorbar = None,

                                  colorscale = [[0,'black'],[0.7,'white'],

                                                [0.8,'red'],[0.9,'blue'],

                                                [1.0,'rgb(255,0,255)']],

                                  zmin = 0,zmax = 500,zauto = False,

                                  hovertext = label))

fig['layout']['yaxis']['autorange'] = "reversed"

fig.update_layout(title = 'The Distortion induced due to PCA while using '+

                  str(components)+' components.',

                  height  = 600,width = 1100,yaxis_tickvals = [0],

                  yaxis_ticktext =[' '],xaxis_tickvals = [0],

                  xaxis_ticktext =[' '],

                  xaxis_title = 'The Original images have a red border while a blue one has been used for their PCA transforms.')

fig.update_traces(showscale = False)

fig.show()
x_train,x_test,y_train,y_test = train_test_split(pca.transform(numpy_train_x),df_train_y,test_size = 0.1)
knn = KNN.KNeighborsClassifier(n_jobs = -1,n_neighbors = 3,algorithm = 'ball_tree')

knn.fit(x_train,y_train.to_numpy().ravel())

pred = knn.predict(x_test)

pred
y_test_np = y_test.to_numpy().ravel()

score=0

for i in range(len(y_test)):

    if pred[i] == y_test_np[i]:

        score = score+1

score /=len(y_test)

print(str(score*100))
predictions = pred

y_test_np = y_test.to_numpy()

classes = [0,1,2,3,4,5,6,7,8,9]





confusion_mat = np.zeros((len(classes),len(classes)))

for i in range(len(predictions)):

    confusion_mat[classes.index(predictions[i])][classes.index(y_test_np[i])]+=1

confusion_mat = confusion_mat.T

confusion_mat_norm = confusion_mat/len(y_test_np)

confusion_mat_norm = (confusion_mat_norm//0.0001)/10000



fig = ff.create_annotated_heatmap(confusion_mat_norm, x=classes, y=classes, 

                                  annotation_text=confusion_mat_norm,

                                  colorscale='Viridis',text = confusion_mat,

                                  hovertemplate='Expected Value: %{y}<br>'+

                                                'Predicted Value: %{x}<br>'+

                                                'No. of datapoints in this category are: %{text}<extra></extra>')

fig.update_layout(title_text='<b>Confusion Matrix for the dataset:</b>',

                  xaxis = {'title':'Predicted Values'},width = 900,

                  yaxis = {'title':'Expected Values','autorange':'reversed'})

fig.update_traces(showscale = True)

fig.show()
knn.fit(pca.transform(numpy_train_x),df_train_y)
df_test = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')

df_test.describe()
np_test = pca.transform(df_test.to_numpy())
df_test['label'] = knn.predict(np_test)
a = []

for i in range(28000):

    a.append(i+1)

df_test['ImageId'] = a

df_test.describe()
df_test[['ImageId','label']].to_csv('submission.csv',index=False)