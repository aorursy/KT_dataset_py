import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# data visualisation

from plotly.offline import init_notebook_mode, iplot

import plotly.graph_objs as go

import seaborn as sns

init_notebook_mode(connected=True)

sns.set(style="whitegrid")
sns.color_palette("viridis",3)
# Based on code from:

# https://www.kaggle.com/romankovalenko/data-distribution-3d-scatter-plots/notebook#2.-Define-functions

def draw_3d_plot(data_pd, title_name, categorical_feature='Purchased', color=None):#, name_dict = cover_type):

    cat_feat_n = data_pd.loc[:,categorical_feature].unique()

    palette = [ (0.229739, 0.322361, 0.9),

                (0.127568, 0.9, 0.550556),

                (0.9, 0.1, 0.1)]

    data = []

    

    for i in cat_feat_n:

        temp_trace = go.Scatter3d(

            x=data_pd[data_pd.loc[:,categorical_feature] == i]['Age'],

            y=data_pd[data_pd.loc[:,categorical_feature] == i]['EstimatedSalary'],

            z=data_pd[data_pd.loc[:,categorical_feature] == i]['Gender'],

            mode='markers',

            name='Purchased' if i==1 else 'Not Purchased' if i == 0 else 'Wrong Answer',

            marker=dict(

                size=3,

                color='rgb'+str(palette[i])

            )

        )

        data.append(temp_trace)

    

    



    layout = dict(title=title_name, autosize=True, 

                  scene=dict(xaxis=dict(title='Age.', titlefont=dict(family='Arial, sans-serif',size=10,color='grey')), 

                             yaxis=dict(title='EstimateSalary.', titlefont=dict(family='Arial, sans-serif',size=10,color='grey')),

                             zaxis=dict(title='Gender', titlefont=dict(family='Arial, sans-serif',size=10,color='grey'))));

    fig = go.Figure(data=data, layout=layout)

    iplot(fig)
df = pd.read_csv("/kaggle/input/Social_Network_Ads.csv")

df['Gender'].replace('Male', 0, inplace=True)

df['Gender'].replace('Female', 1, inplace=True)



ID = df.iloc[:, 0].values

X = df.iloc[:, [1,2,3]].values

Y = df.iloc[:, 4].values



df.head()
from sklearn.model_selection import train_test_split



ID_train, ID_test, X_train, X_test, Y_train, Y_test = train_test_split(ID, X, Y, test_size=0.2, random_state=2)
from sklearn import svm



model = svm.SVC(kernel='linear')

model.fit(X_train, Y_train)
test_df = pd.DataFrame(data={

    'User ID': ID_test, 'Gender': X_test[:, 0], 'Age': X_test[:,1], 'EstimatedSalary': X_test[:,2], 'Purchased': Y_test

})

draw_3d_plot(test_df, "Test Data")
Y_predicted = model.predict(X_test)
Y_combined = [0] * len(Y_predicted)

for i in range(len(X_test)):

    Y_combined[i] = Y_predicted[i] if Y_predicted[i]==Y_test[i] else -1



p_df = pd.DataFrame(data={

    'User ID': ID_test, 'Gender': X_test[:, 0], 'Age': X_test[:,1], 'EstimatedSalary': X_test[:,2],

    'Purchased': Y_combined

})



draw_3d_plot(p_df, "Predicted Values")
from sklearn.metrics import accuracy_score



print(accuracy_score(Y_test, Y_predicted))