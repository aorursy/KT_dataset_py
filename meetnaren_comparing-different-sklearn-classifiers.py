# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from sklearn.naive_bayes import GaussianNB

from sklearn.tree import DecisionTreeClassifier

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import cross_val_score, StratifiedKFold



import plotly

import plotly.graph_objs as go

import plotly.offline as ply

plotly.offline.init_notebook_mode()

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
data=pd.read_csv('../input/creditcard.csv')
data.head()
Y = data['Class']

X = data.drop('Class', axis=1)
classifier_names = ['Naive Bayes', 

                    'Decision Tree', 

                    'Random Forest', 

                    'Nearest Neighbors', 

                    'Neural Network']



classifiers = [GaussianNB(),

               DecisionTreeClassifier(max_depth=10),

               RandomForestClassifier(max_depth=10),

               KNeighborsClassifier(5),

               MLPClassifier()]
plot_data=[]



clf_data=zip(classifier_names, classifiers)
for clf_name, clf in clf_data:

    print('Running '+clf_name)

    kf=StratifiedKFold(n_splits=10, shuffle=True)

    scores=cross_val_score(clf, X, Y, cv=kf)

    print(scores)

    plot_data.append(

        go.Scatter(

            x=[i+1 for i in range(10)],

            y=scores,

            mode='lines',

            name=clf_name

        )

    )
layout = go.Layout(

    xaxis=dict(

        title='Fold no.'

    ),

    yaxis=dict(

        range=[np.min([i['y'] for i in plot_data]), 1],

        title='Accuracy'

    )

)

fig=go.Figure(data=plot_data, layout=layout)

ply.iplot(fig)