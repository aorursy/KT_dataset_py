# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

import plotly.offline as plt

import plotly.graph_objs as go





def knn_classifier(path, target_classes, classifier, sample, k):

    training_set = pd.DataFrame(pd.read_csv(path))

    classes_set = training_set[target_classes]



    data_graph = [go.Scatter3d(

        x=classes_set[target_classes[0]],

        y=classes_set[target_classes[2]],

        z=classes_set[target_classes[2]],

        mode='markers',

        marker=dict(

            color='#212121',

        ),

        name='Points from training set'

    ), go.Scatter3d(

        x=[sample[0]],

        y=[sample[1]],

        z=[sample[2]],

        mode='markers',

        marker=dict(

            size=10,

            color='#FFD600',

        ),

        name='New sample'

    )]



    graph_layout = go.Layout(

        scene=dict(

            xaxis=dict(title='mass'),

            yaxis=dict(title='width'),

            zaxis=dict(title='height')

        ),

        margin=dict(b=10, l=10, t=10)

    )



    data_graph = go.Figure(data=data_graph, layout=graph_layout)



   # plt.plot(data_graph, filename='../output_files/knn_classification.html')



    training_set['dist'] = (classes_set[target_classes] - np.array(sample)).pow(2).sum(1).pow(0.5)

    training_set.sort_values('dist', inplace=True)

    return (training_set.iloc[:k][classifier]).value_counts().idxmax()





if __name__ == '__main__':

    pd.set_option('display.max_columns', 10)

    print(knn_classifier(path="/kaggle/input/braincsv/bt_dataset_t3.csv",

                         target_classes=['Mean', 'Variance', 'Entropy','Skewness','Kurtosis','Contrast','Energy','ASM','Homogeneity','Dissimilarity'],

                         classifier='Target',

                         sample=[8.511352539, 1126.214187, 0.868765065, 3.763142356, 15.10757947, 362.2912134, 0.921786235, 0.849689863,0.94929533,2.765725244],

                         k=1))