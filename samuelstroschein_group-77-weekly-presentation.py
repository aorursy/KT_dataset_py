import os

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



train_path= '../input/chest-xray-pneumonia/chest_xray/train/'

val_path = '../input/chest-xray-pneumonia/chest_xray/val/'

test_path = '../input/chest-xray-pneumonia/chest_xray/test/'



def count_files(path):

    num_normal = 0

    num_pneumonia_virus = 0

    num_pneumonia_bacteria = 0

    for folder_name in os.listdir(path):

        folder_path = path + folder_name

        for image_name in os.listdir(folder_path):

            if 'virus' in image_name:

                num_pneumonia_virus += 1

            elif 'bacteria' in image_name:

                num_pneumonia_bacteria += 1

            elif 'virus' not in image_name and 'bacteria' not in image_name:

                num_normal += 1

            else:

                raise Exception('Unhandled image!')

    num_pneumonia_combined = num_pneumonia_virus + num_pneumonia_bacteria

    return num_normal, num_pneumonia_combined, num_pneumonia_virus, num_pneumonia_bacteria





train_num_normal, train_num_combined, train_num_virus, train_num_bacteria = count_files(train_path)

val_num_normal, val_num_combined, val_num_virus, val_num_bacteria = count_files(val_path)

test_num_normal, test_num_combined, test_num_virus, test_num_bacteria = count_files(test_path)
# Create pandas dataframe for plotly



num_list =[['Training Set', 'Normal', train_num_normal],

#            ['Training Set', 'Pneumonia', train_num_combined],

           ['Training Set', 'Virus', train_num_virus],

           ['Training Set', 'Bacteria', train_num_bacteria],

           

           ['Validation Set', 'Normal', val_num_normal],

#            ['Validation Set', 'Pneumonia', val_num_combined],

           ['Validation Set', 'Virus', val_num_virus],

           ['Validation Set', 'Bacteria', val_num_bacteria],

           

           ['Test Set', 'Normal', test_num_normal],

#            ['Test Set', 'Pneumonia', test_num_combined],

           ['Test Set', 'Virus', test_num_virus],

           ['Test Set', 'Bacteria', test_num_bacteria]]



plotly_df = pd.DataFrame(num_list, columns=

                         ['Data Set', 'Class', 'Amount']

                        )



plotly_df
import plotly.express as px

fig = px.pie(plotly_df, values='Amount', names="Data Set", title="Distribution Data Set")

fig.show()
fig = px.pie(plotly_df, values='Amount', names="Class", color='Class', title="Classes in whole dataset")

fig.show()
fig = px.bar(plotly_df, x='Data Set', y='Amount', color='Class', title="Bar Chart")

fig.show()