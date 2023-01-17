import os

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt





#plotly imports

!pip install chart_studio

import plotly.express as px

import chart_studio.plotly as py

import plotly.graph_objs as go

from plotly.offline import iplot

import cufflinks

cufflinks.go_offline()

cufflinks.set_config_file(world_readable=True, theme='pearl')
train_dir = '../input/osic-pulmonary-fibrosis-progression/train/'

test_dir = '../input/osic-pulmonary-fibrosis-progression/test/'





train_df = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/train.csv')

test_df = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/test.csv')

print('Training Dataframe shape: ', train_df.shape)



train_df.head(10)
# Let's have a look at the detailed info about the dataframes

print('Training Dataframe Details: ')

print(train_df.info())



print('\n\nTest Dataframe Details: ')

print(test_df.info())





print('Number of patients in training set:',

      len(os.listdir(train_dir)))

print('Number of patients in test set:',

     len(os.listdir(test_dir)))
# Creating unique patient lists and their properties. 

patient_ids = os.listdir(train_dir)

patient_ids = sorted(patient_ids)



#Creating new rows

no_of_instances = []

age = []

sex = []

smoking_status = []



for patient_id in patient_ids:

    patient_info = train_df[train_df['Patient'] == patient_id].reset_index()

    no_of_instances.append(len(os.listdir(train_dir + patient_id)))

    age.append(patient_info['Age'][0])

    sex.append(patient_info['Sex'][0])

    smoking_status.append(patient_info['SmokingStatus'][0])



#Creating the dataframe for the patient info    

patient_df = pd.DataFrame(list(zip(patient_ids, no_of_instances, age, sex, smoking_status)), 

                                 columns =['Patient', 'no_of_instances', 'Age', 'Sex', 'SmokingStatus'])

print(patient_df.info())

patient_df.head()
patient_df['Sex'].value_counts(normalize = True).iplot(kind = 'bar', 

                                                        color = 'blue', 

                                                        yTitle = 'Unique patient count',

                                                        xTitle = 'Gender',

                                                        title = 'Gender Distribution of the unique patients')
import scipy



data = patient_df.Age.tolist()

plt.figure(figsize=(18,6))

# Creating the main histogram

_, bins, _ = plt.hist(data, 15, density=1, alpha=0.5)



# Creating the best fitting line with mean and standard deviation

mu, sigma = scipy.stats.norm.fit(data)

best_fit_line = scipy.stats.norm.pdf(bins, mu, sigma)

plt.plot(bins, best_fit_line, color = 'b', linewidth = 3, label = 'fitting curve')

plt.title(f'Age Distribution [ mean = {"{:.2f}".format(mu)}, standard_dev = {"{:.2f}".format(sigma)} ]', fontsize = 18)

plt.xlabel('Age -->')

plt.show()



patient_df['Age'].iplot(kind='hist',bins=25,color='blue',xTitle='Percent distribution',yTitle='Count')
plt.figure(figsize=(16, 6))

sns.kdeplot(patient_df.loc[patient_df['Sex'] == 'Male', 'Age'], label = 'Male',shade=True)

sns.kdeplot(patient_df.loc[patient_df['Sex'] == 'Female', 'Age'], label = 'Female',shade=True)

plt.xlabel('Age (years)'); plt.ylabel('Density'); plt.title('Distribution of Ages');
patient_df['SmokingStatus'].value_counts(normalize=False ).iplot(kind='bar',

                                                      yTitle='Percentage', 

                                                      linecolor='black', 

                                                      opacity=0.8,

                                                      color='blue',

                                                      theme='pearl',

                                                      bargap=0.5,

                                                      title='SmokingStatus Distribution')
patient_df.groupby(['SmokingStatus', 'Sex']).count()['Patient'].unstack().iplot(kind='bar', 

                                                                                yTitle = 'Unique Patient Count',

                                                                                title = 'Gender vs SmokingStatus' )
plt.figure(figsize=(16, 6))

sns.kdeplot(patient_df.loc[patient_df['SmokingStatus'] == 'Ex-smoker', 'Age'], label = 'Ex-smoker',shade=True)

sns.kdeplot(patient_df.loc[patient_df['SmokingStatus'] == 'Never smoked', 'Age'], label = 'Never smoked',shade=True)

sns.kdeplot(patient_df.loc[patient_df['SmokingStatus'] == 'Currently smokes', 'Age'], label = 'Currently smokes',shade=True)

# Labeling of plot

plt.xlabel('Age (years)'); plt.ylabel('Density'); plt.title('Distribution of Ages');
from ipywidgets import interact  #, interactive, IntSlider, ToggleButtons



def patient_lookup(patient_id):

    print(train_df[train_df['Patient'] == patient_id])

    patient_info = train_df[train_df['Patient'] == patient_id].reset_index()

    fig, (ax1,ax2) = plt.subplots(1,2, figsize = (15, 5))

    ax1.plot(patient_info['Weeks'].tolist() , patient_info['FVC'].tolist(), marker = '*', linewidth = 3,color = 'r', markeredgecolor = 'b')

    ax1.set_title('FVC Deterioriation over the Weeks')

    ax1.set_xlabel('Weeks -->')

    ax1.set_ylabel('FVC')

    ax1.grid(True)

    

    ax2.plot(patient_info['Weeks'].tolist() , patient_info['Percent'].tolist(),marker = '*', linewidth = 3,

            color = 'r', markeredgecolor = 'b' )

    ax2.set_title('Percent change over the weeks')

    ax2.set_xlabel('Weeks -->')

    ax2.set_ylabel('Percent(of adult capacity)')

    ax2.grid(True)

    fig.suptitle(f'P_ID: {patient_id}', fontsize = 20) 

    

    

    

interact(patient_lookup, patient_id = patient_ids)
import random

import pydicom

def explore_dicoms(patient_id, instance):

    RefDs = pydicom.dcmread('../input/osic-pulmonary-fibrosis-progression/train/' + 

                            patient_id +

                            '/' + 

                            str(instance) + '.dcm')

    plt.figure(figsize=(10, 5))

    plt.imshow(RefDs.pixel_array, cmap='gray');

    plt.title(f'P_ID: {patient_id}\nInstance: {instance}')

    plt.axis('off')





def show_ct_scans(patient_id):

    no_of_instances = int(patient_df[patient_df['Patient'] == patient_id]['no_of_instances'].values[0])

    files = sorted(random.sample(range(1, no_of_instances), 9))

    rows = 3

    cols = 3

    fig = plt.figure(figsize=(12,12))

    for idx in range(1, rows*cols+1):

        fig.add_subplot(rows, cols, idx)

        RefDs = pydicom.dcmread(train_dir + patient_id + '/' + str(files[idx-1]) + '.dcm')

        plt.imshow(RefDs.pixel_array, cmap='gray')

        plt.title(f'Instance: {files[idx-1]}')

        plt.axis(False)

        fig.add_subplot

    fig.suptitle(f'P_ID: {patient_id}') 

    plt.show()
# show_ct_scans(patient_ids[0])

interact(show_ct_scans,patient_id = patient_ids)
import warnings

warnings.filterwarnings('ignore')

import pandas_profiling as pdp

unique_patient_profile  = pdp.ProfileReport(patient_df)

unique_patient_profile