import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os



# Visualisation libraries

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

sns.set()

from plotly.offline import init_notebook_mode, iplot 

import plotly.graph_objs as go

import plotly.offline as py

import pycountry

py.init_notebook_mode(connected=True)

import folium 

from folium import plugins



import pydicom



# Graphics in retina format 

%config InlineBackend.figure_format = 'retina' 



# Increase the default plot size and set the color scheme

plt.rcParams['figure.figsize'] = 8, 5

#plt.rcParams['image.cmap'] = 'viridis'



# palette of colors to be used for plots

colors = ["steelblue","dodgerblue","lightskyblue","powderblue","cyan","deepskyblue","cyan","darkturquoise","paleturquoise","turquoise"]





# Disable warnings in Anaconda

import warnings

warnings.filterwarnings('ignore')
# Helper functions



def return_count(data,question_part):

    """Counts occurences of each value in a given column"""

    counts_df = data[question_part].value_counts().to_frame()

    return counts_df



def return_percentage(data,question_part):

    """Calculates percent of each value in a given column"""

    total = data[question_part].count()

    counts_df= data[question_part].value_counts().to_frame()

    percentage_df = (counts_df*100)/total

    return percentage_df





    

def plot_graph(data,question,title,x_axis_title,y_axis_title):

    """ plots a percentage bar graph"""

    df = return_percentage(data,question)

    

    trace1 = go.Bar(

                    x = df.index,

                    y = df[question],

                    #orientation='h',

                    marker = dict(color='dodgerblue',

                                 line=dict(color='black',width=1)),

                    text = df.index)

    data = [trace1]

    layout = go.Layout(barmode = "group",title=title,width=800, height=500,

                       xaxis=dict(type='category',categoryorder='array',categoryarray=salary_order,title=y_axis_title),

                       yaxis= dict(title=x_axis_title))

                       

    fig = go.Figure(data = data, layout = layout)

    iplot(fig)    

basepath = '../input/osic-pulmonary-fibrosis-progression/'
train_info = pd.read_csv(basepath + 'train.csv')

train_info.head()
test_info = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/test.csv')

test_info.head()
test_info.shape[0]
train_info.shape[0] / test_info.shape[0]
train_info.Patient.value_counts().max()
test_info.Patient.value_counts().max()
fig, ax = plt.subplots(1,2,figsize=(20,5))

sns.countplot(train_info.Sex, palette="Reds_r", ax=ax[0]);

ax[0].set_xlabel("")

ax[0].set_title("Gender counts in train");



sns.countplot(test_info.Sex, palette="Blues_r", ax=ax[1]);

ax[1].set_xlabel("")

ax[1].set_title("Gender counts in test");
fig, ax = plt.subplots(1,2,figsize=(20,5))



sns.countplot(train_info.Age, color="orangered", ax=ax[0]);

labels = ax[0].get_xticklabels();

ax[0].set_xticklabels(labels, rotation=90);

ax[0].set_xlabel("");

ax[0].set_title("Age distribution in train");



sns.countplot(test_info.Age, color="lightseagreen", ax=ax[1]);

labels = ax[1].get_xticklabels();

ax[1].set_xticklabels(labels, rotation=90);

ax[1].set_xlabel("");

ax[1].set_title("Age distribution in test");
fig, ax = plt.subplots(1,2,figsize=(20,5))



sns.countplot(train_info.SmokingStatus, color="orangered", ax=ax[0]);

labels = ax[0].get_xticklabels();

ax[0].set_xticklabels(labels, rotation=90);

ax[0].set_xlabel("");

ax[0].set_title("Smoking status distribution in train");



sns.countplot(test_info.SmokingStatus, color="lightseagreen", ax=ax[1]);

labels = ax[1].get_xticklabels();

ax[1].set_xticklabels(labels, rotation=90);

ax[1].set_xlabel("");

ax[1].set_title("Smoking status distribution in test");
fig, ax = plt.subplots(2,1,figsize=(20,10))



sns.countplot(train_info.Weeks, color="orangered", ax=ax[0]);

labels = ax[0].get_xticklabels();

ax[0].set_xticklabels(labels, rotation=90);

ax[0].set_xlabel("");

ax[0].set_title("Weeks distribution in train");



sns.countplot(test_info.Weeks, color="lightseagreen", ax=ax[1]);

labels = ax[1].get_xticklabels();

ax[1].set_xticklabels(labels, rotation=90);

ax[1].set_xlabel("");

ax[1].set_title("\nWeeks distribution in test");
fig, ax = plt.subplots(1,2,figsize=(20,5))



sns.countplot(train_info.loc[train_info.Sex == 'Male'].Age, color="orangered", ax=ax[0]);

labels = ax[0].get_xticklabels();

ax[0].set_xticklabels(labels, rotation=90);

ax[0].set_xlabel("");

ax[0].set_title("Distribution of men by age in train");



sns.countplot(train_info.loc[train_info.Sex == 'Female'].Age, color="lightseagreen", ax=ax[1]);

labels = ax[1].get_xticklabels();

ax[1].set_xticklabels(labels, rotation=90);

ax[1].set_xlabel("");

ax[1].set_title("Distribution of women by age in train");
pydicom.__version__
example_folders = os.listdir(basepath + "train/")[:2]

example_folders
example_files = os.listdir(basepath + "train/" + 'ID00060637202187965290703')[0:2]

example_files
train_paths = np.array([])

folders = os.listdir(basepath + "train/")

for folder in folders:

    files = os.listdir(basepath + "train/" + folder)

    for file in files:

        train_paths = np.append(train_paths, [basepath + "train/" + folder + '/' + file])

train_paths[:4]
#https://www.kaggle.com/schlerp/getting-to-know-dicom-and-the-data

def show_dcm_info(file_path):

    dataset = pydicom.dcmread(file_path)

    

    print("Filename.....................:", file_path)

    print()



    pat_name = dataset.PatientName

    display_name = pat_name.family_name + ", " + pat_name.given_name

    print("Patient's name...............:", display_name)

    print("Patient id...................:", dataset.PatientID)

    print("Manufacturer.................:", dataset.Manufacturer)

    print("Manufacturer's Model Name....:", dataset.ManufacturerModelName)

    print("Slice Location...............:", dataset.SliceLocation)

   

    

    

    if 'PixelData' in dataset:

        rows = int(dataset.Rows)

        cols = int(dataset.Columns)

        print("Image size...................: {rows:d} x {cols:d}, {size:d} bytes".format(

            rows=rows, cols=cols, size=len(dataset.PixelData)))

        if 'PixelSpacing' in dataset:

            print("Pixel spacing................:", dataset.PixelSpacing)

            

def plot_pixel_array(dataset, figsize=(5,5)):

    plt.figure(figsize=figsize)

    plt.grid(False)

    plt.imshow(dataset.pixel_array, 'gray')

    plt.show()
show_dcm_info(train_paths[75])



example_dcm = pydicom.dcmread(train_paths[75])

plot_pixel_array(example_dcm)
# source: https://www.kaggle.com/c/siim-isic-melanoma-classification/discussion/154658



def extract_DICOM_attributes(dicom_file_paths):

    df = pd.DataFrame()

    for dicom_file_path in dicom_file_paths:

        dicom_file_dataset = pydicom.read_file(dicom_file_path)

        patient_name = dicom_file_dataset.PatientID

        manufacturer = dicom_file_dataset.Manufacturer

        manufacturer_model_name = dicom_file_dataset.ManufacturerModelName 

        modality = dicom_file_dataset.Modality

        body_part_examined = dicom_file_dataset.BodyPartExamined

        photometric_interpretation = dicom_file_dataset.PhotometricInterpretation

        rows = dicom_file_dataset.Rows

        columns = dicom_file_dataset.Columns



        df = df.append(pd.DataFrame({'patient_name' : patient_name,

                                     'manufacturer' : manufacturer,

                                     'manufacturer_model_name' : manufacturer_model_name,

                                     'modality': modality,

                                     'body_part_examined': body_part_examined,

                                     'photometric_interpretation': photometric_interpretation,

                                     'path': dicom_file_path,

                                     'rows': rows, 

                                     'columns': columns}, index=[0]), ignore_index=True)

    return df
train_dcm_info = extract_DICOM_attributes(train_paths)

train_dcm_info.head()
print(len(train_dcm_info))

columns = ['patient_name', 'manufacturer', 'manufacturer_model_name', 'modality', 'body_part_examined', 'photometric_interpretation',

                                'rows', 'columns']

train_dcm_copy = train_dcm_info[columns].copy()

train_dcm_no_dubl = train_dcm_copy.drop_duplicates(subset=columns)

print(len(train_dcm_no_dubl) )
train_dcm_no_dubl.head()
fig, ax = plt.subplots(figsize=(20,5))



sns.countplot(train_dcm_no_dubl.manufacturer, color="c", ax=ax);

labels = ax.get_xticklabels();

ax.set_xticklabels(labels, rotation=90);

ax.set_xlabel("");

ax.set_title("Distribution of manufacturer in train");

fig, ax = plt.subplots(2,1,figsize=(15,10))



sns.countplot(train_dcm_no_dubl.rows, color="orangered", ax=ax[0]);

labels = ax[0].get_xticklabels();

ax[0].set_xticklabels(labels, rotation=90);

ax[0].set_xlabel("");

ax[0].set_title("Distribution of manufacturer in train");



sns.countplot(train_dcm_no_dubl['columns'], color="lightseagreen", ax=ax[1]);

labels = ax[1].get_xticklabels();

ax[1].set_xticklabels(labels, rotation=90);

ax[1].set_xlabel("");

ax[1].set_title("Distribution of manufacturer in train");
fig, ax = plt.subplots(2,1,figsize=(15,10))



sns.distplot(train_info.FVC, color="g", ax=ax[0]);

ax[0].set_xlabel("");

ax[0].set_title("Distribution of FVC in train");



sns.distplot(train_info.Percent, color="r", ax=ax[1]);

ax[1].set_xlabel("");

ax[1].set_title("Distribution of Percent in train");
percent_100 = train_info.FVC / train_info.Percent * 100

percent_100.mean()
train_info['Percent 100%'] = train_info.FVC / train_info.Percent * 100

train_group_sex = train_info.loc[:, ['FVC', 'Percent', 'Percent 100%','Sex']].groupby(['Sex']).mean()

train_group_sex
train_group_weeks = train_info.loc[:, ['FVC', 'Percent', 'Percent 100%','Weeks']].groupby(['Weeks']).mean()

train_group_weeks['Weeks'] = train_group_weeks.index

train_group_weeks.head()
fig, ax = plt.subplots(3,1,figsize=(15,17))



sns.regplot("Weeks", "FVC", data=train_group_weeks, truncate=False,

                  color="c", order=3, ax=ax[0])

ax[0].set_title("Distribution of average FVC by weeks in train");



sns.regplot("Weeks", "Percent", data=train_group_weeks, truncate=False,

                  color="m", order=3, ax=ax[1]);

ax[1].set_title("Distribution of average Percent by weeks in train");



sns.regplot("Weeks", "Percent 100%", data=train_group_weeks,truncate=False,

                  color="k", order=2, ax=ax[2])

ax[2].set_title("Distribution of average Percent 100% by weeks in train");
train_group_age = train_info.loc[:, ['FVC', 'Percent', 'Percent 100%','Age']].groupby(['Age']).mean()

train_group_age['Age'] = train_group_age.index

train_group_age.head()
fig, ax = plt.subplots(3,1,figsize=(15,17))



sns.regplot("Age", "FVC", data=train_group_age, truncate=False,

                  color="r", order=4, ax=ax[0])

ax[0].set_title("Distribution of average FVC by age in train");



sns.regplot("Age", "Percent", data=train_group_age, truncate=False,

                  color="g", order=4, ax=ax[1]);

ax[1].set_title("Distribution of average Percent by age in train");



sns.regplot("Age", "Percent 100%", data=train_group_age,truncate=False,

                  color="b", order=3, ax=ax[2])

ax[2].set_title("Distribution of average Percent 100% by age in train");
train_group_smoking = train_info.loc[:, ['FVC', 'Percent', 'Percent 100%','SmokingStatus']].groupby(['SmokingStatus']).mean()

train_group_smoking['SmokingStatus'] = train_group_smoking.index

train_group_smoking
fig, ax = plt.subplots(3,1,figsize=(15,17))



sns.barplot(data = train_group_smoking, x = 'SmokingStatus', y ="FVC", ax=ax[0])

ax[0].set_title("Distribution of average FVC by SmokingStatus in train");



sns.barplot(data = train_group_smoking, x = 'SmokingStatus', y ="Percent", ax=ax[1])

ax[1].set_title("Distribution of average Percent by SmokingStatus in train");



sns.barplot(data = train_group_smoking, x = 'SmokingStatus', y ="Percent 100%", ax=ax[2])

ax[2].set_title("Distribution of average Percent 100% by SmokingStatus in train");
def get_tuble(arr, ind):

    ans = np.array([])

    for element in arr:

        ans = np.append(ans, element[ind])

    return ans
train_group_smoking_sex = train_info.loc[:, ['FVC', 'Percent', 'Percent 100%','SmokingStatus', 'Sex']].groupby(['SmokingStatus', 'Sex']).mean()

train_group_smoking_sex['SmokingStatus'] = get_tuble(train_group_smoking_sex.index, 0)

train_group_smoking_sex['Sex'] = get_tuble(train_group_smoking_sex.index, 1)

train_group_smoking_sex
fig, ax = plt.subplots(3,1,figsize=(15,17))



sns.barplot(data = train_group_smoking_sex, x = 'SmokingStatus', y ="FVC", ax=ax[0], hue='Sex')

ax[0].set_title("Distribution of average FVC by SmokingStatus  and gender in train");



sns.barplot(data = train_group_smoking_sex, x = 'SmokingStatus', y ="Percent", ax=ax[1], hue='Sex')

ax[1].set_title("Distribution of average Percent by SmokingStatus and gender in train");



sns.barplot(data = train_group_smoking_sex, x = 'SmokingStatus', y ="Percent 100%", ax=ax[2], hue='Sex' )

ax[2].set_title("Distribution of average Percent 100% by SmokingStatus and gender in train");
train_group_smoking_age = train_info.loc[:, ['FVC', 'Percent', 'Percent 100%','SmokingStatus', 'Age']].groupby(['SmokingStatus', 'Age']).mean()

train_group_smoking_age['SmokingStatus'] = get_tuble(train_group_smoking_age.index, 0)

train_group_smoking_age['Age'] = get_tuble(train_group_smoking_age.index, 1)

train_group_smoking_age.head()
sns.pairplot(train_group_smoking_age, hue="SmokingStatus", palette="husl")