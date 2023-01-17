# import the necessary libraries

import numpy as np 

import pandas as pd 

import os



# Visualisation libraries

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

sns.set()

import pycountry

import plotly.express as px

from plotly.offline import init_notebook_mode, iplot 

import plotly.graph_objs as go

import plotly.offline as py

from plotly.offline import download_plotlyjs,init_notebook_mode,plot,iplot

!pip install chart_studio

import chart_studio.plotly as py

import cufflinks

cufflinks.go_offline()

cufflinks.set_config_file(world_readable=True, theme='pearl')

#py.init_notebook_mode(connected=True)



#Racing Bar Chart

!pip install bar_chart_race

import bar_chart_race as bcr

from IPython.display import HTML



# Increase the default plot size and set the color scheme

plt.rcParams['figure.figsize'] = 8, 5

plt.style.use("fivethirtyeight")# for pretty graphs



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Disable warnings 

import warnings

warnings.filterwarnings('ignore')
train_df = pd.read_csv('/kaggle/input/janatahack-healthcare-analytics-ii/Train_hMYJ020/train.csv')

test_df = pd.read_csv('/kaggle/input/janatahack-healthcare-analytics-ii/Test_ND2Q3bm/test.csv')

sub_df = pd.read_csv('/kaggle/input/janatahack-healthcare-analytics-ii/sample_submission_lfbv3c3.csv')





#Training data

print('Training data shape: ', train_df.shape)

train_df.head(5)
#Test data

print('Test data shape: ', test_df.shape)

test_df.head(5)
# Null values and Data types

print('Train Set')

print(train_df.info())

print('-------------')

print('Test Set')

print(test_df.info())
train_df.isnull().sum()
test_df.isnull().sum()
# Total number of Patients in the dataset(train+test)

print("Total Patients in Train set: ",train_df['patientid'].nunique())

print("Total Patients in Test set: ",test_df['patientid'].nunique())
train_df.columns
#find overlap between train and test sets

cols =  [

            'case_id', 'Hospital_code', 'Hospital_type_code', 'City_Code_Hospital',

       'Hospital_region_code', 'Available Extra Rooms in Hospital',

       'Department', 'Ward_Type', 'Ward_Facility_Code', 'Bed Grade',

       'patientid', 'City_Code_Patient', 'Type of Admission',

       'Severity of Illness', 'Visitors with Patient', 'Age',

       'Admission_Deposit'

          ]

for col in cols:

  print('Total unique '+col  +' values in Train are {}'.format(train_df[col].nunique()))

  print('Total unique '+col  +' values in Test are {}'.format(test_df[col].nunique()))

  print('Common'+col +' values are {}'.format(len(list(set(train_df[col]) & set(test_df[col])))))

  print('**************************')
for col in train_df.columns:

    print('The unique values of '+col+' column in train_df dataset are {} '.format(train_df[col].nunique()))
for col in test_df.columns:

    print('The unique values of '+col+' column in test dataset are {}'.format(test_df[col].nunique()))
train_df['Stay'].value_counts().iplot(kind='bar',yTitle='Count',color='red')
trace0 = go.Box(y=train_df["Age"],name="Age")



data = [trace0]

iplot(data)
trace0 = go.Box(y=train_df["Admission_Deposit"],name="Admission_Deposit")



data = [trace0]

iplot(data)
train_df['Admission_Deposit'].iplot(kind='hist',bins=30,color='orange',xTitle='Admission_Deposit distribution',yTitle='Count')
labels = train_df['Hospital_type_code'].value_counts().index

values = train_df['Hospital_type_code'].value_counts().values

fig = go.Figure(data=[go.Pie(labels=labels, values=values, textinfo='label+percent',

                             insidetextorientation='radial')])

fig.show()
train_df['Hospital_code'].value_counts().iplot(kind='bar',yTitle='Count',color='green')
City_Code_Hospital = train_df['City_Code_Hospital'].value_counts().sort_values(ascending=False)

City_Code_Hospital.iplot(kind='barh', title='City_Code_Hospital')
Department = train_df['Department'].value_counts().sort_values(ascending=False)[:10]

Department.iplot(kind='bar', title='Department', color = 'blue')
labels = train_df['Ward_Type'].value_counts().index

values = train_df['Ward_Type'].value_counts().values

fig = go.Figure(data=[go.Pie(labels=labels, values=values, textinfo='label+percent',

                             insidetextorientation='radial')])

fig.show()
labels = train_df['Ward_Facility_Code'].value_counts().index

values = train_df['Ward_Facility_Code'].value_counts().values

fig = go.Figure(data=[go.Pie(labels=labels, values=values, textinfo='label+percent',

                             insidetextorientation='radial')])

fig.show()
train_df['Severity of Illness'].value_counts(normalize=True)
train_df['Severity of Illness'].value_counts(normalize=True).iplot(kind='bar',

                                                      yTitle='Percentage', 

                                                      linecolor='black', 

                                                      opacity=0.7,

                                                      color='green',

                                                      theme='pearl',

                                                      bargap=0.8,

                                                      gridcolor='white',

                                                     

                                                      title='Distribution of the Severity of Illness column in the training set')

train_df['Hospital_region_code'].value_counts(normalize=True)
train_df['Hospital_region_code'].value_counts(normalize=True).iplot(kind='bar',

                                                      yTitle='Percentage', 

                                                      linecolor='black', 

                                                      opacity=0.7,

                                                      color='pink',

                                                      theme='pearl',

                                                      bargap=0.8,

                                                      gridcolor='white',

                                                     

                                                      title='Distribution of the Hospital_region_code column in the training set')

labels = train_df['Bed Grade'].value_counts().index

values = train_df['Bed Grade'].value_counts().values

fig = go.Figure(data=[go.Pie(labels=labels, values=values, textinfo='label+percent',

                             insidetextorientation='radial')])

fig.show()
train_df.columns
train_df[['Hospital_code', 'Hospital_type_code', 'City_Code_Hospital',

       'Hospital_region_code', 'Available Extra Rooms in Hospital',

       'Department', 'Ward_Type', 'Ward_Facility_Code', 'Bed Grade',

       'patientid', 'City_Code_Patient', 'Type of Admission',

       'Severity of Illness', 'Visitors with Patient', 'Age',

       'Admission_Deposit', 'Stay']].describe(include='all')
def reduce_mem_usage(df, verbose=True):

    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

    start_mem = df.memory_usage().sum() / 1024**2    

    for col in df.columns:

        col_type = df[col].dtypes

        if col_type in numerics:

            c_min = df[col].min()

            c_max = df[col].max()

            if str(col_type)[:3] == 'int':

                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:

                    df[col] = df[col].astype(np.int8)

                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:

                    df[col] = df[col].astype(np.int16)

                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:

                    df[col] = df[col].astype(np.int32)

                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:

                    df[col] = df[col].astype(np.int64)  

            else:

                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:

                    df[col] = df[col].astype(np.float16)

                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:

                    df[col] = df[col].astype(np.float32)

                else:

                    df[col] = df[col].astype(np.float64)    

    end_mem = df.memory_usage().sum() / 1024**2

    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))

    return df

train_df = reduce_mem_usage(train_df)

test_df = reduce_mem_usage(test_df)