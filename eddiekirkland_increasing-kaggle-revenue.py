# importing libraries

import numpy as np

import pandas as pd

import matplotlib as plt

%matplotlib inline



# reading in csv file

mc = pd.read_csv('/kaggle/input/kaggle-survey-2019/multiple_choice_responses.csv')



# exploring overall data

mc.info()

mc.head()
# exploring overall data

mc.info()

mc.head()
# Reassigning header to first row and removing first row

mc.columns = mc.iloc[0,:]

mc = mc[1:]

mc.head()
# Remove duplicates

mc = mc.drop_duplicates()

print(len(mc))
# Removing "text" columns

text_columns = mc.filter(regex=(".*\ Text$"))

mc.drop(text_columns.columns, axis=1, inplace=True)
# Listing column names

mc.columns.values
## Creating 'activities' column

# Assigning Multiple Choice columns into a single column with nan values

test_list = mc.loc[:,'Select any activities that make up an important part of your role at work: (Select all that apply) - Selected Choice - Analyze and understand data to influence product or business decisions':

                   'Select any activities that make up an important part of your role at work: (Select all that apply) - Selected Choice - Other'].apply(

    lambda x: ",".join(x.dropna()), axis=1)

test_list = test_list.replace({'nannannan':np.nan})

mc['activities'] = test_list

# Dropping Multiple Choice columns from range

drop_list = mc.loc[:,'Select any activities that make up an important part of your role at work: (Select all that apply) - Selected Choice - Analyze and understand data to influence product or business decisions':

                   'Select any activities that make up an important part of your role at work: (Select all that apply) - Selected Choice - Other']

drop_list_cols = list(drop_list.columns.values)

mc.drop(drop_list_cols, axis=1, inplace=True)
## Creating 'fav_media' column

# Assigning Multiple Choice columns into a single column with nan values

test_list = mc.loc[:,'Who/what are your favorite media sources that report on data science topics? (Select all that apply) - Selected Choice - Twitter (data science influencers)':

                   'Who/what are your favorite media sources that report on data science topics? (Select all that apply) - Selected Choice - Other'].apply(

    lambda x: ",".join(x.dropna()), axis=1)

test_list = test_list.replace({'nannannan':np.nan})

mc['fav_media'] = test_list

# Dropping Multiple Choice columns from range

drop_list = mc.loc[:,'Who/what are your favorite media sources that report on data science topics? (Select all that apply) - Selected Choice - Twitter (data science influencers)':

                   'Who/what are your favorite media sources that report on data science topics? (Select all that apply) - Selected Choice - Other']

drop_list_cols = list(drop_list.columns.values)

mc.drop(drop_list_cols, axis=1, inplace=True)
## Creating 'dscourse_platforms' column

# Assigning Multiple Choice columns into a single column with nan values

test_list = mc.loc[:,'On which platforms have you begun or completed data science courses? (Select all that apply) - Selected Choice - Udacity':

                   'On which platforms have you begun or completed data science courses? (Select all that apply) - Selected Choice - Other'].apply(

    lambda x: ",".join(x.dropna()), axis=1)

test_list = test_list.replace({'nannannan':np.nan})

mc['dscourse_platforms'] = test_list

# Dropping Multiple Choice columns from range

drop_list = mc.loc[:,'On which platforms have you begun or completed data science courses? (Select all that apply) - Selected Choice - Udacity':

                   'On which platforms have you begun or completed data science courses? (Select all that apply) - Selected Choice - Other']

drop_list_cols = list(drop_list.columns.values)

mc.drop(drop_list_cols, axis=1, inplace=True)
## Creating 'ides_used' column

# Assigning Multiple Choice columns into a single column with nan values

test_list = mc.loc[:,"Which of the following integrated development environments (IDE's) do you use on a regular basis?  (Select all that apply) - Selected Choice - Jupyter (JupyterLab, Jupyter Notebooks, etc) ":

                   "Which of the following integrated development environments (IDE's) do you use on a regular basis?  (Select all that apply) - Selected Choice - Other"].apply(

    lambda x: ",".join(x.dropna()), axis=1)

test_list = test_list.replace({'nannannan':np.nan})

mc['ides_used'] = test_list

# Dropping Multiple Choice columns from range

drop_list = mc.loc[:,"Which of the following integrated development environments (IDE's) do you use on a regular basis?  (Select all that apply) - Selected Choice - Jupyter (JupyterLab, Jupyter Notebooks, etc) ":

                   "Which of the following integrated development environments (IDE's) do you use on a regular basis?  (Select all that apply) - Selected Choice - Other"]

drop_list_cols = list(drop_list.columns.values)

mc.drop(drop_list_cols, axis=1, inplace=True)
## Creating 'notebooks_used' column

# Assigning Multiple Choice columns into a single column with nan values

test_list = mc.loc[:,'Which of the following hosted notebook products do you use on a regular basis?  (Select all that apply) - Selected Choice -  Kaggle Notebooks (Kernels) ':

                   'Which of the following hosted notebook products do you use on a regular basis?  (Select all that apply) - Selected Choice - Other'].apply(

    lambda x: ",".join(x.dropna()), axis=1)

test_list = test_list.replace({'nannannan':np.nan})

mc['notebooks_used'] = test_list

# Dropping Multiple Choice columns from range

drop_list = mc.loc[:,'Which of the following hosted notebook products do you use on a regular basis?  (Select all that apply) - Selected Choice -  Kaggle Notebooks (Kernels) ':

                   'Which of the following hosted notebook products do you use on a regular basis?  (Select all that apply) - Selected Choice - Other']

drop_list_cols = list(drop_list.columns.values)

mc.drop(drop_list_cols, axis=1, inplace=True)
## Creating 'langs_used' column

# Assigning Multiple Choice columns into a single column with nan values

test_list = mc.loc[:,'What programming languages do you use on a regular basis? (Select all that apply) - Selected Choice - Python':

                   'What programming language would you recommend an aspiring data scientist to learn first? - Selected Choice'].apply(

    lambda x: ",".join(x.dropna()), axis=1)

test_list = test_list.replace({'nannannan':np.nan})

mc['langs_used'] = test_list

# Dropping Multiple Choice columns from range

drop_list = mc.loc[:,'What programming languages do you use on a regular basis? (Select all that apply) - Selected Choice - Python':

                   'What programming language would you recommend an aspiring data scientist to learn first? - Selected Choice']

drop_list_cols = list(drop_list.columns.values)

mc.drop(drop_list_cols, axis=1, inplace=True)
## Creating 'dataviz_used' column

# Assigning Multiple Choice columns into a single column with nan values

test_list = mc.loc[:,'What data visualization libraries or tools do you use on a regular basis?  (Select all that apply) - Selected Choice -  Ggplot / ggplot2 ':

                   'What data visualization libraries or tools do you use on a regular basis?  (Select all that apply) - Selected Choice - Other'].apply(

    lambda x: ",".join(x.dropna()), axis=1)

test_list = test_list.replace({'nannannan':np.nan})

mc['dataviz_used'] = test_list

# Dropping Multiple Choice columns from range

drop_list = mc.loc[:,'What data visualization libraries or tools do you use on a regular basis?  (Select all that apply) - Selected Choice -  Ggplot / ggplot2 ':

                   'What data visualization libraries or tools do you use on a regular basis?  (Select all that apply) - Selected Choice - Other']

drop_list_cols = list(drop_list.columns.values)

mc.drop(drop_list_cols, axis=1, inplace=True)
## Creating 'hardware_used' column

# Assigning Multiple Choice columns into a single column with nan values

test_list = mc.loc[:,'Which types of specialized hardware do you use on a regular basis?  (Select all that apply) - Selected Choice - CPUs':

                   'Which types of specialized hardware do you use on a regular basis?  (Select all that apply) - Selected Choice - Other'].apply(

    lambda x: ",".join(x.dropna()), axis=1)

test_list = test_list.replace({'nannannan':np.nan})

mc['hardware_used'] = test_list

# Dropping Multiple Choice columns from range

drop_list = mc.loc[:,'Which types of specialized hardware do you use on a regular basis?  (Select all that apply) - Selected Choice - CPUs':

                   'Which types of specialized hardware do you use on a regular basis?  (Select all that apply) - Selected Choice - Other']

drop_list_cols = list(drop_list.columns.values)

mc.drop(drop_list_cols, axis=1, inplace=True)
## Creating 'ml_alg_used' column

# Assigning Multiple Choice columns into a single column with nan values

test_list = mc.loc[:,'Which of the following ML algorithms do you use on a regular basis? (Select all that apply): - Selected Choice - Linear or Logistic Regression':

                   'Which of the following ML algorithms do you use on a regular basis? (Select all that apply): - Selected Choice - Other'].apply(

    lambda x: ",".join(x.dropna()), axis=1)

test_list = test_list.replace({'nannannan':np.nan})

mc['ml_alg_used'] = test_list

# Dropping Multiple Choice columns from range

drop_list = mc.loc[:,'Which of the following ML algorithms do you use on a regular basis? (Select all that apply): - Selected Choice - Linear or Logistic Regression':

                   'Which of the following ML algorithms do you use on a regular basis? (Select all that apply): - Selected Choice - Other']

drop_list_cols = list(drop_list.columns.values)

mc.drop(drop_list_cols, axis=1, inplace=True)
## Creating 'ml_tools_used' column

# Assigning Multiple Choice columns into a single column with nan values

test_list = mc.loc[:,'Which categories of ML tools do you use on a regular basis?  (Select all that apply) - Selected Choice - Automated data augmentation (e.g. imgaug, albumentations)':

                   'Which categories of ML tools do you use on a regular basis?  (Select all that apply) - Selected Choice - Other'].apply(

    lambda x: ",".join(x.dropna()), axis=1)

test_list = test_list.replace({'nannannan':np.nan})

mc['ml_tools_used'] = test_list

# Dropping Multiple Choice columns from range

drop_list = mc.loc[:,'Which categories of ML tools do you use on a regular basis?  (Select all that apply) - Selected Choice - Automated data augmentation (e.g. imgaug, albumentations)':

                   'Which categories of ML tools do you use on a regular basis?  (Select all that apply) - Selected Choice - Other']

drop_list_cols = list(drop_list.columns.values)

mc.drop(drop_list_cols, axis=1, inplace=True)
## Creating 'cvision_used' column

# Assigning Multiple Choice columns into a single column with nan values

test_list = mc.loc[:,'Which categories of computer vision methods do you use on a regular basis?  (Select all that apply) - Selected Choice - General purpose image/video tools (PIL, cv2, skimage, etc)':

                   'Which categories of computer vision methods do you use on a regular basis?  (Select all that apply) - Selected Choice - Other'].apply(

    lambda x: ",".join(x.dropna()), axis=1)

test_list = test_list.replace({'nannannan':np.nan})

mc['cvision_used'] = test_list

# Dropping Multiple Choice columns from range

drop_list = mc.loc[:,'Which categories of computer vision methods do you use on a regular basis?  (Select all that apply) - Selected Choice - General purpose image/video tools (PIL, cv2, skimage, etc)':

                   'Which categories of computer vision methods do you use on a regular basis?  (Select all that apply) - Selected Choice - Other']

drop_list_cols = list(drop_list.columns.values)

mc.drop(drop_list_cols, axis=1, inplace=True)
## Creating 'nlp_methods_used' column

# Assigning Multiple Choice columns into a single column with nan values

test_list = mc.loc[:,'Which of the following natural language processing (NLP) methods do you use on a regular basis?  (Select all that apply) - Selected Choice - Word embeddings/vectors (GLoVe, fastText, word2vec)':

                   'Which of the following natural language processing (NLP) methods do you use on a regular basis?  (Select all that apply) - Selected Choice - Other'].apply(

    lambda x: ",".join(x.dropna()), axis=1)

test_list = test_list.replace({'nannannan':np.nan})

mc['nlp_methods_used'] = test_list

# Dropping Multiple Choice columns from range

drop_list = mc.loc[:,'Which of the following natural language processing (NLP) methods do you use on a regular basis?  (Select all that apply) - Selected Choice - Word embeddings/vectors (GLoVe, fastText, word2vec)':

                   'Which of the following natural language processing (NLP) methods do you use on a regular basis?  (Select all that apply) - Selected Choice - Other']

drop_list_cols = list(drop_list.columns.values)

mc.drop(drop_list_cols, axis=1, inplace=True)
## Creating 'ml_frameworks_used' column

# Assigning Multiple Choice columns into a single column with nan values

test_list = mc.loc[:,'Which of the following machine learning frameworks do you use on a regular basis? (Select all that apply) - Selected Choice -   Scikit-learn ':

                   'Which of the following machine learning frameworks do you use on a regular basis? (Select all that apply) - Selected Choice - Other'].apply(

    lambda x: ",".join(x.dropna()), axis=1)

test_list = test_list.replace({'nannannan':np.nan})

mc['ml_frameworks_used'] = test_list

# Dropping Multiple Choice columns from range

drop_list = mc.loc[:,'Which of the following machine learning frameworks do you use on a regular basis? (Select all that apply) - Selected Choice -   Scikit-learn ':

                   'Which of the following machine learning frameworks do you use on a regular basis? (Select all that apply) - Selected Choice - Other']

drop_list_cols = list(drop_list.columns.values)

mc.drop(drop_list_cols, axis=1, inplace=True)
## Creating 'cloud_platform_used' column

# Assigning Multiple Choice columns into a single column with nan values

test_list = mc.loc[:,'Which of the following cloud computing platforms do you use on a regular basis? (Select all that apply) - Selected Choice -  Google Cloud Platform (GCP) ':

                   'Which of the following cloud computing platforms do you use on a regular basis? (Select all that apply) - Selected Choice - Other'].apply(

    lambda x: ",".join(x.dropna()), axis=1)

test_list = test_list.replace({'nannannan':np.nan})

mc['cloud_platform_used'] = test_list

# Dropping Multiple Choice columns from range

drop_list = mc.loc[:,'Which of the following cloud computing platforms do you use on a regular basis? (Select all that apply) - Selected Choice -  Google Cloud Platform (GCP) ':

                   'Which of the following cloud computing platforms do you use on a regular basis? (Select all that apply) - Selected Choice - Other']

drop_list_cols = list(drop_list.columns.values)

mc.drop(drop_list_cols, axis=1, inplace=True)
## Creating 'cloud_products_used' column

# Assigning Multiple Choice columns into a single column with nan values

test_list = mc.loc[:,'Which specific cloud computing products do you use on a regular basis? (Select all that apply) - Selected Choice - AWS Elastic Compute Cloud (EC2)':

                   'Which specific cloud computing products do you use on a regular basis? (Select all that apply) - Selected Choice - Other'].apply(

    lambda x: ",".join(x.dropna()), axis=1)

test_list = test_list.replace({'nannannan':np.nan})

mc['cloud_products_used'] = test_list

# Dropping Multiple Choice columns from range

drop_list = mc.loc[:,'Which specific cloud computing products do you use on a regular basis? (Select all that apply) - Selected Choice - AWS Elastic Compute Cloud (EC2)':

                   'Which specific cloud computing products do you use on a regular basis? (Select all that apply) - Selected Choice - Other']

drop_list_cols = list(drop_list.columns.values)

mc.drop(drop_list_cols, axis=1, inplace=True)
## Creating 'bigdataanalytics_used' column

# Assigning Multiple Choice columns into a single column with nan values

test_list = mc.loc[:,'Which specific big data / analytics products do you use on a regular basis? (Select all that apply) - Selected Choice - Google BigQuery':

                   'Which specific big data / analytics products do you use on a regular basis? (Select all that apply) - Selected Choice - Other'].apply(

    lambda x: ",".join(x.dropna()), axis=1)

test_list = test_list.replace({'nannannan':np.nan})

mc['bigdataanalytics_used_used'] = test_list

# Dropping Multiple Choice columns from range

drop_list = mc.loc[:,'Which specific big data / analytics products do you use on a regular basis? (Select all that apply) - Selected Choice - Google BigQuery':

                   'Which specific big data / analytics products do you use on a regular basis? (Select all that apply) - Selected Choice - Other']

drop_list_cols = list(drop_list.columns.values)

mc.drop(drop_list_cols, axis=1, inplace=True)
## Creating 'ml_products_used' column

# Assigning Multiple Choice columns into a single column with nan values

test_list = mc.loc[:,'Which of the following machine learning products do you use on a regular basis? (Select all that apply) - Selected Choice - SAS':

                   'Which of the following machine learning products do you use on a regular basis? (Select all that apply) - Selected Choice - Other'].apply(

    lambda x: ",".join(x.dropna()), axis=1)

test_list = test_list.replace({'nannannan':np.nan})

mc['ml_products_used'] = test_list

# Dropping Multiple Choice columns from range

drop_list = mc.loc[:,'Which of the following machine learning products do you use on a regular basis? (Select all that apply) - Selected Choice - SAS':

                   'Which of the following machine learning products do you use on a regular basis? (Select all that apply) - Selected Choice - Other']

drop_list_cols = list(drop_list.columns.values)

mc.drop(drop_list_cols, axis=1, inplace=True)
## Creating 'automl_tools_used' column

# Assigning Multiple Choice columns into a single column with nan values

test_list = mc.loc[:,'Which automated machine learning tools (or partial AutoML tools) do you use on a regular basis?  (Select all that apply) - Selected Choice -  Google AutoML ':

                   'Which automated machine learning tools (or partial AutoML tools) do you use on a regular basis?  (Select all that apply) - Selected Choice - Other'].apply(

    lambda x: ",".join(x.dropna()), axis=1)

test_list = test_list.replace({'nannannan':np.nan})

mc['automl_tools_used'] = test_list

# Dropping Multiple Choice columns from range

drop_list = mc.loc[:,'Which automated machine learning tools (or partial AutoML tools) do you use on a regular basis?  (Select all that apply) - Selected Choice -  Google AutoML ':

                   'Which automated machine learning tools (or partial AutoML tools) do you use on a regular basis?  (Select all that apply) - Selected Choice - Other']

drop_list_cols = list(drop_list.columns.values)

mc.drop(drop_list_cols, axis=1, inplace=True)
## Creating 'reldb_products_used' column

# Assigning Multiple Choice columns into a single column with nan values

test_list = mc.loc[:,'Which of the following relational database products do you use on a regular basis? (Select all that apply) - Selected Choice - MySQL':

                   'Which of the following relational database products do you use on a regular basis? (Select all that apply) - Selected Choice - Other'].apply(

    lambda x: ",".join(x.dropna()), axis=1)

test_list = test_list.replace({'nannannan':np.nan})

mc['reldb_products_used'] = test_list

# Dropping Multiple Choice columns from range

drop_list = mc.loc[:,'Which of the following relational database products do you use on a regular basis? (Select all that apply) - Selected Choice - MySQL':

                   'Which of the following relational database products do you use on a regular basis? (Select all that apply) - Selected Choice - Other']

drop_list_cols = list(drop_list.columns.values)

mc.drop(drop_list_cols, axis=1, inplace=True)
# Making column names readable

column_clean = {'Duration (in seconds)':'duration',

                'What is your age (# years)?':'age',

                'What is your gender? - Selected Choice':'gender',

                'In which country do you currently reside?':'country',

                'What is the highest level of formal education that you have attained or plan to attain within the next 2 years?':'education',

                'Select the title most similar to your current role (or most recent title if retired): - Selected Choice':'title',

                'What is the size of the company where you are employed?':'comp_size',

                'Approximately how many individuals are responsible for data science workloads at your place of business?':'ds_teamsize',

                'Does your current employer incorporate machine learning methods into their business?':'use_ml',

                'What is your current yearly compensation (approximate $USD)?':'compensation',

                'Approximately how much money have you spent on machine learning and/or cloud computing products at your work in the past 5 years?':'dollars_mlorcloud',

                'What is the primary tool that you use at work or school to analyze data? (Include text response) - Selected Choice':'prim_analaysistool',

                'How long have you been writing code to analyze data (at work or at school)?':'coding_years',

                'Have you ever used a TPU (tensor processing unit)?':'used_tpu',

                'For how many years have you used machine learning methods?':'ml_years'}

mc = mc.rename(columns=column_clean)
#replacing null str with NaN

mc = mc.replace('None', np.nan)

mc = mc.replace('', np.nan)
mc.info()
# Checking for remaining null values

mc.apply(lambda x: sum(x.notnull()))
#Filter categorical variables

categorical_columns = [x for x in mc.dtypes.index if mc.dtypes[x]=='object']

#Exclude ID cols and source:

categorical_columns = [x for x in mc if x not in ['duration', 'activities', 'fav_media', 'dscourse_platforms', 'ides_used',

                      'notebooks_used', 'langs_used', 'dataviz_used', 'hardware_used', 'ml_alg_used',

                      'ml_tools_used', 'cvision_used', 'nlp_methods_used', 'ml_frameworks_used',

                      'cloud_platform_used', 'cloud_products_used', 'bigdataanalytics_used_used',

                      'ml_products_used', 'automl_tools_used', 'reldb_products_used']]

#Print frequency of all relevant categories

for col in categorical_columns:

    print('\nFrequency of Categories for %s'%col)

    print(mc[col].value_counts())
# Creating Age Frequency Table

age_freq = mc['age'].value_counts(normalize=True)

age_freq = age_freq.sort_index(axis=0)

age_freq = age_freq.reset_index()

age_freq = pd.DataFrame(age_freq)
# Setting style for bar graphs

import matplotlib.pyplot as plt

%matplotlib inline



# set font

plt.rcParams['font.family'] = 'sans-serif'

plt.rcParams['font.sans-serif'] = 'Helvetica'



# set the style of the axes and the text color

plt.rcParams['axes.edgecolor']='#333F4B'

plt.rcParams['axes.linewidth']=0.8

plt.rcParams['xtick.color']='#333F4B'

plt.rcParams['ytick.color']='#333F4B'

plt.rcParams['text.color']='#333F4B'



# numeric placeholder for the y axis

my_range=list(range(1,len(age_freq.index)+1))



fig, ax = plt.subplots(figsize=(8,5))



# create for each bin a vertical line that starts at y = 0 with the length 

# represented by the specific percentage.

plt.vlines(x=my_range, ymin=0, ymax=age_freq['age'], color='#007ACC', alpha=0.5, linewidth=30)



# create for each bin a dot at the level of the expense percentage value

# plt.plot(my_range, age_freq['age'], "o", markersize=10, color='#007ACC', alpha=0.6)



# set labels

ax.set_xlabel('Age', fontsize=15, fontweight='black', color = '#333F4B')

ax.set_ylabel('% of Respondents', fontsize=15, fontweight='black', color = '#333F4B')



# set axis

ax.tick_params(axis='both', which='major', labelsize=12)

plt.xticks(my_range, age_freq['index'])



# add an horizonal label for the y axis 

# fig.text(-0.15, 0.5, '% of Respondants', fontsize=15, fontweight='black', color = '#333F4B')



# change the style of the axis spines

ax.spines['top'].set_color('none')

ax.spines['right'].set_color('none')

ax.spines['left'].set_smart_bounds(True)

ax.spines['bottom'].set_smart_bounds(True)



# set the spines position

ax.spines['bottom'].set_position(('axes', -0.04))

ax.spines['left'].set_position(('axes', 0.015))



# set background color

ax.set_facecolor('white')



# add margin to y-axis

ax.margins(0.1, 0)



# add title

fig.suptitle('Age of Survey Respondents', fontsize=18, fontweight='black')



plt.savefig('age_freq.png', dpi=300, bbox_inches='tight')
# Creating Gender Frequency Table

gender_freq = mc['gender'].value_counts(normalize=True)

gender_freq = gender_freq.reset_index()

gender_freq = gender_freq.sort_index(axis=0)

gender_freq = pd.DataFrame(gender_freq)
# Creating Gender Frequency Graph

# Setting style for bar graphs

%matplotlib inline



# set font

plt.rcParams['font.family'] = 'sans-serif'

plt.rcParams['font.sans-serif'] = 'Helvetica'



# set the style of the axes and the text color

plt.rcParams['axes.edgecolor']='#333F4B'

plt.rcParams['axes.linewidth']=0.8

plt.rcParams['xtick.color']='#333F4B'

plt.rcParams['ytick.color']='#333F4B'

plt.rcParams['text.color']='#333F4B'



# numeric placeholder for the y axis

my_range=list(range(1,len(gender_freq.index)+1))



fig, ax = plt.subplots(figsize=(8,5))



# create for each bin a vertical line that starts at y = 0 with the length 

# represented by the specific percentage.

plt.vlines(x=my_range, ymin=0, ymax=gender_freq['gender'], color='#007ACC', alpha=0.5, linewidth=50)



# create for each bin a dot at the level of the expense percentage value

# plt.plot(my_range, age_freq['age'], "o", markersize=10, color='#007ACC', alpha=0.6)



# set labels

ax.set_xlabel('Gender', fontsize=15, fontweight='black', color = '#333F4B')

ax.set_ylabel('% of Respondents', fontsize=15, fontweight='black', color = '#333F4B')



# set axis

ax.tick_params(axis='both', which='major', labelsize=12)

plt.xticks(my_range, gender_freq['index'])



# add an horizonal label for the y axis 

# fig.text(-0.15, 0.5, '% of Respondants', fontsize=15, fontweight='black', color = '#333F4B')



# change the style of the axis spines

ax.spines['top'].set_color('none')

ax.spines['right'].set_color('none')

ax.spines['left'].set_smart_bounds(True)

ax.spines['bottom'].set_smart_bounds(True)



# set the spines position

ax.spines['bottom'].set_position(('axes', -0.04))

ax.spines['left'].set_position(('axes', 0.015))



# set background color

ax.set_facecolor('white')



# add margin to y-axis

ax.margins(0.2, 0)



# add title

fig.suptitle('Gender of Survey Respondents', fontsize=18, fontweight='black')



plt.savefig('gender_freq.png', dpi=300, bbox_inches='tight')
# Renaming long value name

mc['country'] = mc['country'].replace('United Kingdom of Great Britain and Northern Ireland', 'UK')

mc['country'] = mc['country'].replace('United States of America', 'USA')

# Creating Country Frequency Table

country_freq = mc['country'].value_counts(normalize=True, ascending=False)

#country_freq = country_freq.sort_index(axis=0)

country_freq = country_freq.reset_index()

country_freq = pd.DataFrame(country_freq)

country_freq = country_freq.loc[country_freq['country'] >= .01]
# Creating Country Frequency Graph

# Setting style for bar graphs

%matplotlib inline



# set font

plt.rcParams['font.family'] = 'sans-serif'

plt.rcParams['font.sans-serif'] = 'Helvetica'



# set the style of the axes and the text color

plt.rcParams['axes.edgecolor']='#333F4B'

plt.rcParams['axes.linewidth']=0.8

plt.rcParams['xtick.color']='#333F4B'

plt.rcParams['ytick.color']='#333F4B'

plt.rcParams['text.color']='#333F4B'



# numeric placeholder for the y axis

my_range=list(range(1,len(country_freq.index)+1))



fig, ax = plt.subplots(figsize=(20,6))



# create for each bin a vertical line that starts at y = 0 with the length 

# represented by the specific percentage.

plt.vlines(x=my_range, ymin=0, ymax=country_freq['country'], color='#007ACC', alpha=0.5, linewidth=15)



# create for each bin a dot at the level of the expense percentage value

plt.plot(my_range, country_freq['country'], "o", markersize=15, color='#007ACC', alpha=0.9)



# set labels

ax.set_xlabel('Country', fontsize=15, fontweight='black', color = '#333F4B')

ax.set_ylabel('% of Respondents', fontsize=15, fontweight='black', color = '#333F4B')



# set axis

ax.tick_params(axis='both', which='major', labelsize=12)

plt.xticks(my_range, country_freq['index'], rotation=45)



# add an horizonal label for the y axis 

# fig.text(-0.15, 0.5, '% of Respondants', fontsize=15, fontweight='black', color = '#333F4B')



# change the style of the axis spines

ax.spines['top'].set_color('none')

ax.spines['right'].set_color('none')

ax.spines['left'].set_smart_bounds(True)

ax.spines['bottom'].set_smart_bounds(True)



# set the spines position

ax.spines['bottom'].set_position(('axes', -0.04))

ax.spines['left'].set_position(('axes', 0.015))



# set background color

ax.set_facecolor('white')



# add margin to y-axis

ax.margins(0.05, 0.03)

ax.set_xticklabels(country_freq['index'], rotation=90)



# add title

fig.suptitle('Country of Survey Respondents (> 1%)', fontsize=18, fontweight='black')



plt.savefig('country_freq.png', dpi=300, bbox_inches='tight')
# Renaming long descriptors

mc['education'] = mc['education'].replace("Some college/university study without earning a bachelor’s degree", 'Some college')

mc['education'] = mc['education'].replace('No formal education past high school', 'High School')



# Creating Education Frequency Table

education_freq = mc['education'].value_counts(normalize=True)

education_freq = education_freq.reset_index()

gender_freq = education_freq.sort_index(axis=0)

education_freq = pd.DataFrame(education_freq)
# Creating Education Frequency Graph

# Setting style for bar graphs

%matplotlib inline



# set font

plt.rcParams['font.family'] = 'sans-serif'

plt.rcParams['font.sans-serif'] = 'Helvetica'



# set the style of the axes and the text color

plt.rcParams['axes.edgecolor']='#333F4B'

plt.rcParams['axes.linewidth']=0.8

plt.rcParams['xtick.color']='#333F4B'

plt.rcParams['ytick.color']='#333F4B'

plt.rcParams['text.color']='#333F4B'



# numeric placeholder for the y axis

my_range=list(range(1,len(education_freq.index)+1))



fig, ax = plt.subplots(figsize=(8,5))



# create for each bin a vertical line that starts at y = 0 with the length 

# represented by the specific percentage.

plt.vlines(x=my_range, ymin=0, ymax=education_freq['education'], color='#007ACC', alpha=0.5, linewidth=50)



# create for each bin a dot at the level of the expense percentage value

# plt.plot(my_range, age_freq['age'], "o", markersize=10, color='#007ACC', alpha=0.6)



# set labels

ax.set_xlabel('Education Level', fontsize=15, fontweight='black', color = '#333F4B')

ax.set_ylabel('% of Respondents', fontsize=15, fontweight='black', color = '#333F4B')



# set axis

ax.tick_params(axis='both', which='major', labelsize=12)

plt.xticks(my_range, education_freq['index'])



# add an horizonal label for the y axis 

# fig.text(-0.15, 0.5, '% of Respondants', fontsize=15, fontweight='black', color = '#333F4B')



# change the style of the axis spines

ax.spines['top'].set_color('none')

ax.spines['right'].set_color('none')

ax.spines['left'].set_smart_bounds(True)

ax.spines['bottom'].set_smart_bounds(True)



# set the spines position

ax.spines['bottom'].set_position(('axes', -0.04))

ax.spines['left'].set_position(('axes', 0.015))



# set background color

ax.set_facecolor('white')



# add margin to y-axis

ax.margins(0.2, 0)

ax.set_xticklabels(education_freq['index'], rotation=90)



# add title

fig.suptitle('Highest Education Level of Survey Respondents', fontsize=18, fontweight='black')



plt.savefig('education_freq.png', dpi=300, bbox_inches='tight')
# Creating Title Frequency Table

title_freq = mc['title'].value_counts(normalize=True)

title_freq = title_freq.reset_index()

title_freq = title_freq.sort_index(axis=0)

title_freq = pd.DataFrame(title_freq)
# Creating Title Frequency Graph

# Setting style for bar graphs

%matplotlib inline



# set font

plt.rcParams['font.family'] = 'sans-serif'

plt.rcParams['font.sans-serif'] = 'Helvetica'



# set the style of the axes and the text color

plt.rcParams['axes.edgecolor']='#333F4B'

plt.rcParams['axes.linewidth']=0.8

plt.rcParams['xtick.color']='#333F4B'

plt.rcParams['ytick.color']='#333F4B'

plt.rcParams['text.color']='#333F4B'



# numeric placeholder for the y axis

my_range=list(range(1,len(title_freq.index)+1))



fig, ax = plt.subplots(figsize=(10,6))



# create for each bin a vertical line that starts at y = 0 with the length 

# represented by the specific percentage.

plt.vlines(x=my_range, ymin=0, ymax=title_freq['title'], color='#007ACC', alpha=0.5, linewidth=15)



# create for each bin a dot at the level of the expense percentage value

plt.plot(my_range, title_freq['title'], "o", markersize=15, color='#007ACC', alpha=0.9)



# set labels

ax.set_xlabel('Job Title', fontsize=15, fontweight='black', color = '#333F4B')

ax.set_ylabel('% of Respondents', fontsize=15, fontweight='black', color = '#333F4B')



# set axis

ax.tick_params(axis='both', which='major', labelsize=12)

plt.xticks(my_range, title_freq['index'], rotation=45)



# add an horizonal label for the y axis 

# fig.text(-0.15, 0.5, '% of Respondants', fontsize=15, fontweight='black', color = '#333F4B')



# change the style of the axis spines

ax.spines['top'].set_color('none')

ax.spines['right'].set_color('none')

ax.spines['left'].set_smart_bounds(True)

ax.spines['bottom'].set_smart_bounds(True)



# set the spines position

ax.spines['bottom'].set_position(('axes', -0.04))

ax.spines['left'].set_position(('axes', 0.015))



# set background color

ax.set_facecolor('white')



# add margin to y-axis

ax.margins(0.05, 0.03)

ax.set_xticklabels(title_freq['index'], rotation=90)



# add title

fig.suptitle('Job Title of Survey Respondents', fontsize=18, fontweight='black')



plt.savefig('title_freq.png', dpi=300, bbox_inches='tight')
# Creating Data Science Team Size Frequency Table

ds_teamsize_freq = mc['ds_teamsize'].value_counts(normalize=True)

ds_teamsize_freq = ds_teamsize_freq.reset_index()

ds_teamsize_freq = ds_teamsize_freq.sort_index(axis=0)

ds_teamsize_freq = pd.DataFrame(ds_teamsize_freq)
# Creating Data Science Team Size Frequency Graph

# Setting style for bar graphs

%matplotlib inline



# set font

plt.rcParams['font.family'] = 'sans-serif'

plt.rcParams['font.sans-serif'] = 'Helvetica'



# set the style of the axes and the text color

plt.rcParams['axes.edgecolor']='#333F4B'

plt.rcParams['axes.linewidth']=0.8

plt.rcParams['xtick.color']='#333F4B'

plt.rcParams['ytick.color']='#333F4B'

plt.rcParams['text.color']='#333F4B'



# numeric placeholder for the y axis

my_range=list(range(1,len(ds_teamsize_freq.index)+1))



fig, ax = plt.subplots(figsize=(10,6))



# create for each bin a vertical line that starts at y = 0 with the length 

# represented by the specific percentage.

plt.vlines(x=my_range, ymin=0, ymax=ds_teamsize_freq['ds_teamsize'], color='#007ACC', alpha=0.5, linewidth=15)



# create for each bin a dot at the level of the expense percentage value

plt.plot(my_range, ds_teamsize_freq['ds_teamsize'], "o", markersize=15, color='#007ACC', alpha=0.9)



# set labels

ax.set_xlabel('# of People', fontsize=15, fontweight='black', color = '#333F4B')

ax.set_ylabel('% of Respondents', fontsize=15, fontweight='black', color = '#333F4B')



# set axis

ax.tick_params(axis='both', which='major', labelsize=12)

plt.xticks(my_range, ds_teamsize_freq['index'], rotation=45)



# add an horizonal label for the y axis 

# fig.text(-0.15, 0.5, '% of Respondants', fontsize=15, fontweight='black', color = '#333F4B')



# change the style of the axis spines

ax.spines['top'].set_color('none')

ax.spines['right'].set_color('none')

ax.spines['left'].set_smart_bounds(True)

ax.spines['bottom'].set_smart_bounds(True)



# set the spines position

ax.spines['bottom'].set_position(('axes', -0.04))

ax.spines['left'].set_position(('axes', 0.015))



# set background color

ax.set_facecolor('white')



# add margin to y-axis

ax.margins(0.05, 0.03)

ax.set_xticklabels(ds_teamsize_freq['index'], rotation=90)



# add ds_teamsize

fig.suptitle('Data Science Team Size of Survey Respondents', fontsize=18, fontweight='black')



plt.savefig('teamsize_freq.png', dpi=300, bbox_inches='tight')
# Creating larger bins for compensation

mc['compensationhi'] = mc['compensation'].str.extract(r'(?<=-|\$)(.*$)')

mc['compensationhi'] = mc['compensationhi'].str.replace(',','')

mc['compensationhi'] = mc['compensationhi'].str.replace('-','')

mc['compensationhi'] = mc['compensationhi'].str.replace('<','')

mc['compensationhi'] = mc['compensationhi'].astype(float)

mc['compensationbins'] = pd.cut(mc['compensationhi'], [0, 50000, 100000, 200000, 300000, 500000, 1000000], 

                                labels=['$0-50K', '$50-100K', '$100-200K', '$200-300K', '$300-500K', '$500K+'])
# Creating Compensation Frequency Table

compensationbins_freq = mc['compensationbins'].value_counts(normalize=True)

compensationbins_freq = compensationbins_freq.reset_index()

compensationbins_freq = compensationbins_freq.sort_index(axis=0)

compensationbins_freq = pd.DataFrame(compensationbins_freq)
# Creating Compensation Frequency Graph

# Setting style for bar graphs

%matplotlib inline



# set font

plt.rcParams['font.family'] = 'sans-serif'

plt.rcParams['font.sans-serif'] = 'Helvetica'



# set the style of the axes and the text color

plt.rcParams['axes.edgecolor']='#333F4B'

plt.rcParams['axes.linewidth']=0.8

plt.rcParams['xtick.color']='#333F4B'

plt.rcParams['ytick.color']='#333F4B'

plt.rcParams['text.color']='#333F4B'



# numeric placeholder for the y axis

my_range=list(range(1,len(compensationbins_freq.index)+1))



fig, ax = plt.subplots(figsize=(8,5))



# create for each bin a vertical line that starts at y = 0 with the length 

# represented by the specific percentage.

plt.vlines(x=my_range, ymin=0, ymax=compensationbins_freq['compensationbins'], color='#007ACC', alpha=0.5, linewidth=50)



# create for each bin a dot at the level of the expense percentage value

# plt.plot(my_range, age_freq['age'], "o", markersize=10, color='#007ACC', alpha=0.6)



# set labels

ax.set_xlabel('', fontsize=15, fontweight='black', color = '#333F4B')

ax.set_ylabel('% of Respondents', fontsize=15, fontweight='black', color = '#333F4B')



# set axis

ax.tick_params(axis='both', which='major', labelsize=12)

plt.xticks(my_range, compensationbins_freq['index'])



# add an horizonal label for the y axis 

# fig.text(-0.15, 0.5, '% of Respondants', fontsize=15, fontweight='black', color = '#333F4B')



# change the style of the axis spines

ax.spines['top'].set_color('none')

ax.spines['right'].set_color('none')

ax.spines['left'].set_smart_bounds(True)

ax.spines['bottom'].set_smart_bounds(True)



# set the spines position

ax.spines['bottom'].set_position(('axes', -0.04))

ax.spines['left'].set_position(('axes', 0.015))



# set background color

ax.set_facecolor('white')



# add margin to y-axis

ax.margins(0.2, 0)



# add title

fig.suptitle('Annual Compensation of Survey Respondents', fontsize=18, fontweight='black')



plt.savefig('compensationbins_freq.png', dpi=300, bbox_inches='tight')
# Renaming bins for dollars spent

rename = {'$0 (USD)':'$0','$100-$999':'$100-$1k',

          '$1000-$9,999':'$1K-$10K','$1-$99':'$1-$100',

          '$10,000-$99,999':'$10K-$100K','> $100,000 ($USD)':'$100K+'}

mc['dollars_mlorcloud'] = mc['dollars_mlorcloud'].replace(rename)
# Creating dollars spent Frequency Table

dollars_mlorcloud = mc['dollars_mlorcloud'].value_counts(normalize=True)

dollars_mlorcloud = dollars_mlorcloud.reset_index()

dollars_mlorcloud = dollars_mlorcloud.sort_index(axis=0)

dollars_mlorcloud = pd.DataFrame(dollars_mlorcloud)
# Creating Dollars Spent Frequency Graph

# Setting style for bar graphs

%matplotlib inline



# set font

plt.rcParams['font.family'] = 'sans-serif'

plt.rcParams['font.sans-serif'] = 'Helvetica'



# set the style of the axes and the text color

plt.rcParams['axes.edgecolor']='#333F4B'

plt.rcParams['axes.linewidth']=0.8

plt.rcParams['xtick.color']='#333F4B'

plt.rcParams['ytick.color']='#333F4B'

plt.rcParams['text.color']='#333F4B'



# numeric placeholder for the y axis

my_range=list(range(1,len(dollars_mlorcloud.index)+1))



fig, ax = plt.subplots(figsize=(11,5))



# create for each bin a vertical line that starts at y = 0 with the length 

# represented by the specific percentage.

plt.vlines(x=my_range, ymin=0, ymax=dollars_mlorcloud['dollars_mlorcloud'], color='#007ACC', alpha=0.5, linewidth=50)



# create for each bin a dot at the level of the expense percentage value

# plt.plot(my_range, age_freq['age'], "o", markersize=10, color='#007ACC', alpha=0.6)



# set labels

ax.set_xlabel('', fontsize=15, fontweight='black', color = '#333F4B')

ax.set_ylabel('% of Respondents', fontsize=15, fontweight='black', color = '#333F4B')



# set axis

ax.tick_params(axis='both', which='major', labelsize=12)

plt.xticks(my_range, dollars_mlorcloud['index'])



# add an horizonal label for the y axis 

# fig.text(-0.15, 0.5, '% of Respondants', fontsize=15, fontweight='black', color = '#333F4B')



# change the style of the axis spines

ax.spines['top'].set_color('none')

ax.spines['right'].set_color('none')

ax.spines['left'].set_smart_bounds(True)

ax.spines['bottom'].set_smart_bounds(True)



# set the spines position

ax.spines['bottom'].set_position(('axes', -0.04))

ax.spines['left'].set_position(('axes', 0.015))



# set background color

ax.set_facecolor('white')



# add margin to y-axis

ax.margins(0.2, 0)



# add title

fig.suptitle('$ Spent on Machine Learning or Cloud Computing (Per Year)', fontsize=18, fontweight='black')



plt.savefig('dollars_mlorcloud.png', dpi=300, bbox_inches='tight')
# Renaming bins for machine learning years

mc['ml_years'] = mc['ml_years'].str.replace(' years', '')



# Creating machine learning years spent Frequency Table

ml_years = mc['ml_years'].value_counts(normalize=True)

ml_years = ml_years.reset_index()

ml_years = ml_years.sort_index(axis=0)

ml_years = pd.DataFrame(ml_years)
# Creating machine learning years Frequency Graph

# Setting style for bar graphs

%matplotlib inline



# set font

plt.rcParams['font.family'] = 'sans-serif'

plt.rcParams['font.sans-serif'] = 'Helvetica'



# set the style of the axes and the text color

plt.rcParams['axes.edgecolor']='#333F4B'

plt.rcParams['axes.linewidth']=0.8

plt.rcParams['xtick.color']='#333F4B'

plt.rcParams['ytick.color']='#333F4B'

plt.rcParams['text.color']='#333F4B'



# numeric placeholder for the y axis

my_range=list(range(1,len(ml_years.index)+1))



fig, ax = plt.subplots(figsize=(11,5))



# create for each bin a vertical line that starts at y = 0 with the length 

# represented by the specific percentage.

plt.vlines(x=my_range, ymin=0, ymax=ml_years['ml_years'], color='#007ACC', alpha=0.5, linewidth=50)



# create for each bin a dot at the level of the expense percentage value

# plt.plot(my_range, age_freq['age'], "o", markersize=10, color='#007ACC', alpha=0.6)



# set labels

ax.set_xlabel('', fontsize=15, fontweight='black', color = '#333F4B')

ax.set_ylabel('% of Respondents', fontsize=15, fontweight='black', color = '#333F4B')



# set axis

ax.tick_params(axis='both', which='major', labelsize=12)

plt.xticks(my_range, ml_years['index'])



# add an horizonal label for the y axis 

# fig.text(-0.15, 0.5, '% of Respondants', fontsize=15, fontweight='black', color = '#333F4B')



# change the style of the axis spines

ax.spines['top'].set_color('none')

ax.spines['right'].set_color('none')

ax.spines['left'].set_smart_bounds(True)

ax.spines['bottom'].set_smart_bounds(True)



# set the spines position

ax.spines['bottom'].set_position(('axes', -0.04))

ax.spines['left'].set_position(('axes', 0.015))



# set background color

ax.set_facecolor('white')



# add margin to y-axis

ax.margins(0.2, 0)



# add title

fig.suptitle('# of Years Experience with Machine Learning', fontsize=18, fontweight='black')



plt.savefig('ml_years.png', dpi=300, bbox_inches='tight')
# Get names of indexes for which title is student.

indexNames = mc[mc['title'] == 'Student' ].index

 

# Delete these row indexes from dataFrame

mcclean = mc.drop(indexNames)

print(len(mc)-len(mcclean))
# Creating Compensation Frequency Table

compensationbins_clean = mcclean['compensationbins'].value_counts(normalize=True)

compensationbins_clean = compensationbins_clean.reset_index()

compensationbins_clean = compensationbins_clean.sort_index(axis=0)

compensationbins_clean = pd.DataFrame(compensationbins_clean)

print(compensationbins_clean)
# Creating Compensation Frequency Graph

# Setting style for bar graphs

%matplotlib inline



# set font

plt.rcParams['font.family'] = 'sans-serif'

plt.rcParams['font.sans-serif'] = 'Helvetica'



# set the style of the axes and the text color

plt.rcParams['axes.edgecolor']='#333F4B'

plt.rcParams['axes.linewidth']=0.8

plt.rcParams['xtick.color']='#333F4B'

plt.rcParams['ytick.color']='#333F4B'

plt.rcParams['text.color']='#333F4B'



# numeric placeholder for the y axis

my_range=list(range(1,len(compensationbins_freq.index)+1))



fig, ax = plt.subplots(figsize=(8,5))



# create for each bin a vertical line that starts at y = 0 with the length 

# represented by the specific percentage.

plt.vlines(x=my_range, ymin=0, ymax=compensationbins_clean['compensationbins'], color='#007ACC', alpha=0.5, linewidth=50)



# create for each bin a dot at the level of the expense percentage value

# plt.plot(my_range, age_freq['age'], "o", markersize=10, color='#007ACC', alpha=0.6)



# set labels

ax.set_xlabel('', fontsize=15, fontweight='black', color = '#333F4B')

ax.set_ylabel('% of Respondents', fontsize=15, fontweight='black', color = '#333F4B')



# set axis

ax.tick_params(axis='both', which='major', labelsize=12)

plt.xticks(my_range, compensationbins_clean['index'])



# add an horizonal label for the y axis 

# fig.text(-0.15, 0.5, '% of Respondants', fontsize=15, fontweight='black', color = '#333F4B')



# change the style of the axis spines

ax.spines['top'].set_color('none')

ax.spines['right'].set_color('none')

ax.spines['left'].set_smart_bounds(True)

ax.spines['bottom'].set_smart_bounds(True)



# set the spines position

ax.spines['bottom'].set_position(('axes', -0.04))

ax.spines['left'].set_position(('axes', 0.015))



# set background color

ax.set_facecolor('white')



# add margin to y-axis

ax.margins(0.2, 0)



# add title

fig.suptitle('Annual Compensation of Survey Respondents', fontsize=18, fontweight='black')



plt.savefig('compensationbins_clean.png', dpi=300, bbox_inches='tight')
# Creating Data Science Team Size Frequency Table

ds_teamsize_clean = mcclean['ds_teamsize'].value_counts(normalize=True)

ds_teamsize_clean = ds_teamsize_clean.reset_index()

ds_teamsize_clean = ds_teamsize_clean.sort_index(axis=0)

ds_teamsize_clean = pd.DataFrame(ds_teamsize_clean)

print(ds_teamsize_clean.head())
# Creating Data Science Team Size Frequency Graph

# Setting style for bar graphs

%matplotlib inline



# set font

plt.rcParams['font.family'] = 'sans-serif'

plt.rcParams['font.sans-serif'] = 'Helvetica'



# set the style of the axes and the text color

plt.rcParams['axes.edgecolor']='#333F4B'

plt.rcParams['axes.linewidth']=0.8

plt.rcParams['xtick.color']='#333F4B'

plt.rcParams['ytick.color']='#333F4B'

plt.rcParams['text.color']='#333F4B'



# numeric placeholder for the y axis

my_range=list(range(1,len(ds_teamsize_freq.index)+1))



fig, ax = plt.subplots(figsize=(10,6))



# create for each bin a vertical line that starts at y = 0 with the length 

# represented by the specific percentage.

plt.vlines(x=my_range, ymin=0, ymax=ds_teamsize_clean['ds_teamsize'], color='#007ACC', alpha=0.5, linewidth=15)



# create for each bin a dot at the level of the expense percentage value

plt.plot(my_range, ds_teamsize_clean['ds_teamsize'], "o", markersize=15, color='#007ACC', alpha=0.9)



# set labels

ax.set_xlabel('# of People', fontsize=15, fontweight='black', color = '#333F4B')

ax.set_ylabel('% of Respondents', fontsize=15, fontweight='black', color = '#333F4B')



# set axis

ax.tick_params(axis='both', which='major', labelsize=12)

plt.xticks(my_range, ds_teamsize_clean['index'], rotation=45)



# add an horizonal label for the y axis 

# fig.text(-0.15, 0.5, '% of Respondants', fontsize=15, fontweight='black', color = '#333F4B')



# change the style of the axis spines

ax.spines['top'].set_color('none')

ax.spines['right'].set_color('none')

ax.spines['left'].set_smart_bounds(True)

ax.spines['bottom'].set_smart_bounds(True)



# set the spines position

ax.spines['bottom'].set_position(('axes', -0.04))

ax.spines['left'].set_position(('axes', 0.015))



# set background color

ax.set_facecolor('white')



# add margin to y-axis

ax.margins(0.05, 0.03)

ax.set_xticklabels(ds_teamsize_freq['index'], rotation=90)



# add ds_teamsize

fig.suptitle('Data Science Team Size of Survey Respondents', fontsize=18, fontweight='black')



plt.savefig('teamsize_freq_clean.png', dpi=300, bbox_inches='tight')
print(mcclean['comp_size'].unique())
# Renaming bins for company size

rename = {'1000-9,999 employees':'1k - 10k','> 10,000 employees':'10k+',

          '0-49 employees':'0 - 50','50-249 employees':'50 - 250',

          '250-999':'250 - 1k'}

mcclean['comp_size'] = mc['comp_size'].replace(rename)
# Creating company size Frequency Table

comp_size = mcclean['comp_size'].value_counts(normalize=True)

comp_size = comp_size.reset_index()

comp_size = comp_size.sort_index(axis=0)

comp_size = pd.DataFrame(comp_size)

print(comp_size.head())
# Creating Company Size  Frequency Graph

# Setting style for bar graphs

%matplotlib inline



# set font

plt.rcParams['font.family'] = 'sans-serif'

plt.rcParams['font.sans-serif'] = 'Helvetica'



# set the style of the axes and the text color

plt.rcParams['axes.edgecolor']='#333F4B'

plt.rcParams['axes.linewidth']=0.8

plt.rcParams['xtick.color']='#333F4B'

plt.rcParams['ytick.color']='#333F4B'

plt.rcParams['text.color']='#333F4B'



# numeric placeholder for the y axis

my_range=list(range(1,len(comp_size.index)+1))



fig, ax = plt.subplots(figsize=(11,5))



# create for each bin a vertical line that starts at y = 0 with the length 

# represented by the specific percentage.

plt.vlines(x=my_range, ymin=0, ymax=comp_size['comp_size'], color='#007ACC', alpha=0.5, linewidth=50)



# create for each bin a dot at the level of the expense percentage value

# plt.plot(my_range, age_freq['age'], "o", markersize=10, color='#007ACC', alpha=0.6)



# set labels

ax.set_xlabel('', fontsize=15, fontweight='black', color = '#333F4B')

ax.set_ylabel('% of Respondents', fontsize=15, fontweight='black', color = '#333F4B')



# set axis

ax.tick_params(axis='both', which='major', labelsize=12)

plt.xticks(my_range, comp_size['index'])



# add an horizonal label for the y axis 

# fig.text(-0.15, 0.5, '% of Respondants', fontsize=15, fontweight='black', color = '#333F4B')



# change the style of the axis spines

ax.spines['top'].set_color('none')

ax.spines['right'].set_color('none')

ax.spines['left'].set_smart_bounds(True)

ax.spines['bottom'].set_smart_bounds(True)



# set the spines position

ax.spines['bottom'].set_position(('axes', -0.04))

ax.spines['left'].set_position(('axes', 0.015))



# set background color

ax.set_facecolor('white')



# add margin to y-axis

ax.margins(0.2, 0)



# add title

fig.suptitle('Size of Company Where Employed', fontsize=18, fontweight='black')



plt.savefig('comp_size_clean.png', dpi=300, bbox_inches='tight')
# Creating Education Frequency Table

education_freq_clean = mcclean['education'].value_counts(normalize=True)

education_freq_clean = education_freq_clean.reset_index()

gender_freq_clean = education_freq_clean.sort_index(axis=0)

education_freq_clean = pd.DataFrame(education_freq_clean)

education_freq_clean.head()
# Creating Education Frequency Graph

# Setting style for bar graphs

%matplotlib inline



# set font

plt.rcParams['font.family'] = 'sans-serif'

plt.rcParams['font.sans-serif'] = 'Helvetica'



# set the style of the axes and the text color

plt.rcParams['axes.edgecolor']='#333F4B'

plt.rcParams['axes.linewidth']=0.8

plt.rcParams['xtick.color']='#333F4B'

plt.rcParams['ytick.color']='#333F4B'

plt.rcParams['text.color']='#333F4B'



# numeric placeholder for the y axis

my_range=list(range(1,len(education_freq_clean.index)+1))



fig, ax = plt.subplots(figsize=(8,5))



# create for each bin a vertical line that starts at y = 0 with the length 

# represented by the specific percentage.

plt.vlines(x=my_range, ymin=0, ymax=education_freq_clean['education'], color='#007ACC', alpha=0.5, linewidth=50)



# create for each bin a dot at the level of the expense percentage value

# plt.plot(my_range, age_freq['age'], "o", markersize=10, color='#007ACC', alpha=0.6)



# set labels

ax.set_xlabel('Education Level', fontsize=15, fontweight='black', color = '#333F4B')

ax.set_ylabel('% of Respondents', fontsize=15, fontweight='black', color = '#333F4B')



# set axis

ax.tick_params(axis='both', which='major', labelsize=12)

plt.xticks(my_range, education_freq_clean['index'])



# add an horizonal label for the y axis 

# fig.text(-0.15, 0.5, '% of Respondants', fontsize=15, fontweight='black', color = '#333F4B')



# change the style of the axis spines

ax.spines['top'].set_color('none')

ax.spines['right'].set_color('none')

ax.spines['left'].set_smart_bounds(True)

ax.spines['bottom'].set_smart_bounds(True)



# set the spines position

ax.spines['bottom'].set_position(('axes', -0.04))

ax.spines['left'].set_position(('axes', 0.015))



# set background color

ax.set_facecolor('white')



# add margin to y-axis

ax.margins(0.2, 0)

ax.set_xticklabels(education_freq['index'], rotation=90)



# add title

fig.suptitle('Highest Education Level of Survey Respondents', fontsize=18, fontweight='black')



plt.savefig('education_freq_clean.png', dpi=300, bbox_inches='tight')