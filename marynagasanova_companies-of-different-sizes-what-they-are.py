# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# This line makes our plot visible.



%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#    for filename in filenames:

#        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# These lines load all 3 Kaggle surveys, which we have in availability.



survey_2017 = pd.read_csv("/kaggle/input/2017-kaggle-survey/2017_responses.csv", encoding='latin-1', low_memory=False)

survey_2018 = pd.read_csv("/kaggle/input/2018-kaggle-survey/2018_responses.csv", header = 1, low_memory=False)

survey_2019 = pd.read_csv("/kaggle/input/kaggle-survey-2019/multiple_choice_responses.csv", header = 1)
# This cell eximines raw amount of responses per company size. We will use it for the future data rescaling.

company_size_raw = survey_2019['What is the size of the company where you are employed?'].value_counts().to_frame('Total amount')



# We will determine and plot amount of respondents per company size in % to make it human-readable.

company_size_perc = survey_2019["What is the size of the company where you are employed?"].value_counts(normalize=True)*100



# Set plotting variables and plot the output.

plt.rcParams.update({'figure.figsize': (7, 5)})

ax = company_size_perc.plot(kind = 'bar')

ax.set_title('Company size')



#company_size_perc.set_xlabel('Amount ($)')

ax.set_ylabel('% of respondents')
# Function for easy column grouping and counting their content.

# This functions takes dataframe; name of the column we interested in; 

# new column name (we will use it for shorter references and readibility); 

# column for indexing and optional parameter col_param, which is a value in an orig_col_name.



def group_by_func(df, orig_col_name, new_col_name, index_col, col_param = False):

    if col_param is not False:

        # If the col_param is set, we will select data from the original dataframe using this parameter.

        # We need it to group a column, which has multiple values in it.

        temp_df = df[df[orig_col_name] == col_param]

        # Group 2 columns and count amount of each pair

        

    else:

        temp_df = df

    grouped_df = temp_df.groupby([orig_col_name, index_col]).size().to_frame(new_col_name).reset_index()

    # Remove unnecessary column for easier further data manipulation and setting the index

    grouped_df = grouped_df.drop(orig_col_name, 1).set_index(index_col)

    return grouped_df
# Function for re-arrenging data from two corresponding columns to a table. 

# This function is used to create a new dataframe with multiple columns from two given columns.

# It takes a column name we're interested in; a column, which we use for grouping (size of companies) 

# and an optional value of scale for re-scaling.



def column_to_table(orig_col_name, index_col_name, scale = False):

    new_dataframe = pd.DataFrame()

    # Make a list of all possible values of the interesting column

    new_col_name_raw = survey_2019[orig_col_name].unique()

    # Leave only significant valuables

    new_col_name = [x for x in new_col_name_raw if str(x) != 'nan']

    # Iterate over each value to make a column in our new dataframe

    

    for i in new_col_name:

        temp_df = group_by_func(survey_2019, orig_col_name, i, index_col_name, i)

        # Parameter scale is set to True if we need relative data instead of absolute one

        

        if scale is True:

            # Divide our amount of each pair per amount of responders in appropriate group

            temp_df = ((temp_df[i]/company_size_raw['Total amount'])*100).to_frame(i)

            

        # Add newly created column to our new dataframe

        new_dataframe = pd.concat([new_dataframe, temp_df], axis = 1, sort = True)

    return new_dataframe
# Create a new dataframe using multiple columns from the original dataframe.

# This function is used to create a new dataframe from multiple columns in the original dataframe

# with the desired columns.

# It takes:

#   - a column name we are interested in;

#   - new column names for easier plotting;

#   - index column, which represents company size

#   - an optional scale value, which it uses for re-scaling results according to amount of respondents

#     in the specific group.



def multiple_columns_to_table(orig_column_name, new_column_name, index_col, scale = False):

    output_df = pd.DataFrame()

    # j parameter is used to iterate over new_column_name list.

    j = 0

    

    for i in orig_column_name:

        # Creating a column of a new dataframe

        temp_df = group_by_func(survey_2019, i, new_column_name[j], index_col)

        # Re-scale our output column to amount of respondents in each group, if scale parameter is set to True.

        

        if scale is True:

            temp_df = ((temp_df[new_column_name[j]]/company_size_raw['Total amount'])*100).to_frame(new_column_name[j])

        j += 1

        # Add a column to our new dataframe

        output_df = pd.concat([output_df, temp_df], axis = 1, sort = True)

        # Return and plot output dataframe

    return output_df.style.background_gradient(cmap='Blues')
# How many employees are responsible for ML at your business in 2019.



column_to_table('Approximately how many individuals are responsible for data science workloads at your place of business?',

                'What is the size of the company where you are employed?'

               ).style.background_gradient(cmap='Blues')
# How many employees are responsible for ML at your business in 2019, re-scaled data for more deep comparison.



column_to_table(

    'Approximately how many individuals are responsible for data science workloads at your place of business?',

    'What is the size of the company where you are employed?',

    True

).style.background_gradient(cmap='Blues')
# Distribution of activities for company 0-49 with the respondents replied that they don't have anyone responsible

# for machine learning in their company.

# Let's get the list of activities from the survey and make more compact output list for easier plotting



activities_2019_survey = [

    'Select any activities that make up an important part of your role at work: (Select all that apply) - Selected Choice - Analyze and understand data to influence product or business decisions',

    'Select any activities that make up an important part of your role at work: (Select all that apply) - Selected Choice - Build and/or run the data infrastructure that my business uses for storing, analyzing, and operationalizing data',

    'Select any activities that make up an important part of your role at work: (Select all that apply) - Selected Choice - Build prototypes to explore applying machine learning to new areas',

    'Select any activities that make up an important part of your role at work: (Select all that apply) - Selected Choice - Build and/or run a machine learning service that operationally improves my product or workflows',

    'Select any activities that make up an important part of your role at work: (Select all that apply) - Selected Choice - Experimentation and iteration to improve existing ML models',

    'Select any activities that make up an important part of your role at work: (Select all that apply) - Selected Choice - Do research that advances the state of the art of machine learning',

    'Select any activities that make up an important part of your role at work: (Select all that apply) - Selected Choice - None of these activities are an important part of my role at work',

    'Select any activities that make up an important part of your role at work: (Select all that apply) - Selected Choice - Other'

]

activities_0_49_employees_output = [

    'Analyze and understand data',

    'Build/run the data infrastructure for storing/analyzing/operationalizing data',

    'Build prototypes to explore applying ML to new areas',

    'Build/run a ML service that operationally improves my product/workflows',

    'Experimentation and iteration to improve existing ML models',

    'Do research that advances the state of the art of ML',

    'None of these',

    'Other'

]

# Select necessary part of the dataframe related to our group of 0-49 employees, who replied

# that they don't have anyone responsible for the machinee learning

temp_df = survey_2019[(survey_2019['What is the size of the company where you are employed?'] == '0-49 employees') & (survey_2019['Approximately how many individuals are responsible for data science workloads at your place of business?'] == '0')]

activities_0_49_employees_list = []



for i in activities_2019_survey:

    # Count and add each activity to the list.

    activities_0_49_employees_list.append(temp_df[i].count())

    # Create a dataframe from the list.

    

activities_0_49_employees = pd.DataFrame(activities_0_49_employees_list, index = activities_0_49_employees_output, columns =['Total amount of respondents'])



# Set plotting parameters and plot the frame.

plt.rcParams.update({'figure.figsize': (10, 7)})

plt.rcParams.update({'font.size': 18})

ax = activities_0_49_employees.plot(kind = 'barh')

ax.set_title("Activities of '0-49 employees' group")
# Size of company vs. activities of the employees 2019.

# List of activities, represented in the survey.



activities_list = [

    'Select any activities that make up an important part of your role at work: (Select all that apply) - Selected Choice - Analyze and understand data to influence product or business decisions',

    'Select any activities that make up an important part of your role at work: (Select all that apply) - Selected Choice - Build and/or run the data infrastructure that my business uses for storing, analyzing, and operationalizing data',

    'Select any activities that make up an important part of your role at work: (Select all that apply) - Selected Choice - Build prototypes to explore applying machine learning to new areas',

    'Select any activities that make up an important part of your role at work: (Select all that apply) - Selected Choice - Build and/or run a machine learning service that operationally improves my product or workflows',

    'Select any activities that make up an important part of your role at work: (Select all that apply) - Selected Choice - Experimentation and iteration to improve existing ML models',

    'Select any activities that make up an important part of your role at work: (Select all that apply) - Selected Choice - Do research that advances the state of the art of machine learning',

    'Select any activities that make up an important part of your role at work: (Select all that apply) - Selected Choice - None of these activities are an important part of my role at work',

    'Select any activities that make up an important part of your role at work: (Select all that apply) - Selected Choice - Other'

]



# Desired output column names for more readable format.

activities_list_output = [

    'Analyze and understand data to influence product or business decisions',

    'Build and/or run the data infrastructure that my business uses for storing, analyzing, and operationalizing data',

    'Build prototypes to explore applying machine learning to new areas',

    'Build and/or run a machine learning service that operationally improves my product or workflows',

    'Experimentation and iteration to improve existing ML models',

    'Do research that advances the state of the art of machine learning',

    'None of these activities are an important part of my role at work',

    'Other'

]



multiple_columns_to_table(

    activities_list,

    activities_list_output,

    'What is the size of the company where you are employed?'

)
# Size of company vs. activities of the employees normilized.



activities_list = [

    'Select any activities that make up an important part of your role at work: (Select all that apply) - Selected Choice - Analyze and understand data to influence product or business decisions',

    'Select any activities that make up an important part of your role at work: (Select all that apply) - Selected Choice - Build and/or run the data infrastructure that my business uses for storing, analyzing, and operationalizing data',

    'Select any activities that make up an important part of your role at work: (Select all that apply) - Selected Choice - Build prototypes to explore applying machine learning to new areas',

    'Select any activities that make up an important part of your role at work: (Select all that apply) - Selected Choice - Build and/or run a machine learning service that operationally improves my product or workflows',

    'Select any activities that make up an important part of your role at work: (Select all that apply) - Selected Choice - Experimentation and iteration to improve existing ML models',

    'Select any activities that make up an important part of your role at work: (Select all that apply) - Selected Choice - Do research that advances the state of the art of machine learning',

    'Select any activities that make up an important part of your role at work: (Select all that apply) - Selected Choice - None of these activities are an important part of my role at work',

    'Select any activities that make up an important part of your role at work: (Select all that apply) - Selected Choice - Other'

]



activities_list_output = [

    'Analyze and understand data to influence product or business decisions',

    'Build and/or run the data infrastructure that my business uses for storing, analyzing, and operationalizing data',

    'Build prototypes to explore applying machine learning to new areas',

    'Build and/or run a machine learning service that operationally improves my product or workflows',

    'Experimentation and iteration to improve existing ML models',

    'Do research that advances the state of the art of machine learning',

    'None of these activities are an important part of my role at work',

    'Other'

]



multiple_columns_to_table(

    activities_list,

    activities_list_output,

    'What is the size of the company where you are employed?',

    True

)
# Activities in total 2019 vs. 2018 years scaled according to the amount of respondents in each year.

# List of activities.



survey_2018_activities = ['Select any activities that make up an important part of your role at work: (Select all that apply) - Selected Choice - Analyze and understand data to influence product or business decisions', 'Select any activities that make up an important part of your role at work: (Select all that apply) - Selected Choice - Build and/or run a machine learning service that operationally improves my product or workflows', 'Select any activities that make up an important part of your role at work: (Select all that apply) - Selected Choice - Build and/or run the data infrastructure that my business uses for storing, analyzing, and operationalizing data', 'Select any activities that make up an important part of your role at work: (Select all that apply) - Selected Choice - Build prototypes to explore applying machine learning to new areas', 'Select any activities that make up an important part of your role at work: (Select all that apply) - Selected Choice - Do research that advances the state of the art of machine learning', 'Select any activities that make up an important part of your role at work: (Select all that apply) - Selected Choice - None of these activities are an important part of my role at work', 'Select any activities that make up an important part of your role at work: (Select all that apply) - Selected Choice - Other']

survey_2019_activities = ['Select any activities that make up an important part of your role at work: (Select all that apply) - Selected Choice - Analyze and understand data to influence product or business decisions', 'Select any activities that make up an important part of your role at work: (Select all that apply) - Selected Choice - Build and/or run the data infrastructure that my business uses for storing, analyzing, and operationalizing data', 'Select any activities that make up an important part of your role at work: (Select all that apply) - Selected Choice - Build prototypes to explore applying machine learning to new areas', 'Select any activities that make up an important part of your role at work: (Select all that apply) - Selected Choice - Build and/or run a machine learning service that operationally improves my product or workflows', 'Select any activities that make up an important part of your role at work: (Select all that apply) - Selected Choice - Experimentation and iteration to improve existing ML models', 'Select any activities that make up an important part of your role at work: (Select all that apply) - Selected Choice - Do research that advances the state of the art of machine learning', 'Select any activities that make up an important part of your role at work: (Select all that apply) - Selected Choice - None of these activities are an important part of my role at work', 'Select any activities that make up an important part of your role at work: (Select all that apply) - Selected Choice - Other']



# Desired activities name for more readable format.

survey_2018_activities_output = [

    'Analyze and understand data',

    'Build/run a ML service that operationally improves my product/workflows',

    'Build/run the data infrastructure for storing/analyzing/operationalizing data',

    'Build prototypes to explore applying ML to new areas',

    'Do research that advances the state of the art of ML',

    'None of these',

    'Other'

]



survey_2019_activities_output = [

    'Analyze and understand data',

    'Build/run the data infrastructure for storing/analyzing/operationalizing data',

    'Build prototypes to explore applying ML to new areas',

    'Build/run a ML service that operationally improves my product/workflows',

    'Experimentation and iteration to improve existing ML models',

    'Do research that advances the state of the art of ML',

    'None of these',

    'Other'

]



survey_2018_activities_list = []

for i in survey_2018_activities:

    # Count, re-scale and add each activity to the list.

    survey_2018_activities_list.append((survey_2018[i].count()/23859)*100)

    # Create a dataframe from the list.

survey_2018_activities_df = pd.DataFrame(

    survey_2018_activities_list,

    index = survey_2018_activities_output,

    columns =['2018']

) 



survey_2019_activities_list = []



for i in survey_2019_activities:

    # Count, re-scale and add each activity to the list.

    survey_2019_activities_list.append((survey_2019[i].count()/19717)*100)

    # Create a dataframe from the list.

survey_2019_activities_df = pd.DataFrame(survey_2019_activities_list, index = survey_2019_activities_output, columns =['2019'])

# Combine dataframes for both years.



survey_activities_2019_vs_2018 = pd.concat([survey_2018_activities_df, survey_2019_activities_df], axis = 1, sort = True)



# Set plotting parameters and plot the frame.

plt.rcParams.update({'figure.figsize': (10, 7)})

plt.rcParams.update({'font.size': 18})

ax = survey_activities_2019_vs_2018.plot(kind = 'barh')

ax.set_title('The most important activities 2018 vs. 2019')

ax.set_xlabel('% of respondents')
# Distribution of the 'Experimentation and iteration to improve existing ML models' activity across the company types, raw data.



data_science_degree = group_by_func(

    survey_2019,

    'Select any activities that make up an important part of your role at work: (Select all that apply) - Selected Choice - Experimentation and iteration to improve existing ML models',

    'Experimentation and iteration to improve existing ML models',

    'What is the size of the company where you are employed?'

)



# Set plotting parameters and plot the dataframe.

plt.rcParams.update({'figure.figsize': (10, 5)})

plt.rcParams.update({'font.size': 13})

data_science_degree.plot(kind = 'bar')
# Distribution of the 'Experimentation and iteration to improve existing ML models' activity across the company types, re-scaled to the amount of respondents in each group.



data_science_degree = group_by_func(

    survey_2019,

    'Select any activities that make up an important part of your role at work: (Select all that apply) - Selected Choice - Experimentation and iteration to improve existing ML models',

    'Experimentation and iteration to improve existing ML models',

    'What is the size of the company where you are employed?'

)



data_science_degree_percent = (data_science_degree['Experimentation and iteration to improve existing ML models']/company_size_raw['Total amount'])*100



# Set plotting parameters and plot the dataframe.

plt.rcParams.update({'figure.figsize': (7, 5)})

plt.rcParams.update({'font.size': 13})

ax = data_science_degree_percent.plot(kind = 'bar')

ax.set_title('Experimentation and iteration to improve existing ML models')
# Distribution of machine learning methods across the company types, raw amount.



column_to_table(

    'Does your current employer incorporate machine learning methods into their business?',

    'What is the size of the company where you are employed?'

).style.background_gradient(cmap='Blues')
# Distribution of machine learning methods across the company types, re-scaled amount for comparizon.



column_to_table(

    'Does your current employer incorporate machine learning methods into their business?',

    'What is the size of the company where you are employed?',

    True

).style.background_gradient(cmap='Blues')
# Distribution of job titles across the company sizes.



column_to_table(

    'What is the size of the company where you are employed?',

    'Select the title most similar to your current role (or most recent title if retired): - Selected Choice'

).T.style.background_gradient(cmap='Blues')
# Comparizon of job title of respondents in the past 3 surveys, amount re-scaled to the amount of respondents

# in each appropriate group.

# Select job title from each survey, count amount of each group and re-scale according to the amount of respondents

# in each group.



job_title_2017 = ((survey_2017['CurrentJobTitleSelect'].value_counts()/16000)*100).to_frame('2017')

job_title_2019 = ((survey_2019['Select the title most similar to your current role (or most recent title if retired): - Selected Choice'].value_counts()/19717)*100).to_frame('2019')

job_title_2018 = ((survey_2018['Select the title most similar to your current role (or most recent title if retired): - Selected Choice'].value_counts()/23859)*100).to_frame('2018')



# Create a new dataframe by combining 3 dataframes from each year.

new_df = pd.concat([job_title_2017, job_title_2018, job_title_2019], axis = 1, sort = True)



# Set plotting parameters and plot the dataframe.

plt.rcParams.update({'figure.figsize': (10, 20)})

plt.rcParams.update({'font.size': 13})

ax = new_df.plot(kind = 'barh', width = 1.0)

ax.grid()

ax.set_title('Comparison of respondents job titles in 2017, 2018, 2019 years')

ax.set_xlabel('% of respondents in each year')
# Distribution of the 'Product/Project Manager' job title across types of companies, re-scaled according to

# the amount of respondents in each group. 

# Group job title and company size columns using the 'Product/Project Manager' as a key.



product_manager_vs_company_size = group_by_func(

    survey_2019,

    'Select the title most similar to your current role (or most recent title if retired): - Selected Choice',

    'Product/Project Manager',

    'What is the size of the company where you are employed?',

    'Product/Project Manager'

)



# Re-scale the amount of the specific job title holder according to the amount of respondents in each group.

product_manager_vs_company_size_percent = (product_manager_vs_company_size['Product/Project Manager']/company_size_raw['Total amount'])*100



# Set plotting parameters and plot the dataframe.

plt.rcParams.update({'figure.figsize': (7, 5)})

plt.rcParams.update({'font.size': 13})

ax = product_manager_vs_company_size_percent.plot(kind = 'bar', x = 'What is the size of the company where you are employed?', y = '% of responders')

ax.set_title('Product/Project Manager job title distribution')

ax.set_ylabel('% of respondents')
# Distribution of the activities for respondents, who replied 'yes' to job title 'Product/Project Manager'

# Let's get the list of activities from the survey and make more compact output list for easier plotting



activities_2019_survey = ['Select any activities that make up an important part of your role at work: (Select all that apply) - Selected Choice - Analyze and understand data to influence product or business decisions', 

                          'Select any activities that make up an important part of your role at work: (Select all that apply) - Selected Choice - Build and/or run the data infrastructure that my business uses for storing, analyzing, and operationalizing data',

                          'Select any activities that make up an important part of your role at work: (Select all that apply) - Selected Choice - Build prototypes to explore applying machine learning to new areas', 

                          'Select any activities that make up an important part of your role at work: (Select all that apply) - Selected Choice - Build and/or run a machine learning service that operationally improves my product or workflows',

                          'Select any activities that make up an important part of your role at work: (Select all that apply) - Selected Choice - Experimentation and iteration to improve existing ML models', 

                          'Select any activities that make up an important part of your role at work: (Select all that apply) - Selected Choice - Do research that advances the state of the art of machine learning', 

                          'Select any activities that make up an important part of your role at work: (Select all that apply) - Selected Choice - None of these activities are an important part of my role at work', 

                          'Select any activities that make up an important part of your role at work: (Select all that apply) - Selected Choice - Other']



activities_2019_survey_output = ['Analyze and understand data', 

                                 'Build/run the data infrastructure for storing/analyzing/operationalizing data', 

                                 'Build prototypes to explore applying ML to new areas', 

                                 'Build/run a ML service that operationally improves my product/workflows', 

                                 'Experimentation and iteration to improve existing ML models', 

                                 'Do research that advances the state of the art of ML', 

                                 'None of these', 

                                 'Other']



# Select necessary part of the dataframe related to 'Product/Project Manager' job title

temp_df = survey_2019[(survey_2019['Select the title most similar to your current role (or most recent title if retired): - Selected Choice'] == 'Product/Project Manager')]

project_manager_list = []



for i in activities_2019_survey:

    # Count and add each activity to the list.

    project_manager_list.append(temp_df[i].count())

    # Create a dataframe from the list.



project_manager_activities = pd.DataFrame(

    project_manager_list,

    index = activities_2019_survey_output,

    columns =['Count']

)



# Set plotting parameters and plot the frame.

plt.rcParams.update({'figure.figsize': (10, 7)})

plt.rcParams.update({'font.size': 18})

ax = project_manager_activities.plot(kind = 'barh')

ax.set_title('Product/Project Manager activities distribution')

ax.set_xlabel('Amount of respondents')
# Learning platform usage across company sizes, re-scaled according to amount of respondents in

# the appropriate group.

# List of the platforms.



learning_platform = ['On which platforms have you begun or completed data science courses? (Select all that apply) - Selected Choice - Udacity',

                     'On which platforms have you begun or completed data science courses? (Select all that apply) - Selected Choice - Coursera',

                     'On which platforms have you begun or completed data science courses? (Select all that apply) - Selected Choice - edX',

                     'On which platforms have you begun or completed data science courses? (Select all that apply) - Selected Choice - DataCamp', 

                     'On which platforms have you begun or completed data science courses? (Select all that apply) - Selected Choice - DataQuest',

                     'On which platforms have you begun or completed data science courses? (Select all that apply) - Selected Choice - Kaggle Courses (i.e. Kaggle Learn)', 

                     'On which platforms have you begun or completed data science courses? (Select all that apply) - Selected Choice - Fast.ai', 

                     'On which platforms have you begun or completed data science courses? (Select all that apply) - Selected Choice - Udemy', 

                     'On which platforms have you begun or completed data science courses? (Select all that apply) - Selected Choice - LinkedIn Learning', 

                     'On which platforms have you begun or completed data science courses? (Select all that apply) - Selected Choice - University Courses (resulting in a university degree)', 

                     'On which platforms have you begun or completed data science courses? (Select all that apply) - Selected Choice - None', 

                     'On which platforms have you begun or completed data science courses? (Select all that apply) - Selected Choice - Other']



# List of the output platforms for more readable format.

learning_platform_output = ['Udacity', 

                            'Coursera', 

                            'edX', 

                            'DataCamp', 

                            'DataQuest', 

                            'Kaggle Courses', 

                            'Fast.ai', 

                            'Udemy', 

                            'LinkedIn Learning', 

                            'University Courses', 

                            'None', 

                            'Other']



multiple_columns_to_table(

    learning_platform,

    learning_platform_output,

    'What is the size of the company where you are employed?',

    True

)
# Data source usage across company_size, re-scaled according to the amount of respondents in the appropriate group.

# List of the datasources.



data_source = ['Who/what are your favorite media sources that report on data science topics? (Select all that apply) - Selected Choice - Twitter (data science influencers)',

               'Who/what are your favorite media sources that report on data science topics? (Select all that apply) - Selected Choice - Hacker News (https://news.ycombinator.com/)', 

               'Who/what are your favorite media sources that report on data science topics? (Select all that apply) - Selected Choice - Reddit (r/machinelearning, r/datascience, etc)', 

               'Who/what are your favorite media sources that report on data science topics? (Select all that apply) - Selected Choice - Kaggle (forums, blog, social media, etc)', 

               'Who/what are your favorite media sources that report on data science topics? (Select all that apply) - Selected Choice - Course Forums (forums.fast.ai, etc)', 

               'Who/what are your favorite media sources that report on data science topics? (Select all that apply) - Selected Choice - YouTube (Cloud AI Adventures, Siraj Raval, etc)', 

               'Who/what are your favorite media sources that report on data science topics? (Select all that apply) - Selected Choice - Podcasts (Chai Time Data Science, Linear Digressions, etc)', 

               'Who/what are your favorite media sources that report on data science topics? (Select all that apply) - Selected Choice - Blogs (Towards Data Science, Medium, Analytics Vidhya, KDnuggets etc)', 

               'Who/what are your favorite media sources that report on data science topics? (Select all that apply) - Selected Choice - Journal Publications (traditional publications, preprint journals, etc)', 

               'Who/what are your favorite media sources that report on data science topics? (Select all that apply) - Selected Choice - Slack Communities (ods.ai, kagglenoobs, etc)', 

               'Who/what are your favorite media sources that report on data science topics? (Select all that apply) - Selected Choice - None']



# List of the output datasources for more readable format.

data_source_output = ['Twitter', 

                      'Hacker News', 

                      'Reddit', 

                      'Kaggle', 

                      'Course Forums', 

                      'YouTube', 

                      'Podcasts', 

                      'Blogs', 

                      'Journal Publications', 

                      'Slack Communities', 

                      'None']



multiple_columns_to_table(

    data_source,

    data_source_output,

    'What is the size of the company where you are employed?',

    True

)
# Distribution of the coding experience across the company types, re-scaled data for comparison.



column_to_table(

    'How long have you been writing code to analyze data (at work or at school)?',

    'What is the size of the company where you are employed?',

    True

).style.background_gradient(cmap='Blues')