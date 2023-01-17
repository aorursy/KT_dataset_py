import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

plt.style.use('ggplot')

import seaborn as sns

from IPython.display import display, Image

%matplotlib inline
df_schema = pd.read_csv('../input/schema.csv')

display(df_schema.head())
df_conversion_rates = pd.read_csv('../input/conversionRates.csv')

display(df_conversion_rates.head(10))
df_multi_choice = pd.read_csv('../input/multipleChoiceResponses.csv',encoding='latin-1', low_memory=False)

df_multi_choice.head()
print(df_multi_choice.shape)
popular_country = list(df_multi_choice['Country'].value_counts().head(5).index)

print(popular_country)
color = sns.color_palette()

# get value counts of EmploymentStatus column

employment_status = df_multi_choice.EmploymentStatus.value_counts()

# delare plt figure for plotting

plt.figure(figsize=(10, 8))

# seaborn barplot 

sns.barplot(y=employment_status.index, 

            x=employment_status.values,

            color=color[1])

# add a suptitle

plt.suptitle("Employment Status in Data science and Machine Learning Industries", fontsize=14)

# add xlabel

plt.xlabel('Employment Status Count', fontsize=12)

# add ylabel

plt.ylabel('Employment Status', fontsize=12)

# finally show the plot

plt.show()
# get data of popular country

df_employment_popular_country = df_multi_choice[df_multi_choice.Country.isin(popular_country)]

# declare plt figure for plotting

plt.figure(figsize=(10, 8))

plt.title('Employment Status in different country', fontsize=14)

# plot countplot

sns.countplot(x="Country", hue="EmploymentStatus", data=df_employment_popular_country)

plt.ylabel('Employment Status Count')

plt.show()
# get the value_counts of CurrentJobTitleSelect column

df_job_title = df_multi_choice[df_multi_choice.CurrentJobTitleSelect.notnull()]["CurrentJobTitleSelect"].value_counts()

plt.figure(figsize=(8, 6))

plt.title('Popular Job')

sns.barplot(y=df_job_title.index, x= df_job_title)

plt.xlabel('Current Job Title Count')

plt.show()
df_career_switcher = df_multi_choice[df_multi_choice.CareerSwitcher.notnull()]

df_career_switcher = df_career_switcher.loc[df_career_switcher['CareerSwitcher'] == 'Yes']

plt.figure(figsize=(10, 6))

plt.title('Career switcher in different Job field', fontsize=14)

sns.countplot(y="CurrentJobTitleSelect", hue="CareerSwitcher", data=df_career_switcher)

plt.xticks(rotation=90)

plt.xlabel('Career switcher Count')

plt.ylabel('Current Job Title')

plt.show()
df_remote_work = df_multi_choice['RemoteWork'].value_counts()

plt.figure(figsize=(8, 6))

plt.title('Data Scientist work remotely', fontsize=14)

sns.barplot(y=df_remote_work.index, x= df_remote_work)

plt.xlabel('')

plt.show()
df_learning_popular_country = df_multi_choice[df_multi_choice.Country.isin(popular_country)]

df_learning_popular_country = df_learning_popular_country[df_learning_popular_country.LearningDataScience.notnull()]

plt.figure(figsize=(13, 8))

plt.title('Focused on learning Data Science in different country', fontsize=14)

sns.countplot(x="Country", hue="LearningDataScience", data=df_learning_popular_country)

plt.ylabel('Learning Data science')

plt.show()
# learning platform select

df_learning_platform = df_multi_choice[df_multi_choice.LearningPlatformSelect.notnull()]['LearningPlatformSelect']

df_learning_platform = df_learning_platform.astype('str').apply(lambda x: x.split(','), 1)

flat_list = [item for sublist in df_learning_platform for item in sublist]

d = pd.DataFrame(flat_list)[0].value_counts()



plt.figure(figsize=(10, 6))

plt.title('Popular learning platform', fontsize=14)

sns.barplot(x=d, y=d.index)

plt.xlabel('Learning Platform Popularity')

plt.ylabel('Learning Platform Name')

plt.show()
filter_col = [col for col in df_multi_choice if col.startswith('LearningPlatformUsefulness')]

df_learning_platform_usefulness = df_multi_choice[filter_col]

df_learning_platform_usefulness = df_learning_platform_usefulness.rename(columns=lambda x: x.replace('LearningPlatformUsefulness', ''))

plt.figure(figsize=(12, 10))

plt.title('Usefulness of Different learning platform', fontsize=14)

sns.countplot(x="variable", hue="value", data=pd.melt(df_learning_platform_usefulness))

plt.xticks(rotation=90)

plt.xlabel('Learning Platform Name')

plt.ylabel('Learning Platform Usefulness')

plt.show()
df_popular_bpn = df_multi_choice[df_multi_choice.BlogsPodcastsNewslettersSelect.notnull()]['BlogsPodcastsNewslettersSelect']

df_popular_bpn = df_popular_bpn.astype('str').apply(lambda x: x.split(','), 1)

flat_list = [item for sublist in df_popular_bpn for item in sublist]

d = pd.DataFrame(flat_list)[0].value_counts()



plt.figure(figsize=(8, 6))

plt.suptitle('Popular Blogs/Podcasts/Newsletters', fontsize=14)

sns.barplot(x=d, y=d.index)

plt.xlabel('')

plt.ylabel('Blogs/Podcasts/Newsletters Name')

plt.show()
df_uni_imp = df_multi_choice['UniversityImportance'].value_counts()

plt.figure(figsize=(8, 6))

plt.title("University Importance", fontsize=14)

sns.barplot(y=df_uni_imp.index, x= df_uni_imp)

plt.xlabel('')

plt.show()
# get the 'JobHuntTime' column from our data without null values

df_hunt_time = df_multi_choice[df_multi_choice.JobHuntTime.notnull()]["JobHuntTime"]

# let's count each unique time and their occurrences. 

df_hunt_time = df_hunt_time.value_counts()

# let's build a matplotlib figure for plotting

plt.figure(figsize=(8, 6))

plt.title("Time spent per week for job hunting")

# bar plot using seaborn barplot function given y the title and x to value

sns.barplot(y=df_hunt_time.index, x = df_hunt_time)

# rotate the x axis label to 90 degrees 

plt.ylabel('Job Hunting Time Frame')

plt.xlabel('Number of occurrences of each time frame')

plt.xticks(rotation=90)



# finally show the bar plot

plt.show()
con_df = pd.DataFrame(df_multi_choice['Country'].value_counts())

top_country = con_df.head(5).index

df_top_country = df_multi_choice.loc[df_multi_choice['Country'].isin(top_country)]



plt.figure(figsize=(8, 6))

plt.title('Time spent per week for job hunting in different country', fontsize=14)

sns.countplot(x='JobHuntTime', hue='Country', data=df_top_country)

plt.xlabel('Job hunting time frame')

plt.ylabel('Popular time frame in different country')

plt.show()
# get WorkChallengesSelect column without null values

df_working_challenges = df_multi_choice[df_multi_choice.WorkChallengesSelect.notnull()]['WorkChallengesSelect']

# split at ","

df_working_challenges = df_working_challenges.astype('str').apply(lambda x: x.split(','), 1)

# keep only the unique item

flat_list = [item for sublist in df_working_challenges for item in sublist]

# value counts

d = pd.DataFrame(flat_list)[0].value_counts()

# declare the figure

plt.figure(figsize=(10, 8))

plt.title("Challenges faced during work", fontsize=14)

# plot the graph

sns.barplot(x=d, y=d.index)

plt.ylabel('Challenges')

plt.xlabel('')

plt.show()
# work challenges frequency

filter_col = [col for col in df_multi_choice if col.startswith('WorkChallengeFrequency')]

df_work_challenge_frequency = df_multi_choice[filter_col]

df_work_challenge_frequency = df_work_challenge_frequency.rename(columns=lambda x: x.replace('WorkChallengeFrequency', ''))

plt.figure(figsize=(12, 10))

plt.title('Frequency of different challenges at work')

sns.countplot(x="variable", hue="value", data=pd.melt(df_work_challenge_frequency))

plt.ylabel('Frequency')

plt.xlabel('Daily work challenges')

plt.xticks(rotation=90)

plt.show()
# replace CompensationCurrency column value with actual exchangeRate of df_conversion_rates dataframe

df_multi_choice['CompensationCurrency'] = df_multi_choice['CompensationCurrency'].map(df_conversion_rates.set_index('originCountry')['exchangeRate'])

# drop row where df_multi_choice['CompensationCurrency'] column is null

df_compension = df_multi_choice.dropna(axis=0, how='all', subset=['CompensationCurrency'])

# some people put "-" sign before their salary, so we drop it

df_compension = df_compension[df_compension['CompensationAmount'].str.contains("-")==False]

# replace "," with ""

df_compension['CompensationAmount'] = df_compension['CompensationAmount'].str.replace(',', '')

# change df_multi_choice['CompensationAmount'] column datatype to float

df_compension['CompensationAmount'] = df_compension['CompensationAmount'].astype(float)



# multiply CompensationAmount column and CompensationCurrency column and create a new column to hold that value names "SalaryAmount"

df_compension['SalaryAmount'] = df_compension['CompensationAmount']*df_compension['CompensationCurrency']

# convert SalaryAmount column to 'int' datatype

df_compension['SalaryAmount'] = df_compension['SalaryAmount'].astype(int)
# create salary range for plotting

bins = [0, 1000,2500, 5000, 10000, 50000, 100000, 150000, 200000, 250000, 300000]

df_salary_amount_value_counts = pd.cut(df_compension['SalaryAmount'], bins).value_counts()

# let's build a matplotlib figure for plotting

plt.figure(figsize=(8, 6))

plt.title("Salary of Data Scientists", fontsize=14)

# bar plot using seaborn barplot function given y the title and x to value

sns.barplot(y=df_salary_amount_value_counts.index, x = df_salary_amount_value_counts)

# rotate the x axis label to 90 degrees 

plt.ylabel('Salary range in dollar')

plt.xlabel('Number of occurrences of each salary range')

plt.xticks(rotation=90)



# finally show the bar plot

plt.show()
# get salary corresponding to gender and keep only Female and Male

df_gender_salary = df_compension['SalaryAmount'].groupby(df_compension['GenderSelect']).mean()[['Female', 'Male']]

# pyplot figure for plotting

plt.figure(figsize=(8, 8))

plt.title("Gender diversity in Salary")

# plot the diversity

df_gender_salary.plot(kind='bar', color=[['darkorange', 'red']])

plt.xlabel('Gender Name')

plt.ylabel('Salary')

plt.show()
# get salary corresponding to country and keep popular country

df_country_salary = df_compension['SalaryAmount'].groupby(df_compension['Country']).mean()[['United States', 'India', 'Other', 'Russia', "People 's Republic of China", 'Brazil', 'Germany', 'France', 'Canada']]

plt.figure(figsize=(8, 6))

plt.title("Country diversity in salary")

# plot the diversity

df_country_salary.plot(kind='bar', color=[['r', 'g', 'blue', 'purple', 'hotpink', 'orange', 'chocolate', 'skyblue', 'tomato']])

plt.xlabel('Country Name')

plt.ylabel('Salary')

plt.show()
df_dataset = df_multi_choice[df_multi_choice.PublicDatasetsSelect.notnull()]['PublicDatasetsSelect']

df_dataset = df_dataset.astype('str').apply(lambda x: x.split(','), 1)

flat_list = [item for sublist in df_dataset for item in sublist]

d = pd.DataFrame(flat_list)[0].value_counts()

plt.figure(figsize=(6, 6))

plt.title('Popular places to find public datasets to practice data science skills', fontsize=14)

sns.barplot(x=d, y=d.index)

plt.xlabel('')

plt.show()
# what is most popular mltoolsnextyear

df_ml_tools_next_year = df_multi_choice[df_multi_choice.MLToolNextYearSelect

                               .notnull()]["MLToolNextYearSelect"].value_counts().head(20)

plt.figure(figsize=(8, 6))

plt.title('Popular ML tools in next year', fontsize=14)

sns.barplot(y=df_ml_tools_next_year.index, x= df_ml_tools_next_year)

plt.ylabel('ML tools name')

plt.xlabel('ML tools popularity')

plt.show()
df_ml_tools_next_year = df_multi_choice[df_multi_choice.MLToolNextYearSelect

                               .notnull()]

df_ml_tools_next_yearrr = df_ml_tools_next_year['MLToolNextYearSelect'].value_counts().head(5).index

pop = df_ml_tools_next_yearrr.get_values().tolist()

# pop = ', '.join("'{0}'".format(w) for w in pop)



df_ml_tools_next_year = df_ml_tools_next_year[df_ml_tools_next_year['MLToolNextYearSelect'].isin(pop)]

plt.figure(figsize=(10, 6))

plt.title('Next year popular ML tools in different job field', fontsize=14)

sns.countplot(x="CurrentJobTitleSelect", hue='MLToolNextYearSelect', data=df_ml_tools_next_year)

plt.xticks(rotation=90)

plt.xlabel('Job field name')

plt.ylabel('Popularity of ML Tools')

plt.show()
# ml method next year

df_ml_method_next_year = df_multi_choice['MLMethodNextYearSelect'].value_counts()

plt.figure(figsize=(8, 6))

plt.title('Popular ML method in next year', fontsize=14)

sns.barplot(y=df_ml_method_next_year.index, x= df_ml_method_next_year)

plt.ylabel('ML methods name')

plt.xlabel('ML methods popularity')

plt.show()
filter_col = [col for col in df_multi_choice if col.startswith('JobFactor')]

df_factor = df_multi_choice[filter_col]

df_factor = df_factor.rename(columns=lambda x: x.replace('JobFactor', ''))

plt.figure(figsize=(12, 10))

plt.title('Importance of different job factor', fontsize=14)

sns.countplot(x="variable", hue="value", data=pd.melt(df_factor))

plt.xlabel('JobFactor')

plt.ylabel('Job Factor Importance')

plt.xticks(rotation=90)

plt.show()
df_job_find_resource = df_multi_choice['JobSearchResource'].value_counts()

plt.figure(figsize=(8, 6))

plt.title('Popular platform for finding job', fontsize=14)

sns.barplot(y=df_job_find_resource.index, x= df_job_find_resource)

plt.ylabel('Platform name')

plt.xlabel('Platform popularity')

plt.show()
df_job_satisfaction = df_multi_choice['JobSatisfaction'].value_counts()

plt.figure(figsize=(8, 6))

plt.title('Job satisfaction', fontsize=14)

sns.barplot(y=df_job_satisfaction.index, x= df_job_satisfaction)

plt.xlabel('')

plt.show()
df_salary_change = df_multi_choice['SalaryChange'].value_counts()

plt.figure(figsize=(7, 6))

plt.title('Salary changed in the past 3 years', fontsize=14)

sns.barplot(y=df_salary_change.index, x= df_salary_change)

plt.show()
df_work_tool_code_sharing = df_multi_choice[df_multi_choice.WorkCodeSharing.notnull()]['WorkCodeSharing']

df_work_tool_code_sharing = df_work_tool_code_sharing.astype('str').apply(lambda x: x.split(','), 1)

flat_list = [item for sublist in df_work_tool_code_sharing for item in sublist]

df_work_tool_code_sharing = pd.DataFrame(flat_list)[0].value_counts()

plt.figure(figsize=(8, 6))

plt.title('Popular tools for sharing code', fontsize=14)

sns.barplot(x=df_work_tool_code_sharing, y=df_work_tool_code_sharing.index)

plt.ylabel('Tools name')

plt.xlabel('Tools popularity')

plt.show()
df_work_tool_data_sourcing = df_multi_choice[df_multi_choice.WorkDataSourcing.notnull()]['WorkDataSourcing']

df_work_tool_data_sourcing = df_work_tool_data_sourcing.astype('str').apply(lambda x: x.split(','), 1)

flat_list = [item for sublist in df_work_tool_data_sourcing for item in sublist]

df_work_tool_data_sourcing = pd.DataFrame(flat_list)[0].value_counts().head(10)

plt.figure(figsize=(8, 6))

plt.title('Popular tools for sharing source data', fontsize=14)

sns.barplot(x=df_work_tool_data_sourcing, y=df_work_tool_data_sourcing.index)

plt.ylabel('Tools name')

plt.xlabel('Tools popularity')

plt.show()
df_work_data_storage = df_multi_choice[df_multi_choice.WorkDataStorage.notnull()]['WorkDataStorage']

df_work_data_storage = df_work_data_storage.astype('str').apply(lambda x: x.replace('),', ')//').split('//'), 1)

flat_list = [item for sublist in df_work_data_storage for item in sublist]

df_work_data_storage = pd.DataFrame(flat_list)[0].value_counts()

plt.figure(figsize=(7, 6))

plt.title('Popular data storage models', fontsize=14)

sns.barplot(x=df_work_data_storage, y=df_work_data_storage.index)

plt.ylabel('Data storage models name')

plt.xlabel('Data storage models popularity')

plt.show()
df_ds_team_sit = df_multi_choice['WorkMLTeamSeatSelect'].value_counts()

plt.figure(figsize=(8, 6))

plt.title('Data scientist sit within the organization', fontsize=14)

sns.barplot(y=df_ds_team_sit.index, x= df_ds_team_sit)

plt.xlabel('')

plt.show()
df_internal_external_tools = df_multi_choice['WorkInternalVsExternalTools'].value_counts()

plt.figure(figsize=(8, 6))

plt.title('Internal vs external resources used in data science projects', fontsize=14)

sns.barplot(y=df_internal_external_tools.index, x= df_internal_external_tools)

plt.xlabel('')

plt.show()
df_work_data_vis = df_multi_choice['WorkDataVisualizations'].value_counts()

plt.figure(figsize=(8, 6))

plt.title('Percentage of data visualization in analytics projects', fontsize=14)

sns.barplot(y=df_work_data_vis.index, x= df_work_data_vis)

plt.xlabel('')

plt.show()
filter_col = [col for col in df_multi_choice if col.startswith('Time')]

df_time_in_project = df_multi_choice[filter_col]

df_time_in_project = df_time_in_project.rename(columns=lambda x: x.replace('Time', ''))

plt.figure(figsize=(12, 10))

plt.title('Percentage of time in different task in a project', fontsize=14)

df_time_in_project.mean().plot(kind='bar', color=[['red', 'green', 'blue', 'purple', 'hotpink', 'orange']])

plt.xlabel('Task name')

plt.show()
df_work_ml_model = df_multi_choice[df_multi_choice.WorkMethodsSelect.notnull()]['WorkMethodsSelect']

df_work_ml_model = df_work_ml_model.astype('str').apply(lambda x: x.split(','), 1)

flat_list = [item for sublist in df_work_ml_model for item in sublist]

df_work_ml_model = pd.DataFrame(flat_list)[0].value_counts().head(10)

plt.figure(figsize=(8, 6))

plt.title('Popular work method', fontsize=14)

sns.barplot(x=df_work_ml_model, y=df_work_ml_model.index)

plt.xlabel('')

plt.show()
df_work_tool_tech = df_multi_choice[df_multi_choice.WorkToolsSelect.notnull()]['WorkToolsSelect']

df_work_tool_tech = df_work_tool_tech.astype('str').apply(lambda x: x.split(','), 1)

flat_list = [item for sublist in df_work_tool_tech for item in sublist]

df_work_tool_tech = pd.DataFrame(flat_list)[0].value_counts().head(10)

plt.figure(figsize=(8, 6))

plt.title('Popular analytics tools, technologies and languages', fontsize=14)

sns.barplot(x=df_work_tool_tech, y=df_work_tool_tech.index)

plt.xlabel('')

plt.show()
df_work_algorithom = df_multi_choice[df_multi_choice.WorkAlgorithmsSelect.notnull()]['WorkAlgorithmsSelect']

df_work_algorithom = df_work_algorithom.astype('str').apply(lambda x: x.split(','), 1)

flat_list = [item for sublist in df_work_algorithom for item in sublist]

df_work_algorithom = pd.DataFrame(flat_list)[0].value_counts()

plt.figure(figsize=(8, 6))

plt.suptitle('Popular analytic methods')

sns.barplot(x=df_work_algorithom, y=df_work_algorithom.index)

plt.xlabel('')

plt.show()
df_data_size = df_multi_choice['WorkDatasetSize'].value_counts()

plt.figure(figsize=(8, 6))

plt.suptitle("Typical datasets size for training model")

sns.barplot(y=df_data_size.index, x= df_data_size)

plt.xlabel('')

plt.show()
df_work_production_frequ = df_multi_choice['WorkProductionFrequency'].value_counts()

plt.figure(figsize=(8, 6))

plt.title("Frequency of model get put into production at work", fontsize=14)

sns.barplot(y=df_work_production_frequ.index, x= df_work_production_frequ)

plt.show()
df_work_data_type = df_multi_choice[df_multi_choice.WorkDataTypeSelect.notnull()]['WorkDataTypeSelect']

df_work_data_type = df_work_data_type.astype('str').apply(lambda x: x.split(','), 1)

flat_list = [item for sublist in df_work_data_type for item in sublist]

df_work_data_type = pd.DataFrame(flat_list)[0].value_counts()

plt.figure(figsize=(8, 6))

plt.title('Popular data type', fontsize=14)

sns.barplot(x=df_work_data_type, y=df_work_data_type.index)

plt.xlabel('')

plt.show()
df_work_hardware = df_multi_choice[df_multi_choice.WorkHardwareSelect.notnull()]['WorkHardwareSelect']

df_work_hardware = df_work_hardware.astype('str').apply(lambda x: x.replace('),', ')//').split('//'), 1)

flat_list = [item for sublist in df_work_hardware for item in sublist]

df_work_hardware = pd.DataFrame(flat_list)[0].value_counts().head(15)

plt.figure(figsize=(6, 8))

plt.suptitle('Hardware used by Data Scientist')

sns.barplot(x=df_work_hardware, y=df_work_hardware.index)

plt.xlabel('')

plt.show()
df_work_role = df_multi_choice[df_multi_choice.JobFunctionSelect.notnull()]['JobFunctionSelect']

df_work_role = df_work_role.astype('str').apply(lambda x: x.split(','), 1)

flat_list = [item for sublist in df_work_role for item in sublist]

df_work_role = pd.DataFrame(flat_list)[0].value_counts()

plt.figure(figsize=(8, 6))

plt.title('Primary role in job', fontsize=14)

sns.barplot(x=df_work_role, y=df_work_role.index)

plt.xlabel('')

plt.show()
df_algo_under_label = df_multi_choice['AlgorithmUnderstandingLevel'].value_counts()

plt.figure(figsize=(8, 6))

plt.title('Understanding of mathematics behind the algorithom', fontsize=14)

sns.barplot(y=df_algo_under_label.index, x= df_algo_under_label)

plt.xlabel('')

plt.show()
# language recommendation

df_language = df_multi_choice['LanguageRecommendationSelect'].value_counts()

plt.figure(figsize=(8, 6))

plt.title('Popular language', fontsize=14)

sns.barplot(y=df_language.index, x= df_language)

plt.xlabel('Language popularity')

plt.show()
# language recommendation by profession

df_language = df_multi_choice[df_multi_choice.LanguageRecommendationSelect

                               .notnull()]

df_ml_tools_next_yearrr = df_ml_tools_next_year['LanguageRecommendationSelect'].value_counts().head(3).index

pop = df_ml_tools_next_yearrr.get_values().tolist()

# pop = ', '.join("'{0}'".format(w) for w in pop)



df_language = df_language[df_language['LanguageRecommendationSelect'].isin(pop)]

plt.figure(figsize=(10, 6))

plt.title('Popular language in different profession', fontsize=14)

sns.countplot(x="CurrentJobTitleSelect", hue='LanguageRecommendationSelect', data=df_language)

plt.xticks(rotation=90)

plt.xlabel('Job title')

plt.ylabel('Language popularity')

plt.show()