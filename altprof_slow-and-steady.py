import numpy as np # almost everything

import pandas as pd # working with tables

import seaborn as sns # plotting library

%matplotlib inline

from matplotlib import patches

import matplotlib.pyplot as plt # importing pyplot

from scipy import stats # importing some stats that will be useful later on



import warnings

warnings.filterwarnings('ignore') # ignoring all warnings
sns.set_style('whitegrid') # white grid background on plots

sns.set_palette('colorblind', 10) # color palette for colorblinds with 10 colors

current_palette = sns.color_palette() # saving this palette in separate variable

sns.set_context('notebook') # setting notebook context
data = pd.read_csv('../input/kaggle-survey-2019/multiple_choice_responses.csv', low_memory=False).drop(0); # loading data



india = data[data['Q3'] == 'India'].copy()

usa = data[data['Q3'] == 'United States of America'].copy()

brazil = data[data['Q3'] == 'Brazil'].copy()

japan = data[data['Q3'] == 'Japan'].copy()

russia = data[data['Q3'] == 'Russia'].copy()
print('Number of respondents: ', data.shape[0])
salary = data['Q10'].str.replace(',','').str.replace('$','').str.replace('>','').str.strip(' ').str.split('-').apply(pd.Series).astype(np.float32) # getting numeric values of salary

salary.columns = ['min', 'max'] # renaming columns

salary['midrange'] = (salary['max'] + salary['min']) / 2 # calculating mid-range: (max + min) / 2
loglogtime = np.log(np.log(data['Time from Start to Finish (seconds)'].astype(np.int32)))



india_time = loglogtime.loc[india.index]

usa_time = loglogtime.loc[usa.index]

brazil_time = loglogtime.loc[brazil.index]

japan_time = loglogtime.loc[japan.index]

russia_time = loglogtime.loc[russia.index]



q1_all = np.percentile(loglogtime, 25)

q3_all = np.percentile(loglogtime, 75)



q1_india = np.percentile(india_time, 25)

q3_india = np.percentile(india_time, 75)



q1_usa = np.percentile(usa_time, 25)

q3_usa = np.percentile(usa_time, 75)



q1_brazil = np.percentile(brazil_time, 25)

q3_brazil = np.percentile(brazil_time, 75)



q1_japan = np.percentile(japan_time, 25)

q3_japan = np.percentile(japan_time, 75)



q1_russia = np.percentile(russia_time, 25)

q3_russia = np.percentile(russia_time, 75)



fast_all = data.loc[(loglogtime < q1_all)]

slow_all = data.loc[(loglogtime > q3_all)]



fast_india = india.loc[(india_time < q1_india)]

slow_india = india.loc[(india_time > q3_india)]



fast_usa = usa.loc[(usa_time < q1_usa)]

slow_usa = usa.loc[(usa_time > q3_usa)]



fast_brazil = brazil.loc[(brazil_time < q1_brazil)]

slow_brazil = brazil.loc[(brazil_time > q3_brazil)]



fast_japan = japan.loc[(japan_time < q1_japan)]

slow_japan = japan.loc[(japan_time > q3_japan)]



fast_russia = russia.loc[(russia_time < q1_russia)]

slow_russia = russia.loc[(russia_time > q3_russia)]



info_time = pd.DataFrame()

info_time['Countries'] = ['All', 'India', 'USA', 'Brazil', 'Japan', 'Russia']

info_time['Size'] = [fast_all.shape[0], fast_india.shape[0], fast_usa.shape[0], fast_brazil.shape[0], fast_japan.shape[0], fast_russia.shape[0]]

info_time['Median duration for slow respondents, min'] = [int(np.round(slow_all['Time from Start to Finish (seconds)'].astype(np.int32).median() / 60)),

                                                          int(np.round(slow_india['Time from Start to Finish (seconds)'].astype(np.int32).median() / 60)), 

                                                          int(np.round(slow_usa['Time from Start to Finish (seconds)'].astype(np.int32).median() / 60)), 

                                                          int(np.round(slow_brazil['Time from Start to Finish (seconds)'].astype(np.int32).median() / 60)), 

                                                          int(np.round(slow_japan['Time from Start to Finish (seconds)'].astype(np.int32).median() / 60)), 

                                                          int(np.round(slow_russia['Time from Start to Finish (seconds)'].astype(np.int32).median() / 60))]

info_time['Median duration for quick respondents, min'] = [int(np.round(fast_all['Time from Start to Finish (seconds)'].astype(np.int32).median() / 60)),

                                                           int(np.round(fast_india['Time from Start to Finish (seconds)'].astype(np.int32).median() / 60)), 

                                                           int(np.round(fast_usa['Time from Start to Finish (seconds)'].astype(np.int32).median() / 60)), 

                                                           int(np.round(fast_brazil['Time from Start to Finish (seconds)'].astype(np.int32).median() / 60)), 

                                                           int(np.round(fast_japan['Time from Start to Finish (seconds)'].astype(np.int32).median() / 60)), 

                                                           int(np.round(fast_russia['Time from Start to Finish (seconds)'].astype(np.int32).median() / 60))]



info_time
age_order = ['18-21', '22-24', '25-29', '30-34', '35-39', '40-44', '45-49', '50-54', '55-59', '60-69', '70+']
fig, axs = plt.subplots(2, 3, figsize=(16,10));

fig.suptitle('Age comparison', x=0.5, y=1.05)



axs[0][0].set_title('All')

slow_all['Q1'].value_counts(normalize=True).loc[age_order].plot(kind='bar', color=current_palette[0], alpha=0.8, ax=axs[0][0], rot=30);

fast_all['Q1'].value_counts(normalize=True).loc[age_order].plot(kind='bar', color=current_palette[1], alpha=0.8, ax=axs[0][0], rot=30);



axs[0][1].set_title('India')

slow_india['Q1'].value_counts(normalize=True).loc[age_order].plot(kind='bar', color=current_palette[0], alpha=0.8, ax=axs[0][1], rot=30);

fast_india['Q1'].value_counts(normalize=True).loc[age_order].plot(kind='bar', color=current_palette[1], alpha=0.8, ax=axs[0][1], rot=30);



axs[0][2].set_title('USA')

slow_usa['Q1'].value_counts(normalize=True).loc[age_order].plot(kind='bar', color=current_palette[0], alpha=0.8, ax=axs[0][2], rot=30);

fast_usa['Q1'].value_counts(normalize=True).loc[age_order].plot(kind='bar', color=current_palette[1], alpha=0.8, ax=axs[0][2], rot=30);



axs[1][0].set_title('Brazil')

slow_brazil['Q1'].value_counts(normalize=True).loc[age_order].plot(kind='bar', color=current_palette[0], alpha=0.8, ax=axs[1][0], rot=30);

fast_brazil['Q1'].value_counts(normalize=True).loc[age_order].plot(kind='bar', color=current_palette[1], alpha=0.8, ax=axs[1][0], rot=30);



axs[1][1].set_title('Japan')

slow_japan['Q1'].value_counts(normalize=True).loc[age_order].plot(kind='bar', color=current_palette[0], alpha=0.8, ax=axs[1][1], rot=30);

fast_japan['Q1'].value_counts(normalize=True).loc[age_order].plot(kind='bar', color=current_palette[1], alpha=0.8, ax=axs[1][1], rot=30);



axs[1][2].set_title('Russia')

slow_russia['Q1'].value_counts(normalize=True).loc[age_order].plot(kind='bar', color=current_palette[0], alpha=0.8, ax=axs[1][2], rot=30);

fast_russia['Q1'].value_counts(normalize=True).loc[age_order].plot(kind='bar', color=current_palette[1], alpha=0.8, ax=axs[1][2], rot=30);



fig.legend([patches.Patch(facecolor=current_palette[color]) for color in range(2)], ['Slow respondents', 'Quick respondents'], loc='upper right');



fig.tight_layout();
fig, axs = plt.subplots(2, 3, figsize=(16,10));

fig.suptitle('Gender comparison', x=0.5, y=1.05)



axs[0][0].set_title('All')

slow_all['Q2'].value_counts(normalize=True).plot(kind='bar', color=current_palette[0], alpha=0.8, ax=axs[0][0], rot=30);

fast_all['Q2'].value_counts(normalize=True).plot(kind='bar', color=current_palette[1], alpha=0.8, ax=axs[0][0], rot=30);



axs[0][1].set_title('India')

slow_india['Q2'].value_counts(normalize=True).plot(kind='bar', color=current_palette[0], alpha=0.8, ax=axs[0][1], rot=30);

fast_india['Q2'].value_counts(normalize=True).plot(kind='bar', color=current_palette[1], alpha=0.8, ax=axs[0][1], rot=30);



axs[0][2].set_title('USA')

slow_usa['Q2'].value_counts(normalize=True).plot(kind='bar', color=current_palette[0], alpha=0.8, ax=axs[0][2], rot=30);

fast_usa['Q2'].value_counts(normalize=True).plot(kind='bar', color=current_palette[1], alpha=0.8, ax=axs[0][2], rot=30);



axs[1][0].set_title('Brazil')

slow_brazil['Q2'].value_counts(normalize=True).plot(kind='bar', color=current_palette[0], alpha=0.8, ax=axs[1][0], rot=30);

fast_brazil['Q2'].value_counts(normalize=True).plot(kind='bar', color=current_palette[1], alpha=0.8, ax=axs[1][0], rot=30);



axs[1][1].set_title('Japan')

slow_japan['Q2'].value_counts(normalize=True).plot(kind='bar', color=current_palette[0], alpha=0.8, ax=axs[1][1], rot=30);

fast_japan['Q2'].value_counts(normalize=True).plot(kind='bar', color=current_palette[1], alpha=0.8, ax=axs[1][1], rot=30);



axs[1][2].set_title('Russia')

slow_russia['Q2'].value_counts(normalize=True).plot(kind='bar', color=current_palette[0], alpha=0.8, ax=axs[1][2], rot=30);

fast_russia['Q2'].value_counts(normalize=True).plot(kind='bar', color=current_palette[1], alpha=0.8, ax=axs[1][2], rot=30);



fig.legend([patches.Patch(facecolor=current_palette[color]) for color in range(2)], ['Slow respondents', 'Quick respondents'], loc='upper right');



fig.tight_layout();
education_dict = {'Master’s degree': 'Master', 

                  'Bachelor’s degree': 'Bachelor', 

                  'Doctoral degree': 'Doctor', 

                  'Some college/university study without earning a bachelor’s degree': 'Audition',

                  'No formal education past high school': 'High School',

                  'I prefer not to answer': 'Other',

                  'Professional degree': 'Professional'}



education_order = ['High School', 'Professional', 'Audition', 'Bachelor', 'Master', 'Doctor']



fig, axs = plt.subplots(2, 3, figsize=(16,10));

fig.suptitle('Education comparison', x=0.5, y=1.05)



axs[0][0].set_title('All')

slow_all['Q4'].map(education_dict).value_counts(normalize=True).loc[education_order].plot(kind='bar', color=current_palette[0], alpha=0.8, ax=axs[0][0], rot=30);

fast_all['Q4'].map(education_dict).value_counts(normalize=True).loc[education_order].plot(kind='bar', color=current_palette[1], alpha=0.8, ax=axs[0][0], rot=30);



axs[0][1].set_title('India')

slow_india['Q4'].map(education_dict).value_counts(normalize=True).loc[education_order].plot(kind='bar', color=current_palette[0], alpha=0.8, ax=axs[0][1], rot=30);

fast_india['Q4'].map(education_dict).value_counts(normalize=True).loc[education_order].plot(kind='bar', color=current_palette[1], alpha=0.8, ax=axs[0][1], rot=30);



axs[0][2].set_title('USA')

slow_usa['Q4'].map(education_dict).value_counts(normalize=True).loc[education_order].plot(kind='bar', color=current_palette[0], alpha=0.8, ax=axs[0][2], rot=30);

fast_usa['Q4'].map(education_dict).value_counts(normalize=True).loc[education_order].plot(kind='bar', color=current_palette[1], alpha=0.8, ax=axs[0][2], rot=30);



axs[1][0].set_title('Brazil')

slow_brazil['Q4'].map(education_dict).value_counts(normalize=True).loc[education_order].plot(kind='bar', color=current_palette[0], alpha=0.8, ax=axs[1][0], rot=30);

fast_brazil['Q4'].map(education_dict).value_counts(normalize=True).loc[education_order].plot(kind='bar', color=current_palette[1], alpha=0.8, ax=axs[1][0], rot=30);



axs[1][1].set_title('Japan')

slow_japan['Q4'].map(education_dict).value_counts(normalize=True).loc[education_order].plot(kind='bar', color=current_palette[0], alpha=0.8, ax=axs[1][1], rot=30);

fast_japan['Q4'].map(education_dict).value_counts(normalize=True).loc[education_order].plot(kind='bar', color=current_palette[1], alpha=0.8, ax=axs[1][1], rot=30);



axs[1][2].set_title('Russia')

slow_russia['Q4'].map(education_dict).value_counts(normalize=True).loc[education_order].plot(kind='bar', color=current_palette[0], alpha=0.8, ax=axs[1][2], rot=30);

fast_russia['Q4'].map(education_dict).value_counts(normalize=True).loc[education_order].plot(kind='bar', color=current_palette[1], alpha=0.8, ax=axs[1][2], rot=30);



fig.legend([patches.Patch(facecolor=current_palette[color]) for color in range(2)], ['Slow respondents', 'Quick respondents'], loc='upper right');



fig.tight_layout();
size_order = ['0-49 employees', '50-249 employees', '250-999 employees', '1000-9,999 employees', '> 10,000 employees']



fig, axs = plt.subplots(2, 3, figsize=(16,10));

fig.suptitle('Company size charts', x=0.5, y=1.05)



axs[0][0].set_title('All')

slow_all['Q6'].value_counts(normalize=True).loc[size_order].plot(kind='bar', color=current_palette[0], alpha=0.8, ax=axs[0][0], rot=30);

fast_all['Q6'].value_counts(normalize=True).loc[size_order].plot(kind='bar', color=current_palette[1], alpha=0.8, ax=axs[0][0], rot=30);



axs[0][1].set_title('India')

slow_india['Q6'].value_counts(normalize=True).loc[size_order].plot(kind='bar', color=current_palette[0], alpha=0.8, ax=axs[0][1], rot=30);

fast_india['Q6'].value_counts(normalize=True).loc[size_order].plot(kind='bar', color=current_palette[1], alpha=0.8, ax=axs[0][1], rot=30);



axs[0][2].set_title('USA')

slow_usa['Q6'].value_counts(normalize=True).loc[size_order].plot(kind='bar', color=current_palette[0], alpha=0.8, ax=axs[0][2], rot=30);

fast_usa['Q6'].value_counts(normalize=True).loc[size_order].plot(kind='bar', color=current_palette[1], alpha=0.8, ax=axs[0][2], rot=30);



axs[1][0].set_title('Brazil')

slow_brazil['Q6'].value_counts(normalize=True).loc[size_order].plot(kind='bar', color=current_palette[0], alpha=0.8, ax=axs[1][0], rot=30);

fast_brazil['Q6'].value_counts(normalize=True).loc[size_order].plot(kind='bar', color=current_palette[1], alpha=0.8, ax=axs[1][0], rot=30);



axs[1][1].set_title('Japan')

slow_japan['Q6'].value_counts(normalize=True).loc[size_order].plot(kind='bar', color=current_palette[0], alpha=0.8, ax=axs[1][1], rot=30);

fast_japan['Q6'].value_counts(normalize=True).loc[size_order].plot(kind='bar', color=current_palette[1], alpha=0.8, ax=axs[1][1], rot=30);



axs[1][2].set_title('Russia')

slow_russia['Q6'].value_counts(normalize=True).loc[size_order].plot(kind='bar', color=current_palette[0], alpha=0.8, ax=axs[1][2], rot=30);

fast_russia['Q6'].value_counts(normalize=True).loc[size_order].plot(kind='bar', color=current_palette[1], alpha=0.8, ax=axs[1][2], rot=30);



fig.legend([patches.Patch(facecolor=current_palette[color]) for color in range(2)], ['Slow respondents', 'Quick respondents'], loc='upper right');



fig.tight_layout();
ds_num_order = ['0', '1-2', '3-4', '5-9', '10-14', '15-19', '20+']



fig, axs = plt.subplots(2, 3, figsize=(16,10));

fig.suptitle('Number of data scientists in a company', x=0.5, y=1.05)



axs[0][0].set_title('All')

slow_all['Q7'].value_counts(normalize=True).loc[ds_num_order].plot(kind='bar', color=current_palette[0], alpha=0.8, ax=axs[0][0], rot=30);

fast_all['Q7'].value_counts(normalize=True).loc[ds_num_order].plot(kind='bar', color=current_palette[1], alpha=0.8, ax=axs[0][0], rot=30);



axs[0][1].set_title('India')

slow_india['Q7'].value_counts(normalize=True).loc[ds_num_order].plot(kind='bar', color=current_palette[0], alpha=0.8, ax=axs[0][1], rot=30);

fast_india['Q7'].value_counts(normalize=True).loc[ds_num_order].plot(kind='bar', color=current_palette[1], alpha=0.8, ax=axs[0][1], rot=30);



axs[0][2].set_title('USA')

slow_usa['Q7'].value_counts(normalize=True).loc[ds_num_order].plot(kind='bar', color=current_palette[0], alpha=0.8, ax=axs[0][2], rot=30);

fast_usa['Q7'].value_counts(normalize=True).loc[ds_num_order].plot(kind='bar', color=current_palette[1], alpha=0.8, ax=axs[0][2], rot=30);



axs[1][0].set_title('Brazil')

slow_brazil['Q7'].value_counts(normalize=True).loc[ds_num_order].plot(kind='bar', color=current_palette[0], alpha=0.8, ax=axs[1][0], rot=30);

fast_brazil['Q7'].value_counts(normalize=True).loc[ds_num_order].plot(kind='bar', color=current_palette[1], alpha=0.8, ax=axs[1][0], rot=30);



axs[1][1].set_title('Japan')

slow_japan['Q7'].value_counts(normalize=True).loc[ds_num_order].plot(kind='bar', color=current_palette[0], alpha=0.8, ax=axs[1][1], rot=30);

fast_japan['Q7'].value_counts(normalize=True).loc[ds_num_order].plot(kind='bar', color=current_palette[1], alpha=0.8, ax=axs[1][1], rot=30);



axs[1][2].set_title('Russia')

slow_russia['Q7'].value_counts(normalize=True).loc[ds_num_order].plot(kind='bar', color=current_palette[0], alpha=0.8, ax=axs[1][2], rot=30);

fast_russia['Q7'].value_counts(normalize=True).loc[ds_num_order].plot(kind='bar', color=current_palette[1], alpha=0.8, ax=axs[1][2], rot=30);



fig.legend([patches.Patch(facecolor=current_palette[color]) for color in range(2)], ['Slow respondents', 'Quick respondents'], loc='upper right');



fig.tight_layout();
ml_use_order = ['I do not know', 

                'No (we do not use ML methods)',

                'We are exploring ML methods (and may one day put a model into production)',

                'We use ML methods for generating insights (but do not put working models into production)',

                'We recently started using ML methods (i.e., models in production for less than 2 years)',

                'We have well established ML methods (i.e., models in production for more than 2 years)']



fig, axs = plt.subplots(1, 6, figsize=(20,10), sharey=True);

fig.suptitle('Intensivity of ML use in company', x=0.5, y=1.05)



axs[0].set_title('All')

slow_all['Q8'].value_counts(normalize=True).loc[ml_use_order].plot(kind='barh', color=current_palette[0], alpha=0.8, ax=axs[0]);

fast_all['Q8'].value_counts(normalize=True).loc[ml_use_order].plot(kind='barh', color=current_palette[1], alpha=0.8, ax=axs[0]);



axs[1].set_title('India')

slow_india['Q8'].value_counts(normalize=True).loc[ml_use_order].plot(kind='barh', color=current_palette[0], alpha=0.8, ax=axs[1]);

fast_india['Q8'].value_counts(normalize=True).loc[ml_use_order].plot(kind='barh', color=current_palette[1], alpha=0.8, ax=axs[1]);



axs[2].set_title('USA')

slow_usa['Q8'].value_counts(normalize=True).loc[ml_use_order].plot(kind='barh', color=current_palette[0], alpha=0.8, ax=axs[2]);

fast_usa['Q8'].value_counts(normalize=True).loc[ml_use_order].plot(kind='barh', color=current_palette[1], alpha=0.8, ax=axs[2]);



axs[3].set_title('Brazil')

slow_brazil['Q8'].value_counts(normalize=True).loc[ml_use_order].plot(kind='barh', color=current_palette[0], alpha=0.8, ax=axs[3]);

fast_brazil['Q8'].value_counts(normalize=True).loc[ml_use_order].plot(kind='barh', color=current_palette[1], alpha=0.8, ax=axs[3]);



axs[4].set_title('Japan')

slow_japan['Q8'].value_counts(normalize=True).loc[ml_use_order].plot(kind='barh', color=current_palette[0], alpha=0.8, ax=axs[4]);

fast_japan['Q8'].value_counts(normalize=True).loc[ml_use_order].plot(kind='barh', color=current_palette[1], alpha=0.8, ax=axs[4]);



axs[5].set_title('Russia')

slow_russia['Q8'].value_counts(normalize=True).loc[ml_use_order].plot(kind='barh', color=current_palette[0], alpha=0.8, ax=axs[5]);

fast_russia['Q8'].value_counts(normalize=True).loc[ml_use_order].plot(kind='barh', color=current_palette[1], alpha=0.8, ax=axs[5]);



fig.legend([patches.Patch(facecolor=current_palette[color]) for color in range(2)], ['Slow respondents', 'Quick respondents'], loc='upper right');



fig.tight_layout();
fig, axs = plt.subplots(2, 3, figsize=(20,10));

fig.suptitle('Distribution of mid-range salary', x=0.5, y=1.05)



axs[0][0].set_title('All')

salary.loc[slow_all.index]['midrange'].plot(kind='kde', bw_method='silverman', ax=axs[0][0]);

salary.loc[fast_all.index]['midrange'].plot(kind='kde', bw_method='silverman', ax=axs[0][0]);



axs[0][1].set_title('India')

salary.loc[slow_india.index]['midrange'].plot(kind='kde', bw_method='silverman', ax=axs[0][1]);

salary.loc[fast_india.index]['midrange'].plot(kind='kde', bw_method='silverman', ax=axs[0][1]);



axs[0][2].set_title('USA')

salary.loc[slow_usa.index]['midrange'].plot(kind='kde', bw_method='silverman', ax=axs[0][2]);

salary.loc[fast_usa.index]['midrange'].plot(kind='kde', bw_method='silverman', ax=axs[0][2]);



axs[1][0].set_title('Brazil')

salary.loc[slow_brazil.index]['midrange'].plot(kind='kde', bw_method='silverman', ax=axs[1][0]);

salary.loc[fast_brazil.index]['midrange'].plot(kind='kde', bw_method='silverman', ax=axs[1][0]);



axs[1][1].set_title('Japan')

salary.loc[slow_japan.index]['midrange'].plot(kind='kde', bw_method='silverman', ax=axs[1][1]);

salary.loc[fast_japan.index]['midrange'].plot(kind='kde', bw_method='silverman', ax=axs[1][1]);



axs[1][2].set_title('Russia')

salary.loc[slow_russia.index]['midrange'].plot(kind='kde', bw_method='silverman', ax=axs[1][2]);

salary.loc[fast_russia.index]['midrange'].plot(kind='kde', bw_method='silverman', ax=axs[1][2]);



fig.legend([patches.Patch(facecolor=current_palette[color]) for color in range(2)], ['Slow respondents', 'Quick respondents'], loc='upper right');



fig.tight_layout();
slow_median_all = salary.loc[slow_all.index]['midrange'].median()

fast_median_all = salary.loc[fast_all.index]['midrange'].median()

slow_mean_all = salary.loc[slow_all.index]['midrange'].mean()

fast_mean_all = salary.loc[fast_all.index]['midrange'].mean()



slow_median_india = salary.loc[slow_india.index]['midrange'].median()

fast_median_india = salary.loc[fast_india.index]['midrange'].median()

slow_mean_india = salary.loc[slow_india.index]['midrange'].mean()

fast_mean_india = salary.loc[fast_india.index]['midrange'].mean()



slow_median_usa = salary.loc[slow_usa.index]['midrange'].median()

fast_median_usa = salary.loc[fast_usa.index]['midrange'].median()

slow_mean_usa = salary.loc[slow_usa.index]['midrange'].mean()

fast_mean_usa = salary.loc[fast_usa.index]['midrange'].mean()



slow_median_brazil = salary.loc[slow_brazil.index]['midrange'].median()

fast_median_brazil = salary.loc[fast_brazil.index]['midrange'].median()

slow_mean_brazil = salary.loc[slow_brazil.index]['midrange'].mean()

fast_mean_brazil = salary.loc[fast_brazil.index]['midrange'].mean()



slow_median_japan = salary.loc[slow_japan.index]['midrange'].median()

fast_median_japan = salary.loc[slow_brazil.index]['midrange'].median()

slow_mean_japan = salary.loc[slow_japan.index]['midrange'].mean()

fast_mean_japan = salary.loc[slow_brazil.index]['midrange'].mean()



slow_median_russia = salary.loc[slow_russia.index]['midrange'].median()

fast_median_russia = salary.loc[fast_russia.index]['midrange'].median()

slow_mean_russia = salary.loc[slow_russia.index]['midrange'].mean()

fast_mean_russia = salary.loc[fast_russia.index]['midrange'].mean()



info_time['Mean slow salary - Mean quick salary'] = [slow_mean_all-fast_mean_all, 

                                                     slow_mean_india-fast_mean_india, 

                                                     slow_mean_usa-fast_mean_usa, 

                                                     slow_mean_brazil-fast_mean_brazil, 

                                                     slow_mean_japan-fast_mean_japan,

                                                     slow_mean_russia-fast_mean_russia]



info_time['Median slow salary - Median quick salary'] = [slow_median_all-fast_median_all, 

                                                         slow_median_india-fast_median_india, 

                                                         slow_median_usa-fast_median_usa, 

                                                         slow_median_brazil-fast_median_brazil, 

                                                         slow_median_japan-fast_median_japan,

                                                         slow_median_russia-fast_median_russia]



info_time
cc_spent_order = ['$0 (USD)', '$1-$99', '$100-$999', '$1000-$9,999', '$10,000-$99,999', '> $100,000 ($USD)']



fig, axs = plt.subplots(2, 3, figsize=(16,10));

fig.suptitle('Number of data scientists in a company', x=0.5, y=1.05)



axs[0][0].set_title('All')

slow_all['Q11'].value_counts(normalize=True).loc[cc_spent_order].plot(kind='bar', color=current_palette[0], alpha=0.8, ax=axs[0][0], rot=30);

fast_all['Q11'].value_counts(normalize=True).loc[cc_spent_order].plot(kind='bar', color=current_palette[1], alpha=0.8, ax=axs[0][0], rot=30);



axs[0][1].set_title('India')

slow_india['Q11'].value_counts(normalize=True).loc[cc_spent_order].plot(kind='bar', color=current_palette[0], alpha=0.8, ax=axs[0][1], rot=30);

fast_india['Q11'].value_counts(normalize=True).loc[cc_spent_order].plot(kind='bar', color=current_palette[1], alpha=0.8, ax=axs[0][1], rot=30);



axs[0][2].set_title('USA')

slow_usa['Q11'].value_counts(normalize=True).loc[cc_spent_order].plot(kind='bar', color=current_palette[0], alpha=0.8, ax=axs[0][2], rot=30);

fast_usa['Q11'].value_counts(normalize=True).loc[cc_spent_order].plot(kind='bar', color=current_palette[1], alpha=0.8, ax=axs[0][2], rot=30);



axs[1][0].set_title('Brazil')

slow_brazil['Q11'].value_counts(normalize=True).loc[cc_spent_order].plot(kind='bar', color=current_palette[0], alpha=0.8, ax=axs[1][0], rot=30);

fast_brazil['Q11'].value_counts(normalize=True).loc[cc_spent_order].plot(kind='bar', color=current_palette[1], alpha=0.8, ax=axs[1][0], rot=30);



axs[1][1].set_title('Japan')

slow_japan['Q11'].value_counts(normalize=True).loc[cc_spent_order].plot(kind='bar', color=current_palette[0], alpha=0.8, ax=axs[1][1], rot=30);

fast_japan['Q11'].value_counts(normalize=True).loc[cc_spent_order].plot(kind='bar', color=current_palette[1], alpha=0.8, ax=axs[1][1], rot=30);



axs[1][2].set_title('Russia')

slow_russia['Q11'].value_counts(normalize=True).loc[cc_spent_order].plot(kind='bar', color=current_palette[0], alpha=0.8, ax=axs[1][2], rot=30);

fast_russia['Q11'].value_counts(normalize=True).loc[cc_spent_order].plot(kind='bar', color=current_palette[1], alpha=0.8, ax=axs[1][2], rot=30);



fig.legend([patches.Patch(facecolor=current_palette[color]) for color in range(2)], ['Slow respondents', 'Quick respondents'], loc='upper right');



fig.tight_layout();
code_exp_order = ['I have never written code', '< 1 years', '1-2 years', '3-5 years', '5-10 years', '10-20 years', '20+ years']



fig, axs = plt.subplots(1, 6, figsize=(20,10), sharey=True);

fig.suptitle('Coding experience', x=0.5, y=1.05)



axs[0].set_title('All')

slow_all['Q15'].value_counts(normalize=True).loc[code_exp_order].plot(kind='barh', color=current_palette[0], alpha=0.8, ax=axs[0]);

fast_all['Q15'].value_counts(normalize=True).loc[code_exp_order].plot(kind='barh', color=current_palette[1], alpha=0.8, ax=axs[0]);



axs[1].set_title('India')

slow_india['Q15'].value_counts(normalize=True).loc[code_exp_order].plot(kind='barh', color=current_palette[0], alpha=0.8, ax=axs[1]);

fast_india['Q15'].value_counts(normalize=True).loc[code_exp_order].plot(kind='barh', color=current_palette[1], alpha=0.8, ax=axs[1]);



axs[2].set_title('USA')

slow_usa['Q15'].value_counts(normalize=True).loc[code_exp_order].plot(kind='barh', color=current_palette[0], alpha=0.8, ax=axs[2]);

fast_usa['Q15'].value_counts(normalize=True).loc[code_exp_order].plot(kind='barh', color=current_palette[1], alpha=0.8, ax=axs[2]);



axs[3].set_title('Brazil')

slow_brazil['Q15'].value_counts(normalize=True).loc[code_exp_order].plot(kind='barh', color=current_palette[0], alpha=0.8, ax=axs[3]);

fast_brazil['Q15'].value_counts(normalize=True).loc[code_exp_order].plot(kind='barh', color=current_palette[1], alpha=0.8, ax=axs[3]);



axs[4].set_title('Japan')

slow_japan['Q15'].value_counts(normalize=True).loc[code_exp_order].plot(kind='barh', color=current_palette[0], alpha=0.8, ax=axs[4]);

fast_japan['Q15'].value_counts(normalize=True).loc[code_exp_order].plot(kind='barh', color=current_palette[1], alpha=0.8, ax=axs[4]);



axs[5].set_title('Russia')

slow_russia['Q15'].value_counts(normalize=True).loc[code_exp_order].plot(kind='barh', color=current_palette[0], alpha=0.8, ax=axs[5]);

fast_russia['Q15'].value_counts(normalize=True).loc[code_exp_order].plot(kind='barh', color=current_palette[1], alpha=0.8, ax=axs[5]);



fig.legend([patches.Patch(facecolor=current_palette[color]) for color in range(2)], ['Slow respondents', 'Quick respondents'], loc='upper right');



fig.tight_layout();
tpu_use_order = ['Never', 'Once', '2-5 times', '6-24 times', '> 25 times']



fig, axs = plt.subplots(1, 6, figsize=(20,10), sharey=True);

fig.suptitle('TPU usage aka what Google is really interested in', x=0.5, y=1.05)



axs[0].set_title('All')

slow_all['Q22'].value_counts(normalize=True).loc[tpu_use_order].plot(kind='barh', color=current_palette[0], alpha=0.8, ax=axs[0]);

fast_all['Q22'].value_counts(normalize=True).loc[tpu_use_order].plot(kind='barh', color=current_palette[1], alpha=0.8, ax=axs[0]);



axs[1].set_title('India')

slow_india['Q22'].value_counts(normalize=True).loc[tpu_use_order].plot(kind='barh', color=current_palette[0], alpha=0.8, ax=axs[1]);

fast_india['Q22'].value_counts(normalize=True).loc[tpu_use_order].plot(kind='barh', color=current_palette[1], alpha=0.8, ax=axs[1]);



axs[2].set_title('USA')

slow_usa['Q22'].value_counts(normalize=True).loc[tpu_use_order].plot(kind='barh', color=current_palette[0], alpha=0.8, ax=axs[2]);

fast_usa['Q22'].value_counts(normalize=True).loc[tpu_use_order].plot(kind='barh', color=current_palette[1], alpha=0.8, ax=axs[2]);



axs[3].set_title('Brazil')

slow_brazil['Q22'].value_counts(normalize=True).loc[tpu_use_order].plot(kind='barh', color=current_palette[0], alpha=0.8, ax=axs[3]);

fast_brazil['Q22'].value_counts(normalize=True).loc[tpu_use_order].plot(kind='barh', color=current_palette[1], alpha=0.8, ax=axs[3]);



axs[4].set_title('Japan')

slow_japan['Q22'].value_counts(normalize=True).loc[tpu_use_order].plot(kind='barh', color=current_palette[0], alpha=0.8, ax=axs[4]);

fast_japan['Q22'].value_counts(normalize=True).loc[tpu_use_order].plot(kind='barh', color=current_palette[1], alpha=0.8, ax=axs[4]);



axs[5].set_title('Russia')

slow_russia['Q22'].value_counts(normalize=True).loc[tpu_use_order].plot(kind='barh', color=current_palette[0], alpha=0.8, ax=axs[5]);

fast_russia['Q22'].value_counts(normalize=True).loc[tpu_use_order].plot(kind='barh', color=current_palette[1], alpha=0.8, ax=axs[5]);



fig.legend([patches.Patch(facecolor=current_palette[color]) for color in range(2)], ['Slow respondents', 'Quick respondents'], loc='upper right');



fig.tight_layout();
ml_use_exp_order = ['< 1 years', '1-2 years', '2-3 years', '3-4 years', '4-5 years', '5-10 years', '10-15 years', '20+ years']



fig, axs = plt.subplots(2, 3, figsize=(16,10));

fig.suptitle('Machine learning coding experience', x=0.5, y=1.05)



axs[0][0].set_title('All')

slow_all['Q23'].value_counts(normalize=True).loc[ml_use_exp_order].plot(kind='bar', color=current_palette[0], alpha=0.8, ax=axs[0][0], rot=30);

fast_all['Q23'].value_counts(normalize=True).loc[ml_use_exp_order].plot(kind='bar', color=current_palette[1], alpha=0.8, ax=axs[0][0], rot=30);



axs[0][1].set_title('India')

slow_india['Q23'].value_counts(normalize=True).loc[ml_use_exp_order].plot(kind='bar', color=current_palette[0], alpha=0.8, ax=axs[0][1], rot=30);

fast_india['Q23'].value_counts(normalize=True).loc[ml_use_exp_order].plot(kind='bar', color=current_palette[1], alpha=0.8, ax=axs[0][1], rot=30);



axs[0][2].set_title('USA')

slow_usa['Q23'].value_counts(normalize=True).loc[ml_use_exp_order].plot(kind='bar', color=current_palette[0], alpha=0.8, ax=axs[0][2], rot=30);

fast_usa['Q23'].value_counts(normalize=True).loc[ml_use_exp_order].plot(kind='bar', color=current_palette[1], alpha=0.8, ax=axs[0][2], rot=30);



axs[1][0].set_title('Brazil')

slow_brazil['Q23'].value_counts(normalize=True).loc[ml_use_exp_order].plot(kind='bar', color=current_palette[0], alpha=0.8, ax=axs[1][0], rot=30);

fast_brazil['Q23'].value_counts(normalize=True).loc[ml_use_exp_order].plot(kind='bar', color=current_palette[1], alpha=0.8, ax=axs[1][0], rot=30);



axs[1][1].set_title('Japan')

slow_japan['Q23'].value_counts(normalize=True).loc[ml_use_exp_order].plot(kind='bar', color=current_palette[0], alpha=0.8, ax=axs[1][1], rot=30);

fast_japan['Q23'].value_counts(normalize=True).loc[ml_use_exp_order].plot(kind='bar', color=current_palette[1], alpha=0.8, ax=axs[1][1], rot=30);



axs[1][2].set_title('Russia')

slow_russia['Q23'].value_counts(normalize=True).loc[ml_use_exp_order].plot(kind='bar', color=current_palette[0], alpha=0.8, ax=axs[1][2], rot=30);

fast_russia['Q23'].value_counts(normalize=True).loc[ml_use_exp_order].plot(kind='bar', color=current_palette[1], alpha=0.8, ax=axs[1][2], rot=30);



fig.legend([patches.Patch(facecolor=current_palette[color]) for color in range(2)], ['Slow respondents', 'Quick respondents'], loc='upper right');



fig.tight_layout();