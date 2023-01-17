import numpy as np # almost everything

import pandas as pd # working with tables

import seaborn as sns # plotting library

%matplotlib inline

from matplotlib import patches

import matplotlib.pyplot as plt # importing pyplot

from scipy import stats # importing some stats that will be useful later on
sns.set_style('whitegrid') # white grid background on plots

sns.set_palette('colorblind', 10) # color palette for colorblinds with 10 colors

current_palette = sns.color_palette() # saving this palette in separate variable

sns.set_context('poster') # setting notebook context
data = pd.read_csv('../input/kaggle-survey-2019/multiple_choice_responses.csv', low_memory=False); # loading data

USA = data[data['Q3'] == 'United States of America'].copy() # copying only US ones
print('Number of US respondents: ', USA.shape[0])
us_salary = USA['Q10'].str.replace(',','').str.replace('$','').str.replace('>','').str.strip(' ').str.split('-').apply(pd.Series).astype(np.float32) # getting numeric values of salary

us_salary.columns = ['min', 'max'] # renaming columns

us_salary['midrange'] = (us_salary['max'] + us_salary['min']) / 2 # calculating mid-range https://en.wikipedia.org/wiki/Mid-range



# renaming columns for transparency

us_salary_prediction = pd.get_dummies(USA[['Q4', 'Q6', 'Q7', 'Q8', 'Q15', 'Q23']].rename({'Q4': 'Education',

                                                                                          'Q6': 'Company Size',

                                                                                          'Q7': 'DS involvement',

                                                                                          'Q8': 'ML use',

                                                                                          'Q15': 'Overall DS experience',

                                                                                          'Q23': 'Overall ML experience'}, axis='columns'), prefix_sep=': ')
# calculating correlations that we will use in "Salary" section

corrs = pd.DataFrame()

for idx, col in enumerate(us_salary_prediction.columns):

    t_min, p_min = stats.kendalltau(us_salary_prediction[col], us_salary['min'], nan_policy='omit')

    t_max, p_max = stats.kendalltau(us_salary_prediction[col], us_salary['max'], nan_policy='omit')

    t_mr, p_mr = stats.kendalltau(us_salary_prediction[col], us_salary['midrange'], nan_policy='omit')

    corrs.loc[idx, 'Feature'] = col

    corrs.loc[idx, 'Kendall'] = np.min([t_min, t_max, t_mr])

    corrs.loc[idx, 'Pval'] = np.max([p_min, p_max, p_mr])

corrs.set_index('Feature', inplace=True)
USA.loc[USA[(USA['Q1'] == '50-54') | (USA['Q1'] == '55-59') | (USA['Q1'] == '60-69') | (USA['Q1'] == '70+')].index, 'Q1'] = '>50' 

USA.loc[USA[(USA['Q2'] == 'Prefer to self-describe')].index, 'Q2'] = 'Self-described' 

age_gender_plot_data = USA.drop(USA[(USA['Q2'] == 'Prefer not to say')].index, axis=0).groupby('Q1')['Q2'].value_counts().rename('Percentage').reset_index()

age_gender_plot_data = age_gender_plot_data.rename({'Q1': 'Age', 'Q2': 'Gender'}, axis='columns').pivot(columns='Gender', index='Age', values='Percentage')

age_gender_plot_data.plot(kind='barh', stacked=True, title='Age-Gender', figsize=(20,12));
ed_plot_data = USA['Q4'].value_counts(normalize=True).rename('Percentage').reset_index().rename({'index': 'Education'}, axis='columns')

ed_plot_data['Education'] = ed_plot_data['Education'].map({'Master’s degree': 'Master', 

                                                           'Bachelor’s degree': 'Bachelor', 

                                                           'Doctoral degree': 'Doctor', 

                                                           'Some college/university study without earning a bachelor’s degree': 'Audition',

                                                           'No formal education past high school': 'High School',

                                                           'I prefer not to answer': 'Other',

                                                           'Professional degree': 'Professional'})

ed_plot_data.drop(ed_plot_data[ed_plot_data['Education'] == 'Other'].index, axis=0, inplace=True)

ed_order = ['High School', 'Professional', 'Audition', 'Bachelor', 'Master', 'Doctor']

ed_plot_data.set_index('Education').loc[ed_order].plot(kind='bar', figsize=(16,12), legend=False, title='Percentage of people with different degrees');
USA.loc[USA[(USA['Q7'] == '0') | (USA['Q7'] == '1-2') | (USA['Q7'] == '3-4')].index, 'Q7'] = '<5'

USA.loc[USA[(USA['Q7'] == '5-9') | (USA['Q7'] == '10-14') | (USA['Q7'] == '15-19')].index, 'Q7'] = '5-20'

USA.loc[USA[(USA['Q7'] == '20+')].index, 'Q7'] = '>20' 

size_order = ['0-49 employees', '50-249 employees', '250-999 employees', '1000-9,999 employees', '> 10,000 employees']



companies_plot_data = USA.groupby('Q6')['Q7'].value_counts().rename('Count').reset_index()

companies_plot_data = companies_plot_data.rename({'Q6': 'Company size', 'Q7': 'DS involvement'}, axis='columns').pivot(columns='DS involvement', index='Company size', values='Count')

companies_plot_data = companies_plot_data.loc[size_order]



companies_plot_data[['<5', '5-20', '>20']].plot(kind='barh', stacked=True, figsize=(16,12), title='Number of data scientists in companies');
companies_plot_data_norm = USA.groupby('Q6')['Q7'].value_counts(normalize=True).rename('Percentage').reset_index()

companies_plot_data_norm = companies_plot_data_norm.rename({'Q6': 'Company size', 'Q7': 'DS involvement'}, axis='columns').pivot(columns='DS involvement', index='Company size', values='Percentage')

companies_plot_data_norm = companies_plot_data_norm.loc[size_order]



companies_plot_data_norm[['<5', '5-20', '>20']].plot(kind='barh', stacked=True, figsize=(16,12), title='Number of data scientists in companies (normalized)');
USA.loc[USA[(USA['Q8'] == 'No (we do not use ML methods)')].index, 'Q8'] = 'None'

USA.loc[USA[(USA['Q8'] == 'We use ML methods for generating insights (but do not put working models into production)') | (USA['Q8'] == 'We are exploring ML methods (and may one day put a model into production)')].index, 'Q8'] = 'Low'

USA.loc[USA[(USA['Q8'] == 'We recently started using ML methods (i.e., models in production for less than 2 years)') | (USA['Q8'] == 'We have well established ML methods (i.e., models in production for more than 2 years)')].index, 'Q8'] = 'High' 



ds_ml_companies_plot_data = USA.drop(USA[USA['Q8'] == 'I do not know'].index, axis=0).groupby(['Q6', 'Q8'])['Q7'].value_counts().rename('Count').reset_index().rename({'Q6': 'Company size', 'Q7': 'DS involvement', 'Q8': 'ML use'}, axis='columns')



ds_less_5 = ds_ml_companies_plot_data[ds_ml_companies_plot_data['DS involvement'] == '<5'].pivot(columns='ML use', index='Company size', values='Count')

ds_less_5 = ds_less_5.loc[size_order].fillna(0)



ds_from_5_to_20 = ds_ml_companies_plot_data[ds_ml_companies_plot_data['DS involvement'] == '5-20'].pivot(columns='ML use', index='Company size', values='Count')

ds_from_5_to_20 = ds_from_5_to_20.loc[size_order].fillna(0)



ds_more_20 = ds_ml_companies_plot_data[ds_ml_companies_plot_data['DS involvement'] == '>20'].pivot(columns='ML use', index='Company size', values='Count')

ds_more_20 = ds_more_20.loc[size_order].fillna(0)



ds_less_5_norm = (ds_less_5 / ds_less_5.sum(axis=1).values.reshape(5,1))

ds_more_20_norm = (ds_more_20 / ds_more_20.sum(axis=1).values.reshape(5,1))

ds_from_5_to_20_norm = (ds_from_5_to_20 / ds_from_5_to_20.sum(axis=1).values.reshape(5,1))
fig, ax = plt.subplots(figsize=(16,12));

plt.title('Normalized plot of company sizes with different number of data scientists and ML methods usage')



ds_less_5_norm.plot(kind='barh', stacked=True, width=0.2, position=1.5, ax=ax, legend=False, hatch='///');

ds_more_20_norm.plot(kind='barh', stacked=True, width=0.2, position=-0.5, ax=ax, legend=False);

ds_from_5_to_20_norm.plot(kind='barh', stacked=True, width=0.2, position=0.5, ax=ax, legend=False, hatch='...');



q8_hatch_legend = plt.legend([patches.Patch(hatch='///'), patches.Patch(hatch='...'), patches.Patch()], 

                             ['Less than 5 DS', 'From 5 to 20 DS', 'More than 20 DS'], 

                             loc='lower right', bbox_to_anchor=(1.3, 0), borderaxespad=0.)



q8_color_legend = plt.legend([patches.Patch(facecolor=current_palette[0]), patches.Patch(facecolor=current_palette[1]), patches.Patch(facecolor=current_palette[2])], 

                             ['High ML use', 'Low ML use', 'No ML use'],

                             loc='upper right', bbox_to_anchor=(1.245, 1), borderaxespad=0.)



ax = plt.gca().add_artist(q8_hatch_legend)
q9_rename_dict = {'Q9_Part_1': 'Analyze and understand data to influence product or business decisions',

                  'Q9_Part_2': 'Build and/or run the data infrastructure that my business uses for storing, analyzing, and operationalizing data',

                  'Q9_Part_3': 'Build prototypes to explore applying machine learning to new areas',

                  'Q9_Part_4': 'Build and/or run a machine learning service that operationally improves my product or workflows',

                  'Q9_Part_5': 'Experimentation and iteration to improve existing ML models'}



q9_size_plot = USA[['Q6', 'Q9_Part_1', 'Q9_Part_2', 'Q9_Part_3', 'Q9_Part_4', 'Q9_Part_5']].groupby('Q6').count().rename(q9_rename_dict, axis='columns')

q9_size_plot = q9_size_plot.loc[size_order].T

q9_size_plot_norm = q9_size_plot / q9_size_plot.sum(axis=0)



q9_size_plot_norm.plot(kind='pie', subplots=True, figsize=(32,6), legend=False, labels=None, autopct='%.1f', title='Pie charts for comapy sizes vs their ML usage', layout=(1,5));



q9_legend = plt.legend([patches.Patch(facecolor=current_palette[color]) for color in range(6)], 

                       [q9_rename_dict['Q9_Part_1'], q9_rename_dict['Q9_Part_2'], q9_rename_dict['Q9_Part_3'], 

                        q9_rename_dict['Q9_Part_4'], q9_rename_dict['Q9_Part_5']], bbox_to_anchor=(1, -1), loc='lower right', borderaxespad=0.)



plt.gca().add_artist(q9_legend);
mr_male_us_salary = us_salary[USA.Q2 == 'Male']['midrange']

mr_female_us_salary = us_salary[USA.Q2 == 'Female']['midrange']

mr_sd_us_salary = us_salary[USA.Q2 == 'Self-described']['midrange']
fig, ax = plt.subplots(figsize=(16,12));

plt.title('Midrange salary KDE for males and females');



male_mode = mr_male_us_salary.mode()[0]

male_mean = mr_male_us_salary.mean()

male_median = mr_male_us_salary.median()

mr_male_us_salary.rename('Male midrange salary', inplace=True).plot.kde(ax=ax, bw_method='silverman');

plt.axvline(male_mean, color='r', linestyle='dotted', lw=3, label='Male salary mean')

plt.axvline(male_median, color='r', linestyle='dashed', lw=3, label='Male salary median')



female_mode = mr_female_us_salary.mode()[0]

female_mean = mr_female_us_salary.mean()

female_median = mr_female_us_salary.median()

mr_female_us_salary.rename('Female midrange salary', inplace=True).plot.kde(ax=ax, bw_method='silverman');

plt.axvline(female_mean, color='k', linestyle='dotted', lw=3, label='Female salary mean')

plt.axvline(female_median, color='k', linestyle='dashed', lw=3, label='Female salary median')



plt.legend();
fig, ax = plt.subplots(figsize=(16,12));

plt.title('Midrange salary KDE for self-described gender');



sd_mode = mr_sd_us_salary.mode()[0]

sd_mean = mr_sd_us_salary.mean()

sd_median = mr_sd_us_salary.median()

mr_sd_us_salary.rename('Self-described midrange salary', inplace=True).plot.kde(ax=ax, bw_method='silverman');

plt.axvline(sd_mean, color='y', linestyle='dotted', lw=2, label='Salary mean')

plt.axvline(sd_median, color='y', linestyle='dashed', lw=2, label='Salary median')



plt.legend();
corrs[(corrs['Pval'] < 0.01) & (corrs['Kendall'] > 0)]['Kendall'].sort_values().plot(kind='barh', title='Positive Kendall Tau correlation (p < 0.01)', figsize=(16,12));
corrs[(corrs['Pval'] < 0.01) & (corrs['Kendall'] < 0)]['Kendall'].sort_values().plot(kind='barh', title='Negative Kendall Tau correlation (p < 0.01)', figsize=(16,12));
loglogtime = np.log(np.log(USA['Time from Start to Finish (seconds)'].astype(np.int32)))

q1 = np.percentile(loglogtime, 25)

q3 = np.percentile(loglogtime, 75)



fast = loglogtime[loglogtime < q1].index

slow = loglogtime[loglogtime > q3].index



age_order = ['18-21', '22-24', '25-29', '30-34', '35-39', '40-44', '45-49', '>50']



print('Median duration for slow respondents is {} min'.format(int(np.round(USA.loc[slow]['Time from Start to Finish (seconds)'].astype(np.int32).median() / 60))))

print('Median duration for quick respondents is {} min'.format(int(np.round(USA.loc[fast]['Time from Start to Finish (seconds)'].astype(np.int32).median() / 60))))

print('Number of respondents in each group is {}'.format(int(USA.loc[fast].shape[0])))
fig, ax = plt.subplots(figsize=(16,12));

plt.title('Age comparison between slow and quick respondents')

USA.loc[slow]['Q1'].value_counts(normalize=True).loc[age_order].plot(kind='bar', ax=ax, color=current_palette[0], alpha=0.75);

USA.loc[fast]['Q1'].value_counts(normalize=True).loc[age_order].plot(kind='bar', ax=ax, color=current_palette[1], alpha=0.75);

plt.legend([patches.Patch(facecolor=current_palette[color]) for color in range(2)], ['Slow respondents', 'Quick respondents'], loc='upper center');
fig, ax = plt.subplots(figsize=(16,12));

plt.title('Gender comparison between slow and quick respondents')

USA.loc[slow]['Q2'].value_counts(normalize=True).plot(kind='bar', ax=ax, color=current_palette[0], alpha=0.75);

USA.loc[fast]['Q2'].value_counts(normalize=True).plot(kind='bar', ax=ax, color=current_palette[1], alpha=0.75);

plt.legend([patches.Patch(facecolor=current_palette[color]) for color in range(2)], ['Slow respondents', 'Quick respondents'], loc='upper center');
fig, ax = plt.subplots(figsize=(16,12));

plt.title('Degree comparison between slow and quick respondents')

USA.loc[slow]['Q4'].map({'Master’s degree': 'Master', 

                         'Bachelor’s degree': 'Bachelor', 

                         'Doctoral degree': 'Doctor', 

                         'Some college/university study without earning a bachelor’s degree': 'Audition',

                         'No formal education past high school': 'High School',

                         'I prefer not to answer': 'Other',

                         'Professional degree': 'Professional'}).value_counts(normalize=True).loc[ed_order].plot(kind='bar', ax=ax, color=current_palette[0], alpha=0.75);

USA.loc[fast]['Q4'].map({'Master’s degree': 'Master', 

                         'Bachelor’s degree': 'Bachelor', 

                         'Doctoral degree': 'Doctor', 

                         'Some college/university study without earning a bachelor’s degree': 'Audition',

                         'No formal education past high school': 'High School',

                         'I prefer not to answer': 'Other',

                         'Professional degree': 'Professional'}).value_counts(normalize=True).loc[ed_order].plot(kind='bar', ax=ax, color=current_palette[1], alpha=0.75);

plt.legend([patches.Patch(facecolor=current_palette[color]) for color in range(2)], ['Slow respondents', 'Quick respondents'], loc='upper center');
fig, ax = plt.subplots(figsize=(16,12));

plt.title('Company size comparison between slow and quick respondents')

USA.loc[slow]['Q6'].value_counts(normalize=True).loc[size_order].plot(kind='bar', ax=ax, color=current_palette[0], alpha=0.75);

USA.loc[fast]['Q6'].value_counts(normalize=True).loc[size_order].plot(kind='bar', ax=ax, color=current_palette[1], alpha=0.75);

plt.legend([patches.Patch(facecolor=current_palette[color]) for color in range(2)], ['Slow respondents', 'Quick respondents'], loc='upper center');
fig, ax = plt.subplots(figsize=(16,12));

plt.title('Number of DS in company comparison between slow and quick respondents')

USA.loc[slow]['Q7'].value_counts(normalize=True).loc[['<5','5-20','>20']].plot(kind='bar', ax=ax, color=current_palette[0], alpha=0.75);

USA.loc[fast]['Q7'].value_counts(normalize=True).loc[['<5','5-20','>20']].plot(kind='bar', ax=ax, color=current_palette[1], alpha=0.75);

plt.legend([patches.Patch(facecolor=current_palette[color]) for color in range(2)], ['Slow respondents', 'Quick respondents'], loc='upper center');
fig, ax = plt.subplots(figsize=(16,12));

plt.title('ML use company comparison between slow and quick respondents')

USA.loc[slow]['Q8'].value_counts(normalize=True).plot(kind='bar', ax=ax, color=current_palette[0], alpha=0.75);

USA.loc[fast]['Q8'].value_counts(normalize=True).plot(kind='bar', ax=ax, color=current_palette[1], alpha=0.75);

plt.legend([patches.Patch(facecolor=current_palette[color]) for color in range(2)], ['Slow respondents', 'Quick respondents'], loc='upper center');
fig, ax = plt.subplots(figsize=(16,12));

plt.title('Salary comparison between slow and quick respondents')

us_salary.loc[slow]['midrange'].rename('Slow respondents', inplace=True).plot.kde(ax=ax, bw_method='silverman');

us_salary.loc[fast]['midrange'].rename('Quick respondents', inplace=True).plot.kde(ax=ax, bw_method='silverman');



slow_mode = us_salary.loc[slow]['midrange'].mode()[0]

slow_mean = us_salary.loc[slow]['midrange'].mean()

slow_median = us_salary.loc[slow]['midrange'].median()



fast_mode = us_salary.loc[fast]['midrange'].mode()[0]

fast_mean = us_salary.loc[fast]['midrange'].mean()

fast_median = us_salary.loc[fast]['midrange'].median()



plt.axvline(slow_mean, color='b', linestyle='dotted', lw=3, label='Slow respondents salary mean')

plt.axvline(slow_median, color='b', linestyle='dashed', lw=3, label='Slow respondents salary median')

plt.axvline(fast_mean, color='r', linestyle='dotted', lw=3, label='Quick respondents salary mean')

plt.axvline(fast_median, color='r', linestyle='dashed', lw=3, label='Quick respondents salary median')



plt.gca().add_artist(plt.legend(loc='upper right'));