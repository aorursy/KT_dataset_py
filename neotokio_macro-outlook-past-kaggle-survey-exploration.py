import pandas as pd

import matplotlib.pyplot as plt

import matplotlib.cm as cm

import matplotlib.colors as colors

from collections import OrderedDict

import numpy as np



multiple_choice = pd.read_csv('../input/kaggle-survey-2019/multiple_choice_responses.csv')

multiple_choice_2018 = pd.read_csv('../input/kaggle-survey-2018/multipleChoiceResponses.csv')

# other_text_responses = pd.read_csv('../input/kaggle-survey-2019/other_text_responses.csv')

# questions_only = pd.read_csv('../input/kaggle-survey-2019/questions_only.csv')

# survey_schema = pd.read_csv('../input/kaggle-survey-2019/survey_schema.csv')



personal = multiple_choice[['Q1', 'Q2', 'Q3', 'Q4', 'Q6', 'Q10']]
age_gender = personal.groupby(['Q2', 'Q1']).size().rename('count').reset_index()

age_gender.columns = ['gender', 'age', 'count']

age_gender.drop(age_gender.index[44], inplace=True)

age_gender_viz = (age_gender.pivot_table(index='age', columns='gender', values='count'))

ax = age_gender_viz.plot(kind='barh', width=1.0, colors=['r', 'b', 'g', 'm'], alpha=0.3, title='Age/Gender Relation')

ax.set_xlabel('No. of responders')

ax.set_ylabel('Age')
salaries = personal.groupby(['Q3', 'Q10']).size().rename('count').reset_index()

salaries_count = salaries.groupby('Q3').sum()

salaries_reported = salaries_count['count'].nlargest(10)

salaries.set_index('Q3', inplace=True)

top10 = salaries.loc[['India', 'United States of America', 'Brazil', 'Japan', 'Russia', 'Germany', 'United Kingdom of Great Britain and Northern Ireland', 'Spain', 'Canada']]



india = top10.loc['India'].sort_values(by='count').nlargest(columns='count', n=5).reset_index().set_index('Q10')

usa = top10.loc['United States of America'].sort_values(by='count').nlargest(columns='count', n=5).reset_index().set_index('Q10')

brazil = top10.loc['Brazil'].sort_values(by='count').nlargest(columns='count', n=5).reset_index().set_index('Q10')

japan = top10.loc['Japan'].sort_values(by='count').nlargest(columns='count', n=5).reset_index().set_index('Q10')

russia = top10.loc['Russia'].sort_values(by='count').nlargest(columns='count', n=5).reset_index().set_index('Q10')

germany = top10.loc['Germany'].sort_values(by='count').nlargest(columns='count', n=5).reset_index().set_index('Q10')

uk = top10.loc['United Kingdom of Great Britain and Northern Ireland'].sort_values(by='count').nlargest(columns='count', n=5).reset_index().set_index('Q10')

spain = top10.loc['Spain'].sort_values(by='count').nlargest(columns='count', n=5).reset_index().set_index('Q10')

canada = top10.loc['Canada'].sort_values(by='count').nlargest(columns='count', n=5).reset_index().set_index('Q10')



top_df = pd.concat([india, usa, brazil, japan, russia, germany, uk, spain, canada])

top_df.reset_index(inplace=True)

top_df.Q10.str.replace('$', '')

top_df.Q10 = top_df.Q10.str.replace(r'\d+,\d+-', '')

top_df.Q10 = top_df.Q10.str.replace(r'\d+-', '')

top_df.Q10 = top_df.Q10.str.replace(r'\W', '')

top_df.Q3 = top_df.Q3.str.replace('United Kingdom of Great Britain and Northern Ireland', 'UK')

top_df.Q3 = top_df.Q3.str.replace('United States of America', 'USA')

top_df.Q10 = top_df.Q10.astype(int)

top_df['count'] = top_df['count'].astype(int)



fig, ax = plt.subplots()

nrof_labels = len(top_df['Q3'])

colors = cm.rainbow(np.linspace(0, 1, nrof_labels))



for i, c in enumerate(colors):

    x = top_df['count'][i]

    y = top_df['Q10'][i]

    l = top_df['Q3'][i]

    ax.scatter(x, y, label=l, c=c, alpha=0.5)



handles, labels = plt.gca().get_legend_handles_labels()

by_label = OrderedDict(zip(labels, handles))

plt.legend(by_label.values(), by_label.keys())

ax.set_xlabel('No. of people claiming given salary')

ax.set_ylabel('Salary in USD (max value)')

ax.set_title('Income/Country Relation')

plt.show()
edu_plot = pd.Series(personal.Q4.value_counts())

edu_plot.drop(labels='What is the highest level of formal education that you have attained or plan to attain within the next 2 years?', inplace=True)

work_plot = pd.Series(personal.Q6.value_counts())

work_plot.drop(labels='What is the size of the company where you are employed?', inplace=True)



edus = personal.groupby(['Q4', 'Q6']).size().rename('count')

edus_df = pd.DataFrame(edus)

edus_df.drop(index=edus_df.index[-1:], inplace=True)

edus_df.reset_index(inplace=True)

edus_df.Q4 = edus_df.Q4.str.replace('Some college/university study without earning a bachelor’s degree', 'Some study without earning degree')

edus_df.columns = ['Education', 'Size of Company', 'Count']

edus_df['Size of Company'] = edus_df['Size of Company'].str.replace('10,000','10000')

edus_df['Size of Company'] = edus_df['Size of Company'].str.replace('9,999','9999')

edus_df.set_index(['Education', 'Size of Company'], inplace=True)

ax = edus_df.unstack().plot(kind='barh', alpha=0.3)

ax.set_xlabel('No. of Responses')

ax.set_ylabel('Level of Education')

ax.set_title('Education/Size of Company Employer Company')

handles, labels = ax.get_legend_handles_labels()

labels_new = [label.strip('()').split(',')[1] for label in labels]

plt.legend(handles, labels_new)
code_long = multiple_choice[['Q15', 'Q23']]

code_long.drop(code_long[0:1].index, inplace=True)

code_long.dropna(inplace=True)

q15_values = pd.DataFrame(code_long['Q15'].value_counts())

q23_values = pd.DataFrame(code_long['Q23'].value_counts())

entry = pd.DataFrame(q23_values.loc[['2-3 years', '3-4 years', '4-5 years']].sum())





q23_values.drop(['2-3 years', '3-4 years',], inplace=True)

q23_values.rename(index={'4-5 years' : '3-5 years', '10-15 years': '10-20 years'}, inplace=True)

q23_values.Q23.replace(927, 3847, inplace=True)

q23_values.columns = ['ML Methods']

q15_values.columns = ['Years Programming']

ax = pd.concat({'Years Coding':q15_values, 'Years Using ML Methods': q23_values}, axis=1).plot.barh(alpha=0.3, color=['g','b'])

ax.set_xlabel('No. of Answers')

ax.set_ylabel('Years Coding/Years Using ML Methods')

ax.set_title('Relation of experience (in years) in programming to years using ML methods')

handles, labels = ax.get_legend_handles_labels()

labels_new = [label.strip('()').split(',')[1] for label in labels]

plt.legend(handles, labels_new)

plt.show()
favorite_media_df = multiple_choice.filter(regex='Q12_')

fav_count = favorite_media_df.apply(lambda x: x.value_counts())

fav_count.drop(fav_count[11:].index, inplace=True)

fav_count.rename(index={'Kaggle (forums, blog, social media, etc)':'Kaggle forums',

                        'Blogs (Towards Data Science, Medium, Analytics Vidhya, KDnuggets etc)':'Blogs',

                        'Journal Publications (traditional publications, preprint journals, etc)':'Journal Publications',

                        'Course Forums (forums.fast.ai, etc)': 'Course Forums',

                        'Twitter (data science influencers)': 'Twitter',

                        'Reddit (r/machinelearning, r/datascience, etc)': 'Social/Reddit',

                        'Slack Communities (ods.ai, kagglenoobs, etc)': 'Social/Slack',

                        'Podcasts (Chai Time Data Science, Linear Digressions, etc)': 'Podcasts',

                        'Hacker News (https://news.ycombinator.com/)': 'Hacker News'

                        }, inplace=True)

fav_count = fav_count.filter(regex='Q12').bfill(axis=1).iloc[:, 0].fillna('unknown')

fav_count.sort_values(inplace=True)



favorite_media_df_2018 = multiple_choice_2018.filter(regex='Q38')

fav_count_2018 = favorite_media_df_2018.apply(lambda x: x.value_counts())

fav_count_2018.drop(fav_count_2018[22:].index, inplace=True)

fav_count_2018 = fav_count_2018.filter(regex='Q38').bfill(axis=1).iloc[:, 0].fillna('unknown')

fav_count_2018.sort_values(inplace=True)

fav_count_2018.drop(['Journal Publications', 'ArXiv & Preprints'], inplace=True)

fav_count_2018['Journal Publications'] = 5291.0

fav_count_2018.drop(['FiveThirtyEight.com', 'Cloud AI Adventures (YouTube)', 'KDnuggets Blog', 'DataTau News Aggregator', 'FastML Blog', 'Medium Blog Posts'], inplace=True)

fav_count_2018['Blogs'] = 11113.0



ax = pd.concat({'2019 Favorite Media':fav_count, '2018 Favorite Media': fav_count_2018[3:]}, axis=1).plot.barh(alpha=0.3, color=['g','b'])

ax.set_xlabel('No. of Votes')

ax.set_title('Favorite Media Sources in 2019 and 2018')

plt.show()
mltools = multiple_choice.filter(regex='Q8')

mltools_count = mltools.apply(lambda x: x.value_counts())

mltools_count.drop(mltools_count[12:].index, inplace=True)

mltools_count = mltools_count.filter(regex='Q8').bfill(axis=1).iloc[:, 0].fillna('unknown')

mltools_count.drop(['Does your current employer incorporate machine learning methods into their business?'], inplace=True)



mltools_2018 = multiple_choice_2018.filter(regex='Q10')

mltools_count_2018 = mltools_2018.apply(lambda x: x.value_counts())

mltools_count_2018.drop(mltools_count_2018[12:].index, inplace=True)

mltools_count_2018 = mltools_count_2018.filter(regex='Q10').bfill(axis=1).iloc[:, 0].fillna('unknown')

mltools_count_2018.drop(['Does your current employer incorporate machine learning methods into their business?'], inplace=True)





ax = pd.concat({'2019':mltools_count, '2018': mltools_count_2018}, axis=1).plot.barh(alpha=0.3, color=['g','b'])

ax.set_xlabel('No. of Votes')

ax.set_title('Does your current employer incorporate machine learning methods into their business?')

plt.show()

activities = multiple_choice.filter(regex='Q9')

activities_count = activities.apply(lambda x: x.value_counts())

activities_count.drop(activities_count[12:].index, inplace=True)

activities_count = activities_count.filter(regex='Q9').bfill(axis=1).iloc[:, 0].fillna('unknown')

activities_count.drop(['Select any activities that make up an important part of your role at work: (Select all that apply) - Selected Choice - Analyze and understand data to influence product or business decisions',

                       'Select any activities that make up an important part of your role at work: (Select all that apply) - Selected Choice - Build and/or run a machine learning service that operationally improves my product or workflows',

                       'Select any activities that make up an important part of your role at work: (Select all that apply) - Selected Choice - Build and/or run the data infrastructure that my business uses for storing, analyzing, and operationalizing data',

                       'Select any activities that make up an important part of your role at work: (Select all that apply) - Selected Choice - Build prototypes to explore applying machine learning to new areas',

                       ], inplace=True)



activities_2018 = multiple_choice_2018.filter(regex='Q11')

activities_count_2018 = activities_2018.apply(lambda x: x.value_counts())

activities_count_2018.drop(activities_count_2018[12:].index, inplace=True)

activities_count_2018 = activities_count_2018.filter(regex='Q11').bfill(axis=1).iloc[:, 0].fillna('unknown')

activities_count_2018.drop(['Select any activities that make up an important part of your role at work: (Select all that apply) - Selected Choice - Analyze and understand data to influence product or business decisions',

                       'Select any activities that make up an important part of your role at work: (Select all that apply) - Selected Choice - Build and/or run a machine learning service that operationally improves my product or workflows',

                       'Select any activities that make up an important part of your role at work: (Select all that apply) - Selected Choice - Build and/or run the data infrastructure that my business uses for storing, analyzing, and operationalizing data',

                       'Select any activities that make up an important part of your role at work: (Select all that apply) - Selected Choice - Build prototypes to explore applying machine learning to new areas',

                       'Select any activities that make up an important part of your role at work: (Select all that apply) - Selected Choice - Do research that advances the state of the art of machine learning',

                        ], inplace=True)



ax1 = pd.concat({'2019':activities_count, '2018': activities_count_2018}, axis=1).plot.barh(alpha=0.3, color=['g','b'])

ax1.set_xlabel('No. of Votes')

ax1.set_title('Activities that make up an important part of your role at work')

plt.show()
primary_tool = multiple_choice.filter(regex='Q14')

primary_tool_count = primary_tool.apply(lambda x: x.value_counts())

primary_tool_count.drop(primary_tool_count[12:].index, inplace=True)

primary_tool_count = primary_tool_count.filter(regex='Q14').bfill(axis=1).iloc[:, 0].fillna('unknown')

primary_tool_count.drop(primary_tool_count[6:].index, inplace=True)



primary_tool_2018 = multiple_choice_2018.filter(regex='Q12')

primary_tool_count_2018 = primary_tool_2018.apply(lambda x: x.value_counts())

primary_tool_count_2018.drop(primary_tool_count_2018[12:].index, inplace=True)

primary_tool_count_2018 = primary_tool_count_2018.filter(regex='Q12').bfill(axis=1).iloc[:, 0].fillna('unknown')

primary_tool_count_2018.drop(primary_tool_count_2018[6:].index, inplace=True)

primary_tool_count_2018.rename(index={'Local or hosted development environments (RStudio, JupyterLab, etc.)':'Local development environments (RStudio, JupyterLab, etc.)'}, inplace=True)





ax2 = pd.concat({'2019':primary_tool_count, '2018': primary_tool_count_2018}, axis=1).plot.barh(alpha=0.3, color=['g','b'])

ax2.set_xlabel('No. of Votes')

ax2.set_title('What is the primary tool that you use to analyze data?')

plt.show()