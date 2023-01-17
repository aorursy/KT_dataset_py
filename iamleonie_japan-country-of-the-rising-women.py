import numpy as np



import pandas as pd

pd.set_option('display.max_colwidth', -1)



import matplotlib.pyplot as plt

import seaborn as sns



import warnings

warnings.filterwarnings('ignore')



# Load survey data from 2019

multiple_choice_responses_2019= pd.read_csv("../input/kaggle-survey-2019/multiple_choice_responses.csv")



# Load survey data from 2018

multiple_choice_responses_2018 = pd.read_csv("../input/kaggle-survey-2018/multipleChoiceResponses.csv")



# Load survey data from 2017

multiple_choice_responses_2017 = pd.read_csv("../input/kaggle-survey-2017/multipleChoiceResponses.csv",encoding='ISO-8859-1')
### Preprocessing ###



# Copy original responses for further processing

responses_2017_mod = multiple_choice_responses_2017.copy()

responses_2018_mod = multiple_choice_responses_2018.copy()

responses_2019_mod = multiple_choice_responses_2019.copy()



# Drop first row, which does not contain an actual data point 

responses_2018_mod = responses_2018_mod.drop(responses_2018_mod.index[0]).reset_index(drop=True)

responses_2019_mod = responses_2019_mod.drop(responses_2019_mod.index[0]).reset_index(drop=True)



# Rename columns for more easier handling

questions_to_features_2017 = {'GenderSelect' : 'Gender'}



questions_to_features_2018 = {'Q1' : 'Gender',

                              'Q3' : 'Country',

                              'Q5' : 'Major',

                              'Q6' : 'Occupation'}



questions_to_features_2019 = {'Q1' : 'Age',

                              'Q2' : 'Gender',

                              'Q3' : 'Country',

                              'Q4' : 'Education',

                              'Q5' : 'Occupation',

                              'Q10' : 'Salary'}



responses_2017_mod.rename(columns = questions_to_features_2017, inplace = True)

responses_2018_mod.rename(columns = questions_to_features_2018, inplace = True)

responses_2019_mod.rename(columns = questions_to_features_2019, inplace = True)



### Utility functions ###

def convert_to_percentage(df):

    df = df.fillna(0)

    df['sum'] = df.sum(axis = 1)



    for col in df.columns:

        df[col] = (df[col] / df['sum'] * 100).round(1)



    df = df.drop('sum', axis=1)

    

    return df



def convert_range_to_mean(range_str):

    if any(i.isdigit() for i in range_str):

        range_str = range_str.replace(',', '').replace('+', '').replace(' ', '').replace('>', '').replace('<', '').replace('$', '').replace('years', '').replace('employees', '')

        range_array = np.array(range_str.split('-'))

        range_array = range_array.astype(np.float)

        return np.mean(range_array)

    else:

        return 0



def create_dict_from_unique_values(df, column_name):

    unique_values = df[column_name].unique()

    dicts = {}

    for i in range(len(unique_values)):

        if type(unique_values[i]) is str:

            dicts[unique_values[i]] = convert_range_to_mean(unique_values[i])

        else:

            dicts[unique_values[i]] = 0

    return dicts
def get_participation_by_country(df):

    df_mod = df[df.Country != 'Other']

    df_mod = df_mod.groupby(['Country']).agg(Number_of_Respondents = ('Country', 'count')).sort_values('Number_of_Respondents', ascending=False).reset_index()

    df_mod['Number_of_Respondents_percentage'] = df_mod['Number_of_Respondents'].apply(lambda x: (x/df_mod.Number_of_Respondents.sum()*100).round(1))

    return df_mod



# Get annual participation by country in percent

participation_2019 = get_participation_by_country(responses_2019_mod)

participation_2018 = get_participation_by_country(responses_2018_mod)

participation_2017 = get_participation_by_country(responses_2017_mod)



# Get annual participation for Japan in percent

participation_japan = pd.DataFrame(data = [participation_2017[participation_2017.Country == 'Japan'].Number_of_Respondents_percentage,

                                           participation_2018[participation_2018.Country == 'Japan'].Number_of_Respondents_percentage,

                                           participation_2019[participation_2019.Country == 'Japan'].Number_of_Respondents_percentage],

                                   index = ['2017','2018', '2019'])    



participation_japan['percentage'] = participation_japan.max(axis=1)



# Plot figure

fig = participation_japan['percentage'].plot(kind = 'bar',

                                             color = ['#C70025'])

fig.set_title('Percentage of Japanese Respondents', fontsize = 15)                                            

plt.gcf().set_size_inches(5, 5)

plt.xticks(rotation = 0, fontsize = 15)

plt.xlabel('Years of Survey', fontsize = 15)

plt.ylabel('Percentage', fontsize = 15)

fig.set_ylim(0, 5)

plt.show()



def highlight_japan(x):

    if x.Country == 'Japan':

        return ['background-color: #ffe2d8']*3

    else:

        return ['background-color: ']*3



participation_2019.head().style.set_caption('Countries Ranked By Number of Respondents in 2019 Kaggle Survey').apply(highlight_japan, axis=1)
print(responses_2019_mod[responses_2019_mod.Country == 'Japan']['Gender'].value_counts())
# Convert categorical salary column to numerical

label_encoding_salary = {}

label_encoding_salary['Salary'] = create_dict_from_unique_values(responses_2019_mod, 'Salary')

responses_2019_mod.replace(label_encoding_salary, inplace = True)



# Select data scientists from JP and USA with realistic salary

data_scientists_selection = responses_2019_mod[((responses_2019_mod.Country == 'Japan') 

                                                | (responses_2019_mod.Country =='United States of America')) 

                                               & ((responses_2019_mod.Gender == 'Male') 

                                                  | (responses_2019_mod.Gender ==  'Female'))

                                               & (responses_2019_mod.Salary > 20000)

                                               & (responses_2019_mod.Occupation == 'Data Scientist')]



# Plot figure

fig = sns.boxplot(x = "Country", 

                  y = "Salary", 

                  hue = "Gender", 

                  palette = ['#C70025', 'white'], 

                  data = data_scientists_selection, 

                  fliersize = 0)

fig.set_title('Annual Compensation of Data Scientists on Kaggle [USD]', fontsize = 15)

fig.set_ylim(0, 230000)

plt.gcf().set_size_inches(6, 5)

plt.xticks(rotation = 0, fontsize = 15)

plt.ylabel(ylabel = 'Salary', fontsize = 15)

plt.xlabel(xlabel = 'Country', fontsize = 15)

plt.legend(bbox_to_anchor = (0.5,1.2), loc = "center")

plt.show()



print('Disclaimer: \nThis analysis excludes unrealistic annual compensations of less than 20.000 USD for Data Scientists in high cost countries like Japan and USA.')

data_scientists_selection.groupby(['Country', 'Gender']).agg(Number_of_Respondents = ('Salary', 'count')).style.set_caption('Database for Salary Comparison of Data Scientists on Kaggle')
# Select Japanese Men and Women

japan_2019_overview = responses_2019_mod[(responses_2019_mod.Country == 'Japan')

                                         & (responses_2019_mod.Gender != 'Prefer not to say')]



# Create new feature of employment status based on 'Occupation' column

japan_2019_overview['Status'] = japan_2019_overview.Occupation.apply(lambda x: 'Employed' if ((x != 'Student') & (x != 'Not employed')) else x)



# Group by new status 'Status' and Gender

japan_2019_overview= japan_2019_overview.groupby(['Status', 'Gender']).agg(Number_of_respondents = ('Occupation', 'count')).reset_index()

japan_2019_overview = japan_2019_overview.pivot(index='Status',columns='Gender')['Number_of_respondents']



# Plot figure

fig = japan_2019_overview.plot(kind='bar',

                               color=['#C70025', 'white'],

                               linewidth=1,

                               edgecolor='k')

fig.set_title('Gender Dsitribution of Japanese Kagglers', fontsize = 15)                                            



plt.gcf().set_size_inches(8, 5)

plt.xticks(rotation = 0, fontsize = 15)

plt.xlabel('Employment Status', fontsize = 15)

plt.ylabel('Number of Respondents', fontsize = 15)

plt.legend(bbox_to_anchor = (0.5,1.2), loc = "center")

plt.show()
# Summarize other countries as rest of the world (RoW)

responses_2019_mod.Country = responses_2019_mod.Country.apply(lambda x: x if x == 'Japan' else 'RoW')



# Modify 'Occupation' to categories regarding leading role

positions_grouped = {'Data Scientist' : 'Non-leading position', 

                     'Research Scientist' : 'Non-leading position', 

                     'Manager' : 'Leading position',

                     'Research Assistant' : 'Leading position',

                     'Data Analyst': 'Non-leading position', 

                     'Software Engineer': 'Non-leading position', 

                     'Product/Project Manager'  : 'Leading position',

                     'Consultant': 'Non-leading position', 

                     'Marketing Analyst': 'Non-leading position', 

                     'Principal Investigator': 'Leading position',

                     'Data Engineer': 'Non-leading position',

                     'Business Analyst': 'Non-leading position',

                     'Developer Advocate': 'Non-leading position',

                     'Salesperson': 'Non-leading position',

                     'Statistician': 'Non-leading position',

                     'DBA/Database Engineer': 'Non-leading position',

                     'Data Journalist': 'Non-leading position'}



leading_positions = responses_2018_mod.copy()

leading_positions.Occupation.replace(positions_grouped, inplace = True)



# Prepare data

def get_gender_distribution_by_position(df):

    df = df[(df.Occupation != 'Student')

            & (df.Occupation != 'Other') 

            & (df.Occupation != 'Not employed') 

            & ((df.Gender == 'Female') | (df.Gender == 'Male'))] 



    df = df.groupby(['Occupation', 'Gender']).Country.count().to_frame().reset_index() 

    df = df.pivot(index='Occupation',columns='Gender')['Country']

    

    df = convert_to_percentage(df)

    return df



gender_distribution_position = pd.DataFrame(data = [get_gender_distribution_by_position(leading_positions[leading_positions.Country == 'Japan']).Female, 

                                                    get_gender_distribution_by_position(leading_positions[leading_positions.Country != 'Japan']).Female],

                                            index = ['Japan','RoW'])    



# Plot figure

fig = gender_distribution_position.T.plot(kind = 'bar',

                                          color = ['#C70025', '#ffe2d8'])

fig.set_title('Female Kagglers in the Workforce', fontsize = 15)                                            



plt.gcf().set_size_inches(8,5)

plt.xticks(rotation = 0, fontsize = 15)

plt.xlabel('Position', fontsize = 15)

plt.ylabel('Percentage of Respondents', fontsize = 15)

plt.legend(bbox_to_anchor = (0.5,1.2), loc = "center")

plt.show()
# Rename columns for better (human) handling

questions_to_features_algo = {'Q24_Part_1' : 'Linear or Logistic Regression',

                              'Q24_Part_2' : 'Decision Trees or Random Forests',

                              'Q24_Part_3' : 'Gradient Boosting Machines',

                              'Q24_Part_4' : 'Bayesian Approaches',

                              'Q24_Part_5' : 'Evolutionary Approaches',

                              'Q24_Part_6' : 'Dense Neural Networks',

                              'Q24_Part_7' : 'Convolutional Neural Networks',

                              'Q24_Part_8' : 'Generative Adversarial Networks',

                              'Q24_Part_9' : 'Recurrent Neural Networks',

                              'Q24_Part_10' : 'Transformer Networks',

                              'Q24_Part_11' : 'No ML algorithms',

                              'Q24_Part_12' : 'Other ML algorithms',

                              'Q26_Part_1' : 'General purpose image/video tools',

                              'Q26_Part_2' : 'Image segmentation methods',

                              'Q26_Part_3' : 'Object detection methods',

                              'Q26_Part_4' : 'Image classification methods',

                              'Q26_Part_5' : 'Generative Networks',

                              'Q26_Part_6' : 'No CV method',

                              'Q26_Part_7' : 'Other CV method',

                              'Q27_Part_1' : 'Word embeddings/vectors',

                              'Q27_Part_2' : 'Encoder-decoder models',

                              'Q27_Part_3' : 'Contextualized embeddings',

                              'Q27_Part_4' : 'Transformer language models',

                              'Q27_Part_5' : 'No NLP methods',

                              'Q27_Part_6' : 'Other NLP methods'}

                            

responses_2019_mod.rename(columns = questions_to_features_algo, inplace = True)



# Prepare lists

algorithms_ml = ['Linear or Logistic Regression',

                 'Decision Trees or Random Forests',

                 'Gradient Boosting Machines',

                 'Bayesian Approaches',

                 'Evolutionary Approaches',

                 'Dense Neural Networks',

                 'Convolutional Neural Networks',

                 'Generative Adversarial Networks',

                 'Recurrent Neural Networks',

                 'Transformer Networks',

                 #'No ML algorithms',

                 #'Other ML algorithms'

                ]



algorithms_cv = ['General purpose image/video tools',

                 'Image segmentation methods',

                 'Object detection methods',

                 'Image classification methods',

                 'Generative Networks',

                 #'No CV method',

                 #'Other CV method'

                ]



algorithms_nlp = ['Word embeddings/vectors',

                  'Encoder-decoder models',

                  'Contextualized embeddings',

                  'Transformer language models',

                  #'No NLP methods',

                  #'Other NLP methods'

                 ]



algorithms = algorithms_ml + algorithms_cv + algorithms_nlp



# Prepare data

for i in range(len(algorithms)):

    responses_2019_mod[algorithms[i]] = responses_2019_mod[algorithms[i]].apply(lambda x: 1 if (type(x) is str) else 0)



algorithms_2019 = responses_2019_mod.groupby('Country')[algorithms].sum()

algorithms_2019 = convert_to_percentage(algorithms_2019)



# Plot figures

fig, axes = plt.subplots(1,3)

algorithms_2019[algorithms_ml].T.plot(kind = 'bar', 

                                      color = ['#C70025', '#ffe2d8'], 

                                      ax = axes[0], 

                                      title = 'ML Algorithms used by Kagglers', 

                                      fontsize = 15)

algorithms_2019[algorithms_cv].T.plot(kind = 'bar', 

                                      color = ['#C70025', '#ffe2d8'], 

                                      ax = axes[1], 

                                      title = 'CV Methods used by Kagglers', 

                                      fontsize = 15)

algorithms_2019[algorithms_nlp].T.plot(kind = 'bar', 

                                       color = ['#C70025', '#ffe2d8'], 

                                       ax = axes[2], 

                                       title = 'NLP Methods used by Kagglers', 

                                       fontsize = 15)



fig.set_size_inches(18,5)



for i in range(3):

    axes[i].set_ylim(0, 18)

    axes[i].set_xlabel('Methods', fontsize = 15)

    axes[i].set_ylabel('Percentage of Used Methods', fontsize = 15)

    axes[i].get_legend().remove()



fig.legend(bbox_to_anchor=(0.77,0.1), loc="center", labels=['Japan', 'RoW'])

plt.show()
# Dictionary to group STEM disciplines

studies_grouped = {'Computer science (software engineering, etc.)' : 'STEM',

                   'Engineering (non-computer focused)' : 'STEM',

                   'Mathematics or statistics' : 'STEM', 

                   'Physics or astronomy' : 'STEM',

                   'Information technology, networking, or system administration' : 'STEM',

                   'Environmental science or geology' : 'STEM',

                   'Medical or life sciences (biology, chemistry, medicine, etc.)' : 'STEM',

                   'A business discipline (accounting, economics, finance, etc.)' : 'Business',

                   'Social sciences (anthropology, psychology, sociology, etc.)' : 'Social Sciences',

                   'Humanities (history, literature, philosophy, etc.)' : 'Humanities',

                   'Fine arts or performing arts' : 'Arts',

                   'I never declared a major' : 'No major'}



responses_2018_mod.Major.replace(studies_grouped, inplace = True)



# Prepare data

df_female = responses_2018_mod[(responses_2018_mod.Country == 'Japan')

                               & (responses_2018_mod.Gender == 'Female')].groupby('Major')['Major'].count()

df_male = responses_2018_mod[(responses_2018_mod.Country == 'Japan')

                             & (responses_2018_mod.Gender == 'Male')].groupby('Major')['Major'].count()



gender_distribution_japan = pd.DataFrame(data = [df_female, df_male],index = ['Female','Male'])    



gender_distribution_japan['total'] = [df_female.sum(), df_male.sum()]

gender_distribution_japan.fillna(0)



for col in gender_distribution_japan.columns:

    gender_distribution_japan['{} (%)'.format(col)] = gender_distribution_japan[col]/gender_distribution_japan['total']*100



rearranged_columns = ['STEM (%)',

                      'Business (%)',

                      'Social Sciences (%)',

                      'Humanities (%)',

                      'Arts (%)',

                      'Other (%)',

                      'No major (%)' ]



# Plot figure

fig = gender_distribution_japan[rearranged_columns].T[['Female']].plot(kind='bar',color=['#C70025'])

plt.gcf().set_size_inches(15,5)

fig.set_title('Female Participation in Academic Disciplines in Japan', fontsize = 15)

plt.xticks(np.arange(7),

           ('STEM', 'Business', 'Social Sciences', 'Humanities', 'Arts', 'Other', 'No major'), 

           rotation = 0,

           fontsize = 15)

plt.xlabel('Academic Discipline', fontsize = 15)

plt.ylabel('Percentage of Female Respondents', fontsize = 15)

fig.get_legend().remove()

plt.show()
# Prepare list

rearranged_columns = ['Social Sciences',

                      'Humanities',

                      'Arts']



# Plot figure

fig = gender_distribution_japan[rearranged_columns].T[['Male','Female']].plot(kind='bar',

                                                                              color=['white','#C70025'],

                                                                              linewidth=1,

                                                                              edgecolor='k')

fig.set_title('Gender Distribution in Academic Disciplines in Japan', fontsize = 15)                                            

plt.gcf().set_size_inches(8,5)

plt.xticks(rotation = 0, fontsize = 15)

plt.xlabel('Academic Discipline', fontsize = 15)

plt.ylabel('Number of Respondents', fontsize = 15)

plt.legend(bbox_to_anchor=(0.5,1.2), loc="center",labels=['Male','Female'])

plt.show()





non_stem_fields_japan = responses_2018_mod[(responses_2018_mod.Country == 'Japan')

                                           & (responses_2018_mod.Gender != 'Prefer not to say') 

                                           & ((responses_2018_mod.Major == 'Social Sciences') 

                                              | (responses_2018_mod.Major == 'Humanities')

                                              | (responses_2018_mod.Major == 'Arts'))]

non_stem_fields_japan = non_stem_fields_japan.groupby(['Major', 'Gender']).Occupation.unique().to_frame()

non_stem_fields_japan.head(6).style.set_caption('Occupations of Japanese Kagglers Who Have Not Majored in STEM Fields')
# Convert categorical age column to numerical

label_encoding_age = {}

label_encoding_age['Age'] = create_dict_from_unique_values(responses_2019_mod, 'Age')

responses_2019_mod.replace(label_encoding_age, inplace = True)



# Get Japanese females who went to university

japan_females_university_age_2019 = responses_2019_mod[(responses_2019_mod.Country == 'Japan')

                                                       & (responses_2019_mod.Gender== 'Female')

                                                       & (responses_2019_mod.Education != 'No formal education past high school')]



# Prepare data

def reached_unverisity_age_at(current_age):

    if current_age <= 27:

        return 'Later than 2010s'

    elif current_age <= 37:

        return 'Early 2000s'

    elif current_age <=47:

        return 'Around 1990'

    else:

        return '1980s and earlier'

    

japan_females_university_age_2019['university_age_reached'] = japan_females_university_age_2019.Age.apply(lambda x: reached_unverisity_age_at(x))

japan_females_university_age_2019 = japan_females_university_age_2019.groupby(['university_age_reached', 'Education']).Gender.count().to_frame().reset_index()

japan_females_university_age_2019 = japan_females_university_age_2019.pivot(index='university_age_reached',columns='Education')['Gender']#.reset_index(drop=True)



japan_females_university_age_2019 = convert_to_percentage(japan_females_university_age_2019)



# Plot figure

degrees = ['Some college/university study without earning a bachelor’s degree', 'Bachelor’s degree', 'Master’s degree']



fig = japan_females_university_age_2019[degrees].plot(kind = 'bar', 

                                                      color=['#ffe2d8', "#ff9999", "#C70025"])

fig.set_title('Education Level by When Graduated from High School', fontsize = 15)                                            



plt.gcf().set_size_inches(10,5)

plt.xticks(rotation=0, fontsize = 15)

plt.xlabel('Year when Graduated from High School', fontsize = 15)

plt.ylabel('Percentage of Respondents', fontsize = 15)

plt.legend(bbox_to_anchor=(0.5,1.3), loc="center")

plt.show()



print('Disclaimer: The number of respondents aged higher than 47 was very low. It is assumed that this group of Japanese females that are interested in an online data science community are a rarity for their generation. Under this aspect, the high level of education seems to be no surprise.')
# Prepare data

def get_gender_distribution_by_country(df):

    df_mod = df[(df.Gender == 'Female')| (df.Gender == 'Male')]

    df_mod.Country = df_mod.Country.apply(lambda x: x if x == 'Japan' else 'Row')

    df_mod = df_mod.groupby(['Country', 'Gender']).agg(Number_of_respondents = ('Gender', 'count' )).reset_index()

    df_mod = df_mod.pivot(index='Country',columns='Gender')['Number_of_respondents']

    df_mod = convert_to_percentage(df_mod)

    

    return df_mod



students = pd.DataFrame(data = [get_gender_distribution_by_country(responses_2017_mod[responses_2017_mod.StudentStatus == 'Yes']).Female, 

                                get_gender_distribution_by_country(responses_2018_mod[responses_2018_mod.Occupation == 'Student']).Female,

                                get_gender_distribution_by_country(responses_2019_mod[responses_2019_mod.Occupation == 'Student']).Female],

                        index = ['2017','2018','2019'])  



working_people = pd.DataFrame(data = [get_gender_distribution_by_country(responses_2017_mod[((responses_2017_mod.EmploymentStatus == 'Independent contractor, freelancer, or self-employed') |

                                                                                             (responses_2017_mod.EmploymentStatus == 'Employed full-time') |

                                                                                             (responses_2017_mod.EmploymentStatus == 'Employed part-time'))]).Female, 

                                      get_gender_distribution_by_country(responses_2018_mod[((responses_2018_mod.Occupation != 'Student') & 

                                                                                             (responses_2018_mod.Occupation != 'Not employed'))]).Female,

                                      get_gender_distribution_by_country(responses_2019_mod[((responses_2019_mod.Occupation != 'Student') & 

                                                                                             (responses_2019_mod.Occupation != 'Not employed'))]).Female],

                              index = ['2017','2018','2019'])  

# Plot figures

fig, axes = plt.subplots(1,2)



working_people.plot(kind = 'bar',

                    color = ['#C70025', '#ffe2d8'], 

                    ax = axes[0], 

                    title = 'Workforce', 

                    fontsize = 15)



students.plot(kind ='bar', 

              color = ['#C70025', '#ffe2d8'], 

              ax = axes[1], 

              title = 'Students', 

              fontsize = 15)



fig.set_size_inches(12,5)



for i in range(2):

    axes[i].set_ylim(0, 25)

    axes[i].tick_params(labelrotation = 0)

    axes[i].set_xlabel('Year of Survey', fontsize = 15)

    axes[i].set_ylabel('Percentage of Females', fontsize = 15)

    axes[i].get_legend().remove()



fig.legend(bbox_to_anchor=(0.5,1.1), loc="left", labels = ['Japan', 'RoW'])

plt.show()
# Get working people

working_people = responses_2018_mod[(responses_2018_mod.Occupation != 'Student') 

                                    & (responses_2018_mod.Occupation != 'Other') 

                                    & (responses_2018_mod.Occupation != 'Not employed')]



# Convert categorical column (Q23: time spent coding) to numerical column

time_spent_coding = {'0% of my time' : 0,

                     '75% to 99% of my time' : 88,

                     '50% to 74% of my time': 63,

                     '100% of my time' : 100, 

                     '25% to 49% of my time' : 38,

                     '1% to 25% of my time' : 13}

            

working_people.Q23.replace(time_spent_coding, inplace=True)



# Prepare data

working_people = working_people.groupby('Occupation').Q23.mean().reset_index().sort_values('Q23', ascending = False)



# Plot figure

fig = working_people.set_index('Occupation').Q23.plot(kind = 'bar',

                                                      color = ['#C70025'])



fig.set_title('Percentage of Work Time Spent Programming by Occupation', fontsize = 15)                                            



plt.gcf().set_size_inches(20,6)

plt.xlabel('Occupation', fontsize = 15)

plt.ylabel('Percentage of Working Time', fontsize = 15)

fig.set_ylim(0, 105)

plt.show()