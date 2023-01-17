#Importing dependent modules

import numpy as np 

import pandas as pd 

from matplotlib import pyplot as plt
raw_data = pd.read_csv('../input/survey_results_public.csv',dtype=str)

raw_data.head()
raw_data.shape
columns_selected_for_analysis= ['Hobby','OpenSource','Country','Student','Employment',

                                'FormalEducation','UndergradMajor','CompanySize','DevType',

                                'YearsCoding','YearsCodingProf','JobSatisfaction','CareerSatisfaction',

                                'SalaryType','ConvertedSalary','CommunicationTools','TimeFullyProductive',

                                'EducationTypes','SelfTaughtTypes','LanguageWorkedWith',

                                'LanguageDesireNextYear','DatabaseWorkedWith','DatabaseDesireNextYear',

                                'PlatformWorkedWith','PlatformDesireNextYear','FrameworkWorkedWith',

                                'FrameworkDesireNextYear','IDE','OperatingSystem','Methodology',

                                'VersionControl','CheckInCode','WakeTime','HoursComputer','HoursOutside',

                                'SkipMeals','Gender','Age']
analysis_data = raw_data[columns_selected_for_analysis]

analysis_data.shape
analysis_data.describe()
for col in analysis_data.columns :

    print(col)

    print(analysis_data[col].value_counts())

    print()
#Limiting the data to only companies having less than 100 employees

data=analysis_data[analysis_data['CompanySize'].isin([

    '20 to 99 employees','10 to 19 employees','Fewer than 10 employees'])]
data.shape
#Determing the perentage of people in startup/smallscale IT frim

x=['Startup/Small Scale', 'Rest']

y=[len(data)/len(analysis_data),(len(analysis_data)-len(data))/len(analysis_data)]

plt.bar(x,y)

plt.xlabel('Data')

plt.ylabel('Percent of total data')

plt.title('People employed in startup');
#Checking null values in each column

data.isnull().sum()
null_col_data=data.isnull().sum()/len(data)*100

null_col_data.plot(kind='barh',figsize=(12,10));
#Function which plots bar and barh graphs

def plot_bar_graph(df_series, kind='bar', figsize=(10,10),rot=0, x_label=None, y_label=None,title=None):

    '''

    input :

    df_series: Series dataframe object which is to be plotted. 

                Index values taken as x-axis and values as y-axis by defaiul.

    kind : 'bar','barh' (str object).

           Specifies the type of bar graph

    figsize : (x,y) touple object . Determines the dimensions of the figure 

    rot : rotating the x ticks (integer)

    

    Output : plots bar graph and/or saves the image depending on the save argument 

    

    This functions takes a series object and plots the bar graph 

    '''

    plot_object=df_series.plot(kind=kind,rot=rot,title=title,figsize=figsize)

    plot_object.set_xlabel(x_label)

    plot_object.set_ylabel(y_label);

        
data_country_wise=data['Country'].value_counts()*100/len(data)
#Considering only top 10 

data_country_wise=data_country_wise.sort_values(ascending=False)[:10]

plot_bar_graph(data_country_wise,kind='bar',rot=60,title='Percentage of people vs Country',figsize=(7,5),

              x_label='Country',y_label='Percentage')
data['Age'].value_counts().sum()
age_total_cnt=data['Age'].value_counts().sum()

age_distribution_data=data['Age'].value_counts()/age_total_cnt*100

plot_bar_graph(age_distribution_data,kind='barh',rot=0,title='Age distribution',figsize=(12,5),

              x_label='Percentage',y_label='Age Range')
def get_mean_age_value(x):

    '''

    input:

    X(str) - String object

    optupt :

    Returns value based on string 

    

    Returns middle value for the different date ranges

    '''

    

    if x == 'Under 18 years old' :

        return 18

    elif x == '18 - 24 years old' :

        return (18+24)/2

    elif x == '25 - 34 years old' :

        return (25+34)/2

    elif x == '35 - 44 years old' :

        return (35+44)/2

    elif x == '45 - 54 years old' :

        return (45+54)/2

    elif x == '55 - 64 years old' :

        return (55+64)/2

    elif x == '65 years or older' :

        return 65
#adding a new engineered column AgeMean

data['AgeMean'] = data['Age'].apply(lambda x : get_mean_age_value(x) if x is not np.nan else np.nan )



#Getting the mean 

data['AgeMean'].mean()
data['ConvertedSalary']=data['ConvertedSalary'].astype(float);
#Plotting the graph for mean salary for each age range 

salary_data=data.groupby(['Age']).mean()['ConvertedSalary']

plot_bar_graph(salary_data,kind='bar',rot=90,title='Salary vs Age',figsize=(7,5),

              x_label='Age',y_label='Salary in USD($)')

#Getting the average salary

data.ConvertedSalary.mean()
salary_data_education=data.groupby(['FormalEducation']).mean()['ConvertedSalary']

plot_bar_graph(salary_data_education,kind='barh',rot=0,title='Formal Educations vs Average Salary',figsize=(7,5),

              x_label='FormalEducation',y_label='Salary in USD($)')

#Function to split the srting and get the count

def get_split_count(df,m=10):

    '''

    input : df (dataframe) : data to be split and counted.

            m : (int) to get the top m entries default 10

    output : pandas Series object which has entries as index and the count as values.

    

    This function is used to get the split count of the data which is seperated by  ;

    

    '''

    

    lang_dict=dict()

    len_df=len(df)

    for i in range(len_df) :

        lang_str =df.iloc[i]

        if lang_str is not np.nan :

            lang_list = lang_str.split(';')

            for lang in lang_list :

                if lang in lang_dict :

                    lang_dict[lang] += 1

                else :

                    lang_dict[lang] = 1

    lang_dict=pd.Series(lang_dict,index=lang_dict.keys()).sort_values(ascending=False)[:m]

    lang_dict=lang_dict/len_df*100

    return lang_dict
#Get top 10 languages

lang_worked_with = get_split_count(data['LanguageWorkedWith'])

#Plot

plot_bar_graph(lang_worked_with,kind='bar',rot=45,title='Languages Worked/Working With',figsize=(7,5),

              x_label='Programming Languages',y_label='Percentage')
#Plot top 10 lang desired next year

plot_bar_graph(get_split_count(data['LanguageDesireNextYear']),kind='bar',rot=45,title='Languages Desired To Work With Next Year',figsize=(7,5),

              x_label='Programming Languages',y_label='Percentage')
os_data=data['OperatingSystem'].value_counts()

os_data_len=data['OperatingSystem'].value_counts().sum()

os_data=os_data/os_data_len*100

plot_bar_graph(os_data,kind='bar',rot=0,title='Operating Systems Used',figsize=(7,5),

              x_label='Operating systems',y_label='Percentage')
#Plot top version control system used

plot_bar_graph(get_split_count(data['VersionControl']),kind='barh',rot=0,title='Version Control Methods/Systems Used',figsize=(7,5),

              x_label='Percentage',y_label='Version Control')
data['Hobby'].isnull().sum()
hobby_data=data['Hobby'].value_counts()/len(data)*100

plot_bar_graph(hobby_data,kind='bar',rot=0,title='Working as Hobby',figsize=(7,5),

              x_label='Hobby',y_label='Percentage')