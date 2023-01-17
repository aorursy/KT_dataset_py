# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import seaborn as sns

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.

multi = pd.read_csv('../input/multipleChoiceResponses.csv')



multi.head()


#Remove the first row which contains the question text

multi = multi.drop(multi.index[0])
def salary_convert(salary):

    '''A function that converts str in 'Q9' column and converts to int. Salaries in the CSV are listed as ranges. 

    For calcuation I took the mid point of each range.

    

    Keyword Arguments:

    salary = str representing a salary range that was pulled from the 'Q9' in the multi DataFrame

    '''

    avg = salary

    if isinstance(salary, str):

        if  '-' in salary:

            range = salary.split('-')

            avg = (int(range[0]) + int(range[1].split(',')[0]))*500

        

        else:

            avg = np.NaN

    return avg

multi['Salary'] = multi.Q9.apply(salary_convert)
major_salary = multi.groupby(['Q5'], as_index = False).Salary.mean()

major_salary
#ms_graph = convert(major_salary, 'Major')

ms = sns.barplot(x = 'Q5', y= 'Salary' , data = major_salary, color = 'blue')

ms.set_xticklabels(major_salary['Q5'], rotation=90)
multi.groupby(['Q5']).Salary.value_counts()

#Removing rows where the answer to question 6 or 7 indicates they are a student

mask = multi['Q6'] == 'Student'

mask2 = multi['Q7'] == 'I am a student'

multi_no_students = multi[~mask]

multi_no_students_2 = multi_no_students[~mask2]
multi_no_students_2.groupby(['Q5']).mean()['Salary']
multi.Q5.value_counts()
multi[multi['Q5'] == 'Computer science (software engineering, etc.)'].Q3.value_counts()

multi_no_students_2.groupby(['Q3']).mean()['Salary']
country_income = pd.DataFrame(multi_no_students_2.groupby(['Q3']).mean()['Salary'])

country_income['Countries'] = country_income.index

ci = sns.swarmplot(x = 'Countries', y="Salary", data=country_income)

ci.set_xticklabels([],rotation=90)
income_by_country = multi_no_students_2.groupby(['Q3']).mean()['Salary']

income_by_country

mns = multi_no_students_2



high_income = [k for k, v in income_by_country.items() if v > 40000]



def label_income_level(country_name):

    '''A function that returns 1 for high income countries and 0 for low income countries. It will be used to create

    a column of 1's and 0's that will be used to sort respondents into two seperate datasets.

    

    Keyword Arguments:

    country_name = An entry from the 'Q3' column of the multi DataFrame. It is checked against the high_income list

                    and assigned a 1 or 0.

    '''

    if country_name in high_income:

        return 1

    else:

        return 0

    

mns['income_region'] = mns["Q3"].apply(label_income_level)





multi_high = mns[mns['income_region'] == 1]

multi_low = mns[mns['income_region'] == 0]

major_salary_high = multi_high.groupby(['Q5'], as_index = False).Salary.mean()

major_salary_high
msh = sns.barplot(x = 'Q5', y= 'Salary' , data = major_salary_high, color = 'blue')

msh.set_xticklabels(major_salary_high['Q5'], rotation=90)
multi_low.groupby(['Q5']).mean()['Salary']
multi.groupby(['Q5']).Q4.value_counts()
coding = multi.groupby(['Q24'], as_index = False).Salary.mean()

myorder = [8,9,7,0,3,6,1,2,4]

#Need to reorder because default order was nonsensical. We'll need to do the same on ML question

ordered = coding.reindex(myorder, axis = 0)

ordered
#code_graph = convert(coding, 'Coding Experience')

#code_graph

code = sns.barplot(x = 'Salary', y= 'Q24' , data = ordered, color = 'blue')

#code.set_xticklabels(ordered['Q24'], rotation=90)
multi.groupby(['Q25'], as_index = False).Salary.mean()

ml_order = [8,9,7,0,2,4,5,6,1,3]
ml_graph = multi.groupby(['Q25'], as_index = False).Salary.mean()

ml_ordered = ml_graph.reindex(ml_order)

ml = sns.barplot(x = 'Salary', y= 'Q25' , data = ml_ordered, color = 'blue')

multi.groupby(['Q48']).Salary.mean()

ml_graph = multi.groupby(['Q48'], as_index = False).Salary.mean()

ml = sns.barplot(x = 'Q48', y= 'Salary' , data = ml_graph, color = 'blue')

ml.set_xticklabels(ml_graph['Q48'], rotation=90)
def super_groupby (df,comparison,base_text, part_max):

    '''This function is designed to find value counts from multiple columns and group them by the responses in a single column.

        The design is built specifically around the column name formats used in the multipleChoiceResponses.csv file. It iterates

        the mulitple column comparing them to the single column and concatenates the results""

    

        Keyword Arguements:

        df = A DataFrame containing the relevant data

        comparison = A single column

        base_text = The basic format for the columns to be iterated (should include a {} in place of a part number)

        part_max = The highest number for np.arange (should be one more than the highest numbered part # in the columns or,

                more simply, one more than the number of columns)

        '''

    parts = np.arange(1,part_max)



    column_list = []

    for part in parts:

        column_list.append(base_text.format(part))

    count_list = []

    for column in column_list:

        df = df.rename(columns={column: 'current'})

        count = df.groupby([comparison]).current.value_counts()

        df = df.rename(columns={'current': column})

        count_list.append(count)

    concatenated = pd.concat(count_list, sort =False)

    return concatenated.sort_index()



def plot_supergroup(supergroup, column = 'current'):

    '''A function for plotting value count groupby's primarily those generated by super_groupby

    

    Keyword Arguements:

    supergroup = A Pandas Series generated by either groupby or super_groupby

    column = The name of the column generated by groupby or super_groupby (which is named 

            'current' if generated by super_groupby thus that is the default)

    '''

    df = pd.DataFrame(supergroup)

    df = df.rename(columns={column:'Count'})

    df.reset_index(inplace=True)

    ax2= sns.barplot(y = 'Q48', x = 'Count', hue = column, data=df)

    ax2.legend(loc=4, bbox_to_anchor=(0.5, 1.0))
frameworks = super_groupby(multi,'Q48','Q19_Part_{}',20)

plot_supergroup(frameworks)
analysis_tools = multi.groupby(['Q48']).Q12_MULTIPLE_CHOICE.value_counts()

plot_supergroup(analysis_tools,'Q12_MULTIPLE_CHOICE')
role = super_groupby(multi,'Q48','Q11_Part_{}',7)

plot_supergroup(role)
dtype = super_groupby(multi,'Q48','Q31_Part_{}',13)

plot_supergroup(dtype)