# imports

# data wranglers 

import pandas as pd 

# for visualizing with plotly

import plotly.express as px

from plotly.subplots import make_subplots

import plotly.graph_objects as go

# for frequent pattern mining

from mlxtend.frequent_patterns import fpgrowth
# load all main data into dataframe

mcq_resp_df = pd.read_csv('/kaggle/input/kaggle-survey-2019/multiple_choice_responses.csv', low_memory=False)

questions_df = pd.read_csv('/kaggle/input/kaggle-survey-2019/questions_only.csv')

schema_df = pd.read_csv('/kaggle/input/kaggle-survey-2019/survey_schema.csv')

other_resp_df = pd.read_csv('/kaggle/input/kaggle-survey-2019/other_text_responses.csv')
# questions are basically in questions_df. To make life easier I will print out all the questions here for reference.

for q in questions_df.columns:

    print(q +':'+ questions_df[q][0])
# get response by country for different gender types

resp_by_country = mcq_resp_df[1:].groupby(['Q3', 'Q2']).count()['Q1'].reset_index().pivot(index='Q3', columns='Q2', values='Q1')

resp_by_country = resp_by_country.fillna(0).reset_index()

resp_by_country['Total Response'] = resp_by_country['Female'] + resp_by_country['Male'] + resp_by_country['Prefer not to say'] + resp_by_country['Prefer to self-describe']
# get male to female ratio

resp_by_country['Male To Female Ratio'] = resp_by_country['Male'] / resp_by_country['Female'] 
# plot the distribution of the response by country on a choropleth map



fig = px.choropleth(resp_by_country, locations="Q3", locationmode='country names',

                    color="Total Response", # lifeExp is a column of gapminder

                    hover_name="Q3", # column to add to hover information

                    color_continuous_scale=px.colors.sequential.Plasma,

                    title="Kaggle ML & DS Survey Response Distribution" )



fig.show()
# plot the distribution of the male to female response by country on a choropleth map



fig = px.choropleth(resp_by_country, locations="Q3", locationmode='country names',

                    color="Male To Female Ratio", # lifeExp is a column of gapminder

                    hover_name="Q3", # column to add to hover information

                    color_continuous_scale=px.colors.sequential.Plasma,

                    title='Kaggle ML & DS Survey Male To Female Ratio')



fig.show()
# count responds per age group and gender type

age_and_gender = mcq_resp_df[1:].groupby(['Q1', 'Q2']).count()['Q3'].reset_index().pivot(index='Q1', columns='Q2', values='Q3')

age_and_gender = age_and_gender.fillna(0).reset_index()

age_and_gender['Total Response'] = age_and_gender['Female'] + age_and_gender['Male'] + age_and_gender['Prefer not to say'] + age_and_gender['Prefer to self-describe']

age_and_gender['Male To Female Ratio'] = age_and_gender['Male'] / age_and_gender['Female']
# make a scatter plot for male and female responses per age group, and also the ratio

fig = make_subplots(rows=1, cols=2)



fig.add_trace(go.Scatter(x=age_and_gender['Q1'], y=age_and_gender['Male'], mode='markers', name='Male'), row=1, col=1)

fig.add_trace( go.Scatter(x=age_and_gender['Q1'], y=age_and_gender['Female'], mode='markers', name='Female'), row=1, col=1)

fig.add_trace( go.Scatter(x=age_and_gender['Q1'], y=age_and_gender['Male To Female Ratio'], mode='markers', name='Male To Female Ratio'), row=1, col=2)



fig.update_layout(height=600, width=800, title_text="Kaggle Survey Response by Age Group and Gender")

fig.show()
# a function to convert range of salary to a middle figure

def conv_salary_to_num(salary_cat):

    

    if '> $500,000' in salary_cat:

        return 500000.0

    

    salary_cat = salary_cat.replace('$','')

    low = float(salary_cat.split('-')[0].replace(',',''))

    high = float(salary_cat.split('-')[1].replace(',',''))

    

    return low + ((high - low) / 2)
# keep gender, country, and salary range

salary_est_df = mcq_resp_df[1:][['Q2','Q3','Q10']].dropna()

# convert salary range to an estimated salary

salary_est_df['salary_est'] = salary_est_df.apply(lambda row: conv_salary_to_num(row['Q10']), axis=1)
# a function to get salary array for a given country up to an indicated quantile

def get_quantile_salary_est(salary_est_df, country, quantile):



    quantile_salary_est_df = salary_est_df[salary_est_df['Q3']==country]

    quantile_salary_est_df = quantile_salary_est_df[quantile_salary_est_df.salary_est < quantile_salary_est_df.salary_est.quantile(quantile)]

    return quantile_salary_est_df['salary_est']
# histogram of our salary for India, Russia, China, USA, Canada, and Germany

fig = make_subplots(rows=2, cols=3, subplot_titles=("India", "Russia", "China", "USA", "Canada", "Germany"))



fig.add_trace(go.Histogram(x=get_quantile_salary_est(salary_est_df, 'India', 0.95)), row=1, col=1)

fig.add_trace(go.Histogram(x=get_quantile_salary_est(salary_est_df, 'Russia', 0.95)), row=1, col=2)

fig.add_trace(go.Histogram(x=get_quantile_salary_est(salary_est_df, 'China', 0.95)), row=1, col=3)



fig.add_trace(go.Histogram(x=get_quantile_salary_est(salary_est_df, 'United States of America', 0.95)), row=2, col=1)

fig.add_trace(go.Histogram(x=get_quantile_salary_est(salary_est_df, 'Canada', 0.95)), row=2, col=2)

fig.add_trace(go.Histogram(x=get_quantile_salary_est(salary_est_df, 'Germany', 0.95)), row=2, col=3)



fig.update_layout(height=600, width=800, title_text="Kaggle Survey Response by Salary and Country", showlegend=False)

fig.show()
# Get all of the column names (they are given in parts)

q12_col_names = [Q12 for Q12 in list(mcq_resp_df[1:].columns) if 'Q12' in Q12]

# remove the last one as it is the open ended other question living in another file

q12_col_names = q12_col_names[:-1]
# select a subset of data for analysis

q12_resp_df = mcq_resp_df[1:][q12_col_names]
# the actual sources are in the content of the dataframe and not the header

# here we collect the name per column, and then format it and replace it as the header

sources = []



for q12_col_name in q12_col_names:

    sources.append([val for val in mcq_resp_df[1:][q12_col_name].unique() if (type(val)==str)][0])

    

sources = [source.split('(')[0].rstrip() for source in sources]



q12_resp_df.columns = sources  
# do a count per knowledge source

q12_resp_df_count = q12_resp_df.count()
# change the dataframe to a format fpgrowth algorithm library likes

q12_resp_df = q12_resp_df.fillna(0).applymap(lambda x: True if type(x)==str else False)

# get frequent pattern

q12_fp_df = fpgrowth(q12_resp_df, min_support=0.01, use_colnames=True).sort_values(by='support', ascending=False)
# calculate items per row

q12_fp_df['items'] = q12_fp_df['itemsets'].apply(lambda x: len(x))

# format itemsets into nice strings

q12_fp_df['itemsets'] = q12_fp_df['itemsets'].apply(lambda x: ', '.join(x))

# we are interested in more than one item and we will just consider the top 10

q12_fp_df = q12_fp_df[q12_fp_df['items']>1][:10]
# make two plots, one for frequent pattern and another one pie chart

fig = make_subplots(rows=1, cols=2,subplot_titles=("Frequent Pattern", "Pie Chart"), specs=[[{'type':'xy'}, {'type':'domain'}]])

fig.add_trace(go.Pie(labels=q12_resp_df_count.index, values=q12_resp_df_count.values), 1, 2)

fig.add_trace(go.Bar(x=q12_fp_df.itemsets, y=q12_fp_df.support, ), 1, 1)
def get_col_info(question_num, df):

    '''

    return column names and answers for columns that are given in multiple parts

    question_name

    

    Args:

      question_num (str): question number of the question, i.e. Q23

      df (pandas dataframe): dataframe that contains all the parts of the question number

      

    Returns:

      tuple: returns a tuple of list, containing column names and the answers

    

    '''

    # loop over all the column names in the dataframe and create a list that 

    # contains the question number in the column name

    col_names = [q for q in list(df.columns) if question_num in q]

    # pop the OTHER column, as this links to open ended text questions

    col_names = [q for q in col_names if 'OTHER' not in q]

    

    # create a list where all the answers will be stored

    answers = []



    # go over the column names and get the unique value of this column

    # this is the answer people have picked for this part of the question

    for col_name in col_names:

        answers.append([val for val in df[col_name].unique() if (type(val)==str)][0])

    

    # remove the explanation in bracket

    answers = [answer.split('(')[0].strip() for answer in answers]

    

    # return column names and answers

    return (col_names, answers)
def get_count_for_heatmap(df, y_col, y_list, x_list):

    

    '''

    returns a tuple containing x, y, and z value required to do a heatmap given the response 

    dataframe. for each y value, the percentage of x value present is calculated and returned, 

    along with the original x and y variable list.

    

    Args:

      df (pandas dataframe): dataframe that contains all the parts of the question number

      y_col (str): name of the column for which we are interested to get count

      y_list (list): list containing different values which we consider for our y column

      x_list (list): list of columns for which we do the count per y value

      

    Returns:

      tuple: returns a tuple of list, containing x, y, and percentage of x per y needed 

      to do a heatmap

    '''

    

    # a list to contain the x count for each y values

    y_count_list = []

    # loop over all the y values

    for y in y_list:

        # a list to contain all the x values for a given y

        x_count_list = []

        # count total number of samples (used for normalization)

        x_total = len(df[df[y_col] == y])

        # loop over the x value and count it

        for x in x_list:

            count = df[df[y_col]==y][x].value_counts()[0]

            # divide x value by total number of x samples present

            x_count_list.append((count / x_total) * 100)

        # append all x percentage count for a given y

        y_count_list.append(x_count_list)

    return (x_list, y_list, y_count_list)
# question 23 asks how many years have you used machine learning methods.

# take this column and remove all the samples where this is null

q23_resp_df = mcq_resp_df[1:][~ mcq_resp_df[1:]['Q23'].isnull()]
# re order the list of values for q23 so it is chronological in heatmap

exp_list = q23_resp_df['Q23'].unique()

exp_list = exp_list[2], exp_list[0], exp_list[1], exp_list[4], exp_list[5], exp_list[6], exp_list[3], exp_list[7]
fig = make_subplots(rows=1, cols=2,subplot_titles=("ML Algorithm", "ML Framework"), specs=[[{'type':'heatmap'}, {'type':'heatmap'}]])



# get column names and answers for question 25

cols_24, answers_24 = get_col_info('Q24', mcq_resp_df[1:])

x_24, y_24, z_24 = get_count_for_heatmap(q23_resp_df, 'Q23', exp_list, cols_24)



# get column names and answers for question 28

cols_28, answers_28 = get_col_info('Q28', mcq_resp_df[1:])

x_28, y_28, z_28 = get_count_for_heatmap(q23_resp_df, 'Q23', exp_list, cols_28)



fig.add_trace(go.Heatmap(z=z_24, y=y_24, x=answers_24, showscale=False), 1, 1)

fig.add_trace(go.Heatmap(z=z_28, y=y_28, x=answers_28, showscale=False), 1, 2)



fig.show()
fig = make_subplots(rows=1, cols=2,subplot_titles=("ML Tools", "Automated ML Tools"), specs=[[{'type':'heatmap'}, {'type':'heatmap'}]])



# get column names and answers for question 25

cols_25, answers_25 = get_col_info('Q25', mcq_resp_df[1:])

x_25, y_25, z_25 = get_count_for_heatmap(q23_resp_df, 'Q23', exp_list, cols_25)



# get column names and answers for question 28

cols_33, answers_33 = get_col_info('Q33', mcq_resp_df[1:])

x_33, y_33, z_33 = get_count_for_heatmap(q23_resp_df, 'Q23', exp_list, cols_33)



fig.add_trace(go.Heatmap(z=z_25, y=y_25, x=answers_25, showscale=False), 1, 1)

fig.add_trace(go.Heatmap(z=z_33, y=y_33, x=answers_33, showscale=False), 1, 2)

 

fig.show()
fig = make_subplots(rows=1, cols=2,subplot_titles=("Cloud Platforms", "Big Data - SaaS"), specs=[[{'type':'heatmap'}, {'type':'heatmap'}]])



# get column names and answers for question 25

cols_30, answers_30 = get_col_info('Q30', mcq_resp_df[1:])

x_30, y_30, z_30 = get_count_for_heatmap(q23_resp_df, 'Q23', exp_list, cols_30)



# get column names and answers for question 28

cols_31, answers_31 = get_col_info('Q31', mcq_resp_df[1:])

x_31, y_31, z_31 = get_count_for_heatmap(q23_resp_df, 'Q23', exp_list, cols_31)



fig.add_trace(go.Heatmap(z=z_30, y=y_30, x=answers_30, showscale=False), 1, 1)

fig.add_trace(go.Heatmap(z=z_31, y=y_31, x=answers_31, showscale=False), 1, 2)



fig.show()