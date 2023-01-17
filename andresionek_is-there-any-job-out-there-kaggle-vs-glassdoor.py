## Basic Cleaning - Kaggle Survey

import numpy as np

import pandas as pd

import os

import re



# Loading the multiple choices dataset from Kaggle Survey, we will not look to the free form data on this study

kaggle_multiple_choice = pd.read_csv('/kaggle/input/kaggle-survey-2019/multiple_choice_responses.csv', low_memory=False)

kaggle_multiple_choice.head()
# Separating questions from answers

# This Series stores all questions

kaggle_questions = kaggle_multiple_choice.iloc[0,:]

kaggle_questions.head()
# This DataFrame stores all answers

kaggle = kaggle_multiple_choice.iloc[1:,:]

kaggle.head()
# removing everyone that took less than 3 minutes or more than 600 minutes to answer the survey

answers_before = kaggle.shape[0]

print(f'Initial dataset length is {answers_before} answers.')



# Creating a mask to identify those who took less than 3 min

less_3_minutes = kaggle[round(kaggle.iloc[:,0].astype(int) / 60) <= 3].index

# Dropping those rows

kaggle = kaggle.drop(less_3_minutes, axis=0)



# Creating a mask to identify those who took more than 600 min

more_600_minutes = kaggle[round(kaggle.iloc[:,0].astype(int) / 60) >= 600].index

kaggle = kaggle.drop(more_600_minutes, axis=0)



answers_after = kaggle.shape[0]

print('After removing respondents that took less than 3 minutes or more than 600 minutes' \

      f'to answer the survey we were left with {answers_after} answers.')
# removing respondents who are not employed or project/product managers

answers_before = kaggle.shape[0]



# Creating a mask to identify Job titles that are not interesting for this study

students_and_others = kaggle[(kaggle.Q5 == 'Student') | \

                             (kaggle.Q5 == 'Other') | \

                             (kaggle.Q5 == 'Not employed') | \

                             (kaggle.Q5 == 'Product/Project Manager')

                            ].index

# Dropping rows

kaggle = kaggle.drop(list(students_and_others), axis=0)

answers_after = kaggle.shape[0]

print(f'After removing respondents who are not employed or project/product managers we were left with {answers_after} answers.')
# Removing those who didn't disclose compensation (Q10 is NaN)

answers_before = kaggle.shape[0]

kaggle.dropna(subset=['Q10'], inplace=True)

answers_after = kaggle.shape[0]

print(f'After removing respondents who did not disclose compensation there were left {answers_after} answers.')
# Now lets group some data

kaggle.Q5.value_counts()
# Groupping DBA + Data Engineer

kaggle.Q5 = kaggle.Q5.replace('DBA/Database Engineer', 'Data Engineer/DBA')

kaggle.Q5 = kaggle.Q5.replace('Data Engineer', 'Data Engineer/DBA')

kaggle.Q5.value_counts()
# Groupping Statistician + Research Scientist

kaggle.Q5 = kaggle.Q5.replace('Statistician', 'Statistician/Research Scientist')

kaggle.Q5 = kaggle.Q5.replace('Research Scientist', 'Statistician/Research Scientist')

kaggle.Q5.value_counts()
# Simplifying country names

kaggle.Q3 = kaggle.Q3.replace('United Kingdom of Great Britain and Northern Ireland', 'United Kingdom')

kaggle.Q3 = kaggle.Q3.replace('United States of America', 'United States')
# We have 12 columns with answers about programming language (Q18)

# Lets concatenate them into a list

kaggle['ProgLang'] = kaggle[['Q18_Part_1', 'Q18_Part_2', 'Q18_Part_3', 'Q18_Part_4', 'Q18_Part_5',

                             'Q18_Part_6', 'Q18_Part_7', 'Q18_Part_8', 'Q18_Part_9', 'Q18_Part_10', 

                             'Q18_Part_11', 'Q18_Part_12']].values.tolist()

kaggle.ProgLang.head()
# remove nulls from list of programming languages

kaggle.ProgLang = kaggle.ProgLang.apply(lambda x: [item for item in x if not pd.isnull(item)])

kaggle.ProgLang.head()
# Calculates the quantity of different Programming Languages for each user

kaggle['QtyProgLang'] = kaggle.ProgLang.apply(lambda x: len(x))

# If Quantity > 6 then it will be 6

kaggle.loc[kaggle.QtyProgLang > 6, 'QtyProgLang'] = 6

kaggle.QtyProgLang.head()
# We have 12 columns with answers about Cloud Platforms (Q29)

# Lets concatenate them into a list as we did for programming languages

kaggle['CloudPlatf']= kaggle[['Q29_Part_1', 'Q29_Part_2', 'Q29_Part_3', 'Q29_Part_4',

                            'Q29_Part_5', 'Q29_Part_6', 'Q29_Part_7', 'Q29_Part_8',

                            'Q29_Part_9', 'Q29_Part_10', 'Q29_Part_11', 'Q29_Part_12']].values.tolist()



# remove nulls

kaggle.CloudPlatf = kaggle.CloudPlatf.apply(lambda x: [item.strip().lower() for item in x if not pd.isnull(item)])



# Calculates the quantity

kaggle['QtyCloudPlatf'] = kaggle.CloudPlatf.apply(lambda x: len(x))

kaggle.loc[kaggle.QtyCloudPlatf > 6, 'QtyCloudPlatf'] = 6
# Finally the same logic for Databases (Q34)

kaggle['Databases']= kaggle[['Q34_Part_1', 'Q34_Part_2', 'Q34_Part_3', 'Q34_Part_4', 'Q34_Part_5', 'Q34_Part_6',

                             'Q34_Part_7', 'Q34_Part_8', 'Q34_Part_9', 'Q34_Part_10', 'Q34_Part_11', 'Q34_Part_12']].values.tolist()

kaggle.Databases = kaggle.Databases.apply(lambda x: [item.strip().lower() for item in x if not pd.isnull(item)])

kaggle['QtyDatabases'] = kaggle.Databases.apply(lambda x: len(x))

kaggle.loc[kaggle.QtyDatabases > 6, 'QtyDatabases'] = 6
# Now lets rename some columns to have more meaningfull names

kaggle.columns = kaggle.columns.str.replace('Q15', 'TimeWritingCode')

kaggle.columns = kaggle.columns.str.replace('Q10', 'Salary')

kaggle.columns = kaggle.columns.str.replace('Q1', 'Age')

kaggle.columns = kaggle.columns.str.replace('Q5', 'JobTitle')

kaggle.columns = kaggle.columns.str.replace('Q3', 'Country')

kaggle.columns = kaggle.columns.str.replace('Q6', 'CompanySize')
# To create good plots, we need to transform some columns into categories.

# This is because a category might have a logical order, that will be preserved when plotting.

# Otherwise the categories would be sorted by alphabetical order



# Transform TimeWritingCode column into category

time_writting_code = ['I have never written code', '< 1 years', '1-2 years', '3-5 years',

                      '5-10 years', '10-20 years', '20+ years']

cat_dtype = pd.api.types.CategoricalDtype(categories=time_writting_code, ordered=True)

kaggle.TimeWritingCode = kaggle.TimeWritingCode.astype(cat_dtype)

# Now TimeWritingCode has a specific order as defined by the list time_writting_code
# We will do the same for CompanySize and transform the column into category

company_size = ['0-49 employees', '50-249 employees', '250-999 employees', '1000-9,999 employees', '> 10,000 employees']

cat_dtype = pd.api.types.CategoricalDtype(categories=company_size, ordered=True)

kaggle.CompanySize = kaggle.CompanySize.astype(cat_dtype)
# And the same for JobTitle column. Transform it into category

job_titles = ['Business Analyst', 'Data Analyst', 'Data Scientist', 

              'Data Engineer/DBA', 'Software Engineer', 'Statistician/Research Scientist']

cat_dtype = pd.api.types.CategoricalDtype(categories=job_titles, ordered=True)

kaggle.JobTitle = kaggle.JobTitle.astype(cat_dtype)
# Add count column to make groupby easier

kaggle['Count'] = 1
# Transform range of salaries into numerical value

# We are summing up the lowest and highest value for each category, and then dividing by 2.

# Some regex needed to clean the text

compensation = kaggle.Salary.str.replace(r'(?:(?!\d|\-).)*', '').str.replace('500000', '500-500000').str.split('-')

kaggle.Salary = compensation.apply(lambda x: (int(x[0]) * 1000 + int(x[1]))/ 2) / 1000 # it is calculated in thousand dollars
# Transform range of ages into numerical value

# The same we did for salary, we will do for age

age = kaggle.Age.str.replace(r'(?:(?!\d|\-).)*', '').str.replace('70', '70-80').str.split('-')

kaggle.Age = (age.apply(lambda x: (int(x[0]) + int(x[1]))/ 2)).astype(int)
# Filtering only the columns we will need

kaggle = kaggle[['Age', 'Country', 'JobTitle', 'CompanySize', 'Salary', 'TimeWritingCode', 'ProgLang',

                 'QtyProgLang', 'CloudPlatf', 'QtyCloudPlatf', 'Databases', 'QtyDatabases', 'Count']]

# Finally our Dataframe looks like this

kaggle.head(10)
# One more to go! 

# Basic Cleaning - Glassdoor Data

# Loading the main file for glassdoor listings

glassdoor_full = pd.read_csv('/kaggle/input/data-jobs-listings-glassdoor/glassdoor.csv')

glassdoor_full.head()
# Selecting only the columns that are interesting for the purpose of this study

glassdoor_columns = ['header.jobTitle', 'job.description', 'map.country']

glassdoor = glassdoor_full[glassdoor_columns].copy()



# Rename columns for more meaningful names

glassdoor.columns = ['JobTitle', 'JobDescription', 'Country']

glassdoor.head()
# Dropping NaN countries

listings_before = glassdoor.shape[0]

print(f'Initial Glassdoor dataset length is {listings_before} job listings.')



glassdoor.dropna(subset=['Country'], inplace=True)



listings_after = glassdoor.shape[0]

print(f'After removing NaN countries we were left with {listings_after} job listings.')
# As you can see we have some country names writen in full, others are 2 digits codes.

# Lets fix that by replacing 2 digits country codes by the full country name



# This table has a list of country names vs 2 digit codes

country_codes = pd.read_csv('/kaggle/input/data-jobs-listings-glassdoor/country_names_2_digit_codes.csv')

country_codes.head()
# We merge both by 2 digit code, and then fill the NaNs with the full country name

glassdoor = pd.merge(glassdoor, country_codes, left_on='Country', right_on='Code', how='left')

glassdoor.head()
# Then replace the 2 digits codes with full name

glassdoor.Country = glassdoor.Name.fillna(glassdoor.Country)

glassdoor.head()
# Finally drop columns that wont be used

glassdoor = glassdoor.drop(['Name', 'Code'], axis=1)

glassdoor.head()
# After doing this, there were still some countries that didn't match official nomenclature

# We need to remove them

listings_before = glassdoor.shape[0]



# Now we merge two dataframes by Country and Name

glassdoor = pd.merge(glassdoor, country_codes, left_on='Country', right_on='Name', how='left')

glassdoor.head()
# And remove rows were Name is NaN

glassdoor.dropna(subset=['Name'], inplace=True)

glassdoor = glassdoor.drop(['Name', 'Code'], axis=1)

listings_after = glassdoor.shape[0]

print('After removing countries names that don\'t match official nomenclature there' \

      f'were left {listings_after} job listings.')
# Now we focus on job titles. We need to find the specific terms we are interested for this study

listings_before = glassdoor.shape[0]



# List of job titles that are interesting for this study

job_titles = ['data scientist', 'software engineer', 'data analyst', 'research scientist', 'business analyst',

              'data engineer', 'statistician', 'dba', 'database engineer', 'machine learning engineer']



# Creating masks for each job title to identify where they appear

job_masks = [glassdoor.JobTitle.str.contains(job_title, flags=re.IGNORECASE, regex=True) for job_title in job_titles]

# Combining all masks where any value is True, return True

combined_mask = np.vstack(job_masks).any(axis=0)

combined_mask
# Applying the mask to the dataset

glassdoor = glassdoor[combined_mask].reset_index(drop=True)

listings_after = glassdoor.shape[0]

print(f'After removing Job Titles that don\'t match answers from question 5 there were left {listings_after} job listings.')

glassdoor.head()
# Now lets clean job titles even further and remove any word that doesn't match the terms from Kaggle survey question 5

job_titles_regex = '|'.join(job_titles)

glassdoor.JobTitle = glassdoor.JobTitle.str.findall(job_titles_regex, flags=re.IGNORECASE)

glassdoor.JobTitle = glassdoor.JobTitle.str[0]

glassdoor.JobTitle = glassdoor.JobTitle.str.title()

glassdoor.head()
# The same way we did for kaggle, we need to group DBA + Data Engineer

glassdoor.JobTitle = glassdoor.JobTitle.replace('Dba', 'Data Engineer/DBA')

glassdoor.JobTitle = glassdoor.JobTitle.replace('Database Engineer', 'Data Engineer/DBA')

glassdoor.JobTitle = glassdoor.JobTitle.replace('Data Engineer', 'Data Engineer/DBA')



# And group Statistician + Research Scientist

glassdoor.JobTitle = glassdoor.JobTitle.replace('Statistician', 'Statistician/Research Scientist')

glassdoor.JobTitle = glassdoor.JobTitle.replace('Research Scientist', 'Statistician/Research Scientist')



glassdoor.JobTitle.value_counts()
# Finally, we transform JobTitle column into category same way we did for Kaggle

job_titles = ['Business Analyst', 'Data Analyst', 'Data Scientist', 

              'Data Engineer/DBA', 'Software Engineer', 'Statistician/Research Scientist']

cat_dtype = pd.api.types.CategoricalDtype(categories=job_titles, ordered=True)

glassdoor.JobTitle = glassdoor.JobTitle.astype(cat_dtype)
# Add column to make groupby easier

glassdoor['Count'] = 1
# Now lets work with Jod Descriptions

# first, we will lowercase everything

glassdoor.JobDescription = glassdoor.JobDescription.str.lower()
# We need to find mentions to cloud platforms into Job Descriptions

# There are multiple ways to mention those, lets create a dictionary 

# to replace them and make standardized terms appear in Job Description

cloud_platforms = {

    'Alibaba': ' Alibaba Cloud ', 

    'Amazon Web Services': ' Amazon Web Services (AWS) ',

    'AWS': ' Amazon Web Services (AWS) ',

    'Google Cloud Platform': ' Google Cloud Platform (GCP) ', 

    'GCP': ' Google Cloud Platform (GCP) ',

    'Google Cloud': ' Google Cloud Platform (GCP) ',

    'IBM': ' IBM Cloud ', 

    'Azure': ' Microsoft Azure ', 

    'Oracle': ' Oracle Cloud ',

    'Red Hat': ' Red Hat Cloud ',

    'SAP': ' SAP Cloud ', 

    'Salesforce': ' Salesforce Cloud ', 

    'VMware': ' VMware Cloud '

}



# Replacing terms into Job Description

for find, repl in cloud_platforms.items():

    glassdoor.JobDescription = glassdoor.JobDescription.str.replace(find.lower(), repl.lower())
# Doing the same for databases

databases ={

    'dynamodb': ' aws dynamodb ',

    'dynamo': ' aws dynamodb ',

    ' rds ': ' aws relational database service ',

    'relational database service': ' aws relational database service ',

    'azure sql': ' azure sql database ',

    'google cloud sql': ' google cloud sql ',

    'microsoft access': ' microsoft access ', 

    'sql server': ' microsoft sql server ', 

    'my sql': ' mysql ', 

    'oracle db': ' oracle database ', 

    'postgres': ' postgressql ',

    'postgre': ' postgressql ',

    'postgre sql': ' postgressql ',

    'sqlite': 'sqlite '

}



for find, repl in databases.items():

    glassdoor.JobDescription = glassdoor.JobDescription.str.replace(find.lower(), repl.lower())
# Now we will create a class to enable plotting radar charts

import plotly.graph_objects as go

import plotly.offline as pyo

import plotly.express as px

from plotly.subplots import make_subplots





# We will create a class to polar plots because we are going to apply the same settings

# for multiple charts. By creating a class we can standardize that and change configs in only one place.

class PolarPlot():



    def __init__(self, title):

        pyo.init_notebook_mode() # allows plotly to show charts on notebook 

        self.figure = go.Figure() # instatiates plotly figure

        self.range = (0, 0) # define the initial range of polar plots

        self.title = title # Saves the chart title as an attribute

        self.theta = ['Business Analyst', 'Data Analyst', 'Data Scientist', 'Data Engineer/DBA',

                      'Software Engineer', 'Statistician/Research Scientist', 'Business Analyst'] # Those are the Theta values for our plot

    

    def update_common_layout(self):

        """

        Updates general layout characteristics

        """

        self.figure.update_layout(

            showlegend = True,

            title_text = self.title, # add title to chart from attribute

            title_font_color = '#333333', # Grey is always better to not draw much attention

            title_font_size = 14,

            legend_font_color = 'grey', # We don't want to draw attention to the legend 

            legend_itemclick = 'toggleothers', # Change the default behaviour, when click select only that trace

            legend_itemdoubleclick = 'toggle', # Change the default behaviour, when double click ommit that trace

            width = 800, # chart size 

            height = 500 # chart size

        )

        

    def update_commom_polar_layout(self):

        """

        Updates polar layout characteristics

        """

        self.figure.update_layout(

            polar_bgcolor='white', # White background is always better

            polar_radialaxis_visible=True, # we want to show the axis

            polar_radialaxis_showticklabels=True, # we want to show the axis titles

            polar_radialaxis_tickfont_color='darkgrey', # grey to the axis label (Software Engineer, Data Scientist, etc)

            polar_angularaxis_color='grey', # Always grey for all elements that are not important

            polar_angularaxis_showline=False, # hide lines that are not necessary

            polar_radialaxis_showline=False, # hide lines that are not necessary

            polar_radialaxis_layer='below traces', # show the axis bellow all traces

            polar_radialaxis_gridcolor='#F2F2F2', # grey to not draw attention

            polar_radialaxis_range=self.range # gets the range attribute, that is calculated in another method

        )

        

    def add_data(self, data, country, color, hover_template='%{r:0.0f}%'):

        """

        Adds a trace to the figure following the same standard for each trace

        """

        highlight = color != 'lightslategrey' # We only want to highlight a few traces, this will decide if a trace is highlighted or not

        data.append(data[0]) # add the first element to the end of the list, this will "close" the polar chart 

        self.figure.add_trace(

            go.Scatterpolar(

                r=data, # Data points 

                theta=self.theta, # Axis (Software Engineer, Data Scientist, etc)

                mode='lines', # plot mode to lines (not good to show markers, it usually adds cluttering to chart)

                name=country, # name to be exibited on legend and on hover

                hoverinfo='name+r', # what to show on hover (name + data point)

                hovertemplate=hover_template, # Format of data point

                line_color=color, # line color

                showlegend=highlight, # will decide if show or not the legend, only shows if color != lightslategrey. Otherwise we would have one legend for each trace. Too much clutter.

                opacity= 0.8 if highlight else 0.25, # If we want to highlight, then oppacity is 0.8. Otherwise it is 0.25, to reduce clutter.

                line_shape='spline', # This will allow smoothing the lines

                line_smoothing=0.8, # How much the lines will smooth

                line_width=1.6 if highlight else 0.6 # we want highlighted traces to be more evident, otherwise they should stay in the background 

            )

        )

        self.update_range(data) # Calls the method that will update the max range

    

    def update_range(self, data):

        """

        Updates the range to be 110% of maximum value of all traces

        """

        max_range = max(data) * 1.1

        self.range = (0, max_range) if max_range > self.range[1] else self.range # updates the range attribute

        

    def show(self):

        """

        Update layouts and shows the figure

        """

        self.update_common_layout() 

        self.update_commom_polar_layout()

        self.figure.show()
# as we have many charts displaying job titles per country, we will create anoter class to handle this

class JobTitlebyCountry:

    

    def __init__(self, country_color, polar_plot):

        self.countries = list(set(kaggle.Country.tolist())) # defines the list of countries as all countries available on Kaggle dataset

        self.country_color = country_color # It is a dictionary of country names and colors

        self.polar = polar_plot # PolarPlot() instance that we have defined before

        

        for country in self.country_color.keys():

            # Here we remove countries that will be highlighted and add then back to the end of the list

            # this way we avoid having non highlighted traces over them

            self.countries.remove(country)

            self.countries.append(country)

    

    def add_traces_percentage(self, data):

        """

        Calculates the percentage of rows for each job title in each country  and then add it to plot

        """

        for country in self.countries:

            # iterates in the list of countries

            data_filtered = data[data.Country == country] # filters data for a single country

            if len(data_filtered) > 20: # only plotting if there are more than 20 datapoints

                color = self.country_color.get(country, 'lightslategrey') # check if country is in the dict of countries and get color, if it is not then color is lightslategrey

                plot_data = data_filtered.groupby(['JobTitle'], as_index=False).Count.sum().Count.tolist() # group the data by jobtitle, perform count and convert it to a list 

                plot_data = (np.array(plot_data) / sum(plot_data) * 100).tolist() # Transform absolute values into percentages

                self.polar.add_data(plot_data, country, color) # add trace to the figure

    

    def add_traces_average_salary(self, data):

        """

        Calculates the average salary for each job title in each country and then add it to plot

        """

        for country in self.countries:

            data_filtered = data[data.Country == country]

            if len(data_filtered) > 20: # only plotting if there are more than 20 datapoints

                color = self.country_color.get(country, 'lightslategrey')

                plot_data = data_filtered.groupby(['JobTitle'], as_index=False).Salary.mean() # Group by job title and calculate average salary

                plot_data.dropna(inplace=True) # drop all nulls

                plot_data = plot_data.Salary.tolist()

                if len(plot_data) == 6: #removes countries that dont have salary for all job titles

                    self.polar.add_data(plot_data, country, color, 'U$%{r:,.2r}') # add trace to figure

          

    def adjust_layout_percentage(self):

        """

        Adjust layout to show percentages

        """

        self.polar.figure.update_layout(

            polar_radialaxis_tickvals=[25, 50, 75], # show ticks ate 25, 50 and 75

            polar_radialaxis_ticktext=['25%', '50%', '75%'],

            polar_radialaxis_tickmode='array',

        )
pp = PolarPlot('Proportionally USA has more Data Scientists, while China has more Software Engineers.' \

               '<br><span style="font-size:10px"><i>Kaggle Survey Q5: Select the title most similar to ' \

               'your current role (sums to 100% for each country)</span></i>') # instantiate PolarPlot, with chart title



us_ch_color = {

    'United States': '#002366', 

    'China': '#ED2124', 

} # creates dict of colors for highlights



plot = JobTitlebyCountry(us_ch_color, pp) # Instantiates the class that will add traces to plot

plot.add_traces_percentage(kaggle) # add traces

plot.adjust_layout_percentage() # Adjust layout to show percentages

plot.polar.show() # show plot
pp = PolarPlot('USA has more demand for Data Analysts and Scientists. China needs more Software Engineers.' \

               '<br><span style="font-size:10px"><i>Glassdoor Job Listings: Percentage of job titles per country ' \

               ' (sums to 100% for each country)</span></i>')



plot = JobTitlebyCountry(us_ch_color, pp)

plot.add_traces_percentage(glassdoor)

plot.adjust_layout_percentage()

plot.polar.show()
pp = PolarPlot('China pays more to Statisticians compared to other positions. USA salaries are the highest.' \

               '<br><span style="font-size:10px"><i>Kaggle Survey Q10: What is your current yearly compensation (approximate $USD)?</span></i>')



plot = JobTitlebyCountry(us_ch_color, pp)

plot.add_traces_average_salary(kaggle)

plot.polar.show()
pp = PolarPlot('India and Brazil have almost the same shape. The latter has a bit more Research Sientists and Statisticians.' \

               '<br><span style="font-size:10px"><i>Kaggle Survey Q5: Sums to 100% for each country</span></i>')



br_in_color = {

    'India': '#FE9933', 

    'Brazil': '#179B3A',

}



plot = JobTitlebyCountry(br_in_color, pp)

plot.add_traces_percentage(kaggle)

plot.adjust_layout_percentage()

plot.polar.show()
pp = PolarPlot('In Brazil there are more space for Software Engineers while in India for Business Analysts.' \

               '<br><span style="font-size:10px"><i>Glassdoor Job Titles: Sums to 100% for each country</span></i>')



plot = JobTitlebyCountry(br_in_color, pp)

plot.add_traces_percentage(glassdoor)

plot.adjust_layout_percentage()

plot.polar.show()
pp = PolarPlot('Salaries are higher in Brazil than in India. Data engineers are well paid in Brazil compared to others.' \

               '<br><span style="font-size:10px"><i>Kaggle Survey: Average salary per country in USD per year</span></i>')



plot = JobTitlebyCountry(br_in_color, pp)

plot.add_traces_average_salary(kaggle)

plot.polar.show()
# Plots proportion of listings for each job title per country

pp = PolarPlot('European countries have almost the same proportion of professionals.' \

               '<br><span style="font-size:10px"><i>Kaggle Survey Q5: Sums to 100% for each country</span></i>')



ge_uk_ne_colors = { 

    'Germany': '#000000',

    'United Kingdom': '#012169',

    'Netherlands': '#F55900'

}



plot = JobTitlebyCountry(ge_uk_ne_colors, pp)

plot.add_traces_percentage(kaggle)

plot.adjust_layout_percentage()

plot.polar.show()
# Plots proportion of listings for each job title per country

pp = PolarPlot('Proportionally there are more positions for Data and Software Engineers in The Netherlands.' \

               '<br><span style="font-size:10px"><i>Glassdoor Job Titles: Sums to 100% for each country</span></i>')



plot = JobTitlebyCountry(ge_uk_ne_colors, pp)

plot.add_traces_percentage(glassdoor)

plot.adjust_layout_percentage()

plot.polar.show()
# Plots average salary for each job title per country

pp = PolarPlot('Business Analysts in the UK and Germany earn more than Engineers.' \

               '<br><span style="font-size:10px"><i>Kaggle Survey: Average salary per country in USD per year</span></i>')



plot = JobTitlebyCountry(ge_uk_ne_colors, pp)

plot.add_traces_average_salary(kaggle)

plot.polar.show()
# Now we want to plot a set of subplots

# to achieve that we are also building a new class that will handle all configuration

class PolarSubPlot():



    def __init__(self, title, subplot_titles):

        pyo.init_notebook_mode()

        self.figure = go.Figure()

        self.range = (0, 0)

        self.title = title

        self.subplot_titles = subplot_titles # List of subplot tiles

        self.theta = ['Business Analyst', 'Data Analyst', 'Data Scientist', 'Data Engineer/DBA',

                      'Software Engineer', 'Statistician/Research Scientist', 'Business Analyst']

        self.subplot_countries = [] # initiate a list of countries to build subplots

        self.make_subplots() # Execute method that will add subplots to figure

        

    def make_subplots(self):

        """

        Creates 6 subplots in the figure and add titles

        """

        self.figure = make_subplots(

            rows=3, # our subplot will have 3 rows

            cols=2, # and 2 columns

            subplot_titles=self.subplot_titles, # Add titles to subplots

            specs=[[{'type': 'polar'}]*2]*3 # Define chart type for each subplot

        )



        for i in self.figure['layout']['annotations']:

            i['font'] = dict(size=16,color='grey')  # Size of subplot title

    

    def update_common_layout(self):

        """

        Updates general layout characteristics

        """

        self.figure.update_layout(

            showlegend = True,

            title_text = self.title,

            title_font_color = '#333333',

            title_font_size = 16,

            margin_t = 150,

            legend_font_color = 'gray',

            legend_itemclick = False,

            legend_itemdoubleclick = False,

            width = 800,

            height = 900

        )

        

    def update_polar_layout(self):

        """

        Updates polar layout characteristics

        """

        # this is a dictionary with polar plot layout configs, similar to what you saw on PolarPlot

        polar_layout = dict(

            bgcolor = 'white',

            radialaxis = dict(

                visible=True,

                showticklabels=False,

                showline=False,

                layer='below traces',

                gridcolor='#F2F2F2',

                tickvals=[25, 50, 75],

                tickmode='array',

                range=self.range,        

            ),

            angularaxis = dict(

                tickfont_size=9,

                color='grey',

                showline=False

            )

        )



        # We need to update the layout for each subplot

        # We will create the same dictionary of layout for each subplot because we want all of them to have the same layout

        subplot_layout = dict()

        for subplot in ['polar', 'polar2', 'polar3', 'polar4', 'polar5', 'polar6']:

             subplot_layout[subplot] = polar_layout

        

        self.figure.update_layout(subplot_layout)

        

    def add_data(self, data, country, source, x, y, hover_template='%{r:0.0f}%'):

        """

        Adds a trace to the figure following the same standard for each trace

        """

        data.append(data[0])

        self.figure.add_trace(

            go.Scatterpolar(

                r=data,

                theta=self.theta,

                mode='lines',

                name=source,

                hoverinfo='name+r',

                hovertemplate='%{r:0.0f}%',

                text=country,

                line_color= '#1BAB40' if source == 'Glassdoor' else '#27A4D7', # Green for Glassdoor, blue for Kaggle

                showlegend= len(self.subplot_countries) < 2,

                opacity= 0.8,

                line_shape='spline',

                line_smoothing=0.8,

                line_width=1.5

            ), x, y)

        self.update_range(data)

        self.subplot_countries.append(country)

    

    def update_range(self, data):

        """

        Updates the range to be 110% of maximum value of all traces

        """

        max_range = max(data) * 1.1

        self.range = (0, max_range) if max_range > self.range[1] else self.range

        

    def show(self):

        self.update_common_layout()

        self.update_polar_layout()

        self.figure.show()
# Plots Kaggle vs Glassdoor proportion of listings for each job title for 4 countries

countries = ['Brazil', 'China', 'India', 'United States', 'Netherlands', 'United Kingdom']

pp = PolarSubPlot('Comparing Job Titles of Kagglers and jobs listings per country.' \

                  '<br>Jobs listings are not exactly available in the same job titles as Kagglers are.'\

                  '<br><span style="font-size:10px"><i>Kaggle Survey vs Glassdoor Listings: '\

                  'Percentage of Job Titles (sums to 100% for each country)</span></i>',

                  subplot_titles = countries)

   

i = 1 # rows

j = 1 # columns

for country in countries:

    # Add Kaggle trace

    kaggle_filtered = kaggle[kaggle.Country == country] 

    plot_data = kaggle_filtered.groupby(['JobTitle'], as_index=False).Count.sum().Count.tolist()

    plot_data = (np.array(plot_data) / sum(plot_data) * 100).tolist()

    pp.add_data(plot_data, country, 'Kaggle', i, j)

    

    # Add Glassdoor trace

    glassdoor_filtered = glassdoor[glassdoor.Country == country] 

    plot_data = glassdoor_filtered.groupby(['JobTitle'], as_index=False).Count.sum().Count.tolist()

    plot_data = (np.array(plot_data) / sum(plot_data) * 100).tolist()

    pp.add_data(plot_data, country, 'Glassdoor', i, j)

  

    # Chart Positioning

    if j >= 2:

        i += 1

        j = 1

    else:

        j += 1



pp.show()
# Now we create a class to make line plots easier, same logic as before

class LinePlot():



    def __init__(self, title):

        pyo.init_notebook_mode()

        self.figure = go.Figure()

        self.range = (0, 100)

        self.title = title

    

    def update_axis_title(self, x, y):

        self.figure.update_layout(

            xaxis_title_text=x,

            yaxis_title_text=y,

        )

        

    def update_layout(self):

        """

        Creates a clean layout for ploting, adjusting multiple settings

        """

        self.figure.update_layout(

            showlegend=True,

            title_text=self.title,

            title_font_color='#333333',

            legend_font_color='gray',

            legend_itemclick='toggleothers',

            legend_itemdoubleclick='toggle',

            width = 800,

            height=500,

            plot_bgcolor='white',

            xaxis_title_font_color='grey',

            xaxis_color='grey',

            yaxis_title_font_color='grey',

            yaxis_color='grey',

        )

         

    def add_data(self, x_names, y_data, trace_name, trace_color, hover_template):

        """

        Adds a trace to the figure following the same standard for each trace

        """

        highlight = trace_color != 'lightslategrey'

        self.figure.add_trace(

            go.Scatter(

                x=x_names,

                y=y_data,

                mode='lines',

                name=trace_name,

                hoverinfo='name+y',

                hovertemplate=hover_template,

                line_color= trace_color,

                showlegend=highlight,

                opacity= 0.8 if highlight else 0.25,

                line_shape='spline',

                line_smoothing=0.8,

                line_width=1.6 if highlight else 0.5

            )

        )

        

    def show(self):

        self.update_layout()

        self.figure.show()
# This function will be used to create different aggegations for plotting

def plot_lines(line_plot, data, traces, x_names, agg_column, group_column, trace_column, trace_colors, hover_template):

    """

    Creates aggregation to plot

    """

    for trace_name in traces:

        color = trace_colors.get(trace_name, 'lightslategrey') # Get color if trace_name in trace_colors

        data_filtered = data[data[trace_column] == trace_name] # Filter data by trace_name

        plot_data = data_filtered.groupby([group_column], as_index=False).agg({agg_column: ['mean', 'count']}) # Group by group_column and calculate mean and count for agg column

        plot_data = plot_data[agg_column]['mean'].tolist() # convert mean column to list

        line_plot.add_data(x_names, plot_data, trace_name, color, hover_template=hover_template) # add trace to line plot
traces = ['Business Analyst', 'Data Analyst', 'Data Scientist', 'Data Engineer/DBA',

          'Software Engineer', 'Statistician/Research Scientist'] # defines traces (each line)



x_names = ['0 years', '< 1 years', '1-2 years', '3-5 years', '5-10 years', '10-20 years', '20+ years'] # define x axis



trace_colors = {

    'Data Engineer/DBA': '#212F3C',

    'Data Scientist': '#196F3D', 

    'Business Analyst': '#21618C'

} # Define traces that will be highlighted



# Instantiate LinePlot with title

lp = LinePlot('We see a decrease in salary for DS, DE and BA who have just started coding' \

               '<br><span style="font-size:10px"><i>Kaggle Survey Q15: Average salary per years writting code in USD per year</span></i>') 



# call plot lines function

plot_lines(

    lp, # plot a LinePlot

    data=kaggle, # with Kaggle Data

    traces=traces, # With a given list of traces

    x_names=x_names, # and x axis

    agg_column='Salary', # Calculate the aggregation on Salary

    group_column='TimeWritingCode', # Group by TimeWritting Code

    trace_column='JobTitle', # Column that contains traces

    trace_colors=trace_colors, # Dict of traces to highlight

    hover_template='U$%{y:,.2r}' # Number template to show on hover

)



xaxis_title='How long have you been writing code to analyze data (at work or at school)?'

yaxis_title='Average Salary (USD per Year)'



lp.update_axis_title(xaxis_title, yaxis_title) # Update x, and y axis titles

lp.show()


lp = LinePlot('Professionals with no coding experience are older than those with little experience.' \

               '<br><span style="font-size:10px"><i>Kaggle Survey Q1: Average age per years writting code</span></i>')



traces = ['Business Analyst', 'Data Analyst', 'Data Scientist', 'Data Engineer/DBA',

          'Software Engineer', 'Statistician/Research Scientist']



x_names = ['0 years', '< 1 years', '1-2 years', '3-5 years', '5-10 years', '10-20 years', '20+ years']



trace_colors = {

    'Data Engineer/DBA': '#212F3C',

    'Data Scientist': '#196F3D', 

    'Business Analyst': '#21618C'

}



plot_lines(

    lp, 

    data=kaggle,

    traces=traces, 

    x_names=x_names, 

    agg_column='Age', 

    group_column='TimeWritingCode',

    trace_column='JobTitle', 

    trace_colors=trace_colors,

    hover_template='%{y:0.0f}'

)



xaxis_title='How long have you been writing code to analyze data (at work or at school)?'

yaxis_title='Average age in years'

lp.update_axis_title(xaxis_title, yaxis_title)

lp.show()

lp = LinePlot('Statisticians holds the greatest increase in salary when moving to bigger companies.' \

               '<br><span style="font-size:10px"><i>Kaggle Survey Q6: Average salary per company size in USD per year</span></i>')



traces = ['Business Analyst', 'Data Analyst', 'Data Scientist', 'Data Engineer/DBA',

          'Software Engineer', 'Statistician/Research Scientist']



x_names = ['0-49 employees', '50-249 employees', '250-999 employees', '1000-9,999 employees', '> 10,000 employees']



trace_colors = {

    'Data Scientist': '#196F3D', 

    'Business Analyst': '#21618C',

    'Statistician/Research Scientist': '#5B2C6F'

}



plot_lines(

    lp, 

    data=kaggle,

    traces=traces, 

    x_names=x_names, 

    agg_column='Salary', 

    group_column='CompanySize',

    trace_column='JobTitle', 

    trace_colors=trace_colors,

    hover_template='U$%{y:,.2r}'

)



xaxis_title='What is the size of the company where you are employed?'

yaxis_title='Average Salary (USD per Year)'

lp.update_axis_title(xaxis_title, yaxis_title)

lp.show()
# Now we create a function to plot some more polar data

def plot_polar(polar_plot, data, traces, x_names, agg_column, group_column, trace_column, trace_colors, hover_template):

    """

    Creates aggregation to plot

    """

    data_cp = data.copy() # make copy of dataframe to avoid modifying the original

    for trace_name in traces: # for each trace

        color = trace_colors.get(trace_name.strip(), 'lightslategrey') # Check if trace will be highlighted

        if agg_column in ('JobDescription', 'CloudPlatf'): # those columns are in lowercase

            data_cp['TempCol'] = data_cp[agg_column].apply(lambda x: trace_name.lower() in x) # Find rows where trace_name is in agg_column

        else: # Other columns are not lowercase

            data_cp['TempCol'] = data_cp[agg_column].apply(lambda x: trace_name in x) # Find rows where trace_name is in agg_column

        plot_data = data_cp.groupby([group_column], as_index=False).agg({'TempCol': ['sum', 'count']}) # Group by group_column, aggegate on TempCol calculating sum (of boolean) and count

        plot_data['TempColPct'] = plot_data['TempCol']['sum'] / plot_data['TempCol']['count'] * 100 # Transform absolute values into percentages

        plot_data = plot_data.TempColPct.tolist() # Convert to list

        polar_plot.add_data(plot_data, trace_name, color, hover_template) # add trace
pp = PolarPlot('About 3/4 of Kagglers use Python. R is not popular within Software Engineers. ' \

               '<br><span style="font-size:10px"><i>Kaggle Survey Q18: What programming languages do you use on a regular basis?</span></i>')





traces = ['Bash', 'C', 'C++', 'Java', 'Javascript', 'MATLAB', 

          'Other', 'Python', 'R', 'SQL', 'TypeScript'] 



x_names = ['Business Analyst', 'Data Analyst', 'Data Scientist', 'Data Engineer/DBA',

          'Software Engineer', 'Statistician/Research Scientist']



trace_colors = {

    'Python': '#FEC331',

    'SQL': '#66B900',

    'R': '#2063b7',

}



plot_polar(

    pp, 

    data=kaggle,

    traces=traces, 

    x_names=x_names, 

    agg_column='ProgLang', 

    group_column='JobTitle',

    trace_column='ProgLang', 

    trace_colors=trace_colors,

    hover_template='%{r:0.0f}%'

)



pp.figure.update_layout(

    polar_radialaxis_tickvals=[25, 50, 75],

    polar_radialaxis_ticktext=['25%', '50%', '75%'],

    polar_radialaxis_tickmode='array',

)

pp.show()

pp = PolarPlot('R is not a frequent requirements in Jobs Listings. <b>Even for Statisticians.</b>' \

               '<br><span style="font-size:10px"><i>Glassdoor: proportion of programming language appearances in Job Description per Job Title</span></i>')



traces = ['Bash', ' C ', 'C++', 'Java', 'Javascript', 'MATLAB', 

          'Python', ' R ', 'SQL', 'TypeScript'] 



x_names = ['Business Analyst', 'Data Analyst', 'Data Scientist', 'Data Engineer/DBA',

          'Software Engineer', 'Statistician/Research Scientist']



trace_colors = {

    'Python': '#FEC331',

    'SQL': '#66B900',

    'R': '#2063b7',

}



plot_polar(

    pp, 

    data=glassdoor,

    traces=traces, 

    x_names=x_names, 

    agg_column='JobDescription', 

    group_column='JobTitle',

    trace_column='JobDescription', 

    trace_colors=trace_colors,

    hover_template='%{r:0.0f}%'

)



pp.figure.update_layout(

    polar_radialaxis_tickvals=[25, 50, 75],

    polar_radialaxis_ticktext=['25%', '50%', '75%'],

    polar_radialaxis_tickmode='array',

)

pp.show()
# Plots Kaggle vs Glassdoor programming languages percentages of appearances

programming_languages = ['Python', ' R ', 'SQL', 'Bash', 'Java', 'Javascript']

pp = PolarSubPlot('Comparing programming language knowledge of Kagglers and requirements of jobs.' \

                '<br><span style="font-size:10px"><i>Kaggle Survey vs Glassdoor Listings: Percentage '\

                'of job listing with a given language in description vs respondents who use that language</span></i>',

                  subplot_titles = programming_languages)



kaggle_cp = kaggle.copy()

glassdoor_cp = glassdoor.copy()





i = 1

j = 1

for language in programming_languages:

    # Add Kaggle trace

    kaggle_cp['Language'] = kaggle_cp.ProgLang.apply(lambda x: language.strip() in x)

    plot_data = kaggle_cp.groupby('JobTitle', as_index=False).agg({'Language': ['sum', 'count']})

    plot_data['LanguagePct'] = plot_data['Language']['sum'] / plot_data['Language']['count'] * 100

    plot_data = plot_data.LanguagePct.tolist()

    pp.add_data(plot_data, language.strip(), 'Kaggle', i, j)

    

    # Add Glassdoor trace

    glassdoor_cp['Language'] = glassdoor_cp.JobDescription.apply(lambda x: language.lower() in x)

    plot_data = glassdoor_cp.groupby('JobTitle', as_index=False).agg({'Language': ['sum', 'count']})

    plot_data['LanguagePct'] = plot_data['Language']['sum'] / plot_data['Language']['count'] * 100

    plot_data = plot_data.LanguagePct.tolist()

    pp.add_data(plot_data, language.strip(), 'Glassdoor', i, j)

  

    # Chart Positioning

    if j >= 2:

        i += 1

        j = 1

    else:

        j += 1



pp.show()


lp = LinePlot('Learning up to three languages will increase your salary!' \

               '<br><span style="font-size:10px"><i>Kaggle Survey Q18: Quantity of languages used on a regular basis per years writting code</span></i>')



traces = list(set(kaggle.TimeWritingCode.tolist()))



x_names = ['{} languages'.format(x) for x in range(7)]



trace_colors = {}



plot_lines(

    lp, 

    data=kaggle,

    traces=traces, 

    x_names=x_names, 

    agg_column='Salary', 

    group_column='QtyProgLang',

    trace_column='TimeWritingCode', 

    trace_colors=trace_colors,

    hover_template='U$%{y:,.2r}'

)



# Adding Averarage

plot_data = kaggle.groupby(['QtyProgLang'], as_index=False).agg({'Salary': 'mean'})

plot_data = plot_data.Salary.tolist()

lp.add_data(x_names, plot_data, 'Average', 'black', hover_template='U$%{y:,.2r}')





xaxis_title='Quantity of programming languages used on a regular basis'

yaxis_title='Average Salary (USD per Year)'

lp.update_axis_title(xaxis_title, yaxis_title)

lp.show()
pp = PolarPlot('Cloud platforms are more popular within technical positions. AWS is leading whithing Kagglers.' \

               '<br><span style="font-size:10px"><i>Kaggle Survey Q29: Which of the following cloud computing platforms do you use on a regular basis?</span></i>')



traces = ['Alibaba Cloud', 'Amazon Web Services (AWS)', 'Google Cloud Platform (GCP)',

          'IBM Cloud', 'Microsoft Azure', 'Oracle Cloud', 'Red Hat Cloud',

          'SAP Cloud', 'Salesforce Cloud', 'VMware Cloud', 'None']



x_names = ['Business Analyst', 'Data Analyst', 'Data Scientist', 'Data Engineer/DBA',

          'Software Engineer', 'Statistician/Research Scientist']



trace_colors = {

    'Amazon Web Services (AWS)': '#F79500',

    'Google Cloud Platform (GCP)': '#1AA746',

    'Microsoft Azure': '#3278B1',

}



plot_polar(

    pp, 

    data=kaggle,

    traces=traces, 

    x_names=x_names, 

    agg_column='CloudPlatf', 

    group_column='JobTitle',

    trace_column='CloudPlatf', 

    trace_colors=trace_colors,

    hover_template='%{r:0.0f}%'

)



pp.figure.update_layout(

    polar_radialaxis_tickvals=[10, 20, 30],

    polar_radialaxis_ticktext=['10%', '20%', '30%'],

    polar_radialaxis_tickmode='array',

)

pp.show()

pp = PolarPlot('Data engineering is the field which requires more experience with cloud computing products.' \

               '<br><span style="font-size:10px"><i>Glassdoor: percentage of cloud computing platform appearances in Job Descriptions</span></i>')



traces = ['Alibaba Cloud', 'Amazon Web Services (AWS)', 'Google Cloud Platform (GCP)',

          'IBM Cloud', 'Microsoft Azure', 'Oracle Cloud', 'Red Hat Cloud',

          'SAP Cloud', 'Salesforce Cloud', 'VMware Cloud', 'None']



x_names = ['Business Analyst', 'Data Analyst', 'Data Scientist', 'Data Engineer/DBA',

          'Software Engineer', 'Statistician/Research Scientist']



trace_colors = {

    'Amazon Web Services (AWS)': '#F79500',

    'Google Cloud Platform (GCP)': '#1AA746',

    'Microsoft Azure': '#3278B1',

}



plot_polar(

    pp, 

    data=glassdoor,

    traces=traces, 

    x_names=x_names, 

    agg_column='JobDescription', 

    group_column='JobTitle',

    trace_column='CloudPlatf', 

    trace_colors=trace_colors,

    hover_template='%{r:0.0f}%'

)



pp.figure.update_layout(

    polar_radialaxis_tickvals=[10, 20, 30],

    polar_radialaxis_ticktext=['10%', '20%', '30%'],

    polar_radialaxis_tickmode='array',

)

pp.show()

pp = PolarPlot('The most used DBs by Kagglers are MySQL, PostgreSQL and SQL Server.' \

               '<br><span style="font-size:10px"><i>Kaggle Survey Q34: Which of the following relational database products do you use on a regular basis?</span></i>')



traces =  ['aws dynamodb', 'aws relational database service', 'azure sql database', 'google cloud sql',

           'microsoft access', 'microsoft sql server', 'mysql', 'oracle database', 'postgressql', 'sqlite', 'none']



x_names = ['Business Analyst', 'Data Analyst', 'Data Scientist', 'Data Engineer/DBA',

          'Software Engineer', 'Statistician/Research Scientist']



trace_colors = {

    'mysql': '#DE8A01',

    'postgressql': '#32648D',

    'microsoft sql server': '#A41C22',

}



plot_polar(

    pp, 

    data=kaggle,

    traces=traces, 

    x_names=x_names, 

    agg_column='Databases', 

    group_column='JobTitle',

    trace_column='Databases', 

    trace_colors=trace_colors,

    hover_template='%{r:0.0f}%'

)



pp.figure.update_layout(

    polar_radialaxis_tickvals=[10, 20, 30],

    polar_radialaxis_ticktext=['10%', '20%', '30%'],

    polar_radialaxis_tickmode='array',

)

pp.show()

pp = PolarPlot('Notice that many Statisticians and Research Scientists don\'t use DBs.' \

               '<br><span style="font-size:10px"><i>Kaggle Survey Q34: Which of the ' \

               'following relational database products do you use on a regular basis?</span></i>')



traces =  ['aws dynamodb', 'aws relational database service', 'azure sql database', 'google cloud sql',

           'microsoft access', 'microsoft sql server', 'mysql', 'oracle database', 'postgressql', 'sqlite', 'none']



x_names = ['Business Analyst', 'Data Analyst', 'Data Scientist', 'Data Engineer/DBA',

          'Software Engineer', 'Statistician/Research Scientist']



trace_colors = {

    'none': 'grey'

}



plot_polar(

    pp, 

    data=kaggle,

    traces=traces, 

    x_names=x_names, 

    agg_column='Databases', 

    group_column='JobTitle',

    trace_column='Databases', 

    trace_colors=trace_colors,

    hover_template='%{r:0.0f}%'

)



pp.figure.update_layout(

    polar_radialaxis_tickvals=[10, 20, 30],

    polar_radialaxis_ticktext=['10%', '20%', '30%'],

    polar_radialaxis_tickmode='array',

)

pp.show()

pp = PolarPlot('Databases is heavy on Data and Software Engineering. Some Data Analyst positions require SQL Server.' \

               '<br><span style="font-size:10px"><i>Glassdoor: percentage of database names appearances in Job Descriptions</span></i>')



traces =  ['aws dynamodb', 'aws relational database service', 'azure sql database', 'google cloud sql',

           'microsoft access', 'microsoft sql server', 'mysql', 'oracle database', 'postgressql', 'sqlite', 'none']



x_names = ['Business Analyst', 'Data Analyst', 'Data Scientist', 'Data Engineer/DBA',

          'Software Engineer', 'Statistician/Research Scientist']



trace_colors = {

    'mysql': '#DE8A01',

    'postgressql': '#32648D',

    'microsoft sql server': '#A41C22',

}



plot_polar(

    pp, 

    data=glassdoor,

    traces=traces, 

    x_names=x_names, 

    agg_column='JobDescription', 

    group_column='JobTitle',

    trace_column='Databases', 

    trace_colors=trace_colors,

    hover_template='%{r:0.0f}%'

)



pp.figure.update_layout(

    polar_radialaxis_tickvals=[5, 10, 15],

    polar_radialaxis_ticktext=['5%', '10%', '15%'],

    polar_radialaxis_tickmode='array',

)

pp.show()