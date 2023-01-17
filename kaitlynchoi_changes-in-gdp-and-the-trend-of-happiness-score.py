# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



# To enable Jupyter and matplotlib work together effectively and to have the graphs displayed inside the notebook

%matplotlib inline   



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



sns.set(style="white")



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Read all five CSV files

file_2015 = pd.read_csv("/kaggle/input/world-happiness/2015.csv")

file_2016 = pd.read_csv("/kaggle/input/world-happiness/2016.csv")

file_2017 = pd.read_csv("/kaggle/input/world-happiness/2017.csv")

file_2018 = pd.read_csv("/kaggle/input/world-happiness/2018.csv")

file_2019 = pd.read_csv("/kaggle/input/world-happiness/2019.csv")
# Create a single dataframe that shows all the columns from the five datasets.

columns_2015 = list(file_2015.columns)

columns_2016 = list(file_2016.columns)

columns_2017 = list(file_2017.columns)

columns_2018 = list(file_2018.columns)

columns_2019 = list(file_2019.columns)



list_columns = [columns_2015,columns_2016,columns_2017,columns_2018, columns_2019]



df_columns_info = pd.DataFrame(list_columns, index=["2015","2016","2017","2018","2019"])

df_columns_info
# Change the column names so that they are consistent across different datasets.

file_2015=file_2015.rename({"Economy (GDP per Capita)":"GDP per capita","Health (Life Expectancy)":"Life Expectancy","Trust (Government Corruption)":"Government Corruption"},axis='columns')

file_2016=file_2016.rename({"Economy (GDP per Capita)":"GDP per capita","Health (Life Expectancy)":"Life Expectancy","Trust (Government Corruption)":"Government Corruption"},axis='columns')

file_2017=file_2017.rename({"Happiness.Rank":"Happiness Rank","Happiness.Score":"Happiness Score","Economy..GDP.per.Capita.":"GDP per capita","Health..Life.Expectancy.":"Life Expectancy","Trust..Government.Corruption.":"Government Corruption"},axis='columns')

file_2018=file_2018.rename({"Overall rank":"Happiness Rank","Country or region":"Country","Score":"Happiness Score","Healthy life expectancy":"Life Expectancy","Freedom to make life choices":"Freedom","Perceptions of corruption":"Government Corruption"},axis='columns')

file_2019=file_2019.rename({"Overall rank":"Happiness Rank","Country or region":"Country","Score":"Happiness Score","Healthy life expectancy":"Life Expectancy","Freedom to make life choices":"Freedom","Perceptions of corruption":"Government Corruption"},axis='columns')
# Set the "Country" as an index for every file

world_happiness_files =[file_2015,file_2016,file_2017,file_2018,file_2019]



# Set the Country column as index

def country_set_index(files):

    for n in range(len(files)):

        files[n]=files[n].set_index("Country")

    

    return files



world_happiness_files=country_set_index(world_happiness_files)
# Create a dataframe that shows the values of a given factor from 2015 to 2019 for all countries surveyed during this period of time.

def data_15to19(files, factor):

    new_df=files[0][[factor]]

    new_df=new_df.rename({factor:"2015"},axis='columns')

    country_index=new_df.index.tolist()

    

    year=["2015","2016","2017","2018","2019"]

    

    for n in range(1,5):

        new_column=files[n].loc[files[n].index.intersection(country_index), factor]  # Check https://pandas-docs.github.io/pandas-docs-travis/user_guide/indexing.html#indexing-deprecate-loc-reindex-listlike

        new_df[year[n]]=new_column

    

    new_df=new_df.dropna()

    new_df=new_df.astype(float)   # Convert strings to float for any future calculation/plotting

    

    return new_df





# Make a dataframe showing the happiness scores of the countries from 2015 to 2019

scores=data_15to19(world_happiness_files, "Happiness Score")

scores.head()
# Make a dataframe that shows a change from the previous year



def change_from_pre_year(df):

    year=["2015","2016","2017","2018","2019"]

    df=df.reset_index()

    new_df=pd.DataFrame()

    new_df['Country'] = df["Country"]

    

    for n in range(4):     # range(4), not range(5), because the last calculation is to substract 2018 data from 2019.

        next_year = n+1

        new_df[year[next_year]]=df[year[next_year]] - df[year[n]]



    new_df=new_df.set_index("Country")

    new_df.columns=["2015-2016", "2016-2017", "2017-2018", "2018-2019"]

    new_df["Average"]=new_df.mean(numeric_only=True, axis=1)



    # Sort the dataframe by the Average column

    new_df=new_df.sort_values(by=["Average"], ascending=False)

        

    return new_df   



# Create a dataframe showing the annual changes of the happiness score for all the countries

score_change=change_from_pre_year(scores)

score_change.head()
# Create a dataframe with the countries where the change in a given factor is above a certain point.

def went_up(df, points):

    new_df = df[(df["2015-2016"]>=points) &(df["2016-2017"]>=points) &(df["2017-2018"]>=points) &(df["2018-2019"]>=points)]

    

    return new_df



# Find the countries that improved the happiness score (changes > 0)

improved_countries_score=went_up(score_change, 0)
# Countries of my interest

select_countries_improved=improved_countries_score.iloc[:15].index.tolist()



# Plot how the scores changed for the group of countries I selected for further analysis

score_trend=scores.loc[select_countries_improved]



def multi_lineplot(df, x_label,y_label,title):

    

    df=df.T

    plt.figure(figsize=(8,8))



    #dashes=False because The problem is that lineplot uses dashed lines for the second and any further column 

    #but the standard style only supports dashes for the first 6 colors.

    sns.lineplot(data=df, dashes=False)

    plt.legend(bbox_to_anchor=(1.04,0.5), loc="center left")

    plt.xlabel(x_label)

    plt.ylabel(y_label)

    plt.title(title)



    # Placing the legend help: https://stackoverflow.com/questions/4700614/how-to-put-the-legend-out-of-the-plot/43439132#43439132



multi_lineplot(score_trend, "Year", "Happiness Score", "Top 15 countries that constantly improved their happiness scores")
# Create a dataframe with the countries where the change in a given factor is below a certain point.

def went_down(df, points):

    new_df = df[(df["2015-2016"]<=points) &(df["2016-2017"]<=points) &(df["2017-2018"]<=points) &(df["2018-2019"]<=points)]

    

    return new_df



# Obtain a list of 15 worst performing countries

declined_countries_score=went_down(score_change, 0)

declined_countries_list=declined_countries_score.iloc[-15:].index.tolist()



# Plot the happiness score trends of the worst performing countries

score_trend_2=scores.loc[declined_countries_list]

multi_lineplot(score_trend_2, "Year", "Happiness Score", "Bottom 15 countries whose happiness scores constantly decreased over time")
# Create a list of the factors (or columns) that were measured in all five yaers

cols=world_happiness_files[3].columns.tolist()

cols.remove("Social support")    # "Social support" was not measured in 2015, 2016, 2017

standard_columns=cols
# Create a dataframe that I could use to create scatter plots to determine the relationship between changes of a factor and changes of happiness scores.

def changes_for_select_countries(file_list, country_list, column_list):

    df=pd.DataFrame()

    for n in range(len(file_list)-1):

        current_year=file_list[n].loc[country_list, column_list]

        next_year=file_list[n+1].loc[country_list, column_list]

        new_df=next_year.subtract(current_year)

        df=df.append(new_df)



    # Rename the columns since they are showing the change between two years

    new_cols=[]

    for n in range(len(column_list)):

        new_col = "delta "+column_list[n]

        new_cols.append(new_col)

    

    df.columns=new_cols  

        

    # Create a Year column

    year=["2015-2016","2016-2017","2017-2018","2018-2019"]

    year=year*len(country_list)

    year.sort()

    df["Year"] = year

    

    # Sort the dataframe by index and then Year

    df.index.name="Country"

    df=df.sort_values(by=["Country","Year"],ascending = [True, True])

 

    return df  
# Make a dataframe that shows R^2 values for each scatter plot

from scipy import stats



def r_squared_table(df,factors):

    new_df=pd.DataFrame()

    new_df[""]=["R_squared"]

    new_df=new_df.set_index("")

    for n in range(len(factors)):

        x=df[factors[n]]

        y=df["delta Happiness Score"]

        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

        r_squared=r_value**2

        new_df[factors[n]]=[r_squared]

    return new_df
# Make scatter plots to show the relationship between each of the factors and happiness score. Show R^2 as well.

def five_factors_scatter_plot(df, fig_title):

    five_factors=["delta GDP per capita","delta Life Expectancy","delta Freedom","delta Generosity","delta Government Corruption"]

    sns.pairplot(df, x_vars=five_factors, y_vars="delta Happiness Score", hue="Year")

    plt.suptitle(fig_title, y=1.08)

    plt.show()

    

    # Show R^2 

    reg_table=r_squared_table(df,five_factors)

    

    return reg_table
# For 15 best performing countries:

# Get all the information about the select countries from all five datasets

change_info_improved_countries=changes_for_select_countries(world_happiness_files, select_countries_improved, standard_columns)

five_factors_scatter_plot(change_info_improved_countries, "Top 15 countries that constantly improved their happiness scores")
# For 15 best performing countries:

# Get all the information about the select countries from all five datasets

change_info_declined_countries=changes_for_select_countries(world_happiness_files, declined_countries_list, standard_columns)

five_factors_scatter_plot(change_info_declined_countries, "Bottom 15 countries whose happiness scores constantly decreased over time")
# Obtain a list of countries that appeared in all five datasets 

gdp=data_15to19(world_happiness_files, "GDP per capita")

all_countries=gdp.index.tolist()



# Plot changes in GDP vs. changes in Happiness Score and changes in life expectancy vs. changes in Happiness Score for all countries

column_interest=["GDP per capita", "Life Expectancy", "Happiness Score"]

change_info_all_countries=changes_for_select_countries(world_happiness_files, all_countries, column_interest)

two_factors=["delta GDP per capita","delta Life Expectancy"]

sns.pairplot(change_info_all_countries, x_vars=two_factors, y_vars="delta Happiness Score", hue="Year", height=2, aspect=4)

plt.suptitle("Countries included in all five reports", y=1.08)