# Load necessary libraries

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# In order to have the output of plotting function printed to the notebook

%matplotlib inline 



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
main_df=pd.read_csv("/kaggle/input/suicide-rates-overview-1985-to-2016/master.csv")

main_df.head(10)
# Create a summary table for the dataframe



def summary_df(df):

    print(f"Dataset Shape: {df.shape}")

    summary = pd.DataFrame(df.dtypes,columns=['dtypes'])

    summary = summary.reset_index()

    summary['Name'] = summary['index']

    summary = summary[['Name','dtypes']]

    summary['Missing'] = df.isnull().sum().values    

    summary['Uniques'] = df.nunique().values

    

    return summary
summary_df(main_df)
# How many countries were survayed each year?

year_df=main_df.groupby("year").agg({"country":"nunique"}).reset_index()

fig=plt.figure()

plt.bar(year_df["year"], year_df["country"])

plt.show()
# Find the index of the first row of 1990 data

main_df=main_df.sort_values(by=["year"]).reset_index(drop=True)

idx_list_1990=main_df[main_df["year"]==1990].index.values.tolist()

start_year=min(idx_list_1990)



# Find the index of the last row of 2015 data

idx_list_2015=main_df[main_df["year"]==2015].index.values.tolist()

end_year=max(idx_list_2015)



# Slice the Dataframe so that we have only 1990-2005 data

main_df=main_df.iloc[start_year:end_year].reset_index(drop=True)
# After selecting the 1990-2005 rows out of the original dataset

main_df
# Create a DataFrame that has country name, country code, region, and income group information

class_df=pd.read_csv("/kaggle/input/world-bank-country-and-lending-groups/worldbank_classification.csv")

class_df
# Clean the class_df for future use

class_df=class_df.drop(columns=["x","Lending category","Other"])

class_df=class_df.rename(columns={"Economy": "country","Code":"country code"})

countries_218=class_df["country"].tolist()

class_df=class_df.set_index("country")
# Identify which country names are different between main_df and class_df

main_countries=main_df["country"].unique().tolist()



print("The names of the following countries are different between two dataframes:")

for country in main_countries:

    if country in countries_218:

        continue

    else:

        print (country)
# Make the country names consistent between main_df and class_df

main_df['country'] = main_df['country'].replace("Kyrgyzstan","Kyrgyz Republic")

main_df['country'] = main_df['country'].replace("Macau", "Macao SAR")

main_df['country'] = main_df['country'].replace("Republic of Korea", "South Korea")

main_df['country'] = main_df['country'].replace("Russian Federation", "Russia")

main_df['country'] = main_df['country'].replace("Saint Kitts and Nevis", "St. Kitts and Nevis")

main_df['country'] = main_df['country'].replace("Saint Lucia", "St. Lucia")

main_df['country'] = main_df['country'].replace("Saint Vincent and Grenadines", "St. Vincent and the Grenadines")

main_df['country'] = main_df['country'].replace("Slovakia", "Slovak Republic")
# Grab the category information (e.g., Region, ISO code) from the class_df and add it to the new dataframe

def add_classification(df, classification):

    

    group_labels=class_df[classification]

    left_join=pd.merge(df, group_labels, on="country", how='left') 

    

    return left_join
# Create the suicide rate (per 100,000 population) column

def groupby_add_s_rate(df, groupby):

    new_df=df.groupby(groupby).agg({"suicides_no":"sum", "population":"sum"}).reset_index()

    new_df["suicides/100k pop"]=new_df["suicides_no"]/new_df["population"]*100000

    return new_df
# Create a Choropleth Map for a given year

import plotly.express as px



def world_map_year (year):

    new_df=main_df[main_df["year"]==year]

    new_df=new_df.groupby("country").agg({"suicides_no":"sum", "population":"sum"}).reset_index()

    new_df["suicides/100k pop"] = new_df["suicides_no"]/new_df["population"]*100000

    

    # Add ISO alpha3 country code to dataframe

    df_w_code=add_classification(new_df, "country code")

    

    fig = px.choropleth(df_w_code, 

                    locations="country code",

                    color="suicides/100k pop", 

                    hover_name="country", # column to add to hover information

                    color_continuous_scale=px.colors.sequential.Plasma)

    title = 'Suicide rate (per 100,000 population) in {}'.format(year)

    fig.update_layout(title_text=title)

    fig.show()
def world_map_slider(df, column, map_title):



    # Define the max and min of the color bar

    bar_max=df[column].max()

    bar_min=df[column].min()

    

    # Add ISO alpha3 country code to dataframe

    df_w_code=add_classification(df, "country code")

    

    # Make a choropleth map

    fig = px.choropleth(df_w_code, locations="country code",

                    color=column,

                    hover_name="country",

                    animation_frame="year",

                    title = map_title,

                    range_color=(bar_min, bar_max),   # Without setting this up, the color bar range changes

                   color_continuous_scale=px.colors.sequential.Plasma)

    #fig["layout"].pop("updatemenus")  # Uncomment it if you want to manually drag the slider

    fig.show()
world_map_year(2009)
# Groupby year and country

year_country_df=groupby_add_s_rate(main_df, ["year","country"])



# Make a world map with slider

world_map_slider(year_country_df, "suicides/100k pop", "Suicide rate (per 100,000 population) from 1990 to 2015")
# Total suicide rate over time

s_rate_year=groupby_add_s_rate(main_df, ["year"])



# Plot a graph

fig=plt.figure(figsize=(10,5))

plt.plot( "year", "suicides/100k pop", data=s_rate_year, marker='o', 

         markerfacecolor='white', markersize=6, color='red', linewidth=4)

plt.xlabel("Year")

plt.ylim(0,20)

plt.ylabel("Suicide Rate (per 100,000 population)")

fig.suptitle("Worldwide suicide rate from 1990 to 2015, both sexes", fontsize=15)
import seaborn as sns



all_countries_year=groupby_add_s_rate(main_df, ["year","country"])



def country_stat (country):   

    # Extract the country specific data 

    country_df=main_df[main_df["country"]==country]

    

    # Create a plot 

    fig, axes = plt.subplots(3, figsize=(15,10))

    fig.text(0.08, 0.5, 'Suicide rate (per 100,000 population)', va='center', rotation='vertical')



    # Give some space between subplots

    plt.subplots_adjust(hspace=0.4)

    

    # Top plot: Box plot

    sns.boxplot(y=all_countries_year['suicides/100k pop'], x=all_countries_year['year'], palette="YlGn", ax=axes[0])



    # Top plot: line plot (both sexes). IMPORTANT: Change the x values to STRING b/c sns.boxplot convert x.dtype to string.

    suicide_df=groupby_add_s_rate(country_df, "year")

    sns.lineplot(y=suicide_df['suicides/100k pop'], x=suicide_df['year'].astype('str'), 

                 color='red', linewidth=2.5, ax=axes[0])

    axes[0].set(ylabel=None)

    axes[0].set(xlabel=None)

    axes[0].set_title("Box plot: all countries; line plot: {}".format(country))

 

    # Middle plot: line plot (male vs. female)

    mf_df=groupby_add_s_rate(country_df, ["year", "sex"])

 

    for key, grp in mf_df.groupby(['sex']):

        axes[1].plot(grp["year"], grp["suicides/100k pop"], label=key)

    ymax=max(mf_df["suicides/100k pop"])+4

    axes[1].set_ylim([0,ymax])

    axes[1].legend(loc="best")

    axes[1].set_title("Male vs. Female Suicide Rates of {}".format(country))



    # Bottom plot: line plot (age groups)

    age_df=groupby_add_s_rate(country_df, ["year", "age"])



    for key, grp in age_df.groupby(['age']):

        axes[2].plot(grp["year"], grp["suicides/100k pop"], label=key)



    ymax2=max(age_df["suicides/100k pop"])+4

    axes[2].set_ylim([0,ymax2])

    axes[2].set_xlabel("year")

    axes[2].legend(loc="best")

    axes[2].set_title("Suicide Rates of Different Age Groups in {}".format(country))
country_stat("United States")
country_stat("South Korea")
mf_year=main_df.groupby(["year","sex"]).agg({"suicides_no":"sum", "population":"sum"}).reset_index()

mf_year["suicides/100k pop"] = mf_year["suicides_no"]/mf_year["population"]*100000



fig, ax = plt.subplots()



for key, grp in mf_year.groupby(['sex']):

    ax.plot(grp["year"], grp["suicides/100k pop"], label=key)



plt.legend(loc='best')

plt.xlabel("Year")

plt.ylim(0,30)

plt.ylabel("Suicide Rate (per 100,000 population)")

plt.show()
# Load the GNI per capita file and clean it up

gni_capita_df=pd.read_csv("../input/worldincomegroupclassification/GNI per capita.csv")

gni_df = gni_capita_df.copy()

gni_df = gni_df.iloc[4:]

n_col = gni_capita_df.iloc[3,0:4].tolist()

n_col = n_col + (gni_capita_df.iloc[3, 4:].astype("int64").tolist())

gni_df.columns = n_col

gni_df=gni_df.rename(columns={"Country Code":"country code"})



# Load the csv file containing the thresholds for the income group categorization each year

threshold_df=pd.read_csv("../input/worldincomegroupclassification/income classification threshold.csv")



# For the suicide rate dataframe, group by year and country. Add the ISO code.

s_rate_df=groupby_add_s_rate(main_df, ["year","country"])

s_rate_df=add_classification(s_rate_df, "country code")
# Create a list of years from 1990 to 2015. It'll be used in the following for loop.

years=s_rate_df["year"].unique().tolist()



# Add the income group and region information to the dataframe

def add_income_region(df):



    years=s_rate_df["year"].unique().tolist()

    income_group_df=pd.DataFrame()



    for year in years:



        # Add the GNI per capita column to the suicide rate dataframe

        sub_df=df[df["year"]==year]

        gni_sub_df=gni_df[[year, "country code"]]

        gni_sub_df=gni_sub_df.rename(columns={year:"GNI per capita"})

        sub_df=pd.merge(sub_df, gni_sub_df, on="country code", how="inner")

        sub_df.dropna(inplace=True)



        # bin the GNI per capita into four income groups  

        max_gni=max(sub_df["GNI per capita"])

        cut_labels_4 = ['Low', 'Lower-middle', 'Upper-middle', 'High']

        str_year=str(year)

        cut_bins = threshold_df[str_year].tolist()

        cut_bins.append(max_gni)   # WARNING: when append() is used with list, you don't need to assign it to a list (e.g., no need to have "cut_bins= at the beginning)

        sub_df["income group"]=pd.cut(sub_df["GNI per capita"], bins=cut_bins, labels=cut_labels_4)



        income_group_df=income_group_df.append(sub_df)  



    # Add the Region information to the dataframe

    region_df=class_df.reset_index().set_index("country code")["Region"]

    category_df=pd.merge(income_group_df, region_df, on="country code", how="left")

        

    return category_df
category_df=add_income_region(s_rate_df)

category_df
# How many countries in each category were survayed each year?

def country_counts(category):

    dist_df=category_df.groupby(["year", category]).agg({"country code":"nunique"}).reset_index()

    sns.lineplot(x='year', y='country code', data=dist_df, hue=category).set_title("Number of countries surveyed")

    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
country_counts("income group")
def category_and_year(category, year):

    

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 9))

    

    # Left plot: All the groups in the select category over time

    sns.lineplot(x = "year", y = "suicides/100k pop", data=category_df, estimator="median",

                 hue=category, ci=None, ax=ax1)

    # Add a vertical line to show where the select year is



    ymax = max(category_df.groupby(["year",category])["suicides/100k pop"].median())

    ymin = min(category_df.groupby(["year",category])["suicides/100k pop"].median())

    

    ax1.vlines(year, ymin, ymax, linestyles ="dashed", colors ="k", label=year )

    ax1.text(year+0.5, ymax-1, year, fontsize=15)

    

    # Right plot: All the groups in the select category in the chosen year

    sub_df=category_df[category_df["year"]==year]

    

    # Set the descending order for the categories of the violin plot

    inc_order = sub_df.groupby(category)["suicides/100k pop"].median().reset_index()

    inc_order = inc_order.sort_values("suicides/100k pop", ascending=False).set_index(category).index.tolist()

    

    sns.violinplot(x = category, y = "suicides/100k pop", data = sub_df, 

                   scale = 'width', inner = 'quartile', color='w', order=inc_order, ax=ax2)

    ax2.set_title("Suicide rate (per 100,000 population) in {fyr} categorized by {fcat}".format(fyr=year, fcat=category))

      

    # Rotate the x labels for regions

    if category == "Region":

        ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, horizontalalignment='right')
category_and_year("income group", 2003)
country_counts("Region")
category_and_year("Region", 1992)
# Calculate % of total suicide rate for each age group

age_grp_df=groupby_add_s_rate(main_df, ["year","country-year", "age"])

total_s_rate_df=age_grp_df.groupby("country-year").agg({"suicides/100k pop":"sum"}).reset_index()

total_s_rate_df=total_s_rate_df[["country-year", "suicides/100k pop"]]

total_s_rate_df=total_s_rate_df.rename(columns={"suicides/100k pop": "total suicide rate"})



age_s_rate=pd.merge(age_grp_df, total_s_rate_df, on="country-year", how='left') 

age_s_rate["% total suicide rate"]=age_s_rate["suicides/100k pop"]/age_s_rate["total suicide rate"]*100
age_s_rate
# Plot suicide rates of different age groups

sns.lineplot(x="year", y="% total suicide rate", data=age_s_rate, estimator="mean", hue="age")

plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
# Group the suicide rate dataframe by year, country, and age. Add ISO contry code.

main_grp=groupby_add_s_rate(main_df, ["year","country","age"])

main_grp=add_classification(main_grp, "country code")



# Add the income group and region information

main_grp=add_income_region(main_grp)
# Add a label to indicate which group the data point falls into. 

def make_labels(df):

    if df["age"]=="75+ years" and df["income group"]=="High" and df["Region"]=="Europe & Central Asia":

        value="75+ years, high income, Europe"

    elif df["age"]=="15-24 years" and df["income group"]=="Upper-middle" and df["Region"]=="Latin America & Caribbean":

        value="15-24 years, Upper-middle income, Latin America & Caribbean"

    else:

        value="other"



    return value
# Add labels to each row using apply() method

main_grp["label"] = main_grp.apply(make_labels, axis=1)
# Get rid of "other" data so that stripplot only shows two groups of my interest

stripplot_df=main_grp[~(main_grp["label"]=="other")]



# Create a bar plot and a strip plot

fig, ax = plt.subplots(figsize=(20,10))

plt.title("Bar plot: suicide rates of different age groups in different countries; stripplot: data points of two interest groups")

sns.boxplot(y=main_grp['suicides/100k pop'], x=main_grp['year'], color="white")

sns.stripplot(x="year", y="suicides/100k pop", hue="label", data=stripplot_df,

                   size=5,jitter=True, edgecolor="black", linewidth=0.5)
# Run a t-test between two groups



from scipy.stats import ttest_ind



cat1 = main_grp[main_grp['label']=="75+ years, high income, Europe"]

cat2 = main_grp[main_grp['label']=="15-24 years, Upper-middle income, Latin America & Caribbean"]



ttest_ind(cat1["suicides/100k pop"], cat2["suicides/100k pop"])
