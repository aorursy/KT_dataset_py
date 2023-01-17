# Import some necessary libraries

from IPython.display import display, Markdown, Latex

import json

import pandas as pd

import re

import matplotlib.pyplot as plt

import seaborn as sns



# Viz config

plt.style.use('seaborn')

colors = ["#4B88A2", "#BC2C1A", "#DACC3E", "#EA8C55", "#7FB7BE"]

palette = sns.color_palette(colors)

sns.set_palette(palette)

alpha = 0.7
raw_data_file = "/kaggle/input/netherlands-rent-properties/properties.json"



def load_raw_data(filepath):

    raw_data = []

    for line in open(filepath, 'r'):

        raw_data.append(json.loads(line))

    

    df = pd.DataFrame(raw_data)

    

    return df

    

df = load_raw_data(raw_data_file)



Markdown(f"Successfully imported DataFrame with shape: {df.shape}.")
# Functions from: https://www.kaggle.com/juangesino/starter-netherlands-rent-properties



# Define all columns that need to be flatten and the property to extract

flatten_mapper = {

    "_id": "$oid",

    "crawledAt": "$date",

    "firstSeenAt": "$date",

    "lastSeenAt": "$date",

    "detailsCrawledAt": "$date",

}





# Function to do all the work of flattening the columns using the mapper

def flatten_columns(df, mapper):

    

    # Iterate all columns from the mapper

    for column in flatten_mapper:

        prop = flatten_mapper[column]

        raw_column_name = f"{column}_raw"

        

        # Check if the raw column is already there

        if raw_column_name in df.columns:

            # Drop the generated one

            df.drop(columns=[column], inplace=True)

            

            # Rename the raw back to the original

            df.rename(columns={ raw_column_name: column }, inplace=True)        

    

        # To avoid conflicts if re-run, we will rename the columns we will change

        df.rename(columns={

            column: raw_column_name,

        }, inplace=True)



        # Get the value inside the dictionary

        df[column] = df[raw_column_name].apply(lambda obj: obj[prop])

        

    return df





def rename_columns(df):

    # Store a dictionary to be able to rename later

    rename_mapper = {}

    

    # snake_case REGEX pattern

    pattern = re.compile(r'(?<!^)(?=[A-Z])')

    

    # Iterate the DF's columns

    for column in df.columns:

        rename_mapper[column] = pattern.sub('_', column).lower()

        

    # Rename the columns using the mapper

    df.rename(columns=rename_mapper, inplace=True)

    

    return df





def parse_types(df):

    

    df["crawled_at"] = pd.to_datetime(df["crawled_at"])

    df["first_seen_at"] = pd.to_datetime(df["first_seen_at"])

    df["last_seen_at"] = pd.to_datetime(df["last_seen_at"])

    df["details_crawled_at"] = pd.to_datetime(df["details_crawled_at"])

    df["latitude"] = pd.to_numeric(df["latitude"])

    df["longitude"] = pd.to_numeric(df["longitude"])

    

    return df
def add_features(df):

    

    df["rent_per_area"] = df["rent"] / df["area_sqm"]

    

    return df
df = (df

      .pipe(flatten_columns, mapper=flatten_mapper)

      .pipe(rename_columns)

      .pipe(parse_types)

      .pipe(add_features)

     )
df.columns
Markdown(f"""

The dataset contains **{len(df)}** observations with **{len(df.columns)}** feature (columns).

""")
df.info()
df.describe()
cols = ["area_sqm", "city", "property_type", "rent", "rent_per_area"]



display(Markdown("## Top 5 Highest Rent Properties"))

display(df[cols].sort_values(by = ["rent"], ascending=False).head())



display(Markdown("## Top 5 Lowest Rent Properties"))

display(df[cols].sort_values(by = ["rent"], ascending=True).head())



display(Markdown("## Top 5 Highest Area Properties"))

display(df[cols].sort_values(by = ["area_sqm"], ascending=False).head())



display(Markdown("## Top 5 Lowest Area Properties"))

display(df[cols].sort_values(by = ["area_sqm"], ascending=True).head())



display(Markdown("## Top 5 Highest Rent per Sqm"))

display(df[cols].sort_values(by = ["rent_per_area"], ascending=False).head())



display(Markdown("## Top 5 Lowest Rent per Sqm"))

display(df[cols].sort_values(by = ["rent_per_area"], ascending=True).head())
output_md = "<p float='left'>"



for i in range(8):

    img = df.sort_values(by = ["area_sqm"], ascending=False).iloc[i]["cover_image_url"]

    output_md += f"<img src='{img}' width=200 style='float: left; margin: 15px'>"

    

output_md += "</p>"

Markdown(output_md)
fif, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 6))



sns.boxplot(x='rent', data=df, ax=ax1, color=colors[0], boxprops=dict(alpha=alpha))

ax1.set_facecolor('white')

ax1.set_xlabel("Total Rent", labelpad=14)

ax1.set_title("Boxplot Rent", pad=14)



sns.boxplot(x='area_sqm', data=df, ax=ax2, color=colors[1], boxprops=dict(alpha=alpha))

ax2.set_facecolor('white')

ax2.set_xlabel("Area (sqm)", labelpad=14)

ax2.set_title("Boxplot Area", pad=14)



fix2, ax = plt.subplots(1, 1, figsize=(22,8))

sns.boxplot(x='rent_per_area', data=df, ax=ax, color=colors[2], boxprops=dict(alpha=alpha))

ax.set_facecolor('white')

ax.set_xlabel("Rent per Square Meter", labelpad=14)

ax.set_title("Boxplot Rent per Square Meter", pad=14)



plt.show()
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20,6))



sns.distplot(df['rent'], color=colors[0], ax=ax1)

ax1.set_xlabel("Rent", labelpad=14)

ax1.set_title("Distribution of Rent", pad=14)



sns.distplot(df['area_sqm'], color=colors[1], ax=ax2)

ax2.set_xlabel("Area", labelpad=14)

ax2.set_title("Distribution of Area", pad=14)



sns.distplot(df['rent_per_area'], color=colors[2], ax=ax3)

ax3.set_xlabel("Rent per Square Meter", labelpad=14)

ax3.set_title("Distribution of Rent per Square Meter", pad=14)



grid = sns.FacetGrid(df, hue='property_type', palette=palette, height=7, aspect=3)

grid.map(sns.distplot, 'area_sqm', bins=50)

ax = grid.axes[0][0]

ax.legend()

ax.set_xlabel("Area", labelpad=14)

ax.set_title("Distribution of Area by Property Type", pad=14)



grid = sns.FacetGrid(df, hue='property_type', palette=palette, height=7, aspect=3)

grid.map(sns.distplot, 'rent_per_area', bins=50)

ax = grid.axes[0][0]

ax.legend()

ax.set_xlabel("Rent per Square Meter", labelpad=14)

ax.set_title("Distribution of Rent per Square Meter by Property Type", pad=14)

plt.show()
f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(25,5))



ax1.scatter(df['area_sqm'], df['rent'], alpha=0.2)

ax1.set_xlabel(r'Area $(m^2)$', fontsize=15)

ax1.set_ylabel(r'Rent (€)', fontsize=15)

ax1.set_title('Rental price and area')



n, bins, patches = ax2.hist(df['area_sqm'], bins=20, alpha=0.7)

ax2.set_ylabel(r'Frequency', fontsize=15)

ax2.set_xlabel(r'Area $(m^2)$', fontsize=15)

ax2.set_title('Area histogram')



ax3.hist(df['rent'], bins=20, alpha=0.7)

ax3.set_ylabel(r'Frequency', fontsize=15)

ax3.set_xlabel(r'Rent (€)', fontsize=15)

ax3.set_title('Rent histogram')



plt.show()
y_rent = df.groupby(['property_type'])['rent'].mean().index.values

x_rent = df.groupby(['property_type'])['rent'].mean().values



y_area = df.groupby(['property_type'])['area_sqm'].mean().index.values

x_area = df.groupby(['property_type'])['area_sqm'].mean().values



y_rent_sqm = df.groupby(['property_type'])['rent_per_area'].mean().index.values

x_rent_sqm = df.groupby(['property_type'])['rent_per_area'].mean().values



fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,6))



sns.barplot(x=x_rent, y=y_rent, ax=ax1, palette=palette, alpha=alpha)

ax1.set_xlabel("Average Rent", labelpad=14)

ax1.set_title("Average Rent by Propery Type", pad=14)



sns.barplot(x=x_area, y=y_area, ax=ax2, palette=palette, alpha=alpha)

ax2.set_xlabel("Average Area", labelpad=14)

ax2.set_title("Average Area by Propery Type", pad=14)



fig2, ax = plt.subplots(1, 1, figsize=(20,8))

sns.barplot(x=x_rent_sqm, y=y_rent_sqm, ax=ax, palette=palette, alpha=alpha)

ax.set_xlabel("Average Rent per Square Meter", labelpad=14)

ax.set_title("Average Rent per Square Meter by Properly Type", pad=14)

plt.show()
group_city = df.groupby(['city'])['rent', 'area_sqm', 'rent_per_area'].mean()



display(Markdown("## Top 10 Most Expensive Cities"))

display(pd.DataFrame(group_city.sort_values(by = ['rent_per_area'], ascending=False)['rent_per_area'].head(10)))



display(Markdown("## Top 10 Least Expensive Cities"))

display(pd.DataFrame(group_city.sort_values(by = ['rent_per_area'], ascending=True)['rent_per_area']).head(10))
fig, ax = plt.subplots(1, 1, figsize=(20,8))

sns.scatterplot(x="area_sqm", y="rent", data=df, hue="property_type", palette=palette, alpha=alpha)

ax.set_ylabel("Rent", labelpad=14)

ax.set_xlabel("Area", labelpad=14)

ax.set_title("Relationship Between Rent and Area by Property Type", pad=14)

legend = ax.legend()

legend.texts[0].set_text("Property Types")

plt.show()
group = df.groupby(['internet'])

fix, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 6))

sns.barplot(y=group['rent'].count().index.values, x=group['rent'].count().values, palette=palette, ax=ax1, alpha=alpha)

sns.barplot(y=group['rent'].mean().index.values, x=group['rent'].mean().values, palette=palette, ax=ax2, alpha=alpha)

sns.barplot(y=group['rent_per_area'].mean().index.values, x=group['rent_per_area'].mean().values, palette=palette, ax=ax3, alpha=alpha)



ax1.set_title("Number of Properties by Internet Provided")

ax2.set_title("Average Rent by Internet Provided")

ax3.set_title("Average Rent per Square Meter by Internet Provided")



plt.show()
group = df.groupby(['energy_label'])

fix, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 6))

sns.barplot(y=group['rent'].count().index.values, x=group['rent'].count().values, palette=palette, ax=ax1, alpha=alpha)

sns.barplot(y=group['rent'].mean().index.values, x=group['rent'].mean().values, palette=palette, ax=ax2, alpha=alpha)

sns.barplot(y=group['rent_per_area'].mean().index.values, x=group['rent_per_area'].mean().values, palette=palette, ax=ax3, alpha=alpha)



ax1.set_title("Number of Properties by Energy Label")

ax2.set_title("Average Rent by Energy Label")

ax3.set_title("Average Rent per Square Meter by Energy Label")



plt.show()
group = df.groupby(['furnish'])

fix, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 6))

sns.barplot(y=group['rent'].count().index.values, x=group['rent'].count().values, palette=palette, ax=ax1, alpha=alpha)

sns.barplot(y=group['rent'].mean().index.values, x=group['rent'].mean().values, palette=palette, ax=ax2, alpha=alpha)

sns.barplot(y=group['rent_per_area'].mean().index.values, x=group['rent_per_area'].mean().values, palette=palette, ax=ax3, alpha=alpha)



ax1.set_title("Number of Properties by Furnish Status")

ax2.set_title("Average Rent by Furnish Status")

ax3.set_title("Average Rent per Square Meter by Furnish Status")



plt.show()
group = df.groupby(['kitchen'])

fix, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 6))

sns.barplot(y=group['rent'].count().index.values, x=group['rent'].count().values, palette=palette, ax=ax1, alpha=alpha)

sns.barplot(y=group['rent'].mean().index.values, x=group['rent'].mean().values, palette=palette, ax=ax2, alpha=alpha)

sns.barplot(y=group['rent_per_area'].mean().index.values, x=group['rent_per_area'].mean().values, palette=palette, ax=ax3, alpha=alpha)



ax1.set_title("Number of Properties by Kitchen Status")

ax2.set_title("Average Rent by Kitchen Status")

ax3.set_title("Average Rent per Square Meter by Kitchen Status")



plt.show()
group = df.groupby(['toilet'])

fix, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 6))

sns.barplot(y=group['rent'].count().index.values, x=group['rent'].count().values, palette=palette, ax=ax1, alpha=alpha)

sns.barplot(y=group['rent'].mean().index.values, x=group['rent'].mean().values, palette=palette, ax=ax2, alpha=alpha)

sns.barplot(y=group['rent_per_area'].mean().index.values, x=group['rent_per_area'].mean().values, palette=palette, ax=ax3, alpha=alpha)



ax1.set_title("Number of Properties by Toilet Status")

ax2.set_title("Average Rent by Toilet Status")

ax3.set_title("Average Rent per Square Meter by Toilet Status")



plt.show()
group = df.groupby(['roommates'])

fix, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 6))

sns.barplot(y=group['rent'].count().index.values, x=group['rent'].count().values, palette=palette, ax=ax1, alpha=alpha)

sns.barplot(y=group['rent'].mean().index.values, x=group['rent'].mean().values, palette=palette, ax=ax2, alpha=alpha)

sns.barplot(y=group['rent_per_area'].mean().index.values, x=group['rent_per_area'].mean().values, palette=palette, ax=ax3, alpha=alpha)



ax1.set_title("Number of Properties by Number of Roommates")

ax2.set_title("Average Rent by Number of Roommates")

ax3.set_title("Average Rent per Square Meter by Number of Roommates")



plt.show()
group = df.groupby(['match_gender'])

fix, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 6))

sns.barplot(y=group['rent'].count().index.values, x=group['rent'].count().values, palette=palette, ax=ax1)

sns.barplot(y=group['rent'].mean().index.values, x=group['rent'].mean().values, palette=palette, ax=ax2)

sns.barplot(y=group['rent_per_area'].mean().index.values, x=group['rent_per_area'].mean().values, palette=palette, ax=ax3)



ax1.set_title("Number of Properties by Preferred Gender")

ax2.set_title("Average Rent by Preferred Gender")

ax3.set_title("Average Rent per Square Meter by Preferred Gender")



plt.show()