# Let's import some essentials

import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from scipy import stats
df = pd.read_csv("../input/crimes-in-boston/crime.csv",encoding='latin-1')

df.head(5)
df = df.drop("Location", axis = 1)
df = df.drop(["INCIDENT_NUMBER", "OFFENSE_CODE"], axis = 1)

df.head(5)
df['SHOOTING'].describe()
df["SHOOTING"].isna().sum()
df["SHOOTING"].fillna('N', inplace=True)

df.head(5)
df.OFFENSE_CODE_GROUP.isna().sum()
years = [2015, 2016, 2017, 2018]

for year in years:

    print(df[df["YEAR"] == year]["MONTH"].unique())
time_list = ["HOUR", "DAY_OF_WEEK", "MONTH", "OCCURRED_ON_DATE"]

for item in time_list:

    print("No. of NaNs in {} = {}".format(item, df[item].isna().sum()))
locations_list = ["DISTRICT", "REPORTING_AREA", "STREET", "Lat", "Long"]

for item in locations_list:

    print("No. of NaNs in {} = {}".format(item, df[item].isna().sum()))
df["Long"].describe()
df["Lat"].replace(-1, None, inplace=True)

df["Long"].replace(-1, None, inplace=True)

(df["Long"].isna()).sum()
def plot_quantitative(df, col1=None, col2=None, hue=None, k=10, palette=None):

    if col2 == None:

        col2 = col1

    sns.catplot(x=col1, y=col2, kind='count', height=8, aspect=1.5,

                order=df[col2].value_counts().index[0:k],

                hue=hue, data=df, palette=palette)

    plt.show()
plot_quantitative(df, None, "OFFENSE_CODE_GROUP", None, 10)
df_part_one = df[df["UCR_PART"] == "Part One"]

df_part_two = df[df["UCR_PART"] == "Part Two"]

df_part_three = df[df["UCR_PART"] == "Part Three"]

plt.figure(figsize=(14, 7))

plt.subplots_adjust(bottom=0.0001, left=0.01, wspace=0.35, hspace=0.35)

col = "OFFENSE_CODE_GROUP"

k = 5

plt.subplot(221)

plt.title('Part One crimes')

plt.ylabel('OFFENSE_CODE_GROUP')

sns.countplot(y=col, data=df_part_one, hue="YEAR", order=df_part_one[col].value_counts().index[0:k])

plt.subplot(222)

plt.title('Part Two crimes')

plt.ylabel(' ')

sns.countplot(y=col, data=df_part_two, hue="YEAR", order=df_part_two[col].value_counts().index[0:k])

plt.subplot(223)

plt.title('Part Three crimes')

plt.ylabel(' ')

sns.countplot(y=col, data=df_part_three, hue="YEAR", order=df_part_three[col].value_counts().index[0:k])

plt.show()
hue = "YEAR"

col1 = "OFFENSE_CODE_GROUP"

plot_quantitative(df, None, col1, hue, 10)
(df[df["OFFENSE_CODE_GROUP"] == "Towed"]["YEAR"] == 2017).sum()
def construct_crosstab(col1_name, col2_name):

    ct = pd.crosstab(df[col1_name], df[col2_name])

    return ct
ct = construct_crosstab("YEAR", "OFFENSE_CODE_GROUP")
chi2, pvalue, dof, _ = stats.chi2_contingency(ct)
pvalue
def cramersV(chi2, num_samples, num_rows_in_ct, num_cols_in_ct):

    squared_phi = chi2/num_samples

    squared_phi_corr = max(0, squared_phi - ((num_cols_in_ct-1)*(num_rows_in_ct-1))/(num_samples-1))    

    row_corr = num_rows_in_ct - ((num_rows_in_ct-1)**2)/(num_samples-1)

    col_corr = num_cols_in_ct - ((num_cols_in_ct-1)**2)/(num_samples-1)

    return np.sqrt(squared_phi_corr / min( (col_corr-1), (row_corr-1)))



def compute_degree_of_relatedness(col1, col2):

    ct = construct_crosstab(col1, col2)

    chi2, pvalue, dof, _ = stats.chi2_contingency(ct)

    num_samples = ct.sum().sum()

    num_rows, num_cols = ct.shape

    return cramersV(chi2, num_samples, num_rows, num_cols)
degree_year = compute_degree_of_relatedness("YEAR", "OFFENSE_CODE_GROUP")

degree_year
df['OCCURRED_ON_DATE'] = pd.to_datetime(df['OCCURRED_ON_DATE'])

df.head(10)
df["QUARTER"] = df["OCCURRED_ON_DATE"].dt.quarter
hue = "QUARTER"

col1 = "OFFENSE_CODE_GROUP"

plot_quantitative(df, None, col1, hue, 10)
degree_quarter = compute_degree_of_relatedness("QUARTER", "OFFENSE_CODE_GROUP")

degree_quarter
hue = "MONTH"

col1 = "OFFENSE_CODE_GROUP"

plot_quantitative(df, None, col1, hue, 10, "bright")
degree_month = compute_degree_of_relatedness("MONTH", "OFFENSE_CODE_GROUP")

degree_month
degree_hour = compute_degree_of_relatedness("HOUR", "OFFENSE_CODE_GROUP")

degree_hour
loc_list = ["STREET", "REPORTING_AREA", "DISTRICT", "Lat", "Long"]

degree = []

for loc in loc_list:

    degree.append(compute_degree_of_relatedness(loc, "OFFENSE_CODE_GROUP"))

degree
import folium

from folium.plugins import MarkerCluster



plot_data = df[df['UCR_PART'] == 'Part One'].dropna(axis = 0)

boston_crime_map = folium.Map(location = [plot_data['Lat'].mean(), 

                                          plot_data['Long'].mean()], 

                            zoom_start = 11

                             )

mc = MarkerCluster()

for row in plot_data.itertuples():

    mc.add_child(folium.Marker(location = [row.Lat,  row.Long]))



boston_crime_map.add_child(mc)



boston_crime_map
df[df.UCR_PART == 'Part One']['STREET'].value_counts()[:10]
df[df.UCR_PART == 'Part One']['STREET'].value_counts()[-10:]
df[df.UCR_PART != 'Part One']['STREET'].value_counts()[:10]
df['STREET'].value_counts()[:10]