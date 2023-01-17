import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import scipy.stats as stats

df_c = pd.read_csv('../input/boston_crime.csv',

                 sep=',', header=0, parse_dates=["OCCURRED_ON_DATE"])
df_c.head(10)
def ExploreData(dtaframe):

    """

    This function print these features of a data frame:

    1. Number of Rows

    2. Number of Columns

    3. Column names and their data types



    :dtaframe: Takes data frame as the input:



    """

    print("\nNumber of Columns: {}".format(len(dtaframe.columns)))

    print("\nNumber of Rows: {}".format(len(dtaframe)))

    print("\nColumns and their data types: \n\n{}".format(dtaframe.dtypes))
ExploreData(df_c)
def PrintColValues(dataframe):

    """

    This function finds the uniques values present in each column in a data frame and prints it.

    The output is column name along with the unique values it has.

    If a column has more than 30 unique values, then it just print the number of unique values that column has.



    :dataframe: Takes data frame as the input:



    """

    for c in list(dataframe.columns):

        n = dataframe[c].unique()

        if len(n) < 30:

            print(c)

            print(n)

        else:

            print(c + ': ' + str(len(n)) + ' unique values')
PrintColValues(df_c)
df_c = df_c.replace("", np.nan, regex=True)
Col_List = list(df_c.columns)

df_c[Col_List].isnull().sum().sort_values(ascending = False)
# Replace Nan with N in Shooting columns 

df_c['SHOOTING'] = df_c['SHOOTING'].replace(np.nan, 'N')
df_c[Col_List].isnull().sum().sort_values(ascending = False)
df_c = df_c.dropna(how='any')
df_c[Col_List].isnull().sum().sort_values(ascending = False)
new_col = {'A1':'Downtown',

           'A15':'Charlestown',

           'A7':'East Boston',

           'B2':'Roxbury',

           'B3':'Mattapan',

           'C6':'South Boston',

           'C11':'Dorchester',

           'D4':'South End',

           'D14':'Brighton',

           'E5':'West Roxbury',

           'E13':'Jamaica Plain',

           'E18':'Hyde Park'}

df_c['DISTRICT_NAME'] = df_c['DISTRICT'].map(new_col) 

df_c.head()
def top_n(data_series, n):

    """

    This function print the top crime rate based on any varible we pass.



    :data_series: Takes the column name

    :n: The number of top records 



    """

    return data_series.value_counts().iloc[:n]
top_n(df_c['OFFENSE_CODE_GROUP'], n=10)
df_c.Lat.replace(-1, None, inplace=True)

df_c.Long.replace(-1, None, inplace=True)
sns.catplot(y='OFFENSE_CODE_GROUP',

            kind='count',

            height=10,

            aspect = 1.2,

            color = "black",

            order=df_c.OFFENSE_CODE_GROUP.value_counts().index,

            data= df_c)

plt.title('Top Crime Count')
#lets look into the offense description and the frequency with respect to the offense group Motor Vehicle Accident Response

df_c['OFFENSE_DESCRIPTION'][df_c['OFFENSE_CODE_GROUP']== 'Motor Vehicle Accident Response'].value_counts().sort_values().plot.barh()
sns.catplot(x='HOUR',

            kind='count',

            height=8,

            aspect = 2,

            color = "purple",

            data= df_c)

plt.title('crime count in a day')
sns.scatterplot(x='Lat',

                y='Long',

                hue='DISTRICT_NAME',

                alpha = 0.01,

                data=df_c)

plt.title('crime separate by district')
sns.catplot(y='YEAR',

            kind='count',

            height=8,

            aspect =1,

            color = "red",

            order=df_c.YEAR.value_counts().index,

            data= df_c)

plt.title('crime rate in different year')
Transformed_DF = df_c.groupby(['HOUR', 'DAY_OF_WEEK']).INCIDENT_NUMBER.agg('count').reset_index()

Transformed_DF['DAY_OF_WEEK'] = pd.Categorical(Transformed_DF['DAY_OF_WEEK'], categories=

    ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday', 'Sunday'],

    ordered=True)

Transformed_DF = Transformed_DF.pivot("HOUR", "DAY_OF_WEEK", "INCIDENT_NUMBER")

Transformed_DF.head()
fig, ax = plt.subplots(figsize=(10,10))         

sns.heatmap(Transformed_DF, linewidths=.5, cmap="YlGnBu", ax= ax)
Transformed_DF1 = df_c.groupby(['HOUR', 'MONTH']).INCIDENT_NUMBER.agg('count').reset_index()

Transformed_DF1 = Transformed_DF1.pivot("HOUR", "MONTH", "INCIDENT_NUMBER")

Transformed_DF1.head()
fig, ax = plt.subplots(figsize=(10,10))         

sns.heatmap(Transformed_DF1, linewidths=.5, cmap="YlGnBu", ax= ax)
df_c["Day_Crime"] = np.where(np.logical_and(df_c["HOUR"] >= 12, df_c["HOUR"] <= 20), 1, 0)

TTest_DF = df_c.resample('D', on='OCCURRED_ON_DATE').agg({'INCIDENT_NUMBER':'count','Day_Crime':'sum'}).reset_index()

TTest_DF["Night_Crime"] = TTest_DF["INCIDENT_NUMBER"] - TTest_DF["Day_Crime"]

TTest_DF = TTest_DF.rename(columns={"INCIDENT_NUMBER": "Total_Crime"})

print("The new data has {} rows".format(len(TTest_DF)))

print("\n\nBelow are the first 5 rows of the transformed dataframe.\n\n{}".format(TTest_DF.head()))
def Remove_Outliers(dta, col):



    """

    This function is used to detect outliers using 3 sigma rule.

    That means any data point outside of 3 SD from the mean is an outlier.

    It takes 1-D data frame or a column as the input and returns a list of values which are outliers.



    :dta: Input data frame:

    :col: Input list of column or 1D data frame:



    """

    OldLen = len(dta)

    # Remove Outliers

    dta = dta[abs(dta[col] - dta[col].mean()) <= (3 * dta[col].std())]

    NewLen = len(dta)

    Num = OldLen - NewLen

    print("There were total {} outliers removed from the dataframe.".format(Num))
Remove_Outliers(TTest_DF, 'Day_Crime')
Remove_Outliers(TTest_DF, 'Night_Crime')
sns.distplot(TTest_DF['Day_Crime'], color="y", kde_kws={"color": "y", "lw": 2, "label": "Day_Crime"})

sns.distplot(TTest_DF['Night_Crime'],color="k", kde_kws={"color": "k", "lw": 2, "label": "Night_Crime"})

plt.show()
sample1 = TTest_DF['Day_Crime']

sample2 = TTest_DF['Night_Crime']
stats.ttest_ind(sample1,sample2, equal_var = False)