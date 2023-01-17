#Step 1: Load the necessary packages:



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
#Step 2: Load the data set:

test_df=pd.read_csv("../input/californiaddsexpenditures/californiaDDSDataV2.csv")
test_df.head()
test_df.describe()
# Step 4: Our problem statement is to determine if there is Ethnic bias in the Expenditures. Let us find the mean of Expenditures per Ethnicity and check if the claim is valid.



test_df.groupby('Ethnicity').mean().sort_values('Expenditures')
# Step 5: A box plot for the above data is shown below.

bplot=sns.boxplot(y='Ethnicity', x='Expenditures', 

                 data=test_df.sort_values('Ethnicity'), 

                 width=0.5,

                 palette="colorblind")

bplot.axes.set_title("Expenditures By Ethnicity", fontsize=16) 

bplot.set_xlabel("Expenditures", fontsize=14)

bplot.set_ylabel("Ethnicity", fontsize=14) 

bplot.tick_params(labelsize=10)
#Step 6: Let us begin with grouping the Expenditures data on Gender:

test_df.loc[:,['Gender', 'Expenditures']].groupby('Gender').mean().sort_values('Gender')
#Step 7: Let us consider Age. We will use Age Cohort feature available in the data set.

test_df.loc[:,['Age Cohort', 'Expenditures']].groupby('Age Cohort').mean().sort_values('Expenditures')
df = test_df.loc[:,['Age Cohort', 'Expenditures', 'Ethnicity']]
#Step 8: Let us get a perspective of how funds are allocated to different Ethnicities within the Age groups.

zero_to_5 = df['Age Cohort'] == '0 to 5'

six_to_12 = df['Age Cohort'] == '6 to 12'

thirteen_to_17 = df['Age Cohort'] == '13 to 17'

eighteen_to_21 = df['Age Cohort'] == '18 to 21'

twentytwo_to_50 = df['Age Cohort'] == '22 to 50'

fiftyone_plus = df['Age Cohort'] == '51+'
bplot=sns.boxplot(y='Ethnicity', x='Expenditures', 

                 data=df.where(zero_to_5).dropna().sort_values('Ethnicity'), 

                 width=0.5,

                 palette="colorblind")

bplot.axes.set_title("Expenditures By Ethnicity, Age: 0 to 5", fontsize=16) 

bplot.set_xlabel("Expenditures", fontsize=14)

bplot.set_ylabel("Ethnicity", fontsize=14) 

bplot.tick_params(labelsize=10)
bplot=sns.boxplot(y='Ethnicity', x='Expenditures', 

                 data=df.where(six_to_12).dropna().sort_values('Ethnicity'), 

                 width=0.5,

                 palette="colorblind")

bplot.axes.set_title("Expenditures By Ethnicity, Age: 6 to 12", fontsize=16) 

bplot.set_xlabel("Expenditures", fontsize=14)

bplot.set_ylabel("Ethnicity", fontsize=14) 

bplot.tick_params(labelsize=10)
bplot=sns.boxplot(y='Ethnicity', x='Expenditures', 

                 data=df.where(thirteen_to_17).dropna().sort_values('Ethnicity'), 

                 width=0.5,

                 palette="colorblind")

bplot.axes.set_title("Expenditures By Ethnicity, Age: 13 to 17", fontsize=16) 

bplot.set_xlabel("Expenditures", fontsize=14)

bplot.set_ylabel("Ethnicity", fontsize=14) 

bplot.tick_params(labelsize=10)
bplot=sns.boxplot(y='Ethnicity', x='Expenditures', 

                 data=df.where(eighteen_to_21).dropna().sort_values('Ethnicity'), 

                 width=0.5,

                 palette="colorblind")

bplot.axes.set_title("Expenditures By Ethnicity, Age: 18 to 21", fontsize=16) 

bplot.set_xlabel("Expenditures", fontsize=14)

bplot.set_ylabel("Ethnicity", fontsize=14) 

bplot.tick_params(labelsize=10)
bplot=sns.boxplot(y='Ethnicity', x='Expenditures', 

                 data=df.where(twentytwo_to_50).dropna().sort_values('Ethnicity'), 

                 width=0.5,

                 palette="colorblind")

bplot.axes.set_title("Expenditures By Ethnicity, Age: 22 to 50", fontsize=16) 

bplot.set_xlabel("Expenditures", fontsize=14)

bplot.set_ylabel("Ethnicity", fontsize=14) 

bplot.tick_params(labelsize=10)
bplot=sns.boxplot(y='Ethnicity', x='Expenditures', 

                 data=df.where(fiftyone_plus).dropna().sort_values('Ethnicity'), 

                 width=0.5,

                 palette="colorblind")

bplot.axes.set_title("Expenditures By Ethnicity, Age: 50+", fontsize=16) 

bplot.set_xlabel("Expenditures", fontsize=14)

bplot.set_ylabel("Ethnicity", fontsize=14) 

bplot.tick_params(labelsize=10)