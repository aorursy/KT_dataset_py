# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
!pip install seaborn --upgrade
import matplotlib.pyplot as plt

import seaborn as sb



%matplotlib inline
df = pd.read_csv('/kaggle/input/ibrd-statement-of-loans-data/ibrd-statement-of-loans-historical-data.csv')



df = df[df["End of Period"] == "2019-10-31T00:00:00.000"]
sb.__version__
df.info()
df.head()
df.describe()
df[df["Original Principal Amount"] == 0].shape[0]
df[df["Interest Rate"] == 0].shape[0]
df['Loan Type'].value_counts()
df['Loan Status'].value_counts()
# looping through the fields and converting those to datetime type

fields = ['End of Period', 'First Repayment Date', 'Last Repayment Date', 'Agreement Signing Date', 'Board Approval Date', 'Effective Date (Most Recent)', 'Closed Date (Most Recent)', 'Last Disbursement Date']



for field in fields:

    df[field] = pd.to_datetime(df[field])
df.info()
df.drop(columns=['Currency of Commitment'], inplace=True)
df.info()
df.columns = df.columns.str.replace("'s", "")
df.info()
# Assigning base color

base_color = sb.color_palette()[0] 



# Assigning dinominator (1,000,000) to help display big values

mm_var = 1000000



# label for pre-2000

pre_2000 = "Before 2000"



# label for post-2000

post_2000 = "Since 2000"



# variable for displot graphs

plot_height = 6



'''

Common function to set figure size

'''

def setsize(width=8, height=6):

    plt.figure(figsize=(width,height))
df["Board Approval Date"].min(), df["Board Approval Date"].max()
df[["Board Approval Date", "Agreement Signing Date", "Effective Date (Most Recent)"]].sample(10)
df[df['Board Approval Date'] > df['Agreement Signing Date']]
df[df['Agreement Signing Date'] > df['Effective Date (Most Recent)']]
sb.displot(df["Original Principal Amount"]/mm_var, height=plot_height, bins=50, kde=True);

plt.xlabel("Original Principal Amount ($ mm)");
sb.displot(df["Original Principal Amount"]/mm_var, height=plot_height, kde=True);

plt.xlim((0,500))

plt.xlabel("Original Principal Amount ($ mm)");
sb.displot(df["Board Approval Date"].dt.year, height=plot_height, kde=True);
sb.displot(df["Interest Rate"], height=plot_height, kde=True);
setsize()

sb.countplot(y=df["Loan Status"], color=base_color, order=df["Loan Status"].value_counts().index);
setsize()

sb.countplot(y=df["Loan Type"], color=base_color, order=df["Loan Type"].value_counts().index);
# pre-2000 board approved loans

df["post_2000"] = False



# 2000 onwards board approved loans

df["post_2000"] = (df["Board Approval Date"].dt.year >= 2000)



df["post_2000"].value_counts()
sb.displot(data=df, x=df["Original Principal Amount"]/mm_var, hue="post_2000", bins=50, height=plot_height);
ticks = np.arange(0,500+1, 50)

sb.displot(data=df, x=df["Original Principal Amount"]/mm_var, bins=ticks, hue="post_2000", kde=True, height=plot_height);



plt.xlim((0,500))

plt.xticks(ticks);

plt.xlabel("Original Principal Amount ($ mm)");
interest_bins = np.arange(0,df["Interest Rate"].max()+1.0, 1.0)

sb.displot(data=df, x="Interest Rate", bins=interest_bins, hue="post_2000", kde=True, height=6);

plt.xticks(interest_bins);
setsize()

sb.countplot(data=df, x="Loan Type", hue="post_2000");
setsize()

sb.countplot(data=df, y="Loan Status", hue="post_2000");
setsize()

s_ctry_princpl = df.groupby("Country")["Borrower Obligation"].sum().sort_values(ascending=False).head(20)

sb.barplot(y=s_ctry_princpl.index, x=s_ctry_princpl/mm_var, color=base_color);

plt.xlabel("Total Borrower Obligation ($ mm)");
# only considering records with active loans

df_loans_held = df[df["Loans Held"] > 0]



# list of top 20 countries that pay the highest average interest rates

list_loans = df_loans_held.groupby("Country")["Interest Rate"].mean().sort_values(ascending=False).head(20)



# looking at active loans for only those top 20 countries

df_loans_held = df_loans_held[df_loans_held["Country"].isin(list_loans.index)]
setsize()



# using pointplot so see the average interest rate and also range of rates that the country generally gets

sb.pointplot(data=df_loans_held, y="Country", x="Interest Rate", join=False, order=list_loans.index)

plt.xticks(np.arange(0,list_loans.max()+1, 1));
df_loans_held.groupby("Country")["Interest Rate"].describe()
pd.crosstab(df_loans_held["Country"], df_loans_held["Loan Type"])
setsize()



sb.boxplot(data=df, y="Interest Rate", x="Loan Type", color=base_color);
setsize()

sb.scatterplot(y=df["Interest Rate"], x=df["Original Principal Amount"]/mm_var, alpha=0.1, hue=df.post_2000);

plt.xlim((0, 150)) # most of the loans are of less than $150 million

plt.xlabel("Original Principal Amount ($ mm)");
setsize()

df_region_prinpl = df.groupby(["Region", "post_2000"])["Original Principal Amount"].sum().sort_values(ascending=False).reset_index()

sb.barplot(data=df_region_prinpl, y="Region", x=df_region_prinpl["Original Principal Amount"]/mm_var, hue="post_2000")

plt.xlabel("Total Original Principal Amount ($ mm)");
setsize()

df_region_prinpl = df.groupby(["Region", "post_2000"])["Original Principal Amount"].mean().sort_values(ascending=False).reset_index()

sb.barplot(data=df_region_prinpl, y="Region", x=df_region_prinpl["Original Principal Amount"]/mm_var, hue="post_2000");

plt.xlabel("Average Original Principal Amount ($ mm)");
df.groupby(["Region", "post_2000"])["Country"].count().unstack()
df_pre2000 = df[df.post_2000 == False]

df_post2000 = df[df.post_2000 == True]



# countries who have taken most loan before 2000

pre2000 = df_pre2000.groupby("Country")["Original Principal Amount"].sum().sort_values(ascending=False).head(20)



# countries who have taken most loan since 2000

post2000 = df_post2000.groupby("Country")["Original Principal Amount"].sum().sort_values(ascending=False).head(20)



pre2000 = pd.DataFrame(pre2000)

post2000 = pd.DataFrame(post2000)



# joining the two dataframes to have a consolidated view

df_maxloan = pre2000.join(post2000, how="outer", lsuffix="_pre", rsuffix="_post")



# making column names shorter

df_maxloan.columns = ["pre_2000", "post_2000"]



df_maxloan = df_maxloan.reset_index().sort_values("pre_2000", ascending=False)



# melting the pre- and post-2000 principal amount columns so that we can use a barplot and its hue property

df_maxloan = df_maxloan.melt(id_vars=["Country"], var_name="Time", value_name="Original Principal Amount")
setsize(10,8)

sb.barplot(data=df_maxloan, x=df_maxloan["Original Principal Amount"]/mm_var, y="Country", hue="Time")

plt.legend(loc=4);

plt.xlabel("Total Original Principal Amount ($ mm)");
df.groupby("Country")["Original Principal Amount"].sum().sort_values(ascending=False).head(5)
list_toploans = df.groupby("Country")["Original Principal Amount"].sum().sort_values(ascending=False).head(5).index

list_toploans
df_toploans = df[df["Country"].isin(list_toploans)]



ax = sb.FacetGrid(data=df_toploans, col="Country", row="post_2000", hue="Loan Type", margin_titles=True);

ax.map(sb.scatterplot, "Original Principal Amount", "Interest Rate");

ax.add_legend();
pd.crosstab(df_toploans["Country"], [df_toploans["post_2000"], df_toploans["Loan Type"]])
# pivoting data by type and status

df_type_status = df.pivot_table(index="Loan Type", columns="Loan Status", values="Loan Number", aggfunc=len, margins=True, fill_value=0)



# converting counts to percentages

df_type_status = df_type_status.div(df_type_status.iloc[:,-1], axis="index")*100



# resetting index and rounding of decimal places

df_type_status = df_type_status.round(0).reset_index()



# dropping margins created by pivot table

df_type_status.drop(columns=["All"], inplace=True)

df_type_status.drop([13], inplace=True)



# melting all loan statuses so that we can use the dataframe to generate a graph

df_type_status = df_type_status.melt(id_vars=["Loan Type"], var_name="Loan Status", value_name="Percentage")
ax = sb.FacetGrid(data=df_type_status, col="Loan Type", col_wrap=6);

ax.map(sb.barplot, "Loan Status", "Percentage");

ax.set_xticklabels(rotation=90);
ticks = np.arange(0,500+1, 50);

sb.displot(data=df, x=df["Original Principal Amount"]/mm_var, bins=ticks, hue="post_2000", height=plot_height, legend=False);



plt.xlim((0,500));

plt.xticks(ticks);

plt.xlabel("Original Principal Amount ($ mm)");

plt.ylabel("")

plt.title("Distribution of Loans (with Principal Amount less than $500 mm)");

plt.legend([post_2000, pre_2000]);
plt.figure(figsize=(8,6));

sb.countplot(data=df, x="Loan Type", hue="post_2000");

plt.title("Distribution of Loans By Type")

plt.ylabel("")

plt.legend([pre_2000, post_2000]);
# setting legend line colors

from matplotlib.lines import Line2D

custom_lines = [Line2D([0], [0], color="#df8138", lw=4),

                Line2D([0], [0], color="#36749e", lw=4)]



# preparing dataset to analyse Original Principal Amount by region

df_region_prinpl = df.groupby(["Region", "post_2000"])["Original Principal Amount"].agg(['sum', 'mean']).reset_index()



fig, ax = plt.subplots(ncols = 2, figsize = [14,8])

sb.barplot(data=df_region_prinpl, y="Region", x=df_region_prinpl["sum"]/mm_var, hue="post_2000", ax=ax[0])

ax[0].set_xlabel("Total Original Principal Amount ($ mm)");

ax[0].set_ylabel("");

ax[0].get_legend().remove()



sb.barplot(data=df_region_prinpl, y="Region", x=df_region_prinpl["mean"]/mm_var, hue="post_2000", ax=ax[1]);

ax[1].set_xlabel("Average Original Principal Amount ($ mm)");

ax[1].set_yticklabels("");

ax[1].set_ylabel("");

ax[1].get_legend().remove()



fig.legend(custom_lines, [post_2000, pre_2000], 1)

plt.suptitle("Original Principal Amount by Region", size=13);   