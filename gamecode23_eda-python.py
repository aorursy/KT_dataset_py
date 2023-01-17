# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

import warnings

warnings.filterwarnings("ignore")



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
#Get the data and all the null value removal



data = pd.read_csv("../input/train.csv")

metadata = pd.read_excel("../input/LCDataDictionary.xlsx")

new_data = data.dropna(axis = 1,thresh= data.shape[0]*92/100)

not_useful_columns = ["id", "member_id","url"]

new_data.drop(not_useful_columns, axis = 1, inplace = True)



new_data = new_data[new_data["loan_status"] != "Current"]

new_data = new_data[new_data["loan_status"] != "Issued"]

for col_name in new_data.columns:

    if new_data[col_name].dtype == "object":

        mode_var = new_data[col_name].mode()[0]

        new_data[col_name].fillna(mode_var,inplace = True)

    else:

        mean_var = new_data[col_name].mean()

        new_data[col_name].fillna(mean_var,inplace = True)
new_data.columns
%matplotlib notebook

import matplotlib.pyplot as plt

import seaborn as sns

fig,ax = plt.subplots(1,3,sharey = True)



sns.distplot(new_data["loan_amnt"], ax = ax[0],kde=False)

sns.distplot(new_data["funded_amnt"],ax = ax[1],kde=False)

sns.distplot(new_data["funded_amnt_inv"],ax = ax[2],kde=False)

plt.figure()

sns.distplot(new_data["annual_inc"],kde=False)

plt.figure()

sns.distplot(new_data["int_rate"])
plt.figure()

sns.boxplot("annual_inc",data = new_data, color = "blue")
plt.figure(figsize = (7,7))

g = sns.countplot(new_data["loan_status"])

"""

to sort the plot.. we cannot use sns countplot

We can rather use pandas inbuilt plot function as 

new_data.values_count().plot("bar")

"""

plt.xticks(rotation = 90)

plt.tight_layout()
def converts(x):

    values = """Fully Paid

Charged Off

Late (31-120 days)

In Grace Period

Late (16-30 days)

Does not meet the credit policy. Status:Fully Paid

Default

Does not meet the credit policy. Status:Charged Off""".split("\n")

    is_it_positive = [1,0,0,0,0,1,0,0]

    for this_string,ret_value in zip(values,is_it_positive):

        if x == this_string:

            return ret_value





new_data["labels"] = new_data["loan_status"].apply(converts)
plt.figure()

new_data.groupby(["labels"])["loan_amnt"].sum().plot(kind ="bar")
plt.figure(figsize = (7,7))

sns.scatterplot(x = "loan_amnt", y = "annual_inc", hue = "loan_status", data = new_data)

plt.legend(loc = "best")
plt.figure(figsize = (7,7))

sns.scatterplot(x = new_data["loan_amnt"], y = new_data[new_data["annual_inc"] < 400000]["annual_inc"], hue = "loan_status", data = new_data)

plt.legend(loc = "upper right",bbox_to_anchor=(1.45, 0.8))
plt.figure(figsize = (7,7))

trunc_new_data = new_data[new_data["annual_inc"] < 400000]

sns.regplot(x = "loan_amnt", y = "annual_inc", data = trunc_new_data, scatter_kws = {'alpha': 1})

kws = dict(alpha = 0.1)



canv1 = sns.FacetGrid(data = trunc_new_data,row= "loan_status",height=5)

canv1 = canv1.map(sns.scatterplot, "loan_amnt", "annual_inc",**kws)



"""

this maps sns.scatter for those two parametes on every facetgrid combination

"""
new_data['years'] = pd.to_datetime(new_data["issue_d"]).dt.year

new_data['month'] = pd.to_datetime(new_data["issue_d"]).dt.month
plt.figure()

sns.barplot("years","loan_amnt",data = new_data)

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

plt.figure(figsize = (7,7))

sns.barplot("years","loan_amnt",hue = "loan_status",data = new_data,ci = None)

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

plt.figure()
new_data.groupby(["years","loan_status"]).mean()["loan_amnt"].unstack().plot(kind = "line")

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

#Borrowed from another kernel by Janio



clusters_by_regions = {"west" : ['CA', 'OR', 'UT','WA', 'CO', 'NV', 'AK', 'MT', 'HI', 'WY', 'ID'],

"south_west" : ['AZ', 'TX', 'NM', 'OK'],

"south_east" : ['GA', 'NC', 'VA', 'FL', 'KY', 'SC', 'LA', 'AL', 'WV', 'DC', 'AR', 'DE', 'MS', 'TN' ],

"mid_west" : ['IL', 'MO', 'MN', 'OH', 'WI', 'KS', 'MI', 'SD', 'IA', 'NE', 'IN', 'ND'],

"north_east" : ['CT', 'NY', 'PA', 'NJ', 'RI','MA', 'MD', 'VT', 'NH', 'ME']}
def statify(x):

    global clusters_by_regions

    for region in clusters_by_regions.keys():

        if x in clusters_by_regions[region]:

            return region

new_data["region"] = new_data["addr_state"].apply(statify)
plt.figure()

new_data.groupby(["years", "region"])["loan_amnt"].mean().unstack().plot(kind = "line")
#Total in each region

plt.figure()

new_data.groupby(["years", "region"])["loan_amnt"].sum().unstack().plot(kind = "line")
#Total in each region

plt.figure()

new_data.groupby(["region","labels"])["loan_amnt"].sum().unstack().plot(kind = "bar")
#Total in each region

plt.figure()

new_data.groupby(["region","labels"])["loan_amnt"].mean().unstack().plot(kind = "bar")
#Total in each region

plt.figure()

sns.countplot(new_data["region"])
new_data.groupby(["addr_state"])["annual_inc"].mean().sort_values(ascending = False).head(15)
new_data.groupby(["addr_state"])["loan_amnt"].mean().sort_values(ascending = False).head(15)
canvas2 = sns.FacetGrid(new_data,col="labels")

canvas2.map(sns.distplot,"int_rate")

canvas2.fig.suptitle("For all people in my data")





canvas3 = sns.FacetGrid(trunc_new_data,col="labels")

canvas3.map(sns.distplot,"int_rate")

canvas3.fig.suptitle("For all <4L salary ppl")





canvas3 = sns.FacetGrid(new_data[new_data["annual_inc"] > 400000],col="labels")

canvas3.map(sns.distplot,"int_rate")

canvas3.fig.suptitle("For all >4L salary ppl")
new_data["emp_length"].unique()



new_data["employ_exp"] = new_data["emp_length"].replace(['10+ years', '8 years', '3 years', '2 years', '< 1 year', '1 year',

       '5 years', '7 years', '6 years', '9 years', '4 years'],[10,8,3,2,0.5,1,5,7,6,9,4])



pairplot_cols = ["annual_inc","loan_amnt", "employ_exp", "term","int_rate","years"]
trunc_new_data = new_data[new_data["annual_inc"] < 200000]

sns.pairplot(trunc_new_data[pairplot_cols])
#The labelwise counts of purposes

pur_lab = new_data.groupby(by = ["purpose","labels"])["loan_amnt"].count().sort_values(ascending= False)

pur_lab
new_data.groupby(["purpose","labels"])["loan_amnt"].count().sort_values(ascending= False).unstack().plot(kind="bar",)