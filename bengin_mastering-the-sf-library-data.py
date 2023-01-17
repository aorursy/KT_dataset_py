import pandas as pd

import seaborn as sns

%matplotlib inline

import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings('ignore')

import numpy as np
data = pd.read_csv("../input/Library_Usage.csv")

data.head(1)
years = list(range(2003,2017))

member_count = pd.DataFrame({'count' : data.groupby(["Year Patron Registered"]).size()}).reset_index()

ax = sns.barplot(x = "Year Patron Registered",y = "count", data = member_count, order = years, palette = "YlGnBu")

ax = plt.xticks(rotation = 45)

ax = plt.title("Avg. # of Registration Through the Years", fontsize = 18)
month_dict = {"January":"1_", "February":"2_", "March":"3_", "April":"4_", "May":"5_", "June":"6_", "July":"7_", "August":"8_",\

              "September":"9_","October":"10_", "November":"11_", "December":"12_"}



data["Circulation Active Date"] = data["Circulation Active Month"].map(month_dict)  + data["Circulation Active Year"]

data[data["Circulation Active Year"] == "2016"]["Circulation Active Month"].unique()
ax = sns.stripplot(x = "Year Patron Registered", y = "Total Checkouts", data = data, jitter=True)

ax = plt.xticks(fontsize = 12,color="steelblue", alpha=0.8, rotation = 45)

ax = plt.yticks(fontsize = 12,color="steelblue", alpha=0.8)

ax = plt.xlabel("Registration Year", fontsize = 15)

ax = plt.ylabel("Total Checkout #", fontsize = 15)

ax = plt.title("Total Checkout vs Registration Year", fontsize = 18)
ax = sns.stripplot(x = "Year Patron Registered", y = "Total Renewals", data = data, jitter=True)

ax = plt.xticks(fontsize = 12,color="steelblue", alpha=0.8, rotation = 45)

ax = plt.yticks(fontsize = 12,color="steelblue", alpha=0.8)

ax = plt.xlabel("Registration Year",fontsize = 15)

ax = plt.ylabel("Total Renewal #",fontsize = 15)

ax = plt.title("Total Renewal vs Registration Year",fontsize = 18)
dict_age = {'0 to 9 years' : 5, '10 to 19 years' : 15, '20 to 24 years' : 22, '25 to 34 years' : 30, \

            '35 to 44 years': 40, '45 to 54 years' : 50, '55 to 59 years' : 57,'60 to 64 years' : 62, '65 to 74 years' : 70,\

            '75 years and over': 80}

data["Age"] = data["Age Range"].map(dict_age)



def display_corr(values, size):

    sns.set(style="white")



    #the correlation matrix

    corr = values.corr()



    # Generate a mask for the upper triangle

    mask = np.zeros_like(corr, dtype=np.bool)

    mask[np.triu_indices_from(mask)] = True



    # Set up the matplotlib figure

    f, ax = plt.subplots(figsize=(size, size))



    # Generate a custom diverging colormap

    cmap = sns.diverging_palette(220, 10, as_cmap=True)



    # Draw the heatmap with the mask and correct aspect ratio

    sns.heatmap(corr, mask=mask, annot=True,cmap=cmap, vmax=.3,

            square=True,  ax=ax)

    ax = plt.xticks(fontsize = 12,color="steelblue", alpha=0.8, rotation=90)

    ax = plt.yticks(fontsize = 12,color="steelblue", alpha=0.8)

    

display_corr(data, 8)
sns.violinplot(y="Year Patron Registered", data=data[data["Outside of County"] == True], split = True, palette="Set3")
plt.figure(figsize=(10,8))

incidence_count_matrix_long = pd.DataFrame({'count' : data.groupby( [ "Patron Type Definition","Age"] ).size()}).reset_index()

incidence_count_matrix_pivot = incidence_count_matrix_long.pivot("Patron Type Definition","Age","count") 

ax = sns.heatmap(incidence_count_matrix_pivot, annot=True,  linewidths=1, square = False,cbar = False, cmap="Blues") 

ax = plt.xticks(fontsize = 12,color="steelblue", alpha=0.8)

ax = plt.yticks(fontsize = 12,color="steelblue", alpha=0.8)

ax = plt.xlabel("Age", fontsize = 24, color="steelblue")

ax = plt.ylabel("Type", fontsize = 24, color="steelblue")

ax = plt.title("Patron Type and Age Distributions", fontsize = 24, color="steelblue")