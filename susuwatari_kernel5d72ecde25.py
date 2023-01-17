%matplotlib inline

import pandas as pd

import matplotlib.pyplot as plt
# Load

data = "../input/ppp-loan-data-paycheck-protection-program/PPP_data_150k_plus.csv"

df = pd.read_csv(data)



# Gives the plots the same theme as the Seaborn library

plt.style.use("seaborn")
df["JobsRetained"].plot(

    kind="hist", 

    title="Frequency of JobsRetained",

    bins=30,

    alpha=0.3

);
df["JobsRetained"].plot(

    kind="box",

    vert=False,

    title="Distribution of JobsRetaimed"

);
df["LoanRange"].value_counts(normalize=True).plot(

    kind="barh", 

    title="Normazlized Loan Distribution",

    legend = True,

    alpha=0.3

);
df.groupby("State")["JobsRetained"].mean().sort_values().plot(

    kind="bar",

    title = "Mean Jobs Retained In Each State",

    alpha = 0.3,

    legend=True,

)



df.groupby("State")["JobsRetained"].median().sort_values().plot(

    kind="line",

    title = "Mean Jobs Retained In Each State",

    legend=True,

    color="red",

    rot=90,

    alpha=.8,

    figsize=(12,5) # Determines the size 

)

plt.ylabel("Jobs Retained") 

plt.legend(["Median", "Mean"]);
# Create subplots that have 4 rows and one column

fig, ax = plt.subplots(2,2, dpi=100)



df["JobsRetained"].plot(kind="hist", 

                        title="Frequency of JobsRetained",

                        bins=30,

                        ax=ax[0,0], # assigned to 1st row in subplots

                        alpha=0.3

                        )



df["JobsRetained"].plot(kind="box",

                        vert=False,

                        title="Distribution of JobsRetaimed",

                        ax=ax[0,1], #assigned to 2nd row in subplots

                       )



df["LoanRange"].value_counts(normalize=True).plot(kind="barh", 

                                                  title="Normazlized Loan Distribution",

                                                  legend = True,

                                                  ax=ax[1,0], #assigned to 3rd row in subplots

                                                  alpha=0.3

                                                  )



df.groupby("State")["JobsRetained"].mean().sort_values().plot(kind="bar",

                                                              title = "Mean Jobs Retained In Each State",

                                                              alpha = 0.3,

                                                              legend=True,

                                                              ax=ax[1,1], #assigned to 4th row in subplots

                                                             )



df.groupby("State")["JobsRetained"].median().sort_values().plot(kind="line",

                                                                title = "Mean Jobs Retained In Each State",

                                                                legend=True,

                                                                color="red",

                                                                rot=90,

                                                                alpha=.8,

                                                                ax=ax[1,1], #assigned to 4th row in subplots

                                                                figsize=(20,11) # Determines the size of the subplot

                                                               )



plt.ylabel("Jobs Retained") 

plt.legend(["Median", "Mean"]);