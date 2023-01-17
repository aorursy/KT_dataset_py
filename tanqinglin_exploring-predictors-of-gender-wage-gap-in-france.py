import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
salary = pd.read_csv("../input/french-employment-by-town/net_salary_per_town_categories.csv")

industry = pd.read_csv("../input/french-employment-by-town/base_etablissement_par_tranche_effectif.csv")

population = pd.read_csv("../input/french-employment-by-town/population.csv")
salary.head(3)
salary.info() 
industry.head(3)
industry.info()
population.head(3)
population.info()
salary = salary[salary["CODGEO"].apply(lambda x: str(x).isdigit())]
salary["CODGEO"] = salary["CODGEO"].astype(int)
wage_gap = salary["SNHMH14"] - salary["SNHMF14"]

salary["wage_gap"] = wage_gap

mean_vs_gap = pd.DataFrame({'Mean Wages':salary["SNHM14"], 'Wage Gap': salary["wage_gap"]})
sns.set(style="whitegrid")

sns.set(rc={'figure.figsize':(20,10)})



wage_gap_plot = sns.lineplot(data=mean_vs_gap, linewidth=0.7)

wage_gap_plot.axes.set_title("Wage Gap in Every French Cities",fontsize=20)

wage_gap_plot.set_xlabel("Number of Cities", fontsize=15)

wage_gap_plot.set_ylabel("Hourly Wage Gap / €", fontsize=15)



plt.show()
percent_wage_gap = [(salary["wage_gap"]/salary["SNHM14"]) * 100]

mean_percent_wage_gap = np.mean(percent_wage_gap)
sns.set(rc={'figure.figsize':(20,10)})

pwg_plot = sns.lineplot(data=percent_wage_gap, linewidth=0.7, label = 'Percentage by City')

pwg_plot.axhline(mean_percent_wage_gap, ls='--', color = 'orange', linewidth=2.5, label = 'Mean')



pwg_plot.axes.set_title("Wage Gap as a Percentage of Mean Wage in French Cities",fontsize=20)

pwg_plot.set_xlabel("Number of Cities", fontsize=15)

pwg_plot.set_ylabel("Perncentage / %", fontsize=15)

pwg_plot.legend()



plt.show()
# Identifying the French cities where women receives a higher wage than men.

salary["percent_wage_gap"] = (wage_gap / salary["SNHM14"]) * 100

salary.loc[salary["percent_wage_gap"] <= 0]
top_wage_gap = salary.sort_values(by=["percent_wage_gap"], ascending=False).head(30)
# Initialize the matplotlib figure

f, ax = plt.subplots(figsize=(20, 15))



# Plot the Mean Wage

sns.set_color_codes("pastel")

sns.barplot(x="SNHM14", y="LIBGEO", data=top_wage_gap,

            label="Mean Wage", color="b")



# Plot the Gender Wage Gap

sns.set_color_codes("muted")

p = sns.barplot(x="wage_gap", y="LIBGEO", data=top_wage_gap,

            label="Wage Gap", color="b")



# Add a legend and informative axis label

ax.legend(ncol=2, loc="lower right", frameon=True)

ax.set(xlim=(0, 50))

ax.set_title("30 Cities in France with the Highest Percentage Wage Gap", fontsize=20)

ax.set_ylabel("Cities", fontsize=15)

ax.set_xlabel("Percentage Wage Gap", fontsize=15)

sns.despine(left=True, bottom=True)
industry = industry[industry["CODGEO"].apply(lambda x: str(x).isdigit())]
industry["CODGEO"] = industry["CODGEO"].astype(int)
merged_data = salary.merge(industry, how="left", left_on = "CODGEO", right_on="CODGEO")

merged_data.head(3)
economy_vs_pgwg = sns.jointplot("E14TST", "percent_wage_gap", data=merged_data, kind="reg",

                  xlim=(0, 80000), ylim=(-10, 100), height=7)



plt.subplots_adjust(top=0.9)

economy_vs_pgwg.fig.suptitle('Economic Size vs Percentage Gender Wage Gap', fontsize=20)

economy_vs_pgwg.ax_joint.set_xlabel('Number of Firms', fontsize=15)

economy_vs_pgwg.ax_joint.set_ylabel("Perncentage Gender Wage Gap / %", fontsize=15)

plt.show()
round(np.corrcoef(merged_data["E14TST"], merged_data["percent_wage_gap"])[1][0], 5)
merged_data["firm_size_score"] = (1 * merged_data["E14TS1"] + 2 * merged_data["E14TS6"] + 3 * merged_data["E14TS10"] 

                               + 4 * merged_data["E14TS20"] + 5 * merged_data["E14TS50"] + 6 * merged_data["E14TS100"]

                               + 7 * merged_data["E14TS200"] + 8 * merged_data["E14TS500"]) / (merged_data["E14TST"] - merged_data["E14TS0ND"])

merged_data.isnull().sum()
merged_data.dropna(subset=["firm_size_score"], inplace = True)
firm_size_vs_pgwg = sns.jointplot("firm_size_score", "percent_wage_gap", data=merged_data, kind="reg",

                  xlim=(0, 4), ylim=(-10, 100), height=7)



plt.subplots_adjust(top=0.9)

firm_size_vs_pgwg.fig.suptitle('Firm Size vs Percentage Gender Wage Gap', fontsize=20)

firm_size_vs_pgwg.ax_joint.set_xlabel('Firm Size Scores', fontsize=15)

firm_size_vs_pgwg.ax_joint.set_ylabel("Perncentage Gender Wage Gap / %", fontsize=15)

plt.show()
round(np.corrcoef(merged_data["firm_size_score"], merged_data["percent_wage_gap"])[1][0], 5)
population = population[population["CODGEO"].apply(lambda x: str(x).isdigit())]
population["CODGEO"] = population["CODGEO"].astype(int)
total_population = population.groupby(["CODGEO"]).sum()



single_moms = population[(population.MOCO == 23) & (population.SEXE == 2)]

single_moms_per_city = single_moms.groupby(["CODGEO"]).sum()



single_moms_proportion = (single_moms_per_city["NB"] / total_population["NB"]) * 100



merged_data = merged_data.merge(single_moms_proportion, how="left", left_on = "CODGEO", right_on="CODGEO")

merged_data.head(3)
merged_data["NB"].fillna(0, inplace = True)
single_mom_vs_pgwg = sns.jointplot("NB", "percent_wage_gap", data=merged_data, kind="reg",

                  xlim=(0, 25), ylim=(-10, 100), height=7)



plt.subplots_adjust(top=0.9)

single_mom_vs_pgwg.fig.suptitle('Single Mom Families vs Percentage Gender Wage Gap', fontsize=20)

single_mom_vs_pgwg.ax_joint.set_xlabel("Proportion of Single Moms in the City / %", fontsize=15)

single_mom_vs_pgwg.ax_joint.set_ylabel("Perncentage Gender Wage Gap / %", fontsize=15)

plt.show()
round(np.corrcoef(merged_data["NB"], merged_data["percent_wage_gap"])[1][0], 5)