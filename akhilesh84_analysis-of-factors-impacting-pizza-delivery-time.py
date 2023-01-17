import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns
from scipy import stats

import os
raw_data = pd.read_excel(
    io="../input/Module_3_excercises.xls",
    sheet_name="2.Pizza"
)
raw_data = raw_data.drop(["Order", "Driver", "Hour"], axis=1)
raw_data
raw_data.describe()
pairwise_data = raw_data.groupby(by=["Crust", "G.Bread"])["Time to Deliver"].mean()
pairwise_data = pd.DataFrame(pairwise_data)
pairwise_data = pairwise_data.unstack()
pairwise_data.columns = pairwise_data.columns.get_level_values(1)

pairwise_data
sns.set(rc={"figure.figsize": (12, 5)})
sns.barplot(data=raw_data, x="Crust", y="Time to Deliver", hue="G.Bread");
def getGroupData(group_name: iter):
    return raw_data.groupby(by=group_name).groups
# Performing ANOVA for crust type groups
crust_groups = getGroupData("Crust")

thin_crust = raw_data.loc[crust_groups[0]]["Time to Deliver"]
thick_crust = raw_data.loc[crust_groups[1]]["Time to Deliver"]

result = stats.f_oneway(thick_crust, thin_crust)
print("Mean Delivery time for Thick Crust pizza: {0:.2f}\nMean Delivery time for Thin Crust pizza: {1:.2f}".format(np.mean(thick_crust.values), np.mean(thin_crust.values)))
display(result)
# Performing ANOVA for garlic bread ordered or not groups
bread_groups = raw_data.groupby(by='G.Bread').groups

bread_ordered = raw_data.loc[bread_groups[0]]["Time to Deliver"]
bread_not_ordered = raw_data.loc[bread_groups[1]]["Time to Deliver"]
print("Mean Delivery time when bread ordered: {0:.2f}\nMean Delivery time when bread not ordered: {1:.2f}".format(np.mean(bread_ordered.values), np.mean(bread_not_ordered.values)))
result = stats.f_oneway(bread_ordered, bread_not_ordered)
display(result)
grp_by_crust_and_bread = raw_data.groupby(by=['Crust','G.Bread']).groups
print("(Thick=0/Thin=1, Garlic bread ordered = 0 / Not ordered = 1) [Row Indices]")
print("===========================================================================")
for key, value in grp_by_crust_and_bread.items():
    print(key, grp_by_crust_and_bread[key].values)

print("\nPairwise t-test\n")
a = [value.values for key,value in grp_by_crust_and_bread.items()]
for i in range(3):
    for j in range(i+1, 4):
        print("{0}, {1}, {2}".format(a[i], a[j], stats.ttest_ind(raw_data.iloc[a[i], [3]], raw_data.iloc[a[j], [3]])))
