import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
countries = pd.read_csv("../input/countries of the world.csv", decimal=",")
countries = countries.applymap(lambda x: x.strip() if type(x) is str else x)
#countries
# find the number of columns and rows
print("No of rows: ", len(countries.columns))
print("No of columns: ", len(countries.index))
# using an index to filter rows and columns
area = countries.iloc[:, 3]
# using column names
name = countries["Country"]

plt.plot(area, color='green')
plt.title("World Countries Area Comparision ")
plt.xlabel("Country Index")
plt.ylabel("Area in Sq.ft.")
plt.show()
between0and50 = countries[(countries.index >= 0) & (countries.index <= 50)]
between160and250 = countries[(countries.index >= 160) & (countries.index <= 250)]

plt.figure(figsize = (10, 5))
plt.title("Showing outliers using boxplot")
plt.subplot(1, 2, 1)
plt.boxplot(between0and50['Area (sq. mi.)'])
plt.ylabel("Area (sq. mi.)")
plt.title("Showing 4 outliers")
plt.subplot(1, 2, 2)
plt.boxplot(between160and250['Area (sq. mi.)'])
plt.ylabel("Area (sq. mi.)")
plt.title("Showing 2 outliers")
plt.show()
index_of_max_area = between0and50["Area (sq. mi.)"].idxmax()
print("{} is the biggest, from 0 to 50".format(between0and50.loc[index_of_max_area]["Country"]))
fourcountries = between0and50.sort_values(by=['Area (sq. mi.)'], ascending=False).iloc[:, 0:5].head(4)
fourcountries
index_of_max_area = between160and250["Area (sq. mi.)"].idxmax()
print("{} is the biggest, from 160 to 250".format(between160and250.loc[index_of_max_area]["Country"]))
twocountries = between160and250.sort_values(by=['Area (sq. mi.)'], ascending=False).iloc[:, 0:5].head(2)
twocountries
sixcountries = [fourcountries, twocountries]
sixcountries = pd.concat(sixcountries)

plt.figure(figsize = (10, 5))

plt.subplot(1, 2, 1)
plt.pie(sixcountries['Area (sq. mi.)'], labels=sixcountries['Country'],autopct='%1.1f%%')

plt.subplot(1, 2, 2)
plt.pie(sixcountries['Pop. Density (per sq. mi.)'], labels=sixcountries['Country'],autopct='%1.1f%%')

plt.show()
corrs = countries.corr(method='pearson', min_periods=1)

# GDP is highly(positively) correlated with phone, 0.8
gdp_with_phone = corrs.sort_values(by=['GDP ($ per capita)'], ascending=False).iloc[1:2,:]['GDP ($ per capita)']
print ("corr: gdp_phone: ")
print(gdp_with_phone[0])

# birthrate is highly(negatively) correlated with literacy 0.79
br_with_literacy = corrs.sort_values(by=['Birthrate'], ascending=True).iloc[1:2,:]['Birthrate']
print ("corr: brithrate_literacy: " )
print(br_with_literacy[0])

# Now let us plot the scatter plots
plt.figure(figsize = (10, 5))

plt.subplot(1, 2, 1)
plt.title("GDP vs Phones")
plt.xlabel("GDP ($ per capita)")
plt.ylabel("Phones (per 1000)")
plt.scatter(countries["GDP ($ per capita)"], countries["Phones (per 1000)"], alpha=0.5)

plt.subplot(1, 2, 2)
plt.title("Birthrate vs Literacy")
plt.xlabel("Birthrate")
plt.ylabel("Literacy (%)")
plt.scatter(countries["Birthrate"], countries["Literacy (%)"], alpha=0.5)

plt.show()
print("There are {} regions".format(len(set(countries.sort_values(by=["Region"], ascending=False)["Region"]))))
print("These are the 11 regions")
print(set(countries.sort_values(by=["Region"], ascending=False)["Region"]))
countries = countries.loc[countries["Region"].isin(['ASIA (EX. NEAR EAST)','SUB-SAHARAN AFRICA', 'WESTERN EUROPE'])]
# I sorted them starting from Asia to Western Europe
countries = countries.sort_values(by=["Region"], ascending=True)
countries = countries.reset_index()
cat1 = countries.loc[(countries['Literacy (%)'] >= 0) & (countries['Literacy (%)'] <= 40)]
cat2 = countries.loc[(countries['Literacy (%)'] >= 41) & (countries['Literacy (%)'] <= 60)]
cat3 = countries.loc[(countries['Literacy (%)'] >= 61) & (countries['Literacy (%)'] <= 80)]
cat4 = countries.loc[countries['Literacy (%)'] >= 81]
plt.figure(figsize=(10,5))
plt.title("Data Classified by Literacy Level(%)")
plt.xlabel("Countries")
plt.ylabel("Literacy (%)")
plt.scatter(cat1.index, cat1["Literacy (%)"], color = "g" , label = "[0, 40]")
plt.scatter(cat2.index, cat2["Literacy (%)"], color = "b", label = "[41, 60]")
plt.scatter(cat3.index, cat3["Literacy (%)"], color = "r", label = "[61, 80]")
plt.scatter(cat4.index, cat4["Literacy (%)"], color = "y", label = "> 81")
# Separator between ASIA and Sub-Saharan Africa
plt.axvline(x=27, ymin=0, ymax = 100, linewidth=1, color='k')
# Separator between Sub-Saharan Africa and Western Europe
plt.axvline(x=78, ymin=0, ymax = 100, linewidth=1, color='k')
#adding text labels to the categories
plt.text(2, 0.001, "ASIA (EX. NEAR EAST)")
plt.text(40, 0.001, "SUB-SAHARAN AFRICA")
plt.text(80, 0.001, "WESTERN EUROPE")

plt.legend(loc=4)
plt.show()
