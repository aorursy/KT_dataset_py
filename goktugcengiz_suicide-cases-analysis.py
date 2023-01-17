# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
plt.style.use("seaborn-whitegrid") 
import seaborn as sns
from wordcloud import WordCloud

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#Load data
suicideDataRaw = pd.read_csv("../input/suicide-rates-overview-1985-to-2016/master.csv")
suicideData = suicideDataRaw.copy()
suicideData.ndim
suicideData.shape
suicideData.size
suicideData.columns
suicideData.rename(columns={'suicides_no':'suicides_number',
                            'suicides/100k pop':'suicides_per_100k_pop',
                            ' gdp_for_year ($) ':'gdp_for_year', 
                       'gdp_per_capita ($)':'gdp_per_capital'}, inplace=True)
suicideData.columns = suicideData.columns.str.replace(" ","_")
suicideData.columns = suicideData.columns.str.lower()
suicideData.columns
#Observations in the first 5 rows of the data set
suicideData.head()
#Remove country-year feature
suicideData.drop("country-year", axis = 1, inplace = True)
#5 random rows of observation inside the dataset
suicideData.sample(5)
#Some basic statistical details of the numerical values in the dataset
suicideData.describe().T
suicideData.info()
#Missing Value Detection by Features
suicideData.isnull().sum()
#Remove feature
suicideData.drop("hdi_for_year", axis = 1, inplace = True)
suicideData.gdp_for_year = suicideData.gdp_for_year.apply(lambda x: float(''.join(x.split(','))))
suicideData.age = suicideData.age.apply(lambda x: x.replace("years", ""))
suicideData.isnull().sum().sum()
suicideData.sample(5)
suicideData["country"].value_counts().head()
suicideData["country"].value_counts().tail()
suicideData["year"].value_counts().head()
suicideData["year"].value_counts().tail()
suicideData = suicideData.query("year != 2016")
suicideData["year"].value_counts().tail()
suicideData["sex"].value_counts()
suicideData["age"].value_counts()
suicideData["suicides_number"].value_counts().head()
suicideData["population"].value_counts().head()
suicideData["suicides_per_100k_pop"].value_counts().head()
suicideData["gdp_for_year"].value_counts().head()
suicideData["gdp_per_capital"].value_counts().head()
suicideData["generation"].value_counts()
#Average numbers of suicide by gender
suicideData.groupby("sex")["suicides_number"].mean()
#Average suicide rates by gender
suicideData.groupby("sex")["suicides_per_100k_pop"].mean()
#Average suicide rates by country
#The countries with the highest suicide rates are below
suicideData.groupby("country", as_index = False)["suicides_per_100k_pop"].mean().sort_values(by="suicides_per_100k_pop",ascending = False).head()
#Average suicide rates by country
#The countries with the lowest suicide rates are below
suicideData.groupby("country", as_index = False)["suicides_per_100k_pop"].mean().sort_values(by="suicides_per_100k_pop",ascending = False).tail()
#Average numbers of suicide by country
#The countries with the highest numbers of suicide are below
suicideData.groupby("country", as_index = False)["suicides_number"].mean().sort_values(by="suicides_number",ascending = False).head()
#Average numbers of suicide by country
#The countries with the lowest numbers of suicide are below
suicideData.groupby("country", as_index = False)["suicides_number"].mean().sort_values(by="suicides_number",ascending = False).tail()
#Average numbers of suicide by year
suicideData.groupby("year", as_index = False)["suicides_number"].mean().sort_values(by="suicides_number",ascending = False)
#Average suicide rates by year
suicideData.groupby("year", as_index = False)["suicides_per_100k_pop"].mean().sort_values(by="suicides_per_100k_pop",ascending = False)
#Average suicide numbers by age
suicideData.groupby("age", as_index = False)["suicides_number"].mean().sort_values(by="suicides_number",ascending = False)
#Average suicide rates by age
suicideData.groupby("age", as_index = False)["suicides_per_100k_pop"].mean().sort_values(by="suicides_per_100k_pop",ascending = False)
#Average suicide numbers by generation
suicideData.groupby("generation", as_index = False)["suicides_number"].mean().sort_values(by="suicides_number",ascending = False)
#Average suicide rates by generation
suicideData.groupby("generation", as_index = False)["suicides_per_100k_pop"].mean().sort_values(by="suicides_per_100k_pop",ascending = False)
#Average suicide numbers by gender and country
suicideData.pivot_table("suicides_number", index = "sex", columns = "country")
#Average suicide numbers by gender and year
suicideData.pivot_table("suicides_number", index = "sex", columns = "year")
numericVars = ["year","suicides_number", "population", "suicides_per_100k_pop", 
         "gdp_for_year", "gdp_per_capital"]
#Heat Map
sns.heatmap(suicideData[numericVars].corr(), annot = True, fmt = ".2f")
plt.show()
#Histogram
def plot_hist(variable):
    plt.figure(figsize = (9,3))
    plt.hist(suicideData[variable])
    plt.xlabel(variable)
    plt.ylabel("Frequency")
    plt.title("{} distribution with hist".format(variable))
    plt.show()

for i in numericVars:
    plot_hist(i)
#Gender show bar plot
sns.set(style='whitegrid')
ax=sns.barplot(x=suicideData['sex'].value_counts().index,
               y=suicideData['sex'].value_counts().values,
               palette="Blues_d",
               hue=['female','male'])
plt.legend(loc=8)
plt.xlabel('Gender')
plt.ylabel('Frequency')
plt.title('Show of Gender Bar Plot')
plt.show()
#Age show bar plot
sns.set(style='whitegrid')
ax=sns.barplot(x=suicideData['age'].value_counts().index,
               y=suicideData['age'].value_counts().values,
               palette="Blues_d")
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.title('Show of Age Bar Plot')
plt.show()
#Generation show bar plot
sns.set(style='whitegrid')
ax=sns.barplot(x=suicideData['generation'].value_counts().index,
               y=suicideData['generation'].value_counts().values,
               palette="Blues_d")
plt.xticks(rotation = 45)
plt.xlabel('Generation')
plt.ylabel('Frequency')
plt.title('Show of Generation Bar Plot')
plt.show()
#Countries show bar plot
f, ax = plt.subplots(figsize=(6, 15))
sns.set(style='whitegrid')
ax=sns.barplot(x=suicideData['country'].value_counts().values,
               y=suicideData['country'].value_counts().index,
               palette="Blues_d")
plt.xlabel('Frequency')
plt.ylabel('Country')
plt.title('Show of Country Bar Plot')
plt.show()
plt.figure(figsize=(8,5))
sns.barplot(x = "sex", 
            y = "suicides_number", 
            data = suicideData)
plt.xlabel('Gender')
plt.ylabel('Number of Suicide')
plt.title('Suicide Numbers by Gender')
plt.show()
plt.figure(figsize=(8,5))
sns.barplot(x = "sex", 
            y = "suicides_per_100k_pop", 
            data = suicideData)
plt.xlabel('Gender')
plt.ylabel('Suicide Ratio')
plt.title('Suicide Ratio by Gender')
plt.show()
countrySN = suicideData.groupby("country", as_index = False)["suicides_number"].mean().sort_values(by="suicides_number",ascending = False)
plt.figure(figsize=(6,15))
sns.barplot(x = "suicides_number", 
            y = "country", 
            data = countrySN)
plt.xlabel('Number of Suicide')
plt.ylabel('Country')
plt.title('Suicide Numbers by Countries')
plt.show()
countrySR = suicideData.groupby("country", as_index = False)["suicides_per_100k_pop"].mean().sort_values(by="suicides_per_100k_pop",ascending = False)
plt.figure(figsize=(6,15))
sns.barplot(x = "suicides_per_100k_pop", 
            y = "country", 
            data = countrySR)
plt.xlabel('Suicide Ratio')
plt.ylabel('Country')
plt.title('Suicide Ratio by Countries')
plt.show()
yearMeanSN = suicideData.groupby("year", as_index = False)["suicides_number"].mean()
plt.figure(figsize=(15,6))
sns.pointplot(x = "year", 
              y = "suicides_number", 
              data = yearMeanSN)
plt.xlabel('Year')
plt.ylabel('Average Number of Suicide')
plt.title('Average Suicide Numbers by Years')
plt.show()
yearSumSN = suicideData.groupby("year", as_index = False)["suicides_number"].sum()
plt.figure(figsize=(15,6))
sns.pointplot(x = "year", 
              y = "suicides_number", 
              data = yearSumSN)
plt.xlabel('Year')
plt.ylabel('Number of Suicide')
plt.title('Total Suicide Numbers by Years')
plt.show()
yearMeanSR = suicideData.groupby("year", as_index = False)["suicides_per_100k_pop"].mean()
plt.figure(figsize=(15,6))
sns.pointplot(x = "year", 
              y = "suicides_per_100k_pop", 
              data = yearMeanSR)
plt.xlabel('Year')
plt.ylabel('Suicide Ratio')
plt.title('Suicide Ratios by Years')
plt.show()
lithuania = suicideData[suicideData["country"] == "Lithuania"]
srilanka = suicideData[suicideData["country"] == "Sri Lanka"]
russia = suicideData[suicideData["country"] == "Russian Federation"]
hungary = suicideData[suicideData["country"] == "Hungary"]
belarus = suicideData[suicideData["country"] == "Belarus"]
topFiveCountries = pd.concat([lithuania, srilanka, russia, hungary, belarus], ignore_index = True)
topFive = topFiveCountries.groupby(["country", "year"], as_index = False)["suicides_per_100k_pop"].mean()

plt.figure(figsize=(15,8))
sns.lineplot(x="year", 
             y="suicides_per_100k_pop",
             hue="country",
             data=topFive)
plt.xlabel('Year')
plt.ylabel('Suicide Ratio')
plt.title('Suicide Ratios by Years')
plt.show()
ageDataSN = suicideData.groupby("age", as_index = False)["suicides_number"].mean().sort_values(by="suicides_number",ascending = False)
plt.figure(figsize=(8,5))
sns.barplot(x = "age", 
            y = "suicides_number", 
            data = ageDataSN)
plt.xlabel('Age Group')
plt.ylabel('Number of Suicide')
plt.title('Suicide Numbers by Age Groups')
plt.show()
ageDataSR = suicideData.groupby("age", as_index = False)["suicides_per_100k_pop"].mean().sort_values(by="suicides_per_100k_pop",ascending = False)
plt.figure(figsize=(8,5))
sns.barplot(x = "age", 
            y = "suicides_per_100k_pop", 
            data = ageDataSR)
plt.xlabel('Age Group')
plt.ylabel('Suicide Ratio')
plt.title('Suicide Ratio by Age Groups')
plt.show()
genDataSN = suicideData.groupby("generation", as_index = False)["suicides_number"].mean().sort_values(by="suicides_number",ascending = False)
plt.figure(figsize=(8,5))
sns.barplot(x = "generation",
            y = "suicides_number",
            data = genDataSN)
plt.xlabel('Generation')
plt.ylabel('Number of Suicide')
plt.title('Suicide Numbers by Generations')
plt.show()
genDataSR = suicideData.groupby("generation", as_index = False)["suicides_per_100k_pop"].mean().sort_values(by="suicides_per_100k_pop",ascending = False)
plt.figure(figsize=(8,5))
sns.barplot(x = "generation", 
            y = "suicides_per_100k_pop", 
            data = genDataSR)
plt.xlabel('Generations')
plt.ylabel('Suicide Ratio')
plt.title('Suicide Ratios by Generations')
plt.show()
csDataSN = suicideData.groupby(["country","sex"], as_index = False)["suicides_number"].mean().sort_values(by="suicides_number",ascending = False)
plt.figure(figsize=(8,25))
sns.barplot(x = "suicides_number", 
            y = "country", 
            hue="sex", 
            data = csDataSN)
plt.xlabel('Number of Suicide')
plt.ylabel('Country')
plt.title('Suicide Numbers by Countries and Gender')
plt.show()
csDataSR = suicideData.groupby(["country","sex"], as_index = False)["suicides_per_100k_pop"].mean().sort_values(by="suicides_per_100k_pop",ascending = False)
plt.figure(figsize=(8,25))
sns.barplot(x = "suicides_per_100k_pop", 
            y = "country", 
            hue="sex", 
            data = csDataSR)
plt.xlabel('Suicide Ratio')
plt.ylabel('Country')
plt.title('Suicide Ratios by Countries and Gender')
plt.show()
asDataSN = suicideData.groupby(["age","sex"], as_index = False)["suicides_number"].mean().sort_values(by="suicides_number",ascending = False)
plt.figure(figsize=(8,5))
sns.barplot(x = "age", 
            y = "suicides_number", 
            hue="sex", 
            data = asDataSN)
plt.xlabel('Age Group')
plt.ylabel('Suicide Number')
plt.title('Suicide Numbers by Age and Gender')
plt.show()
asDataSR = suicideData.groupby(["age","sex"], as_index = False)["suicides_per_100k_pop"].mean().sort_values(by="suicides_per_100k_pop",ascending = False)
plt.figure(figsize=(8,5))
sns.barplot(x = "age", 
            y = "suicides_per_100k_pop", 
            hue="sex", 
            data = asDataSR)
plt.xlabel('Age Group')
plt.ylabel('Suicide Ratio')
plt.title('Suicide Ratio by Age and Gender')
plt.show()
gsDataSN = suicideData.groupby(["generation","sex"], as_index = False)["suicides_number"].mean().sort_values(by="suicides_number",ascending = False)
plt.figure(figsize=(8,5))
sns.barplot(x = "generation", 
            y = "suicides_number", 
            hue="sex", 
            data = gsDataSN)
plt.xlabel('Generation')
plt.ylabel('Suicide Ratio')
plt.title('Suicide Ratio by Generation and Gender')
plt.show()
gsDataSR = suicideData.groupby(["generation","sex"], as_index = False)["suicides_per_100k_pop"].mean().sort_values(by="suicides_per_100k_pop",ascending = False)
plt.figure(figsize=(8,5))
sns.barplot(x = "generation", 
            y = "suicides_per_100k_pop", 
            hue="sex", 
            data = gsDataSR)
plt.xlabel('Generation')
plt.ylabel('Suicide Ratio')
plt.title('Suicide Ratio by Generation and Gender')
plt.show()
abc = suicideData.groupby(["country","year"], as_index= False)["suicides_per_100k_pop"].mean().sort_values(by = "suicides_per_100k_pop", ascending = False)
x95 = abc.country[abc.year == 1995]
plt.subplots(figsize=(12,12))
wordcloud = WordCloud(
                          background_color='white',
                          width=512,
                          height=384
                         ).generate(" ".join(x95))
plt.imshow(wordcloud)
plt.axis('off')
plt.savefig('graph.png')

plt.show()
#Suicide Numbers vs Population by Generation and Gender
plt.figure(figsize=(12,8))
ax = sns.scatterplot(x="suicides_number", 
                     y="population",
                     hue="generation", 
                     style="sex", 
                     data=suicideData)
plt.xlabel('Suicide Number')
plt.ylabel('Population')
plt.title('Suicide Numbers vs Population by Generation and Gender')
plt.show()
#Suicide Numbers vs GDP for Year by Generation and Gender
plt.figure(figsize=(12,8))
ax = sns.scatterplot(x="suicides_number", 
                     y="gdp_for_year",
                     hue="generation", 
                     style="sex", 
                     data=suicideData)
plt.xlabel('Suicide Number')
plt.ylabel('GDP for Year')
plt.title('Suicide Numbers vs GDP for Year by Generation and Gender')
plt.show()
#GDP for Capital vs GDP for Year by Generation and Gender
plt.figure(figsize=(12,8))
ax = sns.scatterplot(x="gdp_per_capital", 
                     y="gdp_for_year",
                     hue="generation", 
                     style="sex", 
                     data=suicideData)
plt.xlabel('GDP per Capital')
plt.ylabel('GDP for Year')
plt.title('GDP for Capital vs GDP for Year by Generation and Gender')
plt.show()
#GDP for Year vs Population by Generation and Gender
plt.figure(figsize=(12,8))
ax = sns.scatterplot(x="gdp_for_year", 
                     y="population",
                     hue="generation", 
                     style="sex", data=suicideData)
plt.xlabel('GDP for Year')
plt.ylabel('Population')
plt.title('GDP for Year vs Population by Generation and Gender')
plt.show()
tr = suicideData[suicideData["country"] == "Turkey"]
tr
#Year
trYearSum = tr.groupby("year", as_index= False)["suicides_number"].sum()
plt.figure(figsize=(15,6))
sns.pointplot(x = "year", 
              y = "suicides_number",
              color="#bb3f3f",
              data = trYearSum)
plt.xlabel('Year')
plt.ylabel('Total Number of Suicide')
plt.title('Total Suicide Numbers by Years in Turkey Republic')
plt.show()
#Gender
trGenderSum = tr.groupby("sex", as_index= False)["suicides_number"].sum()
trGenderSum
# Data to plot
labels = 'Men', 'Women'
sizes = [7562, 2569]
colors = ['lightskyblue', 'lightcoral']
explode = (0.1, 0)  # explode 1st slice

# Plot
plt.figure(figsize = (7,7))
plt.pie(sizes, explode=explode, labels=labels, colors=colors,
autopct='%1.1f%%', shadow=True, startangle=140)

plt.axis('equal')
plt.show()
#Age
trAgeSum = tr.groupby("age", as_index = False)["suicides_number"].sum().sort_values(by="suicides_number", ascending = False)
plt.figure(figsize=(8,5))
sns.barplot(x = "age", 
            y = "suicides_number", 
            data = trAgeSum)
plt.xlabel('Age Group')
plt.ylabel('Total Suicide Number')
plt.title('Total Suicide Numbers by Age Groups in Turkey Republic')
plt.show()
trAgeRateAvg = tr.groupby("age", as_index = False)["suicides_per_100k_pop"].mean().sort_values(by="suicides_per_100k_pop", ascending = False)
plt.figure(figsize=(8,5))
sns.barplot(x = "age", 
            y = "suicides_per_100k_pop", 
            data = trAgeRateAvg)
plt.xlabel('Age Group')
plt.ylabel('Average Suicide Ratio')
plt.title('Average Suicide Ratio by Age Groups in Turkey Republic')
plt.show()
#Generation
trGenSum = tr.groupby("generation", as_index = False)["suicides_number"].sum().sort_values(by="suicides_number", ascending = False)
plt.figure(figsize=(8,5))
sns.barplot(x = "generation", 
            y = "suicides_number", 
            data = trGenSum)
plt.xlabel('Generation')
plt.ylabel('Total Suicide Number')
plt.title('Total Suicide Numbers by Generations in Turkey Republic')
plt.show()
trGenRateAvg = tr.groupby("generation", as_index = False)["suicides_per_100k_pop"].mean().sort_values(by="suicides_per_100k_pop", ascending = False)
plt.figure(figsize=(8,5))
sns.barplot(x = "generation", 
            y = "suicides_per_100k_pop", 
            data = trGenRateAvg)
plt.xlabel('Generation')
plt.ylabel('Average Suicide Ratio')
plt.title('Average Suicide Ratio by Generation in Turkey Republic')
plt.show()