import matplotlib.pyplot as plt

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import pylab

import seaborn as sns





plt.style.use('fivethirtyeight')

%matplotlib inline

pylab.rcParams['figure.figsize'] = (10.0, 8.0)
df = pd.read_csv("../input/multipleChoiceResponses.csv", encoding="ISO-8859-1")

df.head()
len(df[pd.isnull(df.GenderSelect)])
plot = df[df.GenderSelect.isnull() == False].groupby(df.GenderSelect).GenderSelect.count().plot.bar()

plot = plt.title("Number of Respondents by Gender")
filtered_df = df[(df.GenderSelect.isnull() == False) & (df.Country.isnull() == False)]
def getFemaleMaleRatio(df):

    counts_by_gender = df.groupby('GenderSelect').GenderSelect.count()

    return counts_by_gender[0]/counts_by_gender[1]
group_by_country = filtered_df.groupby(df.Country)

ratios = group_by_country.apply(getFemaleMaleRatio)

print("Maximum Female/Male Ratio: ", ratios.idxmax(), ratios.max())

print("Minimum Female/Male Ratio: ", ratios.idxmin(), ratios.min())
fig, ax = plt.subplots()

df[df.GenderSelect == 'Male'].Age.plot.hist(bins=100, ax=ax, alpha=0.5)

df[df.GenderSelect == 'Female'].Age.plot.hist(bins=100, ax=ax, alpha=0.8)

legend = ax.legend(['Male', 'Female'])

plot = plt.title("Age distribution for Male and Female Data Scientists")
fig, ax = plt.subplots()

df[(df.GenderSelect == 'Male') & (df.Age > 60)].Age.plot.hist(ax=ax, alpha=0.5)

df[(df.GenderSelect == 'Female') & (df.Age > 60)].Age.plot.hist(ax=ax, alpha=0.8)

legend = ax.legend(['Male', 'Female'])

plot = plt.title("Age Distribution for Male and Female Data Scientists above 60 years of age")
fig, ax = plt.subplots()

df[(df.GenderSelect == 'Male') & (df.StudentStatus == 'Yes')].Age.plot.hist(bins=30, ax=ax, alpha=0.5)

df[(df.GenderSelect == 'Female') & (df.StudentStatus == 'Yes')].Age.plot.hist(bins=30, ax=ax, alpha=0.8)

legend = ax.legend(['Male', 'Female'])

plot = plt.title("Age Distribution for Male and Female Student Respondents")
counts_by_gender = df.groupby([df.GenderSelect, df.EmploymentStatus]).size().reset_index(name="Total")
n_male = len(df[df.GenderSelect == "Male"])

n_female = len(df[df.GenderSelect == "Female"])

n_diff_identity = len(df[df.GenderSelect == "A different identity"])

n_other = len(df[df.GenderSelect == "Non-binary, genderqueer, or gender non-conforming"])

print(n_male, n_female, n_diff_identity, n_other)
counts_by_gender_plot = counts_by_gender.pivot("GenderSelect", "EmploymentStatus", "Total")

ax = sns.heatmap(counts_by_gender_plot, linewidths=.5, cmap="Blues")

plot = plt.title("Heatmap of Absolute number of people across Gender & Employment Status")
relative_counts = df.groupby([df.GenderSelect, df.EmploymentStatus]).size().groupby(level=0).apply(lambda x:

                                                 100 * x / float(x.sum())).reset_index(name="percentage")
relative_counts_by_gender_plot = relative_counts.pivot("GenderSelect", "EmploymentStatus", "percentage")

ax = sns.heatmap(relative_counts_by_gender_plot, linewidths=.5, cmap="Blues")

plot = plt.title("Heatmap of Relative number of people across Gender who are in each Employment Category")
jobs_by_gender = df[["GenderSelect", "CurrentJobTitleSelect"]].groupby([df.CurrentJobTitleSelect, df.GenderSelect]).size().reset_index(name="number")

from matplotlib import pyplot



chart = sns.factorplot(x='CurrentJobTitleSelect', y='number', hue='GenderSelect', data=jobs_by_gender, kind='bar', size=15, aspect=2, legend=False)

for ax in plt.gcf().axes:

    ax.set_xlabel("Job Title", fontsize=35)

    ax.set_ylabel("Count", fontsize=35)



for ax in chart.axes.flatten(): 

    ax.set_xticklabels(ax.get_xticklabels(),rotation=90, fontsize=25) 

    ax.set_yticklabels(ax.get_yticklabels(),rotation=0, fontsize=25) 



plot = plt.legend(loc='upper left',  prop={'size': 20})

plot = plt.title("Number of people with Different Job Titles by Gender", fontsize=30)
relative_jobs_by_gender = df[["GenderSelect", "CurrentJobTitleSelect"]].groupby([df.CurrentJobTitleSelect, df.GenderSelect]).size().groupby(level=0).apply(lambda x:

                                                 100 * x / float(x.sum())).reset_index(name="percentage")
values = relative_jobs_by_gender.groupby([relative_jobs_by_gender.CurrentJobTitleSelect, relative_jobs_by_gender.GenderSelect]).percentage.sum().unstack()

values = values[['Male', 'Female', 'A different identity', 'Non-binary, genderqueer, or gender non-conforming']]

values = values.sort_values(by=values.columns[0], axis=0)
plot = values.plot.bar(stacked=True)

plot = plt.title("Relative Number of people from each gender within a job title")
conversion_rates = pd.read_csv("../input/conversionRates.csv")

conversion_rates = conversion_rates.set_index("originCountry").T
def getConversionRate(currency):

    """

        Returns conversion rate for the given currency to USD

        If the currency is not in the conversion table assumes it is USD

        and returns 1

    """

    if currency not in conversion_rates.columns:

        return 1

    return conversion_rates[currency].exchangeRate
def processCompensation(row):

    compensation = row["CompensationAmount"]

    if type(compensation) == type("str"):

        try:

            result = float(compensation.replace(",", ""))

            

            if result < 0:

                result =  -result

            row["CompensationAmount"] = result * getConversionRate(row["CompensationCurrency"])

        except Exception as e:

            row["CompensationAmount"] = np.nan

    return row

df.CompensationAmount = df.apply(processCompensation, axis=1).CompensationAmount
ninetyninth_percentile = df.CompensationAmount.quantile(0.99)

first_percentile = df.CompensationAmount.quantile(.01)

df.CompensationAmount = df[((df.CompensationAmount < ninetyninth_percentile) & (df.CompensationAmount > first_percentile))].CompensationAmount

df[df.GenderSelect == "Male"].CompensationAmount.plot.hist(bins=10, alpha=0.5)

df[df.GenderSelect == "Female"].CompensationAmount.plot.hist(bins=10, alpha=0.8)
fig, ax = plt.subplots()

df[(df.GenderSelect == 'Male') & (df.CompensationAmount.isnull() == False ) & (df.CompensationAmount != 0.0)].groupby([df.Age]).CompensationAmount.median().plot.bar( ax=ax, alpha=0.5)

df[(df.GenderSelect == 'Female') & (df.CompensationAmount.isnull() == False) & ((df.CompensationAmount != 0.0))].groupby([df.Age]).CompensationAmount.median().plot.bar(ax=ax, alpha=0.3, color="red")



legend = ax.legend(['Male', 'Female'])

plot = plt.title("Median Compensation distribution for Male and Female Data Scientists by Age")