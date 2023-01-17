import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import os
import plotly
from plotly.graph_objs import Scatter, Figure, Layout

plotly.offline.init_notebook_mode(connected=True)

print(os.listdir("../input/"))

loans = pd.read_csv("../input/data-science-for-good-kiva-crowdfunding/kiva_loans.csv")
data_regions = pd.read_csv("../input/data-science-for-good-kiva-crowdfunding/kiva_mpi_region_locations.csv")
themes = pd.read_csv("../input/data-science-for-good-kiva-crowdfunding/loan_theme_ids.csv")
themes_regions = pd.read_csv("../input/data-science-for-good-kiva-crowdfunding/loan_themes_by_region.csv")

hdi = pd.read_csv("../input/human-development/human_development.csv")
mpi = pd.read_csv("../input/human-development/multidimensional_poverty.csv")
# look at the data
print(loans.shape)
loans.sample(3)
#checking for missing data
loans.isnull().sum()
print(data_regions.shape)
data_regions.sample(5)
data_regions.isnull().sum()
print(themes.shape)
themes.sample(5)
themes.isnull().sum()
print(themes_regions.shape)
themes_regions.sample(5)
themes_regions.isnull().sum()
print(hdi.shape)
hdi.sample(5)
print(mpi.shape)
mpi.sample(5)
# select only the loans that were funded
funded_loans = loans[loans['funded_time'].isnull()==False]
funded_loans.isnull().sum()
plt.figure(figsize=(10,8))
sns.set(style="whitegrid")
sns.set_color_codes("pastel")
sns.countplot(y=loans['sector'].where(loans['funded_time'].isnull()==False), order=loans['sector'].value_counts().index)
plt.title("Sectors which received loans", fontsize=20)
plt.xlabel('Number of loans', fontsize=18)
plt.ylabel('Sector', fontsize=18)
plt.xticks(fontsize=12)
plt.show()
plt.figure(figsize=(14,35))
sns.set(style="whitegrid")
sns.set_color_codes("pastel")
sns.countplot(y=loans['activity'], order=loans['activity'].value_counts().index)
plt.title("Activities which received loans", fontsize=20)
plt.xlabel('Number of loans', fontsize=18)
plt.ylabel('Activities', fontsize=18)
plt.xticks(fontsize=12)
plt.show()
plt.figure(figsize=(14,40))
plt.subplot(821)
agric_act = loans['activity'].where(loans['sector'] == "Agriculture")
sns.countplot(y=agric_act, order=agric_act.value_counts().iloc[0:10].index)
plt.title("Agriculture", fontsize=16)
plt.subplot(822)
food_act = loans['activity'].where(loans['sector'] == "Food")
sns.countplot(y=food_act, order=food_act.value_counts().iloc[0:10].index)
plt.title("Food", fontsize=16)
plt.subplot(823)
retl_act = loans['activity'].where(loans['sector'] == "Retail")
sns.countplot(y=retl_act, order=retl_act.value_counts().iloc[0:10].index)
plt.title("Retail", fontsize=16)
plt.subplot(824)
serv_act = loans['activity'].where(loans['sector'] == "Services")
sns.countplot(y=serv_act, order=serv_act.value_counts().iloc[0:10].index)
plt.title("Services", fontsize=16)
plt.subplot(825)
pruse_act = loans['activity'].where(loans['sector'] == "Personal Use")
sns.countplot(y=pruse_act, order=pruse_act.value_counts().iloc[0:10].index)
plt.title("Personal Use", fontsize=16)
plt.subplot(826)
house_act = loans['activity'].where(loans['sector'] == "Housing")
sns.countplot(y=house_act, order=house_act.value_counts().iloc[0:10].index)
plt.title("Housing", fontsize=16)
plt.subplot(827)
clth_act = loans['activity'].where(loans['sector'] == "Clothing")
sns.countplot(y=clth_act, order=clth_act.value_counts().iloc[0:10].index)
plt.title("Clothing", fontsize=16)
plt.subplot(828)
edu_act = loans['activity'].where(loans['sector'] == "Education")
sns.countplot(y=edu_act, order=edu_act.value_counts().iloc[0:10].index)
plt.title("Education", fontsize=16)
plt.subplot(829)
trans_act = loans['activity'].where(loans['sector'] == "Transportation")
sns.countplot(y=trans_act, order=trans_act.value_counts().iloc[0:10].index)
plt.title("Transportation", fontsize=16)
plt.subplot(8, 2, 10)
art_act = loans['activity'].where(loans['sector'] == "Arts")
sns.countplot(y=art_act, order=art_act.value_counts().iloc[0:10].index)
plt.title("Arts", fontsize=16)
plt.subplot(8, 2, 11)
hlth_act = loans['activity'].where(loans['sector'] == "Health")
sns.countplot(y=hlth_act, order=hlth_act.value_counts().iloc[0:10].index)
plt.title("Health", fontsize=16)
plt.subplot(8, 2, 12)
ctrn_act = loans['activity'].where(loans['sector'] == "Construction")
sns.countplot(y=ctrn_act, order=ctrn_act.value_counts().iloc[0:10].index)
plt.title("Construction", fontsize=16)
plt.subplot(8, 2, 13)
mnft_act = loans['activity'].where(loans['sector'] == "Manufacturing")
sns.countplot(y=mnft_act, order=mnft_act.value_counts().iloc[0:10].index)
plt.title("Manufacturing", fontsize=16)
plt.subplot(8, 2, 14)
etmt_act = loans['activity'].where(loans['sector'] == "Entertainment")
sns.countplot(y=etmt_act, order=etmt_act.value_counts().iloc[0:10].index)
plt.title("Entertainment", fontsize=16)
plt.subplot(8, 2, 15)
wlsl_act = loans['activity'].where(loans['sector'] == "Wholesale")
sns.countplot(y=wlsl_act, order=wlsl_act.value_counts().iloc[0:10].index)
plt.title("Wholesale", fontsize=16)
plt.figure(figsize=(10,8))
sns.distplot(loans['term_in_months'], bins=80)
plt.title("Loan term in months", fontsize=20)
plt.xlabel('Number of months', fontsize=18)
plt.xticks(fontsize=12)
plt.show()
plt.figure(figsize=(10,8))
sns.set(style="whitegrid")
sns.set_color_codes("pastel")
sns.countplot(y=loans['currency'], order=loans['currency'].value_counts().iloc[0:20].index)
plt.title("Top 20 most common currencies for loans", fontsize=20)
plt.xlabel('Number of loans', fontsize=18)
plt.ylabel('Currency', fontsize=18)
plt.xticks(fontsize=12)
plt.show()
plt.figure(figsize=(10,8))
sns.distplot(loans['loan_amount'], bins=80)
plt.title("Amount of money in the loans", fontsize=20)
plt.xlabel('Money (USD)', fontsize=18)
plt.xticks(fontsize=12)
plt.show()
plt.figure(figsize=(10,8))
sns.distplot(loans['lender_count'], bins=80)
plt.title("Amount of people who helped fund a loan", fontsize=20)
plt.xlabel('People', fontsize=18)
plt.xticks(fontsize=12)
plt.show()
avg = loans['lender_count'].sum() / len(loans)
print("Average amount of people per loan: ", avg)
plt.figure(figsize=(10,8))
plt.scatter(x=loans['lender_count'], y=loans['loan_amount'])
plt.title("Correlation between loan amount and people funding them", fontsize=20)
plt.xlabel('Number of lenders', fontsize=18)
plt.ylabel('Loan amount', fontsize=18)
plt.xticks(fontsize=12)
plt.show()

print("Pearson correlation:\n",np.corrcoef(loans['lender_count'], y=loans['loan_amount']))
loans['date'] = pd.to_datetime(loans['date'], format = "%Y-%m-%d")
plt.figure(figsize=(8,6))
sns.countplot(loans['date'].dt.year)
plt.title("Loans over the years", fontsize=20)
plt.xlabel('Year', fontsize=18)
plt.ylabel('Loans', fontsize=18)
plt.xticks(fontsize=12)
plt.show()
plt.figure(figsize=(8,6))
sns.countplot(loans['date'].dt.month)
plt.title("Quantity of loans per month", fontsize=20)
plt.xlabel('Month', fontsize=18)
plt.ylabel('Loans', fontsize=18)
plt.xticks(fontsize=12)
plt.show()
plt.figure(figsize=(10,6))
sns.countplot(loans['date'].dt.day)
plt.title("Quantity of loans per day of the month", fontsize=20)
plt.xlabel('Day', fontsize=18)
plt.ylabel('Loans', fontsize=18)
plt.xticks(fontsize=12)
plt.show()
sectors = loans['sector'].unique()
money_sec = []
loan_sec = []

for sec in sectors:
    money_sec.append(loans['loan_amount'].where(loans['sector']==sec).sum())
    loan_sec.append((loans['sector']==sec).sum())

df_sector = pd.DataFrame([sectors, money_sec, loan_sec]).T
plt.figure(figsize=(10,8))
sns.set(style="whitegrid")
sns.set_color_codes("pastel")
sns.barplot(x=df_sector[1], y=df_sector[0], order=loans['sector'].value_counts().index)
plt.title("Distribution of money per sectors", fontsize=20)
plt.ylabel('Sectors', fontsize=18)
plt.xlabel('Money (x10^8 USD)', fontsize=18)
plt.xticks(fontsize=12, rotation=90)
plt.show()
plt.figure(figsize=(10,8))
sns.set(style="whitegrid")
sns.set_color_codes("pastel")
sns.barplot(x=df_sector[1]/df_sector[2], y=df_sector[0], order=loans['sector'].value_counts().index)
plt.title("Average amount of money per loan", fontsize=20)
plt.ylabel('Sectors', fontsize=18)
plt.xlabel('Average money per loan (USD)', fontsize=18)
plt.xticks(fontsize=12, rotation=90)
plt.show()
activities = loans['activity'].unique()
money_act = []
loan_act = []

for act in activities:
    money_act.append(loans['loan_amount'].where(loans['activity']==act).sum())
    loan_act.append((loans['activity']==act).sum())

df_activity = pd.DataFrame([activities, money_act, loan_act]).T
plt.figure(figsize=(10,35))
sns.set(style="whitegrid")
sns.set_color_codes("pastel")
sns.barplot(x=df_activity[1], y=df_activity[0], order=loans['activity'].value_counts().index)
plt.title("Distribution of money per activity", fontsize=20)
plt.ylabel('Activities', fontsize=18)
plt.xlabel('Money (x10^8 USD)', fontsize=18)
plt.xticks(fontsize=12, rotation=90)
plt.show()
plt.figure(figsize=(10,35))
sns.set(style="whitegrid")
sns.set_color_codes("pastel")
sns.barplot(x=df_activity[1]/df_activity[2], y=df_activity[0], order=loans['activity'].value_counts().index)
plt.title("Average amount of money per activity", fontsize=20)
plt.ylabel('Activities', fontsize=18)
plt.xlabel('Average money per loan (USD)', fontsize=18)
plt.xticks(fontsize=12, rotation=90)
plt.show()
years = loans['date'].dt.year.unique()
months = loans['date'].dt.month.unique()
days = loans['date'].dt.day.unique()
money_year = []
loan_year = []
money_month = []
loan_month = []
money_day = []
loan_day = []

for year in years:
    money_year.append(loans['loan_amount'].where(loans['date'].dt.year==year).sum())
    loan_year.append((loans['date'].dt.year==year).sum())
    
for month in months:
    money_month.append(loans['loan_amount'].where(loans['date'].dt.month==month).sum())
    loan_month.append((loans['date'].dt.month==month).sum())
    
for day in days:
    money_day.append(loans['loan_amount'].where(loans['date'].dt.day==day).sum())
    loan_day.append((loans['date'].dt.day==day).sum())

df_year = pd.DataFrame([years, money_year, loan_year]).T
df_month = pd.DataFrame([months, money_month, loan_month]).T
df_day = pd.DataFrame([days, money_day, loan_day]).T
plt.figure(figsize=(8,6))
sns.set(style="whitegrid")
sns.set_color_codes("pastel")
sns.barplot(x=df_year[0], y=df_year[1])
plt.title("Money distribution per year", fontsize=20)
plt.xlabel('Years', fontsize=18)
plt.ylabel('Money (x10^8 USD)', fontsize=18)
plt.xticks(fontsize=12)
plt.show()
plt.figure(figsize=(8,6))
sns.set(style="whitegrid")
sns.set_color_codes("pastel")
sns.barplot(x=df_month[0], y=df_month[1])
plt.title("Money distribution per month", fontsize=20)
plt.xlabel('Month', fontsize=18)
plt.ylabel('Money (x10^7 USD)', fontsize=18)
plt.xticks(fontsize=12)
plt.show()
plt.figure(figsize=(14,6))
sns.set(style="whitegrid")
sns.set_color_codes("pastel")
sns.barplot(x=df_day[0], y=df_day[1])
plt.title("Money distribution per day of month", fontsize=20)
plt.xlabel('Day of month', fontsize=18)
plt.ylabel('Money (x10^7 USD)', fontsize=18)
plt.xticks(fontsize=12, rotation=90)
plt.show()
count_country = loans['country'].value_counts().dropna()
codes = loans['country_code'].unique()
countries = loans['country'].unique()
money_ctry = []

for c in countries:
    money_ctry.append(loans['funded_amount'].where(loans['country']==c).sum())
    
dataMap = pd.DataFrame([codes, countries, count_country[countries], money_ctry]).T
data = [ dict(
        type = 'choropleth',
        locations = dataMap[1],
        z = dataMap[2],
        text = dataMap[1],
        locationmode = 'country names',
        colorscale = [[0,"rgb(5, 10, 172)"],[0.35,"rgb(40, 60, 190)"],[0.5,"rgb(70, 100, 245)"],\
            [0.6,"rgb(90, 120, 245)"],[0.7,"rgb(106, 137, 247)"],[1,"rgb(220, 220, 220)"]],
        autocolorscale = False,
        reversescale = True,
        marker = dict(
            line = dict (
                color = 'rgb(180,180,180)',
                width = 0.5
            ) ),
        colorbar = dict(
            autotick = False,
            title = 'Loans per country'),
      ) ]

layout = dict(
    geo = dict(
        showframe = True,
        showcoastlines = True,
        projection = dict(
            type = 'Mercator'
        )
    )
)

fig = dict( data=data, layout=layout )
plotly.offline.iplot(fig, validate=False)
data2 = [ dict(
        type = 'choropleth',
        locations = dataMap[1],
        z = dataMap[3],
        text = dataMap[1],
        locationmode = 'country names',
        colorscale = [[0,"rgb(5, 10, 172)"],[0.35,"rgb(40, 60, 190)"],[0.5,"rgb(70, 100, 245)"],\
            [0.6,"rgb(90, 120, 245)"],[0.7,"rgb(106, 137, 247)"],[1,"rgb(220, 220, 220)"]],
        autocolorscale = False,
        reversescale = True,
        marker = dict(
            line = dict (
                color = 'rgb(180,180,180)',
                width = 0.5
            ) ),
        colorbar = dict(
            autotick = False,
            title = 'Money per country'),
      ) ]

layout2 = dict(
    geo = dict(
        showframe = True,
        showcoastlines = True,
        projection = dict(
            type = 'Mercator'
        )
    )
)

fig2 = dict( data=data2, layout=layout2 )
plotly.offline.iplot(fig2, validate=False)
data3 = [ dict(
        type = 'choropleth',
        locations = dataMap[1],
        z = dataMap[3]/dataMap[2],
        text = dataMap[1],
        locationmode = 'country names',
        colorscale = [[0,"rgb(5, 10, 172)"],[0.35,"rgb(40, 60, 190)"],[0.5,"rgb(70, 100, 245)"],\
            [0.6,"rgb(90, 120, 245)"],[0.7,"rgb(106, 137, 247)"],[1,"rgb(220, 220, 220)"]],
        autocolorscale = False,
        reversescale = True,
        marker = dict(
            line = dict (
                color = 'rgb(180,180,180)',
                width = 0.5
            ) ),
        colorbar = dict(
            autotick = False,
            title = 'Average amount per loan'),
      ) ]

layout3 = dict(
    geo = dict(
        showframe = True,
        showcoastlines = True,
        projection = dict(
            type = 'Mercator'
        )
    )
)

fig3 = dict( data=data3, layout=layout3 )
plotly.offline.iplot(fig3, validate=False)
loans[loans['country']=='Cote D\'Ivoire']
data4 = [ dict(
        type = 'choropleth',
        locations = hdi['Country'],
        z = hdi['Human Development Index (HDI)'],
        text = hdi['Country'],
        locationmode = 'country names',
        colorscale = [[0,"rgb(5, 10, 172)"],[0.35,"rgb(40, 60, 190)"],[0.5,"rgb(70, 100, 245)"],\
            [0.6,"rgb(90, 120, 245)"],[0.7,"rgb(106, 137, 247)"],[1,"rgb(220, 220, 220)"]],
        autocolorscale = False,
        reversescale = True,
        marker = dict(
            line = dict (
                color = 'rgb(180,180,180)',
                width = 0.5
            ) ),
        colorbar = dict(
            autotick = False,
            title = 'HDI'),
      ) ]

layout4 = dict(
    geo = dict(
        showframe = True,
        showcoastlines = True,
        projection = dict(
            type = 'Mercator'
        )
    )
)

fig4 = dict(data=data4, layout=layout4)
plotly.offline.iplot(fig4, validate=False)
hdi_temp = []
leb_temp = []
mye_temp = []
gni_temp = []
mpi_temp = []

for c in dataMap[1]:
    h = hdi[hdi['Country']==c]
    m = mpi[mpi['Country']==c]
    hdi_temp.append(h['Human Development Index (HDI)'].values)
    leb_temp.append(h['Life Expectancy at Birth'].values)
    mye_temp.append(h['Mean Years of Education'].values)
    gni_temp.append(h['Gross National Income (GNI) per Capita'].values)
    mpi_temp.append(m['Multidimensional Poverty Index (MPI, 2010)'].values)

hdi_kiva = []
leb_kiva = []
mye_kiva = []
gni_kiva = []
mpi_kiva = []
c_hdi_kiva = []
c_mpi_kiva = []

for i in range(0, len(dataMap[1])):
    if hdi_temp[i].size:
        hdi_kiva.append(hdi_temp[i][0])
        leb_kiva.append(leb_temp[i][0])
        mye_kiva.append(mye_temp[i][0])
        gni_kiva.append(gni_temp[i][0])
        c_hdi_kiva.append(dataMap[2][i])
    if mpi_temp[i].size:
        mpi_kiva.append(mpi_temp[i][0])
        c_mpi_kiva.append(dataMap[2][i])

df_hdi = pd.DataFrame([c_hdi_kiva, hdi_kiva, leb_kiva, mye_kiva, gni_kiva, c_mpi_kiva, mpi_kiva]).T
plt.figure(figsize=(8,6))
plt.scatter(x=df_hdi[0], y=df_hdi[1])
plt.title("Correlation between quantity of loans per country and its HDI", fontsize=20)
plt.xlabel('Number of loans', fontsize=18)
plt.ylabel('HDI', fontsize=18)
plt.xticks(fontsize=12)
plt.show()
plt.figure(figsize=(16,6))
plt.subplot(121)
plt.scatter(x=df_hdi[0], y=df_hdi[2])
plt.title("Life Expectancy at Birth", fontsize=20)
plt.xlabel('Number of loans', fontsize=18)
plt.ylabel('Life Expectancy at Birth (years)', fontsize=18)
plt.xticks(fontsize=12)

plt.subplot(122)
plt.scatter(x=df_hdi[0], y=df_hdi[2])
plt.title("Life Expectancy at Birth - less than 45k loans", fontsize=20)
plt.xlabel('Number of loans', fontsize=18)
plt.ylabel('Life Expectancy at Birth (years)', fontsize=18)
plt.axis([-2500, 45000, 48, 85])
plt.xticks(fontsize=12)
plt.show()
plt.figure(figsize=(16,6))
plt.subplot(121)
plt.scatter(x=df_hdi[0], y=df_hdi[3])
plt.title("Mean Years of Education", fontsize=20)
plt.xlabel('Number of loans', fontsize=18)
plt.ylabel('Mean Years of Education (years)', fontsize=18)
plt.xticks(fontsize=12)

plt.subplot(122)
plt.scatter(x=df_hdi[0], y=df_hdi[3])
plt.title("Mean Years of Education - less than 45k loans", fontsize=20)
plt.xlabel('Number of loans', fontsize=18)
plt.ylabel('Mean Years of Education (years)', fontsize=18)
plt.axis([-2500, 45000, 1, 14])
plt.xticks(fontsize=12)
plt.show()
plt.figure(figsize=(16,6))
plt.subplot(121)
plt.scatter(x=df_hdi[5], y=df_hdi[6])
plt.title("Multidimensional Poverty Index", fontsize=20)
plt.xlabel('Number of loans', fontsize=18)
plt.ylabel('Multidimensional Poverty Index', fontsize=18)
plt.xticks(fontsize=12)

plt.subplot(122)
plt.scatter(x=df_hdi[5], y=df_hdi[6])
plt.title("Multidimensional Poverty Index - less than 45k loans", fontsize=20)
plt.xlabel('Number of loans', fontsize=18)
plt.ylabel('Multidimensional Poverty Index', fontsize=18)
plt.axis([-2500, 45000, -0.05, 0.6])
plt.xticks(fontsize=12)
plt.show()
# separate the Kiva data from the Philippines
phil = loans[loans['country'] == 'Philippines']
print(phil.shape)
phil.sample(3)
plt.figure(figsize=(10,8))
sns.set(style="whitegrid")
sns.set_color_codes("pastel")
sns.countplot(y=phil['sector'], order=phil['sector'].value_counts().index)
plt.title("Sectors which received loans in the Philippines", fontsize=20)
plt.xlabel('Number of loans', fontsize=18)
plt.ylabel('Sector', fontsize=18)
plt.xticks(fontsize=12)
plt.show()
plt.figure(figsize=(14,6))
plt.subplot(131)
retl_phil = phil['activity'].where(phil['sector'] == "Retail")
sns.countplot(y=retl_phil, order=retl_phil.value_counts().iloc[0:10].index)
plt.title("Retail", fontsize=16)
plt.subplot(132)
food_phil = phil['activity'].where(phil['sector'] == "Food")
sns.countplot(y=food_phil, order=food_phil.value_counts().iloc[0:10].index)
plt.title("Food", fontsize=16)
plt.subplot(133)
agric_phil = phil['activity'].where(phil['sector'] == "Agriculture")
sns.countplot(y=agric_phil, order=agric_phil.value_counts().iloc[0:10].index)
plt.title("Agriculture", fontsize=16)
plt.show()
