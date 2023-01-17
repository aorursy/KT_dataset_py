# Loading libraries

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.offline as ply
ply.init_notebook_mode(connected=True)
from mpl_toolkits.basemap import Basemap

kiva_loans_data = pd.read_csv("../input/kiva_loans.csv")
kiva_mpi_locations_data = pd.read_csv("../input/kiva_mpi_region_locations.csv")
loan_theme_ids_data = pd.read_csv("../input/loan_theme_ids.csv")
loan_themes_by_region_data = pd.read_csv("../input/loan_themes_by_region.csv")
kiva_loans_data.head()
kiva_mpi_locations_data.head()
loan_theme_ids_data.head()
loan_themes_by_region_data.head()
kiva_loans_data.drop(['id'], axis=1).describe()
kiva_loans_data.drop(['id'], axis=1).describe(include=['O'])
kiva_mpi_locations_data.drop(['geo'], axis=1).describe(include=['O'])
loan_themes_by_region_data.drop(['Loan Theme ID', 'geocode_old', 'geocode', 'geo', 'mpi_geo'], axis=1).describe(include=['O'])
plt.figure(figsize=(13,9))
sectors = kiva_loans_data['sector'].value_counts()
sns.barplot(y=sectors.index, x=sectors.values)
plt.xlabel('Number of loans', fontsize=20)
plt.ylabel("Sectors", fontsize=20)
plt.title("Number of loans per sector", size=30)
plt.show()
gender = kiva_loans_data[['borrower_genders']]
gender = gender[gender["borrower_genders"].notnull()]
gender = gender["borrower_genders"].str.upper()
gender = gender.str.replace("FEMALE","F")
gender = gender.str.replace("MALE","M")
gender = pd.DataFrame(gender)
gender["F"] = gender["borrower_genders"].str.count("F")
gender["M"] = gender["borrower_genders"].str.count("M")
gender_list = [sum(gender["M"]),sum(gender["F"])]
gender_label = ["MALE","FEMALE"]
plt.figure(figsize=(7,7))
plt.pie(gender_list,labels=gender_label,shadow=True,colors = ["blue","pink"],autopct="%1.0f%%",
        explode=[0,.1],wedgeprops={"linewidth":2,"edgecolor":"k"})
plt.title("GENDER DISTRIBUTION")
kiva_loans_data.borrower_genders = kiva_loans_data.borrower_genders.astype(str)
gender_data = pd.DataFrame(kiva_loans_data.borrower_genders.str.split(',').tolist())
kiva_loans_data['sex_borrowers'] = gender_data[0]
kiva_loans_data.loc[kiva_loans_data.sex_borrowers == 'nan', 'sex_borrowers'] = np.nan
sex_mean = pd.DataFrame(kiva_loans_data.groupby(['sex_borrowers'])['funded_amount'].mean().sort_values(ascending=False)).reset_index()
print(sex_mean)
g1 = sns.barplot(x='sex_borrowers', y='funded_amount', data=sex_mean)
g1.set_title("Mean funded Amount by Gender ", fontsize=15)
g1.set_xlabel("Gender")
g1.set_ylabel("Average funded Amount(US)", fontsize=12)
f, ax = plt.subplots(figsize=(15, 8))
sns.countplot(x="sex_borrowers", hue='repayment_interval', data=kiva_loans_data).set_title('sex borrowers with repayment_intervals');
plt.figure(figsize=(15,8))
count = kiva_loans_data['repayment_interval'].value_counts().head(10)
sns.barplot(count.values, count.index, )
for i, v in enumerate(count.values):
    plt.text(0.8,i,v,color='k',fontsize=19)
plt.xlabel('Count', fontsize=12)
plt.ylabel('Types of repayment interval', fontsize=12)
plt.title("Types of repayment intervals with their count", fontsize=16)
funded = pd.DataFrame(kiva_loans_data, columns=['funded_amount', 'loan_amount'])
funded['isFunded'] = np.where((kiva_loans_data['funded_amount'] < kiva_loans_data['loan_amount']), 0, 1)

notFunded = 0
Funded=0

for x in funded['isFunded']:
    if x == 0:
        notFunded += 1
    else :
        Funded+=1
    notFunded=notFunded
    Funded=Funded
    
arr=[Funded,notFunded]

objects = ('Repaid', 'NotRepaid')
y_pos = np.arange(len(objects))

plt.bar(y_pos, arr, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel('Amount of loans')

plt.show()
plt.figure(figsize=(12,10))

country_count=kiva_loans_data['country'].value_counts()
top_country=country_count.head(10)
sns.barplot(top_country.values,top_country.index)

plt.xlabel('Loan Counts',fontsize=12)
plt.ylabel('Country Name',fontsize=12)
plt.title('Top countries to take loan from Kiva',fontsize=18)
plt.show()
plt.figure(figsize=(12,6))
sns.distplot(kiva_loans_data[kiva_loans_data['loan_amount'] < kiva_loans_data['loan_amount'].quantile(.95) ]['loan_amount'])
plt.show()
plt.figure(figsize=(15,8))
count = kiva_mpi_locations_data['world_region'].value_counts()
sns.barplot(count.values, count.index, )
plt.xlabel('Count', fontsize=12)
plt.ylabel('world region name', fontsize=12)
plt.title("Distribution of world regions", fontsize=16)
plt.figure(figsize=(15,8))
count = kiva_loans_data['activity'].value_counts().head(30)
sns.barplot(count.values, count.index)
plt.xlabel('Count', fontsize=12)
plt.ylabel('Activity name?', fontsize=12)
plt.title("Top Loan Activity type", fontsize=16)
map = Basemap()

lat = kiva_mpi_locations_data["lat"].tolist()
lon = kiva_mpi_locations_data["lon"].tolist()

x,y = map(lon,lat)

plt.figure(figsize=(15,8))
map.plot(x,y,"go",color ="orange",markersize =6,alpha=.6)
map.shadedrelief()
data = [ dict(
        type = 'scattergeo',
        lat = kiva_mpi_locations_data['lat'],
        lon = kiva_mpi_locations_data['lon'],
        text = kiva_mpi_locations_data['LocationName'],
        marker = dict(
             size = 10,
             line = dict(
                width=1,
                color='rgba(150, 150, 150)'
            ),
            cmin = 0,
            color = kiva_mpi_locations_data['MPI'],
            cmax = kiva_mpi_locations_data['MPI'].max(),
            colorbar=dict(
                title="Poverty Index"
            )
        ))]
layout = dict(title = 'Poverty Index')
fig = dict( data=data, layout=layout )
ply.iplot(fig)