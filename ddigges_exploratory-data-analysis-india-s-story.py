import matplotlib.pyplot as plt
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pylab
import seaborn as sns
plt.style.use('fivethirtyeight')
%matplotlib inline
pylab.rcParams['figure.figsize'] = (10.0, 8.0)
loans_df = pd.read_csv("../input/data-science-for-good-kiva-crowdfunding/kiva_loans.csv", parse_dates=['disbursed_time', 'funded_time', 'posted_time'])
loan_theme_ids_df = pd.read_csv("../input/data-science-for-good-kiva-crowdfunding/loan_theme_ids.csv")
loan_themes_df = pd.read_csv("../input/data-science-for-good-kiva-crowdfunding/loan_themes_by_region.csv")
loan_coords_df = pd.read_csv("../input/additional-kiva-snapshot/loan_coords.csv")
loan_coords_df.columns = ['id', 'latitude', 'longitude']
loans_df.shape
loans_df.head()
# From: https://deparkes.co.uk/2016/11/04/sort-pandas-boxplot/
def boxplot_sorted(df, by, column):
    # use dict comprehension to create new dataframe from the iterable groupby object
    # each group name becomes a column in the new dataframe
    df2 = pd.DataFrame({col:vals[column] for col, vals in df.groupby(by)})
    # find and sort the median values in this new dataframe
    meds = df2.median().sort_values()
    # use the columns in the dataframe, ordered sorted by median value
    # return axes so changes can be made outside the function
    return df2[meds.index].plot(kind='box', logy=True)
pylab.rcParams['figure.figsize'] = (8.0, 25.0)
plt.style.use('fivethirtyeight')
loans_df.groupby(loans_df.country).id.count().sort_values().plot.barh(color='cornflowerblue');
plt.ylabel('Loan Count')
plt.title("Loan Count by Country");
pylab.rcParams['figure.figsize'] = (6.0, 6.0)
loans_df.groupby(loans_df.sector).id.count().sort_values().plot.bar(color='cornflowerblue');
plt.ylabel('Loan Count')
plt.title("Loan Count by Sector");
loans_df.term_in_months.plot.hist(bins=100);
plt.ylabel('Loan Count')
plt.title("Loan Count by Loan Duration");
loans_df.lender_count.plot(kind='box', logy=True);
plt.title("Distribution of Number of Lenders per loan");
plt.xlabel("Number of lenders in powers of 10");
loans_df[loans_df.lender_count > 150].shape[0]/loans_df.shape[0]
axes = plt.gca()
axes.set_xlim([0,150])
loans_df.lender_count.plot.hist(bins=1000);
plt.xlabel('Number of Lenders')
plt.title("Distribution of Number of Lenders where number < 150");
max(loans_df.lender_count)
loans_df[loans_df.lender_count == max(loans_df.lender_count)]
def process_gender(x):
    
    if type(x) is float and np.isnan(x):
        return "nan"
    genders = x.split(",")
    male_count = sum(g.strip() == 'male' for g in genders)
    female_count = sum(g.strip() == 'female' for g in genders)
    
    if(male_count > 0 and female_count > 0):
        return "MF"
    elif(female_count > 0):
        return "F"
    elif (male_count > 0):
        return "M"
loans_df.borrower_genders = loans_df.borrower_genders.apply(process_gender)
loans_df.borrower_genders.value_counts().plot.bar(color='cornflowerblue');
plt.xlabel('Borrower Group/Individual Gender')
plt.ylabel('Count')
plt.title("Loan Count by Gender of Borrower");
loans_df.funded_amount.plot(kind='box', logy=True);
plt.title("Distribution of Loan Funded Amount");
loans_df.funded_amount.describe()
# Q3 + 1.5 * IQR
IQR = loans_df.funded_amount.quantile(0.75) - loans_df.funded_amount.quantile(0.25)
upper_whisker = loans_df.funded_amount.quantile(0.75) + 1.5 * IQR
loans_above_upper_whisker = loans_df[loans_df.funded_amount > upper_whisker]
loans_above_upper_whisker.shape
# percentage of loans above upper whisker
loans_above_upper_whisker.shape[0]/loans_df.shape[0]
loans_zero = loans_df[loans_df.funded_amount == 0]
print("Number of unfunded loans", loans_zero.shape)
print("% of unfunded loans", loans_zero.shape[0]/loans_df.shape[0])
loans_below_upper_whisker = loans_df[loans_df.funded_amount < upper_whisker]
loans_below_upper_whisker.funded_amount.plot.hist();
plt.xlabel('Funded Amount')
plt.title("Distribution of Loan Funded amount < $2000");
df = loans_above_upper_whisker[loans_above_upper_whisker.funded_amount < 20000]
df.funded_amount.plot.hist();
plt.xlabel('Funded Amount')
plt.title("Distribution of Loan Funded Amount between \$2,000 and \$20,000");
df.shape
df = loans_above_upper_whisker[(loans_above_upper_whisker.funded_amount > 20000) & (loans_above_upper_whisker.funded_amount < 60000)]
df.funded_amount.plot.hist()
plt.xlabel('Funded Amount')
plt.title("Distribution of Loan Funded Amount between \$20,000 and \$60,000");
df.sector.value_counts().sort_values().plot.bar(color='cornflowerblue');
plt.ylabel('Count')
plt.xlabel('Sector')
plt.title("Loan Count by Sector for Loan Amount between \$20,000 and \$60,000");
loans_df[loans_df.funded_amount > 60000]
pylab.rcParams['figure.figsize'] = (16.0, 8.0)
boxplot_sorted(loans_df[loans_df.funded_amount < 10000], by=["sector"], column="funded_amount");
plt.xticks(rotation=90);
plt.ylabel('Funded Amount')
plt.xlabel('Sector')
plt.title('Funded Amount by Sector');
pylab.rcParams['figure.figsize'] = (6.0, 6.0)
boxplot_sorted(loans_df[(loans_df.funded_amount < 10000) & (loans_df.borrower_genders != "nan")], by=["borrower_genders"], column="funded_amount");
plt.title('Funded Amount by Gender')
plt.ylabel('Funded Amount')
plt.xlabel('Gender')
loan_amount_values = loans_df[(loans_df.funded_amount < 10000) & (loans_df.borrower_genders != "nan")].groupby("borrower_genders").loan_amount
loan_amount_values.median()
loan_amount_values.quantile(0.75) - loan_amount_values.quantile(0.25)
pylab.rcParams['figure.figsize'] = (24.0, 8.0)
boxplot_sorted(loans_df[(loans_df.funded_amount < 10000) & (loans_df.borrower_genders != "nan")], by=["country"], column="funded_amount");
plt.xticks(rotation=90);
plt.title('Funded Amount by Country')
plt.ylabel('Funded Amount')
plt.xlabel('Country');
loans_df[loans_df.country == 'Afghanistan']
pylab.rcParams['figure.figsize'] = (6.0, 6.0)
loans_df[loans_df.country == 'Chile'].sector.value_counts().plot.bar(color='cornflowerblue');
plt.title("Loan Count by Sector in Chile")
plt.xlabel("Sector")
plt.ylabel("Loan Count")
time_to_fund = (loans_df.funded_time - loans_df.posted_time)
time_to_fund_in_days = (time_to_fund.astype('timedelta64[s]')/(3600 * 24))
loans_df = loans_df.assign(time_to_fund=time_to_fund)
loans_df = loans_df.assign(time_to_fund_in_days=time_to_fund_in_days)


max(time_to_fund_in_days)
lower = loans_df.time_to_fund_in_days.quantile(0.01)
upper = loans_df.time_to_fund_in_days.quantile(0.99)
loans_df[(loans_df.time_to_fund_in_days > lower)].time_to_fund_in_days.plot.hist();
plt.title('Loan Count by Time taken to fund')
plt.xlabel('Time taken to fund')
plt.ylabel('Loan Count')
loans_df[(loans_df.time_to_fund_in_days > 100)].shape
loans_df[(loans_df.time_to_fund_in_days > 100)].shape[0]/loans_df.shape[0]
loans_df[(loans_df.time_to_fund_in_days > 100)].time_to_fund_in_days.plot.hist();
pylab.rcParams['figure.figsize'] = (8.0, 8.0)
boxplot_sorted(loans_df[loans_df.borrower_genders != 'nan'], by=["borrower_genders"], column="time_to_fund_in_days");
pylab.rcParams['figure.figsize'] = (24.0, 8.0)
#loans_df[["time_to_fund_in_days", "country"]].boxplot(by="country");
axes = boxplot_sorted(loans_df, by=["country"], column="time_to_fund_in_days")
axes.set_title("Time to Fund by country in days")
plt.xticks(rotation=90);
df_india = loans_df[loans_df.country == 'India']
df_india.shape
pylab.rcParams['figure.figsize'] = (8.0, 8.0)
df_india.groupby('sector').id.count().sort_values().plot.bar(color='cornflowerblue');
pylab.rcParams['figure.figsize'] = (8.0, 8.0)
df_india.groupby('region').id.count().sort_values(ascending=False).head(20).plot.bar(color='cornflowerblue');
df_india_top_ten = df_india.groupby('region').id.count().sort_values(ascending=False).head(10)
df_india_top_ten
df_india_top_ten.plot.bar(color='cornflowerblue');
for region in df_india_top_ten.index:
    plt.title(region)
    df_india[df_india.region==region].groupby('sector').id.count().sort_values(ascending=False).plot.bar(color='cornflowerblue')
    plt.show()
pd.Series(list(set(df_india.id) & set(loan_coords_df.id))).shape[0]/df_india.shape[0]
df_india = df_india.merge(loan_coords_df, on='id', how='inner')
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
longitudes = list(df_india.longitude)
latitudes = list(df_india.latitude)
plt.figure(figsize=(14, 8))
earth = Basemap(projection='lcc',
                resolution='h',
                llcrnrlon=67,
                llcrnrlat=5,
                urcrnrlon=99,
                urcrnrlat=37,
                lat_0=28,
                lon_0=77
)
earth.drawcoastlines()
earth.drawcountries()
earth.drawstates(color='#555566', linewidth=1)
earth.drawmapboundary(fill_color='#46bcec')
earth.fillcontinents(color = 'white',lake_color='#46bcec')
# convert lat and lon to map projection coordinates
longitudes, latitudes = earth(longitudes, latitudes)
plt.scatter(longitudes, latitudes, 
            c='red',alpha=0.5, zorder=10)
plt.savefig('Loans Disbursed in India', dpi=350)
MPI = pd.read_csv("../input/mpi/MPI_subnational.csv")
MPI[MPI.Country=='India'].shape
# Load data
MPI = pd.read_csv("../input/mpi/MPI_subnational.csv")[['Country', 'Sub-national region', 'World region', 'MPI Regional']]
MPInat = pd.read_csv("../input/mpi/MPI_national.csv")[['ISO','Country','MPI Rural', 'MPI Urban']].set_index('ISO')
LT = pd.read_csv("../input/data-science-for-good-kiva-crowdfunding/loan_themes_by_region.csv")[['country','Partner ID', 'Loan Theme ID', 'region', 'mpi_region', 'ISO', 'number', 'amount','rural_pct', 'LocationName', 'Loan Theme Type']]
# Create new column mpi_region and join MPI data to Loan themes on it
MPI['mpi_region'] = MPI[['Sub-national region', 'Country']].apply(lambda x: ', '.join(x), axis=1)
MPI = MPI.set_index('mpi_region')
LT = LT.join(MPI, on='mpi_region', rsuffix='_mpi') #[['country','Partner ID', 'Loan Theme ID', 'Country', 'ISO', 'mpi_region', 'MPI Regional', 'number', 'amount','Loan Theme Type']]
#~ Pull in country-level MPI Scores for when there aren't regional MPI Scores
LT = LT.join(MPInat, on='ISO',rsuffix='_mpinat')
LT['Rural'] = LT['rural_pct']/100        #~ Convert rural percentage to 0-1
LT['MPI Natl'] = LT['Rural']*LT['MPI Rural'] + (1-LT['Rural'])*LT['MPI Urban']
LT['MPI Regional'] = LT['MPI Regional'].fillna(LT['MPI Natl'])
#~ Get "Scores": volume-weighted average of MPI Region within each loan theme.
Scores = LT.groupby('Loan Theme ID').apply(lambda df: np.average(df['MPI Regional'], weights=df['amount'])).to_frame()
Scores.columns=["MPI Score"]
#~ Pull loan theme details
LT = LT.groupby('Loan Theme ID').first().join(Scores)#.join(LT_['MPI Natl'])
LT['Loan Theme ID'] = LT.index
loans_with_mpi_df = loans_df.merge(loan_theme_ids_df, on='id').merge(LT, on='Loan Theme ID')
loans_with_mpi_india_df = loans_with_mpi_df[loans_with_mpi_df.Country == 'India']
loans_with_mpi_india_df.shape
df_poverty_rate = pd.read_csv("../input/poverty-rate-of-indian-states/IndiaPovertyRate.csv", encoding = "ISO-8859-1")
latitudes = list(df_poverty_rate.Latitude)
longitudes = list(df_poverty_rate.Longitude)
poverty_rate = list(df_poverty_rate.PovertyRate)
plt.figure(figsize=(14, 8))
earth = Basemap(projection='lcc',
                resolution='h',
                llcrnrlon=67,
                llcrnrlat=5,
                urcrnrlon=99,
                urcrnrlat=37,
                lat_0=28,
                lon_0=77
)
earth.drawcoastlines()
earth.drawcountries()
earth.drawstates(color='#555566', linewidth=1)
earth.drawmapboundary(fill_color='#46bcec')
earth.fillcontinents(color = 'white',lake_color='#46bcec')
# convert lat and lon to map projection coordinates
longitudes, latitudes = earth(longitudes, latitudes)
plt.scatter(longitudes, latitudes, 
            c=poverty_rate, zorder=10, cmap='bwr')
plt.savefig('Loans Disbursed in India', dpi=350)
