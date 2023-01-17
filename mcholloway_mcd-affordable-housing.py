from IPython.display import Image

Image("../input/pandas-logo/pandas_logo.png")
import pandas as pd
houses_2009 = pd.read_csv('../input/appraisal-data/2009SINGLEFAMILYSF.txt')
houses_2009.head(n=10)
houses_2009.tail()
houses_2009.shape
houses_2009.info()
houses_2009.columns
houses_2009["2009 TOTAL APPR"]
houses_2009.columns = ['APN', 'DistrictCode', 'CouncilDistrict', 'AddressFullAddress',

       'AddressCity', 'AddressPostalCode', 'LAND', 'IMPR', 'TOTALAPPR',

       'TOTALASSD', 'FinishedArea']

houses_2009.columns
houses_2009.TOTALAPPR
# Your code here
# Or load the solution

%load ../input/exercisesolutions/soln_003.py
houses_2009.AddressCity.unique()
houses_2009.AddressCity.nunique()
houses_2009.AddressCity.value_counts()
# Your Code Here
# Or load the solution

%load ../input/exercisesolutions/soln_101.py
houses_2009.AddressFullAddress.value_counts() > 1
(houses_2009.AddressFullAddress.value_counts() > 1).sum()
houses_2009.AddressCity == 'BRENTWOOD'
houses_2009.loc[houses_2009.AddressCity == 'BRENTWOOD']
# Your Code Here
# Or load the solution

%load ../input/exercisesolutions/soln_102.py
# Your Code Here
# Or load the solution

%load ../input/exercisesolutions/soln_103.py
houses_2009.AddressCity.str.lower()
# Your Code Here
# Or load the solution

%load ../input/exercisesolutions/soln_104.py
houses_2009 = houses_2009.drop_duplicates('AddressFullAddress')

houses_2013 = houses_2013.drop_duplicates('AddressFullAddress')

houses_2017 = houses_2017.drop_duplicates('AddressFullAddress')
houses_2009.loc[100:105,['AddressFullAddress', 'AddressCity']]
houses_2009.loc[[1000, 2000, 3000], 'CouncilDistrict']
# Your Code Here
# Or load the solution

%load ../input/exercisesolutions/soln_105.py
import matplotlib.pyplot as plt

%matplotlib inline
houses_2009.CouncilDistrict.value_counts().plot.bar();
fig = houses_2009.CouncilDistrict.value_counts().plot.bar(figsize = (14,6), width = 0.75,

                                                         rot = 0, color = 'plum')

fig.set_xlabel('District')

fig.set_title('Number of Single-Family Homes by District, 2009', fontweight = 'bold');
houses_2009.CouncilDistrict.value_counts().loc[range(1,36)]
fig = houses_2009.CouncilDistrict.value_counts().loc[list(range(1,36))].plot.bar(figsize = (14,6), width = 0.75,

                                                         rot = 0, color = 'plum')

fig.set_xlabel('District')

fig.set_title('Number of Single-Family Homes by District, 2009', fontweight = 'bold');
# Your Code Here
# Or load the solution

%load ../input/exercisesolutions/soln_106.py
fig = houses_2009.FinishedArea.plot.hist(figsize = (10,4))

fig.set_title('Distribution of Homes by Square Footage', fontweight = 'bold');
# Your Code Here
# Or load the solution

%load ../input/exercisesolutions/soln_107.py
houses_2009.loc[houses_2009.FinishedArea < 10000].FinishedArea.plot.hist(figsize = (10,4), bins = 50)

plt.title('Distribution of Homes by Square Footage', fontweight = 'bold');
houses_2009.TOTALAPPR.plot.hist();
houses_2009.TOTALAPPR.describe()
houses_2009.loc[houses_2009.TOTALAPPR <= 750000].TOTALAPPR.plot.hist(bins = 50);
fig = houses_2009.loc[houses_2009.TOTALAPPR <= 750000].TOTALAPPR.plot.hist(bins = 50, alpha = 0.6, density = True, label = '2009', figsize = (10,5))

houses_2017.loc[houses_2017.TOTALAPPR <= 750000].TOTALAPPR.plot.hist(bins = 50, alpha = 0.6, density = True, label = '2017');

fig.axes.get_yaxis().set_visible(False)

fig.set_title('Distribution of Appraisal Values, 2009 vs 2017')

fig.legend();
# Your Code Here
# Or load the solution

%load ../input/exercisesolutions/soln_108.py
# Your Code Here
# Or load the solution

%load ../input/exercisesolutions/soln_109.py
houses_2009.groupby('CouncilDistrict').count()
houses_2009.groupby('CouncilDistrict').APN.count()
houses_2009.groupby('CouncilDistrict').TOTALAPPR.mean()
houses_2009.groupby('CouncilDistrict').TOTALAPPR.agg(['mean', 'median'])
houses_2009.groupby('CouncilDistrict').agg({'TOTALAPPR':['mean', 'median'], 'FinishedArea': ['mean', 'median']})
agg_df = houses_2009.groupby('CouncilDistrict').agg({'TOTALAPPR':['mean', 'median'], 'FinishedArea': ['mean', 'median']})

agg_df.columns
agg_df = houses_2009.groupby(['CouncilDistrict', 'AddressPostalCode']).TOTALAPPR.median()

agg_df
agg_df.loc[25]
agg_df.loc[(25, 37205)]
# Your Code Here
# Or Load the Solution

%load ../input/exercisesolutions/soln_110.py
fig, ax = plt.subplots(figsize = (12,5))

ax2 = ax.twinx()



width = 0.4



houses_2009.groupby('CouncilDistrict').TOTALAPPR.mean().plot.bar(color='olive', ax=ax, width=width, position=1, edgecolor = 'black', rot = 0)

houses_2009.groupby('CouncilDistrict').FinishedArea.mean().plot.bar(color='lightcoral', ax=ax2, width=width, position=0, edgecolor= 'black')



ax.set_ylabel('Median Appraisal Value', color = 'olive', fontweight = 'bold')

ax2.set_ylabel('Average Square Footage', color = 'lightcoral', fontweight = 'bold')



plt.xlim(-1,35)

plt.title('Housing Snapshot by District, 2009', fontweight = 'bold');
ACS = pd.read_csv('../input/census/ACS.csv')

ACS = ACS.set_index('district')

ACS.head()
# Your Code Here
# Or load the solution

%load ../input/exercisesolutions/soln_111.py
fig, ax = plt.subplots(figsize = (10,6))

plt.scatter(x = ACS.loc[ACS.year == 2017].median_income, 

         y = houses_2017.groupby('CouncilDistrict').TOTALAPPR.median(),

           alpha = 0.75)

plt.xlabel('Median Income')

plt.ylabel('Median Household Appraisal Value');
fig, ax = plt.subplots(figsize = (10,6))

plt.scatter(x = ACS.loc[ACS.year == 2017].median_income, 

         y = houses_2017.groupby('CouncilDistrict').TOTALAPPR.median(),

           alpha = 0.75)

plt.xlabel('Median Income')

plt.ylabel('Median Household Appraisal Value')

for i in range(1,36):

    plt.annotate(xy = (ACS.loc[ACS.year == 2017].median_income.loc[i], houses_2017.groupby('CouncilDistrict').TOTALAPPR.median().loc[i]),

                s = str(i));
import geopandas as gpd
council_districts = gpd.read_file('../input/shapefiles/Council_District_Outlines.geojson')
council_districts.head()
council_districts.plot();
fig, ax = plt.subplots(figsize = (10,10))

council_districts.plot(ax = ax);
council_districts['coords'] = council_districts.geometry.map(lambda x: x.representative_point().coords[:][0])

council_districts.head()
rows = council_districts.iterrows()
next(rows)
idx, row = next(rows)

print(idx)

print(row)
row['district']
for idx, row in rows:

    print(row['district'])
#Your Code Here
# Or load the solution

%load ../input/exercisesolutions/soln_203.py
def shift_coord(district, amount, direction):

    old_coord = council_districts.loc[council_districts.district == district, 'coords'].values[0]

    if direction == 'up':

        new_coord = (old_coord[0], old_coord[1] + amount)

    if direction == 'down':

        new_coord = (old_coord[0], old_coord[1] - amount)

    if direction == 'left':

        new_coord = (old_coord[0] - amount, old_coord[1])

    if direction == 'right':

        new_coord = (old_coord[0] + amount, old_coord[1])

    council_districts.loc[council_districts.district == district, 'lng'] = new_coord[0]

    council_districts.loc[council_districts.district == district, 'lat'] = new_coord[1]



    council_districts.loc[council_districts.district == district, 'coords'] = council_districts.loc[council_districts.district == district, ['lng', 'lat']].apply(tuple, axis = 1) 
shift_coord(district='15', amount = 0.005, direction = 'left')

shift_coord(district='9', amount = 0.005, direction = 'down')

shift_coord(district='15', amount = 0.02, direction = 'down')

shift_coord(district='28', amount = 0.003, direction = 'down')

shift_coord(district='6', amount = 0.005, direction = 'down')

shift_coord(district='27', amount = 0.004, direction = 'left')

shift_coord(district='27', amount = 0.005, direction = 'down')

shift_coord(district='11', amount = 0.01, direction = 'down')

shift_coord(district='18', amount = 0.005, direction = 'down')

shift_coord(district='22', amount = 0.01, direction = 'down')

shift_coord(district='25', amount = 0.006, direction = 'down')

shift_coord(district='21', amount = 0.005, direction = 'right')

shift_coord(district='24', amount = 0.005, direction = 'right')

shift_coord(district='3', amount = 0.01, direction = 'down')

shift_coord(district='3', amount = 0.005, direction = 'left')

shift_coord(district='7', amount = 0.015, direction = 'down')
fig, ax = plt.subplots(figsize = (10,10))

council_districts.plot(ax = ax)

for idx, row in council_districts.iterrows():

    plt.annotate(s=row['district'], xy=row['coords'],

                 horizontalalignment='center', fontweight = 'bold')
homes_per_district = houses_2009.CouncilDistrict.value_counts()

homes_per_district
type(homes_per_district)
homes_per_district = homes_per_district.reset_index()

homes_per_district.head(5)
homes_per_district.columns = ['district', 'num_homes_2009']

homes_per_district.head(2)
pd.merge(left = council_districts, right = homes_per_district)
council_districts.district = council_districts.district.astype(int)
council_districts = pd.merge(left = council_districts, right = homes_per_district)
type(council_districts)
fig, ax = plt.subplots(figsize = (10,10))

council_districts.plot(ax = ax, column = 'num_homes_2009')

for idx, row in council_districts.iterrows():

    plt.annotate(s=row['district'], xy=row['coords'],

                 horizontalalignment='center', fontweight = 'bold');
fig, ax = plt.subplots(figsize = (10,10))

council_districts.plot(ax = ax, column = 'num_homes_2009', legend = True)

for idx, row in council_districts.iterrows():

    plt.annotate(s=row['district'], xy=row['coords'],

                 horizontalalignment='center', fontweight = 'bold')
fig, ax = plt.subplots(figsize = (10,10))

council_districts.plot(ax = ax, column = 'num_homes_2009', legend = True, cmap = 'YlOrRd',edgecolor = 'grey')

for idx, row in council_districts.iterrows():

    plt.annotate(s=row['district'], xy=row['coords'],

                 horizontalalignment='center', fontweight = 'bold')
from matplotlib import cm

from matplotlib.colors import Normalize

from mpl_toolkits.axes_grid1 import make_axes_locatable
fig, ax = plt.subplots(figsize = (10,10))



council_districts.plot(ax = ax, column = 'num_homes_2009', cmap = 'YlOrRd', edgecolor = 'grey')



for idx, row in council_districts.iterrows():

    plt.annotate(s=row['district'], xy=row['coords'],

                 horizontalalignment='center', fontweight = 'bold', color = 'black')



divider = make_axes_locatable(ax)

cax = divider.append_axes("right", size="5%", pad=0.1)



cmap = cm.ScalarMappable(

      norm = Normalize(council_districts.num_homes_2009.min(), council_districts.num_homes_2009.max()), 

      cmap = 'YlOrRd')

cmap.set_array([])    

fig.colorbar(mappable=cmap, cax = cax);
# Your Code Here
# Or load the solution 

%load ../input/exercisesolutions/soln_202.py
fig, ax = plt.subplots(figsize = (10,10))



council_districts.plot(ax = ax, column = 'num_homes_2009', cmap = 'YlOrRd', edgecolor = 'grey')



for idx, row in council_districts.iterrows():

    plt.annotate(s=row['district'], xy=row['coords'],

                 horizontalalignment='center', fontweight = 'bold', color = choose_color(row['num_homes_2009']))



plt.title('Number of Single-Family Homes by Council District, 2009', fontweight = 'bold', fontsize = 14)

plt.axis('off')



divider = make_axes_locatable(ax)

cax = divider.append_axes("right", size="5%", pad=0.1)



cmap = cm.ScalarMappable(

      norm = Normalize(council_districts.num_homes_2009.min(), council_districts.num_homes_2009.max()), 

      cmap = 'YlOrRd')

cmap.set_array([])    

fig.colorbar(mappable=cmap, cax = cax);
interstates = gpd.read_file('../input/shapefiles/tl_2016_us_primaryroads.shp')
interstates.head()
interstates.plot()
print(interstates.crs)

print(council_districts.crs)
interstates = interstates.to_crs(council_districts.crs)
interstates = gpd.sjoin(interstates, council_districts, how="inner", op='intersects')
fig, ax = plt.subplots(figsize = (10,10))

council_districts.plot(ax = ax)

for idx, row in council_districts.iterrows():

    plt.annotate(s=row['district'], xy=row['coords'],

                 horizontalalignment='center', fontweight = 'bold')

interstates.plot(color = 'black', ax = ax);
fig, ax = plt.subplots(figsize = (10,10))

council_districts.plot(ax = ax)

xlims = plt.xlim()

ylims = plt.ylim()

for idx, row in council_districts.iterrows():

    plt.annotate(s=row['district'], xy=row['coords'],

                 horizontalalignment='center', fontweight = 'bold')

interstates.plot(color = 'black', ax = ax)

plt.xlim(xlims)

plt.ylim(ylims);
fig, ax = plt.subplots(figsize = (10,10))



council_districts.plot(ax = ax, column = 'num_homes_2009', cmap = 'YlOrRd', edgecolor = 'grey')



xlims = plt.xlim()

ylims = plt.ylim()



interstates.plot(color = 'black', ax = ax)

plt.xlim(xlims)

plt.ylim(ylims)





for idx, row in council_districts.iterrows():

    plt.annotate(s=row['district'], xy=row['coords'],

                 horizontalalignment='center', fontweight = 'bold', color = choose_color(row['num_homes_2009']))



plt.title('Number of Single-Family Homes by Council District, 2009', fontweight = 'bold', fontsize = 14)

plt.axis('off')



divider = make_axes_locatable(ax)

cax = divider.append_axes("right", size="5%", pad=0.1)



cmap = cm.ScalarMappable(

      norm = Normalize(council_districts.num_homes_2009.min(), council_districts.num_homes_2009.max()), 

      cmap = 'YlOrRd')

cmap.set_array([])    

fig.colorbar(mappable=cmap, cax = cax); 
#Your Code Here
# Or load the solution 

%load ../input/exercisesolutions/soln_204.py
homes_per_district_2013 = pd.DataFrame(houses_2013.CouncilDistrict.value_counts().sort_values()).reset_index()

homes_per_district_2013.columns = ['district', 'num_homes_2013']



homes_per_district_2017 = pd.DataFrame(houses_2017.CouncilDistrict.value_counts().sort_values()).reset_index()

homes_per_district_2017.columns = ['district', 'num_homes_2017']
council_districts = pd.merge(left = pd.merge(left = council_districts, right = homes_per_district_2013), right = homes_per_district_2017)

#Your Code Here
# Or load the solution

%load ../input/exercisesolutions/soln_302.py
generate_map(2013)
from ipywidgets import interact
@interact(x=5)

def square(x):

    return x**2
@interact(x = 5, y = 5)

def sum_squares(x,y):

    return x**2 + y**2
@interact(k = [1/4, 1/3, 1/2, 1, 2, 3, 4])

def plot_power_function(k):

    xs = range(50)

    dynamic_ys = [x ** k for x in xs]

    plt.plot(xs, dynamic_ys)
# Your Code Here
# Or load the solution 

%load ../input/exercisesolutions/soln_303.py
def choose_color_scaled(num_homes, vmin, vmax):

    if num_homes < (vmin + vmax) / 2: return "black"

    return "white"
vmin = council_districts[['num_homes_2009', 'num_homes_2013', 'num_homes_2017']].values.min()

vmax = council_districts[['num_homes_2009', 'num_homes_2013', 'num_homes_2017']].values.max()



@interact(year = ['2009', '2013', '2017'])

def generate_map(year):

    fig, ax = plt.subplots(figsize = (10,10))

    column = 'num_homes_' + year

    



    council_districts.plot(ax = ax, column = column, cmap = 'YlOrRd', edgecolor = 'grey', vmin = vmin, vmax = vmax)



    xlims = plt.xlim()

    ylims = plt.ylim()

    interstates.plot(color = 'black', ax = ax)

    plt.xlim(xlims)

    plt.ylim(ylims)

    

    for idx, row in council_districts.iterrows():

        plt.annotate(s=row['district'], xy=row['coords'],

                     horizontalalignment='center', fontweight = 'bold', color = choose_color_scaled(row[column], vmin, vmax))



    plt.title(f'Number of Single-Family Homes by Council District, {year}', fontweight = 'bold', fontsize = 14)

    plt.axis('off')



    divider = make_axes_locatable(ax)

    cax = divider.append_axes("right", size="5%", pad=0.1)



    cmap = cm.ScalarMappable(

          norm = Normalize(vmin, vmax), 

          cmap = 'YlOrRd')

    cmap.set_array([])    

    fig.colorbar(mappable=cmap, cax = cax);   
#Your Code Here
# Or load the solution

%load ../input/exercisesolutions/soln_304.py
council_districts[['district', 'absolute_change']].sort_values('absolute_change').head()
council_districts[['district', 'absolute_change']].sort_values('absolute_change', ascending = False).head()
council_districts[['district', 'relative_change']].sort_values('relative_change', ascending = False).head()
#Your Code Here
# Or load the solution 

%load ../input/exercisesolutions/soln_305.py
houses_2017.TOTALAPPR.describe()
houses_2017.nlargest(n=5, columns='TOTALAPPR')
def find_mortgage_payment(TOTALAPPR,years = 30, rate = 4, down_payment = 20):

    P = TOTALAPPR * (1 - (down_payment / 100))

    n = 12 * years

    r = rate / (100 * 12)

    M = P * (r * (1 + r)**n) / ((1 + r)**n - 1)

    return M
houses_2009['est_mortgage_cost'] = houses_2009.TOTALAPPR.apply(lambda x: 12*find_mortgage_payment(x))

houses_2013['est_mortgage_cost'] = houses_2013.TOTALAPPR.apply(lambda x: 12*find_mortgage_payment(x))

houses_2017['est_mortgage_cost'] = houses_2017.TOTALAPPR.apply(lambda x: 12*find_mortgage_payment(x))

tax_rates = {'USD' : 3.155/100,

            'GSD' : 2.755/100,

             'GO' : 3.503 / 100,

             'FH' : 2.755/100,

             'OH' : 2.755/100,

             'BM' : 3.012/100,

             'BH' : 2.755/100,

            'CBID' : 3.2844 / 100,

            'GBID': 3.2631 / 100,

            'RT' : 3.437/100,

            'LW' : 2.755/100}
def calculate_property_taxes(row):

    return row.TOTALASSD * tax_rates[row.DistrictCode]
houses_2009['est_property_tax'] = houses_2009.apply(calculate_property_taxes, axis = 1)
houses_2009.DistrictCode = houses_2009.DistrictCode.str.strip()

houses_2013.DistrictCode = houses_2013.DistrictCode.str.strip()

houses_2017.DistrictCode = houses_2017.DistrictCode.str.strip()
houses_2009['est_property_tax'] = houses_2009.apply(calculate_property_taxes, axis = 1)

houses_2013['est_property_tax'] = houses_2013.apply(calculate_property_taxes, axis = 1)

houses_2017['est_property_tax'] = houses_2017.apply(calculate_property_taxes, axis = 1)
houses_2009['est_yearly_cost'] = houses_2009.est_mortgage_cost + houses_2009.est_property_tax + 720

houses_2013['est_yearly_cost'] = houses_2013.est_mortgage_cost + houses_2013.est_property_tax + 720

houses_2017['est_yearly_cost'] = houses_2017.est_mortgage_cost + houses_2017.est_property_tax + 720
def classify_house(value, AMI):

    if value <= 0.3 * 0.3*AMI:

        return 'AFF_1'

    elif value <= 0.3 * 0.6 * AMI:

        return 'AFF_2'

    elif value <= 0.3* 0.9 * AMI:

        return 'WF_1'

    elif value <= 0.3 * 1.2*AMI:

        return 'WF_2'

    else:

        return 'AWF'
houses_2009['category'] = houses_2009.est_yearly_cost.apply(lambda x: classify_house(x, 64900))

houses_2013['category'] = houses_2013.est_yearly_cost.apply(lambda x: classify_house(x, 62300))

houses_2017['category'] = houses_2017.est_yearly_cost.apply(lambda x: classify_house(x, 68000))
plt.figure(figsize = (10,6))

houses_2017.category.value_counts()[['AFF_1', 'AFF_2', 'WF_1', 'WF_2', 'AWF']].plot.bar(rot = 0)

plt.title('Number of Single-Family Homes by Category, 2017');
chess_players = pd.DataFrame({'player': ['Magnus Carlsen', 'Fabiano Caruana', 'Ding Liren'],

                       'wins': [962, 793,414],

                        'draws': [930,821,575],

                       'losses': [334,459,186]})

chess_players= chess_players.set_index('player')

chess_players
fig, ax = plt.subplots(figsize = (7,5))

chess_players.plot.bar(ax = ax, edgecolor = 'black', lw = 1.5, rot = 0, width = 0.8)

plt.title('Top 3 Chess Players by ELO', fontweight = 'bold')

plt.xlabel('')

ax.legend(bbox_to_anchor=(1, 0.6));
houses_2009['year'] = 2009

houses_2013['year'] = 2013

houses_2017['year'] = 2017
houses = pd.concat([houses_2009, houses_2013, houses_2017])
houses.head()
#Your Code Here
# Or load the soution 

%load ../input/exercisesolutions/soln_402.py
melted_chess = chess_players.reset_index().melt(id_vars=['player'], var_name = 'outcome')

melted_chess
melted_chess.pivot(index='player', columns='outcome', values='value')
#Your Code Here
# Or load the solution

%load ../input/exercisesolutions/soln_403.py
pivot_df = pivot_df.loc[['AFF_1', 'AFF_2', 'WF_1', 'WF_2', 'AWF']]

pivot_df
# Your Code Here
# Or load the solution

%load ../input/exercisesolutions/soln_404.py
chess_players
fig, ax = plt.subplots(figsize = (8,6))

chess_players.plot.bar(stacked=True, edgecolor = 'black', lw = 1.5, 

                       rot = 0, ax = ax, width = 0.75,title = 'Top 3 Chess Players by ELO')

plt.xlabel('')

handles, labels = ax.get_legend_handles_labels()

ax.legend(handles[::-1], labels[::-1], bbox_to_anchor=(1, 0.6));
import numpy as np
fig, ax = plt.subplots(figsize = (8,6))

chess_players.plot.bar(stacked=True, edgecolor = 'black', lw = 1.5, 

                       rot = 0, ax = ax, width = 0.75,title = 'Top 3 Chess Players by ELO')

plt.xlabel('')

handles, labels = ax.get_legend_handles_labels()

ax.legend(handles[::-1], labels[::-1], bbox_to_anchor=(1, 0.6))



rows = chess_players.iterrows()

for i in range(3):

    values = next(rows)[1]

    heights = np.array([0] + list(values.cumsum()[:-1])) + values/2

    for height, value in zip(heights,values):

        plt.text(x = i, y = height, s = f'{value:,}', color = 'white', ha = 'center', va = 'center', fontweight = 'bold');
#Your Code Here
# Or load the solution

%load ../input/exercisesolutions/soln_405.py
#Your Code Here
# Or load the solution

%load ../input/exercisesolutions/soln_406.py
#Your Code Here
# Or load the solution

%load ../input/exercisesolutions/soln_407.py
overall_pct_change = 100*(houses.loc[houses.year == 2017].TOTALAPPR.median() - houses.loc[houses.year==2013].TOTALAPPR.median()) / houses.loc[houses.year == 2013].TOTALAPPR.median()

overall_pct_change
#Your Code Here
# Or load the solution

%load ../input/exercisesolutions/soln_408.py
median_appr = median_appr.pivot(index = 'CouncilDistrict', columns='year', values='TOTALAPPR')

median_appr.head()
median_appr.columns
median_appr['pct_change'] = 100*(median_appr[2017] - median_appr[2013]) / median_appr[2013]
median_appr.head()
median_appr = median_appr.reset_index()

median_appr.head()
fig, ax = plt.subplots()

median_appr.plot.scatter(x = 'CouncilDistrict', y = 'pct_change', figsize = (14,6), ax = ax, s=75)

ax.set_xticks(list(range(1,36)))

ax.set_ylim(0,140)

plt.grid(axis = 'both', linestyle = '--', alpha = 0.5, lw = 2)



plt.title('Percent Change in Median Appraisal Value, 2013 - 2017', fontweight = 'bold', fontsize = 14)

plt.xlabel('Council District', fontsize = 14)

plt.ylabel('Percent Change (%)', fontsize = 14)

plt.xticks(fontsize = 12, fontweight = 'bold')

plt.yticks(fontsize = 14);
fig, ax = plt.subplots()

median_appr.plot.scatter(x = 'CouncilDistrict', y = 'pct_change', figsize = (14,6), ax = ax, s=75)

ax.set_xticks(list(range(1,36)))

ax.set_ylim(0,140)

plt.grid(axis = 'both', linestyle = '--', alpha = 0.5, lw = 2)

plt.axhline(y=overall_pct_change, color='r', lw = 1)





plt.title('Percent Change in Median Appraisal Value, 2013 - 2017', fontweight = 'bold', fontsize = 14)

plt.xlabel('Council District', fontsize = 14)

plt.ylabel('Percent Change (%)', fontsize = 14)

plt.xticks(fontsize = 12, fontweight = 'bold')

plt.yticks(fontsize = 14);
from matplotlib import collections  as mc
for x in zip([1,2,3], ['a', 'b', 'c']):

    print(x)
lines = [[(x,y),(x,overall_pct_change)] for x,y in zip(range(1,36), median_appr['pct_change'])]



fig, ax = plt.subplots()

median_appr.plot.scatter(x = 'CouncilDistrict', y = 'pct_change', figsize = (14,6), ax = ax, s=75)

ax.set_xticks(list(range(1,36)))

ax.set_ylim(0,140)

plt.grid(axis = 'both', linestyle = '--', alpha = 0.5, lw = 2)

plt.axhline(y=overall_pct_change, color='r', lw = 1)

lc = mc.LineCollection(lines, linewidths=2)

ax.add_collection(lc)



plt.title('Percent Change in Median Appraisal Value, 2013 - 2017', fontweight = 'bold', fontsize = 14)

plt.xlabel('Council District', fontsize = 14)

plt.ylabel('Percent Change (%)', fontsize = 14)

plt.xticks(fontsize = 12, fontweight = 'bold')

plt.yticks(fontsize = 14);
lines = [[(x,y),(x,overall_pct_change)] for x,y in zip(range(1,36), median_appr['pct_change'])]



fig, ax = plt.subplots()

median_appr.plot.scatter(x = 'CouncilDistrict', y = 'pct_change', figsize = (14,6), ax = ax, s=75)

ax.set_xticks(list(range(1,36)))

ax.set_ylim(0,140)

plt.grid(axis = 'both', linestyle = '--', alpha = 0.5, lw = 2)

plt.axhline(y=overall_pct_change, color='r', lw = 1)

lc = mc.LineCollection(lines, linewidths=2)

ax.add_collection(lc)



plt.title('Percent Change in Median Appraisal Value, 2013 - 2017', fontweight = 'bold', fontsize = 14)

plt.xlabel('Council District', fontsize = 14)

plt.ylabel('Percent Change (%)', fontsize = 14)

plt.xticks(fontsize = 12, fontweight = 'bold')

plt.yticks(fontsize = 14)



ax.annotate("Percent Change for\nDavidson County\n(" + "{:.1f}".format(overall_pct_change)+ "%)", xy=(36, overall_pct_change), 

            xytext=(33, 90), fontsize = 12, ha = 'center', va = 'center', color = 'red', fontweight = "bold",

            arrowprops=dict(arrowstyle="->", lw = 2))

ax.annotate("District 5\n(" + "{:.1f}".format(median_appr['pct_change'].max())+ "%)", xy=(5.5, median_appr['pct_change'].max()-1), 

            xytext=(9, 120), fontsize = 12, ha = 'center', va = 'center', color = 'red', fontweight = "bold",

            arrowprops=dict(arrowstyle="->", lw = 2));
#Your Code Here
%load ../input/exercisesolutions/soln_409.py
import seaborn as sns
fig = plt.figure(figsize = (10,6))

sns.boxplot(data = houses_2017.loc[houses_2017.CouncilDistrict.isin([1,2,3,4])], 

            x = 'CouncilDistrict', 

            y = 'TOTALAPPR')

plt.title('Home Appriasal Values, 2017')

plt.ylim(0, 1000000);
fig = plt.figure(figsize = (10,6))

sns.violinplot(data = houses_2017.loc[houses_2017.CouncilDistrict.isin([1,2,3,4])], x = 'CouncilDistrict', 

               y = 'TOTALAPPR')

plt.title('Home Appraisal Values, 2017')

plt.ylim(0, 1000000);
fig = plt.figure(figsize = (10,6))

sns.boxplot(data = houses, x = 'year', y = 'TOTALAPPR')

plt.title('Home Appraisal Values')

plt.ylim(0, 1000000);
fig = plt.figure(figsize = (10,6))

sns.violinplot(data = houses, x = 'year', y = 'TOTALAPPR')

plt.title('Home Appraisal Values')

plt.ylim(0, 1000000);
np.percentile(houses.loc[houses.CouncilDistrict == 34, 'TOTALAPPR'], 90)
@interact(district = range(1,36), plot_type = ['box', 'violin'])

def plot_dist(district, plot_type):

    fig = plt.figure(figsize = (10,6))

    if plot_type == 'box':

        sns.boxplot(data = houses.loc[houses.CouncilDistrict == district], x = 'year', y = 'TOTALAPPR')

    if plot_type == 'violin':

        sns.violinplot(data = houses.loc[houses.CouncilDistrict == district], x = 'year', y = 'TOTALAPPR')

    ymax = np.percentile(houses.loc[(houses.CouncilDistrict == district) & (houses.year == 2017), 'TOTALAPPR'], 99.9)

    plt.ylim(0, ymax)

    plt.title('Total Appraised Value, District ' + str(district));
cd = council_districts[['district', 'geometry']]
import warnings

warnings.filterwarnings('ignore')
@interact(district = range(1,36), plot_type = ['box', 'violin'])

def plot_dist(district, plot_type):

    fig = plt.figure(figsize = (10,6))

    if plot_type == 'box':

        sns.boxplot(data = houses.loc[houses.CouncilDistrict == district], x = 'year', y = 'TOTALAPPR')

    if plot_type == 'violin':

        sns.violinplot(data = houses.loc[houses.CouncilDistrict == district], x = 'year', y = 'TOTALAPPR')

    ymax = np.percentile(houses.loc[(houses.CouncilDistrict == district) & (houses.year == 2017), 'TOTALAPPR'], 99.9)

    plt.ylim(0, ymax)

    plt.title('Total Appraised Value, District ' + str(district))

    

    cd['chosen_district'] = 0

    cd.loc[cd.district == district, 'chosen_district'] = 1

    

    mini_map = plt.axes([.85, .3, .4, .4]) #[left, bottom, width, height]

    cd.plot(column = 'chosen_district', ax = mini_map, legend = False, edgecolor = 'black', cmap = 'binary')

    plt.axis('off')

    plt.title('District ' + str(district));
df = houses.loc[houses.CouncilDistrict == 20]



target_0 = df.loc[df.year == 2009]

target_1 = df.loc[df.year == 2013]

target_2 = df.loc[df.year == 2017]



sns.distplot(target_0[['TOTALAPPR']], hist=False, label = '2009')

sns.distplot(target_1[['TOTALAPPR']], hist=False, label = '2013')

g = sns.distplot(target_2[['TOTALAPPR']], hist=False, label = '2017')



g.set(xlim=(0, 500000));
df = houses.loc[houses.CouncilDistrict == 20]



target_0 = df.loc[df.year == 2009]

target_1 = df.loc[df.year == 2013]

target_2 = df.loc[df.year == 2017]



sns.distplot(target_0[['TOTALAPPR']], hist=True, label = '2009')

sns.distplot(target_1[['TOTALAPPR']], hist=True, label = '2013')

g = sns.distplot(target_2[['TOTALAPPR']], hist=True, label = '2017')



g.set(xlim=(0, 500000));
@interact(district = range(1,36))

def make_dist_plot(district):

    plt.figure(figsize = (10,6))

    

    df = houses.loc[houses.CouncilDistrict == district]



    target_0 = df.loc[df.year == 2009]

    target_1 = df.loc[df.year == 2013]

    target_2 = df.loc[df.year == 2017]



    sns.distplot(target_0[['TOTALAPPR']], hist=False, label = '2009', kde_kws={'lw': 2.5})

    sns.distplot(target_1[['TOTALAPPR']], hist=False, label = '2013', kde_kws={'lw': 2.5})

    g = sns.distplot(target_2[['TOTALAPPR']], hist=False, label = '2017', kde_kws={'lw': 2.5}, color = 'purple')



    xmax = np.percentile(houses.loc[(houses.CouncilDistrict == district) & (houses.year == 2017), 'TOTALAPPR'], 95)



    g.set(xlim=(0, xmax))

    g.set(yticks = [])

    g.set(title="Distribution of Appraisal Values, District " + str(district));
import plotly.express as px
chess_players
melted_chess = chess_players.reset_index().melt(id_vars=['player'], var_name = 'outcome')

melted_chess
fig = px.bar(melted_chess, x='player', y='value', color = 'outcome', 

             width = 800, height = 500,

            category_orders = {'outcome' : ['wins', 'losses', 'draws']})

fig.update_layout(title_text = 'Top Rated Chess Players', title_font_size = 24)

fig.update_yaxes(title_text = 'Number of Games', title_font_size = 20, tickfont_size = 14,)

fig.update_xaxes(title_text = '', tickfont_size = 18)

fig.update_layout(legend_traceorder = 'reversed')

fig.show()
category_count.head()
category_count.year = category_count.year.astype('category')
@interact(district = range(1, 36))

def make_plot(district):

    df = houses.loc[houses.CouncilDistrict == district].groupby(['category', 'year']).APN.count().reset_index().rename(columns = {'APN' : 'count'})

    #df.year = df.year.astype('category')

    fig = px.bar(df, x='year', y='count', color = 'category', width = 800, height = 500,

                category_orders = {'category' : ['AFF_1', 'AFF_2', 'WF_1', 'WF_2', 'AWF']})

    fig.update_yaxes(title_text = 'Number of Homes', title_font_size = 18)

    fig.update_xaxes(title_text = '', tick0=2009, dtick=4, tickfont_size = 18)

    fig.update_layout(title_text = 'Affordable Housing Profile, District ' + str(district), title_font_size = 20)

    fig.update_layout(legend_traceorder = 'reversed')

    fig.show()
district_counts = houses.groupby(['year', 'CouncilDistrict', 'category']).APN.count().reset_index().rename(columns = {'APN' : 'num_homes'})

district_counts = district_counts.loc[district_counts.CouncilDistrict.isin(list(range(1,36)))]
@interact(year = [2009, 2013, 2017])

def make_plot(year):

    df = district_counts.loc[district_counts.year == year]

    fig = px.bar(df, x='CouncilDistrict', y='num_homes', color = 'category', width = 900, height = 500,

                category_orders = {'category' : ['AFF_1', 'AFF_2', 'WF_1', 'WF_2', 'AWF']})

    fig.update_yaxes(title_text = 'Number of Homes', title_font_size = 18, range = [0,8300])

    fig.update_xaxes(title_text = 'District', tick0=1, dtick=1, tickfont_size = 14, tickangle = 0)

    fig.update_layout(title_text = 'Davidson County Affordable Housing Profile by District, ' + str(year), title_font_size = 20)

    fig.update_layout(legend_traceorder = 'reversed')

    fig.show()
@interact(district = range(1,36))

def make_plotly(district):

    df = houses.loc[houses.CouncilDistrict == district]

    ymax = np.percentile(df.TOTALAPPR, 99.9)

    

    fig = px.box(df, x="year", y="TOTALAPPR", width = 800, height = 500)

    fig.update_yaxes(range=[0,ymax], title_text = 'Appraised Value', title_font_size = 18)

    fig.update_xaxes(title_text = '', tickfont_size = 18)

    fig.update_layout(title_text = 'Appraised Values, District ' + str(district), title_font_size = 20)



    fig.show()
@interact(district = range(1,36))

def make_plotly(district):

    df = houses.loc[houses.CouncilDistrict == district]

    ymax = np.percentile(df.TOTALAPPR, 99.9)

    

    fig = px.violin(df, x="year", y="TOTALAPPR", width = 800, height = 500, box = True)

    fig.update_yaxes(range=[0,ymax], title_text = 'Appraised Value', title_font_size = 18)

    fig.update_xaxes(title_text = '', tickfont_size = 18)

    fig.update_layout(title_text = 'Appraised Values, District ' + str(district), title_font_size = 20)



    fig.show()