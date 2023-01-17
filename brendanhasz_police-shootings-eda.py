# Packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import poisson 
import urllib.request, json 

# Plot settings
%matplotlib inline
%config InlineBackend.figure_format = 'svg'
sns.set()
PK = pd.read_csv('../input/fatal-police-shootings-data.csv', encoding='latin1')
PK.sample(n=10)
# Show info about each column
for col in PK:
    print(col, '\nNum NaNs:', PK[col].isnull().sum())
    print(PK[col].describe(), '\n')
# Import bokeh components
from bokeh.plotting import figure, show
from bokeh.io import output_notebook
from bokeh.models import ColumnDataSource, Circle, HoverTool, ColorBar, LinearColorMapper
from bokeh.palettes import Viridis256
from bokeh.tile_providers import CARTODBPOSITRON_RETINA
output_notebook()

# Functions to convert lat and lon to Mercator coords
def lat_to_mercY(lat):
    """Convert Latitude to Mercator Y"""
    return np.log(np.tan(np.pi / 4 + np.radians(lat) / 2)) * 6378137.0

def lon_to_mercX(lon):
    """Convert Longitude to Mercator X"""
    return np.radians(lon) * 6378137.0

# Count shootings per city
PK['citystate'] = PK.city.astype(str)+' '+PK.state.astype(str)
PKc = PK.citystate.value_counts().to_frame()

# this doesn't actually work in Kaggle Kernels b/c networking is disabled
# Look up latitude and longitude of each city using OSM geocoding
#cities = [c.replace(' ', '%20') for c in PKc.index.tolist()]
#Surl = 'http://nominatim.openstreetmap.org/search/' #search part of url
#Furl = '%20USA?format=json' #format part of url
#X = []
#Y = []
#for c in cities:
#    try:
#        with urllib.request.urlopen(Surl+c+Furl) as url:
#            data = json.loads(url.read().decode())
#        X.append(float(data[0]['lon']))
#        Y.append(float(data[0]['lat']))
#    except (IndexError, UnicodeEncodeError): #could not geocode this city
#        X.append(np.nan)
#        Y.append(np.nan)

# Load location data instead of geocoding it
CL = pd.read_csv('../input/CityLocations.csv')
PKcl = PKc.join(CL.set_index('city'))
X = PKcl.lon
Y = PKcl.lat

# Data source table for Bokeh
source = ColumnDataSource(data=dict(
    X = lon_to_mercX(X),
    Y = lat_to_mercY(Y),
    size = 7*np.sqrt(PKc.citystate),
    num_shot = PKc.citystate,
    city = PKc.index.tolist()
))

# Tooltips (on hover)
tooltips=[('People shot', "@num_shot"), 
          ('City', "@city")]

# Plot the points
p = figure(plot_width=800, plot_height=500, 
           x_range=(-7300000, -14500000), y_range=(2500000,7000000),
           title='Police Shootings by City', 
           active_scroll="wheel_zoom", tooltips=tooltips)
p.add_tile(CARTODBPOSITRON_RETINA) #set background map
p.circle('X', 'Y', source=source, size='size', #plot each station
         fill_alpha=0.5, line_color=None, color='crimson')
p.axis.visible = False
show(p)
# Plot the manner of death
ax = sns.countplot(x='manner_of_death', data=PK)
plt.title('Manner of death')
for p in ax.patches:
    x = p.get_bbox().get_points()[:,0]
    y = p.get_bbox().get_points()[1,1]
    ax.annotate('{:.2g}%'.format(100.*y/len(PK)), (x.mean(), y), ha='center', va='bottom')
plt.show()
# Plot the 20 most common weapons used by individuals shot
plt.figure(figsize=(4.7, 5))
ax = sns.countplot(y='armed', data=PK,
                   order = PK.armed.value_counts().iloc[:20].index)
plt.title('Weapon used by person shot')
plt.show()
sns.distplot(PK.age[~np.isnan(PK.age)])
plt.title('Age of individuals shot by police')
plt.show()
PK[PK.age<16]
# Plot police killings by gender
ax = sns.countplot(x = "gender", data = PK,
                   order = PK.gender.value_counts().index)
for p in ax.patches:
    x = p.get_bbox().get_points()[:,0]
    y = p.get_bbox().get_points()[1,1]
    ax.annotate('{:.2g}%'.format(100.*y/len(PK)), (x.mean(), y), ha='center', va='bottom')
plt.title('Police Killings by Gender')
plt.show()
def ActualVsPopulation(df, pop, group):
    """Get dataframe with actual per-group percentage vs population group percentage"""
    d = {group: [], 'type': [], 'percent': []}
    tot_pop = float(sum(pop.values()))
    for g in df[group].dropna().unique(): #for each group

        # Actual percentages
        d[group].append(g)
        d['type'].append('Killings')
        d['percent'].append(100*df[df[group]==g].id.count()/df.id.count())

        # Percentages if statistic followed population distribution
        d[group].append(g)
        d['type'].append('Population') #based on population percentage
        d['percent'].append(100*pop[g]/tot_pop)
        
    return pd.DataFrame(data=d)
# Plot percent police killings by gender vs population percentages
pop_g = {'M': 49.2, 'F': 50.8} #percent population by gender https://www.census.gov/quickfacts/fact/table/US
df = ActualVsPopulation(PK, pop_g, 'gender')
sns.barplot(x="gender", y="percent", hue="type", data=df, palette=["r", "C0"])
plt.title('Actual Police Killings vs Population Distribution (by Gender)')
plt.show()
# Plot police killings by race
ax = sns.countplot(x="race", data=PK,
                   order = PK.race.value_counts().index)
for p in ax.patches:
    x = p.get_bbox().get_points()[:,0]
    y = p.get_bbox().get_points()[1,1]
    ax.annotate('{:.2g}%'.format(100.*y/len(PK)), (x.mean(), y), ha='center', va='bottom')
plt.title('Police Killings by Race')
plt.show()
# Plot percent police killings by race vs population percentages
pop_r = {'W': 60.7, # white - % population by race https://www.census.gov/quickfacts/fact/table/US
         'B': 13.4, # black or african american
         'H': 18.1, # hispanic or latino
         'A': 5.8,  # asian
         'N': 1.5,  # american indian, alaska native, naitive hawaiian, and other pacific islander
         'O': 0.5}  # other
df = ActualVsPopulation(PK, pop_r, 'race')
sns.barplot(x="race", y="percent", hue="type", data=df,
            order = PK.race.value_counts().index, palette=["r", "C0"])
plt.title('Actual Police Killings vs Population Distribution (by Race)')
plt.show()
plt.figure(figsize=(6, 13))
sns.countplot(y="state", 
              data=PK,
              order=PK.state.value_counts().index)
plt.title('Police Killings By State')
plt.show()
# Population for each state 
# https://www.census.gov/data/datasets/2017/demo/popest/state-total.html
pop_s ={'AL': 4874747,
        'AK': 739795,
        'AZ': 7016270,
        'AR': 3004279,
        'CA': 39536653,
        'CO': 5607154,
        'CT': 3588184,
        'DE': 961939,
        'DC': 693972,
        'FL': 20984400,
        'GA': 10429379,
        'HI': 1427538,
        'ID': 1716943,
        'IL': 12802023,
        'IN': 6666818,
        'IA': 3145711,
        'KS': 2913123,
        'KY': 4454189,
        'LA': 4684333,
        'ME': 1335907,
        'MD': 6052177,
        'MA': 6859819,
        'MI': 9962311,
        'MN': 5576606,
        'MS': 2984100,
        'MO': 6113532,
        'MT': 1050493,
        'NE': 1920076,
        'NV': 2998039,
        'NH': 1342795,
        'NJ': 9005644,
        'NM': 2088070,
        'NY': 19849399,
        'NC': 10273419,
        'ND': 755393,
        'OH': 11658609,
        'OK': 3930864,
        'OR': 4142776,
        'PA': 12805537,
        'RI': 1059639,
        'SC': 5024369,
        'SD': 869666,
        'TN': 6715984,
        'TX': 28304596,
        'UT': 3101833,
        'VT': 623657,
        'VA': 8470020,
        'WA': 7405743,
        'WV': 1815857,
        'WI': 5795483,
        'WY': 579315}

# Plot percent police killings by state vs population percentages
df = ActualVsPopulation(PK, pop_s, 'state')
plt.figure(figsize=(6, 13))
sns.barplot(y="state", x="percent", hue="type", data=df,
            order = PK.state.value_counts().index, palette=["r", "C0"])
plt.title('Actual Police Killings vs Population Distribution (by State)')
plt.show()
lam = 10 #lambda, the "expected" number
k1 = 5   #k, the actual number, when more than expected
k2 = 15  #k, the actual number, when less than expected

xx = np.arange(0,30)
ppd = poisson.pmf(xx, lam) #poisson probability distribution
cppd = poisson.cdf(xx, lam) #cumulative poisson distribution

p_k1 = poisson.pmf(k1, lam) #probability of observing k1
p_k2 = poisson.pmf(k2, lam) #and k2
c_k1 = poisson.cdf(k1, lam) #cumulative probability of observing k1
c_k2 = poisson.cdf(k2, lam) #and k2

plt.figure()
plt.plot(xx,ppd, 'o')
plt.plot([k1, k1], [0, p_k1], 'g--')
plt.plot([0, k1], [p_k1, p_k1], 'g--')
plt.plot([k2, k2], [0, p_k2], 'r--')
plt.plot([0, k2], [p_k2, p_k2], 'r--')
plt.title(r'Poisson Distribution, $\lambda=10$')
plt.xlabel('k')
plt.ylabel('Probability')
plt.xlim([0, 30])
plt.ylim([0, 0.14])
plt.show()

plt.figure()
plt.plot(xx,cppd, 'o')
c_k1 = poisson.cdf(k1, lam)
plt.plot([k1, k1], [0, c_k1], 'g--')
plt.plot([0, k1], [c_k1, c_k1], 'g--')
plt.plot([k2, k2], [0, c_k2], 'r--')
plt.plot([0, k2], [c_k2, c_k2], 'r--')
plt.title(r'Poisson Cumulative Distribution, $\lambda=10$')
plt.xlabel('k')
plt.ylabel('Cumulative Probability')
plt.xlim([0, 30])
plt.ylim([0, 1])
plt.show()
# Compute the cumulative Poisson distribution value for each state
natl_rate = PK.shape[0]/sum(pop_s.values()) #national average rate of police shootings per person
cum_poisson_vals = [] #to store cumulative poisson values
for state in pop_s.keys():
    lam = natl_rate*pop_s[state] #num shootings for this state if rate matched national avg (lambda)
    k = PK[PK.state==state].shape[0] #actual number of shootings in this state
    cum_poisson_vals.append(poisson.cdf(k, lam))

# Plot cumulative Poisson distribution values
plt.figure(figsize=(6, 13))
EK = pd.DataFrame(data={'state': list(pop_s.keys()), 
                        'cum_poisson': cum_poisson_vals})
EK.sort_values('cum_poisson', inplace=True)
y = np.arange(len(EK.state))
plt.barh(y, EK.cum_poisson-0.5, left=0.5)
ax = plt.gca()
ax.set_yticks(y)
ax.set_yticklabels(EK.state)
ax.invert_yaxis()
plt.plot([0.5, 0.5], [y[0]-1, y[-1]+1], 'k', linewidth=2)
plt.title('Police Shootings: Cumulative Probability\ngiven national average, by State')
plt.ylabel('State')
plt.xlabel('Cumulative Poisson Probability')
plt.text(0.505, y[0]-1.5, 
         r'Less shootings $\leftarrow$ $\rightarrow$ More shootings',
         horizontalalignment='center', color='gray')
plt.show()
# Compute the cumulative Poisson distribution value for each race
cum_poisson_vals = []
for race in pop_r.keys():
    lam = PK.shape[0]*pop_r[race]/100 #num shootings of people of this race if killings were random (lambda)
    k = PK[PK.race==race].shape[0] #actual number of shootings of people of this race
    cum_poisson_vals.append(poisson.cdf(k, lam))

# Plot cumulative Poisson distribution values
plt.figure()
EK = pd.DataFrame(data={'race': list(pop_r.keys()), 
                        'cum_poisson': cum_poisson_vals})
EK.sort_values('cum_poisson', inplace=True)
y = np.arange(len(EK.race))
plt.barh(y, EK.cum_poisson-0.5, left=0.5)
plt.plot([0.5, 0.5], [y[0]-1, y[-1]+1], 'k', linewidth=2)
ax = plt.gca()
ax.set_yticks(y)
ax.set_yticklabels(EK.race)
ax.invert_yaxis()
plt.title('Police Shootings: Cumulative Probability\ngiven national average, by Race')
plt.ylabel('Race')
plt.xlabel('Cumulative Poisson Probability')
plt.text(0.506, y[0]-0.75, 
         r'Less shootings $\leftarrow$ $\rightarrow$ More shootings',
         horizontalalignment='center', color='gray')
plt.show()
# Convert date from object to datetime
PK.date = pd.to_datetime(PK.date)
# Plot shootings by month
plt.figure(figsize=(5,8))
sns.countplot(y=PK.date.dt.strftime('%Y %m %B'), 
              order=sorted(PK.date.dt.strftime('%Y %m %B').unique()))
plt.title('Police Killings By Month since start of dataset')
plt.show()
# Plot shootings by day of week
dow_map={0:'M', 1:'T', 2:'W', 3:'Th', 4:'F', 5:'Sa', 6:'Su'}
sns.countplot(x=PK.date.dt.dayofweek.map(dow_map), order=dow_map.values())
plt.title('Police Killings By Day of Week')
plt.show()
# Plot how many individuals showed signs of mental illness
ax = sns.countplot(x='signs_of_mental_illness', data=PK)
for p in ax.patches:
    x = p.get_bbox().get_points()[:,0]
    y = p.get_bbox().get_points()[1,1]
    ax.annotate('{:.2g}%'.format(100.*y/len(PK)), (x.mean(), y), ha='center', va='bottom')
plt.title('Individuals showing signs of mental illness\nwhen shot by police')
plt.show()
# Plot how many individuals were fleeing when shot
ax = sns.countplot(x='flee', data=PK)
for p in ax.patches:
    x = p.get_bbox().get_points()[:,0]
    y = p.get_bbox().get_points()[1,1]
    ax.annotate('{:.2g}%'.format(100.*y/len(PK)), (x.mean(), y), ha='center', va='bottom')
plt.title('Method of fleeing\nwhen shot by police')
plt.show()
# Plot how many shootings have body camera footage
ax = sns.countplot(x='body_camera', data=PK)
for p in ax.patches:
    x = p.get_bbox().get_points()[:,0]
    y = p.get_bbox().get_points()[1,1]
    ax.annotate('{:.2g}%'.format(100.*y/len(PK)), (x.mean(), y), ha='center', va='bottom')
plt.title('Whether an officer was wearing a body camera \nat the time of the shooting')
plt.show()
# Count the proportion of shootings w/ body camera by state
pcPK = PK.groupby('state').agg({'body_camera': 'mean'})

# Plot percent shootings w/ body camera by state
plt.figure(figsize=(6, 13))
sns.barplot(y=pcPK.index, 
            x=100.*pcPK.values.flatten(),
            order=pcPK.body_camera.sort_values(ascending=False).index)
plt.title('Percent of Police Killings\nwith body camera, by State')
plt.xlabel('Percent of Shootings with body camera')
plt.xlim([0, 100])
plt.show()