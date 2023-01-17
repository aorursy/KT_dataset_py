from IPython.display import HTML



HTML('''<script>

code_show=true; 

function code_toggle() {

 if (code_show){

 $('div.input').hide();

 } else {

 $('div.input').show();

 }

 code_show = !code_show

} 

$( document ).ready(code_toggle);

</script>

<form action="javascript:code_toggle()"><input type="submit" value="Click here to toggle on/off the raw code."></form>''')
# Including a title Image

from IPython.display import Image

%matplotlib inline

Image("../input/world-map-countries/World-Map.jpg", width=600, height=600)
import numpy as np

import matplotlib.pyplot as plt

import pandas as pd



import plotly

# plotly standard imports

import plotly.graph_objs as go

import plotly.plotly as py



# Cufflinks wrapper on plotly

import cufflinks as cf



# Options for pandas

#pd.options.display.max_columns = 30



# Display all cell outputs

from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = 'all'



from plotly.offline import iplot, init_notebook_mode, plot

cf.go_offline()



init_notebook_mode(connected=True)



# Set global theme

cf.set_config_file(world_readable=True, theme='pearl')



import warnings  

warnings.filterwarnings('ignore')

np.seterr(divide='ignore', invalid='ignore')
dataset = pd.read_excel('../input/world-countries/countries_of_the_world.xls',skiprows=3)
dataset.head()
dataset.tail(2)
%matplotlib inline

Image("../input/how-many-countries/how-many-countries-2018.png", width=600, height=600)
# Fixing messy column names

dataset.columns = dataset.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('(', '').str.replace(')', '')

dataset.info()
dataset.isnull().sum()
# Keep a copy of the original dataframe

orig_dataset = dataset.copy()





dataset['area_sq._km.'] = dataset['area_sq._mi.']*2.59

dataset['pop._density_per_sq._km.'] = dataset['pop._density_per_sq._mi.']*.3861
dataset['area_sq._mi.'][9],dataset['area_sq._km.'][9]
dataset['pop._density_per_sq._mi.'][9],dataset['pop._density_per_sq._km.'][9]
dataset['region'] = dataset['region'].str.strip().str.lower().str.replace(' ', '_').str.replace('(', '').str.replace(')', '')

dataset['country'] = dataset['country'].str.strip().str.lower().str.replace(' ', '_').str.replace('(', '').str.replace(')', '')
dataset.describe()
import seaborn as sns; sns.set(style='white')

sns.palplot(sns.color_palette("Blues"))



descending_order = dataset['region'].value_counts().sort_values(ascending=False).index



%matplotlib inline

fig = plt.figure(figsize=(10,8))

ax = fig.add_subplot(111)

sns.countplot(y=dataset['region'],color='lightgray',order=descending_order)

#Get rid of top and right border:

sns.despine(offset=0, trim=False)

# Change the colors of the left and bottom borders (fade into the background)

ax.spines['left'].set_color('lightgray')

ax.spines['bottom'].set_color('lightgray')

# Attract attention

ax.patches[0].set_fc('cornflowerblue')





plt.title('Distribution of Regions',fontsize=14)

plt.xlabel('Number of Countries',fontsize=12)

plt.ylabel('Region',fontsize=12)
from wordcloud import WordCloud, STOPWORDS



%matplotlib inline

text = dataset['region'].values

wordcloud = WordCloud(

    width = 3000,

    height = 2000,

    background_color = 'lightgrey',

    stopwords = STOPWORDS).generate(str(text))



fig = plt.figure(

    figsize = (14, 10),

    facecolor = 'lightgray',

    edgecolor = 'lightgray')

plt.imshow(wordcloud, interpolation = 'bilinear')

plt.axis('off')

plt.tight_layout(pad=0)
sub_saharan_africa = dataset[dataset['region']=='sub-saharan_africa']
sub_saharan_africa.describe()

# Total population in Sub-Saharan Africa....not sure as of what year?

print('Sub-Saharan Africa has a total population of;',(sub_saharan_africa['population'].sum())/1000000,'Million')
fig, ax = plt.subplots(1, 2, figsize=(16,6))



sns.distplot(sub_saharan_africa['population'], ax=ax[0], color='darkgray')

ax[0].set_title('Distribution of Sub-Saharan Africa population', fontsize=14)

#Get rid of top and right border:

sns.despine(offset=0, trim=False)

ax[0].set_xlim([min(sub_saharan_africa['population']), max(sub_saharan_africa['population'])])

ax[0].set_xlabel('Population')

# Attract attention

ax[0].patches[0].set_fc('cornflowerblue')



sns.distplot(sub_saharan_africa['pop._density_per_sq._km.'], ax=ax[1], color='darkgray')

ax[1].set_title('Distribution of Sub-Saharan Africa population density', fontsize=14)

ax[1].set_xlim([min(sub_saharan_africa['pop._density_per_sq._km.']), max(sub_saharan_africa['pop._density_per_sq._km.'])])

ax[1].set_xlabel('Population density (people per square km)')
fig, ax = plt.subplots(1, 2, figsize=(16,6))



sns.distplot(sub_saharan_africa['infant_mortality_per_1000_births'].fillna(0), ax=ax[0], color='lightgreen')

ax[0].set_title(r'Distribution of Sub-Saharan Africa infant mortality per 1000 births', fontsize=14)

#Get rid of top and right border:

sns.despine(offset=0, trim=False)

ax[0].set_xlim([min(sub_saharan_africa['infant_mortality_per_1000_births']), max(sub_saharan_africa['infant_mortality_per_1000_births'])])

ax[0].set_xlabel('Infant mortality per 1000 births')



sns.distplot(sub_saharan_africa['net_migration'].fillna(0), ax=ax[1], color='lightgray')

ax[1].set_title('Distribution of Sub-Saharan Africa net migration', fontsize=14)

ax[1].set_xlim(-1.0,1.0)

ax[1].set_xlabel('Net migration')

# Attract attention

#ax[1].patches[10].set_fc('cornflowerblue')
fig, ax = plt.subplots(1, 2, figsize=(16,6))



sns.distplot(sub_saharan_africa['literacy_%'].fillna(0), ax=ax[0], color='lightgray')

ax[0].set_title(r'Distribution of literacy in Sub-Saharan Africa', fontsize=14)

#Get rid of top and right border:

sns.despine(offset=0, trim=False)

ax[0].set_xlim([min(sub_saharan_africa['literacy_%']), max(sub_saharan_africa['literacy_%'])])

ax[0].set_xlabel('Level of literacy')



sns.distplot(sub_saharan_africa['industry'].fillna(0), ax=ax[1], color='lightgray')

ax[1].set_title('Distribution of industry in Sub-Saharan Africa', fontsize=14)

ax[1].set_xlim([min(sub_saharan_africa['industry']), max(sub_saharan_africa['industry'])])

ax[1].set_xlabel('Industry')

# Attract attention

#ax[1].patches[23].set_fc('cornflowerblue')
fig, ax = plt.subplots(1, 2, figsize=(16,6))



sns.distplot(sub_saharan_africa['birthrate'].fillna(0), ax=ax[0], color='lightgray')

ax[0].set_title(r'Distribution of birthrate in Sub-Saharan Africa',fontsize=14)

#Get rid of top and right border:

sns.despine(offset=0, trim=False)

ax[0].set_xlim([min(sub_saharan_africa['birthrate']), max(sub_saharan_africa['birthrate'])])

ax[0].set_xlabel('Birthrate')



sns.distplot(sub_saharan_africa['deathrate'].fillna(0), ax=ax[1], color='lightgray')

ax[1].set_title('Distribution of deathrate in Sub-Saharan Africa',fontsize=14)

ax[1].set_xlim([min(sub_saharan_africa['deathrate']), max(sub_saharan_africa['deathrate'])])

ax[1].set_xlabel('Deathrate')

# Attract attention

#ax[1].patches[23].set_fc('cornflowerblue')
#sub_saharan_africa.loc['namibia']
lat_lon_df = pd.read_csv('../input/world-countries/sub_saharan_africa.csv')
lat_lon_df['latitude'] = lat_lon_df['latitude'].astype(int)

lat_lon_df['longitude'] = lat_lon_df['longitude'].astype(int)
#Plotting an image as the background and the data on the forefront :)

import matplotlib.image as mpimg

africa_img=mpimg.imread('../input/map-of-africa/Africa_map.png')

ax = lat_lon_df.plot(kind="scatter", x='longitude', y='latitude', figsize=(14,12),

                       s=lat_lon_df['population']/100000, label="Population",

                       c='population', cmap=plt.get_cmap("jet"),

                       colorbar=False, alpha=0.4,

                      )

plt.imshow(africa_img, extent=[-25, 58, -33, 23], alpha=0.2,

           aspect='auto',cmap=plt.get_cmap("jet"))

plt.ylabel("Latitude", fontsize=14)

plt.xlabel("Longitude", fontsize=14)



population = lat_lon_df['population']

tick_values = np.linspace(population.min(), population.max(), 10)

cbar = plt.colorbar()

cbar.ax.set_yticklabels(["%dM"%(round(v)/1000000) for v in tick_values], fontsize=14)

cbar.set_label('Population', fontsize=16)



plt.legend(fontsize=16)