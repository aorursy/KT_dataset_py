### Standard imports
import numpy as np
import pandas as pd
pd.options.display.max_columns = 50

# Will count number of unique words in a sentence
from collections import Counter

### Standard plotting imports
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
sns.set()

### Fancy plot of Earth (This library is really cool and fast!)
import folium
from folium.plugins import MarkerCluster

### Advanced plotting import
# Altair
import altair as alt
# alt.renderers.enable('notebook')
# Plotly
from plotly.offline import init_notebook_mode, iplot
import plotly.graph_objs as go
import plotly.plotly as py
from plotly import tools
init_notebook_mode(connected=True)

# Helps visualize wordy features
from wordcloud import WordCloud

### Removes warnings from output
import warnings
warnings.filterwarnings('ignore')

DATA_PATH = "../input/"
import json  # need it for json.dumps
from altair.vega import v3
from IPython.display import HTML

# Create the correct URLs for require.js to find the Javascript libraries
vega_url = 'https://cdn.jsdelivr.net/npm/vega@' + v3.SCHEMA_VERSION
vega_lib_url = 'https://cdn.jsdelivr.net/npm/vega-lib'
vega_lite_url = 'https://cdn.jsdelivr.net/npm/vega-lite@' + alt.SCHEMA_VERSION
vega_embed_url = 'https://cdn.jsdelivr.net/npm/vega-embed@3'
noext = "?noext"

altair_paths = {
    'vega': vega_url + noext,
    'vega-lib': vega_lib_url + noext,
    'vega-lite': vega_lite_url + noext,
    'vega-embed': vega_embed_url + noext
}

workaround = """
requirejs.config({{
    baseUrl: 'https://cdn.jsdelivr.net/npm/',
    paths: {paths}
}});
"""

# Define the function for rendering
def add_autoincrement(render_func):
    # Keep track of unique <div/> IDs
    cache = {}
    def wrapped(chart, id="vega-chart", autoincrement=True):
        """Render an altair chart directly via javascript.
        
        This is a workaround for functioning export to HTML.
        (It probably messes up other ways to export.) It will
        cache and autoincrement the ID suffixed with a
        number (e.g. vega-chart-1) so you don't have to deal
        with that.
        """
        if autoincrement:
            if id in cache:
                counter = 1 + cache[id]
                cache[id] = counter
            else:
                cache[id] = 0
            actual_id = id if cache[id] == 0 else id + '-' + str(cache[id])
        else:
            if id not in cache:
                cache[id] = 0
            actual_id = id
        return render_func(chart, id=actual_id)
    # Cache will stay defined and keep track of the unique div Ids
    return wrapped


@add_autoincrement
def render_alt(chart, id="vega-chart"):
    # This below is the javascript to make the chart directly using vegaEmbed
    chart_str = """
    <div id="{id}"></div><script>
    require(["vega-embed"], function(vegaEmbed) {{
        const spec = {chart};     
        vegaEmbed("#{id}", spec, {{defaultStyle: true}}).catch(console.warn);
    }});
    </script>
    """
    return HTML(
        chart_str.format(
            id=id,
            chart=json.dumps(chart) if isinstance(chart, dict) else chart.to_json(indent=None)
        )
    )

HTML("".join((
    "<script>",
    workaround.format(paths=json.dumps(altair_paths)),
    "</script>"
)))
# A short hand way to plot most bar graphs
def pretty_bar(data, ax, xlabel=None, ylabel=None, title=None, int_text=False, x=None, y=None):
    
    if x is None:
        x = data.values
    if y is None:
        y = data.index
    
    # Plots the data
    fig = sns.barplot(x, y, ax=ax)
    
    # Places text for each value in data
    for i, v in enumerate(x):
        
        # Decides whether the text should be rounded or left as floats
        if int_text:
            ax.text(0, i, int(v), color='k', fontsize=14)
        else:
            ax.text(0, i, round(v, 3), color='k', fontsize=14)
     
    ### Labels plot
    ylabel != None and fig.set(ylabel=ylabel)
    xlabel != None and fig.set(xlabel=xlabel)
    title != None and fig.set(title=title)

def plotly_bar(df, col):
    value_counts = df[col].value_counts()
    labels = list(value_counts.index)
    values = list(value_counts.values)
    trace = go.Bar(x=labels, y=values)
    return trace
    
### Used to style Python print statements
class color:
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'
HTML('<iframe width="560" height="315" src="https://www.youtube.com/embed/DxEpyjWDB6I?rel=0" frameborder="0" allow="autoplay; encrypted-media" allowfullscreen></iframe>')
HTML('<iframe width="560" height="315" src="https://www.youtube.com/embed/UD8sHa84M_Q?rel=0" frameborder="0" allow="autoplay; encrypted-media" allowfullscreen></iframe>')
HTML('<iframe width="560" height="315" src="https://www.youtube.com/embed/JBOArELXkqA?rel=0" frameborder="0" allow="autoplay; encrypted-media" allowfullscreen></iframe>')
sites = pd.read_csv(DATA_PATH + "sites.csv")

m = folium.Map(location=[60, -125], zoom_start=3)
marker_cluster = MarkerCluster().add_to(m)

for lat, lon, nst_site in sites[["latitude", "longitude", "nst_site"]].values:    
    folium.Marker(
        location = [lat, lon],
        popup = f"<h1>Site: {nst_site}</h1><p>Latitude: {round(lat, 3)}</p><p>Longitude: {round(lon, 3)}</p>",
        icon = folium.Icon(color='green', icon='ok-sign'),
    ).add_to(marker_cluster)

m
pollutants = pd.read_csv(DATA_PATH + "pollutants.csv")
print("Shape of pollutants data:", pollutants.shape)

num_cols = [col for col in pollutants.columns if pollutants[col].dtype != object]
cat_cols = [col for col in pollutants.columns if pollutants[col].dtype == object]

print("\n{}Numeric columns:{}".format(color.UNDERLINE, color.END))
print(" --- ".join(num_cols))

print("\n{}Categoric columns:{}".format(color.UNDERLINE, color.END))
print(" --- ".join(cat_cols))

pollutants.head()
missing_cols = pollutants.isnull().sum()
missing_cols = missing_cols[missing_cols > 0].index

temp = pd.DataFrame(pollutants[missing_cols].isnull().sum()).reset_index()
temp.columns = ["Feature", "Number Missing"]

temp
temp = pollutants.fiscal_year.value_counts().sort_index()

trace = go.Scatter(
        x=temp.index,
        y=temp.values,
    )

layout = go.Layout(
    xaxis=dict(
        linewidth=2,
        tickfont=dict(
            family='Arial',
            size=12
        )
    ),
    title="Number of studies over the years")



fig = go.Figure(data=[trace], layout=layout)
iplot(fig)
pollutants[(pollutants.general_location.isnull()) | (pollutants.specific_location.isnull())]
pollutants.loc[7761, "specific_location"] = "Chrome Bay"
pollutants.loc[7761, "general_location"]  = "Cook Inlet"

pollutants.loc[7780, "specific_location"] = "LOCATION UNKNOWN"
pollutants.loc[7780, "general_location"]  = "LOCATION UNKNOWN"

pollutants.loc[7790, "specific_location"] = "Tutka Bay"
pollutants.loc[7790, "general_location" ] = "Kachemak Bay"
fig, ax = plt.subplots(1, 1, figsize=(15, 8))
temp = pollutants.general_location.value_counts().head(20)
pretty_bar(temp, ax, title="Most popular General Locations", xlabel="Count")
temp = (pollutants
        .groupby("general_location")
        .specific_location
        .value_counts()
        .rename(columns={"specific_location": "Count"})
        .sort_values(ascending=False)
        .head(17))
temp = pd.DataFrame(temp).reset_index()
temp.columns = ["General Location", "Specific Location", "Count"]

render_alt(
    alt.Chart(temp).mark_bar().encode(
        x='Count',
        y='Specific Location',
        color="General Location",
        tooltip=['Count', "Specific Location", "General Location"])
    .properties(width=700, title="Most popular Specific Locations")
)
def substance_map(x):
    if x in pahs:
        return "PAH"
    elif x in ddts:
        return "DDT"
    elif x in org_chr:
        return "Organochloride insecticides"
    elif x in pcbs:
        return "PCB"
    elif x in other:
        return x
    return np.nan

other = {"Copper", "Cadmium", "Lead", "Chromium", "Zinc", "Nickel", "Silver", "Tin", "Mercury", "Selenium", 
         "Manganese", "Iron", "Aluminum", "Antimony", "Thallium", "Arsenic"}

# NOTE: I couldn't find much information on Pentachloroanisole.  It has the elements of a typical organochlorine compound
# NOTE: Endosulfan II and Endosulfan Sulfate have sulfur in it
org_chr = {"Gamma-Hexachlorocyclohexane", "Aldrin", "Mirex", "Dieldrin", "Heptachlor", "Heptachlor-Epoxide",
           "Endrin", "Oxychlordane", "Delta-Hexachlorocyclohexane", "Beta-Hexachlorocyclohexane",
           "Alpha-Hexachlorocyclohexane", "Pentachlorobenzene", "Endosulfan I", "Alpha-Chlordane",
           "Gamma-Chlordane", "Pentachloroanisole", "1,2,3,4-Tetrachlorobenzene", "Cis-Nonachlor",
           "Endosulfan II", "Endosulfan Sulfate", "1,2,4,5-Tetrachlorobenzene", "Trans-Nonachlor"}

# NOTE: Phenanthrenes_Anthracenes is here because I see articles relating it to petroleum (also it ends in -ene)
pahs = {"Benzo[a]pyrene", "Benz[a]anthracene", "Dibenzo[a,h]anthracene", "Benzo[e]pyrene", "Anthracene",
        "Fluorene", "Perylene", "Chrysene", "Fluoranthene", "Pyrene", "Acenaphthene", "Indeno[1,2,3-c,d]pyrene",
        "Benzo[g,h,i]perylene", "Phenanthrene", "Acenaphthylene", "Benzo[b]fluoranthene", "Naphthalene",
        "Hexachlorobenzene", "Benzo[k]fluoranthene", "C1-Naphthalenes", "C2-Naphthalenes",
        "C3-Naphthalenes", "C4-Naphthalenes", "C4-Chrysenes", "C3-Chrysenes", "C1-Chrysenes",
        "C2-Chrysenes", "C1-Fluorenes", "C2-Fluorenes", "C3-Fluorenes", "1-Methylnaphthalene",
        "2-Methylnaphthalene", "1-Methylphenanthrene", "C3-Phenanthrenes_Anthracenes",
        "C4-Phenanthrenes_Anthracenes", "C2-Phenanthrenes_Anthracenes", "C1-Phenanthrenes_Anthracenes",
        "2,6-Dimethylnaphthalene", "1,6,7-Trimethylnaphthalene", "C3-Fluoranthenes_Pyrenes",
        "C2-Fluoranthenes_Pyrenes", "Dibenzofuran", "Naphthobenzothiophene",
        "C3-Naphthobenzothiophene", "C2-Naphthobenzothiophene", "C1-Naphthobenzothiophene",
        "C2-Decalin", "C1-Decalin", "C4-Decalin", "C3-Decalin", "Decalin",
        "C1-Fluoranthenes_Pyrenes"}

pcbs = {"PCB195_208", "PCB66", "PCB128", "PCB105", "PCB206", "PCB44", "PCB101_90", "PCB180",
        "PCB138_160", "PCB8_5", "PCB153_132_168", "PCB18", "PCB170_190", "PCB52", "PCB28",
        "PCB187", "PCB118", "PCB209", "PCB201_173_157", "PCB149_123", "PCB31", "PCB49",
        "PCB110_77", "PCB29", "PCB70", "PCB158", "PCB156_171_202", "PCB183", "PCB194",
        "PCB151", "PCB174", "PCB95", "PCB99", "PCB87_115", 'PCB169', 'PCB 126', 'PCB56_60',
        'PCB45', 'PCB 104', 'PCB188', 'PCB146', 'PCB74_61', 'PCB77', 'PCB 154', 'PCB112',
        'PCB126', 'PCB', 'PCB18_17', 'PCB199'}

ddts = {"4,4'-DDT", "2,4'-DDT", "4,4'-DDD", "2,4'-DDD", "2,4'-DDE", "4,4'-DDE"}

pollutants["substance"] = pollutants.parameter.map(substance_map)
pollutants = pollutants.drop(pollutants[pollutants.substance.isnull()].index, axis=0)
fig, ax = plt.subplots(1, 1, figsize=(12, 12))
pretty_bar(pollutants.substance.value_counts(), ax, title="Substances", xlabel="Count")
def organism_map(x):
    if x in sediment:
        return "Sediment"
    elif x in bivalves:
        return "Bivalves"
    elif x in fish:
        return "Fish"
    return np.nan

def specific_bivalve_map(x):
    if x in oysters:
        return "Oyster"
    elif x in clams:
        return "Clam"
    elif x in mussels:
        return "Mussel"
    return np.nan

### Bivalves
oysters  = {"Crassostrea virginica", "Ostrea sandvicensis", "Crassostrea rhizophorae",
           "Isognomon alatus", "Crassostrea corteziensis"}

clams    = {"Clinocardium nuttallii", "Mya Arenaria", "Anadara tuberculosa",
           "Protothaca staminea", "Anadara similis", "mytella guyanensis",
           "Donax denticulatus", "Siliqua Patula", "Corbicula fluminea",
           "Protothaca grata", "Mytella falcata", "Ctenoides scabra", "Chama sinuosa"}

mussels  = {"Mytilus species", "Mytilus edulis", "Dreissena", "Dreissena species", "Perumytilus purpuratus",
           "Aulacomya ater", "Perna Perna", "Semimytilus algosus", "Mytilus platensis",
           "Bracchidonies rodrigezii", "Choromytilus chorus", "Geukensia demissa"}

### Fish
flounder = {"Pleuronectes americanus", "Platichthys stellatus", "Flatfish"}

fish     = {"Lutjanus griseus", "Diapterus auratus", "Osmerus mordax",
            "Diplodus argenteus", "Hexagrammos decagrammus", "Shrimp", "Starfish"}

salmon   = {"Oncorhynchus keta", "Oncorhynchus nerka"}

### Sediment          
sediment = {"Sediment", "Surface Sediment"}

bivalves  = set.union(oysters, clams, mussels)

fish      = set.union(flounder, fish, salmon)

pollutants["organism"] = pollutants.scientific_name.map(organism_map)
pollutants["specific_bivalve"] = pollutants.scientific_name.map(specific_bivalve_map)
pollutants = pollutants.drop(pollutants[pollutants.organism.isnull()].index, axis=0)
fig, axarr = plt.subplots(2, 1, figsize=(14, 12))

pretty_bar(pollutants.organism.value_counts(), axarr[0], title="Organism", xlabel="Counts")
pretty_bar(pollutants.specific_bivalve.value_counts(), axarr[1], title="Specific groups within Bivalves", xlabel="Counts")
### Convert all units to the same unit
is_mg = pollutants.units.map(lambda x: 999 * int(x == "micrograms per dry gram") + 1)
pollutants.result = pollutants.result * is_mg
pollutants.units = pollutants.units.map(lambda x: "ng/dry g" if x == "micrograms per dry gram" else x)
fig, axarr = plt.subplots(3, 1, figsize=(10, 20))

### Find the median result
temp = pd.DataFrame(pollutants.groupby(["organism", "substance"])["result"].median())
temp = temp[temp.result > 0].reset_index().groupby("organism")

plt.suptitle("Median result found for each Organism", fontsize=24)
i = 0
for organism, group in temp:
    ax = axarr[i]
    pretty_bar(None, x=group["result"], y=group["substance"], ax=ax, title=organism, ylabel="", xlabel="ng/dry g")
    i += 1
fig, axarr = plt.subplots(3, 1, figsize=(10, 20))

### Find the median result
temp = pd.DataFrame(pollutants.groupby(["specific_bivalve", "substance"])["result"].median())
temp = temp[temp.result > 0].reset_index().groupby("specific_bivalve")

plt.suptitle("Median result found for a specific Bivalve", fontsize=24)
i = 0
for organism, group in temp:
    ax = axarr[i]
    pretty_bar(None, x=group["result"], y=group["substance"], ax=ax, title=organism, ylabel="", xlabel="ng/dry g")
    i += 1
### Create individual figures
fig = tools.make_subplots(rows=1, cols=1)

### Past
past = pollutants[pollutants.fiscal_year < 2001]
past = pd.DataFrame(past.groupby(["organism", "substance"])["result"].median())
past = past[past.result > 0].reset_index().groupby("organism")

### Current
curr = pollutants[pollutants.fiscal_year >= 2001]
curr = pd.DataFrame(curr.groupby(["organism", "substance"])["result"].median())
curr = curr[curr.result > 0].reset_index().groupby("organism")    

for label, period in (("Before 2001", past), ("After 2001", curr)):
    for organism, group in period:
        trace = go.Bar(
                        y=group["result"],
                        x=group["substance"],
                        name=label
                    )
        fig.append_trace(trace, 1, 1)

### Create buttons for drop down menu
organisms = ['Bivalves', 'Fish', 'Sediment']

buttons = []
for i in range(len(organisms)):
    visibility = [i==j for j in range(len(organisms))]
    button = dict(
                 label =  organisms[i],
                 method = 'update',
                 args = [{'visible': visibility},
            {'title': organisms[i]}])
    buttons.append(button)

updatemenus = list([
    dict(active=-1, x=-0.15, buttons=buttons)
])

layout = go.Layout(
    title="Before/After comparison of substance measurements",
    yaxis=dict(title="nanograms per dry gram"),
    updatemenus=updatemenus,
    height=600, 
    width=800
)

fig['layout'] = layout
iplot(fig)
subs = set(pollutants.substance.unique())

datas = []
for sub in subs:
    temp = pollutants[(pollutants.scientific_name == "Sediment") & (pollutants.substance == sub)]
    datas.append((sub, temp.groupby(["fiscal_year"])["result"].median()))

traces = []

for sub, data in datas:
    traces.append(go.Scatter(
                            x=data.index,
                            y=data.values, 
                            name=sub
                        ))
buttons = []

for i, sub in enumerate(subs):
    visibility = [i==j for j in range(len(subs))]
    button = dict(
                 label =  sub,
                 method = 'update',
                 args = [{'visible': visibility},
                     {'title': f"{sub} Levels" }])
    buttons.append(button)

updatemenus = list([
    dict(active=-1,
         x=-0.15,
         buttons=buttons
    )
])

layout = dict(title='Substance levels over time', 
              showlegend=False,
              updatemenus=updatemenus,
              yaxis=dict(title="nanograms per dry gram"))

fig = dict(data=traces, layout=layout)
fig['layout'].update(height=800, width=800)

iplot(fig)
# Here I load the histopaths data and provide basic information of the dataset.
histopaths = pd.read_csv(DATA_PATH + "histopaths.csv")
print(f"Shape of histopaths data: {color.BOLD}{histopaths.shape}{color.END}")
### Interestingly, there are several columns that are completely empty!
### These features are: 
cols = ["edema", "gonad_subsample_wet_weight", "hydra_gill",
        "nemertine_gill", "other_trematode_sporocyst_gill",
        "other_trematode_sporocyst_gut", "tumor"]

### In addition there are also many columns that only have 10 entries present!
### These features are:
cols.extend(["abnormality", "abnormality_description", "chlamydia", 
        "metacercaria", "pseudoklossia", 
        "rickettsia_digestive_tubule", "rickettsia_gut"])

### There are also some columns that are boring and contain only one unique value.
### These should be removed as well
cols.extend(["multinucleated_sphere_x",
             "pea_crab", "proctoeces", "neoplasm",
             "unusual_digestive_tubule", "unidentified_gonoduct_organism",
             "station_letter", "multinucleated_sphere_x_description",
             "focal_necrosis", "diffuse_necrosis", "ciliate_large_gill"])

### Columns lacking information
cols.extend(["unidentified_organism", "focal_inflammation", 
             "diffuse_inflammation", "trematode_metacercariae_description", 
             "condition_code"])

N = histopaths.shape[1]
histopaths = histopaths.drop(cols, axis=1)
print(f"Deleted {color.BOLD}{N - histopaths.shape[1]}{color.END} features")
num_cols = [col for col in histopaths.columns if histopaths[col].dtype != object]
cat_cols = [col for col in histopaths.columns if histopaths[col].dtype == object]

print(f"\n{color.UNDERLINE}Numeric columns:{color.END}")
print(" --- ".join(num_cols))

print(f"\n{color.UNDERLINE}Categoric columns:{color.END}")
print(" --- ".join(cat_cols))

histopaths.head()
missing_cols = histopaths.isnull().sum()
missing_cols = histopaths[missing_cols[missing_cols > 0].index].isnull().sum()
missing_cols = missing_cols.reset_index()
missing_cols.columns = ["Column", "Num missing"]

missing_cols
plt.figure(figsize=(16, 15))

### Nuanced way of creating subplots
ax1 = plt.subplot2grid((3, 2), (0, 0))
ax2 = plt.subplot2grid((3, 2), (0, 1))
ax3 = plt.subplot2grid((3, 2), (1, 0))
ax4 = plt.subplot2grid((3, 2), (1, 1))
ax5 = plt.subplot2grid((3, 2), (2, 0), colspan=2)

for col in ("source_file", "sex", "species_name", "study_name"):
    histopaths[col].fillna("NaN", inplace=True)
    
### source_file
pretty_bar(histopaths.source_file.value_counts(), ax1,
           title="Source File", xlabel="Counts")

### species_name
pretty_bar(histopaths.species_name.value_counts(), ax2,
           title="Species name", xlabel="Counts")

### study_name
pretty_bar(histopaths.study_name.value_counts(), ax3,
           title="Study name", xlabel="Counts")

### sex
sns.countplot(histopaths.sex, ax=ax4).set(title="Sex", xlabel="Sex", ylabel="Counts")

ax5.plot(histopaths.fiscal_year.value_counts().sort_index())
ax5.plot(histopaths.fiscal_year.value_counts().sort_index(), ".", markersize=13)
ax5.set_title("Fiscal Year")
ax5.set_ylabel("Counts");
plt.figure(figsize=(16, 12))

### Nuanced way of creating subplots
ax1 = plt.subplot2grid((2, 2), (0, 0))
ax2 = plt.subplot2grid((2, 2), (0, 1))
ax3 = plt.subplot2grid((2, 2), (1, 0), colspan=2)

for ax, col, title in [(ax1, "general_location",        "General Location"),
                       (ax2, "specific_location",       "Specific Location"),
                       (ax3, "coastal_ecological_area", "Coastal Ecological Area")]:
    text = histopaths[col].value_counts()
    
    wc = WordCloud().generate_from_frequencies(text)
    ax.imshow(wc)
    ax.axis('off')
    ax.set_title(title, fontsize=28)
plt.figure(figsize=(16, 12))

### Nuanced way of creating subplots
ax1 = plt.subplot2grid((2, 2), (0, 0))
ax2 = plt.subplot2grid((2, 2), (0, 1))
ax3 = plt.subplot2grid((2, 2), (1, 0), colspan=2)

### specific_region
pretty_bar(histopaths.specific_region.value_counts(), ax1,
           xlabel="Counts", title="Specific regions")

### region
explode = [0.02, 0.02, 0.4, 0.6]
(histopaths.region
 .value_counts()
 .plot(kind="pie", ax=ax2, explode=explode, shadow=True)
 .set(ylabel="", title="Region"))

### state_name
(sns.countplot(histopaths.state_name, ax=ax3)
    .set(title="Activity in States", xlabel="State name", ylabel="Counts"));
def remove_stopwords(text, stopwords):
    for word in stopwords:
        text = text.replace(" " + word + " ", " ")
    return text

stopwords = ["and", "but", "to", "or", "as", "the", "then", "than", "of"]

### condition_code_description
histopaths['condition_code_description'].fillna("", inplace=True)
text = " ".join(histopaths['condition_code_description']).lower()
text = remove_stopwords(text, stopwords)
freq = Counter(text.split())
wc = WordCloud(colormap='winter_r', background_color='White').generate_from_frequencies(freq)
plt.figure(figsize=(13,13))
plt.imshow(wc)
plt.axis('off')
plt.title('Condition Code Description', fontsize=28)

### digestive_tubule_atrophy_description
histopaths['digestive_tubule_atrophy_description'].fillna("", inplace=True)
text = " ".join(histopaths['digestive_tubule_atrophy_description']).lower()
text = remove_stopwords(text, stopwords)
freq = Counter(text.split())
wc = WordCloud(colormap='winter_r', background_color='Black').generate_from_frequencies(freq)
plt.figure(figsize=(13,13))
plt.imshow(wc)
plt.axis('off')
plt.title('Digestive Tubule Atrophy Description', fontsize=28);
f, ax = plt.subplots(figsize=(15, 15))

corr = histopaths.corr()

### Format the visualization from a square to a triangle
mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask)] = True

sns.heatmap(corr, vmax=.8, cmap="ocean", mask=mask);
### I assume if a values NaN, it means none showed up
cols = ["ciliate_small_gill", "ciliate_gut",
        "cestode_gill", "cestode_body", "cestode_mantle",
        "nematopsis_gill", "nematopsis_body", "nematopsis_mantle",
        "copepod_gill", "copepod_body", "copepod_gut_digestive_tubule"]
histopaths[cols] = histopaths[cols].fillna(0)

# Ciliate
s1 = histopaths.ciliate_small_gill.value_counts().reset_index()
s2 = histopaths.ciliate_gut.value_counts().reset_index()
s3 = pd.DataFrame({"Mantle": [len(histopaths)], "index": [0]})
df1 = s1.merge(s2, how="outer", on="index")
df1 = df1.merge(s3, how="outer", on="index")
df1.columns = ["Parasite Count", "Percent of Gills affected", "Percent of Bodies affected", "Percent of Mantles affected"]

# Cestode
s1 = histopaths.cestode_gill.value_counts().reset_index()
s2 = histopaths.cestode_body.value_counts().reset_index()
s3 = histopaths.cestode_mantle.value_counts().reset_index()
df2 = s1.merge(s2, how="outer", on="index")
df2 = df2.merge(s3, how="outer", on="index")
df2.columns = ["Parasite Count", "Percent of Gills affected", "Percent of Bodies affected", "Percent of Mantles affected"]

# Nemotopsis
s1 = histopaths.nematopsis_gill.value_counts().reset_index()
s2 = histopaths.nematopsis_body.value_counts().reset_index()
s3 = histopaths.nematopsis_mantle.value_counts().reset_index()
df3 = s1.merge(s2, how="outer", on="index")
df3 = df3.merge(s3, how="outer", on="index")
df3.columns = ["Parasite Count", "Percent of Gills affected", "Percent of Bodies affected", "Percent of Mantles affected"]

# Nemotopsis
s1 = histopaths.copepod_gill.value_counts().reset_index()
s2 = (histopaths.copepod_body + histopaths.copepod_gut_digestive_tubule).value_counts().reset_index()
s3 = pd.DataFrame({"Mantle": [len(histopaths)], "index": [0]})
df4 = s1.merge(s2, how="outer", on="index")
df4 = df4.merge(s3, how="outer", on="index")
df4.columns = ["Parasite Count", "Percent of Gills affected", "Percent of Bodies affected", "Percent of Mantles affected"]


df1["Organism"] = ["Ciliate"]    * len(df1)
df2["Organism"] = ["Cestode"]    * len(df2)
df3["Organism"] = ["Nematopsis"] * len(df3)
df4["Organism"] = ["Copepod"]    * len(df4)
df1 = df1.merge(df2, how="outer")
df3 = df3.merge(df4, how="outer")
df  = df1.merge(df3, how="outer")

cols = ["Percent of Gills affected", "Percent of Bodies affected", "Percent of Mantles affected"]
df[cols] = (100 * df[cols] / len(histopaths)).round(decimals=2)
df.fillna(0, inplace=True)
df.drop(df[df["Parasite Count"] == 0].index, inplace=True)

### Plotting - Gills
click = alt.selection_multi(fields=['Organism'])

render_alt(
    alt.Chart(df).mark_circle(size=60).encode(
    alt.X('Parasite Count:Q',
        scale=alt.Scale(type="log")
    ),
    y='Percent of Gills affected',
    color=alt.condition(click,'Organism', alt.value('lightgray')),
    tooltip=['Organism', "Parasite Count", "Percent of Bodies affected"]
).properties(selection=click, width=700, title="Different Organisms Inside the Gills"))
### Plotting - Bodies
click = alt.selection_multi(fields=['Organism'])

render_alt(
    alt.Chart(df).mark_circle(size=60).encode(
        alt.X('Parasite Count:Q',
            scale=alt.Scale(type="log")
        ),
        y='Percent of Bodies affected',
        color=alt.condition(click,'Organism', alt.value('lightgray')),
        tooltip=['Organism', "Parasite Count", "Percent of Bodies affected"])
    .properties(selection=click, width=700, title="Different Organisms Inside the Body").
    interactive()
)
### Plotting - Mantles
click = alt.selection_multi(fields=['Organism'])

render_alt(
    alt.Chart(df).mark_circle(size=60).encode(
        alt.X('Parasite Count:Q',
            scale=alt.Scale(type="log")
        ),
        y='Percent of Mantles affected',
        color=alt.condition(click,'Organism', alt.value('lightgray')),
        tooltip=['Organism', "Parasite Count", "Percent of Mantles affected"])
    .properties(selection=click, width=700, title="Different Organisms Inside the Mantle")
    .interactive()
)
fig, axarr = plt.subplots(1, 2, figsize=(15, 6))

histopaths.nematode.fillna(0, inplace=True)
axarr[0].plot(histopaths.nematode.value_counts(), ".", markersize=14)
axarr[0].set_yscale('log')
axarr[0].set_title("Nematode")
axarr[0].set_xlabel("Number of nematodes")
axarr[0].set_ylabel("Number Bivalves affected")

histopaths.ceroid.fillna(0, inplace=True)
axarr[1].plot(histopaths.ceroid.value_counts(), ".", markersize=7)
axarr[1].set_yscale('log')
axarr[1].set_title("Ceroid")
axarr[1].set_xlabel("Number of ceriods")
axarr[1].set_ylabel("Number Bivalves affected");
from scipy.stats import linregress

plt.figure(figsize=(16, 5))
plt.title("Full Displacement and Empty Displacement relationship", fontsize=18)
plt.xlabel("Full Displacement Volume")
plt.ylabel("Empty Displacement Volume")

### Actual data
temp1 = histopaths[(histopaths.full_displacement_volume.notnull()) & (histopaths.empty_displacement_volume.notnull())]
x1 = temp1["full_displacement_volume"]
y1 = temp1["empty_displacement_volume"]
plt.plot(x1, y1, "d", markersize=10)

### r2 value with outliers
slope, intercept, r_value, _, _ = linregress(x1, y1)
plt.text(100, 5200, s="R value: {} (outliers included)".format(int(r_value*1000)/1000), fontsize=14)

### removing outliers
temp2 = temp1[temp1.empty_displacement_volume < 5000]
x1 = temp2["full_displacement_volume"]
y1 = temp2["empty_displacement_volume"]

### Fit line and new r2 value
slope, intercept, r_value, _, _ = linregress(x1, y1)
x2 = np.linspace(min(x1[x1.notnull()]), max(x1[x1.notnull()]), 2)
y2 = [slope * x2[0] + intercept, slope * x2[1] + intercept]
plt.plot(x2, y2)
plt.text(100, 4800, s="R value: {} (outliers removed)".format(int(r_value*1000)/1000), fontsize=14)

del x1, x2, y1, y2, temp1, temp2, slope, intercept, r_value
temp = histopaths[histopaths.empty_displacement_volume > 5000].nst_sample_id.value_counts()
print(f"Number of outliers: {color.BOLD}{sum(temp.values)}{color.END}")
print(f"ID(s) associated with outliers: {color.BOLD}{', '.join(temp.index)}{color.END}")
histopaths[histopaths.empty_displacement_volume > 5000]
temp = histopaths[["length", "wet_weight", "specific_region"]]
temp.columns = ["Length (inches ?)", "Wet Weight (ounces ?)", "Specific region"]

click = alt.selection_multi(fields=['Specific region'])
palette = alt.Scale(domain=['Alaska', 'Great Lakes', 'Delaware Bay to Gulf of Maine',
                           'Florida Gulf Coast', 'Alabama to Brazos River',
                           'Florida Keys to Cape Hatteras', 'Brazos River to Rio Grande'],
                    range=["purple", "#3498db", "#34495e", "#e74c3c", "teal", "#2ecc71", "olive"])

render_alt(
    alt.Chart(temp).mark_square().encode(
        x='Length (inches ?)',
        y='Wet Weight (ounces ?)',
        opacity=alt.value(0.7),
        tooltip=['Specific region', "Length (inches ?)", "Wet Weight (ounces ?)"],
        color=alt.condition(click,
                            'Specific region', alt.value('lightgray'), scale=palette))
    .properties(selection=click, width=700, title="Width and Length relationship")
    .interactive()
)
locs = ("Florida", "Texas", "Louisiana", "Alabama")
histopaths[histopaths.state_name.isin(locs)].fiscal_year.value_counts()
