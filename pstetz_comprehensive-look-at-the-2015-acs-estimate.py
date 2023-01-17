### Data handling imports
import pandas as pd
import numpy as np

### Plotting imports
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
%matplotlib inline

# Advanced plotting... Plotly
from plotly import tools
import plotly.offline as py
from plotly.offline import init_notebook_mode, iplot, plot
import plotly.graph_objs as go
init_notebook_mode(connected=True)

# Statistics imports
import scipy, scipy.stats

# df.head() displays all the columns without truncating
pd.set_option('display.max_columns', None)

import warnings
warnings.filterwarnings('ignore')
# A short hand way to plot most bar graphs
def pretty_bar(data, ax, xlabel=None, ylabel=None, title=None, int_text=False):
    
    # Plots the data
    fig = sns.barplot(data.values, data.index, ax=ax)
    
    # Places text for each value in data
    for i, v in enumerate(data.values):
        
        # Decides whether the text should be rounded or left as floats
        if int_text:
            ax.text(0, i, int(v), color='k', fontsize=14)
        else:
            ax.text(0, i, round(v, 3), color='k', fontsize=14)
     
    ### Labels plot
    ylabel != None and fig.set(ylabel=ylabel)
    xlabel != None and fig.set(xlabel=xlabel)
    title != None and fig.set(title=title)

    
### Used to style Python print statements
class color:
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'
county = pd.read_csv("../input/acs2015_county_data.csv")
tract = pd.read_csv("../input/acs2015_census_tract_data.csv")
before_N = len(tract)
tract = tract.drop(tract[tract.TotalPop == 0].index)
after_N = len(tract)

print("Number of rows removed with zero population: {}{}{}".format(color.BOLD, before_N - after_N, color.END))
del before_N, after_N
print("Shape of county", county.shape)
print("Shape of tract", tract.shape)
print("Columns", county.columns)
county.head()
max_tract = tract.iloc[np.argmax(tract.TotalPop)][["CensusTract", "State", "County"]]
min_tract = tract.iloc[np.argmin(tract.TotalPop)][["CensusTract", "State", "County"]]

print("The most populated Tract is: {}{}, {}{}".format(color.BOLD, max_tract.County, max_tract.State, color.END),
      "with a population of: {}{}{} people".format(color.BOLD, max(tract.TotalPop), color.END))
print("The least populated Tract is: {}{}, {}{} ".format(color.BOLD, min_tract.County, min_tract.State, color.END),
      "with a population of: {}{}{} people".format(color.BOLD, min(tract.TotalPop), color.END))
print("The median number of people sampled in a Tract is: {}{}{}".format(color.BOLD, int(tract.TotalPop.median()), color.END))

### Plotting the different distributions
fig, axarr = plt.subplots(2, 2, figsize=(14, 8))
fig.subplots_adjust(hspace=0.3)
fig.suptitle("Distribution of Tract populations", fontsize=18)

sns.distplot(tract.TotalPop, ax=axarr[0][0]).set(title="KDE Plot")
sns.violinplot(tract.TotalPop, ax=axarr[0][1]).set(title="Violin Plot")
sns.boxplot(tract.TotalPop, ax=axarr[1][0]).set(title="Box Plot")
sorted_data = tract.TotalPop.sort_values().reset_index().drop("index", axis=1)
axarr[1][1].plot(sorted_data, ".")
axarr[1][1].set_title("Tract Populations")
axarr[1][1].set_xlabel("Tract index (after sorting)")
axarr[1][1].set_ylabel("Population")
del sorted_data, min_tract, max_tract
county_pop = county.groupby(["State", "County"]).TotalPop.sum()
print("The most populated County is: {}{}{}".format(color.BOLD, ", ".join(np.argmax(county_pop)[::-1]), color.END),
      "with a population of: {}{}{} people".format(color.BOLD, max(county_pop), color.END))
print("The least populated County is: {}{}{}".format(color.BOLD, ", ".join(np.argmin(county_pop)[::-1]), color.END),
      "with a population of: {}{}{} people".format(color.BOLD, min(county_pop), color.END))
print("The median number of people living in a County is: {}{}{}".format(color.BOLD, int(county_pop.median()), color.END))


### Plotting the different distributions
fig, axarr = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("County overview", fontsize=18)

counties = sorted(county.groupby("State").County.agg(len))
x = np.linspace(1, len(counties), len(counties))
counties = pd.DataFrame({"x":x, "Counties": counties})
(
    sns.regplot(x="x", y="Counties", data=counties, fit_reg=False, ax=axarr[0])
       .set(xlabel="State index (after sorting)", ylabel="Number of counties", title="Number of counties (in each state)")
)

sns.violinplot(county.TotalPop, ax=axarr[1]).set(title="County populations")
del county_pop, counties, x
state_pop = county.groupby("State").TotalPop.sum()

print("The most populated State is: {}{}{}".format(color.BOLD, np.argmax(state_pop), color.END),
      "with a population of: {}{}{} people".format(color.BOLD, max(state_pop), color.END))
print("The least populated State is: {}{}{}".format(color.BOLD, np.argmin(state_pop), color.END),
      "with a population of: {}{}{} people".format(color.BOLD, min(state_pop), color.END))
print("The median number of people living in a State is: {}{}{}".format(color.BOLD, int(state_pop.median()), color.END))

### Plotting the different distributions
fig, axarr = plt.subplots(2, 2, figsize=(14, 8))
fig.subplots_adjust(hspace=0.3)
fig.suptitle("Distribution of State populations", fontsize=18)

sns.distplot(state_pop, ax=axarr[0][0]).set(title="KDE Plot")
sns.violinplot(state_pop, ax=axarr[0][1]).set(title="Violin Plot")
sns.boxplot(state_pop, ax=axarr[1][0]).set(title="Box Plot")

axarr[1][1].plot(state_pop.sort_values().reset_index().drop("State", axis=1), ".")
axarr[1][1].set_title("State Populations")
axarr[1][1].set_xlabel("State index (after sorting)")
axarr[1][1].set_ylabel("Population")
del state_pop
missing_cols = [col for col in county.columns if any(county[col].isnull())]
print(county[missing_cols].isnull().sum())

# Look at rows with missing values
county[county.isnull().any(axis=1)]
missing_cols = [col for col in tract.columns if any(tract[col].isnull())]
print(tract[missing_cols].isnull().sum())

# Look at rows with missing values
tract[tract.isnull().any(axis=1)].head()
tract.sort_values("TotalPop").head(20)
pd.DataFrame({
    "Population": [tract.TotalPop.sum(), county.TotalPop.sum()],
    "Women": [tract.Women.sum(), county.Women.sum()],
    "Men": [tract.Men.sum(), county.Men.sum()],
    "Citizens": [tract.Citizen.sum(), county.Citizen.sum()],
    "States": [len(tract.State.unique()), len(county.State.unique())],
    "Counties": [len(tract.groupby(["State", "County"])), len(county.groupby(["State", "County"]))],
    "Employed": [tract.Employed.sum(), county.Employed.sum()],
}, index=["Tract data", "County data"])
fig, axarr = plt.subplots(3, 1, figsize=(16, 42))
data = county.drop("CensusId", axis=1).corr()

sns.heatmap(data.head(12).transpose(), annot=True, cmap="coolwarm", ax=axarr[0])
sns.heatmap(data.iloc[12:21].transpose(), annot=True, cmap="coolwarm", ax=axarr[1])
sns.heatmap(data.tail(13).transpose(), annot=True, cmap="coolwarm", ax=axarr[2])
del data
dup_counties = (county
 .groupby("County")
 .apply(len)
 .sort_values(ascending=False)
)
dup_counties.where(dup_counties > 1).dropna()
##### County Plots

fig, axarr = plt.subplots(1, 2, figsize=(16,6))
fig.subplots_adjust(wspace=0.3)
fig.suptitle("Population extremes in Counties", fontsize=18)

county_pop = county.groupby(["State", "County"]).TotalPop.median().sort_values(ascending=False)

pretty_bar(county_pop.head(10), axarr[0], title="Most populated Counties")
pretty_bar(county_pop.tail(10), axarr[1], title="Least populated Counties")
plt.show()

##### State Plots

fig, axarr = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle("Total population in all 52 states", fontsize=18)

state_pops = county.groupby("State")["TotalPop"].sum().sort_values(ascending=False)

pretty_bar(state_pops.head(13), axarr[0][0], title="Largest population")
pretty_bar(state_pops.iloc[13:26], axarr[0][1], title="2nd Largest population", ylabel="")
pretty_bar(state_pops.iloc[26:39], axarr[1][0], title="2nd Smallest population")
pretty_bar(state_pops.tail(13), axarr[1][1], title="Smallest population", ylabel="")
del county_pop, state_pops
transportations = ['Drive', 'Carpool', 'Transit', 'Walk', 'OtherTransp']

datas = []
for tran in transportations:
    datas.append(county.groupby(["State", "County"])[tran].median().sort_values(ascending=False).head(10))

traces = []

for data in datas:
    traces.append(go.Box(
                            x=data.index,
                            y=data.values, 
                            showlegend=False
                        ))
buttons = []

for i, tran in enumerate(transportations):
    visibility = [i==j for j in range(len(transportations))]
    button = dict(
                 label =  tran,
                 method = 'update',
                 args = [{'visible': visibility},
                     {'title': 'Top counties for {}'.format(tran)}])
    buttons.append(button)

updatemenus = list([
    dict(active=-1,
         x=-0.15,
         buttons=buttons
    )
])

layout = dict(title='Counties with most popular transportation methods', 
              showlegend=False,
              updatemenus=updatemenus)

fig = dict(data=traces, layout=layout)

iplot(fig, filename='dropdown')
transportations = ['Drive', 'Carpool', 'Transit', 'Walk', 'OtherTransp']

datas = []
for tran in transportations:
    county["trans"] = county.TotalPop * county[tran]
    data = county.groupby("State")["trans"].sum() / county.groupby("State")["TotalPop"].sum()
    datas.append(data.sort_values(ascending=False))

### Create individual figures
fig = tools.make_subplots(rows=2, cols=2, subplot_titles=('1st Quartile', '2nd Quartile',
                                                          '3rd Quartile', '4th Quartile'))

for i in range(4):
    for data in datas:
        start_i = 13 * i
        end_i   = start_i + 13
        
        trace = go.Bar(
                        x=data.iloc[start_i: end_i].index,
                        y=data.iloc[start_i: end_i].values, 
                        showlegend=False
                    )
        
        row_num = 1 + (i // 2)
        col_num = 1 + (i % 2)
        fig.append_trace(trace, row_num, col_num)

### Create buttons for drop down menu
buttons = []
for i, label in enumerate(transportations):
    visibility = [i==j for j in range(len(transportations))]
    button = dict(
                 label =  label,
                 method = 'update',
                 args = [{'visible': visibility},
                     {'title': label}])
    buttons.append(button)

updatemenus = list([
    dict(active=-1,
         x=-0.15,
         buttons=buttons
    )
])

fig['layout']['title'] = 'Transportation across the states'
fig['layout']['showlegend'] = False
fig['layout']['updatemenus'] = updatemenus
fig['layout'].update(height=800, width=1000)

# Remove created column
county = county.drop("trans", axis=1)
del transportations

iplot(fig, filename='dropdown')
fig, axarr = plt.subplots(1, 2, figsize=(16, 8))
fig.subplots_adjust(wspace=0.8)

commute = county.groupby(["State", "County"])["MeanCommute"].median().sort_values(ascending=False)

pretty_bar(commute.head(20), axarr[0], title="Greatest commute times")
pretty_bar(commute.tail(20), axarr[1], title="Lowest commute times")
del commute
##### County Plots

fig, axarr = plt.subplots(1, 2, figsize=(18,8))
fig.subplots_adjust(hspace=0.8)
fig.suptitle("Unemployment extremes in Counties", fontsize=18)

unemployment = county.groupby(["State", "County"])["Unemployment"].median().sort_values(ascending=False)

pretty_bar(unemployment.head(12), axarr[0], title="Highest Unemployment")
pretty_bar(unemployment.tail(12), axarr[1], title="Lowest Unemployment")
plt.show()

##### State Plots

fig, axarr = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle("Unemployment percentage in all 52 states", fontsize=18)

county["Tot_Unemployment"] = county.Unemployment * county.TotalPop
unemployment = county.groupby("State").Tot_Unemployment.sum() / county.groupby("State").TotalPop.sum()
unemployment = unemployment.sort_values(ascending=False)

pretty_bar(unemployment.head(13), axarr[0][0], title="1st Quartile")
pretty_bar(unemployment.iloc[13:26], axarr[0][1], title="2nd Quartile", ylabel="")
pretty_bar(unemployment.iloc[26:39], axarr[1][0], title="3rd Quartile")
pretty_bar(unemployment.tail(13), axarr[1][1], title="4th Quartile", ylabel="")

# Remove created column
county = county.drop("Tot_Unemployment", axis=1)
del unemployment
fig, axarr = plt.subplots(2, 2, figsize=(14,12))
fig.subplots_adjust(wspace=0.5)

county_income_per_cap = county.groupby(["State", "County"])["IncomePerCap"].median().sort_values(ascending=False)
county_income = county.groupby(["State", "County"])["Income"].median().sort_values(ascending=False)

pretty_bar(county_income_per_cap.head(10), axarr[0][0], title="Richest IncomePerCap Counties")
pretty_bar(county_income_per_cap.tail(10), axarr[0][1], title="Poorest IncomePerCap Counties", ylabel="")

pretty_bar(county_income.head(10), axarr[1][0], title="Richest Income Counties")
pretty_bar(county_income.tail(10), axarr[1][1], title="Poorest Income Counties", ylabel="")
del county_income, county_income_per_cap
fig, axarr = plt.subplots(2, 2, figsize=(14,12))
fig.subplots_adjust(wspace=0.5)

poverty = county.groupby(["State", "County"])["Poverty"].median().sort_values(ascending=False)
child_poverty = county.groupby(["State", "County"])["ChildPoverty"].median().sort_values(ascending=False)

pretty_bar(poverty.head(10), axarr[0][0], title="Highest in Poverty")
pretty_bar(poverty.tail(10), axarr[0][1], title="Lowest in Poverty", ylabel="")

pretty_bar(child_poverty.head(10), axarr[1][0], title="Highest in Child Poverty")
pretty_bar(child_poverty.tail(10), axarr[1][1], title="Lowest in Child Poverty", ylabel="")
del poverty, child_poverty
sectors = ['PrivateWork', 'PublicWork', 'SelfEmployed', 'FamilyWork']

fig = tools.make_subplots(rows=1, cols=2, subplot_titles=('Highest', 'Lowest',))

datas = []
for sector in sectors:
    data = county.groupby(["State", "County"])[sector].median().sort_values(ascending=False)
    data = data.reset_index()
    data["Place"] = data["County"] + ", " + data["State"]
    data = data.rename(columns={sector: "Values"})
    datas.append(data)
    
for data in datas:
    trace = go.Bar(
                    x=data.head(10).Place,
                    y=data.head(10).Values, 
                    showlegend=False
                )
    fig.append_trace(trace, 1, 1)

for data in datas:
    trace = go.Bar(
                    x=data.tail(10).Place,
                    y=data.tail(10).Values, 
                    showlegend=False
                )
    fig.append_trace(trace, 1, 2)

### Create buttons for drop down menu
buttons = []
for i, label in enumerate(sectors):
    visibility = [i==j for j in range(len(sectors))]
    button = dict(
                 label =  label,
                 method = 'update',
                 args = [{'visible': visibility},
                     {'title': label}])
    buttons.append(button)

### Create menu
updatemenus = list([
    dict(active=-1,
         x=-0.15,
         buttons=buttons
    )
])

### Final figure edits
fig['layout']['title'] = 'Sectors'
fig['layout']['showlegend'] = False
fig['layout']['updatemenus'] = updatemenus

iplot(fig, filename='dropdown')
careers = ['Professional', 'Service', 'Office', 'Construction', 'Production']

fig = tools.make_subplots(rows=1, cols=2, subplot_titles=('Highest', 'Lowest',))

datas = []
for career in careers:
    data = county.groupby(["State", "County"])[career].median().sort_values(ascending=False)
    data = data.reset_index()
    data["Place"] = data["County"] + ", " + data["State"]
    data = data.rename(columns={career: "Values"})
    datas.append(data)
    
for data in datas:
    trace = go.Bar(
                    x=data.head(10).Place,
                    y=data.head(10).Values, 
                    showlegend=False
                )
    fig.append_trace(trace, 1, 1)

for data in datas:
    trace = go.Bar(
                    x=data.tail(10).Place,
                    y=data.tail(10).Values, 
                    showlegend=False
                )
    fig.append_trace(trace, 1, 2)

### Create buttons for drop down menu
buttons = []
for i, label in enumerate(careers):
    visibility = [i==j for j in range(len(careers))]
    button = dict(
                 label =  label,
                 method = 'update',
                 args = [{'visible': visibility},
                     {'title': label}])
    buttons.append(button)

### Create menu
updatemenus = list([
    dict(active=-1,
         x=-0.15,
         buttons=buttons
    )
])

### Final figure edits
fig['layout']['title'] = 'Careers'
fig['layout']['showlegend'] = False
fig['layout']['updatemenus'] = updatemenus

iplot(fig, filename='dropdown')
################ Setup ################

#### Create new column: total population for each race

races = ['Hispanic', 'White', 'Black', 'Native', 'Asian', 'Pacific']

for race in races:
    county[race + "_pop"] = (county[race] * county.TotalPop) / 100
races = ['Hispanic', 'White', 'Black', 'Native', 'Asian', 'Pacific']

### Create individual figures
fig = tools.make_subplots(rows=2, cols=2, subplot_titles=('Highest Population',     'Lowest Population',
                                                          'Highest Representation', 'Lowest Representation'))

###################################### Population ######################################

datas = []
for race in races:
    data = county.groupby(["State", "County"])[race + "_pop"].sum().map(int).sort_values(ascending=False)
    data = data.reset_index()
    data["Place"] = data["County"] + ", " + data["State"]
    data = data.rename(columns={race + "_pop": "Values"})
    datas.append(data)
    

for data in datas:
    trace = go.Bar(
                    x=data.head(10).Place,
                    y=data.head(10).Values, 
                    showlegend=False
                )
    fig.append_trace(trace, 1, 1)
    
for data in datas:
    trace = go.Bar(
                    x=data.tail(10).Place,
                    y=data.tail(10).Values, 
                    showlegend=False
                )
    fig.append_trace(trace, 1, 2)
    
###################################### Representation ######################################

datas = []
for race in races:
    data = county.groupby(["State", "County"])[race].median().sort_values(ascending=False)
    data = data.reset_index()
    data["Place"] = data["County"] + ", " + data["State"]
    data = data.rename(columns={race: "Values"})
    datas.append(data)

for data in datas:
    trace = go.Bar(
                    x=data.head(10).Place,
                    y=data.head(10).Values, 
                    showlegend=False
                )
    fig.append_trace(trace, 2, 1)
    
for data in datas:
    trace = go.Bar(
                    x=data.tail(10).Place,
                    y=data.tail(10).Values, 
                    showlegend=False
                )
    fig.append_trace(trace, 2, 2)

### Create buttons for drop down menu
buttons = []
for i, label in enumerate(races):
    visibility = [i==j for j in range(len(races))]
    button = dict(
                 label =  label,
                 method = 'update',
                 args = [{'visible': visibility},
                     {'title': label}])
    buttons.append(button)

### Create menu
updatemenus = list([
    dict(active=-1,
         x=-0.15,
         buttons=buttons
    )
])

### Final figure edits
fig['layout']['title'] = 'Racial Population in Counties'
fig['layout']['showlegend'] = False
fig['layout']['updatemenus'] = updatemenus
fig['layout'].update(height=800, width=1000)

iplot(fig, filename='dropdown')
races = ['Hispanic', 'White', 'Black', 'Native', 'Asian', 'Pacific']

datas = []
for race in races:
    data = county.groupby("State")[race + "_pop"].sum().sort_values(ascending=False)
    datas.append(data)

### Create individual figures
fig = tools.make_subplots(rows=2, cols=2, subplot_titles=('1st Quartile', '2nd Quartile',
                                                          '3rd Quartile', '4th Quartile'))

for i in range(4):
    for data in datas:
        start_i = 13 * i
        end_i   = start_i + 13
        
        trace = go.Bar(
                        x=data.iloc[start_i: end_i].index,
                        y=data.iloc[start_i: end_i].values, 
                        showlegend=False
                    )
        
        row_num = 1 + (i // 2)
        col_num = 1 + (i % 2)
        fig.append_trace(trace, row_num, col_num)

### Create buttons for drop down menu
buttons = []
for i, label in enumerate(races):
    visibility = [i==j for j in range(len(races))]
    button = dict(
                 label =  label,
                 method = 'update',
                 args = [{'visible': visibility},
                     {'title': label}])
    buttons.append(button)

updatemenus = list([
    dict(active=-1,
         x=-0.15,
         buttons=buttons
    )
])

fig['layout']['title'] = 'Racial Population in all 52 States'
fig['layout']['showlegend'] = False
fig['layout']['updatemenus'] = updatemenus
fig['layout'].update(height=800, width=1000)

iplot(fig, filename='dropdown')
#### Remove created variables

races = ['Hispanic', 'White', 'Black', 'Native', 'Asian', 'Pacific']
county = county.drop([race + "_pop" for race in races], axis=1)

del races, datas, fig, buttons
numeric_cols = ['Poverty', 'Transit', 'IncomePerCap', 'MeanCommute', 'Unemployment', "Carpool"]

sns.pairplot(county[numeric_cols].sample(1000))
del numeric_cols
sns.jointplot(x='Unemployment', y='Poverty', data=county, kind="reg")
_ = sns.jointplot(x='Unemployment', y='Poverty', data=county, kind='kde')
sns.jointplot(x='Poverty', y='Income', data=county, kind="reg")
_ = sns.jointplot(x='Poverty', y='Income', data=county, kind="kde")
sns.jointplot(x='Poverty', y='Carpool', data=county, kind="reg")
_ = sns.jointplot(x='Poverty', y='Carpool', data=county, kind="kde")
sns.jointplot(x='MeanCommute', y='Transit', data=county, kind="reg")
_ = sns.jointplot(x='MeanCommute', y='Transit', data=county, kind="kde")
high = county[county.Income > 80000]
mid  = county[(county.Income < 80000) & (county.Income > 32000)]
low  = county[county.Income < 32000]

print("Number of low income counties: {}{}{}".format(color.BOLD, len(low), color.END),
      "  Number of middle income counties: {}{}{}".format(color.BOLD, len(mid), color.END),
      "  Number of high income counties: {}{}{}".format(color.BOLD, len(high), color.END))
#########################   Income Distribution Plots   #########################

fig, axarr = plt.subplots(1, 2, figsize=(16, 5))

income = county.groupby(["State", "County"])["Income"].median().sort_values().values
axarr[0].plot(income)
axarr[0].set(title="Sorted Incomes", xlabel="County index (after sorting)", ylabel="Income")

(
        county
            .groupby(["State", "County"])["Income"]
            .median()
            .sort_values()
            .plot(kind="kde", ax=axarr[1])
            .set(title="KDE plot of income", xlabel="Income")
)
plt.show()

#########################   Career Type Plots   #########################

works = [ 'Professional', 'Service', 'Office', 'Construction','Production']

pd.DataFrame({
    "Small income (< $32,000)":  low[works].sum(axis=0) / low[works].sum(axis=0).sum(),
    "Mid income":  mid[works].sum(axis=0) / mid[works].sum(axis=0).sum(),
    "High income (> $80,000)": high[works].sum(axis=0) / high[works].sum(axis=0).sum()
}).transpose().sort_index(ascending=False).plot(kind="bar", rot=0, stacked=True, fontsize=14, figsize=(16, 6))

plt.ylabel("Fraction of workers", fontsize=14)
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), fontsize=12)
plt.title("Career distribution", fontsize=18)
plt.show()

#########################   Career Sector Plots   #########################

works = ['PrivateWork', 'PublicWork', 'SelfEmployed', 'FamilyWork']

pd.DataFrame({
    "Small income (< $32,000)":  low[works].sum(axis=0) / low[works].sum(axis=0).sum(),
    "Mid income":  mid[works].sum(axis=0) / mid[works].sum(axis=0).sum(),
    "High income (> $80,000)": high[works].sum(axis=0) / high[works].sum(axis=0).sum()
}).transpose().sort_index(ascending=False).plot(kind="bar", rot=0, stacked=True, fontsize=14, figsize=(16, 6))

plt.ylabel("Fraction of workers", fontsize=14)
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), fontsize=12)
plt.title("Sector distribution", fontsize=18)
del high, mid, low, income, works
high = county[county.MeanCommute > 32]
mid = county[(county.MeanCommute < 32) & (county.MeanCommute > 15)]
low  = county[county.MeanCommute < 15]
print("Number of short commutes: {}{}{}".format(color.BOLD, len(low), color.END),
      "  Number of average commutes: {}{}{}".format(color.BOLD, len(mid), color.END),
      "  Number of long commutes: {}{}{}".format(color.BOLD, len(high), color.END))
#########################   Commute Distribution Plots   #########################

fig, axarr = plt.subplots(1, 2, figsize=(16, 5))

commute_times = county.groupby(["State", "County"])["MeanCommute"].median().sort_values().values
axarr[0].plot(commute_times)
axarr[0].set(title="Sorted Commute times", xlabel="County index (after sorting)", ylabel="Commute time (min)")

_ = (
        county
            .groupby(["State", "County"])["MeanCommute"]
            .median()
            .sort_values()
            .plot(kind="kde", ax=axarr[1])
            .set(title="KDE plot of commute times", xlabel="Commute time (min)", xlim=(0,60))
)
plt.show()

#########################   Commute Transportation Plots   #########################

trans = ['Drive', 'Carpool', 'Transit', 'Walk', 'OtherTransp', "WorkAtHome"]

pd.DataFrame({
    "Short commutes (< 15min)":  low[trans].sum(axis=0) / low[trans].sum(axis=0).sum(),
    "Medium commutes":  mid[trans].sum(axis=0) / mid[trans].sum(axis=0).sum(),
    "Long commutes (> 32min)": high[trans].sum(axis=0) / high[trans].sum(axis=0).sum()
}).transpose().sort_index(ascending=False).plot(kind="bar", rot=0, stacked=True, fontsize=14, figsize=(16, 6))

plt.ylabel("Fraction of commuters", fontsize=14)
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), fontsize=12)
plt.title("Commute time", fontsize=18)
del high, mid, low, commute_times, trans
longest_county_name_on_census_dataset_index = np.argmax(county.County.map(len))
s_i = np.argmin(county.County.map(len))

county[(county.index == longest_county_name_on_census_dataset_index) | (county.index == s_i)]
max_income_err  = county[county.IncomeErr == max(county.IncomeErr)]
max_income_place = (max_income_err.County + ", " + max_income_err.State).sum()

max_per_cap_err = county[county.IncomePerCapErr == max(county.IncomePerCapErr)]
max_per_cap_place = (max_per_cap_err.County + ", " + max_per_cap_err.State).sum()

print("The County with the biggest income error is: {}{}{}".format(color.BOLD, max_income_place, color.END),
      "with an error of:", color.BOLD, "$" + str(max_income_err.IncomeErr.median()), color.END)
print("The County with the biggest income per cap error is: {}{}{}".format(color.BOLD, max_per_cap_place, color.END),
      "with an error of:", color.BOLD, "$" + str(max_per_cap_err.IncomeErr.median()), color.END)
del max_income_err, max_income_place, max_per_cap_err, max_per_cap_place
county["Men to women"] = county.Men / county.Women
men_to_women = county.groupby(["County", "State"])["Men to women"].median().sort_values(ascending=False)

fig, axarr = plt.subplots(1, 2, figsize=(18,8))
fig.subplots_adjust(wspace=0.3)

pretty_bar(men_to_women.head(10), axarr[0], title="Men to Women")
pretty_bar(men_to_women.tail(10), axarr[1], title="Men to Women")
del men_to_women
################  Configure me!!  ################

state = "California"

##################################################

print("{}{}NOTE{}{}: This is just to help you explore different counties{}"
      .format(color.UNDERLINE, color.BOLD, color.END, color.UNDERLINE, color.END))

county[county.State == state].County.unique()
################  Configure me!!  ################

counties = [("Santa Clara", "California"),   ("San Diego", "California"),
            ("Monterey", "California"),      ("Alameda", "California"),
            ("San Francisco", "California"), ("Contra Costa", "California"),
            ("Los Angeles", "California"),   ("Fresno", "California")]

##################################################
commute, income, income_percap, men, women = ([],[],[],[],[])
hispanic, white, black, native, asian, pacific = ([],[],[],[],[],[])

def total_race(df, race):
    total_pop = df[race] * df.TotalPop
    frac_pop = (total_pop / 100).sum()
    return int(frac_pop)
    
for c, s in counties:
    curr_county = county[(county.County == c) & (county.State == s)]

    commute.append(curr_county.MeanCommute.median())
    men.append(   int(curr_county.Men.median())   )
    women.append( int(curr_county.Women.median()) )
    
    ### NOTE: These demographics are
    hispanic.append( total_race(curr_county, "Hispanic") )
    white.append(    total_race(curr_county, "White")    )
    black.append(    total_race(curr_county, "Black")    )
    native.append(   total_race(curr_county, "Native")   )
    asian.append(    total_race(curr_county, "Asian")    )
    pacific.append(  total_race(curr_county, "Pacific")  )
    income.append(curr_county.Income.median())
    income_percap.append(curr_county.IncomePerCap.median())

counties = pd.DataFrame({
                "Women": women, "Men": men, "Mean Commute": commute,
                "Hispanic": hispanic, "White": white, "Black": black,
                "Native": native, "Asian": asian, "Pacific": pacific,
                "IncomePerCap": income_percap, "Income": income
            }, index=counties)

counties["Men to women"] = counties.Men / counties.Women
del commute, income, income_percap, men, women, hispanic, white, black, native, asian, pacific
counties.head()
plt.figure(figsize=(16, 12))

### Nuanced way of creating subplots
ax1 = plt.subplot2grid((3, 2), (0, 0))
ax2 = plt.subplot2grid((3, 2), (0, 1))
ax3 = plt.subplot2grid((3, 2), (1, 0), colspan=2)
ax4 = plt.subplot2grid((3, 2), (2, 0))
ax5 = plt.subplot2grid((3, 2), (2, 1))

plt.suptitle(", ".join([c for c,s in counties.index]), fontsize=18)

pretty_bar(counties["Mean Commute"], ax1, title="Mean Commute")
pretty_bar(counties["Men to women"], ax2, title="Men to women")

races = ['Hispanic', 'White', 'Black', 'Native', 'Asian', 'Pacific']
counties[races].plot(kind="bar", title="Population make up", stacked=True, ax=ax3, rot=0)

pretty_bar(counties["IncomePerCap"], ax4, title="Income per capita")
pretty_bar(counties["Income"], ax5, title="Income")
del races
################  Configure me!!  ################

selected_county = ("Monterey", "California")

##################################################
# Gets the selected county from the data
selected_county = county[(county.State == selected_county[1]) & (county.County == selected_county[0])]

### Gets the total population and the number of men
n = selected_county.TotalPop.sum()
men = selected_county.Men.median()
women = selected_county.Women.median()

# Calculates the number of standard deivations
distance = abs((n / 2) - men)
sigma = distance / np.sqrt(n / 4)

# Get the probability distribution for a population this size
x = np.linspace(.4*n, .6*n, n+1)
pmf = scipy.stats.binom.pmf(x, n, 0.5)

### Plots the probability distribution and the actual value
plt.figure(figsize=(12, 6))
plt.plot(x, pmf, label="Gender PMF")
plt.axvline(men, color="red", label="Male Actual")
plt.axvline(women, color="k", label="Female Actual")

### Limits the plot to the only interesting sectiton
llim, rlim = n/2 - 1.2*distance, n/2 + 1.2*distance
plt.xlim(llim, rlim)

# Labels the plot
plt.title("{} - Ratio is {} $\sigma$ away".format(selected_county.County.iloc[0], round(sigma, 3)), fontsize=14)
plt.xlabel("Number of people")
plt.ylabel("Probability")
_ = plt.legend(frameon=True, bbox_to_anchor=(1.1, 1.05)).get_frame().set_edgecolor('black')
