# Data Manipulation Packages
import pandas as pd
import numpy as np

# Visualization Packages
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline


# read train data as train_data and 
# test data as test_data
train_data = pd.read_csv("../input/train.csv")
test_data = pd.read_csv("../input/test.csv")

# Check the dimensions of the table
print("The dimension of the train table is: ", train_data.shape)
print("The dimension of the test table is: ", test_data.shape)
train_data.head()
train_data.info()
from plotly.offline import init_notebook_mode, iplot
import plotly.graph_objs as go
import plotly.plotly as py
from plotly import tools
import plotly.figure_factory as ff
import matplotlib.pyplot as plt
import seaborn as sns
import random 
import warnings
import operator
warnings.filterwarnings("ignore")
init_notebook_mode(connected=True)
train_levels = train_data.loc[(train_data['Target'].notnull()) & (train_data['parentesco1'] == 1), ['Target', 'idhogar']]
label_counts = train_levels['Target'].value_counts().sort_index().to_frame()
target = label_counts
levels = ["Extreme Poverty", "Moderate Poverty", "Vulnerable", "Non Vulnerable"]
trace = go.Bar(y=target.Target, x=levels, marker=dict(color=['#FF0000', '#FFA500', '#0000FF', '#008000'], opacity=0.6))
layout = dict(title="Household Poverty Levels", margin=dict(l=200), width=800, height=400)
data = [trace]
fig = go.Figure(data=data, layout=layout)
iplot(fig)
print(label_counts)
train_data.select_dtypes('object').head()
import numpy as np
mapping = {"yes": 1, "no": 0}

# Apply same operation to both train and test
for df in [train_data, test_data]:
    # Fill in the values with the correct mapping
    df['dependency'] = df['dependency'].replace(mapping).astype(np.float64)
    df['edjefa'] = df['edjefa'].replace(mapping).astype(np.float64)
    df['edjefe'] = df['edjefe'].replace(mapping).astype(np.float64)

train_data[['dependency', 'edjefa', 'edjefe']].describe()
train_data.select_dtypes('float').head()
from collections import OrderedDict
# Color mapping
colors = OrderedDict({1: 'red', 2: 'orange', 3: 'blue', 4: 'green'})
poverty_mapping = OrderedDict({1: 'extreme', 2: 'moderate', 3: 'vulnerable', 4: 'non vulnerable'})
plt.figure(figsize = (30, 16))

# Iterate through the float columns
for i, col in enumerate(train_data.select_dtypes('float')):
    ax = plt.subplot(6, 2, i + 1)
    # Iterate through the poverty levels
    for poverty_level, color in colors.items():
        # Plot each poverty level as a separate line
        sns.kdeplot(train_data.loc[train_data['Target'] == poverty_level, col].dropna(), 
                    ax = ax, color = color, label = poverty_mapping[poverty_level])
        
    plt.title(f'{col.capitalize()} Distribution'); 
    plt.xlabel(f'{col}'); plt.ylabel('Density')

plt.subplots_adjust(top = 2)
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings("ignore")

labels = []
values = []
for col in train_data.columns:
    if col not in ["Id", "Target"]:
        labels.append(col)
        values.append(spearmanr(train_data[col].values, train_data["Target"].values)[0])
corr_df = pd.DataFrame({'col_labels':labels, 'corr_values':values})
corr_df = corr_df.sort_values(by='corr_values')
 
plt.figure(figsize=(15,15))
sns.heatmap(train_data[corr_df.col_labels[:10]].corr(), annot=True)
# Groupby the household and figure out the number of unique values
all_equal = train_data.groupby('idhogar')['Target'].apply(lambda x: x.nunique() == 1)

# Households where targets are not all equal
not_equal = all_equal[all_equal != True]
print('There are {} households where the family members do not all have the same target.'.format(len(not_equal)))
train_data[train_data['idhogar'] == not_equal.index[0]][['idhogar', 'parentesco1', 'Target']]
households_leader = train_data.groupby('idhogar')['parentesco1'].sum()

# Find households without a head
households_no_head = train_data.loc[train_data['idhogar'].isin(households_leader[households_leader == 0].index), :]

print('There are {} households without a head.'.format(households_no_head['idhogar'].nunique()))
# Find households without a head and where labels are different
households_no_head_equal = households_no_head.groupby('idhogar')['Target'].apply(lambda x: x.nunique() == 1)
print('{} Households with no head have different labels.'.format(sum(households_no_head_equal == False)))
# Iterate through each household
for household in not_equal.index:
    # Find the correct label (for the head of household)
    true_target = int(train_data[(train_data['idhogar'] == household) & (train_data['parentesco1'] == 1.0)]['Target'])
    
    # Set the correct label for all members in the household
    train_data.loc[train_data['idhogar'] == household, 'Target'] = true_target
    
    
# Groupby the household and figure out the number of unique values
all_equal = train_data.groupby('idhogar')['Target'].apply(lambda x: x.nunique() == 1)

# Households where targets are not all equal
not_equal = all_equal[all_equal != True]
print('There are {} households where the family members do not all have the same target.'.format(len(not_equal)))
# Add null Target column to test
test_data['Target'] = np.nan
add_data = train_data.append(test_data, ignore_index = True)

# Number of missing in each column
missing = pd.DataFrame(add_data.isnull().sum()).rename(columns = {0: 'total'})

# Create a percentage missing
missing['percent'] = missing['total'] / len(add_data)

missing.sort_values('percent', ascending = False).head(10).drop('Target')
def compare_plot(col, title):
    tr1 = train_data[train_data['Target'] == 1][col].value_counts().to_dict()
    tr2 = train_data[train_data['Target'] == 2][col].value_counts().to_dict()
    tr3 = train_data[train_data['Target'] == 3][col].value_counts().to_dict()
    tr4 = train_data[train_data['Target'] == 4][col].value_counts().to_dict()
    
    xx = ['Extereme', 'Moderate', 'Vulnerable', 'NonVulnerable']
    trace1 = go.Bar(y=[tr1[0], tr2[0], tr3[0], tr4[0]], name="Not Present", x=xx, marker=dict(color="orange", opacity=0.6))
    trace2 = go.Bar(y=[tr1[1], tr2[1], tr3[1], tr4[1]], name="Present", x=xx, marker=dict(color="purple", opacity=0.6))
    
    return trace1, trace2 
    
tr1, tr2 = compare_plot("v18q", "Tablet")
tr3, tr4 = compare_plot("refrig", "Refrigerator")
tr5, tr6 = compare_plot("computer", "Computer")
tr7, tr8 = compare_plot("television", "Television")
tr9, tr10 = compare_plot("mobilephone", "MobilePhone")
titles = ["Tablet", "Refrigerator", "Computer", "Television", "MobilePhone"]

fig = tools.make_subplots(rows=3, cols=2, print_grid=False, subplot_titles=titles)
fig.append_trace(tr1, 1, 1)
fig.append_trace(tr2, 1, 1)
fig.append_trace(tr3, 1, 2)
fig.append_trace(tr4, 1, 2)
fig.append_trace(tr5, 2, 1)
fig.append_trace(tr6, 2, 1)
fig.append_trace(tr7, 2, 2)
fig.append_trace(tr8, 2, 2)
fig.append_trace(tr9, 3, 1)
fig.append_trace(tr10, 3, 1)

fig['layout'].update(height=1000, title="What do Households Own", barmode="stack", showlegend=False)
iplot(fig)
heads = train_data.loc[add_data['parentesco1'] == 1].copy()
target = heads['v18q1'].value_counts().to_frame()
levels = ["1.0", "2.0", "3.0", "4.0", "5.0", "6.0"]
trace = go.Bar(y=target['v18q1'], x=levels, marker=dict(color='orange', opacity=0.6))
layout = dict(title="v18q1 Value Counts", margin=dict(l=200), width=800, height=400)
data = [trace]
fig = go.Figure(data=data, layout=layout)
iplot(fig)
heads.groupby('v18q')['v18q1'].apply(lambda x: x.isnull().sum())
add_data['v18q1'] = add_data['v18q1'].fillna(0)
def compare_dists(col, title):
    trace1 = go.Histogram(name="Extereme", x=add_data[add_data['Target']==1][col])
    trace2 = go.Histogram(name="Moderate", x=add_data[add_data['Target']==2][col])
    trace3 = go.Histogram(name="Vulnerable", x=add_data[add_data['Target']==3][col])
    trace4 = go.Histogram(name="NonVulnerable", x=add_data[add_data['Target']==4][col])

    fig = tools.make_subplots(rows=2, cols=2, print_grid=False)
    fig.append_trace(trace1, 1, 1)
    fig.append_trace(trace2, 1, 2)
    fig.append_trace(trace3, 2, 1)
    fig.append_trace(trace4, 2, 2)

    fig['layout'].update(height=400, showlegend=False, title=title)
    iplot(fig)

compare_dists('v2a1', "Monthy Rent for four groups of houses")
own_variables = [x for x in add_data if x.startswith('tipo')]
target = add_data.loc[add_data['v2a1'].isnull(), own_variables].sum().to_frame()
levels = ["Owns and Paid Off", "Owns and Paying", "Rented", "Precarious", "Other"]
trace = go.Bar(y=target[0], x=levels, marker=dict(color='orange', opacity=0.6))
layout = dict(title="Home Ownership Status for Household Missing Rent Payments", margin=dict(l=200), width=800, height=400)
data = [trace]
fig = go.Figure(data=data, layout=layout)
iplot(fig)


# tipovivi1, =1 own and fully paid house
# tipovivi2, "=1 own,  paying in installments"
# tipovivi3, =1 rented
# tipovivi4, =1 precarious
# tipovivi5, "=1 other(assigned,  borrowed)"
# Fill in households that own the house with 0 rent payment
add_data.loc[(add_data['tipovivi1'] == 1), 'v2a1'] = 0

# Create missing rent payment column
add_data['v2a1-missing'] = add_data['v2a1'].isnull()

add_data['v2a1-missing'].value_counts()
# This variable is only collected for people between 7 and 19 years of age 
# and it is the difference between the years of education a person should have 
# and the years of education he/she has. it is capped at 5.
print(add_data['rez_esc'].isnull().value_counts())

add_data.loc[add_data['rez_esc'].notnull()]['age'].describe()
def find_prominent(row, mats):
    for c in mats:
        if row[c] == 1:
            return c
    return 

def combine2(starter, colname, title, replacemap, plotme = True):
    mats = [c for c in add_data.columns if c.startswith(starter)]
    add_data[colname] = add_data.apply(lambda row : find_prominent(row, mats), axis=1)
    add_data[colname] = add_data[colname].apply(lambda x : replacemap[x] if x != None else x )

    om1 = add_data[add_data['Target'] == 1][colname].value_counts().to_frame()
    om2 = add_data[add_data['Target'] == 2][colname].value_counts().to_frame()
    om3 = add_data[add_data['Target'] == 3][colname].value_counts().to_frame()
    om4 = add_data[add_data['Target'] == 4][colname].value_counts().to_frame()

    trace1 = go.Bar(y=om1[colname], x=om1.index, name="Extereme", marker=dict(color='red', opacity=0.9))
    trace2 = go.Bar(y=om2[colname], x=om2.index, name="Moderate", marker=dict(color='red', opacity=0.5))
    trace3 = go.Bar(y=om3[colname], x=om3.index, name="Vulnerable", marker=dict(color='orange', opacity=0.9))
    trace4 = go.Bar(y=om4[colname], x=om4.index, name="NonVulnerable", marker=dict(color='orange', opacity=0.5))

    data = [trace1, trace2, trace3, trace4]
    layout = dict(title=title, legend=dict(y=1.1, orientation="h"), barmode="stack", margin=dict(l=50), height=400)
    fig = go.Figure(data=data, layout=layout)
    if plotme:
        iplot(fig)
        
flr = {"instlevel1": "No Education", "instlevel2": "Incomplete Primary", "instlevel3": "Complete Primary", 
       "instlevel4": "Incomplete Sc.", "instlevel5": "Complete Sc.", "instlevel6": "Incomplete Tech Sc.",
       "instlevel7": "Complete Tech Sc.", "instlevel8": "Undergraduation", "instlevel9": "Postgraduation"}
combine2("instl", "education_details", "Education Details of Family Members", flr) 
# If individual is over 19 or younger than 7 and missing years behind, set it to 0
add_data.loc[((add_data['age'] > 19) | (add_data['age'] < 7)) & (add_data['rez_esc'].isnull()), 'rez_esc'] = 0

# Add a flag for those between 7 and 19 with a missing value
add_data['rez_esc-missing'] = add_data['rez_esc'].isnull()
add_data.loc[add_data['rez_esc'] > 5, 'rez_esc'] = 5
data = add_data[['rez_esc', 'Target']].head(5)
print(data)
def plot_categoricals(x, y, data, annotate = True):
    """Plot counts of two categoricals.
    Size is raw count for each grouping.
    Percentages are for a given value of y."""
    
    # Raw counts 
    raw_counts = pd.DataFrame(data.groupby(y)[x].value_counts(normalize = False))
    raw_counts = raw_counts.rename(columns = {x: 'raw_count'})
    
    # Calculate counts for each group of x and y
    counts = pd.DataFrame(data.groupby(y)[x].value_counts(normalize = True))
    
    # Rename the column and reset the index
    counts = counts.rename(columns = {x: 'normalized_count'}).reset_index()
    counts['percent'] = 100 * counts['normalized_count']
    
    # Add the raw count
    counts['raw_count'] = list(raw_counts['raw_count'])
    
    plt.figure(figsize = (14, 10))
    # Scatter plot sized by percent
    plt.scatter(counts[x], counts[y], edgecolor = 'k', color = 'lightgreen',
                s = 100 * np.sqrt(counts['raw_count']), marker = 'o',
                alpha = 0.6, linewidth = 1.5)
    
    if annotate:
        # Annotate the plot with text
        for i, row in counts.iterrows():
            # Put text with appropriate offsets
            plt.annotate(xy = (row[x] - (1 / counts[x].nunique()), 
                               row[y] - (0.15 / counts[y].nunique())),
                         color = 'navy',
                         s = f"{round(row['percent'], 1)}%")
        
    # Set tick marks
    plt.yticks(counts[y].unique())
    plt.xticks(counts[x].unique())
    
    # Transform min and max to evenly space in square root domain
    sqr_min = int(np.sqrt(raw_counts['raw_count'].min()))
    sqr_max = int(np.sqrt(raw_counts['raw_count'].max()))
    
    # 5 sizes for legend
    msizes = list(range(sqr_min, sqr_max,
                        int(( sqr_max - sqr_min) / 5)))
    markers = []
    
    # Markers for legend
    for size in msizes:
        markers.append(plt.scatter([], [], s = 100 * size, 
                                   label = f'{int(round(np.square(size) / 100) * 100)}', 
                                   color = 'lightgreen',
                                   alpha = 0.6, edgecolor = 'k', linewidth = 1.5))
        
    # Legend and formatting
    plt.legend(handles = markers, title = 'Counts',
               labelspacing = 3, handletextpad = 2,
               fontsize = 16,
               loc = (1.10, 0.19))
    
    plt.annotate(f'* Size represents raw count while % is for a given y value.',
                 xy = (0, 1), xycoords = 'figure points', size = 10)
    
    # Adjust axes limits
    plt.xlim((counts[x].min() - (6 / counts[x].nunique()), 
              counts[x].max() + (6 / counts[x].nunique())))
    plt.ylim((counts[y].min() - (4 / counts[y].nunique()), 
              counts[y].max() + (4 / counts[y].nunique())))
    plt.grid(None)
    plt.xlabel(f"{x}"); plt.ylabel(f"{y}"); plt.title(f"{y} vs {x}");
plot_categoricals('rez_esc', 'Target', add_data);
plot_categoricals('escolari', 'Target', add_data, annotate = False)
heads = add_data[(add_data['rez_esc-missing'] == 1)].copy()
target = heads['Target'].value_counts().to_frame()
print(target)
 #target = heads_target.value_counts().to_frame()
levels = ["4.0", "3.0", "2.0", "1.0"]
trace = go.Bar(y=target['Target'], x=levels, marker=dict(color='orange', opacity=0.6))
layout = dict(title="Target Value Counts", margin=dict(l=200), width=800, height=500)
data = [trace]
fig = go.Figure(data=data, layout=layout)
iplot(fig)
heads = add_data[(add_data['v2a1-missing'] == 1)].copy()
target = heads['Target'].value_counts().to_frame()
print(target)
 #target = heads_target.value_counts().to_frame()
levels = ["4.0", "3.0", "2.0", "1.0"]
trace = go.Bar(y=target['Target'], x=levels, marker=dict(color='orange', opacity=0.6))
layout = dict(title="Target Value Counts", margin=dict(l=200), width=800, height=500)
data = [trace]
fig = go.Figure(data=data, layout=layout)
iplot(fig)
# Education Details, Status and Members
def combine2(starter, colname, title, replacemap, plotme = True):
    mats = [c for c in add_data.columns if c.startswith(starter)]
    add_data[colname] = add_data.apply(lambda row : find_prominent(row, mats), axis=1)
    add_data[colname] = add_data[colname].apply(lambda x : replacemap[x] if x != None else x )

    om1 = add_data[add_data['Target'] == 1][colname].value_counts().to_frame()
    om2 = add_data[add_data['Target'] == 2][colname].value_counts().to_frame()
    om3 = add_data[add_data['Target'] == 3][colname].value_counts().to_frame()
    om4 = add_data[add_data['Target'] == 4][colname].value_counts().to_frame()

    trace1 = go.Bar(y=om1[colname], x=om1.index, name="Extreme", marker=dict(color='red', opacity=0.9))
    trace2 = go.Bar(y=om2[colname], x=om2.index, name="Moderate", marker=dict(color='red', opacity=0.5))
    trace3 = go.Bar(y=om3[colname], x=om3.index, name="Vulnerable", marker=dict(color='orange', opacity=0.9))
    trace4 = go.Bar(y=om4[colname], x=om4.index, name="NonVulnerable", marker=dict(color='orange', opacity=0.5))

    data = [trace1, trace2, trace3, trace4]
    layout = dict(title=title, legend=dict(y=1.1, orientation="h"), barmode="stack", margin=dict(l=50), height=400)
    fig = go.Figure(data=data, layout=layout)
    if plotme:
        iplot(fig)


flr = {"instlevel1": "No Education", "instlevel2": "Incomplete Primary", "instlevel3": "Complete Primary", 
       "instlevel4": "Incomplete Sc.", "instlevel5": "Complete Sc.", "instlevel6": "Incomplete Tech Sc.",
       "instlevel7": "Complete Tech Sc.", "instlevel8": "Undergraduation", "instlevel9": "Postgraduation"}
combine2("instl", "education_details", "Education Details of Family Members", flr)  

flr = {"estadocivil1": "< 10 years", "estadocivil2": "Free / Coupled union", "estadocivil3": "Married", 
       "estadocivil4": "Divorced", "estadocivil5": "Separated", "estadocivil6": "Widow",
       "estadocivil7": "Single"}
combine2("estado", "status_members", "Status of Family Members", flr)  

flr = {"parentesco1": "Household Head", "parentesco2": "Spouse/Partner", "parentesco3": "Son/Daughter", 
       "parentesco4": "Stepson/Daughter", "parentesco5" : "Son/Daughter in Law" , "parentesco6": "Grandson/Daughter", 
       "parentesco7": "Mother/Father", "parentesco8": "Mother/Father in Law", "parentesco9" : "Brother/Sister" , 
       "parentesco10" : "Brother/Sister in law", "parentesco11" : "Other Family Member", "parentesco12" : "Other Non Family Member"}
combine2("parentesc", "family_members", "Family Members in the Households", flr)  

flr = {"lugar1": "Central", "lugar2": "Chorotega", "lugar3": "PacÃƒÂ­fico central", 
       "lugar4": "Brunca", "lugar5": "Huetar AtlÃƒÂ¡ntica", "lugar6": "Huetar Norte"}
combine2("lugar", "region", "Region of the Households", flr, plotme=False) 
# Gender and Age Distributions

def agbr(col):
    temp1 = train_data[add_data['Target'] == 1][col].value_counts()
    trace1 = go.Bar(x=temp1.index, y=temp1.values, marker=dict(color="red", opacity=0.89), name="Extreme")

    temp2 = train_data[add_data['Target'] == 2][col].value_counts()
    trace2 = go.Bar(x=temp2.index, y=temp2.values, marker=dict(color="orange", opacity=0.79), name="Moderate")

    temp3 = train_data[add_data['Target'] == 3][col].value_counts()
    trace3 = go.Bar(x=temp3.index, y=temp3.values, marker=dict(color="purple", opacity=0.89), name="Vulnerable")

    temp4 = train_data[add_data['Target'] == 4][col].value_counts()
    trace4 = go.Bar(x=temp4.index, y=temp4.values, marker=dict(color="green", opacity=0.79), name="NonVulnerable")
    
    return [trace1, trace2, trace3, trace4]
    layout = dict(height=400)
    fig = go.Figure(data=[trace1, trace2, trace3, trace4], layout=layout)
    iplot(fig)

titles = ["Total Persons", "< 12 Yrs", ">= 12 Yrs", "Total Males", "Males < 12 Yrs", "Males >= 12 Yrs", 
         "Total Females", "Females < 12 Yrs", "Females >= 12 Yrs"]
fig = tools.make_subplots(rows=3, cols=3, print_grid=False, subplot_titles=titles)

res = agbr('r4t1')
for x in res:
    fig.append_trace(x, 1, 1)
res = agbr('r4t2')
for x in res:
    fig.append_trace(x, 1, 2)
res = agbr('r4t3')
for x in res:
    fig.append_trace(x, 1, 3)

res = agbr('r4h1')
for x in res:
    fig.append_trace(x, 2, 1)
res = agbr('r4h2')
for x in res:
    fig.append_trace(x, 2, 2)
res = agbr('r4h3')
for x in res:
    fig.append_trace(x, 2, 3)

res = agbr('r4m1')
for x in res:
    fig.append_trace(x, 3, 1)
res = agbr('r4m2')
for x in res:
    fig.append_trace(x, 3, 2)
res = agbr('r4m3')
for x in res:
    fig.append_trace(x, 3, 3)

    
fig['layout'].update(height=750, showlegend=False, title="Gender and Age Distributions")
iplot(fig)
# Age groups among the household
titles = ["Children", "Adults", "65+ Old"]
fig = tools.make_subplots(rows=1, cols=3, print_grid=False, subplot_titles=titles)

res = agbr("hogar_nin")
for x in res:
    fig.append_trace(x, 1, 1)
res = agbr("hogar_adul")
for x in res:
    fig.append_trace(x, 1, 2)
res = agbr("hogar_mayor")
for x in res:
    fig.append_trace(x, 1, 3)

fig['layout'].update(height=350, title="People Distribution in Households", barmode="stack", showlegend=False)
iplot(fig)
# Age groups among the households
titles = ["Children", "Adults", "65+ Old"]
fig = tools.make_subplots(rows=1, cols=3, print_grid=False, subplot_titles=titles)

res = agbr("hogar_nin")
for x in res:
    fig.append_trace(x, 1, 1)
res = agbr("hogar_adul")
for x in res:
    fig.append_trace(x, 1, 2)
res = agbr("hogar_mayor")
for x in res:
    fig.append_trace(x, 1, 3)

fig['layout'].update(height=350, title="People Distribution in Households", barmode="stack", showlegend=False)
iplot(fig)

# Household size
tm = agbr('tamhog')
layout = dict(title="Household People Size", margin=dict(l=100), height=400, legend=dict(orientation="h", y=1))
fig = go.Figure(data=tm, layout=layout)
iplot(fig)
# Poverty Levels with respect to Monthly Rent and Age of the House
trace0 = go.Scatter(x=train_data['v2a1'], y=train_data['age'], name="Extereme", 
                    mode='markers', marker=dict(color=train_data['Target'], opacity=1, size=16 - train_data['Target']**2))
layout = go.Layout(xaxis=dict(title="Monthly Rent of the house", range=(0,400000)), yaxis=dict(title="Age of the House"))
fig = go.Figure(data =[trace0], layout=layout)
iplot(fig)
# Area/Location Details
# Area Type with Respect to Poverty Levels
train_data['area_type'] = train_data['area1'].apply(lambda x: "urbal" if x==1 else "rural")

cols = ['area_type', 'Target']
colmap = sns.light_palette("yellow", as_cmap=True)
pd.crosstab(add_data[cols[1]], train_data[cols[0]]).style.background_gradient(cmap = colmap)
# Region with respect to Poverty Levels

cols = ['region', 'Target']
colmap = sns.light_palette("orange", as_cmap=True)
pd.crosstab(add_data[cols[0]], add_data[cols[1]]).style.background_gradient(cmap = colmap)
id_var = ['Id', 'idhogar', 'Target']
ind_bool = ['v18q', 'dis', 'male', 'female', 'estadocivil1', 'estadocivil2', 'estadocivil3', 
            'estadocivil4', 'estadocivil5', 'estadocivil6', 'estadocivil7', 
            'parentesco1', 'parentesco2',  'parentesco3', 'parentesco4', 'parentesco5', 
            'parentesco6', 'parentesco7', 'parentesco8',  'parentesco9', 'parentesco10', 
            'parentesco11', 'parentesco12', 'instlevel1', 'instlevel2', 'instlevel3', 
            'instlevel4', 'instlevel5', 'instlevel6', 'instlevel7', 'instlevel8', 
            'instlevel9', 'mobilephone', 'rez_esc-missing']

ind_ordered = ['rez_esc', 'escolari', 'age']
hh_bool = ['hacdor', 'hacapo', 'v14a', 'refrig', 'paredblolad', 'paredzocalo', 
           'paredpreb','pisocemento', 'pareddes', 'paredmad',
           'paredzinc', 'paredfibras', 'paredother', 'pisomoscer', 'pisoother', 
           'pisonatur', 'pisonotiene', 'pisomadera',
           'techozinc', 'techoentrepiso', 'techocane', 'techootro', 'cielorazo', 
           'abastaguadentro', 'abastaguafuera', 'abastaguano',
            'public', 'planpri', 'noelec', 'coopele', 'sanitario1', 
           'sanitario2', 'sanitario3', 'sanitario5',   'sanitario6',
           'energcocinar1', 'energcocinar2', 'energcocinar3', 'energcocinar4', 
           'elimbasu1', 'elimbasu2', 'elimbasu3', 'elimbasu4', 
           'elimbasu5', 'elimbasu6', 'epared1', 'epared2', 'epared3',
           'etecho1', 'etecho2', 'etecho3', 'eviv1', 'eviv2', 'eviv3', 
           'tipovivi1', 'tipovivi2', 'tipovivi3', 'tipovivi4', 'tipovivi5', 
           'computer', 'television', 'lugar1', 'lugar2', 'lugar3',
           'lugar4', 'lugar5', 'lugar6', 'area1', 'area2', 'v2a1-missing']

hh_ordered = [ 'rooms', 'r4h1', 'r4h2', 'r4h3', 'r4m1','r4m2','r4m3', 'r4t1',  'r4t2', 
              'r4t3', 'v18q1', 'tamhog','tamviv','hhsize','hogar_nin',
              'hogar_adul','hogar_mayor','hogar_total',  'bedrooms', 'qmobilephone']

hh_cont = ['v2a1', 'dependency', 'edjefe', 'edjefa', 'meaneduc', 'overcrowding']
sqr_ = ['SQBescolari', 'SQBage', 'SQBhogar_total', 'SQBedjefe', 
        'SQBhogar_nin', 'SQBovercrowding', 'SQBdependency', 'SQBmeaned', 'agesq']
x = ind_bool + ind_ordered + ind_bool + hh_bool + hh_ordered + hh_cont + sqr_

from collections import Counter

print('There are no repeats: ', np.all(np.array(list(Counter(x).values())) == 1))
print('We covered every variable: ', len(x) == add_data.shape[1])
sns.lmplot('age', 'SQBage', data = add_data, fit_reg=False);
plt.title('Squared Age versus Age');
# Creating trace1
trace1 = go.Scatter(
                    x = add_data['age'],
                    y = add_data['SQBage'],
                    mode = "markers")

data = [trace1]
layout = dict(title = 'Squared Age versus Age',
              xaxis= dict(title= 'World Rank',ticklen= 5,zeroline= False)
             )
fig = dict(data = data, layout = layout)
iplot(fig)
# Remove squared variables
add_data = add_data.drop(columns = sqr_)
add_data.shape
print(add_data.columns)
from collections import Counter

print('There are no repeats: ', np.all(np.array(list(Counter(x).values())) == 1))
print('We covered every variable: ', len(x) == add_data.shape[1])
