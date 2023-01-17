import pandas
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
df = pandas.read_csv("../input/march18_myspeed.csv")
# set na to None
df.loc[df['Signal_strength'] == 'na', 'Signal_strength'] = None
df.loc[df.isnull()['Signal_strength']]

# convert 'Signal_strength' to float
df['Signal_strength'] = pandas.to_numeric(df.loc[:,'Signal_strength'])
print(df.head())
print(df.info())
print(df.describe())
df.isnull().sum()
columns = ['Technology', 'Test_type', 'Service Provider', 'LSA']

for c in columns:
    v = df[c].unique()
    g = df.groupby(by=c)[c].count().sort_values(ascending=True)
    r = np.arange(len(v))
    print(g.head())
    plt.figure(figsize = (6, len(v)/2 +1))
    plt.barh(y = r, width = g.head(len(v)))
    total = sum(g.head(len(v)))
    print(total)
    for (i, u) in enumerate(g.head(len(v))):
        plt.text(x = u + 0.2, y = i - 0.08, s = str(round(u/total*100, 2))+'%', color = 'blue', fontweight = 'bold')
    plt.margins(x = 0.2)
    plt.yticks(r, g.index)
    plt.show()    
def sel(df, column_name, value):
    data = df.loc[(df[column_name] == value)]
    return data

pandas.DataFrame.mask = sel
def plot_graphs(provider, state):
    # plot distributions of speeds
    #columns = ['Technology', 'Test_type', 'Service Provider', 'LSA']
    data4g = df.mask('Test_type', 'Download').mask('Technology', '4G')
    data3g = df.mask('Test_type', 'Download').mask('Technology', '3G')
    
    if provider != 'All':
        data4g = data4g.mask('Service Provider', provider)
        data3g = data3g.mask('Service Provider', provider)
    
    if state != 'All':
        data4g = data4g.mask('LSA', state)
        data3g = data3g.mask('LSA', state)
        
    x1 = data4g['Data Speed(Mbps)']
    x2 = data3g['Data Speed(Mbps)']

    fig, axes = plt.subplots(nrows = 1, ncols = 2, figsize = (9, 5))
    #print(x1)
    axes[0].clear()
    axes[0].hist(x1, bins=100, label = '4G', normed = True)
    axes[0].axvline(x1.mean(), color = 'k', linewidth = 1, label = 'avg:'+str(round(x1.mean(), 2)))
    axes[0].legend(loc = 'upper right')
    axes[0].set_xlabel('Count')


    # print(x2)
    axes[1].clear()
    axes[1].hist(x2, bins=100, label = '3G', color = 'g', normed = True)
    axes[1].axvline(x2.mean(), color = 'k', linewidth = 1, label = 'avg:'+str(round(x2.mean(), 2)))
    axes[1].legend(loc = 'upper right')
    axes[1].set_xlabel('Count')

    fig.canvas.set_window_title('Provider-' + provider + ' ' + 'State-' + state)
    #plot both histogram in one figure
    # plt.figure(figsize = (9, 5))
    # plt.hist(x1, bins=100, label = '4G', alpha = 0.5, normed = True)
    # plt.axvline(x1.mean(), linewidth = 1, color = 'b', label = 'avg:'+str(round(x1.mean(), 2)))
    # plt.legend(loc = 'upper right')
    # plt.hist(x2, bins=100, label = '3G', color = 'g', alpha = 0.5, normed = True)
    # plt.axvline(x2.mean(), linewidth = 1, color = 'g', label = 'avg:'+str(round(x.mean(), 2)))
    # plt.legend(loc = 'upper right')

    plt.suptitle('State-' + state + "    " + 'Provider-' + provider, fontsize = 16)
    plt.show()
import ipywidgets as widgets
from ipywidgets import HBox
state_select = widgets.Dropdown(
    options=['All', 'North East', 'Kolkata', 'Bihar', 'Chennai', 'Jammu & Kashmir', 'Delhi',
       'Tamil Nadu', 'Maharashtra', 'Punjab', 'UP East', 'Rajasthan',
       'Gujarat', 'West Bengal', 'Mumbai', 'Kerala', 'Andhra Pradesh',
       'UP West', 'Orissa', 'Assam', 'Madhya Pradesh', 'Karnataka', 'Haryana',
       'Himachal Pradesh'],
    value='All',
    description='States:',
    disabled=False,
)

provider_select = widgets.Dropdown(
    options=['All', 'JIO', 'VODAFONE', 'AIRTEL', 'IDEA', 'CELLONE', 'UNINOR', 'DOLPHIN', 'AIRCEL'],
    value='All',
    description='Provider:',
    disabled=False,
)


def on_change(change):
    if change['type'] == 'change' and change['name'] == 'value':
        plot_graphs(provider_select.value, state_select.value)


state_select.observe(on_change)
provider_select.observe(on_change)

hb = HBox([state_select, provider_select])
display(hb)
plot_graphs(provider_select.value, state_select.value)
#2d histograms
#columns = ['Technology', 'Test_type', 'Service Provider', 'LSA']
import matplotlib.colors as colors

data = df.dropna().mask('Test_type', 'Download').mask('Technology', '4G')
#print(data.isnull().sum())
x = data['Signal_strength']
y = data['Data Speed(Mbps)']
plt.hist2d(x, y, bins = 40, norm=colors.LogNorm())
plt.ylabel('Data Speed(Mbps)')
plt.xlabel('Signal_strength')
plt.show()
# avg speeds of states and service providers
#columns = ['Technology', 'Test_type', 'Service Provider', 'LSA']

state = 'LSA'
service = 'Service Provider'
speed = 'Data Speed(Mbps)'

values = df[state].unique()
r = np.arange(len(values))

plt.figure(figsize = (8, len(values)/2 +1))
plt.xlabel(speed)


# 4g
data = df.mask('Test_type', 'Download').mask('Technology', '4G')
group = data.groupby(by=state)[speed].mean().sort_values(ascending = True)
plt.barh(y = r, width = group.head(len(values)), label = '4G')
for (i, v) in enumerate(group.head(len(values))):
    plt.text(x = v + 0.2, y = i - 0.1, s = str(round(v, 2)), color = 'blue', fontweight = 'bold')
plt.yticks(r, group.index)

# 3g
data = df.mask('Test_type', 'Download').mask('Technology', '3G')
temp = data.groupby(by=state)[speed].mean()
# get correct positions of width according to previous sorting
for v in group.index:
    group.head(len(values))[v] = temp.head(len(values))[v]
    

plt.barh(y = r, width = group.head(len(values)), color = 'y', label = '3G')
for (i, v) in enumerate(group.head(len(values))):
    plt.text(x = v + 0.2, y = i - 0.1, s = str(round(v, 2)), color = 'yellow', fontweight = 'bold')


plt.margins(x = 0.15)
plt.legend(loc = 'lower right')
plt.show()
# avg speeds of states and service providers
#columns = ['Technology', 'Test_type', 'Service Provider', 'LSA']

#4g
print(df[service].unique())
data = df.mask('Test_type', 'Download').mask('Technology', '4G')
values = data[service].unique()
group = data.groupby(by=service)[speed].mean().sort_values(ascending = True)
r = np.arange(len(values))

plt.figure(figsize = (6, len(values)/2 +1))
plt.barh(y = r, width = group.head(len(values)), label = '4G')
for (i, v) in enumerate(group.head(len(values))):
    plt.text(x = v + 0.2, y = i - 0.1, s = str(round(v, 2)), color = 'blue', fontweight = 'bold')
plt.yticks(r, group.index)
plt.xlabel(speed)
plt.margins(x = 0.15)
plt.legend(loc = 'lower right')

#3g
data = df.mask('Test_type', 'Download').mask('Technology', '3G')
values = data[service].unique()
group = data.groupby(by=service)[speed].mean().sort_values(ascending = True)
r = np.arange(len(values))

plt.figure(figsize = (6, len(values)/2 +1))
plt.barh(y = r, width = group.head(len(values)), color = 'g', label = '3G')
for (i, v) in enumerate(group.head(len(values))):
    plt.text(x = v + 0.02, y = i - 0.1, s = str(round(v, 2)), color = 'green', fontweight = 'bold')
plt.yticks(r, group.index)
plt.xlabel(speed)
plt.margins(x = 0.15)
plt.legend(loc = 'lower right')
plt.show()