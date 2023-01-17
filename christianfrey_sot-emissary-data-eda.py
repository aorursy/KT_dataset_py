import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

from scipy.optimize import curve_fit



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df = pd.read_csv('/kaggle/input/sea-of-thieves-emissary-data/SeaoThievesEmissaryDataMay.csv')

df.head(25)
# Set palette for emissary colors

sea_palette = sns.color_palette(['#c3922e', '#764b74', '#44f3ec', '#326ac9', '#b13314'])

sns.palplot(sea_palette)
# Set palette for ranks within emissaries 

rank_palette = sns.color_palette(['#54473b', '#cc7e5e', '#98b6b7', '#c99a50'])

rank_palette2 = sns.color_palette(['#cc7e5e', '#98b6b7', '#c99a50'])



sns.palplot(rank_palette)
# change date to only day

def date_fix(date):

    date = int(date.split('/')[1])

    return date



df['Date'] = df['Date'].apply(lambda x: date_fix(x))

df.head(25)
df.dtypes
# Now plot graphs for the amount of coins needed to reach an emissary rank





plt.figure(figsize=(12,8))

sns.set(font_scale = 1.3)

sns.set_style("whitegrid")

sns.lineplot(x=df[df['Level']==3]['Date'],y=df[df['Level']==3]['Value'], palette=sea_palette, linewidth=3, hue=df[df['Level']==3]['Emissary'], marker='X').set_title("Value Needed for Highest Rank")
# value and position needed for top rank at

# the end of the month

res=df[df['Level']==3]

res.tail(5)
plt.figure(figsize=(12,8))

sns.set(font_scale = 1.3)

sns.set_style("whitegrid")

a = sns.lineplot(x=df[df['Level']==2]['Date'],y=df[df['Level']==2]['Value'], palette=sea_palette, linewidth=3, hue=df[df['Level']==2]['Emissary'], marker='X').set_title("Value Needed for 3rd Rank")
plt.figure(figsize=(12,8))

sns.set(font_scale = 1.2)

sns.set_style("whitegrid")

a = sns.lineplot(x=df[df['Level']==1]['Date'],y=df[df['Level']==1]['Value'], palette=sea_palette, linewidth=3, hue=df[df['Level']==1]['Emissary'], marker='X').set_title("Value Needed for 2nd Rank")
plt.figure(figsize=(18,22))

sns.set(font_scale = 1.3)

sns.set_style("whitegrid")

a = sns.scatterplot(x=df[df['Level']!=4]['Date'],y=df[df['Level']!=4]['Value'], palette=sea_palette, s=100, hue=df[df['Level']!=4]['Emissary']).set_title("All Rank Values (excluding top player)")
plt.figure(figsize=(12,8))



sns.set(font_scale = 1.5)

sns.set_style("whitegrid")

a = sns.lineplot(x=df[df['Level']==4]['Date'],y=df[df['Level']==4]['Value'], palette=sea_palette, linewidth=3, hue=df[df['Level']==4]['Emissary'], marker='X').set_title("Top Player's Value (Hundred Millions)")
# top player's value at the end of the month

res = df[df['Level']==4]

res.tail(5)
# Now plot position needed to reach higher emissary level



plt.figure(figsize=(12,8))

sns.set(font_scale = 1.5)

sns.set_style("whitegrid")

sns.lineplot(x=df[df['Level']==3]['Date'],y=df[df['Level']==3]['Position'], palette=sea_palette, linewidth=3, hue=df[df['Level']==3]['Emissary'], marker='X').set_title("Position Needed for Highest Rank")
# Now plot position needed to reach higher emissary level



plt.figure(figsize=(12,8))

sns.set(font_scale = 1.5)

sns.set_style("whitegrid")

sns.lineplot(x=df[df['Level']==2]['Date'],y=df[df['Level']==2]['Position'], palette=sea_palette, linewidth=3, hue=df[df['Level']==2]['Emissary'], marker='X').set_title("Position Needed for 2rd Rank")
# Now plot position needed to reach higher emissary level



plt.figure(figsize=(12,8))

sns.set(font_scale = 1.5)

sns.set_style("whitegrid")

sns.lineplot(x=df[df['Level']==1]['Date'],y=df[df['Level']==1]['Position'], palette=sea_palette, linewidth=3, marker='X', hue=df[df['Level']==1]['Emissary']).set_title("Position Needed for 3rd Rank")
# create function to calculate total players in each emissary

# since each emissary is split into even quartiles as levels, 

# the 3rd quartile's position value multiplied by 4 will 

# equal the total amount of players in the emissary



def tot_calc(data):

    data = data*4

    return data



df_tots = df[df['Level']==3]

df_tots = df_tots.reset_index().drop('index', axis=1)

df_tots['PlayerTotal'] = df_tots['Position'].apply(lambda x: tot_calc(x))

df_tots.head(20)
plt.figure(figsize=(12,8))

sns.set(font_scale = 1.5)

sns.set_style("whitegrid")

sns.lineplot(x=df_tots['Date'],y=df_tots['PlayerTotal'], linewidth=3, hue=df_tots['Emissary'], palette=sea_palette, marker='X').set_title("Total Emissary Member Count")
# Total player count at the end of the month

df_tots.tail(5)
# now find percent change rate for player counts

# by creating a new dataframe for each emissary

# then combining the resulting frames



df_rbc = df_tots[df_tots['Emissary'] == "Reaper's Bones"]

df_rbc = df_rbc.reset_index().drop('index', axis=1)



df_rbc['PTotal_%change'] = df_rbc['PlayerTotal'].pct_change()





df_mac = df_tots[df_tots['Emissary'] == "Merchant Alliance"]

df_mac = df_mac.reset_index().drop('index', axis=1)



df_mac['PTotal_%change'] = df_mac['PlayerTotal'].pct_change()





df_ghc = df_tots[df_tots['Emissary'] == "Gold Hoarders"]

df_ghc = df_ghc.reset_index().drop('index', axis=1)



df_ghc['PTotal_%change'] = df_ghc['PlayerTotal'].pct_change()





df_afc = df_tots[df_tots['Emissary'] == "Athena's Fortune"]

df_afc = df_afc.reset_index().drop('index', axis=1)



df_afc['PTotal_%change'] = df_afc['PlayerTotal'].pct_change()



df_osc = df_tots[df_tots['Emissary'] == "Order of Souls"]

df_osc = df_osc.reset_index().drop('index', axis=1)



df_osc['PTotal_%change'] = df_osc['PlayerTotal'].pct_change()





af_allc = df_ghc.append([df_osc, df_afc, df_mac, df_rbc])

af_allc
plt.figure(figsize=(14,12))

sns.set(font_scale = 1.5)

sns.set_style("whitegrid")

sns.lineplot(x=af_allc['Date'],y=af_allc['PTotal_%change'], palette=sea_palette, linewidth=3, hue=af_allc['Emissary'], marker='X').set_title("Percent Change in Total Members")
# now find percent change rate for value

# by creating a new dataframe for each emissary

# then combining the resulting frames



df_rbc = df_tots[df_tots['Emissary'] == "Reaper's Bones"]

df_rbc = df_rbc.reset_index().drop('index', axis=1)



df_rbc['Value_%change'] = df_rbc['Value'].pct_change()



df_mac = df_tots[df_tots['Emissary'] == "Merchant Alliance"]

df_mac = df_mac.reset_index().drop('index', axis=1)



df_mac['Value_%change'] = df_mac['Value'].pct_change()



df_ghc = df_tots[df_tots['Emissary'] == "Gold Hoarders"]

df_ghc = df_ghc.reset_index().drop('index', axis=1)



df_ghc['Value_%change'] = df_ghc['Value'].pct_change()



df_afc = df_tots[df_tots['Emissary'] == "Athena's Fortune"]

df_afc = df_afc.reset_index().drop('index', axis=1)



df_afc['Value_%change'] = df_afc['Value'].pct_change()



df_osc = df_tots[df_tots['Emissary'] == "Order of Souls"]

df_osc = df_osc.reset_index().drop('index', axis=1)



df_osc['Value_%change'] = df_osc['Value'].pct_change()



af_allc = df_ghc.append([df_osc, df_afc, df_mac, df_rbc])





plt.figure(figsize=(14,12))

sns.set(font_scale = 1.5)

sns.set_style("whitegrid")

sns.lineplot(x=af_allc['Date'],y=af_allc['Value_%change'], palette=sea_palette, linewidth=3, hue=af_allc['Emissary'], marker='X').set_title("Percent Change in Min. Highest Rank Value")
# Now plot the difference in value needed to rank up 

# within each emissary (not including top player)



df_af = df[df['Emissary']=="Athena's Fortune"]

df_af = df_af[df_af['Level']!=4]

df_af = df_af.reset_index().drop('index', axis=1)





plt.figure(figsize=(12,8))

sns.set(font_scale = 1.5)

sns.set_style("whitegrid")

sns.lineplot(x=df_af['Date'],y=df_af['Value'], linewidth=3, hue=df_af['Level'],  palette=rank_palette2, marker='X').set_title("Value for Each Athena's Fortune Rank")
df_af = df[df['Emissary']=="Gold Hoarders"]

df_af = df_af[df_af['Level']!=4]

df_af = df_af.reset_index().drop('index', axis=1)





plt.figure(figsize=(12,8))

sns.set(font_scale = 1.5)

sns.set_style("whitegrid")

sns.lineplot(x=df_af['Date'],y=df_af['Value'], linewidth=3, hue=df_af['Level'],  palette=rank_palette2, marker='X').set_title("Value for Each Gold Hoarders Rank")
df_af = df[df['Emissary']=="Merchant Alliance"]

df_af = df_af[df_af['Level']!=4]

df_af = df_af.reset_index().drop('index', axis=1)





plt.figure(figsize=(12,8))

sns.set(font_scale = 1.5)

sns.set_style("whitegrid")

sns.lineplot(x=df_af['Date'],y=df_af['Value'], linewidth=3, hue=df_af['Level'],  palette=rank_palette2, marker='X').set_title("Value for Each Merchant Alliance Rank")
df_af = df[df['Emissary']=="Reaper's Bones"]

df_af = df_af[df_af['Level']!=4]

df_af = df_af.reset_index().drop('index', axis=1)





plt.figure(figsize=(12,8))

sns.set(font_scale = 1.5)

sns.set_style("whitegrid")

sns.lineplot(x=df_af['Date'],y=df_af['Value'], linewidth=3, hue=df_af['Level'],  palette=rank_palette2, marker='X').set_title("Value for Each Reaper's Bones Rank")
df_af = df[df['Emissary']=="Order of Souls"]

df_af = df_af[df_af['Level']!=4]

df_af = df_af.reset_index().drop('index', axis=1)





plt.figure(figsize=(12,8))

sns.set(font_scale = 1.5)

sns.set_style("whitegrid")

sns.lineplot(x=df_af['Date'],y=df_af['Value'], linewidth=3, hue=df_af['Level'],  palette=rank_palette2, marker='X').set_title("Value for Each Order of Souls Rank")
# Now approximate total value earned by each emissary

# this was done by graphing position vs. vlaue by emissary

# take the log of value, then fit a linear regression onto the data.

# One the best fit line is found, find the area under the line

# to the final total player count on May 31st

# this method is far from exact, but is decent given how little 

# continuous data is provided 
emissary = "Reaper's Bones"





df_af = df[df['Emissary'] == emissary]

df_af = df_af.reset_index().drop('index', axis=1)



#find total player count on last day of the month

n = df_tots[(df_tots['Date'] == 31) & (df_tots['Emissary'] == emissary)]['PlayerTotal'].mean()



# take log of value

def log_transf(data):

    data = np.log(data)

    return data



df_af['Value_Log'] = df_af['Value'].apply(lambda x: log_transf(x))



X = df_af['Position']

y = df_af['Value_Log']



#fit the regression to the data and record coefficients

curve_fit = np.polyfit(X, y, 1)

print(curve_fit)

a = curve_fit[0]

b = curve_fit[1]



X_curve1 = np.arange(1,n)

df_curve1 = pd.DataFrame(X_curve1, columns=['X'])



# create a df for the line's coordinates

def curve_calc(data):

    data = a*data + b

    return data



df_curve1['y'] = df_curve1['X'].apply(lambda x: curve_calc(x))







plt.figure(figsize=(12,12))

sns.scatterplot(df_af['Position'], df_af['Value_Log'])

sns.lineplot(df_curve1['X'] ,y = df_curve1['y']).set(xlim=(0,n+(.1*n)),ylim=(0,20))



pos_1_val = a*0 + b

pos_max_val = a*n + b



total_value = ((pos_1_val + pos_max_val)/2)*n







print('y = ' + str(a) + 'x + ' + str(b))

print('Position 1 value: ' + str(pos_1_val))

print('Position max value: ' + str(pos_max_val))

print('Total value: e^' + str(total_value))
emissary = "Reaper's Bones"





df_af = df[df['Emissary'] == emissary]

df_af = df_af.reset_index().drop('index', axis=1)



#find total player count on last day of the month

n = df_tots[(df_tots['Date'] == 31) & (df_tots['Emissary'] == emissary)]['PlayerTotal'].mean()





def log_transf(data):

    data = np.log(data)

    return data



df_af['Value_Log'] = df_af['Value'].apply(lambda x: log_transf(x))



X = df_af['Position']

y = df_af['Value_Log']

curve_fit = np.polyfit(X, y, 1)

print(curve_fit)

a = curve_fit[0]

b = curve_fit[1]



X_curve1 = np.arange(1,n)

df_curve1 = pd.DataFrame(X_curve1, columns=['X'])



def curve_calc(data):

    data = a*data + b

    return data



df_curve1['y'] = df_curve1['X'].apply(lambda x: curve_calc(x))





plt.figure(figsize=(12,12))

sns.scatterplot(df_af['Position'], df_af['Value_Log'])

sns.lineplot(df_curve1['X'] ,y = df_curve1['y']).set(xlim=(0,n+(.1*n)),ylim=(0,20))



pos_1_val = a*0 + b

pos_max_val = a*n + b



total_value = ((pos_1_val + pos_max_val)/2)*n







print('y = ' + str(a) + 'x + ' + str(b))

print('Position 1 value: ' + str(pos_1_val))

print('Position max value: ' + str(pos_max_val))

print('Total value: e^' + str(total_value))
emissary = "Order of Souls"





df_af = df[df['Emissary'] == emissary]

df_af = df_af.reset_index().drop('index', axis=1)



#find total player count on last day of the month

n = df_tots[(df_tots['Date'] == 31) & (df_tots['Emissary'] == emissary)]['PlayerTotal'].mean()





def log_transf(data):

    data = np.log(data)

    return data



df_af['Value_Log'] = df_af['Value'].apply(lambda x: log_transf(x))



X = df_af['Position']

y = df_af['Value_Log']

curve_fit = np.polyfit(X, y, 1)

print(curve_fit)

a = curve_fit[0]

b = curve_fit[1]



X_curve1 = np.arange(1,n)

df_curve1 = pd.DataFrame(X_curve1, columns=['X'])



def curve_calc(data):

    data = a*data + b

    return data



df_curve1['y'] = df_curve1['X'].apply(lambda x: curve_calc(x))





plt.figure(figsize=(12,12))

sns.scatterplot(df_af['Position'], df_af['Value_Log'])

sns.lineplot(df_curve1['X'] ,y = df_curve1['y']).set(xlim=(0,n+(.1*n)),ylim=(0,20))



pos_1_val = a*0 + b

pos_max_val = a*n + b



total_value = ((pos_1_val + pos_max_val)/2)*n







print('y = ' + str(a) + 'x + ' + str(b))

print('Position 1 value: ' + str(pos_1_val))

print('Position max value: ' + str(pos_max_val))

print('Total value: e^' + str(total_value))
emissary = "Athena's Fortune"





df_af = df[df['Emissary'] == emissary]

df_af = df_af.reset_index().drop('index', axis=1)



#find total player count on last day of the month

n = df_tots[(df_tots['Date'] == 31) & (df_tots['Emissary'] == emissary)]['PlayerTotal'].mean()





def log_transf(data):

    data = np.log(data)

    return data



df_af['Value_Log'] = df_af['Value'].apply(lambda x: log_transf(x))



X = df_af['Position']

y = df_af['Value_Log']

curve_fit = np.polyfit(X, y, 1)

print(curve_fit)

a = curve_fit[0]

b = curve_fit[1]



X_curve1 = np.arange(1,n)

df_curve1 = pd.DataFrame(X_curve1, columns=['X'])



def curve_calc(data):

    data = a*data + b

    return data



df_curve1['y'] = df_curve1['X'].apply(lambda x: curve_calc(x))





plt.figure(figsize=(12,12))

sns.scatterplot(df_af['Position'], df_af['Value_Log'])

sns.lineplot(df_curve1['X'] ,y = df_curve1['y']).set(xlim=(0,n+(.1*n)),ylim=(0,20))



pos_1_val = a*0 + b

pos_max_val = a*n + b



total_value = ((pos_1_val + pos_max_val)/2)*n







print('y = ' + str(a) + 'x + ' + str(b))

print('Position 1 value: ' + str(pos_1_val))

print('Position max value: ' + str(pos_max_val))

print('Total value: e^' + str(total_value))
emissary = "Merchant Alliance"





df_af = df[df['Emissary'] == emissary]

df_af = df_af.reset_index().drop('index', axis=1)



#find total player count on last day of the month

n = df_tots[(df_tots['Date'] == 31) & (df_tots['Emissary'] == emissary)]['PlayerTotal'].mean()





def log_transf(data):

    data = np.log(data)

    return data



df_af['Value_Log'] = df_af['Value'].apply(lambda x: log_transf(x))



X = df_af['Position']

y = df_af['Value_Log']

curve_fit = np.polyfit(X, y, 1)

print(curve_fit)

a = curve_fit[0]

b = curve_fit[1]



X_curve1 = np.arange(1,n)

df_curve1 = pd.DataFrame(X_curve1, columns=['X'])



def curve_calc(data):

    data = a*data + b

    return data



df_curve1['y'] = df_curve1['X'].apply(lambda x: curve_calc(x))





plt.figure(figsize=(12,12))

sns.scatterplot(df_af['Position'], df_af['Value_Log'])

sns.lineplot(df_curve1['X'] ,y = df_curve1['y']).set(xlim=(0,n+(.1*n)),ylim=(0,20))



pos_1_val = a*0 + b

pos_max_val = a*n + b



total_value = ((pos_1_val + pos_max_val)/2)*n







print('y = ' + str(a) + 'x + ' + str(b))

print('Position 1 value: ' + str(pos_1_val))

print('Position max value: ' + str(pos_max_val))

print('Total value: e^' + str(total_value))
# Emissary Rankings: Total Value Earned estimator



# 1  GH 5,937,469.24

# 2  OS 4,349,759.95

# 3  RB 4,258,547.00

# 4  MA 3,284,335.62

# 5  AF 2,419,895.25