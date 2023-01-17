import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import matplotlib.cm as cm

import re

sns.set_style("darkgrid")

from sklearn import linear_model

from sklearn.metrics import mean_squared_error, r2_score

from sklearn.preprocessing import PolynomialFeatures

from IPython.core.display import display, HTML, Javascript

from string import Template

import json

import IPython.display
df = pd.read_csv('../input/CompleteDataset.csv')

df.columns
df = df[['Name', 'Age', 'Nationality', 'Overall', 'Potential', 'Club', 'Value', 'Preferred Positions', 'Photo']]

df.head(10)
# get remaining potential

df['Remaining Potential'] = df['Potential'] - df['Overall']





# get only one preferred position (first only)

df['Preferred Position'] = df['Preferred Positions'].str.split().str[0]



# convert K to M

df['Unit'] = df['Value'].str[-1]

df['Value (M)'] = np.where(df['Unit'] == '0', 0, df['Value'].str[1:-1].replace(r'[a-zA-Z]',''))

df['Value (M)'] = df['Value (M)'].astype(float)

df['Value (M)'] = np.where(df['Unit'] == 'M', df['Value (M)'], df['Value (M)']/1000)

df = df.drop('Unit', 1)



df.head(10)
# 'ST', 'RW', 'LW', 'GK', 'CDM', 'CB', 'RM', 'CM', 'LM', 'LB', 'CAM','RB', 'CF', 'RWB', 'LWB'



def get_best_squad(position):

    df_copy = df.copy()

    store = []

    for i in position:

        store.append([i,df_copy.loc[[df_copy[df_copy['Preferred Position'] == i]['Overall'].idxmax()]]['Name'].to_string(index = False), df_copy[df_copy['Preferred Position'] == i]['Overall'].max()])

        df_copy.drop(df_copy[df_copy['Preferred Position'] == i]['Overall'].idxmax(), inplace = True)

    #return store

    return pd.DataFrame(np.array(store).reshape(11,3), columns = ['Position', 'Player', 'Overall']).to_string(index = False)



# 4-3-3

squad_433 = ['GK', 'LB', 'CB', 'CB', 'RB', 'LM', 'CDM', 'RM', 'LW', 'ST', 'RW']

print ('4-3-3')

print (get_best_squad(squad_433))

# 3-5-2

squad_352 = ['GK', 'LWB', 'CB', 'RWB', 'LM', 'CDM', 'CAM', 'CM', 'RM', 'LW', 'RW']

print ('3-5-2')

print (get_best_squad(squad_352))
df_p = df.groupby(['Age'])['Potential'].mean()

df_o = df.groupby(['Age'])['Overall'].mean()



df_summary = pd.concat([df_p, df_o], axis=1)



ax = df_summary.plot()

ax.set_ylabel('Rating')

ax.set_title('Average Rating by Age')
def get_best_squad(position, club = '*', measurement = 'Overall'):

    df_copy = df.copy()

    df_copy = df_copy[df_copy['Club'] == club]

    store = []

    for i in position:

        store.append([df_copy.loc[[df_copy[df_copy['Preferred Position'].str.contains(i)][measurement].idxmax()]]['Preferred Position'].to_string(index = False),df_copy.loc[[df_copy[df_copy['Preferred Position'].str.contains(i)][measurement].idxmax()]]['Name'].to_string(index = False), df_copy[df_copy['Preferred Position'].str.contains(i)][measurement].max(), float(df_copy.loc[[df_copy[df_copy['Preferred Position'].str.contains(i)][measurement].idxmax()]]['Value (M)'].to_string(index = False))])

        df_copy.drop(df_copy[df_copy['Preferred Position'].str.contains(i)][measurement].idxmax(), inplace = True)

    #return store

    return np.mean([x[2] for x in store]).round(1), pd.DataFrame(np.array(store).reshape(11,4), columns = ['Position', 'Player', measurement, 'Value (M)']).to_string(index = False), np.sum([x[3] for x in store]).round(1)



# easier constraint

squad_433_adj = ['GK', 'B$', 'B$', 'B$', 'B$', 'M$', 'M$', 'M$', 'W$|T$', 'W$|T$', 'W$|T$']



# Example Output for Chelsea

rating_433_Chelsea_Overall, best_list_433_Chelsea_Overall, value_433_Chelsea_Overall = get_best_squad(squad_433_adj, 'Chelsea', 'Overall')

rating_433_Chelsea_Potential, best_list_433_Chelsea_Potential, value_433_Chelsea_Potential  = get_best_squad(squad_433_adj, 'Chelsea', 'Potential')



print('-Overall-')

print('Average rating: {:.1f}'.format(rating_433_Chelsea_Overall))

print('Total Value (M): {:.1f}'.format(value_433_Chelsea_Overall))

print(best_list_433_Chelsea_Overall)



print('-Potential-')

print('Average rating: {:.1f}'.format(rating_433_Chelsea_Potential))

print('Total Value (M): {:.1f}'.format(value_433_Chelsea_Potential))

print(best_list_433_Chelsea_Potential)

# very easy constraint since some club do not have strict squad

squad_352_adj = ['GK', 'B$', 'B$', 'B$', 'M$|W$|T$', 'M$|W$|T$', 'M$|W$|T$', 'M$|W$|T$', 'M$|W$|T$', 'W$|T$|M$', 'W$|T$|M$']



By_club = df.groupby(['Club'])['Overall'].mean()



def get_summary(squad):

    OP = []

    # only get top 100 clubs for shorter run-time

    for i in By_club.sort_values(ascending = False).index[0:100]:

        # for overall rating

        O_temp_rating, _, _  = get_best_squad(squad, club = i, measurement = 'Overall')

        # for potential rating & corresponding value

        P_temp_rating, _, P_temp_value = get_best_squad(squad, club = i, measurement = 'Potential')

        OP.append([i, O_temp_rating, P_temp_rating, P_temp_value])

    return OP



OP_df = pd.DataFrame(np.array(get_summary(squad_352_adj)).reshape(-1,4), columns = ['Club', 'Overall', 'Potential', 'Value of highest Potential squad'])

OP_df.set_index('Club', inplace = True)

OP_df = OP_df.astype(float)





print (OP_df.head(10))

    
fig, ax = plt.subplots()

OP_df.plot(kind = 'scatter', x = 'Overall', y = 'Potential', c = 'Value of highest Potential squad', s = 50, figsize = (15,15), xlim = (70, 90), ylim = (70, 90), title = 'Current Rating vs Potential Rating by Club: 3-5-2', ax = ax)

fig, ax = plt.subplots()

OP_df.plot(kind = 'scatter', x = 'Overall', y = 'Potential', c = 'Value of highest Potential squad', s = 50, figsize = (15,15), xlim = (80, 90), ylim = (85, 90), title = 'Current Rating vs Potential Rating by Club: 3-5-2', ax = ax)



def label_point(x, y, val, ax):

    a = pd.concat({'x': x, 'y': y, 'val': val}, axis=1)

    for i, point in a.iterrows():

        ax.text(point['x'], point['y'], str(point['val']))

       

OP_df['Club_label'] = OP_df.index



OP_df_sub = OP_df[(OP_df['Potential']>=85) & (OP_df['Value of highest Potential squad']<=350)]



label_point(OP_df_sub['Overall'], OP_df_sub['Potential'], OP_df_sub['Club_label'], ax)

squad_352_adj = ['GK', 'B$', 'B$', 'B$', 'M$|W$|T$', 'M$|W$|T$', 'M$|W$|T$', 'M$|W$|T$', 'M$|W$|T$', 'W$|T$|M$', 'W$|T$|M$']



rating_352_TH_Overall, best_list_352_TH_Overall, value_352_TH_Overall = get_best_squad(squad_352_adj, 'Tottenham Hotspur', 'Overall')

rating_352_TH_Potential, best_list_352_TH_Potential, value_352_TH_Potential  = get_best_squad(squad_352_adj, 'Tottenham Hotspur', 'Potential')



print('-Overall-')

print('Average rating: {:.1f}'.format(rating_352_TH_Overall))

print('Total Value (M): {:.1f}'.format(value_352_TH_Overall))

print(best_list_352_TH_Overall)



print('-Potential-')

print('Average rating: {:.1f}'.format(rating_352_TH_Potential))

print('Total Value (M): {:.1f}'.format(value_352_TH_Potential))

print(best_list_352_TH_Potential)

def get_best_squad_n(position, nationality, measurement = 'Overall'):

    df_copy = df.copy()

    df_copy = df_copy[df_copy['Nationality'] == nationality]

    store = []

    for i in position:

        store.append([df_copy.loc[[df_copy[df_copy['Preferred Position'].str.contains(i)][measurement].idxmax()]]['Preferred Position'].to_string(index = False),df_copy.loc[[df_copy[df_copy['Preferred Position'].str.contains(i)][measurement].idxmax()]]['Name'].to_string(index = False), df_copy[df_copy['Preferred Position'].str.contains(i)][measurement].max()])

        df_copy.drop(df_copy[df_copy['Preferred Position'].str.contains(i)][measurement].idxmax(), inplace = True)

    #return store

    return np.mean([x[2] for x in store]).round(2), pd.DataFrame(np.array(store).reshape(11,3), columns = ['Position', 'Player', measurement]).to_string(index = False)



def get_summary_n(squad_list, squad_name, nationality_list):

    OP_n = []



    for i in nationality_list:

        count = 0

        for j in squad_list:

            # for overall rating

            O_temp_rating, _  = get_best_squad_n(position = j, nationality = i, measurement = 'Overall')

            # for potential rating & corresponding value

            P_temp_rating, _ = get_best_squad_n(position = j, nationality = i, measurement = 'Potential')

            OP_n.append([i, squad_name[count], O_temp_rating.round(2), P_temp_rating.round(2)])    

            count += 1

    return OP_n



squad_352_strict = ['GK', 'LB|LWB', 'CB', 'RB|RWB', 'LM|W$', 'RM|W$', 'CM', 'CM|CAM|CDM', 'CM|CAM|CDM', 'W$|T$', 'W$|T$']

squad_442_strict = ['GK', 'LB|LWB', 'CB', 'CB', 'RB|RWB', 'LM|W$', 'RM|W$', 'CM', 'CM|CAM|CDM', 'W$|T$', 'W$|T$']

squad_433_strict = ['GK', 'LB|LWB', 'CB', 'CB', 'RB|RWB', 'CM|LM|W$', 'CM|RM|W$', 'CM|CAM|CDM', 'W$|T$', 'W$|T$', 'W$|T$']

squad_343_strict = ['GK', 'LB|LWB', 'CB', 'RB|RWB', 'LM|W$', 'RM|W$', 'CM|CAM|CDM', 'CM|CAM|CDM', 'W$|T$', 'W$|T$', 'W$|T$']

squad_532_strict = ['GK', 'LB|LWB', 'CB|LWB|RWB', 'CB|LWB|RWB', 'CB|LWB|RWB', 'RB|RWB', 'M$|W$', 'M$|W$', 'M$|W$', 'W$|T$', 'W$|T$']





squad_list = [squad_352_strict, squad_442_strict, squad_433_strict, squad_343_strict, squad_532_strict]

squad_name = ['3-5-2', '4-4-2', '4-3-3', '3-4-3', '5-3-2']
rating_352_EN_Overall, best_list_352_EN_Overall = get_best_squad_n(squad_352_strict, 'England', 'Overall')

rating_352_EN_Potential, best_list_352_EN_Potential = get_best_squad_n(squad_352_strict, 'England', 'Potential')



print('-Overall-')

print('Average rating: {:.1f}'.format(rating_352_EN_Overall))

print(best_list_352_EN_Overall)



print('-Potential-')

print('Average rating: {:.1f}'.format(rating_352_EN_Potential))

print(best_list_352_EN_Potential)
OP_df_n = pd.DataFrame(np.array(get_summary_n(squad_list, squad_name, ['England'])).reshape(-1,4), columns = ['Nationality', 'Squad', 'Overall', 'Potential'])

OP_df_n.set_index('Nationality', inplace = True)

OP_df_n[['Overall', 'Potential']] = OP_df_n[['Overall', 'Potential']].astype(float)



print (OP_df_n)
fig, ax = plt.subplots()





OP_df_n.plot(kind = 'barh', x = 'Squad', y = ['Overall', 'Potential'], edgecolor = 'black', color = ['white', 'lightgrey'], figsize = (15,10), title = 'Current and potential rating (Best 11) by squad (England)', ax = ax)





#print (OP_df_n[OP_df_n['Overall'] == OP_df_n['Overall'].max()]['Squad'])



def get_text_y(look_for):

    count = 0

    for i in squad_name:

        if i == look_for:

            return count

        else:

            count += 1



ax.text(OP_df_n['Overall'].max()/2, get_text_y(OP_df_n[OP_df_n['Overall'] == OP_df_n['Overall'].max()]['Squad'].tolist()[0])-0.2, 'Highest Current Rating: {}'.format(OP_df_n['Overall'].max()))

ax.text(OP_df_n['Potential'].max()/2, get_text_y(OP_df_n[OP_df_n['Potential'] == OP_df_n['Potential'].max()]['Squad'].tolist()[0])+0.1, 'Highest Potential Rating: {}'.format(OP_df_n['Potential'].max()))



Country_list = ['Spain','Germany','Brazil','Argentina','Italy']



OP_df_n = pd.DataFrame(np.array(get_summary_n(squad_list, squad_name, Country_list)).reshape(-1,4), columns = ['Nationality', 'Squad', 'Overall', 'Potential'])

OP_df_n.set_index('Nationality', inplace = True)

OP_df_n[['Overall', 'Potential']] = OP_df_n[['Overall', 'Potential']].astype(float)



for i in Country_list:

    OP_df_n_copy = OP_df_n.copy()

    OP_df_n_copy = OP_df_n_copy[OP_df_n_copy.index == i]

    fig, ax = plt.subplots()

    OP_df_n_copy.plot(kind = 'barh', x = 'Squad', y = ['Overall', 'Potential'], edgecolor = 'black', color = ['white', 'lightgrey'], figsize = (15,10), title = 'Current and potential rating (Best 11) by squad ({})'.format(i), ax = ax)



    ax.text(OP_df_n_copy['Overall'].max()/2, get_text_y(OP_df_n_copy[OP_df_n_copy['Overall'] == OP_df_n_copy['Overall'].max()]['Squad'].tolist()[0])-0.2, 'Highest Current Rating: {}'.format(OP_df_n_copy['Overall'].max()))

    ax.text(OP_df_n_copy['Potential'].max()/2, get_text_y(OP_df_n_copy[OP_df_n_copy['Potential'] == OP_df_n_copy['Potential'].max()]['Squad'].tolist()[0])+0.1, 'Highest Potential Rating: {}'.format(OP_df_n_copy['Potential'].max()))
X = df['Overall'].values.reshape(-1,1)

y = df['Value (M)'].values.reshape(-1,1)

regr = linear_model.LinearRegression().fit(X, y)



y_pred = regr.predict(X)

print('Coefficients: ', regr.coef_)

print("Mean squared error: %.2f"% mean_squared_error(y, y_pred))

print('Variance score: %.2f'% r2_score(y, y_pred))



def plot_chart(X, y, y_pred, x_l, x_h, y_l, y_h, c):

    plt.figure(figsize = (15,10))

    plt.scatter(X, y, color=c)

    plt.plot(X, y_pred, color='blue', linewidth=3)



    plt.title('Player value (M) vs rating')

    plt.ylim(y_l,y_h)

    plt.xlim(x_l,x_h)

    plt.ylabel('Value (M)')

    plt.xlabel('Player ratings')

    

plot_chart(X, y, y_pred, 40, 100, 0, 130, 'black')
df_2 = df[df['Value (M)'] != 0]

X_2 = df_2['Overall'].values.reshape(-1,1)

y_2 = df_2['Value (M)'].values.reshape(-1,1)



poly = PolynomialFeatures(degree=2)

X_2_p = poly.fit_transform(X_2)

clf = linear_model.LinearRegression().fit(X_2_p, y_2)

y_2_pred = clf.predict(X_2_p)



print('Coefficients: ', clf.coef_)

print("Mean squared error: %.2f"% mean_squared_error(y_2, y_2_pred))

print('Variance score: %.2f'% r2_score(y_2, y_2_pred))



color_dict = {'ST': 'red', 'CF': 'red', 'LW': 'red', 'RW': 'red',

              'LM': 'blue', 'RM': 'blue', 'CM': 'blue', 'CAM': 'blue', 'CDM': 'blue',

              'LB': 'green', 'RB': 'green', 'CB': 'green', 'LWB': 'green', 'RWB': 'green',

              'GK': 'purple'}



c = df_2['Preferred Position'].map(color_dict)



plot_chart(X_2, y_2, y_2_pred, 40, 100, 0, 130, 'black')



# by positions

plot_chart(X_2, y_2, y_2_pred, 40, 100, 0, 130, c)

plt.text(50, 120, 'Forward', color = 'red')

plt.text(50, 115, 'Midfielder', color = 'blue')

plt.text(50, 110, 'Defender', color = 'green')

plt.text(50, 105, 'Goalkeeper', color = 'purple')
x_low = 85

y_max = 60

plot_chart(X_2, y_2, y_2_pred, x_low, 94, 0, y_max, c)

plt.text(92, 20, 'Forward', color = 'red')

plt.text(92, 17, 'Midfielder', color = 'blue')

plt.text(92, 14, 'Defender', color = 'green')

plt.text(92, 11, 'Goalkeeper', color = 'purple')

ax = plt.gca()

for index, row in df_2.iterrows():

    if row['Overall'] > x_low and row['Value (M)'] < y_max:

        ax.text(row['Overall'], row['Value (M)'], '{}, {}'.format(row['Name'],row['Age']))