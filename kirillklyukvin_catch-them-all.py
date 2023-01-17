import numpy as np

import pandas as pd 

import re



import seaborn as sns



import matplotlib.pyplot as plt

import matplotlib.style as style

import matplotlib.patches as mpatches



import plotly.graph_objects as go

import plotly.express as px

import plotly.figure_factory as ff

from plotly.subplots import make_subplots



from matplotlib.colors import ListedColormap

from matplotlib import cm



from pandas_profiling import ProfileReport
pokemons = pd.read_csv('/kaggle/input/pokemon/Pokemon.csv')
pro_rep = ProfileReport(pokemons,minimal=True)
pro_rep.to_widgets()
display(pokemons.describe())

display(pokemons.info())

display(pokemons.head())
pokemons['Type 2'].fillna(value='None', inplace=True)
pokemons['Types'] = pokemons['Type 1'] + '_' + pokemons['Type 2']
temp_1 = pokemons['Type 1'].value_counts().reset_index()



sns.set_style('whitegrid')



plt.figure(figsize=(9,5))

sns.barplot(y=temp_1['Type 1'], x=temp_1['index'], facecolor='white', linewidth=2, edgecolor='black')





plt.xlabel('')

plt.ylabel('')

plt.xticks(rotation=45)

plt.title('Pokemons of Type 1', size=15);
temp_2 = pokemons['Type 2'].value_counts().reset_index().drop(0)



plt.figure(figsize=(9,5))

sns.barplot(y=temp_2['Type 2'], x=temp_2['index'], facecolor='white', linewidth=2, edgecolor='black')





plt.xlabel('')

plt.ylabel('')

plt.xticks(rotation=45)

plt.title('Pokemons of Type 2', size=15);
c_map_grey = ['rgb(220, 220, 220)','rgb(222, 222, 222)','rgb(224, 224, 224)',

      'rgb(226, 226, 226))','rgb(228, 228, 228)','rgb(230, 230, 230',

      'rgb(232, 232, 232)','rgb(234, 234, 234)','rgb(236, 236, 236)',

      'rgb(238, 238, 238)','rgb(240, 240, 240)','rgb(242, 242, 242)',

      'rgb(244, 244, 244)','rgb(246, 246, 246)','rgb(248, 248, 248)',

      'rgb(250, 250, 250)','rgb(252, 252, 252)','rgb(254, 254, 254)']
fig = go.Figure(data=[go.Pie(labels=temp_1['index'], values=temp_1['Type 1'], 

                             textinfo='label+percent',insidetextorientation='radial', marker_colors=c_map_grey

                            )])



fig.update_traces(textposition='inside',

                  marker=dict(line=dict(color='darkgrey', width=0.3)))

fig.update_layout(title_text='Pokemon types pie-chart distribution', title_x=0.5)

fig.show()
temp_3 = pokemons[['Type 1', 'Total']].groupby('Type 1', as_index=False).median().sort_values(by='Total', ascending=False)



plt.figure(figsize=(14,6))



ax = sns.boxplot(x=pokemons['Type 1'], y=pokemons['Total'], linewidth=2.5,order=temp_3['Type 1'])



for i,box in enumerate(ax.artists):

    box.set_edgecolor('black')

    box.set_facecolor('white')



    

    for j in range(6*i,6*(i+1)):

        ax.lines[j].set_color('black')

        ax.lines[j].set_mfc('black')

        ax.lines[j].set_mec('black')

            



plt.xlabel('')

plt.ylabel('')

plt.title('Distribution of "Total" metric between different Types 1', size=15)

plt.xticks(rotation=45, size=12);
best_total = (pokemons[['Types','Total']].groupby('Types')

              .median().sort_values(by='Total', ascending=False)

              .head(10).style.background_gradient(cmap='binary'))

              

best_total 
temp_legendary = pokemons['Legendary'].value_counts().reset_index()

temp_legendary['index'].replace(True,'Legendary',inplace=True)

temp_legendary['index'].replace(False,'Common',inplace=True)
fig = px.pie(values=temp_legendary['Legendary'],

             names=temp_legendary['index'],

             color_discrete_sequence=px.colors.sequential.Greys)



fig.update_traces(textinfo="value+percent+label",

                  textfont_size=10,

                  marker=dict(line=dict(color='darkgrey', width=2)))



fig.update_layout(title_text='Percentage of rare pokemon', title_x=0.5)

fig.show()
legendary = pokemons.query('Legendary == True')

legendary
green = mpatches.Patch(color='darkgreen', label='Legendary')

grey = mpatches.Patch(color='dimgrey', label='Common')



plt.figure(figsize=(3,5))



sns.pointplot(y="Total", data=pokemons.query('Legendary == False'), color='dimgrey')

sns.pointplot(y="Total", data=pokemons.query('Legendary == True'), color='darkgreen')



plt.legend(handles=[green,grey])

plt.title('Legendary and common pokemons Total points', size=15);
temp = pokemons[['Name', 'Legendary','Generation']].groupby(['Generation','Legendary'], as_index=False).count()
temp
plt.figure(figsize=(10,6))



sns.barplot(data=temp, x='Generation', y='Name', hue='Legendary', 

            palette='Greys', linewidth=2, edgecolor='black')



plt.title('Number of legendary and common pokemons by generation', size=15)

plt.xlabel('Generation', size=12)

plt.ylabel('');
common = pokemons.query('Legendary == False').reset_index(drop=True)

common[['Name', 'Total']].sort_values(by='Total', ascending=False).head()
common['Is_mega'] = [True if 'Mega' in name else False for name in common['Name']]



common.head()
common.query('Is_mega == False')[['Name', 'Total']].sort_values(by='Total', ascending=False).head()
common.query('Is_mega == False')[['Name', 'Total', 'Types']].sort_values(by='Total').head()
biggest_classes = common['Type 1'].value_counts().head().index.tolist()

biggest_classes
### Create the data for analysis



hist_data_1 = [

            common.loc[common['Type 1'] == "Water"]['Attack'], 

            common.loc[common['Type 1'] == "Water"]['Defense'], 

            common.loc[common['Type 1'] == "Water"]['Speed'], 

            common.loc[common['Type 1'] == "Water"]['HP']]



hist_data_2 = [

            common.loc[common['Type 1'] == "Normal"]['Attack'], 

            common.loc[common['Type 1'] == "Normal"]['Defense'], 

            common.loc[common['Type 1'] == "Normal"]['Speed'], 

            common.loc[common['Type 1'] == "Normal"]['HP']]



hist_data_3 = [

            common.loc[common['Type 1'] == "Bug"]['Attack'], 

            common.loc[common['Type 1'] == "Bug"]['Defense'], 

            common.loc[common['Type 1'] == "Bug"]['Speed'], 

            common.loc[common['Type 1'] == "Bug"]['HP']]



hist_data_4 = [

            common.loc[common['Type 1'] == "Grass"]['Attack'], 

            common.loc[common['Type 1'] == "Grass"]['Defense'], 

            common.loc[common['Type 1'] == "Grass"]['Speed'], 

            common.loc[common['Type 1'] == "Grass"]['HP']]



hist_data_5 = [

            common.loc[common['Type 1'] == "Fire"]['Attack'], 

            common.loc[common['Type 1'] == "Fire"]['Defense'], 

            common.loc[common['Type 1'] == "Fire"]['Speed'], 

            common.loc[common['Type 1'] == "Fire"]['HP']]



group_labels=['Attack','Deffence','Speed','HP']



colors=['red', 'blue','yellow','green']
### Using Plotly.create_distplots to create data for  draw.

fig1 = ff.create_distplot(

        hist_data_1, group_labels, colors=colors,

        show_hist=False, show_rug=False)

    

fig2 = ff.create_distplot(

        hist_data_2, group_labels, colors=colors,

        show_hist=False, show_rug=False)



fig3 = ff.create_distplot(

        hist_data_3, group_labels, colors=colors,

        show_hist=False, show_rug=False)



fig4 = ff.create_distplot(

        hist_data_4, group_labels, colors=colors,

        show_hist=False, show_rug=False)



fig5 = ff.create_distplot(

        hist_data_5, group_labels, colors=colors,

        show_hist=False, show_rug=False)



### Initialize figure with subplots

fig = make_subplots(rows=5, cols=1,

                    subplot_titles=(biggest_classes),

                    shared_xaxes=True)





### Add kde for Water type

fig.add_trace(go.Scatter(fig1['data'][0]), row=1, col=1)

fig.add_trace(go.Scatter(fig1['data'][1]), row=1, col=1)

fig.add_trace(go.Scatter(fig1['data'][2]), row=1, col=1)

fig.add_trace(go.Scatter(fig1['data'][3]), row=1, col=1)



### Add kde for Normal type

fig.add_trace(go.Scatter(fig2['data'][0]), row=2, col=1)

fig.add_trace(go.Scatter(fig2['data'][1]), row=2, col=1)

fig.add_trace(go.Scatter(fig2['data'][2]), row=2, col=1)

fig.add_trace(go.Scatter(fig2['data'][3]), row=2, col=1)



### Add kde for Bug type

fig.add_trace(go.Scatter(fig3['data'][0]), row=3, col=1)

fig.add_trace(go.Scatter(fig3['data'][1]), row=3, col=1)

fig.add_trace(go.Scatter(fig3['data'][2]), row=3, col=1)

fig.add_trace(go.Scatter(fig3['data'][3]), row=3, col=1)



### Add kde for Grass type

fig.add_trace(go.Scatter(fig4['data'][0]), row=4, col=1)

fig.add_trace(go.Scatter(fig4['data'][1]), row=4, col=1)

fig.add_trace(go.Scatter(fig4['data'][2]), row=4, col=1)

fig.add_trace(go.Scatter(fig4['data'][3]), row=4, col=1)



### Add kde for Fire type

fig.add_trace(go.Scatter(fig5['data'][0]), row=5, col=1)

fig.add_trace(go.Scatter(fig5['data'][1]), row=5, col=1)

fig.add_trace(go.Scatter(fig5['data'][2]), row=5, col=1)

fig.add_trace(go.Scatter(fig5['data'][3]), row=5, col=1)

                 



### Tune layout settings

fig.update_layout(

    height=1400, width=1000,

    title_text='Distribution of stats', title_x=0.5, title_font=dict(size=20),

    template='plotly_white',

   )



fig.show()
temp_speed = common[['Name','Sp. Atk','Sp. Def','Speed', 'Type 1']]
corr_atk = temp_speed['Sp. Atk'].corr(temp_speed['Speed'])

corr_def = temp_speed['Sp. Def'].corr(temp_speed['Speed'])



fig, axes = plt.subplots(2,1, figsize=(10,10))



sns.scatterplot(temp_speed['Sp. Atk'], temp_speed['Speed'], ax=axes[0], color='dimgrey', label='Correlation = {:.2}'.format(corr_atk))

sns.scatterplot(temp_speed['Sp. Def'], temp_speed['Speed'], ax=axes[1], color='dimgrey', label='Correlation = {:.2}'.format(corr_def))



axes[0].set_title('Attack speed distribution', size=12)

axes[1].set_title('Defense speed distribution', size=12)



axes[0].set_xlabel('Attack speed', size=9)

axes[1].set_xlabel('Defense speed', size=9)



axes[0].legend(loc="center right",prop={'size': 12})

axes[1].legend(loc="center right",prop={'size': 12})



plt.suptitle('Correlation beetween movespeed and fighting skills', size=18, y=(1.03))

plt.tight_layout()

plt.show()
atk_sp_threshold = int(np.ceil(common['Sp. Atk'].median()))

attack_threshold = int(np.ceil(common['Attack'].median()))



hp_quant = common['HP'].quantile(q=.35)
def is_agile(row):

    

    atk_sp = row['Sp. Atk']

    atk = row['Attack']

    hp = row['HP']

    

    if atk_sp_threshold <= atk_sp and attack_threshold <= atk and hp <= hp_quant and atk / hp <= 2.5 :

        return True

    else:

        return False

    

common['Is_agile'] = common.apply(is_agile, axis=1)
common[common['Is_agile']==True]
### time for some longer codes 





pokemons['Family'] = 0

fam_numb = 0



for i in range(len(pokemons)):

    

    if i == 0:

        pokemons.loc[i, 'Family'] = fam_numb

        

        

    

    elif i != int(len(pokemons) - 1) and i != 0:

        

        if pokemons.loc[i-1,'Types'] == pokemons.loc[i,'Types'] == pokemons.loc[i+1,'Types'] :   

            fam_numb = fam_numb

            pokemons.loc[i, 'Family'] = fam_numb

       

    

    

        elif pokemons.loc[i-1,'Types'] == pokemons.loc[i,'Types'] != pokemons.loc[i+1,'Types'] :   

            

            if pokemons.loc[i-1,'Type 2'] == 'None' and pokemons.loc[i,'Type 2'] == 'None':

                if pokemons.loc[i-1,'Type 1'] == pokemons.loc[i,'Type 1']:

                    pokemons.loc[i, 'Family'] = fam_numb

                

                else :

                    pokemons.loc[i, 'Family'] = pokemons.loc[i-1,'Type 1'] + 1

                    fam_numb = pokemons.loc[i, 'Family']

                

            elif pokemons.loc[i-1,'Type 2'] == pokemons.loc[i,'Type 2'] :

                

                if pokemons.loc[i-1,'Type 1'] == pokemons.loc[i,'Type 1'] :

                    pokemons.loc[i, 'Family'] = pokemons.loc[i-1, 'Family']

            

                else :

                    pokemons.loc[i, 'Family'] = pokemons.loc[i-1, 'Family'] + 1

                    fam_numb = pokemons.loc[i, 'Family']

                

            elif pokemons.loc[i-1,'Type 2'] == 'None' and pokemons.loc[i,'Type 2'] != 'None':

                pokemons.loc[i, 'Family'] = pokemons.loc[i-1, 'Family']

        

        

        

        elif pokemons.loc[i-1,'Types'] != pokemons.loc[i,'Types'] != pokemons.loc[i+1,'Types'] :

            

            if pokemons.loc[i-1,'Type 2'] == 'None' :

                if pokemons.loc[i-1,'Type 1'] == pokemons.loc[i,'Type 1']: 

                    pokemons.loc[i, 'Family'] = pokemons.loc[i-1,'Family']

                else: 

                    pokemons.loc[i, 'Family'] = pokemons.loc[i-1,'Family'] + 1

                    fam_numb = pokemons.loc[i, 'Family']

            

            elif pokemons.loc[i-1,'Type 1'] == pokemons.loc[i,'Type 1']:

                pokemons.loc[i, 'Family'] = pokemons.loc[i-1,'Family']

                

            else:

                pokemons.loc[i, 'Family'] = pokemons.loc[i-1, 'Family'] + 1

                fam_numb = pokemons.loc[i, 'Family']

                  

        else:  

            fam_numb += 1

            pokemons.loc[i, 'Family'] = fam_numb

         

        

    else:

        pokemons.loc[i, 'Family'] = fam_numb +1 

        
common_upd = pokemons.query('Legendary==False').reset_index(drop=True)

common_upd = common_upd.join(common['Is_agile'])

common_upd.head()
common_upd[72:77]
fam_dict = dict(common_upd.groupby(['Family'])['Is_agile'].apply(lambda x: x.max()))



common_upd['Is_agile'] = common_upd['Is_agile'].replace(False, common_upd['Family'].map(fam_dict))
common_upd[72:80]
temp.interpolate()
temp = common_upd['Is_agile'].value_counts(normalize=True).reset_index()



fig, ax = plt.subplots(figsize=(4,5))



ax.vlines(x=temp.index, ymin=0, ymax=temp.Is_agile, color='dimgrey', linewidth=20)



for i, z in enumerate(temp.Is_agile):

    ax.text(i, z+0.05, np.around(z, 3), horizontalalignment='center')



plt.xticks(temp.index, ['other','agility based'], horizontalalignment='right', fontsize=12)

plt.ylabel('')

plt.xlabel('')

plt.ylim(0,1.15)

plt.title('Percentage of agile pokemons', size=15, y=(1.03));
### Work in progress
def two_pokemon_compare(first, second):

    

    attributes = pokemons.columns.tolist()[5:11]

    

    ind_first = list(pokemons.query('Name == @first').index)[0]

    ind_second = list(pokemons.query('Name == @second').index)[0]

    

    r1 = list(pokemons[attributes].iloc[ind_first])

    last_1 = r1[0]

    r1.append(last_1)



    r2 = list(pokemons[attributes].iloc[ind_second])

    last_2 = r2[0]

    r2.append(last_2)



    title_temp = first + ' VS ' + second

    

    fig = go.Figure()

    

    fig.add_trace(go.Scatterpolar(

        r = r1,

        theta = ['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed', 'HP'],

        name = first,

        line_color = 'black' 

    ))



    fig.add_trace(go.Scatterpolar(

        r = r2,

        theta = ['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed', 'HP'],

        name = second,

        line_color = 'darkgreen' 

    ))



    fig.update_traces(

        fill='none',

        line_width = 3,

        marker_size = 8)



    annotations = []

    annotations.append(dict(xref='paper', yref='paper', x=0.5, y=1.05,

                              xanchor='center', yanchor='bottom',

                              text=title_temp,

                              font=dict(family='Comic Sans MS',

                                        size=30,

                                        color='rgb(37,37,37)'),

                              showarrow=False))

    

    fig.update_layout(

        template=None,

        polar = dict(

              radialaxis_angle = 45),

        annotations=annotations)



    fig.show()
two_pokemon_compare('Pikachu', 'Mewtwo')
#two_pokemon_compare('pokemon_1', 'pokemon_2')