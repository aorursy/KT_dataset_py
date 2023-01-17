import os
import gc
import pandas as pd
import numpy as np
from IPython.display import HTML
import seaborn as sns
import squarify
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import plotly.plotly as py
from plotly.offline import download_plotlyjs, init_notebook_mode, plot,iplot
import plotly.figure_factory as ff
#Always run this the command before at the start of notebook
init_notebook_mode(connected=True)
import plotly.graph_objs as go
import cufflinks as cf
cf.set_config_file(offline=True, world_readable=True, theme='ggplot')
import warnings
warnings.filterwarnings('ignore')
sns.set_style("whitegrid")
#plt.style.use('seaborn')


from random import shuffle
import matplotlib.animation as animation
#%matplotlib notebook
pd.set_option('display.max_columns', 100)
def game_data():
    dir_path = os.getcwd()
    game_data =  pd.read_csv('../input/' + 'game_data.csv')
        
    return game_data

def get_NGS_data_pre():
    dir_path = os.getcwd()
    files = os.listdir('../input/')
    substring = 'pre'
    
    NGS_files = []
    for item in files:
        if substring in item:
            NGS_files.append(item)
    
    data = pd.read_csv('../input/' + NGS_files[0], memory_map=True)
    data['x'].astype(np.float16, inplace=True)
    data['y'].astype(np.float16, inplace=True)
    
    chunksize = 1000000
    for file in NGS_files[1:]:
        for chunk in pd.read_csv('../input/' + file, memory_map=True, chunksize = chunksize):
            chunk['x'] = chunk['x'].astype(np.float16, inplace=True)
            chunk['y'] = chunk['y'].astype(np.float16, inplace=True)
            data = data.append(chunk)
        
    return data.round(1)


def get_NGS_data_post():
    dir_path = os.getcwd()
    files = os.listdir('../input/')
    substring = 'post'
    
    NGS_files = []
    for item in files:
        if substring in item:
            NGS_files.append(item)
    
    data = pd.read_csv('../input/' + NGS_files[0], memory_map=True)
    data['x'].astype(np.float16, inplace=True)
    data['y'].astype(np.float16, inplace=True)
    
    chunksize = 1000000
    for file in NGS_files[1:]:
        for chunk in pd.read_csv('../input/' + file, memory_map=True, chunksize = chunksize):
            chunk['x'] = chunk['x'].astype(np.float16, inplace=True)
            chunk['y'] = chunk['y'].astype(np.float16, inplace=True)
            data = data.append(chunk)
        
    return data.round(1)


def get_NGS_data_reg16(usecols):
    dir_path = os.getcwd()
    files = os.listdir('../input/')
    substring = 'reg'
    
    NGS_files = []
    for item in files:
        if substring in item and '16' in item:
            NGS_files.append(item)
    
    data = pd.read_csv('../input/' + NGS_files[0], memory_map=True, usecols=usecols)
    data['x'].astype(np.float16, inplace=True)
    data['y'].astype(np.float16, inplace=True)
    data['Season_Year'] = data['Season_Year'].astype(np.int16, inplace=True)
    data['GameKey'] = data['GameKey'].astype(np.int32, inplace=True)
            
    data['PlayID'] = data['PlayID'].astype(np.int32, inplace=True)
    #data['GSISID'] = data['GSISID'].astype(np.int32, inplace=True)
    
    chunksize = 1000000
    for file in NGS_files[1:]:
        for chunk in pd.read_csv('../input/' + file, memory_map=True, chunksize = chunksize, usecols=usecols):
            chunk['x'] = chunk['x'].astype(np.float16, inplace=True)
            chunk['y'] = chunk['y'].astype(np.float16, inplace=True)
            
            chunk['Season_Year'] = chunk['Season_Year'].astype(np.int16, inplace=True)
            chunk['GameKey'] = chunk['GameKey'].astype(np.int32, inplace=True)
            
            chunk['PlayID'] = chunk['PlayID'].astype(np.int32, inplace=True)
            #chunk['GSISID'] = chunk['GSISID'].astype(np.int32, inplace=True)
            data = data.append(chunk)
        
    return data.round(1)


def get_NGS_data_reg17(usecols):
    dir_path = os.getcwd()
    files = os.listdir('../input/')
    substring = 'reg'
    
    NGS_files = []
    for item in files:
        if substring in item and '17' in item:
            NGS_files.append(item)
    
    data = pd.read_csv('../input/' + NGS_files[0], memory_map=True, usecols=usecols)
    data['x'].astype(np.float16, inplace=True)
    data['y'].astype(np.float16, inplace=True)
    data['Season_Year'] = data['Season_Year'].astype(np.int16, inplace=True)
    data['GameKey'] = data['GameKey'].astype(np.int32, inplace=True)
            
    data['PlayID'] = data['PlayID'].astype(np.int32, inplace=True)
    #data['GSISID'] = data['GSISID'].astype(np.int32, inplace=True)
    
    chunksize = 1000000
    for file in NGS_files[1:]:
        for chunk in pd.read_csv('../input/' + file, memory_map=True, chunksize = chunksize, usecols=usecols):
            chunk['x'] = chunk['x'].astype(np.float16, inplace=True)
            chunk['y'] = chunk['y'].astype(np.float16, inplace=True)
            
            chunk['Season_Year'] = chunk['Season_Year'].astype(np.int16, inplace=True)
            chunk['GameKey'] = chunk['GameKey'].astype(np.int32, inplace=True)
            
            chunk['PlayID'] = chunk['PlayID'].astype(np.int32, inplace=True)
            #chunk['GSISID'] = chunk['GSISID'].astype(np.int32, inplace=True)
            data = data.append(chunk)
        
    return data.round(1)


def vid_review_data():
    dir_path = os.getcwd()
    vid_data =  pd.read_csv('../input/' + 'video_review.csv')
        
    return vid_data


def vid_injury_data():
    dir_path = os.getcwd()
    vid_data =  pd.read_csv('../input/' + 'video_footage-injury.csv')
        
    return vid_data


def player_role_data():
    dir_path = os.getcwd()
    role_data =  pd.read_csv('../input/' + 'play_player_role_data.csv')
        
    return role_data


def player_punt_data():
    dir_path = os.getcwd()
    punt_data =  pd.read_csv('../input/' + 'player_punt_data.csv')
        
    return punt_data


def play_info_data():
    dir_path = os.getcwd()
    play_data =  pd.read_csv('../input/' + 'play_information.csv')
        
    return play_data


def generate_table(data):
    
    trace = go.Table(
    header=dict(values=[data.columns[0], data.columns[1]],
                fill = dict(color='#C2D4FF'),
                align = ['left'] * 5),
    cells=dict(values=[data[data.columns[0]], data[data.columns[1]]],
               fill = dict(color='#F5F8FF'),
               align = ['left'] * 5))
    layout = go.Layout(
     autosize=True)

    data = [trace]
    filename = 'pandas_table'
    
    fig = go.Figure(data=data, layout=layout)
    return [fig, filename]


def generate_table3(data):
    
    trace = go.Table(
    header=dict(values=[data.columns[0], data.columns[1], data.columns[2]],
                fill = dict(color='#C2D4FF'),
                align = ['left'] * 5),
    cells=dict(values=[data[data.columns[0]], data[data.columns[1]], data[data.columns[2]]],
               fill = dict(color='#F5F8FF'),
               align = ['left'] * 5))
    layout = go.Layout(
     autosize=True)

    data = [trace]
    filename = 'pandas_table'
    
    fig = go.Figure(data=data, layout=layout)
    return [fig, filename]


def generate_bar_plot(x_data, y_data, data, pal, xaxis_title, yaxis_title, title, hue):
    fig, axes = plt.subplots(1,1,figsize=(10,10))
    ax = sns.barplot(x = x_data, y=y_data, data=data, palette=pal, hue = hue)
    ax.set_ylabel(yaxis_title,fontsize=15)
    ax.set_xlabel(xaxis_title,fontsize=15)
    ax.set_title(title ,fontsize=15)
    ax.tick_params(labelsize=12.5)
    try:
        plt.setp(ax.get_legend().get_texts(), fontsize='15') # for legend text
        plt.setp(ax.get_legend().get_title(), fontsize='15') # for legend title
        
        plt.show()
    except:
        plt.show()
        

def generate_line_plot(x_data, y_data, data, xaxis_title, yaxis_title, title):
    fig, axes = plt.subplots(1,1,figsize=(10,10))
    ax = sns.lineplot(x = x_data, y=y_data, data=data)
    ax.set_ylabel(yaxis_title,fontsize=15)
    ax.set_xlabel(xaxis_title,fontsize=15)
    ax.set_title(title ,fontsize=15)
    ax.tick_params(labelsize=12.5)
    try:
        plt.setp(ax.get_legend().get_texts(), fontsize='15') # for legend text
        plt.setp(ax.get_legend().get_title(), fontsize='15') # for legend title
        
        plt.show()
    except:
        plt.show()


def pie_chart(data, x, y, colour, title):    
    data.iplot(kind='pie', labels = y, values = x, pull=.1, hole=.1,  
          colorscale = colour, textposition='outside', 
        title = title)    

    
def scatter_plot(x, y, label, colour, xaxis_title, yaxis_title, title):
    
    fig,ax = plt.subplots(1,1,figsize=(10, 10))
    plt.scatter(x, y, color= colour, alpha=1, label= label)
    ax.set_ylabel(yaxis_title,fontsize=15)
    ax.set_xlabel(xaxis_title,fontsize=15)
    ax.set_title(title ,fontsize=15)
    ax.tick_params(labelsize=10)
    plt.legend()
    plt.show()
    
    
def bubble_plot(data, x, y, colour, xaxis_title, yaxis_title, title, size, categories):
    data.iplot(kind ='bubble', colorscale = colour, categories= categories, x = x, y = y, size = size,
                xTitle = xaxis_title, yTitle = yaxis_title, title = title)
    
    
def plotly_stacked_bar(data, categories, x_data, y_data, x_title, y_title, title):
    temp_data = []
   
    for index, item in enumerate(categories):
        trace = go.Bar(
        x = data[index][x_data],
        y = data[index][y_data],
        name=item)
        temp_data.append(trace)

    data = temp_data
    layout = go.Layout(
    barmode='stack',
    title=title,
    xaxis=go.layout.XAxis(title=x_title),
    yaxis=go.layout.YAxis(title=y_title)
    )

    fig = go.Figure(data=data, layout=layout)
    iplot(fig, filename='stacked-bar')
    

def pyramid_plot(data_1, data_2, name_1, name_2, x_title, title, category, category_2):

    values = data_2['Percentage'] * -1
    values = values.append(data_1['Percentage'])
    values = values.sort_values(ascending=True)

    min_v = int(values.min())
    max_v = int(values.max())
    values = [int(i) for i in values]

    labels = data_2['Percentage'] * -1
    labels = labels.append(data_1['Percentage'])
    labels = labels.sort_values(ascending=True)
    labels = [int(i) for i in labels]

    new_labels =[]
    for item in labels:
        if item < 0:
            item = item * -1
            new_labels.append(item)
        else:
            new_labels.append(item)

    data_2['Percentage'] = data_2['Percentage'] * -1


    layout = go.Layout(title=title,
                       yaxis=go.layout.YAxis(tickangle=-15),
                       xaxis=go.layout.XAxis(
                           tickangle=-55,
                           range=[min_v, max_v],
                           tickvals= [int(i) for i in values],
                           ticktext= new_labels,
                           title=x_title),
                       barmode='overlay',
                       bargap=0.5,
                       height=500,
                      width=900, 
                      margin=go.layout.Margin(l=225, r=0))

    data = [go.Bar(y=data_1[category],
                   x=data_1['Percentage'],
                   orientation='h',
                   name=name_1,
                   marker=dict(color='green')
                   ),
            go.Bar(y=data_1[category],
                   x=data_2['Percentage'],
                   orientation='h',
                   name=name_2,
                   marker=dict(color='orange')
                   )]

    iplot(dict(data=data, layout=layout), filename='EXAMPLES/bar_pyramid')
    
    
def draw_pitch(data, col1, col2, title, poss_team, oppostion):
    #layout sourced from https://fcpython.com/visualisation/drawing-pitchmap-adding-lines-circles-matplotlib
    #pitch is 53 yards by 100 yards excluding two 10 yard touchdown zones.
    labels = ['Goal','10','20','30','40','50','40','30','20','10','Goal']
    fig = plt.figure(facecolor='white', figsize=(12.5,5))
    ax = fig.add_subplot(1,1,1)
    ax.set_facecolor('green')
    plt.yticks([]) # disable yticks
    
    start_x = -10
    bottom_y = 0
    top_y = 53
    
    ticks = [item * 10 for item in range(0,11)]
    #(x1,x2) (y1,y2)
    
    plt.plot([-10, 110],[0, 0], color='white', linewidth=4)
    plt.plot([-10, 110],[53, 53], color='white', linewidth=4)

    
    for item in range(0,28):
        if item == 0:
            plt.plot([start_x, start_x],[bottom_y, top_y], color='white', linewidth=4)
        
        if item >=1  and item <= 28:
            if item % 2 == 1:
                if item == 0 or item == 27:
                    plt.plot([start_x, start_x],[bottom_y, top_y], color='white', linewidth=4)
                    start_x = start_x + 5
                else:
                    plt.plot([start_x, start_x],[bottom_y, top_y], color='white', linestyle="dashed")
                    start_x = start_x + 5
                
            else:
                if start_x >=0 and start_x < 110:
                    plt.plot([start_x, start_x],[bottom_y, top_y], color='white', linewidth=4)
                    start_x = start_x + 5 
                    
    y_value = []
    for i in range(len(data)):
        y_value.append(10 + i * 5)
                    
    for item in range(len(data)):
        plt.scatter(data[col1][item], y_value[item], s=80, color="red")
        plt.scatter(data[col2][item], y_value[item], s=80, color="yellow")
        ax.text(data[col1][item], y_value[item], poss_team[item], ha='left', size=12.5, color='black')
        ax.text(data[col2][item], y_value[item], oppostion[item], ha='left', size=12.5, color='black')

    plt.xticks(ticks, labels, size=15)
    plt.title(title, fontsize=20)
    plt.show()
    
    
def draw_heatmap(data, col1, col2, title, levels):
    #layout sourced from https://fcpython.com/visualisation/drawing-pitchmap-adding-lines-circles-matplotlib
    #pitch is 53 yards by 100 yards excluding two 10 yard touchdown zones.
    labels = ['Goal','10','20','30','40','50','40','30','20','10','Goal']
    fig = plt.figure(facecolor='white', figsize=(12.5,5))
    ax = fig.add_subplot(1,1,1)
    ax.set_facecolor('darkgreen')
    plt.yticks([]) # disable yticks
    
    start_x = -10
    bottom_y = 0
    top_y = 53
    
    ticks = [item * 10 for item in range(0,11)]
    #(x1,x2) (y1,y2)
    
    ax = sns.kdeplot(shade=True, n_levels=levels, data=data, cmap='Greens_r')
    #ax = sns.scatterplot(col1,col2, data=data, alpha=.4)
    
    plt.plot([-10, 110],[0, 0], color='white', linewidth=4)
    plt.plot([-10, 110],[53, 53], color='white', linewidth=4)

    
    for item in range(0,28):
        if item == 0:
            plt.plot([start_x, start_x],[bottom_y, top_y], color='white', linewidth=4)
        
        if item >=1  and item <= 28:
            if item % 2 == 1:
                if item == 0 or item == 27:
                    plt.plot([start_x, start_x],[bottom_y, top_y], color='white', linewidth=4)
                    start_x = start_x + 5
                else:
                    plt.plot([start_x, start_x],[bottom_y, top_y], color='white', linestyle="dashed")
                    start_x = start_x + 5
                
            else:
                if start_x >=0 and start_x < 110:
                    plt.plot([start_x, start_x],[bottom_y, top_y], color='white', linewidth=4)
                    start_x = start_x + 5 

    
    plt.ylim(0, 53)
    plt.xlim(-10, 110)
    plt.xticks(ticks, labels, size=15)
    plt.title(title, fontsize=20)
    plt.show()
    
    
    
def plotly_scatter(x, y, mode, categories, xaxis, yaxis, title, data ):
    data.iplot(kind='scatter', mode=mode, categories=categories, x=x, y=y, xTitle=xaxis_title,
               yTitle=yaxis_title, title=title, filename='cufflinks/simple-scatter')
    

def plotly_line_graph(data, x_data, y_data, names, xaxis, yaxis, title):
    temp = []
    for index, item in enumerate(data):
        trace = go.Scatter(x=item[x_data],
                          y=item[y_data],
                          name = names[index])
        temp.append(trace)
        
    layout = go.Layout(title=title,
                       yaxis=go.layout.YAxis(tickangle=-15, title=yaxis),
                       xaxis=go.layout.XAxis(tickangle=-45, title=xaxis))

    fig = dict(data=temp, layout=layout)
    iplot(fig, filename='simple-connectgaps')
    
    
def perform_merge(data1, data2, columns):
    merged_data = pd.merge(data1, data2, left_on=columns, right_on=columns, suffixes=['','_1'], how='left')
    return merged_data


def punt_received(data):
    yards_gained = []
    
    for row in data['PlayDescription']:
        temp = row.split('punts')[1].split(' ')[1]
        yards_gained.append(int(temp))
    
    data['kicked_to'] = yards_gained
    data['kicked_to'] = data['kicked_to'] + data['kicked_from']
    return data


def punt_from(data):
    yardline = []
    
    for row in data['YardLine']:
        temp = row.split(' ')[1]
        yardline.append(int(temp))
    
    data['kicked_from'] = yardline
    return data


def opposition_team(data): 
    opposition = []
    
    for item in data.iterrows():
        teams = item[1]['Home_Team_Visit_Team'].split('-')
        poss_team = item[1]['Poss_Team']
        for element in teams:
            if poss_team != element:
                opposition.append(element)
    data['oppostion'] = opposition

    return data


def visiting_data(data):
    score_away = []
    away_team = []
    for item in data['Score_Home_Visiting']:
        scores = item.split('-')
        temp =  int(scores[1])
        score_away.append(temp)
        
    for item in data['Home_Team_Visit_Team']:
        teams = item.split('-')
        temp =  teams[1].strip()
        away_team.append(temp)
        
    data['visiting_team'] = away_team    
    data['visit_score'] = score_away
    
    return data


def home_data(data):
    home_score = []
    home_team = []
    for item in data['Score_Home_Visiting']:
        scores = item.split('-')
        temp =  int(scores[0])
        home_score.append(temp)
        
    for item in data['Home_Team_Visit_Team']:
        teams = item.split('-')
        temp =  teams[0].strip()
        home_team.append(temp)

    data['home_team'] = home_team     
    data['home_score'] = home_score
    
    return data


def score_difference(data):
    data['score_diff'] = abs(data['home_score'] - data['visit_score'])
    
    return data


def check_winning(data):
    win_or_lose = []
    for row in data.iterrows():
        items = row[1]
        if items['Poss_Team'] == items['home_team']:
            if items['home_score'] < items['visit_score']:
                win_or_lose.append('Losing')
            if items['home_score'] > items['visit_score']:
                win_or_lose.append('Winning')
                
        elif items['Poss_Team'] == items['visiting_team']:
            if items['visit_score'] < items['home_score']:
                win_or_lose.append('Losing')
            if items['visit_score'] > items['home_score']:
                win_or_lose.append('Winning')
                
        if items['home_score'] == items['visit_score']:
            win_or_lose.append('Draw Game')
    
    data['Poss_Team_Status'] = win_or_lose
    
    return data


def missing_data(data):
    missing = pd.DataFrame(data.isnull().sum()).reset_index()
    missing.columns = ['Column', 'Count']
    missing['Percentage_Observations_Missing'] =  missing['Count'] / len(data) * 100
    return missing


def count_agg(group_columns, data):
    temp_data = data.groupby(group_columns, as_index=False).size()
    agg = pd.DataFrame(temp_data.reset_index())
    group_columns.append('Count')
    agg.columns = group_columns
    agg['Percentage'] = agg['Count'] / agg['Count'].sum() * 100
    
    return agg


def merge_all(play_data, dataframes):
    for item in dataframes:
        cols = list(play_data.columns.intersection(item.columns))
        play_data = perform_merge(play_data, item, cols)
    
    return play_data


def get_merged():
    game = game_data() #general game data, date, time, location etc.
    play = play_info_data()
    video_data = vid_review_data() #concussion data, how it happened etc.
    role = player_role_data() #players role during the pun
    punt = player_punt_data() #player postion
    injury = vid_injury_data()

    injury = injury.rename(columns={'gamekey':'GameKey', 'playid':'PlayID'})

    dataframes = [role, punt]
    all_data = merge_all(game, dataframes)
    all_data = perform_merge(all_data, play, columns=['GameKey', 'PlayID','Season_Year'])

    del(game)
    del(play)
    del(role)
    del(punt)
    gc.collect()
    
    return all_data


def get_NGS_data(data):
    usecols = None
    
    NGS = get_NGS_data_pre()
    cols = list(data.columns.intersection(NGS.columns))
    pre = data.merge(NGS, on = cols)
    
    NGS = get_NGS_data_reg16(usecols)
    cols = list(data.columns.intersection(NGS.columns))
    reg16 = data.merge(NGS, on = cols)
    
    NGS = get_NGS_data_reg17(usecols)
    cols = list(data.columns.intersection(NGS.columns))
    reg17 = data.merge(NGS, on = cols)
    
    NGS = get_NGS_data_post()
    cols = list(data.columns.intersection(NGS.columns))
    post = data.merge(NGS, on = cols)

    pre = pre.append(reg16)
    pre = pre.append(reg17)
    pre = pre.append(post)
    
    return pre


def group_interactive_data(data):
    grouped = data.groupby(by=['GameKey', 'GSISID', 'PlayID'])
    return grouped


def interactive_plot_visible(visible, total_items):
    vis = []
    
    if visible == 0:
        vis = [True]
        for item in range(total_items - 1):
            vis.append(False)
    else:
        for index, item in enumerate(range(total_items)):
            if index == visible:
                vis.append(True)
            else:
                vis.append(False)
    
    return vis


def gen_interactive_plot(button_labels, dataframes, col1, col2):
    traces = []
    menu_items = []
    buttons = []
    list_buttons = dict(active=0)
    
    for index, item in enumerate(dataframes):
        item = item.sort_values('Time')
        item = item.drop_duplicates(subset=['x', 'y'])
        item = assign_colours(item)
        #each trace contains all coordinates for entire dataframe
        trace = go.Scatter(x=list(item[col1]),
                            y=list(item[col2]),
                            #name=button_labels[index],
                           text = item['Event'],
                            mode = 'lines+markers',
                          marker = dict(
                            size = 10,
                            color = item['Colour']))
        traces.append(trace)
        
    data = traces
    
    button = dict(label = 'All Plays',
                 method = 'update',
                 args = [{'visible': [True for item in range(len(button_labels)) ]},
                         {'title': 'Player Movement: '}])
    buttons.append(button)
    
    for index, item in enumerate(button_labels):
        button = dict(label = item,
                 method = 'update',
                 args = [{'visible': interactive_plot_visible(index, len(button_labels)) },
                         {'title': 'Player Movement: ' + item}])
        buttons.append(button)
    
                      
    list_buttons.update({'buttons':buttons, 'direction':'down',
        'pad': {'r': 10, 't': 10},
        'showactive': True,
        'x': 0.1,
        'xanchor': 'left',
        'y': 1.15,
        'yanchor': 'top'})
    
    updatemenus = [list_buttons]
    
    #Generate field lines
    x0 = 10
    x1 = 10
    y0 = 0
    y1 = 53
    
    #the two side lines
    lines = []
    
    sideline1 = {'type': 'line', 'x0': 0, 'y0': 0, 'x1': 120, 'y1': 0,'line': {'color': 'white','width': 5,},}
    
    sideline2 = {'type': 'line', 'x0': 0, 'y0': 53, 'x1': 120, 'y1': 53,'line': {'color': 'white','width': 5,},}

    
    for item in range(0,22):        
        if item >=1  and item <= 21:
            if item % 2 == 1:
                if item == 0 or item == 21:
                    line = {'type': 'line', 'layer':'below', 'x0': x0, 'y0': y0, 'x1': x1, 'y1': y1,
            'line': {'color': 'white','width': 3,},}
                    lines.append(line)
                    x0 = x0 + 5
                    x1 = x1 + 5
                else:
                    #dashed lines
                    line = {'type': 'line', 'layer':'below', 'x0': x0, 'y0': y0, 'x1': x1, 'y1': y1,
            'line': {'color': 'white','width': 3, },}
                    lines.append(line)
                    x0 = x0 + 5
                    x1 = x1 + 5
                
            else:
                if x0 >=0 and x0 < 110:
                    line = {'type': 'line', 'layer':'below', 'x0': x0, 'y0': y0, 'x1': x1, 'y1': y1,
            'line': {'color': 'white','width': 3, 'dash': 'dashdot'},}
                    lines.append(line)
                    x0 = x0 + 5
                    x1 = x1 + 5
    
    lines.append(sideline1)
    lines.append(sideline2)

    layout = dict(title= 'Movements Leading To Injury', showlegend=False,
                  width=900,
                  height=520,
                  margin=dict(t=0, b=60, l=0, r=0),
                  updatemenus=updatemenus,
                  plot_bgcolor='MediumSeaGreen',
                  xaxis=dict(
                    title='',
                      range = [0,120],
                      tickmode='array',
                      showgrid=False,
                      tickangle=45,
                    tickvals=[10,20, 30, 40, 50, 60, 70, 80, 90, 100, 110],
                    ticktext=['End Zone', 10, 20, 30, 40, 50, 40, 30, 20, 10, 'End Zone'],
                    showticklabels=True
                  ),
                  yaxis=dict(
                    title='',
                      showgrid=False,
                      range = [0,53]
                  ),
                 shapes = lines,
                 annotations=[
        dict(
            x=5,
            y=25,
            showarrow=False,
            text='End Zone',
            textangle=-90,
            font=dict(
                family='Overpass',
                size=22,
                color='yellow'
            ),
            xref='x',
            yref='y'
        ),
        dict(
            x=115,
            y=25,
            showarrow=False,
            text='End Zone',
            textangle=-90,
            font=dict(
                family='Overpass',
                size=22,
                color='yellow'
            ),
            xref='x',
            yref='y'
        )
        
    ])
    

    fig = dict(data=data, layout=layout)
    filename='update_dropdown'
    return [fig, filename]


def assign_colours(dataframe):
    colours = """aliceblue, antiquewhite, aqua, aquamarine, azure,
            beige, bisque, black, blanchedalmond, blue,
            blueviolet, brown, burlywood, cadetblue,
            chartreuse, chocolate, coral, cornflowerblue,
            cornsilk, crimson, cyan, darkblue, darkcyan,
            darkgoldenrod, darkgray, darkgrey, darkgreen,
            darkkhaki, darkmagenta, darkolivegreen, darkorange,
            darkorchid, darkred, darksalmon, darkseagreen,
            darkslateblue, darkslategray, darkslategrey,
            darkturquoise, darkviolet, deeppink, deepskyblue,
            dimgray, dimgrey, dodgerblue, firebrick,
            floralwhite, forestgreen, fuchsia, gainsboro,
            ghostwhite, gold, goldenrod, gray, grey, green,
            greenyellow, honeydew, hotpink, indianred, indigo,
            ivory, khaki, lavender, lavenderblush, lawngreen,
            lemonchiffon, lightblue, lightcoral, lightcyan,
            lightgoldenrodyellow, lightgray, lightgrey,
            lightgreen, lightpink, lightsalmon, lightseagreen,
            lightskyblue, lightslategray, lightslategrey,
            lightsteelblue, lightyellow, lime, limegreen,
            linen, magenta, maroon, mediumaquamarine,
            mediumblue, mediumorchid, mediumpurple,
            mediumseagreen, mediumslateblue, mediumspringgreen,
            mediumturquoise, mediumvioletred, midnightblue,
            mintcream, mistyrose, moccasin, navajowhite, navy,
            oldlace, olive, olivedrab, orange, orangered,
            orchid, palegoldenrod, palegreen, paleturquoise,
            palevioletred, papayawhip, peachpuff, peru, pink,
            plum, powderblue, purple, red, rosybrown,
            royalblue, saddlebrown, salmon, sandybrown,
            seagreen, seashell, sienna, silver, skyblue,
            slateblue, slategray, slategrey, snow, springgreen,
            steelblue, tan, teal, thistle, tomato, turquoise,
            violet, wheat, white, whitesmoke, yellow,
            yellowgreen"""

    colour = []
    for item in colours.split(','):
        item = str(item.strip())
        colour.append(item)
    

    shuffle(colour)
    dataframe['Event'] = dataframe['Event'].replace(np.nan, 'General Play')
    dataframe['Colour'] = 'blue'
    colour.remove('blue')
    colour.remove('green')
    
    for index, event in enumerate(pd.unique(dataframe['Event'])):
        dataframe.loc[dataframe['Event'] == event, 'Colour'] = colour[index]
        colour.remove(colour[index]) #stop colours being duplicated

    return dataframe


def assign_matplot_colours(dataframe):
    colour_list = []
    for key in mcolors.CSS4_COLORS:
        colour_list.append(key)
    
    shuffle(colour_list)
    dataframe['Event'] = dataframe['Event'].replace(np.nan, 'General Play')
    dataframe['Colour'] = 'blue'
    colour_list.remove('blue')
    colour_list.remove('green')
    colour_list.remove('white')
    colour_list.remove('red')
    
    for index, event in enumerate(pd.unique(dataframe['Event'])):
        dataframe.loc[dataframe['Event'] == event, 'Colour'] = colour_list[index]
        colour_list.remove(colour_list[index])#stop colours being duplicated
        

    return dataframe


def plot_play(dataframe):
    dataframe = dataframe.sort_values('Time')
    dataframe = assign_colours(dataframe)
    dataframe = dataframe.drop_duplicates(subset=['x', 'y'])
    list_coords = []
    coordinates = {}

    title = list(dataframe['Home_Team'])[0] + ' vs. ' + list(dataframe['Visit_Team'])[0]


    ####################################################
     #Generate field lines
    x0 = 10
    x1 = 10
    y0 = 0
    y1 = 53

    #the two side lines
    lines = []

    sideline1 = {'type': 'line', 'x0': 0, 'y0': 0, 'x1': 120, 'y1': 0,'line': {'color': 'white','width': 5,},}

    sideline2 = {'type': 'line', 'x0': 0, 'y0': 53, 'x1': 120, 'y1': 53,'line': {'color': 'white','width': 5,},}


    for item in range(0,22):        
        if item >=1  and item <= 21:
            if item % 2 == 1:
                if item == 0 or item == 21:
                    line = {'type': 'line', 'layer':'below', 'x0': x0, 'y0': y0, 'x1': x1, 'y1': y1,
            'line': {'color': 'white','width': 3,},}
                    lines.append(line)
                    x0 = x0 + 5
                    x1 = x1 + 5
                else:
                    #dashed lines
                    line = {'type': 'line', 'layer':'below', 'x0': x0, 'y0': y0, 'x1': x1, 'y1': y1,
            'line': {'color': 'white','width': 3, },}
                    lines.append(line)
                    x0 = x0 + 5
                    x1 = x1 + 5

            else:
                if x0 >=0 and x0 < 110:
                    line = {'type': 'line', 'layer':'below', 'x0': x0, 'y0': y0, 'x1': x1, 'y1': y1,
            'line': {'color': 'white','width': 3, 'dash': 'dashdot'},}
                    lines.append(line)
                    x0 = x0 + 5
                    x1 = x1 + 5

    lines.append(sideline1)
    lines.append(sideline2)
    

    for item in dataframe.iterrows():
        data = item[1]
        list_coords.append({'data' : [{'x': [data['x']], 'y':[data['y']], 'type':'scattergl', 'text': data['Event'], 'name': data['Event'], 'mode':'lines+markers', 'marker': {'size' : 15, 'color': data['Colour']}}]})
        
    
    figure = {'data': list_coords[0]['data'],
              'layout': {'shapes' : lines,
                         'showlegend' :True,
                        'plot_bgcolor' :'MediumSeaGreen',
                  'xaxis': {'range': [0, 120], 'autorange': False,
                            'tickvals' :[10,20, 30, 40, 50, 60, 70, 80, 90, 100, 110],
                            'ticktext' : ['End Zone', 10, 20, 30, 40, 50, 40, 30, 20, 10, 'End Zone'],
                           'showgrid' : False,
                          'tickangle' : 45},
                  'yaxis': {'range': [0, 53], 'autorange': False, 'showgrid' : False},
                         'updatemenus': [{'type': 'buttons',
                                          'active':-1,  
                                          'buttons': [
                                              {
                                        'args': [None, {'frame': {'duration': 200, 'redraw': False}, 'mode': "immediate",
                                                 'fromcurrent': True, 'transition': {'duration': 200, 'easing': 'quadratic-in-out',}}],
                                        'label': 'Play',
                                        'method': 'animate',
                                        "execute": True,
                                    },
                                    {
                                        'args': [[None],{'frame': {'duration': 10, 'redraw': False}, 'mode': 'immediate',
                                        'transition': {'duration': 0}}],
                                        'label': 'Pause',
                                        'method': 'animate',
                                        "execute": True
                                    }
                                              ]}],
                         
                         'annotations' :[
                                            dict(
                                                x=5,
                                                y=25,
                                                showarrow=False,
                                                text='End Zone',
                                                textangle=-90,
                                                font=dict(
                                                    family='Overpass',
                                                    size=22,
                                                    color='yellow'
                                                ),
                                                xref='x',
                                                yref='y'
                                            ),
                                            dict(
                                                x=115,
                                                y=25,
                                                showarrow=False,
                                                text='End Zone',
                                                textangle=-90,
                                                font=dict(
                                                    family='Overpass',
                                                    size=22,
                                                    color='yellow'
                                                ),
                                                xref='x',
                                                yref='y'
                                            )],
                         'title': title},
              'frames': list_coords}
    return figure


def animate_play(data, title):
    %matplotlib inline

    #layout sourced from https://fcpython.com/visualisation/drawing-pitchmap-adding-lines-circles-matplotlib
    #pitch is 53 yards by 100 yards excluding two 10 yard touchdown zones.
    labels = ['Goal','10','20','30','40','50','40','30','20','10','Goal']
    fig = plt.figure(facecolor='white', figsize=(12.5,8))
    ax = fig.add_subplot(1,1,1)
    ax.set_facecolor('MediumSeaGreen')
    plt.xlim(-15, 115)
    plt.ylim(-5, 58)
    plt.yticks([]) # disable yticks
    
    start_x = -10
    bottom_y = 0
    top_y = 53
    
    ticks = [item * 10 for item in range(0,11)]
    #(x1,x2) (y1,y2)
    
    plt.plot([-10, 110],[0, 0], color='white', linewidth=4)
    plt.plot([-10, 110],[53, 53], color='white', linewidth=4)

    
    for item in range(0,28):
        if item == 0:
            plt.plot([start_x, start_x],[bottom_y, top_y], color='white', linewidth=4)
        
        if item >=1  and item <= 28:
            if item % 2 == 1:
                if item == 0 or item == 27:
                    plt.plot([start_x, start_x],[bottom_y, top_y], color='white', linewidth=4)
                    start_x = start_x + 5
                else:
                    plt.plot([start_x, start_x],[bottom_y, top_y], color='white', linestyle="dashed")
                    start_x = start_x + 5
                
            else:
                if start_x >=0 and start_x < 110:
                    plt.plot([start_x, start_x],[bottom_y, top_y], color='white', linewidth=4)
                    start_x = start_x + 5 
    
    
    data = home_data(data)
    data = visiting_data(data)
    data = check_winning(data)
    data['x'] = data['x'] - 10
    
    data = data.sort_values('Time')
    
    data = data.drop_duplicates(subset=['x','y','Event'])
   
    partners_data = partner_data(data)
    data = assign_matplot_colours(data)
    
    play = data.iloc[0] #select data range

    title = 'Home Team: ' + str(play['home_team']) + '. Visiting Team: ' + str(play['visiting_team']) + '.\n Home Score: ' + str(play['home_score']) + ' Visiting Team Score: ' + str(play['visit_score'])
    title = title + '.\n Possession Team: ' + str(play['Poss_Team']) + '. Possession Team Status: ' +  str(play['Poss_Team_Status'])
    plt.title(title, fontsize=20)
    
    x = play['x']
    y = play['y']
    label = play['Event'] # Event
    colour = play['Colour'] # Event colour
    p1 = plt.scatter(x=x, y=y, s=100, c=colour, label=label, alpha=0.4)
    
    if len(partners_data) > 0:
        
        partner_play = partners_data.iloc[0] #select data range
        partner_x = partner_play['x']
        partner_y = partner_play['y']
        p2 = plt.scatter(x=partner_x, y=partner_y, s=100, c='red', alpha=0.4)
        annotate = 'Primary Partner'
        plt.text(partner_x, partner_y, annotate, ha='left', size=12.5, color='black')
    
    annotate = 'Player'
    plt.text(x, y, annotate, ha='left', size=12.5, color='black')
    plt.legend(fontsize=15)
    
    
    def animate(i):
        if i > 1:
            play = data.iloc[int(i+1)] #select data range
            x = play['x']
            y = play['y']
                        
                
            if i+1 < len(partners_data):
                partner_play = partners_data.iloc[int(i+1)] #select data range
                partner_x = partner_play['x']
                partner_y = partner_play['y']
                
            label = play['Event']
            if label != 'General Play':
                colour = play['Colour']
                
                p1 = plt.scatter(x=x, y=y, s=100, c=colour, label=label, alpha=0.4)
                
                if i+1 < len(partners_data):
                    p2 = plt.scatter(x=partner_x, y=partner_y, s=100, c='red', alpha=0.4)
                    
                plt.legend(fontsize=15)
            else:
                colour = play['Colour']
                
                p1 = plt.scatter(x=x, y=y, s=100, c=colour, alpha=0.4)
                if i+1 < len(partners_data):
                    p2 = plt.scatter(x=partner_x, y=partner_y, s=100, c='red', alpha=0.4)

                
                plt.legend(fontsize=15)
                
        if i == 1:
            play = data.iloc[int(i+1)] #select data range
            x = play['x']
            y = play['y']
            
            label = play['Event']
            colour = play['Colour']
            p1 = plt.scatter(x=x, y=y, s=100, c=colour, label=label, alpha=0.4)
            if i+1 < len(partners_data):
                partner_play = partners_data.iloc[int(i+1)] #select data range
                partner_x = partner_play['x']
                partner_y = partner_play['y']
                p2 = plt.scatter(x=partner_x, y=partner_y, s=100, c='red', alpha=0.4)
                
            plt.legend(fontsize=15)
            
    
    plt.xticks(ticks, labels, size=15)
    ax.legend(bbox_to_anchor=(1.1, 1.05))
            
    myAnimation = animation.FuncAnimation(fig, animate, frames=len(data)-1, interval=100, repeat=False)
    
    return HTML(myAnimation.to_jshtml(fps=5)) 


def partner_data(data):
    temp = data[['Season_Year', 'GameKey', 'PlayID', 'Primary_Partner_GSISID', 'Season_Type', 'Week', 'Game_Date']]
    temp = temp.rename(columns={'Primary_Partner_GSISID': 'GSISID'})
    temp['GSISID'] = temp['GSISID'].astype(np.float32)
    
    usecols = None    
    
    if temp['Season_Type'].iloc[0] == 'Pre':
        NGS = get_NGS_data_pre()
        cols = list(temp.columns.intersection(NGS.columns))
        temp = temp.merge(NGS, on = cols)
            
    if temp['Season_Type'].iloc[0] == 'Reg':
        if temp['Season_Year'].iloc[0] == 2016:
            NGS = get_NGS_data_reg16(usecols)
            cols = list(temp.columns.intersection(NGS.columns))
            temp = temp.merge(NGS, on = cols)
        else:
            NGS = get_NGS_data_reg17(usecols)
            cols = list(temp.columns.intersection(NGS.columns))
            temp = temp.merge(NGS, on = cols)
           
    elif temp['Season_Type'].iloc[0] == 'Post':
        NGS = get_NGS_data_post()
        cols = list(temp.columns.intersection(NGS.columns))
        temp = temp.merge(NGS, on = cols)
            
    
    temp = temp.rename(columns={'GSISID':'Primary_Partner_GSISID'})
    temp['x'] = temp['x'] - 10 
    temp['Event'] = temp['Event'].replace(np.nan, 'General Play')
    temp = temp.sort_values('Time')
    temp = temp.drop_duplicates(subset=['x','y','Event'])

    return temp
merged_data = get_merged()
all_data = merged_data[merged_data['Role'] == 'P']
NGS = get_NGS_data_pre() #player position, direction etc.
NGS['x'] = NGS['x'] - 10
#NGS.drop_duplicates(subset=['x', 'y'], inplace=True)
columns = ['GameKey' , 'GSISID', 'PlayID']
temp = all_data[all_data['Season_Type'] == 'Pre']
punts = temp.merge(NGS, on= columns)

del(NGS)
gc.collect()

punts = punts.rename(columns = {'x':'X', 'y':'Y'})
punts = punts[(punts['X'] >= -10) | (punts['X'] <= 110)]
punts = punts[(punts['Y'] >= 0) | (punts['Y'] <= 53)]

punts_2016 = punts[punts['Season_Year_x'] == 2016]
punts_2017 = punts[punts['Season_Year_x'] == 2017]

levels = 50
draw_heatmap(punts_2016[['X', 'Y']], 'X', 'Y', 'Where were punts taken from in the 2016 Pre-Season?', levels)
draw_heatmap(punts_2017[['X', 'Y']], 'X', 'Y', 'Where were punts taken from in 2017 Pre-Season?', levels)

del(punts_2016)
del(punts_2017)
del(punts)
gc.collect()
print('')
usecols = ['Season_Year', 'GameKey', 'PlayID', 'GSISID', 'Time', 'x', 'y']
NGS = get_NGS_data_reg16(usecols) #player position, direction etc.
NGS['x'] = NGS['x'] - 10
#NGS.drop_duplicates(subset=['x', 'y'], inplace=True) 

columns = ['GameKey' , 'GSISID', 'PlayID']
temp = all_data[(all_data['Season_Type'] == 'Reg') & (all_data['Season_Year'] == 2016)]
punts = temp.merge(NGS, on= columns)

del(NGS)
gc.collect()

punts = punts.rename(columns = {'x':'X', 'y':'Y'})
punts = punts[(punts['X'] >= -10) | (punts['X'] <= 110)]
punts = punts[(punts['Y'] >= 0) | (punts['Y'] <= 53)]

punts = count_agg(['X', 'Y'], punts)
punts.sort_values(by='Count', inplace=True, ascending=False)

levels = 50
draw_heatmap(punts[['X', 'Y']], 'X', 'Y', 'Where were punts taken from in the 2016 Regular Season?', levels)

del(punts)
gc.collect()
print('')
usecols = ['Season_Year', 'GameKey', 'PlayID', 'GSISID', 'Time', 'x', 'y']
NGS = get_NGS_data_reg17(usecols) #player position, direction etc.
NGS['x'] = NGS['x'] - 10
#NGS.drop_duplicates(subset=['x', 'y'], inplace=True)

columns = ['GameKey' , 'GSISID', 'PlayID']
punts = all_data.merge(NGS, on= columns)

del(NGS)
gc.collect()

punts = punts.rename(columns = {'x':'X', 'y':'Y'})
punts = punts[(punts['X'] >= -10) | (punts['X'] <= 110)]
punts = punts[(punts['Y'] >= 0) | (punts['Y'] <= 53)]

punts = count_agg(['X', 'Y'], punts)
punts.sort_values(by='Count', inplace=True, ascending=False)

levels = 50
draw_heatmap(punts[['X', 'Y']], 'X', 'Y', 'Where were punts taken from in 2017 Regular Season?', levels)

del(punts)
gc.collect()
print('')
NGS = get_NGS_data_post() #player position, direction etc.
NGS['x'] = NGS['x'] - 10

columns = ['GameKey' , 'GSISID', 'PlayID']
punts = all_data.merge(NGS, on= columns)

del(NGS)
gc.collect()

punts = punts.rename(columns = {'x':'X', 'y':'Y'})
punts = punts[(punts['X'] >= -10) | (punts['X'] <= 110)]
punts = punts[(punts['Y'] >= 0) | (punts['Y'] <= 53)]

punts_2017 = punts[punts['Season_Year_x'] == 2017] 
punts_2016 = punts[punts['Season_Year_x'] == 2016] 

levels = 50
draw_heatmap(punts_2016[['X', 'Y']], 'X', 'Y', 'Where were punts taken from in the 2016 Post Season?', levels)
draw_heatmap(punts_2017[['X', 'Y']], 'X', 'Y', 'Where were punts taken from in 2017 Post Season?', levels)

del(punts_2016)
del(punts_2017)
del(punts)
gc.collect()
print('')
merged_data = get_merged()
all_data = merged_data[merged_data['Role'] == 'PR']
NGS = get_NGS_data_pre() #player position, direction etc.
NGS['x'] = NGS['x'] - 10
#NGS.drop_duplicates(subset=['x', 'y'], inplace=True)
columns = ['GameKey' , 'GSISID', 'PlayID']
temp = all_data[all_data['Season_Type'] == 'Pre']
punts = temp.merge(NGS, on= columns)

del(NGS)
gc.collect()

punts = punts.rename(columns = {'x':'X', 'y':'Y'})
punts = punts[(punts['X'] >= -10) | (punts['X'] <= 110)]
punts = punts[(punts['Y'] >= 0) | (punts['Y'] <= 53)]
punts = punts[punts['Event'] == 'punt_received']

punts_2016 = punts[punts['Season_Year_x'] == 2016]
punts_2017 = punts[punts['Season_Year_x'] == 2017]

levels = 50
draw_heatmap(punts_2016[['X', 'Y']], 'X', 'Y', 'Where were punts received in the 2016 Pre-Season?', levels)
draw_heatmap(punts_2017[['X', 'Y']], 'X', 'Y', 'Where were punts received in 2017 Pre-Season?', levels)

del(punts_2016)
del(punts_2017)
del(punts)
gc.collect()
print('')
usecols = ['Season_Year', 'GameKey', 'PlayID', 'GSISID', 'Time', 'x', 'y', 'Event']
NGS = get_NGS_data_reg16(usecols) #player position, direction etc.
NGS['x'] = NGS['x'] - 10
#NGS.drop_duplicates(subset=['x', 'y'], inplace=True)

columns = ['GameKey' , 'GSISID', 'PlayID']
temp = all_data[(all_data['Season_Type'] == 'Reg') & (all_data['Season_Year'] == 2016)]
punts = temp.merge(NGS, on= columns)

del(NGS)
gc.collect()

punts = punts.rename(columns = {'x':'X', 'y':'Y'})
punts = punts[(punts['X'] >= -10) | (punts['X'] <= 110)]
punts = punts[(punts['Y'] >= 0) | (punts['Y'] <= 53)]
punts = punts[punts['Event'] == 'punt_received']


punts = count_agg(['X', 'Y'], punts)
punts.sort_values(by='Count', inplace=True, ascending=False)

levels = 50
draw_heatmap(punts[['X', 'Y']], 'X', 'Y', 'Where were punts received in the 2016 Regular Season?', levels)

del(punts)
gc.collect()
print('')
usecols = ['Season_Year', 'GameKey', 'PlayID', 'GSISID', 'Time', 'x', 'y', 'Event']
NGS = get_NGS_data_reg17(usecols) #player position, direction etc.
NGS['x'] = NGS['x'] - 10
#NGS.drop_duplicates(subset=['x', 'y'], inplace=True)

columns = ['GameKey' , 'GSISID', 'PlayID']
punts = all_data.merge(NGS, on= columns)

del(NGS)
gc.collect()

punts = punts.rename(columns = {'x':'X', 'y':'Y'})
punts = punts[(punts['X'] >= -10) | (punts['X'] <= 110)]
punts = punts[(punts['Y'] >= 0) | (punts['Y'] <= 53)]
punts = punts[punts['Event'] == 'punt_received']


punts = count_agg(['X', 'Y'], punts)
punts.sort_values(by='Count', inplace=True, ascending=False)

levels = 50
draw_heatmap(punts[['X', 'Y']], 'X', 'Y', 'Where were punts received in 2017 Regular Season?', levels)

del(punts)
gc.collect()
print('')
NGS = get_NGS_data_post() #player position, direction etc.
NGS['x'] = NGS['x'] - 10

columns = ['GameKey' , 'GSISID', 'PlayID']
punts = all_data.merge(NGS, on= columns)

del(NGS)
gc.collect()

punts = punts.rename(columns = {'x':'X', 'y':'Y'})
punts = punts[(punts['X'] >= -10) | (punts['X'] <= 110)]
punts = punts[(punts['Y'] >= 0) | (punts['Y'] <= 53)]
punts = punts[punts['Event'] == 'punt_received']

punts_2017 = punts[punts['Season_Year_x'] == 2017] 
punts_2016 = punts[punts['Season_Year_x'] == 2016]


levels = 50
draw_heatmap(punts_2016[['X', 'Y']], 'X', 'Y', 'Where were punts received in the 2016 Post Season?', levels)
draw_heatmap(punts_2017[['X', 'Y']], 'X', 'Y', 'Where were punts received in 2017 Post Season?', levels)

del(punts_2016)
del(punts_2017)
del(punts)
gc.collect()
print('')
merged_data = get_merged()
temp = merged_data[merged_data['Role'] == 'PR']
punts_per_game = count_agg(['Play_Type', 'Season_Year', 'Season_Type'], temp)

x_data = 'Season_Type' 
y_data = 'Percentage'
hue = 'Season_Year'

pal = 'Greens_r'
xaxis_title = 'Football Season'
yaxis_title = 'Percentage'
title = 'Distribution of punts per football season'

generate_bar_plot(x_data, y_data, punts_per_game, pal, xaxis_title, yaxis_title, title, hue)
merged_data = get_merged()
all_data = merged_data[merged_data['Role'] == 'P']
temp = all_data
punts_per_game = count_agg(['Poss_Team', 'Play_Type', 'Season_Year'], temp)
punts_per_game['Season_Year'] = punts_per_game['Season_Year'].astype(str)

_2016_Data = punts_per_game[punts_per_game['Season_Year'] == '2016'].sort_values('Poss_Team', ascending=False)
_2017_Data = punts_per_game[punts_per_game['Season_Year'] == '2017'].sort_values('Poss_Team', ascending=False)
_2017_Data = _2017_Data[_2017_Data['Poss_Team'] != 'LAC']

x_data = 'Poss_Team'
y_data = 'Count'
xaxis = 'Team'
yaxis = 'Count of Punts Per Year'
title = 'Teams Utilising the Punt the most.'

names = ['2016', '2017']
plotly_line_graph([_2016_Data, _2017_Data], x_data, y_data, names, xaxis, yaxis, title)
merged_data = get_merged()
all_data = merged_data[merged_data['Role'] == 'PR']
temp = all_data
punts_per_game = count_agg(['Poss_Team', 'Play_Type', 'Season_Year'], temp)
punts_per_game['Season_Year'] = punts_per_game['Season_Year'].astype(str)

_2016_Data = punts_per_game[punts_per_game['Season_Year'] == '2016'].sort_values('Poss_Team', ascending=False)
_2017_Data = punts_per_game[punts_per_game['Season_Year'] == '2017'].sort_values('Poss_Team', ascending=False)
_2017_Data = _2017_Data[_2017_Data['Poss_Team'] != 'LAC']

x_data = 'Poss_Team'
y_data = 'Count'
xaxis = 'Team'
yaxis = 'Count of Punts Per Year'
title = 'Teams Receiving the most Punts.'

names = ['2016', '2017']
plotly_line_graph([_2016_Data, _2017_Data], x_data, y_data, names, xaxis, yaxis, title)
merged_data = get_merged()
all_data = merged_data[merged_data['Role'] == 'P']

punt_usage = all_data
punt_usage = punt_usage.dropna(subset=['Score_Home_Visiting']) # remove 4 rows containing nan in this column
punt_usage = home_data(punt_usage)
punt_usage = visiting_data(punt_usage)
punt_usage = check_winning(punt_usage)


punt_usage['Quarter'] = punt_usage['Quarter'].astype(np.int8)
punt_usage['Quarter'] = punt_usage['Quarter'].astype(str)
punt_usage['Quarter'] = punt_usage['Quarter'].replace('5.0', 'Extra Time')


punt_usage = count_agg(['Quarter', 'Poss_Team_Status'], punt_usage)

x_data = 'Quarter'
y_data = 'Count'
categories = pd.unique(punt_usage['Poss_Team_Status'])
x_title = 'Quarter'
y_title = 'Count'
title = 'When are punt plays predominantly used and whats the Possession teams status?'
temp = []
data = []
for row in punt_usage.iterrows():
    for cat in categories:
        item = row[1]
        temp = punt_usage[punt_usage['Poss_Team_Status'] == cat]
        data.append(temp)
        temp = []
    
plotly_stacked_bar(data, categories, x_data, y_data, x_title, y_title, title)
merged_data = get_merged()
all_data = merged_data[merged_data['Role'] == 'P']
punts_per_game = count_agg(['Season_Type', 'Week'], all_data)
punts_per_game['Week'] = punts_per_game['Week'].astype(str)
punts_per_game


x_data = 'Week'
y_data = 'Count'
categories = pd.unique(punts_per_game['Season_Type'])
x_title = 'Week'
y_title = 'Count'
title = 'Weekly punt distribution per Season Type'
temp = []
data = []
for row in punts_per_game.iterrows():
    for cat in categories:
        item = row[1]
        temp = punts_per_game[punts_per_game['Season_Type'] == cat]
        data.append(temp)
        temp = []
    
plotly_stacked_bar(data, categories, x_data, y_data, x_title, y_title, title)
all_data['Game_Clock'] = all_data['Game_Clock'].astype(str)
all_data['Game_Clock_Minute'] = all_data['Game_Clock'].apply(lambda x: x.split(':')[0])

punts_per_min = count_agg(['Quarter', 'Game_Clock_Minute'], all_data)

punts_per_min


x_data = 'Game_Clock_Minute'
y_data = 'Count'
categories = pd.unique(punts_per_min['Quarter'])
x_title = 'Game Clock Minute'
y_title = 'Count'
title = 'Game Clock Minute at time of Punt per Quarter'
temp = []
data = []
for row in punts_per_min.iterrows():
    for cat in categories:
        item = row[1]
        temp = punts_per_min[punts_per_min['Quarter'] == cat]
        data.append(temp)
        temp = []
    
plotly_stacked_bar(data, categories, x_data, y_data, x_title, y_title, title)
game = game_data() #general game data, date, time, location etc.
video_data = vid_review_data() #concussion data, how it happened etc.
role = player_role_data() #players role during the pun
punt = player_punt_data() #player postion
injury = vid_injury_data()
play = play_info_data()

injury = injury.rename(columns={'gamekey':'GameKey', 'playid':'PlayID'})
print('The number of concussion observations is: ', len(video_data))
video_data.head()
columns = ['Player_Activity_Derived']
data = count_agg(columns, video_data)

x_data = 'Player_Activity_Derived' 
y_data = 'Percentage'
hue = 'Player_Activity_Derived'

pal = 'Greens_r'
xaxis_title = 'Player Activity'
yaxis_title = 'Percentage'
title = 'Activities Causing Concussion'


generate_bar_plot(x_data, y_data, data, pal, xaxis_title, yaxis_title, title, hue)


columns = ['Primary_Impact_Type']
data = count_agg(columns, video_data)

x_data = 'Primary_Impact_Type' 
y_data = 'Percentage'
hue = 'Primary_Impact_Type'

pal = 'Oranges_r'
xaxis_title = 'Type of Impact'
yaxis_title = 'Percentage'
title = 'Impacts Causing Concussion'

generate_bar_plot(x_data, y_data, data, pal, xaxis_title, yaxis_title, title, hue)
data = video_data[['Player_Activity_Derived', 'Primary_Partner_Activity_Derived']]
result = generate_table(data)

iplot(result[0], result[1])
temp = video_data[video_data['Primary_Partner_Activity_Derived'].isnull()]
temp
temp = video_data[video_data['Primary_Impact_Type'] == 'Helmet-to-ground']
temp
temp = video_data[video_data['Player_Activity_Derived'] == video_data['Primary_Partner_Activity_Derived']]
print('The number of observations where player and primary partner activities are the same is: ', len(temp))
temp
columns = ['Primary_Impact_Type', 'Player_Activity_Derived']
data = count_agg(columns, video_data)

x_data = 'Player_Activity_Derived' 
y_data = 'Percentage'
hue = 'Primary_Impact_Type'

pal = 'viridis'
xaxis_title = 'Player Activity'
yaxis_title = 'Percentage'
title = 'Player Activity by Impact Type'

generate_bar_plot(x_data, y_data, data, pal, xaxis_title, yaxis_title, title, hue)

columns = ['Primary_Impact_Type', 'Primary_Partner_Activity_Derived']
primary = count_agg(columns, video_data)

x_data = 'Primary_Partner_Activity_Derived' 
y_data = 'Percentage'
hue = 'Primary_Impact_Type'

pal = 'viridis'
xaxis_title = 'Primary Partner Activity'
yaxis_title = 'Percentage'
title = 'Primary Partner Activity by Impact Type'

generate_bar_plot(x_data, y_data, primary, pal, xaxis_title, yaxis_title, title, hue)
columns = ['GameKey']
play_counts = count_agg(columns, video_data)
print('Number of games with more than one concussion: ', len(play_counts[play_counts['Count'] >= 2]))
play_counts[play_counts['Count'] >= 2]
columns = ['GameKey', 'GSISID' ,'PlayID']
concussion_role = perform_merge(video_data, role, columns)

columns = ['Role', 'Player_Activity_Derived']
data = count_agg(columns, concussion_role)
data
x = 'Role'
y = 'Percentage'
size = 'Percentage'
categories = 'Role'


colour = None
xaxis_title = 'Player Role'
yaxis_title = 'Percentage of Concussions'
title = 'Player Role vs. Percentage of Concussions'

bubble_plot(data, x, y, colour, xaxis_title, yaxis_title, title, size, categories)  

missing = missing_data(game)
data = missing[['Column', 'Percentage_Observations_Missing']]
result = generate_table(data)

iplot(result[0], result[1])
punt_returners = concussion_role[concussion_role['Role'] == 'PR']

columns = ['GameKey']
punt_returners = perform_merge(punt_returners, game, columns)

columns = ['Season_Type']
PR_season = count_agg(columns, punt_returners)

data = PR_season[['Season_Type', 'Percentage']]
result = generate_table(data)

iplot(result[0], result[1])

columns = ['Season_Year']
PR_season = count_agg(columns, punt_returners)

data = PR_season[['Season_Year', 'Percentage']]
result = generate_table(data)

iplot(result[0], result[1])
columns = ['Player_Activity_Derived', 'Primary_Impact_Type']
PR_Concussions = count_agg(columns, punt_returners)

x_data = 'Player_Activity_Derived' 
y_data = 'Percentage'
hue = 'Primary_Impact_Type'

pal = 'viridis'
xaxis_title = 'Player Activity'
yaxis_title = 'Percentage'
title = 'Punt Returner Activity and Impacts Causing Concussion'

generate_bar_plot(x_data, y_data, PR_Concussions, pal, xaxis_title, yaxis_title, title, hue)
data = punt_returners[['Primary_Impact_Type', 'Week']]
result = generate_table(data.sort_values(by=['Week']))

iplot(result[0], result[1])
columns = ['GameKey', 'PlayID']
PR = perform_merge(punt_returners, play, columns)
PR = visiting_data(PR)
PR = home_data(PR)
PR = score_difference(PR)

data = PR[['Quarter','score_diff' ]]

x_data = 'Quarter'
y_data = 'score_diff'
mode='markers'
categories=''
xaxis_title = 'Score defecit'
yaxis_title = 'Quarter'
title = 'Score defecit vs Quarter'

plotly_scatter(y_data, x_data, mode, categories, xaxis_title, yaxis_title, title, data)
columns = ['Quarter']
PR_quarter = count_agg(columns, PR)

y = 'Quarter'
x = 'Count'
colour = 'Spectral'
title = 'What Quarter Punt Returners Receive Concussions'

pie_chart(PR_quarter, x, y, colour, title)
columns = PR.columns
PR = get_NGS_data(PR)
grouped = group_interactive_data(PR)
dropdown = []
dataframes = []

for key, value in grouped:

    key = str(key).strip('(').strip(')').split(',')
    temp = 'GameKey: ' + str(key[0]) #+ '  GSISID: ' + str(key[1]) + '  PlayID: ' + str(key[2])
    dropdown.append(temp)
    dataframes.append(value)

result = gen_interactive_plot(dropdown, dataframes, 'x', 'y')
iplot(result[0], result[1])
dataframes[0] = dataframes[0].reset_index(drop=True)
animated_play = animate_play(dataframes[0],'')
plt.close()
animated_play
print('Player Activity: ' , dataframes[0]['Player_Activity_Derived'][0])
print('Player Role: ' , dataframes[0]['Role'][0])
print('Impact: ' , dataframes[0]['Primary_Impact_Type'][0])
dataframes[1] = dataframes[1].reset_index(drop=True)
animated_play = animate_play(dataframes[1],'')
plt.close()
animated_play
print('Player Activity: ' , dataframes[1]['Player_Activity_Derived'][0])
print('Player Role: ' , dataframes[1]['Role'][0])
print('Impact: ' , dataframes[1]['Primary_Impact_Type'][0])
dataframes[2] = dataframes[2].reset_index(drop=True)
animated_play = animate_play(dataframes[2],'')
plt.close()
animated_play
print('Player Activity: ' , dataframes[2]['Player_Activity_Derived'][0])
print('Player Role: ' , dataframes[2]['Role'][0])
print('Impact: ' , dataframes[2]['Primary_Impact_Type'][0])
dataframes[3] = dataframes[3].reset_index(drop=True)
animated_play = animate_play(dataframes[3],'')
plt.close()
animated_play
print('Player Activity: ' , dataframes[3]['Player_Activity_Derived'][0])
print('Player Role: ' , dataframes[3]['Role'][0])
print('Impact: ' , dataframes[3]['Primary_Impact_Type'][0])
dataframes[4] = dataframes[4].reset_index(drop=True)
animated_play = animate_play(dataframes[4],'')
plt.close()
animated_play
print('Player Activity: ' , dataframes[4]['Player_Activity_Derived'][0])
print('Player Role: ' , dataframes[4]['Role'][0])
print('Impact: ' , dataframes[4]['Primary_Impact_Type'][0])
HTML('<video width="560" height="315" controls> <source src="https://nfl-vod.cdn.anvato.net/league/5691/18/11/25/284956/284956_12D27120C06E4DB994040750FB43991D_181125_284956_way_punt_3200.mp4" type="video/mp4"></video>')

prlg = concussion_role[(concussion_role['Role'] == 'PRG') | (concussion_role['Role'] == 'PLG')]

columns = ['GameKey']
prlg = perform_merge(prlg, game, columns)

columns = ['Role', 'Season_Type']
PRLG_season = count_agg(columns, prlg)

data = PRLG_season[['Role', 'Season_Type', 'Count']]
generate_bar_plot('Role', 'Count', data, 'viridis', 'Player Role', 'Count', 'Number of Concussions per Role by Season Type', 'Season_Type')


columns = ['Role', 'Season_Year']
PRLG_season = count_agg(columns, prlg)

data = PRLG_season[['Role', 'Season_Year', 'Count']]
generate_bar_plot('Role', 'Count', data, 'viridis', 'Player Role', 'Count', 'Number of Concussions per Role by Year', 'Season_Year')

data = prlg[['Role','Primary_Impact_Type']]
columns = ['Role','Primary_Impact_Type']
data = count_agg(columns, data)

y = 'Primary_Impact_Type'
x = 'Count'
colour = 'Spectral'
title = "What Impacts are causing PRG's and PLG's to get concussions?"

pie_chart(data, x, y, colour, title)

data = prlg[['Role','Player_Activity_Derived', 'Primary_Impact_Type']]
columns = ['Role','Player_Activity_Derived', 'Primary_Impact_Type']
data = count_agg(columns, data)

generate_bar_plot('Player_Activity_Derived', 'Percentage', data, 'viridis', 'Player Activity', 'Percentage of Concussions', 'Percentage of Concussions per Player Activity coloured by Impact Type', 'Primary_Impact_Type')

data = prlg[['Role','Primary_Impact_Type']]
columns = ['Role','Primary_Impact_Type']
data = count_agg(columns, data)

data = data[data['Role'] == 'PRG']

y = 'Primary_Impact_Type'
x = 'Count'
colour = 'PiYG'
title = "What Impacts are causing PRG's to get concussions?"

pie_chart(data, x, y, colour, title)


data = prlg[['Role','Primary_Impact_Type']]
columns = ['Role','Primary_Impact_Type']
data = count_agg(columns, data)

data = data[data['Role'] == 'PLG']

y = 'Primary_Impact_Type'
x = 'Count'
colour = 'Spectral'
title = "What Impacts are causing PLG's to get concussions?"

pie_chart(data, x, y, colour, title)
data = prlg[['Role','Week','Season_Type']]
data = data[data['Season_Type'] == 'Reg']

result = generate_table3(data)
iplot(result[0], result[1])
play = play_info_data()
columns = ['GameKey', 'PlayID']
prlg = perform_merge(prlg, play, columns)

columns = prlg.columns
prlg = get_NGS_data(prlg)
grouped = group_interactive_data(prlg)
dropdown = []
dataframes = []

for key, value in grouped:

    key = str(key).strip('(').strip(')').split(',')
    temp = 'GameKey: ' + str(key[0]) #+ '  GSISID: ' + str(key[1]) + '  PlayID: ' + str(key[2])
    dropdown.append(temp)
    dataframes.append(value)

result = gen_interactive_plot(dropdown, dataframes, 'x', 'y')
iplot(result[0], result[1]) 
dataframes[0] = dataframes[0].reset_index(drop=True)
animated_play = animate_play(dataframes[0],'')
plt.close()
animated_play
print('Player Activity: ' , dataframes[0]['Player_Activity_Derived'][0])
print('Player Role: ' , dataframes[0]['Role'][0])
print('Impact: ' , dataframes[0]['Primary_Impact_Type'][0])
dataframes[1] = dataframes[1].reset_index(drop=True)
animated_play = animate_play(dataframes[1],'')
plt.close()
animated_play
print('Player Activity: ' , dataframes[1]['Player_Activity_Derived'][0])
print('Player Role: ' , dataframes[1]['Role'][0])
print('Impact: ' , dataframes[1]['Primary_Impact_Type'][0])
dataframes[2] = dataframes[2].reset_index(drop=True)
animated_play = animate_play(dataframes[2],'')
plt.close()
animated_play
print('Player Activity: ' , dataframes[2]['Player_Activity_Derived'][0])
print('Player Role: ' , dataframes[2]['Role'][0])
print('Impact: ' , dataframes[2]['Primary_Impact_Type'][0])
dataframes[3] = dataframes[3].reset_index(drop=True)
animated_play = animate_play(dataframes[3],'')
plt.close()
animated_play
print('Player Activity: ' , dataframes[3]['Player_Activity_Derived'][0])
print('Player Role: ' , dataframes[3]['Role'][0])
print('Impact: ' , dataframes[3]['Primary_Impact_Type'][0])
dataframes[4] = dataframes[4].reset_index(drop=True)
animated_play = animate_play(dataframes[4],'')
plt.close()
animated_play
print('Player Activity: ' , dataframes[4]['Player_Activity_Derived'][0])
print('Player Role: ' , dataframes[4]['Role'][0])
print('Impact: ' , dataframes[4]['Primary_Impact_Type'][0])
dataframes[5] = dataframes[5].reset_index(drop=True)
animated_play = animate_play(dataframes[5],'')
plt.close()
animated_play
print('Player Activity: ' , dataframes[5]['Player_Activity_Derived'][0])
print('Player Role: ' , dataframes[5]['Role'][0])
print('Impact: ' , dataframes[5]['Primary_Impact_Type'][0])
dataframes[6] = dataframes[6].reset_index(drop=True)
animated_play = animate_play(dataframes[6],'')
plt.close()
animated_play
print('Player Activity: ' , dataframes[6]['Player_Activity_Derived'][0])
print('Player Role: ' , dataframes[6]['Role'][0])
print('Impact: ' , dataframes[6]['Primary_Impact_Type'][0])
dataframes[7] = dataframes[7].reset_index(drop=True)
animated_play = animate_play(dataframes[7],'')
plt.close()
animated_play
print('Player Activity: ' , dataframes[7]['Player_Activity_Derived'][0])
print('Player Role: ' , dataframes[7]['Role'][0])
print('Impact: ' , dataframes[7]['Primary_Impact_Type'][0])
HTML('<video width="560" height="315" controls> <source src="https://nfl-vod.cdn.anvato.net/league/5691/18/11/25/284954/284954_75F12432BA90408C92660A696C1A12C8_181125_284954_huber_punt_3200.mp4" type="video/mp4"></video>')









