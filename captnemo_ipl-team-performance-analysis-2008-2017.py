import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

%matplotlib notebook

teams = pd.Series([])
for i in range(1,9):
    _ = '../input/ipl-team-wise-performance20082017/'+str(i)+'.csv'
    temp = pd.read_csv(_)
    teams[str(i)] = temp[['season', 'team1', 'team2', 'winner']]

teams['1'].head()
seasons = list()
for i in range(2008,2018):
    seasons.append(str(i))
seasons
q = teams['1']
q.head()
type(q['season'][0])
team_names = list()
def create_list(x):
    team_names.append(x['team1'][0])
    
teams.apply(lambda x:(create_list(x)))

team_names
def update(x,wins):
#     print(x)
    if (x['team1'] == x['winner']):
        _[str(x['season'])] += 1

win = list()
for i in teams:
    i = i.T
    _ = pd.Series([0]*len(seasons),index = seasons)
    i.apply(lambda x:(update(x,_)))
    win.append(_)
win = pd.DataFrame(win, index = team_names)
win
print(win[str(2008)])
%matplotlib notebook

def plotPie(clickedSeason):
    # Data to plot
    plt.subplot(2, 1, 2)
    plt.cla()
    plt.title('Maximum Wins in Selected Season: {}'.format(clickedSeason), fontsize=16)
    selected_Season = win[str(clickedSeason)].copy()
    labels = team_names
    explode = [0]*len(team_names)  # explode 1st slice

    adj = list()
    F_labels = list()

    for i in range(len(selected_Season)):
        if(selected_Season[i] != 0):
            adj.append(selected_Season[i])
            F_labels.append(labels[i])
        else:
            explode.pop()
            
    array_for_pie = np.asarray(adj)
    sizes = array_for_pie
    total = sum(sizes)
    
    k = adj.index(max(adj))
    explode[k] = 0.2

    # Plot
    plt.pie(sizes, explode=explode, labels=F_labels, autopct=lambda p: '{:.0f}'.format(p * total / 100), shadow=True, startangle=140)

    plt.axis('equal')
    plt.show()
plt.figure(figsize=(9, 12)).suptitle("IPL Match Database Analysis", fontsize = 20)

def plotBeautify(ssn,aim):
    plt.subplot(2, 1, 1)
    plt.cla()
    tm_names = []
    lwid = [0.1]*len(team_names)
    if(ssn != 0 and aim != 0):
        scores = list(win[str(seasons[ssn])])
        trgt = list()
        for i in range(len(scores)):
            if scores[i] == aim:
                trgt.append(i)
        for i in trgt:
            lwid[i] = 2.0
            tm_names.append(team_names[i])

    for i in range(len(win)):
        plt.plot(seasons, win.iloc[i], '-o', label=str(team_names[i]), linewidth = lwid[i])
    plt.xlabel('SEASON', fontsize=12)
    plt.ylabel('WINS', fontsize=12)
    plt.ylim(-0.5, max(win.apply(lambda k:max(k)))+1)
    plt.legend(loc = 3, ncol=3, shadow = False, fontsize = 'xx-small')
    plt.title("{} in Season {}".format(', '.join(tm_names), seasons[ssn]))
    
plotBeautify(0,0)

def onclick(event):
    plt.cla()
    plotBeautify(int(event.xdata+0.5),int(event.ydata+0.5))
    plt.plot([int(event.xdata + 0.5), int(event.xdata + 0.5)], [16, -0.5], color='red')
    plt.gca().text(8.6, 13, 'Season : {}\nSeason wins : {}/14\n'.format(seasons[int(event.xdata + 0.5)], int(event.ydata + 0.5)),
                  bbox={'facecolor':'w','pad':5},ha="right", va="top")
    plotPie(seasons[int(event.xdata + 0.5)])
    
#tell mpl_connect we want to pass a 'button_press_event' into onclick when the event is detected
cid = plt.gcf().canvas.mpl_connect('button_press_event', onclick)

cid

#If interactivity not working, re-run the first cell and then re-run the cell with def PlotPie and downwards

