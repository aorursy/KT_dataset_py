%matplotlib inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv("../input/leagueoflegends/LeagueofLegends.csv")
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
names = ['Blue Wins', 'Red Wins']
values = [data.bResult.sum(),data.rResult.sum()]
bar = ax.bar(names,values)
bar[1].set_color('r')
plt.show()
number_of_top_champs = 20
red_top_most_picked_champions = data.groupby(["redTopChamp"]).size().nlargest(number_of_top_champs)
red_jungle_most_picked_champions = data.groupby(["redJungleChamp"]).size().nlargest(number_of_top_champs)
red_mid_most_picked_champions = data.groupby(["redMiddleChamp"]).size().nlargest(number_of_top_champs)
red_adc_most_picked_champions = data.groupby(["redADCChamp"]).size().nlargest(number_of_top_champs)
red_sup_most_picked_champions = data.groupby(["redSupportChamp"]).size().nlargest(number_of_top_champs)

blue_top_most_picked_champions = data.groupby(["blueTopChamp"]).size().nlargest(number_of_top_champs)
blue_jungle_most_picked_champions = data.groupby(["blueJungleChamp"]).size().nlargest(number_of_top_champs)
blue_mid_most_picked_champions = data.groupby(["blueMiddleChamp"]).size().nlargest(number_of_top_champs)
blue_adc_most_picked_champions = data.groupby(["blueADCChamp"]).size().nlargest(number_of_top_champs)
blue_sup_most_picked_champions = data.groupby(["blueSupportChamp"]).size().nlargest(number_of_top_champs)


data_top_champions = pd.DataFrame({"Blue Top Picks":blue_top_most_picked_champions,"Red Top Picks": red_top_most_picked_champions})
data_jungle_champions = pd.DataFrame({"Blue Jungle Picks":blue_jungle_most_picked_champions,"Red Jungle Picks": red_jungle_most_picked_champions})
data_mid_champions = pd.DataFrame({"Blue Mid Picks":blue_mid_most_picked_champions,"Red Mid Picks": red_mid_most_picked_champions})
data_adc_champions = pd.DataFrame({"Blue ADC Picks":blue_adc_most_picked_champions,"Red ADC Picks": red_adc_most_picked_champions})
data_sup_champions = pd.DataFrame({"Blue Sup Picks":blue_sup_most_picked_champions,"Red Sup Picks": red_sup_most_picked_champions})
data_top_champions.plot.bar(color={"red","blue"})
data_jungle_champions.plot.bar(color={"red","blue"})
data_mid_champions.plot.bar(color={"red","blue"})
data_adc_champions.plot.bar(color={"red","blue"})
data_sup_champions.plot.bar(color={"red","blue"})
killData = pd.read_csv("../input/leagueoflegends/kills.csv")
killData = killData.round({'Time':0})
blueKills = killData[killData.Team == 'bKills']
redKills = killData[killData.Team == 'rKills']
blueKills = blueKills[blueKills.Time.notnull()]
redKills = redKills[redKills.Time.notnull()]


blueKills = blueKills.groupby("Time").size()
redKills = redKills.groupby("Time").size()
deaths_data = pd.DataFrame({"Blue Kills":blueKills,"Red Kills": redKills})
deaths_data.plot.line(color={"red","blue"})
goldData = pd.read_csv("../input/leagueoflegends/gold.csv")
# Create minute column names for pd.melt()
minutes = ['min_' + str(x + 1) for x in range(81)]
goldData = pd.melt(goldData, id_vars=['Address', 'Type'], value_vars=minutes, 
                   var_name='minute', value_name='gold')
# Changet the minute variable into a integer.
goldData.minute = goldData.minute.str.strip('min_').astype(int)
goldData.head()

sections = ['Top', 'Jungle', 'Middle', 'ADC', 'Support']
num_sections = len(sections)
for i in range(num_sections):
    plt.figure(i)
    plt.plot(goldData[goldData.Type == 'goldblue' + sections[i]].groupby('minute').gold.mean(), 'b-')
    plt.plot(goldData[goldData.Type == 'goldred' + sections[i]].groupby('minute').gold.mean(), 'r-')
    plt.xlabel('Minute')
    plt.ylabel('Gold')
    plt.title(sections[i])
monsterData = pd.read_csv("../input/leagueoflegends/monsters.csv")
monsterData = monsterData.round({'Time':0})
sections = ["Dragons","Barons","Heralds"]
num_sections = len(sections)
for i in range(num_sections):
    plt.figure(i)
    plt.plot(monsterData[monsterData.Team == 'b' + sections[i]].groupby('Time').size(), 'b-')
    plt.plot(monsterData[monsterData.Team == 'r' + sections[i]].groupby('Time').size(), 'r-')
    plt.xlabel('Minute')
    plt.ylabel(sections[i])
    plt.title(sections[i])
structure_data = pd.read_csv("../input/leagueoflegends/structures.csv")
structure_data = structure_data.round({'Time':0})
sections = ["Towers","Inhibs"]
num_sections = len(sections)
for i in range(num_sections):
    plt.figure(i)
    plt.plot(structure_data[structure_data.Team == 'b' + sections[i]].groupby('Time').size(), 'b-')
    plt.plot(structure_data[structure_data.Team == 'r' + sections[i]].groupby('Time').size(), 'r-')
    plt.xlabel('Minute')
    plt.ylabel(sections[i])
    plt.title(sections[i])
