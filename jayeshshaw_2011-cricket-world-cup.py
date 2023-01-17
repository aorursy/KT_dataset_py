import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
mr = pd.read_excel("../input/mostruns/Most Runs.xlsx")
mr
plt.figure(figsize=(15,10))
sns.barplot(mr["Player"],mr["RUNS"]);
mw = pd.read_excel("../input/mostwickets/Most Wickets.xlsx")
mw
plt.figure(figsize=(15,10))
sns.barplot(mw["Player"],mw["Wickets"]);
hs = pd.read_excel("../input/highestscores/Highest Scores.xlsx")
hs
plt.figure(figsize=(15,10))
sns.barplot(hs["Player "],hs["Score "]);
sc = pd.read_excel("../input/stadiumcapacity/Stadium CAPACITY.xlsx")
sc
plt.figure(figsize=(15,10))
sns.barplot(sc["Place"],sc["Capacity"]);
ga = pd.read_excel("../input/grpastandings/Group A Standings.xlsx")
ga
gas = pd.read_excel("../input/grpastatistics/Group Matches.xlsx")
gas
gas["First Innings Score"].describe()
density = np.array([gas["First Innings Score"]])


sns.set(rc={'figure.figsize':(15.7,6.27)})
plt.title("Density Distribuition of 1st innings scores")
sns.distplot(density, hist=True, rug=True);
#Run rate of 1st innings
gas["run_rate_1"] = gas["First Innings Score"]/gas["Overs played 1"]
gas
gas["run_rate_1"].describe()
plt.figure(figsize=(15,10))
sns.barplot(gas["Stadium"],gas["run_rate_1"]);
density = np.array([gas["run_rate_1"]])


sns.set(rc={'figure.figsize':(15,6.27)})
plt.title("Density Distribuition of 1st innings run_rates")
sns.distplot(density, hist=True, rug=True);
gas["Wickets fell(1st Innings)"].describe()
plt.figure(figsize=(22,10))
plt.title("Graph of wickets fell in 1st innings")
sns.barplot(gas["Match no."],gas["Wickets fell(1st Innings)"]);
gas["Overs played 1"].describe()
plt.figure(figsize=(22,10))
plt.title("Graph of no. of overs played in 1st innings")
sns.barplot(gas["Match no."],gas["Overs played 1"]);
gas["Second Innings Score"].describe()
density = np.array([gas["Second Innings Score"]])


sns.set(rc={'figure.figsize':(15.7,6.27)})
plt.title("Density Distribuition of 2nd innings scores")
sns.distplot(density, hist=True, rug=True);
#Run rate of 2nd innings
gas["run_rate_2"] = gas["Second Innings Score"]/gas["Overs played 2"]
gas
gas["run_rate_2"].describe()
plt.figure(figsize=(15,10))
sns.barplot(gas["Stadium"],gas["run_rate_2"]);
gas["Wickets fell(2nd Innings)"].describe()
plt.figure(figsize=(22,10))
plt.title("Graph of wickets fell in 2nd innings")
sns.barplot(gas["Match no."],gas["Wickets fell(2nd Innings)"]);
gas["Overs played 2"].describe()
ta = pd.read_excel("../input/groupatossdecisions/Grp A toss.xlsx")
ta
import matplotlib.pyplot as plt

# Pie chart, where the slices will be ordered and plotted counter-clockwise:
labels = "Yes", "No"
sizes = [100-52.38, 52.38]
explode = (0, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')

fig1, ax1 = plt.subplots()
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.show()
gbp = pd.read_excel("../input/grpbstandings/Group B Standings.xlsx")
gbp
gbs = pd.read_excel("../input/grpbmatches/Group B matches.xlsx")
gbs
gbs["1st innings score"].describe()
print(np.median(gbs["1st innings score"]))
density = np.array([gbs["1st innings score"]])


sns.set(rc={'figure.figsize':(15.7,6.27)})
plt.title("Density Distribuition of 1st innings scores")
sns.distplot(density, hist=True, rug=True);
gbs["run_rate_1"] = gbs["1st innings score"]/gbs["Overs played 1"]
gbs
gbs["run_rate_1"].describe()
print(np.median(gbs["run_rate_1"]))
plt.figure(figsize=(15,10))
sns.barplot(gbs["Stadium"],gbs["run_rate_1"]);
density = np.array([gbs["run_rate_1"]])


sns.set(rc={'figure.figsize':(15,6.27)})
plt.title("Density Distribuition of 1st innings run_rates")
sns.distplot(density, hist=True, rug=True);
gbs["Wickets fell_1"].describe()
plt.figure(figsize=(22,10))
plt.title("Graph of wickets fell in 1st innings")
sns.barplot(gbs["Match no."],gbs["Wickets fell_1"]);
gbs["Overs played 1"].describe()
print(np.median(gbs["Overs played 1"]))
plt.figure(figsize=(22,10))
plt.title("Graph of no. of overs played in 1st innings")
sns.barplot(gbs["Match no."],gbs["Overs played 1"]);
gbs["2nd innings score"].describe()
density = np.array([gbs["2nd innings score"]])


sns.set(rc={'figure.figsize':(15,6.27)})
plt.title("Density Distribuition of 2nd innings scores")
sns.distplot(density, hist=True, rug=True);
#Run rate of 2nd innings
gbs["run_rate_2"] = gbs["2nd innings score"]/gbs["Overs played 2"]
gbs
gbs["run_rate_2"].describe()
plt.figure(figsize=(15,10))
sns.barplot(gbs['Stadium'],gbs["run_rate_2"]);
gbs["Wickets fell_2"].describe()
plt.figure(figsize=(25,10))
plt.title('Wickets fell in each match')
sns.barplot(gbs['Match no.'],gbs["Wickets fell_2"]);
gbs["Overs played 2"].describe()
gbt = pd.read_excel("../input/grpbtoss/Group B toss.xlsx")
gbt

import matplotlib.pyplot as plt

# Pie chart, where the slices will be ordered and plotted counter-clockwise:
labels = "Yes", "No"
sizes = [100-42.85, 42.85]
explode = (0, 0)  

fig1, ax1 = plt.subplots()
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.show()
qpm = pd.read_excel("../input/qfstandings/Quarter finals.xlsx")
qpm
qfm = pd.read_excel("../input/qfmatches/QF-Matches.xlsx")
qfm
plt.figure(figsize=(10,10))
plt.title("Runs scored in 1st innings")
sns.barplot(qfm['Match'],qfm[" Runs scored 1"]);
plt.figure(figsize=(10,10))
plt.title("Runs scored in 2nd innings")
sns.barplot(qfm['Match'],qfm["Runs scored 2"]);
ns = pd.read_excel("../input/nzvssl/NZ vs SL.xlsx")
ns.dropna()
plt.figure(figsize=(35,10))
plt.title("Performance of Batmsen")
sns.barplot(ns['Batsmen'],ns["Runs"]);
ns1 = pd.read_excel("../input/sf1bowl/NZ vs SL Bowl.xlsx")

ns1["econ"] = ns1["Runs"]/ns1["Overs"]
ns1
plt.figure(figsize=(35,10))
plt.title("Runs given by the bowlers")
sns.barplot(ns1['Bowler'],ns1["Runs"]);
plt.figure(figsize=(35,10))
plt.title("Economy of the bowlers")
sns.barplot(ns1['Bowler'],ns1["econ"]);
ns2 = pd.read_excel("../input/sf1-bat-1/SF IND BAT.xlsx")
ns2
plt.figure(figsize=(25,10))
plt.title("Runs scored by the batsman")
sns.barplot(ns2['Batsman'],ns2["Runs"]);
ns3 = pd.read_excel("../input/sf-2-2/SF PAK BOWL.xlsx")
ns3
ns4 = pd.read_excel("../input/sfbat2/SF PAK BAT.xlsx")
ns4
plt.figure(figsize=(25,10))
plt.title("Runs scored by the batsman")
sns.barplot(ns4['Player'],ns4["Runs"]);
ns5 = pd.read_excel("../input/indbowl/SF-IND-BOWL.xlsx")
ns5["econ"] = ns5["Runs"]/ns5["Overs"]
ns5
slf = pd.read_excel("../input/sl-dat-final/SL BAT FINAL.xlsx")
slf["SR"] = slf["Runs"]/slf["Balls"]*100
slf
plt.figure(figsize=(25,10))
plt.title("Runs scored by the batsman")
sns.barplot(slf['Player'],slf["Runs"]);
ibowl = pd.read_excel("../input/i-bowl/IND BOWL.xlsx")
ibowl["econ"] = ibowl["Runs"]/ibowl["Overs"]
ibowl
fb = pd.read_excel("../input/indbat/IND BAT FINAL.xlsx")
fb
plt.figure(figsize=(25,10))
plt.title("Runs scored by the batsman")
sns.barplot(fb['Player'],fb["Runs"]);
FB = pd.read_excel("../input/slbowl/SRI BOWL.xlsx")
FB["econ"] = FB["Runs"]/FB["Overs"]
FB