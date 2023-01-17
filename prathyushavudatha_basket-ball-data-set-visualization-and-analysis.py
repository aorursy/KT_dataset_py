#Import numpy
import numpy as np

#Seasons
Seasons = ["2005","2006","2007","2008","2009","2010","2011","2012","2013","2014"]
Sdict = {"2005":0,"2006":1,"2007":2,"2008":3,"2009":4,"2010":5,"2011":6,"2012":7,"2013":8,"2014":9}

#Players
Players = ["KobeBryant","JoeJohnson","LeBronJames","CarmeloAnthony","DwightHoward","ChrisBosh","ChrisPaul","KevinDurant","DerrickRose","DwayneWade"]
Pdict = {"KobeBryant":0,"JoeJohnson":1,"LeBronJames":2,"CarmeloAnthony":3,"DwightHoward":4,"ChrisBosh":5,"ChrisPaul":6,"KevinDurant":7,"DerrickRose":8,"DwayneWade":9}

#Salaries
KobeBryant_Salary = [15946875,17718750,19490625,21262500,23034375,24806250,25244493,27849149,30453805,23500000]
JoeJohnson_Salary = [12000000,12744189,13488377,14232567,14976754,16324500,18038573,19752645,21466718,23180790]
LeBronJames_Salary = [4621800,5828090,13041250,14410581,15779912,14500000,16022500,17545000,19067500,20644400]
CarmeloAnthony_Salary = [3713640,4694041,13041250,14410581,15779912,17149243,18518574,19450000,22407474,22458000]
DwightHoward_Salary = [4493160,4806720,6061274,13758000,15202590,16647180,18091770,19536360,20513178,21436271]
ChrisBosh_Salary = [3348000,4235220,12455000,14410581,15779912,14500000,16022500,17545000,19067500,20644400]
ChrisPaul_Salary = [3144240,3380160,3615960,4574189,13520500,14940153,16359805,17779458,18668431,20068563]
KevinDurant_Salary = [0,0,4171200,4484040,4796880,6053663,15506632,16669630,17832627,18995624]
DerrickRose_Salary = [0,0,0,4822800,5184480,5546160,6993708,16402500,17632688,18862875]
DwayneWade_Salary = [3031920,3841443,13041250,14410581,15779912,14200000,15691000,17182000,18673000,15000000]
#Matrix
Salary = np.array([KobeBryant_Salary, JoeJohnson_Salary, LeBronJames_Salary, CarmeloAnthony_Salary, DwightHoward_Salary, ChrisBosh_Salary, ChrisPaul_Salary, KevinDurant_Salary, DerrickRose_Salary, DwayneWade_Salary])

#Games 
KobeBryant_G = [80,77,82,82,73,82,58,78,6,35]
JoeJohnson_G = [82,57,82,79,76,72,60,72,79,80]
LeBronJames_G = [79,78,75,81,76,79,62,76,77,69]
CarmeloAnthony_G = [80,65,77,66,69,77,55,67,77,40]
DwightHoward_G = [82,82,82,79,82,78,54,76,71,41]
ChrisBosh_G = [70,69,67,77,70,77,57,74,79,44]
ChrisPaul_G = [78,64,80,78,45,80,60,70,62,82]
KevinDurant_G = [35,35,80,74,82,78,66,81,81,27]
DerrickRose_G = [40,40,40,81,78,81,39,0,10,51]
DwayneWade_G = [75,51,51,79,77,76,49,69,54,62]
#Matrix
Games = np.array([KobeBryant_G, JoeJohnson_G, LeBronJames_G, CarmeloAnthony_G, DwightHoward_G, ChrisBosh_G, ChrisPaul_G, KevinDurant_G, DerrickRose_G, DwayneWade_G])

#Minutes Played
KobeBryant_MP = [3277,3140,3192,2960,2835,2779,2232,3013,177,1207]
JoeJohnson_MP = [3340,2359,3343,3124,2886,2554,2127,2642,2575,2791]
LeBronJames_MP = [3361,3190,3027,3054,2966,3063,2326,2877,2902,2493]
CarmeloAnthony_MP = [2941,2486,2806,2277,2634,2751,1876,2482,2982,1428]
DwightHoward_MP = [3021,3023,3088,2821,2843,2935,2070,2722,2396,1223]
ChrisBosh_MP = [2751,2658,2425,2928,2526,2795,2007,2454,2531,1556]
ChrisPaul_MP = [2808,2353,3006,3002,1712,2880,2181,2335,2171,2857]
KevinDurant_MP = [1255,1255,2768,2885,3239,3038,2546,3119,3122,913]
DerrickRose_MP = [1168,1168,1168,3000,2871,3026,1375,0,311,1530]
DwayneWade_MP = [2892,1931,1954,3048,2792,2823,1625,2391,1775,1971]
#Matrix
MinutesPlayed = np.array([KobeBryant_MP, JoeJohnson_MP, LeBronJames_MP, CarmeloAnthony_MP, DwightHoward_MP, ChrisBosh_MP, ChrisPaul_MP, KevinDurant_MP, DerrickRose_MP, DwayneWade_MP])

#Field Goals
KobeBryant_FG = [978,813,775,800,716,740,574,738,31,266]
JoeJohnson_FG = [632,536,647,620,635,514,423,445,462,446]
LeBronJames_FG = [875,772,794,789,768,758,621,765,767,624]
CarmeloAnthony_FG = [756,691,728,535,688,684,441,669,743,358]
DwightHoward_FG = [468,526,583,560,510,619,416,470,473,251]
ChrisBosh_FG = [549,543,507,615,600,524,393,485,492,343]
ChrisPaul_FG = [407,381,630,631,314,430,425,412,406,568]
KevinDurant_FG = [306,306,587,661,794,711,643,731,849,238]
DerrickRose_FG = [208,208,208,574,672,711,302,0,58,338]
DwayneWade_FG = [699,472,439,854,719,692,416,569,415,509]
#Matrix
FieldGoals  = np.array([KobeBryant_FG, JoeJohnson_FG, LeBronJames_FG, CarmeloAnthony_FG, DwightHoward_FG, ChrisBosh_FG, ChrisPaul_FG, KevinDurant_FG, DerrickRose_FG, DwayneWade_FG])

#Field Goal Attempts
KobeBryant_FGA = [2173,1757,1690,1712,1569,1639,1336,1595,73,713]
JoeJohnson_FGA = [1395,1139,1497,1420,1386,1161,931,1052,1018,1025]
LeBronJames_FGA = [1823,1621,1642,1613,1528,1485,1169,1354,1353,1279]
CarmeloAnthony_FGA = [1572,1453,1481,1207,1502,1503,1025,1489,1643,806]
DwightHoward_FGA = [881,873,974,979,834,1044,726,813,800,423]
ChrisBosh_FGA = [1087,1094,1027,1263,1158,1056,807,907,953,745]
ChrisPaul_FGA = [947,871,1291,1255,637,928,890,856,870,1170]
KevinDurant_FGA = [647,647,1366,1390,1668,1538,1297,1433,1688,467]
DerrickRose_FGA = [436,436,436,1208,1373,1597,695,0,164,835]
DwayneWade_FGA = [1413,962,937,1739,1511,1384,837,1093,761,1084]
#Matrix
FieldGoalAttempts = np.array([KobeBryant_FGA, JoeJohnson_FGA, LeBronJames_FGA, CarmeloAnthony_FGA, DwightHoward_FGA, ChrisBosh_FGA, ChrisPaul_FGA, KevinDurant_FGA, DerrickRose_FGA, DwayneWade_FGA])

#Points
KobeBryant_PTS = [2832,2430,2323,2201,1970,2078,1616,2133,83,782]
JoeJohnson_PTS = [1653,1426,1779,1688,1619,1312,1129,1170,1245,1154]
LeBronJames_PTS = [2478,2132,2250,2304,2258,2111,1683,2036,2089,1743]
CarmeloAnthony_PTS = [2122,1881,1978,1504,1943,1970,1245,1920,2112,966]
DwightHoward_PTS = [1292,1443,1695,1624,1503,1784,1113,1296,1297,646]
ChrisBosh_PTS = [1572,1561,1496,1746,1678,1438,1025,1232,1281,928]
ChrisPaul_PTS = [1258,1104,1684,1781,841,1268,1189,1186,1185,1564]
KevinDurant_PTS = [903,903,1624,1871,2472,2161,1850,2280,2593,686]
DerrickRose_PTS = [597,597,597,1361,1619,2026,852,0,159,904]
DwayneWade_PTS = [2040,1397,1254,2386,2045,1941,1082,1463,1028,1331]
#Matrix
Points = np.array([KobeBryant_PTS, JoeJohnson_PTS, LeBronJames_PTS, CarmeloAnthony_PTS, DwightHoward_PTS, ChrisBosh_PTS, ChrisPaul_PTS, KevinDurant_PTS, DerrickRose_PTS, DwayneWade_PTS])             
                  
Salary[0]
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
plt.rcParams["figure.figsize" ]= 10,4
plt.plot(Salary[0]) 
plt.show()
plt.plot(Salary[0], c = "black",ls = "--", marker = "s", ms= "7", label = Players[0])
plt.xticks(list(range(0,9)),Seasons, rotation="vertical")
plt.show()
plt.plot(Salary[0], c = "black",ls = "--", marker = "s", ms= "7", label = Players[0])
plt.plot(Salary[1], c = "red",ls = "--", marker = "o", ms= "7", label = Players[1])
plt.plot(Salary[2], c = "green",ls = "--", marker = "^", ms= "7", label = Players[2])
plt.plot(Salary[3], c = "blue",ls = "--", marker = "D", ms= "7", label = Players[3])
plt.xticks(list(range(0,9)),Seasons, rotation="vertical")
plt.show()
plt.plot(Salary[0], c = "black",ls = "--", marker = "s", ms= "7", label = Players[0])
plt.plot(Salary[1], c = "red",ls = "--", marker = "o", ms= "7", label = Players[1])
plt.plot(Salary[2], c = "green",ls = "--", marker = "^", ms= "7", label = Players[2])
plt.plot(Salary[3], c = "blue",ls = "--", marker = "D", ms= "7", label = Players[3])
plt.legend(loc = "upper left",bbox_to_anchor = (1,1) )
plt.xticks(list(range(0,9)),Seasons, rotation="vertical")
plt.show()
plt.plot(Salary[0], c = "black",ls = "--", marker = "s", ms= "7", label = Players[0])
plt.plot(Salary[1], c = "red",ls = "--", marker = "o", ms= "7", label = Players[1])
plt.plot(Salary[2], c = "green",ls = "--", marker = "^", ms= "7", label = Players[2])
plt.plot(Salary[3], c = "blue",ls = "--", marker = "D", ms= "7", label = Players[3])
plt.plot(Salary[4], c = "magenta",ls = "--", marker = "s", ms= "7", label = Players[4])
plt.plot(Salary[5], c = "black",ls = "--", marker = "o", ms= "7", label = Players[5])
plt.plot(Salary[6], c = "red",ls = "--", marker = "^", ms= "7", label = Players[6])
plt.plot(Salary[7], c = "green",ls = "--", marker = "d", ms= "7", label = Players[7])
plt.plot(Salary[8], c = "blue",ls = "--", marker = "s", ms= "7", label = Players[8])
plt.plot(Salary[9], c = "magenta",ls = "--", marker = "o", ms= "7", label = Players[9])
plt.legend(loc = "upper left", bbox_to_anchor = (1,1))
plt.xticks(list(range(0,9)),Seasons, rotation="vertical")
plt.show()
plt.plot(Games[0], c = "black",ls = "--", marker = "s", ms= "7", label = Players[0])
plt.plot(Games[1], c = "red",ls = "--", marker = "o", ms= "7", label = Players[1])
plt.plot(Games[2], c = "green",ls = "--", marker = "^", ms= "7", label = Players[2])
plt.plot(Games[3], c = "blue",ls = "--", marker = "D", ms= "7", label = Players[3])
plt.plot(Games[4], c = "magenta",ls = "--", marker = "s", ms= "7", label = Players[4])
plt.plot(Games[5], c = "black",ls = "--", marker = "o", ms= "7", label = Players[5])
plt.plot(Games[6], c = "red",ls = "--", marker = "^", ms= "7", label = Players[6])
plt.plot(Games[7], c = "green",ls = "--", marker = "d", ms= "7", label = Players[7])
plt.plot(Games[8], c = "blue",ls = "--", marker = "s", ms= "7", label = Players[8])
plt.plot(Games[9], c = "magenta",ls = "--", marker = "o", ms= "7", label = Players[9])
plt.legend(loc = "upper left", bbox_to_anchor = (1,1))
plt.xticks(list(range(0,9)),Seasons, rotation="vertical")
plt.show()

#Adding all the players. Initially started first with one player.
# Defining a Function to plot for all players.

def myplot(playlist):
    for name in playlist:   
        plt.plot(Games[Pdict[name]], c = "black",ls = "--", marker = "s", ms= "7", label = name)
    plt.legend(loc = "upper left", bbox_to_anchor = (1,1))
    plt.xticks(list(range(0,9)),Seasons, rotation="vertical")
    plt.show()
myplot(["JoeJohnson","CarmeloAnthony","ChrisBosh"])

#function advancement
def myplot(playlist):
    col =  {"KobeBryant":"black","JoeJohnson":"red","LeBronJames":"green","CarmeloAnthony":"blue","DwightHoward":"magenta","ChrisBosh":"black","ChrisPaul":"red","KevinDurant":"green","DerrickRose":"blue","DwayneWade":"magenta"}
    mar =  {"KobeBryant":"s","JoeJohnson":"o","LeBronJames":"^","CarmeloAnthony":"d","DwightHoward":"s","ChrisBosh":"o","ChrisPaul":"^","KevinDurant":"d","DerrickRose":"s","DwayneWade":"o"}
    for name in playlist:   
        plt.plot(Games[Pdict[name]], c = col[name],ls = "--", marker = mar[name], ms= "7", label = name)
    plt.legend(loc = "upper left", bbox_to_anchor = (1,1))
    plt.xticks(list(range(0,9)),Seasons, rotation="vertical")
    plt.show()
myplot(Players)

#setting default parameter
def myplot(data,playlist = Players):
    col =  {"KobeBryant":"black","JoeJohnson":"red","LeBronJames":"green","CarmeloAnthony":"blue","DwightHoward":"magenta","ChrisBosh":"black","ChrisPaul":"red","KevinDurant":"green","DerrickRose":"blue","DwayneWade":"magenta"}
    mar =  {"KobeBryant":"s","JoeJohnson":"o","LeBronJames":"^","CarmeloAnthony":"d","DwightHoward":"s","ChrisBosh":"o","ChrisPaul":"^","KevinDurant":"d","DerrickRose":"s","DwayneWade":"o"}
    for name in playlist:   
        plt.plot(data[Pdict[name]], c = col[name],ls = "--", marker = mar[name], ms= "7", label = name)
    plt.legend(loc = "upper left", bbox_to_anchor = (1,1))
    plt.xticks(list(range(0,9)),Seasons, rotation="vertical")
    plt.show()

myplot(Salary,["LeBronJames","DwightHoward"] )
myplot(Salary)
#setting default parameter
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
plt.rcParams["figure.figsize" ]= 8,4
Players = ["KobeBryant","JoeJohnson","LeBronJames","CarmeloAnthony","DwightHoward","ChrisBosh","ChrisPaul","KevinDurant","DerrickRose","DwayneWade"]

def myplot(data,playlist = Players):
    col =  {"KobeBryant":"black","JoeJohnson":"red","LeBronJames":"green","CarmeloAnthony":"blue","DwightHoward":"magenta","ChrisBosh":"black","ChrisPaul":"red","KevinDurant":"green","DerrickRose":"blue","DwayneWade":"magenta"}
    mar =  {"KobeBryant":"s","JoeJohnson":"o","LeBronJames":"^","CarmeloAnthony":"d","DwightHoward":"s","ChrisBosh":"o","ChrisPaul":"^","KevinDurant":"d","DerrickRose":"s","DwayneWade":"o"}
    for name in playlist:   
        plt.plot(data[Pdict[name]], c = col[name],ls = "--", marker = mar[name], ms= "7", label = name)
    plt.legend(loc = "upper left", bbox_to_anchor = (1,1))
    plt.xticks(list(range(0,9)),Seasons, rotation="vertical")
    plt.show()
    
import warnings
warnings.filterwarnings('ignore')
myplot(Salary)
# we can see that KoberBraynt has the the highest salary amoung all other players.
myplot(Salary/Games) 
myplot(Salary/FieldGoals)
#There are anamolities in the data,there is a huge deviation in DerrickRose and KobeBryant graphs.
#In Game Martices
myplot (MinutesPlayed)
myplot (Points)

#injuries are effecting our data


#in game matries normalized
# observe DwightHoward's pattern, the number of field goals/game are same but the the number of field goals/field goal attempts are relatively high.
# From this we can conclude that, DwightHowards makes averge number of field goals per game but he succeeds in most attempts he make. 

import warnings
warnings.filterwarnings('ignore')

myplot (FieldGoals/Games)
myplot (FieldGoals/FieldGoalAttempts)

myplot (Points/Games)
myplot (FieldGoalAttempts/ Games)
# Even here if we observe DwightHoward's pattern, the number of points he makes per game is avergae but the number of attempts he make is relatively low.
# From this we can draw an insight that, although he makes less attempts he succeeds in them.
myplot(MinutesPlayed/ Games) 
myplot(Games)               

# From the below graph we can observe that suprisingly the minutes playes per game is dropping over the years. 
# However, the number of games played remains same.

myplot( FieldGoals / MinutesPlayed)
# We can see that all the players are trying to maintian their efficiency same every year.
import warnings
warnings.filterwarnings('ignore')
myplot(Points/ FieldGoals )
#Seems like they all have differnt points for goals as the graph looks scattered.
