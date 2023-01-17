import pandas as pd

pd.set_option('display.max_rows', 500)

pd.set_option('display.max_columns', 500)

import numpy as np

import os 

import networkx as nx

from collections import defaultdict

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

import seaborn as sns 

from matplotlib import colors

plt.figure(figsize=(10,10))



from tqdm import tqdm_notebook as tqdm

#from tqdm import tqdm

%matplotlib inline
print(os.listdir("/kaggle/"))

data=pd.read_csv("../input/ufcdata/data.csv")

data.head()
for index,fight in data[::-1].iterrows():

    if fight.Winner == "Blue":

        break

print("The first",len(data)-index,"fights have been recorded as red wins")        

DataToDisplay=data[::-1]

display(DataToDisplay.head(10))

print("........................................................................")

display(DataToDisplay[int(len(data)-index-10):int(len(data)-index)])
data.describe()
PrepropData=pd.read_csv("/kaggle/input/ufcdata/preprocessed_data.csv")

PrepropData.head()
RawFighterDetails=pd.read_csv("/kaggle/input/ufcdata/raw_fighter_details.csv")

RawFighterDetails.head()
print("The number of non nan Reach entries is",len(list(RawFighterDetails[RawFighterDetails["Reach"]!= float("nan")])))

print("The number of non nan Stance entries is",len(list(RawFighterDetails[RawFighterDetails["Stance"]!= float("nan")])))

print("The number of non nan DOB entries is",len(list(RawFighterDetails[RawFighterDetails["DOB"]!= float("nan")])))

print("The total number of entries is ",len(RawFighterDetails["fighter_name"]))
RawTotalFightData=pd.read_csv("/kaggle/input/ufcdata/raw_total_fight_data.csv",delimiter =";")

RawTotalFightData.head()
data.dropna()

data.head()

Data=data.join(RawTotalFightData,lsuffix="",rsuffix="_raw").copy()

for key in Data.keys():

    if key[-4:] == "_raw":

        print(key)

        Data=Data.drop([str(key)],axis=1)

print("These are all duplicates or will give no indication of who is going to win the fight.")

        

Data["date_time"]=pd.to_datetime(Data["date"])

Data=Data.drop(["date"],axis=1)

Data=Data.sort_values(["date_time"],ascending=True)



#there is some error in the way to data is recorded the first fight winners are all red

Data=Data.reset_index(drop=True)

for index,fight in Data.iterrows():

    if fight.Winner == "Blue":

        StartPoint=index

        break

Data=Data[StartPoint:]

Data=Data.reset_index(drop=True)

display(Data)

AllData=Data
display(AllData.head(0))
CorneredThingsToKeep=["fighter","current_lose_streak","current_win_streak","wins",

                      "losses","draw","age","SIG_STR.","TOTAL_STR.","TD","total_title_bouts",

                     "win_by_KO/TKO","KD","total_time_fought(seconds)"]



ThingsToKeep=["Winner","win_by","last_round","last_round_time","date_time"]

for thing in CorneredThingsToKeep:

    ThingsToKeep.append("R_"+thing)

    ThingsToKeep.append("B_"+thing)

Data=AllData[ThingsToKeep]

display(Data.head(1))
def ConvertToLandedAttempted(DF,Feature):

    for corner in ["R_","B_"]:

        try:

            DF[[corner+Feature+"_landed",corner+Feature+"_attempted"]] = DF[corner+Feature].str.split(" of ",expand=True)

            for label in [corner+Feature+"_landed",corner+Feature+"_attempted"]:

                DF[label]=pd.to_numeric(DF[label])

            DF=DF.drop([corner+Feature],axis=1)

        except:

            pass

    return DF



FeaturesToSplit=["SIG_STR.","TOTAL_STR.","TD"]

    

for feat in FeaturesToSplit:

    Data=ConvertToLandedAttempted(Data,feat)



for corner in ["R_","B_"]:

    try:

        Data[corner+"total_no._fights"]=pd.to_numeric(Data[corner+"draw"]+Data[corner+"wins"]+Data[corner+"losses"])

        Data=Data.drop([corner+"draw",corner+"losses"],axis=1)

    except:

        pass

    

    

Data.head().T
Data[(Data.R_fighter == "Jon Jones") | (Data.B_fighter == "Jon Jones")][["R_fighter","B_fighter","R_total_time_fought(seconds)","B_total_time_fought(seconds)","date_time"]]
Data=Data.drop([corner+"_total_time_fought(seconds)" for corner in ["R","B"]],axis=1)
Data.head(2)
# now lets create totals for all the revelevent factors



CorneredThingsToKeep=["fighter","current_lose_streak","current_win_streak","wins",

                      "losses","draw","age","SIG_STR.","TOTAL_STR.","TD","total_title_bouts",

                     "win_by_KO/TKO","KD","total_time_fought(seconds)"]



ThingsToKeep=["Winner","win_by","last_round","last_round_time","date_time"]



ThingstoTotal=["SIG_STR.","TOTAL_STR.","TD"]



for corner in ["R_","B_"]:

    for feature in ThingstoTotal:

        for Type in ["_attempted","_landed"]:

            stat=corner+feature+Type

            Data[stat+"_total"]=np.zeros(len(Data))

            

            

for index,fight in tqdm(Data.iterrows(),total=len(Data)):

    for corner in ["R_","B_"]:

        for feature in ThingstoTotal:

            for Type in ["_attempted","_landed"]:

                stat=corner+feature+Type

                fighter=str(fight[corner+"fighter"])

                

                #find the index of the previou fight involving the fighter 

                PreviousFighterFights=Data[(Data.R_fighter == fighter) | (Data.B_fighter == fighter)]

                PreviousFightsIndex=PreviousFighterFights[PreviousFighterFights.index < index].index

                if len(PreviousFightsIndex)> 0:

                    PrevIndex=PreviousFightsIndex[-1]

                else:

                    continue 

                PrevCorner="R_" if Data["R_fighter"][PrevIndex] == fighter else "B_"

                PrevStatLabel=PrevCorner+feature+Type

                PrevTotal=Data[PrevStatLabel+"_total"][PrevIndex] 

                PrevStat=Data[PrevStatLabel][PrevIndex]

                Data[stat+"_total"][index]=PrevTotal+PrevStat

                

def TestIfWorking(Data,Features):

    display(Data[(Data.R_fighter == "Jon Jones") | (Data.B_fighter == "Jon Jones")][["R_fighter","B_fighter",*Features]])

                
def TestIfWorking(Data,Features):

    display(Data[(Data.R_fighter == "Jon Jones") | (Data.B_fighter == "Jon Jones")][["R_fighter","B_fighter"]+Features])

            

TestIfWorking(Data,[corner+"SIG_STR._attempted_total" for corner in ["R_","B_"]]+[corner+"SIG_STR._attempted" for corner in ["R_","B_"]])            
Data["last_round_time"]=[int(minutes)*60 + int(seconds) for minutes,seconds in Data.last_round_time.str.split(":")]

Data.last_round=pd.to_numeric(Data.last_round)

Data["fight_time"]=(Data.last_round-1)*60*5+Data.last_round_time





#totalling the time

def TotalStat(Stats,Data,CornerDependence=True,Descriptor=""):

    if type(Stats)==str:

        Stats=[Stats]

    for stat in Stats:

        for corner in ["R_","B_"]:

            try:

                Data[corner+stat+Descriptor+"_total"]

            except:

                Data[corner+stat+Descriptor+"_total"]=np.zeros(len(Data))

        for index,fight in tqdm(Data.iterrows(),total=len(Data),desc="Totaling "+stat):

            for corner in ["R_","B_"]:

                fighter=str(fight[corner+"fighter"])

                PreviousFighterFights=Data[(Data.R_fighter == fighter) | (Data.B_fighter == fighter)]

                PreviousFightsIndex=PreviousFighterFights[PreviousFighterFights.index < index].index

                if len(PreviousFightsIndex)> 0:

                    PrevIndex=PreviousFightsIndex[-1]

                else:

                    continue

                if CornerDependence==True:

                    PrevCornerTotal="R_" if Data["R_fighter"][PrevIndex] == fighter else "B_"

                    PrevCorner="R_" if Data["R_fighter"][PrevIndex] == fighter else "B_"

                elif CornerDependence==False:

                    PrevCornerTotal="R_" if Data["R_fighter"][PrevIndex] == fighter else "B_"

                    PrevCorner=""

                elif CornerDependece == "Reverse":

                    PrevCorner="B_" if Data["R_fighter"][PrevIndex] == fighter else "R_"

                    PrevCornerTotal="R_" if Data["R_fighter"][PrevIndex] == fighter else "B_"

                Data[corner+stat+Descriptor+"_total"][index]=Data[PrevCornerTotal+stat+Descriptor+"_total"][PrevIndex]+Data[PrevCorner+stat][PrevIndex]

    return Data





            

Data=TotalStat("fight_time",Data,CornerDependence=False)







TestIfWorking(Data,[corner+"fight_time"+"_total" for corner in ["R_","B_"] ]+["fight_time"])

#totaling the defensive stats



def TotalDefenceStat(Stats,Data,CornerDependence="Reverse"):

    if type(Stats)==str:

        Stats=[Stats]

    for stat in Stats:

        for ending,Descriptor in [["_attempted","_faced"],["_landed","_defended"]]:

            StatEnd=stat+ending

            for corner in ["R_","B_"]:

                try:

                    Data[corner+stat+Descriptor+"_total"]

                except:

                    Data[corner+stat+Descriptor+"_total"]=np.zeros(len(Data))

            for index,fight in tqdm(Data.iterrows(),total=len(Data),desc="Totaling "+stat):

                for corner in ["R_","B_"]:

                    fighter=str(fight[corner+"fighter"])

                    PreviousFighterFights=Data[(Data.R_fighter == fighter) | (Data.B_fighter == fighter)]

                    PreviousFightsIndex=PreviousFighterFights[PreviousFighterFights.index < index].index

                    if len(PreviousFightsIndex)> 0:

                        PrevIndex=PreviousFightsIndex[-1]

                    else:

                        continue

                    if CornerDependence==True:

                        PrevCornerTotal="R_" if Data["R_fighter"][PrevIndex] == fighter else "B_"

                        PrevCorner="R_" if Data["R_fighter"][PrevIndex] == fighter else "B_"

                    elif CornerDependence==False:

                        PrevCornerTotal="R_" if Data["R_fighter"][PrevIndex] == fighter else "B_"

                        PrevCorner=""

                    elif CornerDependence == "Reverse":

                        PrevCorner="B_" if Data["R_fighter"][PrevIndex] == fighter else "R_"

                        PrevCornerTotal="R_" if Data["R_fighter"][PrevIndex] == fighter else "B_"

                    if Descriptor == "_faced":

                        Data[corner+stat+Descriptor+"_total"][index]=Data[PrevCornerTotal+stat+Descriptor+"_total"][PrevIndex]+Data[PrevCorner+StatEnd][PrevIndex]

                    else:

                        Data[corner+stat+Descriptor+"_total"][index]=Data[PrevCornerTotal+stat+Descriptor+"_total"][PrevIndex]+(Data[PrevCorner+stat+"_attempted"][PrevIndex]-Data[PrevCorner+StatEnd][PrevIndex])

    return Data



Features=[stat for stat in ["SIG_STR.","TOTAL_STR.","TD"]]









    

Data=TotalDefenceStat(Features,Data)

TestIfWorking(Data,[key for key in Data.keys() if "SIG_STR." in key])
def TotalStat(Stats,Data,CornerDependence=True,Descriptor=""):

    if type(Stats)==str:

        Stats=[Stats]

    for stat in Stats:

        for corner in ["R_","B_"]:

            try:

                Data[corner+stat+Descriptor+"_total"]

            except:

                Data[corner+stat+Descriptor+"_total"]=np.zeros(len(Data))

        for index,fight in tqdm(Data.iterrows(),total=len(Data),desc="Totaling "+stat):

            for corner in ["R_","B_"]:

                fighter=str(fight[corner+"fighter"])

                PreviousFighterFights=Data[(Data.R_fighter == fighter) | (Data.B_fighter == fighter)]

                PreviousFightsIndex=PreviousFighterFights[PreviousFighterFights.index < index].index

                if len(PreviousFightsIndex)> 0:

                    PrevIndex=PreviousFightsIndex[-1]

                else:

                    continue

                if CornerDependence==True:

                    PrevCornerTotal="R_" if Data["R_fighter"][PrevIndex] == fighter else "B_"

                    PrevCorner="R_" if Data["R_fighter"][PrevIndex] == fighter else "B_"

                elif CornerDependence==False:

                    PrevCornerTotal="R_" if Data["R_fighter"][PrevIndex] == fighter else "B_"

                    PrevCorner=""

                elif CornerDependece == "Reverse":

                    PrevCorner="B_" if Data["R_fighter"][PrevIndex] == fighter else "R_"

                    PrevCornerTotal="R_" if Data["R_fighter"][PrevIndex] == fighter else "B_"

                Data[corner+stat+Descriptor+"_total"][index]=Data[PrevCornerTotal+stat+Descriptor+"_total"][PrevIndex]+Data[PrevCorner+stat][PrevIndex]

    return Data





Data=TotalStat("KD",Data)

TestIfWorking(Data,[key for key in Data.keys() if "KD" in key])
#calculting the finnish probability for each fighter 

# we will do this by totaling number of finnishes and then dividing by total fight time

display("The different win_by options ",Data.win_by.unique())



def IsFinnish(DF,corner="Undefined"):

    CornerPrefix="R_" if corner=="Red" else "B_"

    if DF["Winner"]==corner and DF["win_by"] in ['KO/TKO', 'Submission']:

        return 1

    return 0



for corner in ["Red","Blue"]:        

    Data[corner[0]+"_finish"]=Data.apply(IsFinnish,**{"corner":corner},axis=1)



Data=TotalStat("finish",Data)

TestIfWorking(Data,[key for key in Data.keys() if "finish" in key ])
Features=[]

for corner in ["R_","B_"]:

    for AttackForm in ["SIG_STR.","TOTAL_STR.","TD"]:

        Data[corner+AttackForm+"_probability"]=Data[corner+AttackForm+"_landed_total"]/Data[corner+AttackForm+"_attempted_total"]

        Features.append(AttackForm+"_probability")

        Data[corner+AttackForm+"_defence_probability"]=Data[corner+AttackForm+"_defended_total"]/Data[corner+AttackForm+"_faced_total"]

        Features.append(AttackForm+"_defence_probability")

        Data[corner+AttackForm+"_rate"]=Data[corner+AttackForm+"_attempted_total"]/Data[corner+"fight_time"+"_total"]

        Features.append(AttackForm+"_rate")

        

    Data[corner+"KD_rate"]=Data[corner+"KD"+"_total"]/Data[corner+"fight_time"+"_total"]

    Features.append("KD_rate")

    

    Data[corner+"win_probability"]=Data[corner+"wins"]/Data[corner+"total_no._fights"]

    Features.append("win_probability")

    

    Data[corner+"finish_rate"]=Data[corner+"finish_total"]/Data[corner+"fight_time"+"_total"]

    Features.append("finish_rate")

    

    Features+=[corner+factor for factor in ["age","total_no._fights","total_title_bouts","current_win_streak","current_lose_streak"]]

display(Data[-10:])
Data=Data[~Data.win_by.isin(["DQ","TKO - Doctor's Stoppage","Overturned"])]



def BinaryWinner(DF):

    if DF["Winner"]=="Red":

        return 1

    elif DF["Winner"]=="Blue":

        return 0 

    else:

        return np.nan

Data["Winner"]=Data.apply(BinaryWinner,axis=1)

Data.head()
print(len(Data.dropna())*100.0/len(Data),"% of the fights don't have NaN in them")

print(Data.Winner.mean()*100,"% of the fights were won ")
FightNetwork=nx.MultiDiGraph()

for index,fight in tqdm(Data.iterrows(),total=len(Data)):

    if fight.Winner == 1:

        FightNetwork.add_edge(fight.B_fighter,fight.R_fighter,key=fight.date_time)

    elif fight.Winner == 0:

        FightNetwork.add_edge(fight.R_fighter,fight.B_fighter,key=fight.date_time)

fig=plt.figure(figsize=(12,12),dpi=100)

nx.draw(FightNetwork,pos=nx.spring_layout(FightNetwork,k=0.4),arrowsize=0.1,font_size=1,width=0.02,node_size=0,with_labels=True)

fig.show()
Centralities=nx.katz_centrality(nx.DiGraph(FightNetwork),tol=0.001)
%%time

Centralities=nx.katz_centrality(nx.DiGraph(FightNetwork),tol=0.001)
%%time

Centralities=nx.katz_centrality(nx.DiGraph(FightNetwork),tol=0.001,alpha=0.1,beta=1)
def PrintScores(Scores,number):

    for i in  sorted(Scores,key=Scores.get,reverse=True)[:number]:

        print(i,Scores[i])



PrintScores(Centralities,30)
RedScores,BlueScores=[],[]

RedFighters,BlueFighters=[],[]

TotalCentralities=[]

Cententralities={}



FightNetwork=nx.MultiDiGraph()

SumCentralities=np.nan

Indexes=list(Data.index)

for (index,fight),NextIndex in tqdm(zip(Data.iterrows(),Indexes[1:]),total=len(Data)-1):

    try:

        RedScores.append(Cententralities[fight.R_fighter])

    except:

        RedScores+=[np.median(RedScores) if len(RedScores)>0 else np.nan]

    try:

        BlueScores.append(Cententralities[fight.B_fighter])

    except:

        BlueScores+=[np.median(BlueScores) if len(BlueScores)>0 else np.nan]

    

    if fight.Winner == 1:

        FightNetwork.add_edge(fight.B_fighter,fight.R_fighter,key=fight.date_time)

    elif fight.Winner == 0:

        FightNetwork.add_edge(fight.R_fighter,fight.B_fighter,key=fight.date_time)

    

    TotalCentralities.append(SumCentralities)    

    if fight.date_time!=Data.date_time[NextIndex]:

        Cententralities=nx.katz_centrality(nx.DiGraph(FightNetwork),tol=0.001,alpha=0.1,beta=1)

        SumCentralities=np.sum(list(Cententralities.values()))





try:

    RedScores.append(Cententralities[list(Data.R_fighter)[-1]])

except:

    RedScores+=[np.median(RedScores)]

try:

    BlueScores.append(Cententralities[list(Data.B_fighter)[-1]])

except:

    BlueScores+=[np.median(BlueScores)]



TotalCentralities.append(TotalCentralities[-1])

    

Data["R_centrality"]=RedScores

Data["B_centrality"]=BlueScores

Data["total_centrality"]=TotalCentralities

  

    
Data["total_centrality"].describe()
print("The first 10 non NaN maximum centrality scores are:")

count=0

for index,fight in Data.iterrows():

    if fight.R_centrality >0 or fight.B_centrality > 0:

        print(np.nanmax([fight.R_centrality,fight.B_centrality]))

        count+=1

        if count == 10:

            break

            

print("\nThe last 10 maximum centrality scores are:")

count=0

for index,fight in Data[::-1].iterrows():

    if fight.R_centrality >0 or fight.B_centrality > 0:

        print(np.nanmax([fight.R_centrality,fight.B_centrality]))

        count+=1

        if count == 10:

            break

            

print("\nThe maximum centrality is",np.nanmax(list(Data.R_centrality)+list(Data.B_centrality)))
import random

Cententralities=nx.katz_centrality(nx.DiGraph(FightNetwork),tol=0.001,alpha=0.1,beta=1)



# R_SIG_STR._attempted is just a random thing to plot against 

CentralititesData=Data[["R_fighter","B_fighter","Winner","R_SIG_STR._attempted"]]



CentralititesData["R_centrality"]=[Cententralities[fighter] if fighter in Cententralities.keys() else np.median(list(Cententralities.values())) for fighter in list(CentralititesData["R_fighter"])]

CentralititesData["B_centrality"]=[Cententralities[fighter] if fighter in Cententralities.keys() else np.median(list(Cententralities.values())) for fighter in list(CentralititesData["B_fighter"])]

CentralititesData["centrality_difference"]=(CentralititesData["R_centrality"]-CentralititesData["B_centrality"])/(CentralititesData["R_centrality"]+CentralititesData["B_centrality"])



Winners=list(CentralititesData["Winner"])

cols=["b","r"]

winner=[0.0,1.0]

Colors=[cols[winner.index(x)] if x in [0.0,1.0] else random.choice(cols) for x in Winners]

plt.scatter(CentralititesData["centrality_difference"],CentralititesData["R_SIG_STR._attempted"],color=Colors,alpha=0.1)

plt.xlabel("Centrality Difference")

plt.ylabel("Red Fighter's Number of Attempted Strikes")



plt.show()
Data["centrality_difference/total"] = (Data["R_centrality"] - Data["B_centrality"])/Data["total_centrality"]

Data["centrality_difference/product"] = (Data["R_centrality"] - Data["B_centrality"])/(Data["R_centrality"]*Data["B_centrality"])



Features=[]

for AttackForm in ["SIG_STR.","TOTAL_STR.","TD"]:

    Features.append(AttackForm+"_probability")

    Features.append(AttackForm+"_defence_probability")

    Features.append(AttackForm+"_rate")

Features.append("KD_rate")

Features.append("win_probability")

Features.append("finish_rate")

Features+=[factor for factor in ["age","total_no._fights","total_title_bouts","current_win_streak","current_lose_streak"]]

Features=[corner+feat for corner in ["R_","B_"] for feat in Features]

Features+=["R_centrality","B_centrality","centrality_difference/total","centrality_difference/product"]

print(Features)
import seaborn as sns 

StuffToPlot=Data[Features+["Winner"]].fillna(Data[Features+["Winner"]].median())

PairPlot=sns.pairplot(Data[Features+["Winner"]],hue="Winner",palette={1:"r",0:"b"},plot_kws={"alpha":0.03})

#ELR = estimated landing rate

ELRFeatures=["TD","SIG_STR.","TOTAL_STR."]



NewFeatures=[]

Corners=["R_","B_"]

for feature in ELRFeatures:

    for corner in Corners:

        OtherCorner=Corners[Corners.index(corner)-1]

        NewFeature=corner+feature+"_ELR"  

        Start=corner+feature #  prob of landing     prob of not defending                                rate of attempts

        Data[NewFeature]=Data[Start+"_probability"]*(1-Data[OtherCorner+feature+"_defence_probability"])*Data[Start+"_rate"]

        NewFeatures.append(NewFeature)

        

ELRFeatures=NewFeatures 

import seaborn as sns 



DataToPlot=Data[Data["R_total_no._fights"] > 3]

DataToPlot=DataToPlot[DataToPlot["B_total_no._fights"] > 3]

DataToPlot=DataToPlot[NewFeatures+["Winner"]]

DataToPlot=DataToPlot.dropna()

print(len(DataToPlot))



StuffToPlot=DataToPlot

PairPlot=sns.pairplot(DataToPlot,hue="Winner",palette={1:"r",0:"b"},plot_kws={"alpha":0.01})
ELRDifferenceFeatures=[]

for feature in ["TD","SIG_STR.","TOTAL_STR."]: 

    NewFeature=feature+"_ELR_difference"

    Data[NewFeature]=(Data["R_"+feature+"_ELR"]-Data["B_"+feature+"_ELR"])/(Data["R_"+feature+"_ELR"]+Data["B_"+feature+"_ELR"])

    ELRDifferenceFeatures.append(NewFeature)

    
DataToPlot=Data[Data["R_total_no._fights"] > 3]

DataToPlot=DataToPlot[DataToPlot["B_total_no._fights"] > 3]

DataToPlot=DataToPlot[ELRDifferenceFeatures+["Winner"]]

DataToPlot=DataToPlot.dropna()



DataToPlot=DataToPlot[ELRDifferenceFeatures+["Winner"]]

print(len(DataToPlot))





PairPlot=sns.pairplot(DataToPlot,hue="Winner",palette={1:"r",0:"b"},plot_kws={"alpha":0.05})

DataToPlot=Data[ELRDifferenceFeatures+["Winner"]]

DataToPlot=DataToPlot.dropna()

print(len(DataToPlot))



PairPlot=sns.pairplot(DataToPlot,hue="Winner",palette={1:"r",0:"b"},plot_kws={"alpha":0.05})
def TotalStat(Stats,Data,CornerDependence=True,Descriptor=""):

    if type(Stats)==str:

        Stats=[Stats]

    for stat in Stats:

        for corner in ["R_","B_"]:

            try:

                Data[corner+stat+Descriptor+"_total"]

            except:

                Data[corner+stat+Descriptor+"_total"]=np.zeros(len(Data))

        for index,fight in tqdm(Data.iterrows(),total=len(Data),desc="Totaling "+stat):

            for corner in ["R_","B_"]:

                fighter=str(fight[corner+"fighter"])

                PreviousFighterFights=Data[(Data.R_fighter == fighter) | (Data.B_fighter == fighter)]

                PreviousFightsIndex=PreviousFighterFights[PreviousFighterFights.index < index].index

                if len(PreviousFightsIndex)> 0:

                    PrevIndex=PreviousFightsIndex[-1]

                else:

                    continue

                if CornerDependence==True:

                    PrevCornerTotal="R_" if Data["R_fighter"][PrevIndex] == fighter else "B_"

                    PrevCorner="R_" if Data["R_fighter"][PrevIndex] == fighter else "B_"

                elif CornerDependence==False:

                    PrevCornerTotal="R_" if Data["R_fighter"][PrevIndex] == fighter else "B_"

                    PrevCorner=""

                elif CornerDependece == "Reverse":

                    PrevCorner="B_" if Data["R_fighter"][PrevIndex] == fighter else "R_"

                    PrevCornerTotal="R_" if Data["R_fighter"][PrevIndex] == fighter else "B_"

                    

                PrevStat=Data[PrevCorner+stat][PrevIndex] if not np.isnan(Data[PrevCorner+stat][PrevIndex]) else 0

                PrevTotal=Data[PrevCornerTotal+stat+Descriptor+"_total"][PrevIndex] if not np.isnan(Data[PrevCornerTotal+stat+Descriptor+"_total"][PrevIndex]) else 0

            

                Data[corner+stat+Descriptor+"_total"][index]=PrevTotal+PrevStat

    return Data





AttackForms=["SIG_STR.","TOTAL_STR.","TD"]



for index,corner in enumerate(["R_","B_"]):

    OtherCorner=["R_","B_"][index-1]

    for attack in AttackForms:

        Data[corner+attack+"_weighted_landed"]=Data[OtherCorner+attack+"_defence_probability"]*Data[corner+attack+"_landed"]



WeightedLanded=[attack+"_weighted_landed" for attack in AttackForms]



Data=TotalStat(WeightedLanded,Data)



WeightedProbs=[]

for corner in ["R_","B_"]:

    for attack in AttackForms:

        Data[corner+attack+"_weighted_probability"]=Data[corner+attack+"_weighted_landed_total"]/(Data[corner+attack+"_attempted_total"]+0.0000000001)

        WeightedProbs.append(corner+attack+"_weighted_probability")



Data.replace({feature:0.0 for feature in WeightedProbs},np.nan)

    

def TestIfWorking(Data,Features):

    display(Data[(Data.R_fighter == "Jon Jones") | (Data.B_fighter == "Jon Jones")][["R_fighter","B_fighter",*Features]])



    

TestIfWorking(Data,[key for key in list(Data.keys()) if "SIG_STR." in str(key)])
DataToPlot=Data

DataToPlot=DataToPlot[DataToPlot["R_total_no._fights"] > 3]

DataToPlot=DataToPlot[DataToPlot["B_total_no._fights"] > 3]

DataToPlot=DataToPlot[[corner + x + "_weighted_probability" for x in AttackForms for corner in ["R_","B_"]]+["Winner"]]



for feature in [corner + x + "_weighted_probability" for x in AttackForms for corner in ["R_","B_"]]:

    DataToPlot[DataToPlot[feature] != 0]

DataToPlot=DataToPlot.dropna()



print("plot for",len(DataToPlot),"fights")





PairPlot=sns.pairplot(DataToPlot,hue="Winner",palette={1:"r",0:"b"},plot_kws={"alpha":0.05})

for attack in AttackForms:

    Data[attack+"_weighted_probability_difference"]=(Data["R_"+attack+"_weighted_probability"]-Data["B_"+attack+"_weighted_probability"])/(Data["R_"+attack+"_weighted_probability"]+Data["B_"+attack+"_weighted_probability"])
DataToPlot=Data

DataToPlot=DataToPlot[DataToPlot["R_total_no._fights"] > 3]

DataToPlot=DataToPlot[DataToPlot["B_total_no._fights"] > 3]

DataToPlot=DataToPlot[[attack+"_weighted_probability_difference" for attack in AttackForms]+["Winner"]]



for feature in [attack+"_weighted_probability_difference" for attack in AttackForms]:

    DataToPlot=DataToPlot[DataToPlot[feature] != 0]



DataToPlot=DataToPlot.dropna()

print("plot for",len(DataToPlot),"fights")





PairPlot=sns.pairplot(DataToPlot,hue="Winner",palette={1:"r",0:"b"},plot_kws={"alpha":0.05})

Corners=["R_","B_"]

for feature in ["TD","SIG_STR.","TOTAL_STR."]:

    for corner in Corners:

        OtherCorner=Corners[Corners.index(corner)-1]

        Data[corner+feature+"_weighted_ELR"]=Data[corner+feature+"_weighted_probability"]*(1-Data[OtherCorner+feature+"_defence_probability"])*Data[corner+feature+"_rate"]



FeaturesToPlot=[]

for feature in ["TD","SIG_STR.","TOTAL_STR."]:

    Data[feature+"_weighted_ELR_difference"]=(Data["R_"+feature+"_weighted_ELR"]-Data["B_"+feature+"_weighted_ELR"])/(Data["R_"+feature+"_weighted_ELR"]+Data["B_"+feature+"_weighted_ELR"])

    FeaturesToPlot.append(feature+"_weighted_ELR_difference")



Data["centrality_difference/total"] = (Data["R_centrality"] - Data["B_centrality"])/(Data["R_centrality"] + Data["B_centrality"])

Data["centrality_difference/product"] = (Data["R_centrality"] - Data["B_centrality"])/(Data["R_centrality"]*Data["B_centrality"])

import seaborn as sns



DataToPlot=Data

DataToPlot=DataToPlot[DataToPlot["R_total_no._fights"] > 3]

DataToPlot=DataToPlot[DataToPlot["B_total_no._fights"] > 3]

DataToPlot=DataToPlot[FeaturesToPlot+["Winner"]]





DataToPlot=DataToPlot.dropna()

print("plot for",len(DataToPlot),"fights")



PairPlot=sns.pairplot(DataToPlot,hue="Winner",palette={1:"r",0:"b"},plot_kws={"alpha":0.05})
import pandas as pd

import numpy as np 



from sklearn.model_selection import train_test_split

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import StandardScaler

from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import make_scorer



Features=[]

for AttackForm in ["SIG_STR.","TOTAL_STR.","TD"]:

    Features.append(AttackForm+"_probability")

    Features.append(AttackForm+"_defence_probability")

    Features.append(AttackForm+"_rate")

Features.append("KD_rate")

Features.append("win_probability")

Features.append("finish_rate")

Features+=[factor for factor in ["age","total_no._fights","total_title_bouts","current_win_streak","current_lose_streak","centrality"]]

Features=[corner+feat for corner in ["R_","B_"] for feat in Features]

for AttackForm in ["SIG_STR.","TOTAL_STR.","TD"]:

    Features.append(AttackForm+"_ELR_difference")

Features.append("centrality_difference/total")

Features.append("centrality_difference/product")



DataToUse=Data

DataToUse=DataToUse[DataToUse["R_total_no._fights"] > 3 ]

DataToUse=DataToUse[DataToUse["B_total_no._fights"] > 3 ]

DataToUse=DataToUse[Features+["Winner"]]

DataToUse=DataToUse.dropna()

print("We have data for",len(DataToUse),"fights")

TransformData=StandardScaler()



Continue=True

while Continue:

    TrainX,TestX,TrainY,TestY=train_test_split(DataToUse[Features].copy(),DataToUse["Winner"].copy(),test_size=0.25)

    if 0.99<np.mean(TrainY)/np.mean(TestY)<1.01:

        Continue=False



print("Red wins",np.mean(TrainY)*100,"% of the training fights")

print("Red wins",np.mean(TestY)*100,"% of the test fights")

TransformData.fit_transform(TrainX)

TransformData.transform(TestX)



parameters = {'kernel':['rbf'], 'C':[0.2,0.3,0.4,0.5]}

print("For a rbf SVC")

Model=SVC(gamma="auto")

ModelTuner = GridSearchCV(Model,parameters,n_jobs=-1,cv=5,scoring="accuracy")

ModelTuner.fit(TrainX,TrainY)

print("We get an accuracy of",ModelTuner.score(TrainX,TrainY),"on the training set")

print("We get an accuracy of ",ModelTuner.score(TestX,TestY),"on the test set\n")

print("With hyper paremetres of",ModelTuner.best_estimator_.get_params())
print("For a polynomial SVC of degree 2")

Model=SVC(gamma="auto")

parameters = {'kernel':["poly"],"degree":[2], 'C':[0.1,0.3,0.7,0.9,0.5,1]}

ModelTuner = GridSearchCV(Model,parameters,n_jobs=-1,cv=5,scoring="accuracy")

ModelTuner.fit(TrainX,TrainY)

print("We get an accuracy of",ModelTuner.score(TrainX,TrainY),"on the training set")

print("We get an accuracy of ",ModelTuner.score(TestX,TestY),"on the test set\n")

print("With hyper paremetres of",ModelTuner.best_estimator_.get_params())
print("For a linear SVC")

Model=SVC(gamma="auto")

parameters = {'kernel':["linear"], 'C':[1,3,5,7,10]}

ModelTuner = GridSearchCV(Model,parameters,n_jobs=-1,cv=5,scoring="accuracy")

ModelTuner.fit(TrainX,TrainY)

print("We get an accuracy of",ModelTuner.score(TrainX,TrainY),"on the training set")

print("We get an accuracy of ",ModelTuner.score(TestX,TestY),"on the test set\n")

print("With hyper paremetres of",ModelTuner.best_estimator_.get_params())
print("For a Random Forrest")

Model=RandomForestClassifier(n_jobs=-1)

parameters = {'n_estimators':[100,200,300,1000],"max_depth":[None,200,120,100,80,10],"bootstrap":[False]}

ModelTuner = GridSearchCV(Model,parameters,n_jobs=-1,cv=5,scoring="accuracy")

ModelTuner.fit(TrainX,TrainY)

print("We get an accuracy of",ModelTuner.score(TrainX,TrainY),"on the training set")

print("We get an accuracy of ",ModelTuner.score(TestX,TestY),"on the test set\n")

print("With hyper paremetres of",ModelTuner.best_estimator_.get_params())
print("For KNN")

Model=KNeighborsClassifier()

parameters = {'n_neighbors':[4,5,7,8],"weights":["uniform","distance"]}

ModelTuner = GridSearchCV(Model,parameters,n_jobs=-1,cv=5,scoring="accuracy")

ModelTuner.fit(TrainX,TrainY)

print("We get an accuracy of",ModelTuner.score(TrainX,TrainY),"on the training set")

print("We get an accuracy of ",ModelTuner.score(TestX,TestY),"on the test set\n")

print("With hyper paremetres of",ModelTuner.best_estimator_.get_params())
print("For Logistic Regression")

Model=LogisticRegression(max_iter=1000000000,tol=0.00001)

parameters = {'penalty':["l1","l2"],"C":[0.01,0.1,0.2,0.33,1.0]}

ModelTuner = GridSearchCV(Model,parameters,n_jobs=-1,cv=5,scoring="accuracy")

ModelTuner.fit(TrainX,TrainY)

print("We get an accuracy of",ModelTuner.score(TrainX,TrainY),"on the training set")

print("We get an accuracy of ",ModelTuner.score(TestX,TestY),"on the test set\n")

print("With hyper paremetres of",ModelTuner.best_estimator_.get_params())
import tensorflow as tf

#from tensorflow.keras import backend.tensorflow_backend.set_session 

from tensorflow import keras

           

TrainX = np.array(TrainX.astype("float64"))

TestX = np.array(TestX.astype("float64"))



TrainY = np.array(TrainY.astype("float64"))

TestY = np.array(TestY.astype("float64"))

  

Optimiser=keras.optimizers.SGD(learning_rate=0.00006,momentum=0.7)



Model=keras.Sequential()

Model.add(keras.layers.Dense(32,input_dim=int(np.shape(TrainX)[1]),activation='elu'))

Model.add(keras.layers.Dense(32,activation='elu'))

Model.add(keras.layers.Dense(32,activation='elu'))

Model.add(keras.layers.Dense(32,activation='elu'))

Model.add(keras.layers.Dense(32,activation='elu'))

Model.add(keras.layers.Dense(32,activation='elu'))

Model.add(keras.layers.Dense(1,activation="sigmoid"))



Model.compile(loss="binary_crossentropy",optimizer=Optimiser,metrics=["accuracy"])

History=Model.fit(TrainX,TrainY,batch_size=60,epochs=4000,validation_data=(TestX,TestY),verbose=0)
def MovingAverage(a, n=10) :

    ret = np.cumsum(a, dtype=float)

    ret[n:] = ret[n:] - ret[:-n]

    MovAv=list(a[:n-1])+list(ret[n - 1:] / n)

    return MovAv



Xaxis=range(0,len(History.history["accuracy"]),10)

YAcc=MovingAverage(History.history["accuracy"][::10])

YValAcc=MovingAverage(History.history["val_accuracy"][::10])

plt.plot(Xaxis,YAcc,label="Train set")

plt.plot(Xaxis,YValAcc,label= "Test set")

plt.grid(linestyle="-.")



plt.legend()

plt.show()    



print("Red wins",np.mean(TrainY),"% and",np.mean(TestY),"% of the training and test fights respectivly")

print("The max accurcy on the test set is",np.max(History.history["val_accuracy"]))
Optimiser=keras.optimizers.SGD(learning_rate=0.00004,momentum=0.7)



Model=keras.Sequential()

Model.add(keras.layers.Dense(32,input_dim=np.shape(TrainX)[1],activation='elu'))

Model.add(keras.layers.Dense(32,activation='elu'))

Model.add(keras.layers.Dense(32,activation='elu'))

Model.add(keras.layers.Dense(32,activation='elu'))

Model.add(keras.layers.Dense(32,activation='elu'))

Model.add(keras.layers.Dense(32,activation='elu'))

Model.add(keras.layers.Dense(32,activation='elu'))

Model.add(keras.layers.Dense(32,activation='elu'))

Model.add(keras.layers.Dense(32,activation='elu'))

Model.add(keras.layers.Dense(32,activation='elu'))

Model.add(keras.layers.Dense(32,activation='elu'))

Model.add(keras.layers.Dense(1,activation="sigmoid"))



Model.compile(loss="binary_crossentropy",optimizer=Optimiser,metrics=["accuracy"])





BigNetHistory=Model.fit(TrainX,TrainY,batch_size=60,epochs=4000,validation_data=(TestX,TestY),verbose=0)
Xaxis=range(0,len(BigNetHistory.history["accuracy"]),10)



YAcc=MovingAverage(BigNetHistory.history["accuracy"][::10])

YValAcc=MovingAverage(BigNetHistory.history["val_accuracy"][::10])



plt.plot(Xaxis,YAcc,label="Train Set")

plt.plot(Xaxis,YValAcc,label="Test Set")

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend()

plt.grid(linestyle="-.")

plt.show()    



print("Red wins",np.mean(TrainY),"% and",np.mean(TestY),"% of the training and test fights respectivly")

print("The max accurcy on the test set is",np.max(BigNetHistory.history["val_accuracy"]))