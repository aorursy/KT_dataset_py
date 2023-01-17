from pandas import Series,DataFrame
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
sns.set_style("whitegrid")
df=pd.read_csv("matches.csv")
df1=DataFrame([df["city"],df["team1"],df["team2"],df["toss_winner"],df["toss_decision"],df["winner"],df["venue"]]).T
df1.head()
def team(name):
    hy=[]
    ct=[]
    t2=[]
    td=[]
    w=[]
    v=[]
    t1=[]
    for i in np.arange(len(df1["city"])):
        if df1["team1"][i]==name:
            ct=np.append(ct,df1["city"][i])
            t1=np.append(t1,name)
            t2=np.append(t2,df1["team2"][i])
            td=np.append(td,df1["toss_decision"][i])
            w=np.append(w,df1["winner"][i])
            v=np.append(v,df1["venue"][i])
    teamdf=DataFrame([ct,t1,t2,td,w,v],index=["city","team1","team2","toss_decision","winner","venue"]).T
    return teamdf
team("Kolkata Knight Riders").head()
def winfrq(team2): 
    kkr=team(team2) 
    pw=0
    for i in np.arange(len(kkr["city"])):
        if (kkr["winner"][i]==team2):
            pw=(pw+1)
    return pw
def px4w(tem):
    kkr=team(tem)     
    pw1=winfrq(tem)
                            #this is the function taking the team and this giving the winnning proobablity for fielding and batting respectively
    pb=0
    pf=0
    for i in np.arange(len(kkr["city"])):
        if (kkr["toss_decision"][i]=="field") & (kkr["winner"][i]==tem):
             pf=(pf+1)
        elif (kkr["toss_decision"][i]=="bat") & (kkr["winner"][i]==tem):
             pb=(pb+1)
    return pb/pw1,pf/pw1
(px4w("Kolkata Knight Riders"))               #so this give sthe fdact that the prob of kkr winning field first is more than bat first
lb=["field_first","bat_first"]
plt.pie(px4w("Kolkata Knight Riders"),labels=lb)
plt.pie(px4w("Mumbai Indians"),labels=lb)
plt.pie(px4w("Sunrisers Hyderabad"),labels=lb,shadow=True,radius=1.5,colors=["red","green"])
venues=Series(np.unique(df1["venue"]))
def ven(tem,ground):
    kkr=team(tem)  
    pw1=winfrq(tem)
    pb=0
    for i in np.arange(len(kkr["city"])):
          if (kkr["venue"][i]==ground) & (kkr["winner"][i]==tem):
                pb=pb+1
    return pb/pw1 
ven("Kolkata Knight Riders","M Chinnaswamy Stadium")
#ven("Sunrisers Hyderabad","Sharjah Cricket Stadium")
def winven(team):
    s=[]
    for i in venues:
        s=np.append(s,(ven(team,i)))
    dfven=DataFrame(s,index=venues,columns=["prob"])   
    return dfven
def home(team,venu):
    ph=(winven(team)["prob"])[venu]
    pa=1-ph
    return ph,pa
(winven("Kolkata Knight Riders")["prob"])   
(winven("Kolkata Knight Riders")["prob"]).plot(kind="bar",label=True,figsize=(15,5))
(winven("Mumbai Indians")["prob"]).plot(kind="bar",label=True,figsize=(15,5),alpha=1)
(winven("Mumbai Indians")["prob"]).plot(kind="bar",label=True,figsize=(15,5),alpha=.8)
(winven("Kolkata Knight Riders")["prob"]).plot(kind="bar",label=True,figsize=(15,5),alpha=.2)
teams=np.unique(df1["team2"])
teams
def opps(tem,opp):
    kkr=team(tem)  
    pw1=winfrq(tem)
    pb=0
    for i in np.arange(len(kkr["city"])):
          if (kkr["team2"][i]==opp) & (kkr["winner"][i]==tem):
                pb=pb+1
    return pb/pw1 
def oppdis(team):
    s=[]
    for i in teams:
        s=np.append(s,(opps(team,i)))
    dfwdis=DataFrame(s,index=teams,columns=["prob"]) 
    return dfwdis
venues
oppdis("Royal Challengers Bangalore")
oppdis("Mumbai Indians").plot(kind="bar",figsize=(10,5),label=True)
oppdis("Chennai Super Kings").plot(kind="bar",figsize=(10,5),color="yellow",alpha=1)
(px4w("Kolkata Knight Riders")[0])*((winven("Kolkata Knight Riders")["prob"])[4])*((oppdis("Kolkata Knight Riders")["prob"])[2])

def conprob(yrtm,opp,ven,toss,in51):
    if toss=="field":
        toss=0
    if toss=="bat":
        toss=1
    if in51=="yes":
        pyrtm=home(yrtm,ven)[0]
    else:
        pyrtm=home(yrtm,ven)[1]
    py=(px4w(yrtm)[toss])*pyrtm*((oppdis(yrtm)["prob"])[opp])
    if ((winven(yrtm)["prob"])[ven])==0:
        tem=.000001
        postpre=(px4w(yrtm)[toss])*tem*((oppdis(yrtm)["prob"])[opp])
    else :
        postpre=(px4w(yrtm)[toss])*((winven(yrtm)["prob"])[ven])*((oppdis(yrtm)["prob"])[opp])
    return py
conprob("Kolkata Knight Riders","Sunrisers Hyderabad","Eden Gardens","field","yes")
conprob("Kolkata Knight Riders","Sunrisers Hyderabad","Wankhede Stadium","bat","no")
in1=input(" plz enter team1 ", )      
in2=input(" plz enter team2 ", )       
in3=input(" plz enter venue ", ) 
in51=input("is the venue  home groung for team1(yes/no) ", )
in61=input("is the venue home ground for team2(yes/no) ",  )
in4=input(" plz enter your bat or field first ", )       
if in4=="field":
    in5="bat"
elif in4=="bat":
    in5="field"
p1=conprob(in1,in2,in3,in4,in51)
p2=conprob(in2,in1,in3,in5,in61)
if p1>p2:
    print(in1,"has more probablity to win in the given conditions")
elif p2>p1:
    print(in2,"has a greater chance of winning in the given conditions")
elif p2==p1:
    print("Both teams",in1,"and",in2,"has equal chances of winning under the given condtions")
elif p1==0 or p2==0 :
    print("Data inadequate")
else :
     print("Data inadequate")
p1
p2
currdf=pd.read_csv("ipl2019.csv")
currdf["win"]
def postpro(tem1):
    for i in np.arange(len(currdf["teams"])):
        if (currdf)["teams"][i]==tem1:
            ppwin=(currdf)["win"][i]/((currdf)["win"][i]+(currdf)["loss"][i])
    return ppwin
postpro(in1)*p1
postpro(in2)*p2
