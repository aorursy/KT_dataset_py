import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
batting = pd.read_csv('../input/batting.csv')
player = pd.read_csv('../input/player.csv')
salary = pd.read_csv('../input/salary.csv')
#work on recent times
start_year = 1990
end_year = 2015
ext='-01-01'
player = player[(player["debut"] > str(start_year)+ext) & (player["debut"] < str(end_year)+ext)]
batting = batting[batting["year"] > start_year]
salary = salary[salary["year"] > start_year]

#merge stints, too annoying to deal with separately
batting = batting.groupby(["player_id","year"]).sum()
batting = batting.reset_index()
batting["single"] = batting["h"] - (batting["double"] + batting["triple"] + batting["hr"])
batting["tb"] = batting["single"] + 2*batting["double"] + 3*batting["triple"] + 4*batting["hr"]
A = batting["h"] + batting["bb"] - batting["cs"] + batting["hbp"] - batting["g_idp"] #total bases
B = batting["tb"] + 0.26*(batting["bb"] - batting["ibb"] + batting["hbp"])           #base on balls
C = 0.52*(batting["sh"] + batting["sf"] + batting["sb"])                             #sacrifices
D = batting["ab"] + batting["bb"] + batting["hbp"] + batting["sh"] + batting["sf"]   #total at bats
batting["rc"] = (A*B + C)/D

#For players that don't have the required stats for complex RC, use simple RC
batting.loc[batting["rc"].isnull()==True,"rc"]=(batting["h"]+batting["bb"])*batting["tb"]/(batting["ab"]+batting["bb"])

#set remaining isnan and isinf values to 0
batting.loc[batting["rc"].isnull()==True,"rc"] = 0
batting.loc[np.isinf(batting["rc"]),"rc"] = 0
def get_rc_of_payday(plyr_id,player,salary,batting,SITR,SMT,GPMT,rc_post_avg):
    plyr_slry = salary[salary["player_id"] == plyr_id]
    plyr_slry["salary_ratio"] = plyr_slry["salary"].div(plyr_slry["salary"].shift(1))
    check_payday = (plyr_slry["salary_ratio"] > SITR) & (plyr_slry["salary"] > SMT)
    payday_year = plyr_slry["year"].loc[check_payday].values
    plyr_bat = batting.loc[batting["player_id"] == plyr_id]
    data = []
    for i,y in enumerate(payday_year):
        #check that enough games were played for accurate rc in current and previous year?
        if (np.sum(plyr_bat["g"].loc[plyr_bat["year"] == y]) > GPMT) & (np.sum(plyr_bat["g"].loc[plyr_bat["year"] == y-1]) > GPMT):
            yrs = plyr_bat["year"]
            rc_post = plyr_bat.loc[(yrs>=y)&(yrs<y+rc_post_avg),"rc"].mean()
            rc_prev = plyr_bat.loc[yrs==y-1,"rc"].values[0]
            rcratio = rc_post/rc_prev
            if np.isnan(rcratio) == 0:
                data.append(plyr_id)
                data.append(player.loc[player["player_id"] == plyr_id,["name_first","name_last"]].values[0].sum())
                data.append(y)
                data.append(rc_post)
                data.append(rcratio)
                data.append(plyr_slry.loc[plyr_slry["year"]==y,"salary_ratio"].values[0])
    return data
#Let's conduct this analysis and store the results in a new dataframe "data".
slry_inc_thresh_ratio = 2      #min increase ratio player received
slry_min_thresh = 2e6          #min final salary after the payday
gp_thresh = 100                #min number of games played
rc_post_avg = 4                   #when calc'ing rc_i, avg. over how many years? 

columns = ["player_id","player_name","year","rcpost","R","payday"]
data = pd.DataFrame(columns=columns)

for i,plyr_id in enumerate(player["player_id"]):
    d = get_rc_of_payday(plyr_id,player,salary,batting,slry_inc_thresh_ratio,slry_min_thresh,gp_thresh,rc_post_avg)
    while len(d) > 0: #gotta be a better way to do this
        data = data.append({col:d[i] for i,col in enumerate(columns)}, ignore_index=True)
        d = d[len(columns):]
#plot
font=15
plt.figure(figsize=(10,5))
norm = (data["rcpost"] - batting["rc"].loc[batting["rc"] != 0.].mean())/batting["rc"].loc[batting["rc"] != 0.].std()
plt.scatter(data["payday"],data["R"], c=norm, cmap='rainbow',lw=0)
cbar = plt.colorbar()
cbar.set_label("$(rc_{post} - rc_{post,mean}) / \sigma_{rc_{post}}$",size=font)
plt.xlabel('fractional salary increase',fontsize=font)
plt.ylabel('R = rc$_{post}$ / rc$_{prev}$',fontsize=font)
plt.title('Runs Created Ratio vs. salary increase',fontsize=font)

#some stats
rcmax = data["R"].max()
rcmin = data["R"].min()
paymax = data["payday"].max()
print("Best rc ratio is "+str(rcmax)+" from "+str(data.loc[data["R"] == rcmax,["player_name","player_id"]].values[0]))
print("Worst rc ratio is "+str(rcmin)+" from "+str(data.loc[data["R"] == rcmin,["player_name","player_id"]].values[0]))
print("Biggest Salary increase is "+str(paymax)+"x from "+str(data.loc[data["payday"] == paymax,["player_name","player_id"]].values[0]))
print("Fraction of players with rc_post/rc_prev < 0.85:",data[data["R"]<0.85].shape[0]/float(data["R"].shape[0]))
print("Fraction of players with rc_post/rc_prev > 1.15:",data[data["R"]>1.15].shape[0]/float(data["R"].shape[0]))
print("Fraction of players with rc_post >1 std of the mean:",norm[norm>1].shape[0]/float(norm.shape[0]))
#Top R values
data[data["R"]>1.5].sort_values(by="R",ascending=False)
id='moustmi01'
batting[batting["player_id"] == id]
plyr_bat = batting.loc[batting["player_id"] == id,["player_id","year","g","ab","h","hr","rc"]]
plyr_bat
salary[salary["player_id"] == id]