import pandas as pd
import altair as alt
import statistics as sta
import numpy as np
wc = pd.read_csv("../input/world-cup-538/world_cup_comparisons.csv")
pd.set_option('display.min_rows',20)
wc = wc.iloc[::-1]
wc["experience"] = wc.groupby(wc.player).cumcount()
wc["attack"] = (wc[["goals_z","xg_z","crosses_z","boxtouches_z"]].sum(axis=1)/4)
wc["midfield"] = (wc[["passes_z","progpasses_z","takeons_z","progruns_z"]].sum(axis=1)/4)
wc["defense"] = (wc[["tackles_z","interceptions_z","clearances_z","blocks_z"]].sum(axis=1)/4)

def posi(x):
  if x["attack"]>x["defense"] and x["attack"]>x["midfield"]:
    return "att"
  elif x["defense"]>x["midfield"]:
    return "def"
  else:
    return "mid"

wc["position"] = wc.apply(posi,axis=1)

secondplace = wc[((wc["team"]=="West Germany")&(wc["season"]==1966))|((wc["team"]=="Italy")&(wc["season"]==1970))|((wc["team"]=="Netherlands")&(wc["season"]==1974))|((wc["team"]=="Netherlands")&(wc["season"]==1978))|((wc["team"]=="West Germany")&(wc["season"]==1982))|((wc["team"]=="West Germany")&(wc["season"]==1986))|((wc["team"]=="Argentina")&(wc["season"]==1990))|((wc["team"]=="Italy")&(wc["season"]==1994))|((wc["team"]=="Brazil")&(wc["season"]==1998))|((wc["team"]=="Germany")&(wc["season"]==2002))|((wc["team"]=="France")&(wc["season"]==2006))|((wc["team"]=="Netherlands")&(wc["season"]==2010))|((wc["team"]=="Argentina")&(wc["season"]==2014))|((wc["team"]=="Croatia")&(wc["season"]==2018))]
worldcup=wc[(wc["team"]=="England")|(wc["team"]=="Brazil")|(wc["team"]=="West Germany")|(wc["team"]=="Argentina")|(wc["team"]=="Italy")|(wc["team"]=="France")|(wc["team"]=="Spain")|(wc["team"]=="Germany")]
champions1966 = worldcup[(worldcup["season"]==1966)&(worldcup["team"]=="England")]
champions1970 = worldcup[(worldcup["season"]==1970)&(worldcup["team"]=="Brazil")]
champions1974 = worldcup[(worldcup["season"]==1974)&(worldcup["team"]=="West Germany")]
champions1978 = worldcup[(worldcup["season"]==1978)&(worldcup["team"]=="Argentina")]
champions1982 = worldcup[(worldcup["season"]==1982)&(worldcup["team"]=="Italy")]
champions1986 = worldcup[(worldcup["season"]==1986)&(worldcup["team"]=="Argentina")]
champions1990 = worldcup[(worldcup["season"]==1990)&(worldcup["team"]=="West Germany")]
champions1994 = worldcup[(worldcup["season"]==1994)&(worldcup["team"]=="Brazil")]
champions1998 = worldcup[(worldcup["season"]==1998)&(worldcup["team"]=="France")]
champions2002 = worldcup[(worldcup["season"]==2002)&(worldcup["team"]=="Brazil")]
champions2006 = worldcup[(worldcup["season"]==2006)&(worldcup["team"]=="Italy")]
champions2010 = worldcup[(worldcup["season"]==2010)&(worldcup["team"]=="Spain")]
champions2014 = worldcup[(worldcup["season"]==2014)&(worldcup["team"]=="Germany")]
champions2018 = worldcup[(worldcup["season"]==2018)&(worldcup["team"]=="France")]
champions = pd.concat([champions1966,champions1970,champions1974,champions1978,champions1982,champions1986,champions1990,champions1994,champions1998,champions2002,champions2006,champions2010,champions2014,champions2018])
champions_def = champions[champions["position"]=="def"]
champions_att = champions[champions["position"]=="att"]
champions_mid = champions[champions["position"]=="mid"]
champ_best5_def = champions_def["defense"].groupby(champions_def.season).apply(lambda grp: grp.nlargest(5).mean())
champ_best5_mid = champions_mid["midfield"].groupby(champions_mid.season).apply(lambda grp: grp.nlargest(5).mean())
champ_best5_att = champions_att["attack"].groupby(champions_att.season).apply(lambda grp: grp.nlargest(5).mean())

champ_experience_att = champions_att["experience"].groupby(champions_att.season).apply(lambda grp: grp.nlargest(5).mean())
champ_experience_def = champions_def["experience"].groupby(champions_def.season).apply(lambda grp: grp.nlargest(5).mean())
champ_experience_mid = champions_mid["experience"].groupby(champions_mid.season).apply(lambda grp: grp.nlargest(5).mean())

a = pd.merge(champ_best5_def,champ_best5_mid,how="inner",on="season")
b = pd.merge(a,champ_best5_att,how="inner",on="season")
c = pd.merge(b,champ_experience_def,how="inner",on="season")
d = pd.merge(c,champ_experience_mid,how="inner",on="season")
final_champions = pd.merge(d,champ_experience_att,how="inner",on="season").reset_index()
final_champions["status"] = "champion"
final_champions.columns = ["season","defense_performance","midfield_performance","attack_performance","defense_experience","midfield_experience","attack_experience","status"]
elite1966 = worldcup[(worldcup["season"]==1966)&(worldcup["team"]!="England")]
elite1970 = worldcup[(worldcup["season"]==1970)&(worldcup["team"]!="Brazil")]
elite1974 = worldcup[(worldcup["season"]==1974)&(worldcup["team"]!="West Germany")]
elite1978 = worldcup[(worldcup["season"]==1978)&(worldcup["team"]!="Argentina")]
elite1982 = worldcup[(worldcup["season"]==1982)&(worldcup["team"]!="Italy")]
elite1986 = worldcup[(worldcup["season"]==1986)&(worldcup["team"]!="Argentina")]
elite1990 = worldcup[(worldcup["season"]==1990)&(worldcup["team"]!="West Germany")]
elite1994 = worldcup[(worldcup["season"]==1994)&(worldcup["team"]!="Brazil")]
elite1998 = worldcup[(worldcup["season"]==1998)&(worldcup["team"]!="France")]
elite2002 = worldcup[(worldcup["season"]==2002)&(worldcup["team"]!="Brazil")]
elite2006 = worldcup[(worldcup["season"]==2006)&(worldcup["team"]!="Italy")]
elite2010 = worldcup[(worldcup["season"]==2010)&(worldcup["team"]!="Spain")]
elite2014 = worldcup[(worldcup["season"]==2014)&(worldcup["team"]!="Germany")]
elite2018 = worldcup[(worldcup["season"]==2018)&(worldcup["team"]!="France")]
elite = pd.concat([elite1966,elite1970,elite1974,elite1978,elite1982,elite1986,elite1990,elite1994,elite1998,elite2002,elite2006,elite2010,elite2014,elite2018])
elite_def = elite[elite["position"]=="def"]
elite_att = elite[elite["position"]=="att"]
elite_mid = elite[elite["position"]=="mid"]
elite_best5_def = elite_def["defense"].groupby(elite_def.season).mean()
elite_best5_mid = elite_mid["midfield"].groupby(elite_mid.season).mean()
elite_best5_att = elite_att["attack"].groupby(elite_att.season).mean()

elite_experience_att = elite_att["experience"].groupby(elite_att.season).mean()
elite_experience_def = elite_def["experience"].groupby(elite_def.season).mean()
elite_experience_mid = elite_mid["experience"].groupby(elite_mid.season).mean()

e = pd.merge(elite_best5_def,elite_best5_mid,how="inner",on="season")
f = pd.merge(e,elite_best5_att,how="inner",on="season")
g = pd.merge(f,elite_experience_def,how="inner",on="season")
h = pd.merge(g,elite_experience_mid,how="inner",on="season")
final_elite = pd.merge(h,elite_experience_att,how="inner",on="season").reset_index()
final_elite["status"] = "elite"
final_elite.columns = ["season","defense_performance","midfield_performance","attack_performance","defense_experience","midfield_experience","attack_experience","status"]
splace_def = secondplace[secondplace["position"]=="def"]
splace_att = secondplace[secondplace["position"]=="att"]
splace_mid = secondplace[secondplace["position"]=="mid"]
s_best5_def = splace_def["defense"].groupby(splace_def.season).mean()
s_best5_mid = splace_mid["midfield"].groupby(splace_mid.season).mean()
s_best5_att = splace_att["attack"].groupby(splace_att.season).mean()

s_experience_att = splace_att["experience"].groupby(splace_att.season).apply(lambda grp: grp.nlargest(5).mean())
s_experience_def = splace_def["experience"].groupby(splace_def.season).apply(lambda grp: grp.nlargest(5).mean())
s_experience_mid = splace_mid["experience"].groupby(splace_mid.season).apply(lambda grp: grp.nlargest(5).mean())

i = pd.merge(s_best5_def,champ_best5_mid,how="inner",on="season")
j = pd.merge(i,s_best5_att,how="inner",on="season")
k = pd.merge(j,s_experience_def,how="inner",on="season")
l = pd.merge(k,s_experience_mid,how="inner",on="season")
final_2 = pd.merge(l,s_experience_att,how="inner",on="season").reset_index()
final_2["status"] = "second"
final_2.columns = ["season","defense_performance","midfield_performance","attack_performance","defense_experience","midfield_experience","attack_experience","status"]
frames = [final_champions,final_2,final_elite]
soccer = pd.concat(frames)
soccer["team"]=["England","Brazil","Germany","Argentina","Italy","Argentina","Germany","Brazil","France","Brazil","Italy","Spain","Germany","France","Germany","Italy","Netherlands","Netherlands","Germany","Germany","Argentina","Italy","Brazil","Germany","France","Netherlands","Argentina","Croatia","Elite","Elite","Elite","Elite","Elite","Elite","Elite","Elite","Elite","Elite","Elite","Elite","Elite","Elite"]
labs= ["champion","second","elite"]
cols= ["#f3e062","#B5B6F4","#DFDDDC"]

alt.Chart(soccer,title="FIFA World Cup (1966-2018) - 1st & 2nd places").mark_point(filled=True).encode(
    alt.X("defense_performance",scale=alt.Scale(zero=False)),
    alt.Y("attack_performance",scale=alt.Scale(zero=False, padding=1)),
    color=alt.Color("status",scale=alt.Scale(domain=labs,range=cols)),
    size="midfield_performance",
    tooltip=["season","team","defense_experience","midfield_experience","attack_experience"]
    ).properties(width=500,height=600)
wc18 = wc[(wc["season"]==2018)&((wc["team"]=="Spain")|(wc["team"]=="France")|(wc["team"]=="Argentina")|(wc["team"]=="Belgium")|(wc["team"]=="Brazil")|(wc["team"]=="England")|(wc["team"]=="Germany")|(wc["team"]=="Croatia"))]

wc18_def = wc18[wc18["position"]=="def"]
wc18_att = wc18[wc18["position"]=="att"]
wc18_mid = wc18[wc18["position"]=="mid"]

wc18_best5_def = wc18["defense"].groupby(wc18.team).apply(lambda grp: grp.nlargest(5).mean())
wc18_best5_mid = wc18["midfield"].groupby(wc18.team).apply(lambda grp: grp.nlargest(5).mean())
wc18_best5_att = wc18["attack"].groupby(wc18.team).apply(lambda grp: grp.nlargest(5).mean())

wc18_experience_att = wc18_att["experience"].groupby(wc18_att.team).apply(lambda grp: grp.nlargest(5).mean())
wc18_experience_def = wc18_def["experience"].groupby(wc18_def.team).apply(lambda grp: grp.nlargest(5).mean())
wc18_experience_mid = wc18_mid["experience"].groupby(wc18_mid.team).apply(lambda grp: grp.nlargest(5).mean())

m = pd.merge(wc18_best5_def,wc18_best5_mid,how="inner",on="team")
n = pd.merge(m,wc18_best5_att,how="inner",on="team")
o = pd.merge(n,wc18_experience_def,how="inner",on="team")
p = pd.merge(o,wc18_experience_mid,how="inner",on="team")
final_wc18 = pd.merge(p,wc18_experience_att,how="inner",on="team").reset_index()
final_wc18["status"] = ["elite","elite","elite","second","elite","champion","elite","elite"]
final_wc18.columns = ["team","defense_performance","midfield_performance","attack_performance","defense_experience","midfield_experience","attack_experience","status"]

statuz= ["champion","second","elite"]
cols= ["#f3e062","#B5B6F4","#DFDDDC"]

alt.Chart(final_wc18,title="2018 FIFA World Cup").mark_point(filled=True).encode(
    alt.X("defense_performance",scale=alt.Scale(domain=[.2,2.2])),
    alt.Y("attack_performance",scale=alt.Scale(domain=[-.4,2.4])),
    color=alt.Color("status",scale=alt.Scale(domain=statuz,range=cols)),
    size="midfield_performance",
    tooltip=["team","defense_experience","midfield_experience","attack_experience"]
    ).properties(width=500,height=600)
wcBrazil = wc[(wc["team"]=="Brazil")]

wcBrazil_def = wcBrazil[wcBrazil["position"]=="def"]
wcBrazil_att = wcBrazil[wcBrazil["position"]=="att"]
wcBrazil_mid = wcBrazil[wcBrazil["position"]=="mid"]

wcBrazil_best5_def = wcBrazil["defense"].groupby(wcBrazil.season).apply(lambda grp: grp.nlargest(5).mean())
wcBrazil_best5_mid = wcBrazil["midfield"].groupby(wcBrazil.season).apply(lambda grp: grp.nlargest(5).mean())
wcBrazil_best5_att = wcBrazil["attack"].groupby(wcBrazil.season).apply(lambda grp: grp.nlargest(5).mean())

wcBrazil_experience_att = wcBrazil_att["experience"].groupby(wcBrazil_att.season).apply(lambda grp: grp.nlargest(5).mean())
wcBrazil_experience_def = wcBrazil_def["experience"].groupby(wcBrazil_def.season).apply(lambda grp: grp.nlargest(5).mean())
wcBrazil_experience_mid = wcBrazil_mid["experience"].groupby(wcBrazil_mid.season).apply(lambda grp: grp.nlargest(5).mean())

q = pd.merge(wcBrazil_best5_def,wcBrazil_best5_mid,how="inner",on="season")
r = pd.merge(q,wcBrazil_best5_att,how="inner",on="season")
s = pd.merge(r,wcBrazil_experience_def,how="inner",on="season")
t = pd.merge(s,wcBrazil_experience_mid,how="inner",on="season")
final_wcBrazil = pd.merge(t,wcBrazil_experience_att,how="inner",on="season").reset_index()
final_wcBrazil["status"] = ["eliminated","champion","eliminated","eliminated","eliminated","eliminated","eliminated","champion","second","champion","eliminated","eliminated","eliminated","2018"]
final_wcBrazil.columns = ["season","defense_performance","midfield_performance","attack_performance","defense_experience","midfield_experience","attack_experience","status"]

statuz= ["champion","second","eliminated","2018"]
cols= ["#f3e062","#B5B6F4","#DFDDDC","#545454"]
alt.Chart(final_wcBrazil,title="Brazil Squads at FIFA World Cup (1966-2018").mark_point(filled=True).encode(
    alt.X("defense_performance",scale=alt.Scale(domain=[.2,2.2])),
    alt.Y("attack_performance",scale=alt.Scale(domain=[-.4,2.4])),
    color=alt.Color("status",scale=alt.Scale(domain=statuz,range=cols)),
    size="midfield_performance",
    tooltip=["season","defense_experience","midfield_experience","attack_experience"]
    ).properties(width=500,height=600)
wcGermany = wc[(wc["team"]=="Germany")|(wc["team"]=="West Germany")]

wcGermany_def = wcGermany[wcGermany["position"]=="def"]
wcGermany_att = wcGermany[wcGermany["position"]=="att"]
wcGermany_mid = wcGermany[wcGermany["position"]=="mid"]

wcGermany_best5_def = wcGermany["defense"].groupby(wcGermany.season).apply(lambda grp: grp.nlargest(5).mean())
wcGermany_best5_mid = wcGermany["midfield"].groupby(wcGermany.season).apply(lambda grp: grp.nlargest(5).mean())
wcGermany_best5_att = wcGermany["attack"].groupby(wcGermany.season).apply(lambda grp: grp.nlargest(5).mean())

wcGermany_experience_att = wcGermany_att["experience"].groupby(wcGermany_att.season).apply(lambda grp: grp.nlargest(5).mean())
wcGermany_experience_def = wcGermany_def["experience"].groupby(wcGermany_def.season).apply(lambda grp: grp.nlargest(5).mean())
wcGermany_experience_mid = wcGermany_mid["experience"].groupby(wcGermany_mid.season).apply(lambda grp: grp.nlargest(5).mean())

u = pd.merge(wcGermany_best5_def,wcGermany_best5_mid,how="inner",on="season")
v = pd.merge(u,wcGermany_best5_att,how="inner",on="season")
w = pd.merge(v,wcGermany_experience_def,how="inner",on="season")
x = pd.merge(w,wcGermany_experience_mid,how="inner",on="season")
final_wcGermany = pd.merge(x,wcGermany_experience_att,how="inner",on="season").reset_index()
final_wcGermany["status"] = ["second","eliminated","champion","eliminated","second","second","champion","eliminated","eliminated","second","eliminated","eliminated","champion","2018"]
final_wcGermany.columns = ["season","defense_performance","midfield_performance","attack_performance","defense_experience","midfield_experience","attack_experience","status"]

statuz= ["champion","second","eliminated","2018"]
cols= ["#f3e062","#B5B6F4","#DFDDDC","#545454"]

alt.Chart(final_wcGermany,title="Germany Squads at FIFA World Cup (1966-2018)").mark_point(filled=True).encode(
    alt.X("defense_performance",scale=alt.Scale(domain=[.2,2.2])),
    alt.Y("attack_performance",scale=alt.Scale(domain=[-.4,2.4])),
    color=alt.Color("status",scale=alt.Scale(domain=statuz,range=cols)),
    size="midfield_performance",
    tooltip=["season","defense_experience","midfield_experience","attack_experience"]
    ).properties(width=500,height=600)
wcFrance = wc[(wc["team"]=="France")]

wcFrance_def = wcFrance[wcFrance["position"]=="def"]
wcFrance_att = wcFrance[wcFrance["position"]=="att"]
wcFrance_mid = wcFrance[wcFrance["position"]=="mid"]

wcFrance_best5_def = wcFrance["defense"].groupby(wcFrance.season).apply(lambda grp: grp.nlargest(5).mean())
wcFrance_best5_mid = wcFrance["midfield"].groupby(wcFrance.season).apply(lambda grp: grp.nlargest(5).mean())
wcFrance_best5_att = wcFrance["attack"].groupby(wcFrance.season).apply(lambda grp: grp.nlargest(5).mean())

wcFrance_experience_att = wcFrance_att["experience"].groupby(wcFrance_att.season).apply(lambda grp: grp.nlargest(5).mean())
wcFrance_experience_def = wcFrance_def["experience"].groupby(wcFrance_def.season).apply(lambda grp: grp.nlargest(5).mean())
wcFrance_experience_mid = wcFrance_mid["experience"].groupby(wcFrance_mid.season).apply(lambda grp: grp.nlargest(5).mean())

y = pd.merge(wcFrance_best5_def,wcFrance_best5_mid,how="inner",on="season")
z = pd.merge(y,wcFrance_best5_att,how="inner",on="season")
aa = pd.merge(z,wcFrance_experience_def,how="inner",on="season")
bb = pd.merge(aa,wcFrance_experience_mid,how="inner",on="season")
final_wcFrance = pd.merge(bb,wcFrance_experience_att,how="inner",on="season").reset_index()
final_wcFrance["status"] = ["eliminated","eliminated","eliminated","eliminated","champion","eliminated","second","eliminated","eliminated","2018"]
final_wcFrance.columns = ["season","defense_performance","midfield_performance","attack_performance","defense_experience","midfield_experience","attack_experience","status"]

statuz= ["champion","second","eliminated","2018"]
cols= ["#f3e062","#B5B6F4","#DFDDDC","#F1C611"]

alt.Chart(final_wcFrance,title="France Squads at FIFA World Cup (1966-2018)").mark_point(filled=True).encode(
    alt.X("defense_performance",scale=alt.Scale(domain=[.2,2.2])),
    alt.Y("attack_performance",scale=alt.Scale(domain=[-.4,2.4])),
    color=alt.Color("status",scale=alt.Scale(domain=statuz,range=cols)),
    size="midfield_performance",
    tooltip=["season","defense_experience","midfield_experience","attack_experience"]
    ).properties(width=500,height=600)
wcMexico = wc[(wc["team"]=="Mexico")]

wcMexico_def = wcMexico[wcMexico["position"]=="def"]
wcMexico_att = wcMexico[wcMexico["position"]=="att"]
wcMexico_mid = wcMexico[wcMexico["position"]=="mid"]

wcMexico_best5_def = wcMexico["defense"].groupby(wcMexico.season).apply(lambda grp: grp.nlargest(5).mean())
wcMexico_best5_mid = wcMexico["midfield"].groupby(wcMexico.season).apply(lambda grp: grp.nlargest(5).mean())
wcMexico_best5_att = wcMexico["attack"].groupby(wcMexico.season).apply(lambda grp: grp.nlargest(5).mean())

wcMexico_experience_att = wcMexico_att["experience"].groupby(wcMexico_att.season).apply(lambda grp: grp.nlargest(5).mean())
wcMexico_experience_def = wcMexico_def["experience"].groupby(wcMexico_def.season).apply(lambda grp: grp.nlargest(5).mean())
wcMexico_experience_mid = wcMexico_mid["experience"].groupby(wcMexico_mid.season).apply(lambda grp: grp.nlargest(5).mean())

cc = pd.merge(wcMexico_best5_def,wcMexico_best5_mid,how="inner",on="season")
dd = pd.merge(cc,wcMexico_best5_att,how="inner",on="season")
ee = pd.merge(dd,wcMexico_experience_def,how="inner",on="season")
ff = pd.merge(ee,wcMexico_experience_mid,how="inner",on="season")
final_wcMexico = pd.merge(ff,wcMexico_experience_att,how="inner",on="season").reset_index()
final_wcMexico["status"] = ["Fase de Grupos","Cuartos de Final","Fase de Grupos","Cuartos de Final","Octavos de Final","Octavos de Final","Octavos de Final","Octavos de Final","Octavos de Final","Octavos de Final","2018"]
final_wcMexico.columns = ["season","defense_performance","midfield_performance","attack_performance","defense_experience","midfield_experience","attack_experience","status"]

statuz= ["Fase de Grupos","Octavos de Final","Cuartos de Final","2018"]
cols= ["#DFDDDC","#B5B6F4","#F4A5DC","#7577FA"]

alt.Chart(final_wcMexico,title="Mexico Squads at FIFA World Cup (1966-2018)").mark_point(filled=True).encode(
    alt.X("defense_performance",scale=alt.Scale(domain=[.2,2.2])),
    alt.Y("attack_performance",scale=alt.Scale(domain=[-.4,2.4])),
    color=alt.Color("status",scale=alt.Scale(domain=statuz,range=cols)),
    size="midfield_performance",
    tooltip=["season","defense_experience","midfield_experience","attack_experience"]
    ).properties(width=500,height=600)