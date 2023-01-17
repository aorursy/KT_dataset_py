import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

plt.style.use(['ggplot'])
plt.rcParams['figure.figsize']= (15,7)
df = pd.read_csv("/kaggle/input/us-police-shootings/shootings.csv")
df.head(3)
plt.subplot(121)
plt.pie(df["gender"].value_counts(), labels=["Male", "Female"], shadow=True, autopct="%.2f %%", explode=[0.2, 0.0])
plt.legend()
plt.subplot(122)
sns.boxplot(y="gender",x="age", data=df, orient='h')
plt.show()
scores = {'M_true':0, "M_false":0, "F_true":0, 'F_false':0}
for i in df[["gender","signs_of_mental_illness"]].values:
    if i[1]:
        scores[i[0]+"_true"] += 1
    else:
        scores[i[0]+"_false"] += 1

plt.subplot(121)
plt.pie([*scores.values()][:2], labels=["True", "False"], autopct="%.2f %%", explode=[0.1, 0.0], shadow=True)
plt.title("Males with signs")

plt.subplot(122)
plt.pie([*scores.values()][2:], labels=["True", "False"], autopct="%.2f %%", explode=[0.1, 0.0], shadow=True)
plt.title("Females with signs")
plt.show()
plt.subplot(122)
plt.pie(df["race"].value_counts(), labels=df["race"].value_counts().index, shadow=True, autopct="%.2f %%", explode=[0.1,0.1,0.1,0.1,0.3,0.4])
plt.legend()

plt.subplot(121)
sns.boxplot(x="race",y="age", data=df)
plt.show()
states = [*df["state"].value_counts().index]
races = df["race"].value_counts().index

scores = {r:pd.Series([0 for _ in range(len(states))]) for r in [*races, "all"]}
scores["state"] = states

for i in range(len(df)):
    scores[df["race"][i]][states.index(df["state"][i])] += 1
    scores["all"][states.index(df["state"][i])] += 1
    
scores = pd.DataFrame.from_dict(scores)
sns.barplot(x="state", y="all", data=scores)
plt.ylabel("Deaths")
plt.xlabel("State")
plt.show()
_, axes = plt.subplots(3,2, sharex=True, figsize=(35,20))
sns.barplot(x="state", y="White", data=scores,ax=axes[0,0])
sns.barplot(x="state", y="Black", data=scores,ax=axes[0,1])
sns.barplot(x="state", y="Asian", data=scores,ax=axes[1,0])
sns.barplot(x="state", y="Hispanic", data=scores,ax=axes[1,1])
sns.barplot(x="state", y="Native", data=scores,ax=axes[2,0])
sns.barplot(x="state", y="Other", data=scores,ax=axes[2,1])

plt.show()
state = "FL"

cities = [*set(df[df["state"] == state]["city"])]
races = df["race"].value_counts().index

scores = {r:pd.Series([0 for _ in range(len(cities))]) for r in [*races, "all"]}
scores["city"] = cities
for i in range(len(df)):
    if df["state"][i] == state:
        scores[df["race"][i]][cities.index(df["city"][i])] += 1
        scores["all"][cities.index(df["city"][i])] += 1
    
scores = pd.DataFrame.from_dict(scores)

_, axes = plt.subplots(3,2, sharex=True, sharey=True, figsize=(30,10))
for i in axes.flat:
    i.get_xaxis().set_ticks([])

sns.lineplot(x="city", y="White", data=scores,ax=axes[0,0])
sns.lineplot(x="city", y="Black", data=scores,ax=axes[0,1])
sns.lineplot(x="city", y="Asian", data=scores,ax=axes[1,0])
sns.lineplot(x="city", y="Hispanic", data=scores,ax=axes[1,1])
sns.lineplot(x="city", y="Native", data=scores,ax=axes[2,0])
sns.lineplot(x="city", y="Other", data=scores,ax=axes[2,1])

axes[0,0].set_title("White")
axes[0,1].set_title("Black")
axes[1,0].set_title("Asian")
axes[1,1].set_title("Hispanic")
axes[2,0].set_title("Native")
axes[2,1].set_title("Other")

plt.show()

print(f"Top 3 Cities (peaks in graph above) : Black    : {[*scores.sort_values('Black')[::-1]['city'][:3]]}")
print(f"Top 3 Cities (peaks in graph above) : Hispanic : {[*scores.sort_values('Hispanic')[::-1]['city'][:3]]}")
print(f"Top 3 Cities (peaks in graph above) : White    : {[*scores.sort_values('White')[::-1]['city'][:3]]}")
body_cam_off = df[df["body_camera"] == False]
body_cam_on = df[df["body_camera"] == True]

_, axes = plt.subplots(2, 2, figsize=(15,15))

axes[0, 0].pie(body_cam_on["race"].value_counts(), labels=body_cam_on["race"].value_counts().index, autopct="%.2f %%", shadow=True, explode=[0.1, 0.1, 0.1, 0.3, 0.2, 0.1])
axes[0, 0].set_title(f"Ratio of body camera on [{len(body_cam_on)} victims]")

axes[0, 1].pie(body_cam_off["race"].value_counts(), labels=body_cam_off["race"].value_counts().index, autopct="%.2f %%", shadow=True, explode=[0.1, 0.1, 0.1, 0.3, 0.2, 0.1])
axes[0, 1].set_title(f"Ratio of body camera off [{len(body_cam_off)} victims]")

sns.barplot(body_cam_on["race"].value_counts(), body_cam_on["race"].value_counts().index, ax=axes[1, 0])
axes[1, 0].set_title(f"Ratio of body camera on [{len(body_cam_on)} victims]")

sns.barplot(body_cam_off["race"].value_counts(), body_cam_off["race"].value_counts().index, ax=axes[1, 1])
axes[1, 1].set_title(f"Ratio of body camera off [{len(body_cam_off)} victims]")

plt.show()
_, axes = plt.subplots(2,1, figsize=(20, 20))
sns.barplot(x=body_cam_on["state"].value_counts(), y=body_cam_on["state"].value_counts().index, ax=axes[0])
sns.barplot(x=body_cam_off["state"].value_counts(), y=body_cam_off["state"].value_counts().index, ax=axes[1])

axes[0].set_title("Body Camera On")
axes[1].set_title("Body Camera Off")
plt.show()
weapons = [*df["armed"]]
_, ax = plt.subplots(1,1, figsize=(17,10))
wordcloud = WordCloud(colormap="OrRd_r", width=400, height=400).generate(" ".join(weapons))
ax.imshow(wordcloud, interpolation='bilinear')
ax.axis('off')
ax.set_title("Weapons", fontsize=30)
plt.show()
unarmed = df[df["arms_category"] == "Unarmed"]
unarmed_notFleeing = unarmed[unarmed["flee"] == "Not fleeing"]
unarmed_notFleeing_notAttacking = unarmed_notFleeing[unarmed_notFleeing["threat_level"] != "attack"]

plt.pie(df["arms_category"].value_counts(), labels=df["arms_category"].value_counts().index, autopct="%.2f %%", shadow=True, explode=[0.1 + (0.02*i) for i in range(len(set(df["arms_category"])))])
plt.legend()
plt.show()

_, axes = plt.subplots(1,3, figsize=(30,10))
sns.countplot(x="flee", data=unarmed, ax=axes[0])
sns.countplot(x="threat_level", data=unarmed_notFleeing, ax=axes[1])
axes[2].pie(unarmed_notFleeing_notAttacking["race"].value_counts(), labels=unarmed_notFleeing_notAttacking["race"].value_counts().index, autopct="%.2f %%", shadow=True, explode=[0.1, 0.1, 0.1, 0.1, 0.2, 0.3])


axes[0].set_title("Unarmed and how they are fleeing")
axes[1].set_title("Unarmed, not fleeing, by threat level")
axes[2].set_title("Unarmed, not fleeing, not attacking, by race")
axes[2].legend()
plt.show()