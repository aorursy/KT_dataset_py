import matplotlib.pyplot as plt

import seaborn as sns

import pandas as pd
tg= pd.read_csv("../input/guardian_tweets_exe (1).csv")

display(tg.head(3))
ute= pd.read_csv("../input/user_tweets_exe.csv")

display(ute.head(3))
tgute=pd.merge(tg, ute, left_on = ['id'], right_on=['article_id'])   #merging id from users and aticle_id in order to evaluate the tweet of guardian and tweet of user time
tgute["new_x"] = pd.to_datetime(tgute["timestamp_x"])

tgute["new_y"]= pd.to_datetime(tgute["timestamp_y"])



#x tweet guardian

#y tweet users







tgute["new_x"] = pd.to_datetime(tgute["timestamp_x"])

tgute["new_y"]= pd.to_datetime(tgute["timestamp_y"])



    

tgute['delay'] = tgute["new_y"]-tgute["new_x"]

print(type(tgute["new_y"]))

tgute.head(10)
tgute['delay'].describe()
data = tgute[['user_x','user_y','delay']]

data1 = data.tail(1000)

print(type(data1['delay']))

#.astype(int)

print(type(data1['delay']))



temporanea = []

for i in range(len(data)):

    temporanea.append(data['delay'][i].total_seconds()/3600)


fig = plt.figure(figsize=(16,7))

plt.title('Delay [Tweet Time user - Tweet Time The Guardian]')

sns.distplot(temporanea)

plt.xlabel('Hours')
#tweet del the guardian istogramma per orario(ore, minuti, secondi) ma senza data 





temporanea1 = []

for i in range(len(tgute)):

    temporanea1.append(

((tgute["new_x"][i].hour*60+tgute["new_x"][i].minute)*60+tgute["new_x"][i].second)/3600

    )

    

import matplotlib.pyplot as plt

import seaborn as sns

fig = plt.figure(figsize=(16,7))

plt.title('Distribution of tweet by The Guardian')

sns.distplot(temporanea1)    

plt.xlabel('Hours')
#tweet degli utenti istogramma per orario(ore, minuti, secondi) ma senza data 



temporanea2 = []

for i in range(len(tgute)):

    temporanea2.append(

((tgute["new_y"][i].hour*60+tgute["new_y"][i].minute)*60+tgute["new_y"][i].second)/3600

    )

    

import matplotlib.pyplot as plt

import seaborn as sns

fig = plt.figure(figsize=(16,7))

sns.distplot(temporanea2) 

plt.title('Distribution of tweet by Users')

plt.xlabel('Hours')