import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
df_full=pd.read_csv('../input/FullData.csv')

df_ronaldo_messi=df_full.head(2)

df_ronaldo_messi.head()
def plot_player(player):

    img = plt.imread('../input/'+'Pictures/'+player+'.png')

    plt.figure(figsize=(2.5, 2.5))

    plt.imshow(img, aspect='auto')

    plt.axis('off')

    plt.title(player)

    plt.show() 
plot_player('Lionel Messi')
plot_player('Cristiano Ronaldo')
cristiano_messi=df_ronaldo_messi.max()

cristiano_messi
df=df_full[2:]

df.head(2)

#Removing Messi and Ronaldo's entry 
df=df.append(cristiano_messi,ignore_index=True)

df[df['Name'] == 'Lionel Messi']

#adding Mr.Cristiano Messi's entry
df.Name[df.Name=='Lionel Messi'] = 'Cristiano Messi'

df[df['Name'] == 'Cristiano Messi']
df1=df[df['Rating'] > 88]

df1=df1[df1['Age'] < 33]

df1=df1[df1['National_Position'] != 'GK'].head(15) #Excluding Goalkeepers

df1.head(1)
df2 = df1.drop(["Name", "Nationality", "National_Position","National_Kit","Club","Club_Position","Club_Kit","Club_Joining","Contract_Expiry","Preffered_Foot","Birth_Date","Preffered_Position","Work_Rate","Height","Weight"], axis = 1)

df2.head(1) 

#Considering only the numeric data for further calculations
from sklearn.cluster import KMeans

km = KMeans(n_clusters=3, init='k-means++', n_init=10)

km.fit(df2)

x = km.fit_predict(df2)

x
df1["Cluster"]= x

df1.head(15)
df1=df1[df1['Cluster'] == 1].head(15)

df1.head(15)
for name in df1['Name']:

    try:

        plot_player(name)

    except:

        pass

plot_player('Luis Surez')

plot_player('Mesut zil')

plot_player('Gonzalo Higuan')

plot_player('Sergio Agero')

#Had to hard code some names as the names of the image files didn't match with the names in the csv file