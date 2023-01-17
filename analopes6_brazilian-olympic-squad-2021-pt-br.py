import pandas as pd

import numpy as np

pd.plotting.register_matplotlib_converters()

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

print("Setup Complete")
df = pd.read_csv('../input/fifa19/data.csv', index_col='ID')

df.head()
brazilian_players = df.loc[df['Nationality']=='Brazil']

brazilian_players.head(10)
sns.kdeplot(data=brazilian_players['Age'],shade=True)

plt.title('Distribuição das idades dos jogadores brasileiros')

plt.show()
plt.figure(figsize=(12,6))

plt.subplot(1,2,1)

plt.title('Comportamento do potencial em função da idade')

sns.regplot(x=brazilian_players['Age'],y=brazilian_players['Potential'])

plt.subplot(1,2,2)

plt.title('Comportamento do overall em função da idade')

sns.regplot(x=brazilian_players['Age'],y=brazilian_players['Overall'])



plt.show()
mask = brazilian_players['Age'] < 22

under22 = brazilian_players.loc[mask]
plt.figure(figsize=(10,6))

plt.title('Distribuição das idades dos jogadores brasileiros com idade olímpica em 2021')

sns.kdeplot(data=under22['Age'],shade=True)

plt.show()
plt.figure(figsize=(10,6))

plt.title('Distribuição do potencial e overall dos jogadores brasileiros com idade olímpica em 2021')

sns.kdeplot(data=under22['Potential'], label='Potential', shade=True)

sns.kdeplot(data=under22['Overall'], label='Overall', shade=True)

plt.show()
def area(df):

    ans = []

    gk = ['GK']

    dff = ['RB','RCB','LB','LCB','CB']

    mid = ['CDM','CM','CAM','LM','RM','RCM','LCM']

    atk = ['CF','ST','RS','LS','RW','LW']

    for index, row in df.iterrows():

        if gk.count(row['Position']) > 0:

            ans.append('GK')

        elif dff.count(row['Position']) > 0:

            ans.append('DFF')

        elif mid.count(row['Position']) > 0:

            ans.append('MID')

        elif atk.count(row['Position']) > 0:

            ans.append('ATK')

        else:

            ans.append('None')

            

    return ans

    
under22['Area'] = area(under22)
under22p = under22.sort_values(['Potential'], ascending=False)

list_gk_p = under22p.loc[under22p['Area']=='GK']

list_dff_p = under22p.loc[under22p['Area']=='DFF']

list_mid_p = under22p.loc[under22p['Area']=='MID']

list_atk_p = under22p.loc[under22p['Area']=='ATK']
under22o = under22.sort_values(['Overall'], ascending=False)

list_gk_o = under22o.loc[under22p['Area']=='GK']

list_dff_o = under22o.loc[under22p['Area']=='DFF']

list_mid_o = under22o.loc[under22p['Area']=='MID']

list_atk_o = under22o.loc[under22p['Area']=='ATK']
#Desenho do campo de futebol

#fonte: http://petermckeever.com/2019/01/plotting-pitches-in-python/



def draw_pitch(pitch, line, orientation,view):

    fig,ax = plt.subplots(figsize=(20.8,13.6))

    plt.xlim(-1,105)

    plt.ylim(-1,69)

    ax.axis('off') # this hides the x and y ticks

    

    # side and goal lines #

    ly1 = [0,0,68,68,0]

    lx1 = [0,104,104,0,0]



    plt.plot(lx1,ly1,color=line,zorder=5)



    # boxes, 6 yard box and goals



    #outer boxes#

    ly2 = [13.84,13.84,54.16,54.16] 

    lx2 = [104,87.5,87.5,104]

    plt.plot(lx2,ly2,color=line,zorder=5)



    ly3 = [13.84,13.84,54.16,54.16] 

    lx3 = [0,16.5,16.5,0]

    plt.plot(lx3,ly3,color=line,zorder=5)



            #goals#

    ly4 = [30.34,30.34,37.66,37.66]

    lx4 = [104,104.2,104.2,104]

    plt.plot(lx4,ly4,color=line,zorder=5)



    ly5 = [30.34,30.34,37.66,37.66]

    lx5 = [0,-0.2,-0.2,0]

    plt.plot(lx5,ly5,color=line,zorder=5)





           #6 yard boxes#

    ly6 = [24.84,24.84,43.16,43.16]

    lx6 = [104,99.5,99.5,104]

    plt.plot(lx6,ly6,color=line,zorder=5)



    ly7 = [24.84,24.84,43.16,43.16]

    lx7 = [0,4.5,4.5,0]

    plt.plot(lx7,ly7,color=line,zorder=5)



        #Halfway line, penalty spots, and kickoff spot

    ly8 = [0,68] 

    lx8 = [52,52]

    plt.plot(lx8,ly8,color=line,zorder=5)





    plt.scatter(93,34,color=line,zorder=5)

    plt.scatter(11,34,color=line,zorder=5)

    plt.scatter(52,34,color=line,zorder=5)



    circle1 = plt.Circle((93.5,34), 9.15,ls='solid',lw=1.5,color=line, fill=False, zorder=1,alpha=1)

    circle2 = plt.Circle((10.5,34), 9.15,ls='solid',lw=1.5,color=line, fill=False, zorder=1,alpha=1)

    circle3 = plt.Circle((52, 34), 9.15,ls='solid',lw=1.5,color=line, fill=False, zorder=2,alpha=1)



        ## Rectangles in boxes

    rec1 = plt.Rectangle((87.5,20), 16,30,ls='-',color=pitch, zorder=1,alpha=1)

    rec2 = plt.Rectangle((0, 20), 16.5,30,ls='-',color=pitch, zorder=1,alpha=1)



        ## Pitch rectangle

    rec3 = plt.Rectangle((-1, -1), 106,70,ls='-',color=pitch, zorder=1,alpha=1)



    ax.add_artist(rec3)

    ax.add_artist(circle1)

    ax.add_artist(circle2)

    ax.add_artist(rec1)

    ax.add_artist(rec2)

    ax.add_artist(circle3)

        

    
def sel_best(n,lst,p):

    ans = []

    for index,row in lst.iterrows():

        if (p=='p'):

            ans.append(row['Name'] + '(' + str(row['Potential']) + ')')

        else:

            ans.append(row['Name'] + '(' + str(row['Overall']) + ')')

            

        n = n-1

        if (n==0):    

            return ans
#Selecionando os melhores por potencial

best_gk_p = sel_best(2,list_gk_p,'p')

best_dff_p = sel_best(6,list_dff_p,'p')

best_mid_p = sel_best(5,list_mid_p,'p')

best_atk_p = sel_best(5,list_atk_p,'p')



#Selecionando os melhores por overall

best_gk_o = sel_best(2,list_gk_o,'o')

best_dff_o = sel_best(6,list_dff_o,'o')

best_mid_o = sel_best(5,list_mid_o,'o')

best_atk_o = sel_best(5,list_atk_o,'o')
def add_jogadores(lst,x,y):

    for i, type in enumerate(lst):

        x_c = x[i]

        y_c = y[i]

        plt.scatter(x_c, y_c, marker='o', color='red', edgecolors="black", zorder=10)

        plt.text(x_c, y_c+1, type, fontsize=16, fontweight='bold')
x_gk = [5,5]

y_gk = [26,42]



x_dff = [30,30,30,30,30,30]

y_dff = [4,16,28,40,52,64]



x_mid = [55,55,55,55,55]

y_mid = [4,16,34,52,64]



x_atk = [80,80,80,80,80]

y_atk = [4,16,34,52,64]
draw_pitch("#195905","#faf0e6","v","full")

plt.title('Seleção Olímpica brasileira por potencial no Fifa19', fontsize=40)

add_jogadores(best_gk_p,x_gk,y_gk)

add_jogadores(best_dff_p,x_dff,y_dff)

add_jogadores(best_mid_p,x_mid,y_mid)

add_jogadores(best_atk_p,x_atk,y_atk)
ind = np.arange(4)



potential_means_1 = (list_gk_p[0:2]['Potential'].mean(),

                  list_dff_p[0:6]['Potential'].mean(),

                  list_mid_p[0:5]['Potential'].mean(),

                  list_atk_p[0:5]['Potential'].mean())



overall_means_1 = (list_gk_p[0:2]['Overall'].mean(),

                  list_dff_p[0:6]['Overall'].mean(),

                  list_mid_p[0:5]['Overall'].mean(),

                  list_atk_p[0:5]['Overall'].mean())



potential_std_1 = (list_gk_p[0:2]['Potential'].std(),

                  list_dff_p[0:6]['Potential'].std(),

                  list_mid_p[0:5]['Potential'].std(),

                  list_atk_p[0:5]['Potential'].std())



overall_std_1 = (list_gk_p[0:2]['Overall'].std(),

                  list_dff_p[0:6]['Overall'].std(),

                  list_mid_p[0:5]['Overall'].std(),

                  list_atk_p[0:5]['Overall'].std())



plt.bar(ind,potential_means_1,yerr=potential_std_1,label='Potenciall')

plt.bar(ind,overall_means_1,yerr=overall_std_1,label='Overall')

plt.xticks(ind, ('GK','DFF','MID','ATK'))

plt.legend()

plt.show()
draw_pitch("#195905","#faf0e6","v","full")

plt.title('Seleção Olímpica brasileira por overall no Fifa19', fontsize=40)

add_jogadores(best_gk_o,x_gk,y_gk)

add_jogadores(best_dff_o,x_dff,y_dff)

add_jogadores(best_mid_o,x_mid,y_mid)

add_jogadores(best_atk_o,x_atk,y_atk)
ind = np.arange(4)



potential_means_2 = (list_gk_o[0:2]['Potential'].mean(),

                  list_dff_o[0:6]['Potential'].mean(),

                  list_mid_o[0:5]['Potential'].mean(),

                  list_atk_o[0:5]['Potential'].mean())



overall_means_2 = (list_gk_o[0:2]['Overall'].mean(),

                  list_dff_o[0:6]['Overall'].mean(),

                  list_mid_o[0:5]['Overall'].mean(),

                  list_atk_o[0:5]['Overall'].mean())



potential_std_2 = (list_gk_o[0:2]['Potential'].std(),

                  list_dff_o[0:6]['Potential'].std(),

                  list_mid_o[0:5]['Potential'].std(),

                  list_atk_o[0:5]['Potential'].std())



overall_std_2 = (list_gk_o[0:2]['Overall'].std(),

                  list_dff_o[0:6]['Overall'].std(),

                  list_mid_o[0:5]['Overall'].std(),

                  list_atk_o[0:5]['Overall'].std())



plt.bar(ind,potential_means_2,yerr=potential_std_2,label='Potenciall')

plt.bar(ind,overall_means_2,yerr=overall_std_2,label='Overall')

plt.xticks(ind, ('GK','DFF','MID','ATK'))

plt.legend()

plt.show()