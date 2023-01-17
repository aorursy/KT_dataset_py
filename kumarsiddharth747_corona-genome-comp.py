# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# -*- coding: utf-8 -*-

"""

Created on Thu Apr 30 17:05:19 2020



@author: Siddharth

"""

##############################################          subtask1     ##############################

from textwrap import wrap

cols=['sequence']

df1  = pd.read_csv('/kaggle/input/covid19-genomes/LC528232.txt',names=cols)

df1['new'] = df1['sequence'].str.cat(sep='')

l1=df1["new"][0]

p1=wrap(l1, 1)

#-----------------------------virus2---------------------------

df2  = pd.read_csv('/kaggle/input/covid19-genomes/LC528233.txt',names=cols)

df2['new'] = df2['sequence'].str.cat(sep='')

l2=df2["new"][0]

p2=wrap(l2, 1)

##############################end of virus 2###################



#-----------------------------virus3---------------------------

df3  = pd.read_csv('/kaggle/input/covid19-genomes/LC529905.txt',names=cols)

df3['new'] = df3['sequence'].str.cat(sep='')

l3=df3["new"][0]

p3=wrap(l3, 1)

##############################end of virus 3###################



#-----------------------------virus4---------------------------

df4  = pd.read_csv('/kaggle/input/covid19-genomes/LC534418.txt',names=cols)

df4['new'] = df4['sequence'].str.cat(sep='')

l4=df4["new"][0]

p4=wrap(l4, 1)

#print('p4',len(p4))

##############################end of virus 4###################



#-----------------------------virus5---------------------------

df5  = pd.read_csv('/kaggle/input/covid19-genomes/LC534419.txt',names=cols)

df5['new'] = df5['sequence'].str.cat(sep='')

l5=df5["new"][0]

p5=wrap(l5, 1)

#print('p5',len(p5))

##############################end of virus 5###################







#-----------------------------virus6---------------------------

df6  = pd.read_csv('/kaggle/input/covid19-genomes/LR757995.txt',names=cols)

df6['new'] = df6['sequence'].str.cat(sep='')

l6=df6["new"][0]

p6=wrap(l6, 1)

#print('p6',len(p6))

##############################end of virus 6###################



#-----------------------------virus7---------------------------

df7  = pd.read_csv('/kaggle/input/covid19-genomes/LR757996.txt',names=cols)

df7['new'] = df7['sequence'].str.cat(sep='')

l7=df7["new"][0]

p7=wrap(l7, 1)

#print('p7',len(p7))

##############################end of virus 7###################



#-----------------------------virus8---------------------------

df8  = pd.read_csv('/kaggle/input/covid19-genomes/LR757998.txt',names=cols)

df8['new'] = df8['sequence'].str.cat(sep='')

l8=df8["new"][0]

p8=wrap(l8, 1)

#print('p8',len(p8))

##############################end of virus 8###################







#-----------------------------virus9---------------------------

df9  = pd.read_csv('/kaggle/input/covid19-genomes/MN908947.txt',names=cols)

df9['new'] = df9['sequence'].str.cat(sep='')

l9=df9["new"][0]

p9=wrap(l9, 1)

#print('p9',len(p9))

##############################end of virus9###################



#-----------------------------virus10---------------------------

df10  = pd.read_csv('/kaggle/input/covid19-genomes/MN938384.txt',names=cols)

df10['new'] = df10['sequence'].str.cat(sep='')

l10=df10["new"][0]

p10=wrap(l10, 1)

#print('p10',len(p10))

##############################end of virus 10###################



#-----------------------------virus11---------------------------

df11  = pd.read_csv('/kaggle/input/covid19-genomes/MN975262.txt',names=cols)

df11['new'] = df11['sequence'].str.cat(sep='')

l11=df11["new"][0]

p11=wrap(l11, 1)

#print('p11',len(p11))

##############################end of virus 13###################



#-----------------------------virus12---------------------------

df12  = pd.read_csv('/kaggle/input/covid19-genomes/MN985325.txt',names=cols)

df12['new'] = df12['sequence'].str.cat(sep='')

l12=df12["new"][0]

p12=wrap(l12, 1)

#print('p12',len(p12))

##############################end of virus 12###################



#-----------------------------virus13---------------------------

df13  = pd.read_csv('/kaggle/input/covid19-genomes/MN988668.txt',names=cols)

df13['new'] = df13['sequence'].str.cat(sep='')

l13=df13["new"][0]

p13=wrap(l13, 1)

#print('p13',len(p13))

##############################end of virus 13###################



#-----------------------------virus14---------------------------

df14  = pd.read_csv('/kaggle/input/covid19-genomes/MN988669.txt',names=cols)

df14['new'] = df14['sequence'].str.cat(sep='')

l14=df14["new"][0]

p14=wrap(l14, 1)

#print('p14',len(p14))

##############################end of virus 16###################



#-----------------------------virus15---------------------------

df15  = pd.read_csv('/kaggle/input/covid19-genomes/MN988713.txt',names=cols)

df15['new'] = df15['sequence'].str.cat(sep='')

l15=df15["new"][0]

p15=wrap(l15, 1)

#print('p15',len(p15))

##############################end of virus 15###################



#-----------------------------virus16---------------------------

df16  = pd.read_csv('/kaggle/input/covid19-genomes/MN994467.txt',names=cols)

df16['new'] = df16['sequence'].str.cat(sep='')

l16=df16["new"][0]

p16=wrap(l16, 1)

#print('p16',len(p16))

##############################end of virus 16###################



#-----------------------------virus17---------------------------

df17  = pd.read_csv('/kaggle/input/covid19-genomes/MN994468.txt',names=cols)

df17['new'] = df17['sequence'].str.cat(sep='')

l17=df17["new"][0]

p17=wrap(l17, 1)

#print('p17',len(p17))

##############################end of virus 17###################



#-----------------------------virus18---------------------------

df18  = pd.read_csv('/kaggle/input/covid19-genomes/MN996527.txt',names=cols)

df18['new'] = df18['sequence'].str.cat(sep='')

l18=df18["new"][0]

p18=wrap(l18, 1)

#print('p18',len(p18))

##############################end of virus 18###################



#-----------------------------virus19---------------------------

df19  = pd.read_csv('/kaggle/input/covid19-genomes/MN996529.txt',names=cols)

df19['new'] = df19['sequence'].str.cat(sep='')

l19=df19["new"][0]

p19=wrap(l19, 1)

#print('p19',len(p19))

##############################end of virus 21###################



#-----------------------------virus20---------------------------

df20  = pd.read_csv('/kaggle/input/covid19-genomes/MN996530.txt',names=cols)

df20['new'] = df20['sequence'].str.cat(sep='')

l20=df20["new"][0]

p20=wrap(l20,1)

#print('p20',len(p20))

##############################end of virus 20###################











#-----------------------------data frame-----------------------

df11 = pd.DataFrame({'seq_virus1':pd.Series(p1),'seq_virus2':pd.Series(p2),

                      'seq_virus3':pd.Series(p3),'seq_virus4':pd.Series(p4),

                      'seq_virus5':pd.Series(p5),'seq_virus6':pd.Series(p6),

                      'seq_virus7':pd.Series(p7),'seq_virus8':pd.Series(p8),

                      'seq_virus9':pd.Series(p9),'seq_virus10':pd.Series(p10),

                      'seq_virus11':pd.Series(p11),'seq_virus12':pd.Series(p12),

                     'seq_virus13':pd.Series(p13),'seq_virus14':pd.Series(p14),

                     'seq_virus15':pd.Series(p15),'seq_virus16':pd.Series(p16),

                     'seq_virus17':pd.Series(p17),'seq_virus18':pd.Series(p18),

                     'seq_virus19':pd.Series(p19),'seq_virus20':pd.Series(p20)})



print(df11)
############################################################   subtask2    #####################################



d1 =df11['seq_virus1'].unique().tolist()

d2 =df11['seq_virus2'].unique().tolist()

d3 =df11['seq_virus3'].unique().tolist()

d4 =df11['seq_virus4'].unique().tolist()

d5 =df11['seq_virus5'].unique().tolist()

d6 =df11['seq_virus6'].unique().tolist()

d7 =df11['seq_virus7'].unique().tolist()

d8 =df11['seq_virus8'].unique().tolist()

d9 =df11['seq_virus9'].unique().tolist()

d10=df11['seq_virus10'].unique().tolist()

d11=df11['seq_virus11'].unique().tolist()

d12=df11['seq_virus12'].unique().tolist()

d13=df11['seq_virus13'].unique().tolist()

d14=df11['seq_virus14'].unique().tolist()

d15=df11['seq_virus15'].unique().tolist()

d16=df11['seq_virus16'].unique().tolist()

d17=df11['seq_virus17'].unique().tolist()

d18=df11['seq_virus18'].unique().tolist()

d19=df11['seq_virus19'].unique().tolist()

d20=df11['seq_virus20'].unique().tolist()



a=d1+d2+d3+d4+d5+d6+d7+d8+d9+d10+d11+d12+d13+d14+d15+d16+d17+d18+d19+d20

def unique(list1): 

  

    # intilize a null list 

    unique_list = [] 

    u=[]  

    # traverse for all elements 

    for x in list1: 

        # check if exists in unique_list or not 

        if x not in unique_list: 

            unique_list.append(x) 

    # print list 

    for x in unique_list: 

        u.append(x)

        #print (x)

        #print(u)

unique(a)



x=['T','G','C','A','W','Y','S','7']

x=sorted(x)

s=list(range(2,10))

df11=df11.fillna(1)

df11=df11.replace(x,s)



print(df11)
############################################################# subtask3 #############################################

import matplotlib.pyplot as plt

a=df11['seq_virus1']-df11['seq_virus2']

b=df11['seq_virus1']-df11['seq_virus3']

c=df11['seq_virus1']-df11['seq_virus4']

d=df11['seq_virus1']-df11['seq_virus5']

e=df11['seq_virus1']-df11['seq_virus6']

f=df11['seq_virus1']-df11['seq_virus7']

g=df11['seq_virus1']-df11['seq_virus8']

h=df11['seq_virus1']-df11['seq_virus9']

i=df11['seq_virus1']-df11['seq_virus10']

j=df11['seq_virus1']-df11['seq_virus11']

k=df11['seq_virus1']-df11['seq_virus12']

l=df11['seq_virus1']-df11['seq_virus13']

m=df11['seq_virus1']-df11['seq_virus14']

n=df11['seq_virus1']-df11['seq_virus15']

o=df11['seq_virus1']-df11['seq_virus16']

p=df11['seq_virus1']-df11['seq_virus17']

q=df11['seq_virus1']-df11['seq_virus18']

r=df11['seq_virus1']-df11['seq_virus19']

s=df11['seq_virus1']-df11['seq_virus20']



frame = {'values_a': a,'values_b': b,'values_c': c,'values_d': d,'values_e': e,'values_f': f,

         'values_g': g,'values_h': h,'values_i': i,'values_j': j,'values_k': k,'values_l': l,

         'values_m': m,'values_n': n,'values_o': o,'values_p': p,'values_q': q,

         'values_r': r,'values_s': s} 

df3 = pd.DataFrame(frame) 

df3['final']=(df3['values_a']+df3['values_b']+df3['values_c']+df3['values_d']+df3['values_e']+df3['values_f']+

             df3['values_g']+df3['values_h']+df3['values_i']+df3['values_j']+df3['values_k']+df3['values_l']+

             df3['values_m']+df3['values_n']+df3['values_o']+df3['values_p']+df3['values_q']+df3['values_r']+

             df3['values_s'])





df3['final1']=df3['final']

df3["final"]= df3["final"].replace(0, 19000)

df3['final12']=df3['final']-df3['final1']

df3["final12"]= df3["final12"].replace(19000, 10)

plt.plot(df3['final12'])