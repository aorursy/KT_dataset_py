import matplotlib.pyplot as plt

%matplotlib inline

import pandas as pd

import numpy as np
data = {'Diameter':[15.2,15.2,15.2,15.4,15.4,15.4,14.9,14.9,14.9],

        'Heigh':[11.5,11.7,11.6,11.5,11.7,11.6,11.5,11.7,11.6],

        'Chamfer':['1*50','1.5*30','1*30','1.5*30','1*30','1*50','1*30','1*50','1.5*30'],

        'Oil_pressure':[15,17,20,20,15,17,17,20,15]    

}

df = pd.DataFrame(data)

df
df['Breakaway'] = [856,950,908,877,972,898,802,1029,926]



df
sub_df = pd.DataFrame(df)

sub_df['Diameter'] = sub_df.Diameter.map(lambda x:1 if x==15.2 else x) 

sub_df['Diameter'] = sub_df.Diameter.map(lambda x:2 if x==15.4 else x)

sub_df['Diameter'] = sub_df.Diameter.map(lambda x:3 if x==14.9 else x)



sub_df['Heigh'] = sub_df.Heigh.map(lambda x:1 if x==11.5 else x)

sub_df['Heigh'] = sub_df.Heigh.map(lambda x:2 if x==11.7 else x)

sub_df['Heigh'] = sub_df.Heigh.map(lambda x:3 if x==11.6 else x)



sub_df['Chamfer'] = sub_df.Chamfer.map(lambda x:1 if x=='1*50' else x)

sub_df['Chamfer'] = sub_df.Chamfer.map(lambda x:2 if x=='1.5*30' else x)

sub_df['Chamfer'] = sub_df.Chamfer.map(lambda x:3 if x=='1*30' else x)



sub_df['Oil_pressure'] = sub_df.Oil_pressure.map(lambda x:1 if x==15 else x)

sub_df['Oil_pressure'] = sub_df.Oil_pressure.map(lambda x:2 if x==17 else x)

sub_df['Oil_pressure'] = sub_df.Oil_pressure.map(lambda x:3 if x==20 else x)



sub_df = sub_df.astype(int)

sub_df
for i in sub_df['Diameter']:

    if (i == 1):

        D_1 = sub_df[sub_df['Diameter']==i]['Breakaway'].sum()

    if (i == 2):

        D_2 = sub_df[sub_df['Diameter']==i]['Breakaway'].sum()

    if (i == 3):

        D_3 = sub_df[sub_df['Diameter']==i]['Breakaway'].sum()

        

for i in sub_df['Heigh']:

    if (i == 1):

        H_1 = sub_df[sub_df['Heigh']==i]['Breakaway'].sum()

    if (i == 2):

        H_2 = sub_df[sub_df['Heigh']==i]['Breakaway'].sum()

    if (i == 3):

        H_3 = sub_df[sub_df['Heigh']==i]['Breakaway'].sum()

        

for i in sub_df['Chamfer']:

    if (i == 1):

        C_1 = sub_df[sub_df['Chamfer']==i]['Breakaway'].sum()

    if (i == 2):

        C_2 = sub_df[sub_df['Chamfer']==i]['Breakaway'].sum()

    if (i == 3):

        C_3 = sub_df[sub_df['Chamfer']==i]['Breakaway'].sum()        



for i in sub_df['Oil_pressure']:

    if (i == 1):

        O_1 = sub_df[sub_df['Oil_pressure']==i]['Breakaway'].sum()

    if (i == 2):

        O_2 = sub_df[sub_df['Oil_pressure']==i]['Breakaway'].sum()

    if (i == 3):

        O_3 = sub_df[sub_df['Oil_pressure']==i]['Breakaway'].sum()   
total_Breakaway = {'A':[D_1,D_2,D_3],

                  'B':[H_1,H_2,H_3],

                  'C':[C_1,C_2,C_3],

                  'D':[O_1,O_2,O_3]

}

tot_df = pd.DataFrame(total_Breakaway)

tot_df
max_A = tot_df['A'].argmax()

max_B = tot_df['B'].argmax()

max_C = tot_df['C'].argmax()

max_D = tot_df['D'].argmax()

print('A max:',max_A,'   B max:',max_B,'   C max:',max_C,'   D max:',max_D)
R_A = tot_df['A'].max() - tot_df['A'].min()

R_B = tot_df['B'].max() - tot_df['B'].min()

R_C = tot_df['C'].max() - tot_df['C'].min()

R_D = tot_df['D'].max() - tot_df['D'].min()

print('R_A:',R_A ,'  R_B:',R_B,'   R_C:',R_C,'   R_D:',R_D)
print('the best order: B{} D{} C{} A{}'.format(max_B,max_D,max_C,max_A))
y_order = {'A':[2757,2714,2740],

           'B':[2535,2732,2951],

           'C':[2783,2753,2682],

           'D':[2754,2650,2814]

}

y_order_df = pd.DataFrame(y_order)

y_order_df
x_order = {'A':[14.9,15.2,15.4],

           'B':[11.5,11.6,11.7],

           'C':['1*50','1.5*30','1*30'],

           'D':[15,17,20]

}

x_order_df = pd.DataFrame(x_order)

x_order_df
f,ax = plt.subplots(1,4,figsize=(20,5))

f.suptitle('Analyze by the table')

ax[0].plot(x_order_df['A'],y_order_df['A'])

ax[1].plot(x_order_df['B'],y_order_df['B'])

ax[2].plot(x_order_df['C'],y_order_df['C'])

ax[3].plot(x_order_df['D'],y_order_df['D'])