import pandas as pd

import matplotlib.pyplot as plt

import numpy as np
#first and second derivatives

class derivatives():

    def dS(var):

        return -var['beta']*var['S'][-1]*var['I'][-1]/var['N']

    def dI(var):

        return var['beta']*var['S'][-1]*var['I'][-1]/var['N']-var['gamma_h']*var['I'][-1]-var['gamma_r']*var['I'][-1]

    def dH(var):

        return var['gamma_h']*var['I'][-1]-var['delta_d']*var['H'][-1]-var['delta_r']*var['H'][-1]

    def dR(var):

        return var['gamma_r']*var['I'][-1]+var['delta_r']*var['H'][-1]

    def dD(var):

        return var['delta_d']*var['H'][-1]

    def d2S(var):

        return -var['beta']*var['S'][-1]/var['N']*derivatives.dI(var)-var['beta']*var['I'][-1]/var['N']*derivatives.dS(var)

    def d2I(var):

        return -derivatives.d2S(var)-var['gamma_h']*derivatives.dI(var)-var['gamma_r']*derivatives.dI(var)

    def d2H(var):

        return var['gamma_h']*derivatives.dI(var)-var['delta_d']*derivatives.dH(var)-var['delta_r']*derivatives.dH(var)

    def d2R(var):

        return var['gamma_r']*derivatives.dI(var) + var['delta_r']*derivatives.dH(var)

    def d2D(var):

        return var['delta_d']*derivatives.dH(var)
#Truncated Taylor series used to integrate and determine the values which is not the most efficient method but 

# provided the time step is sufficiently small, which it is, the outcomes are accurate

def simulate(N=3.2e6,Io=29,steps=4e4,tstep=0.01,transitions={'Day0,RO4':[0,4],'Day10,RO3':[10,3]},

             beta=0.33,gamma_h=0.012,gamma_r=0.15,delta_r=0.02,delta_d=0.015,):

    var = {'time':[0],'S':[N],'I':[Io],'R':[0],'H':[0],'D':[0],

           'beta':beta,'gamma_h':gamma_h,'gamma_r':gamma_r,'delta_r':delta_r,'delta_d':delta_d,'N':N}

    nohospitalbeds=5000

    for i in range(steps):

        for each in var:

            if type(var[each])==type([]):

                for tran in transitions:

                    if i*tstep>transitions[tran][0]:

                        var['beta'] = transitions[tran][1]*(var['gamma_h']+var['gamma_r'])

                    if var['H'][-1]>nohospitalbeds:

                        var['delta_d']=delta_d*3

                    else:

                        var['delta_d']=delta_d

                if each!='time':

                    fderiv = getattr(derivatives,'d'+str(each))

                    sderiv = getattr(derivatives,'d2'+str(each))

                    var[each].append(var[each][-1]+fderiv(var)*tstep+0.5*sderiv(var)*tstep**2)

                else:

                    var[each].append(var[each][-1]+tstep)

    dataframe = pd.DataFrame(dict((k, var[k]) for k in ('time','S', 'I', 'R', 'H', 'D')))

    return dataframe.set_index('time')
def covidplt(df,withdata=True):

    fig = plt.figure(figsize=(6, 4))

    ax = fig.add_subplot(111)

    plt.grid()

    if withdata:

        ax.scatter(np.arange(27),[29,51,63,78,112,136,181,257,298,346,402,480,602,719,806,887,1012,

                                  1246,1428,1675,1738,1846,2100,2200,2303,2363,2412],color='maroon',label='total recorded infected')

        ax.scatter([26,27],[201,213],label='total hospitalized') #day 27 is on Apr. 14, 2020

        ax.scatter([26,27],[18,19],label='total deceased')

    ax.semilogy(df.index,df['S'],label='susceptible');ax.plot(df.index,df['I'],label='infected')

    ax.plot(df.index,df['R'],label='recovered');ax.plot(df.index,df['H'],label='hospitalized')

    ax.plot(df.index,df['D'],label='deceased')

    plt.xlabel('days');plt.ylabel('No of People in Utah')

    plt.ylim(1,3.5e6);plt.legend(loc='upper right')

    plt.show()
df1=simulate(N=3.2e6,Io=29,steps=40000,tstep=0.005,transitions={'Day0,RO4':[0,3]},

            beta=0.33,gamma_h=0.012,gamma_r=0.2,delta_r=0.02,delta_d=0.012)

covidplt(df1,withdata=False)
df1.iloc[df1.index.get_loc(200,method='nearest')]
df2=simulate(N=3.2e6,Io=29,steps=80000,tstep=0.005,transitions={'Day0,RO4':[0,3],'Day5,RO2':[5,1.5],'Day26,RO1':[26,1]},

            beta=0.33,gamma_h=0.012,gamma_r=0.2,delta_r=0.02,delta_d=0.012)

covidplt(df2)
df2.iloc[df2.index.get_loc(400,method='nearest')]
df1.iloc[df1.index.get_loc(27,method='nearest')]
df3=simulate(N=3.2e6,Io=29,steps=80000,tstep=0.005,transitions={'Day0,RO4':[0,3],'Day5,RO2':[5,1.5],'Day26,RO1':[26,1],

                                                               'Day43,RO2':[43,1.5]},

            beta=0.33,gamma_h=0.012,gamma_r=0.2,delta_r=0.02,delta_d=0.012)

covidplt(df3)
df3.iloc[df3.index.get_loc(300,method='nearest')]
df4=simulate(N=3.2e6,Io=29,steps=80000,tstep=0.005,transitions={'Day0,RO4':[0,3],'Day5,RO2':[5,1.5],'Day26,RO1':[26,1],

                                                               'Day27,RO2':[43,1.5],'D':[75,1],'D2':[105,1.5]},

            beta=0.33,gamma_h=0.012,gamma_r=0.2,delta_r=0.02,delta_d=0.012)

covidplt(df4)
df4.iloc[df4.index.get_loc(400,method='nearest')]
df5=simulate(N=3.2e6,Io=29,steps=80000,tstep=0.005,transitions={'Day0,RO4':[0,3],'Day5,RO2':[5,1.5],'Day26,RO1':[26,1],

                                                               'Day43,RO2':[73,1.5]},

            beta=0.33,gamma_h=0.012,gamma_r=0.2,delta_r=0.02,delta_d=0.012)

covidplt(df5)
df5.iloc[df5.index.get_loc(300,method='nearest')]