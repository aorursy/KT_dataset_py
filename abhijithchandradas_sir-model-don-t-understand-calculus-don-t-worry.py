import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



import warnings

warnings.filterwarnings("ignore")
def sir_model(I0=0.01, beta=0.6, gamma=0.1):

    """

    Function will take in initial state for infected population,

    Transmission rate (beta) and recovery rate(gamma) as input.

    

    """

    N=1          #Total population

    I=I0         #Initial state of I default value 1% of population

    S=N-I        #Initial state of S

    R=0          #Initial State of R

    C=I          #Initial State of Total Cases

    beta=beta    #Transmission Rate

    gamma=gamma  #Recovery Rate



    inf=[]       # List of Infectious population for each day

    day=[]       # Time period in day

    suc=[]       # List of Susceptible population for each day

    rec=[]       # List of Recovered population for each day

    conf=[]      # List of Total Cases population for each day

    

    for i in range(60):

        day.append(i)

        inf.append(I)

        suc.append(S)

        rec.append(R)

        conf.append(C)



        new_inf= I*S*beta       #New infections equation (1)   

        new_rec= I*gamma        #New Recoveries equation (2)

        

        I=I+new_inf-new_rec     #Total infectious population for next day

        S=S-new_inf             #Total infectious population for next day

        R=R+new_rec             #Total recovered population for next day

        C=C+new_inf             #Total confirmed cases for next day

    

    max_inf=round(np.array(inf).max()*100,2)     #Peak infectious population in percentage

    max_conf=round(np.array(conf).max()*100,2)   #Overal infected population in percentage

    

    print(f"Maximum Infectious population at a time :{max_inf}%")

    print(f"Total Infected population :{max_conf}%")

    

    #Visualizing the model

    sns.set(style="darkgrid")

    plt.figure(figsize=(10,6))

    plt.title(f"SIR Model: R0 = {round(beta/gamma,2)}", fontsize=18)

    sns.lineplot(day,inf, label="Infected")

    sns.lineplot(day,suc,label="Succeptible")

    sns.lineplot(day,rec, label="Recovered")

    #sns.lineplot(day,conf, label="Confirmed") #Generally total infected population is not plotted 

    plt.legend()

    plt.xlabel("Time (in days)")

    plt.ylabel("Fraction of Population")

    plt.show()
def sir_model_betalist(I0=0.01, betalist=[0.5,0.8], gammalist=[0.15,0.25,0.5]):

    """

    Function takes Initial Infected Population(I0), list of transmission rates (betalist)

    and list of recovery rates(gammalist) as arguments.

    Plots Infectious population and Infected Population vs time for input parameters

    """

    

    for gamma in gammalist:

        # Plotting Infectious Population

        plt.figure(figsize=(10,6))

        sns.set(style="darkgrid")

        plt.title("SIR Model: Infectious Population", fontsize=18)

        

        for beta in betalist:

            N=1

            I=I0

            S=N-I

            gamma=gamma

            R0=beta/gamma

            

            inf=[]

            day=[]

            for i in range(50):

                day.append(i)

                inf.append(I)

                new_inf= I*S*beta

                new_rec= I*gamma

                I=I+new_inf-new_rec

                S=S-new_inf

            

            inf_max=round(np.array(inf).max()*100,1)

            sns.lineplot(day,inf, label=f"R0: {round(R0,2)} Peak: {inf_max}%")

            plt.legend()

        plt.show()

        

        # Plotting Total Infected Population

        plt.figure(figsize=(10,6))

        plt.title("SIR Model: Total Confirmed Cases", fontsize=18)       

        for beta in betalist:

            N=1

            I=I0

            S=N-I

            C=I

            gamma=gamma

            R0=beta/gamma

            day=[]

            conf=[]

            for i in range(50):

                day.append(i)

                conf.append(C)



                new_inf= I*S*beta

                new_rec= I*gamma

                I=I+new_inf-new_rec

                S=S-new_inf

                C=C+new_inf

            conf_max=round(np.array(conf).max()*100,1)

            sns.lineplot(day,conf, label=f"R0: {round(R0,2)} Total :{conf_max}%")

            plt.legend()

        plt.show()
sir_model(beta=0.75,gamma=0.15)
sir_model(beta=0.45, gamma=0.15)
sir_model(beta=0.45,gamma=0.3)
sir_model_betalist(I0=0.01,betalist=[0.15,0.2, 0.25,0.30,0.4,0.6,0.8,1,1.2], gammalist=[0.20])