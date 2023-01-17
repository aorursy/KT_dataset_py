import numpy as np 

import pandas as pd 

import math

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.metrics import mean_squared_error

import plotly.express as px
data = pd.read_csv("../input/HardRockData.csv")

data
#fig = px.parallel_categories(data, color="ROP [ft/hr]", color_continuous_scale=px.colors.sequential.Inferno)

#fig.show()
def Plot_data_seaborn():

    sns.set()

    columns = ['Dc','Blade','Nc','UCS (psi)','WOB (lbf)','ROP [ft/hr]','IFAave','RPM','Db (inch)','BR','SR']

    sns.pairplot(data[columns], size = 1.5 , kind ='scatter')

    plt.show()

    

    

def Turn_data_to_seperate_lists():

    

    Dc = data['Dc']*0.0393701

    NOB = data['Blade']

    NOC = data['Nc']

    UCS = data['UCS (psi)']

    WOB = data['WOB (lbf)']

    ROP_Data = data['ROP [ft/hr]']

    IFA = data['IFAave']

    RPM = data['RPM']

    Db = data['Db (inch)']

    BR = data['BR']

    SR = data['SR']

    #IFA = 0

    #G = (NOB*Dc*np.cos(np.radians(SR))/(NOC*Db**2*np.tan(np.radians(BR+IFA))))

    #G = (NOB*Dc*np.cos(np.radians(SR)))/(NOC*Db*np.tan(np.radians(BR)))

    

    return WOB,ROP_Data,Db,RPM,UCS,NOC,BR,SR,Dc,NOB,IFA#,G





def AA_Model(WOB,Db,RPM,UCS,NOC,BR,SR,Dc,NOB,w):

    

    ARS = UCS*(1+w[0]*(RPM/100)**w[1])

    G = ((NOB**w[2])*Dc*np.cos(np.radians(SR)))/((NOC**w[3])*Db*(np.tan(np.radians(BR))**w[4]))

    G_k1 = w[5]*G

    A = G_k1*WOB*RPM/ARS

    B = (G_k1**w[6])*(WOB**w[7])*(RPM**w[8])/(ARS**w[9])

    ROP = A - B



    return ROP





'''

def AA_Model(WOB,RPM,UCS,G,NOC,BR,w):



    G_k1 = w[0]*G

    WOBt = w[1] * ((NOC) / math.cos(math.radians(BR)))**w[2]

    ROPt = w[3] * ((NOC) / math.cos(math.radians(BR)))**w[4]

    ROP = G_k1*(WOB-WOBt)*RPM/UCS + ROPt



    return ROP

'''





def Objective_Function(w):

    

    Error = 0

    ROP_pred_list = []

    ROP_pred_list = [AA_Model(WOB,Db,RPM,UCS,NOC,BR,SR,Dc,NOB,w) for WOB,Db,RPM,UCS,NOC,BR,SR,Dc,NOB in zip(WOB,Db,RPM,UCS,NOC,BR,SR,Dc,NOB)]

    rmse = np.sqrt(mean_squared_error(ROP_pred_list, ROP_Data))

    

    return rmse





'''

def Objective_Function(w):

    

    Error = 0

    ROP_pred_list = []

    ROP_pred_list = [AA_Model(WOB,RPM,UCS,G,NOC,BR,w) for WOB,RPM,UCS,G,NOC,BR in zip(WOB,RPM,UCS,G,NOC,BR)]

    

    rmse = np.sqrt(mean_squared_error(ROP_pred_list, ROP_Data))

    

    return rmse

'''









def De_Algorithm(fobj, bounds, mut=0.8, crossp=0.7, popsize=100, its=1000):

    dimensions = len(bounds)

    pop = np.random.rand(popsize, dimensions)

    min_b, max_b = np.asarray(bounds).T

    diff = np.fabs(min_b - max_b)

    pop_denorm = min_b + pop * diff

    fitness = np.asarray([fobj(ind) for ind in pop_denorm])

    best_idx = np.argmin(fitness)

    best = pop_denorm[best_idx]

    for i in range(its):

        for j in range(popsize):

            idxs = [idx for idx in range(popsize) if idx != j]

            a, b, c = pop[np.random.choice(idxs, 3, replace = False)]

            mutant = np.clip(a + mut * (b - c), 0, 1)

            cross_points = np.random.rand(dimensions) < crossp

            if not np.any(cross_points):

                cross_points[np.random.randint(0, dimensions)] = True

            trial = np.where(cross_points, mutant, pop[j])

            trial_denorm = min_b + trial * diff

            f = fobj(trial_denorm)

            if f < fitness[j]:

                fitness[j] = f

                pop[j] = trial

                if f < fitness[best_idx]:

                    best_idx = j

                    best = trial_denorm

                    

                    

        #print("Iteration number= %s" % (i))

        #print("Best Fitness= %s" % (fitness[best_idx]))

        #print("Best values= %s" % (best))

        yield best, fitness[best_idx]



def Run_DEA(ite):

    results = []

    result = list(De_Algorithm(Objective_Function, 

                 [(0.3, 0.5),#ARS

                  (0.4, 0.7),#ARS

                  (0.1, 0.5),#NOB

                  (0.1, 0.5),#NOC

                  (0.1, 0.5),#BR

                  (0, 100),#G_k1 cons

                  (0, 1),#G_k1 pow

                  (0, 0.5),#WOB pow

                  (0.9, 1.1),#RPM pow

                  (0.5, 1)],#ARS pow   

                  mut=0.8, crossp=0.8, popsize=15, its=ite))

    

    df = pd.DataFrame(result)

    return results, df



'''

def Run_DEA(ite):

    results = []

    result = list(De_Algorithm(Objective_Function, 

                 [(0, 1000), # w[0]*G   

                  (0,3000),  #   

                  (0.1, 3), # w[0]*G   

                  (0.1,3000),  #   

                  (0.1, 3)],   # 

                  mut=0.7, crossp=0.8, popsize=30, its=ite))

    

    df = pd.DataFrame(result)

    return results, df

'''





def Best_coffs(df):

    

    #df['w1'], df['w2'], df['w3'], df['w4'], df['w5'], df['w6'], df['w7']  = zip(*df[0]) # Unzip

    #df['w1'], df['w2'], df['w3'], df['w4'], df['w5']  = zip(*df[0]) # Unzip

    df['w1'], df['w2'], df['w3'], df['w4'], df['w5'], df['w6'], df['w7'], df['w8'], df['w9'], df['w10']  = zip(*df[0]) # Unzip

    cols = [0] # Drop the first column

    df.drop(df.columns[cols],axis = 1,inplace = True) # Drop the first column

    df.columns.values[0] = "Fitness" # name the first column as Fitness

    best_coff = df.iloc[len(df)-1,1:] # insert the best coefficients into the best_coff

    

    return best_coff



def Plot_DEA_Evolution(df):

    

    data_ncol=len(df.columns) # number of paramters 

    fig = plt.figure(figsize=(15,15)) # you may change these to change the distance between plots.



    for i in range(1,(data_ncol+1)):

        if i<data_ncol:

            plt.subplot(3, 4, i)

            plt.plot(df['w{}'.format(i)],'bo', markersize=4)

            plt.xlabel('Iteration')

            plt.ylabel('w{}'.format(i))

            plt.grid(True)

        else:       

            plt.subplot(3, 4, data_ncol)

            plt.plot(df['Fitness'],'red', markersize=4)

            plt.xlabel('Iteration')

            plt.ylabel('Fitness')

            plt.grid(True)

    plt.show()



def Plot_variables(x, y, xlab, ylab, xmin, xmax, ymin, ymax):

    

    fig = plt.figure(figsize=(7,7))

    plt.scatter(x, y)

    plt.xlim(xmin,xmax)

    plt.ylim(ymin,ymax)

    plt.xlabel(xlab)

    plt.ylabel(ylab)
# WOB,RPM,UCS,G,w

#w = [0.1,1,0.1,1,1,0.1,1.2,0.11]

#AA_Model(10000,100,28000,500, w)
fig = px.scatter_3d(data, x='RPM', y='WOB', z='ROP [ft/hr]',

                    width=1000, height=700, color="ROP [ft/hr]")

fig.show()
#Plot_data_seaborn()
WOB,ROP_Data,Db,RPM,UCS,NOC,BR,SR,Dc,NOB,IFA = Turn_data_to_seperate_lists() #,G
results, df = Run_DEA(2000)

best_coff = Best_coffs(df)
print(best_coff)
#ARS

#ARS

#NOB

#NOC

#BR

#G_k1 cons

#G_k1 pow

#WOB pow

#RPM pow

#ARS pow 
Plot_DEA_Evolution(df)
#Est_ROP_Model = [AA_Model(WOB,RPM,UCS,G,best_coff) for WOB,RPM,UCS,G in zip(WOB,RPM,UCS,G)]

Est_ROP_Model = [AA_Model(WOB,Db,RPM,UCS,NOC,BR,SR,Dc,NOB,best_coff) for WOB,Db,RPM,UCS,NOC,BR,SR,Dc,NOB in zip(WOB,Db,RPM,UCS,NOC,BR,SR,Dc,NOB)]

Est_ROP_Model = pd.DataFrame(Est_ROP_Model)

ROP_Data = pd.DataFrame(ROP_Data)
Plot_variables(ROP_Data, Est_ROP_Model, 'ROP Data ft/hr', 'ROP Model ft/hr', 0, 100, 0, 100)
data['Model'] = Est_ROP_Model

fig = px.scatter(data, x="ROP [ft/hr]", y="Model", trendline = "ols", width=600, height=600, color="WOB")

fig.show()

results = px.get_trendline_results(fig)

results.px_fit_results.iloc[0].summary()
fig = plt.figure(figsize=(5,5))

plt.scatter(WOB[0:9], ROP_Data[0:9], label='Data')

plt.plot(WOB[0:9], Est_ROP_Model[0:9],'r', label='Model')

plt.title('SNL 2015,4B, SWG, 100rpm')

plt.xlabel("WOB lbf")

plt.ylabel("ROP ft/hr")

plt.xlim(0,5000)

plt.ylim(0,50)

plt.legend()
fig = plt.figure(figsize=(5,5))

plt.scatter(WOB[9:16], ROP_Data[9:16], label='Data')

plt.plot(WOB[9:16], Est_ROP_Model[9:16],'r', label='Model')

plt.title('NOV, 4B, SWG, 80rpm')

plt.xlabel("WOB lbf")

plt.ylabel("ROP ft/hr")

plt.xlim(0,5000)

plt.ylim(0,60)

plt.legend()
fig = plt.figure(figsize=(5,5))

plt.scatter(WOB[16:21], ROP_Data[16:21], label='Data')

plt.plot(WOB[16:21], Est_ROP_Model[16:21],'r', label='Model')

plt.title('SNL 2015,4B, SWG, 150rpm')

plt.xlabel("WOB lbf")

plt.ylabel("ROP ft/hr")

plt.xlim(0,5000)

plt.ylim(0,60)

plt.legend()
fig = plt.figure(figsize=(5,5))

plt.scatter(WOB[21:26], ROP_Data[21:26], label='Data')

plt.plot(WOB[21:26], Est_ROP_Model[21:26],'r', label='Model')

plt.title('SNL 2015,4B, SWG, 150rpm')

plt.xlabel("WOB lbf")

plt.ylabel("ROP ft/hr")

plt.xlim(0,5000)

plt.ylim(0,60)

plt.legend()
fig = plt.figure(figsize=(5,5))

plt.scatter(WOB[26:31], ROP_Data[26:31], label='Data')

plt.plot(WOB[26:31], Est_ROP_Model[26:31],'r', label='Model')

plt.title('SNL 2015,4B, SWG, 150rpm')

plt.xlabel("WOB lbf")

plt.ylabel("ROP ft/hr")

plt.xlim(0,5000)

plt.ylim(0,60)

plt.legend()
fig = plt.figure(figsize=(5,5))

plt.scatter(WOB[31:39], ROP_Data[31:39], label='Data')

plt.plot(WOB[31:39], Est_ROP_Model[31:39],'r', label='Model')

plt.title('SNL 2015,5B, SWG, 150rpm')

plt.xlabel("WOB lbf")

plt.ylabel("ROP ft/hr")

plt.xlim(0,5000)

plt.ylim(0,60)

plt.legend()
fig = plt.figure(figsize=(5,5))

plt.scatter(WOB[39:46], ROP_Data[39:46], label='Data')

plt.plot(WOB[39:46], Est_ROP_Model[39:46],'r', label='Model')

plt.title('SNL 2015,5B, SWG, 150rpm')

plt.xlabel("WOB lbf")

plt.ylabel("ROP ft/hr")

plt.xlim(0,5000)

plt.ylim(0,60)

plt.legend()
fig = plt.figure(figsize=(5,5))

plt.scatter(WOB[46:51], ROP_Data[46:51], label='Data')

plt.plot(WOB[46:51], Est_ROP_Model[46:51],'r', label='Model')

plt.title('SNL 2015,4B, SWG, 150rpm')

plt.xlabel("WOB lbf")

plt.ylabel("ROP ft/hr")

plt.xlim(0,5000)

plt.ylim(0,60)

plt.legend()
fig = plt.figure(figsize=(5,5))

plt.scatter(WOB[51:58], ROP_Data[51:58], label='Data')

plt.plot(WOB[51:58], Est_ROP_Model[51:58],'r', label='Model')

plt.title('SNL 2015,4B, SWG, 150rpm')

plt.xlabel("WOB lbf")

plt.ylabel("ROP ft/hr")

plt.xlim(0,8000)

plt.ylim(0,60)

plt.legend()
fig = plt.figure(figsize=(5,5))

plt.scatter(WOB[58:63], ROP_Data[58:63], label='Data')

plt.plot(WOB[58:63], Est_ROP_Model[58:63],'r', label='Model')

plt.title('SNL 2015,4B, SWG, 150rpm')

plt.xlabel("WOB lbf")

plt.ylabel("ROP ft/hr")

plt.xlim(0,8000)

plt.ylim(0,60)

plt.legend()