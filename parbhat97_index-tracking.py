!pip install mip



import os

import pandas as pd

import numpy as np

import random

import matplotlib.pyplot as plt

import mip.model as mip

from matplotlib.collections import LineCollection
def excess_return(returns,price,X_1,C):

    import mip.model as mip

    T=returns.shape[0]

    z=[]

    for t in range(1,T+1):

        portfolio_return=[]

        for j in range(1,returns.shape[1]):

            r_jt=returns["security_{}".format(j)][t]

            q_jT=price["security_{}".format(j)][T]

            portfolio_return.append(r_jt*q_jT*X_1["security_{}".format(j)])

        benchmark_return=returns["index"][t]*C

        z.append(mip.xsum(portfolio_return)-benchmark_return)

    return (mip.xsum(z)/T)



def deviation(price,returns,C,X_1,t):

    import mip.model as mip

    theta=C/price["index"].iloc[-1]

    z=[] #For d

    for j in range(1,returns.shape[1]):

        q_jt=price["security_{}".format(j)][t]

        z.append(q_jt*X_1["security_{}".format(j)])

    return (theta*price["index"][t]-mip.xsum(z))
file=5

price=pd.read_csv("../input/index_{}.csv".format(file))

#price=price[0:201]

returns=(price-price.shift(1))/price.shift(1)

returns.drop([0],axis=0,inplace=True)
C=100000 #Capital available

tau=0 #Additional Cash Fund

lamda=0.1 # lower_bound for capital invested in jth stock

nuh=0.3  # upper_bound

k= 12 #Cardinality Constraint

pho=0.2 #Transaction Cost Proportion

c_b=0.01 #Constant for buying cost

c_s=0.01 #Constant for selling cost

f=min(price.min())/3 #Fixed Txn Cost

xii=0.25 #Proportion cosntant for TrE



""" Create the input variables """

n=price.shape[1]-1

X_0=np.zeros((n,1)) #Gives units of jth stock in original portfolio

T=200 #Input for training

theta=C/price["index"][T]

for j in random.sample(range(1,n+1),k):

    X_0[j-1]=(C/k)/price["security_{}".format(j)].iloc[0]
"""Initialisation Phase"""



failure=False

#Solve LP Relaxation

LP = mip.Model("Linear Relaxation of EIT",mip.MAXIMIZE)



#Gives units of jth stock in rebalaced portfolio

X_1 = {x:LP.add_var(name="x1_{}".format(x),var_type="C",lb=0,ub=0) for x in list(returns.columns)[1:]}

#Binary Variable depicting if investor holds stock j

y = {x:LP.add_var(name="y_{}".format(x),var_type="C",lb=0,ub=1) for x in list(returns.columns)[1:]}

#Binary Variable depicting if stock j is traded

w={x:LP.add_var(name="w_{}".format(x),var_type="C",lb=0,ub=1) for x in list(returns.columns)[1:]}

#Buying cost of jth stock

b= {x:LP.add_var(name="b_{}".format(x),var_type="C",lb=0) for x in list(returns.columns)[1:]}

#Selling cost of jth stock

s= {x:LP.add_var(name="s_{}".format(x),var_type="C",lb=0) for x in list(returns.columns)[1:]}

#Downside Devaition

d={x:LP.add_var(name="d_t{}".format(x),var_type="C",lb=0) for x in list(returns.index)}

#Upside Devaition

u={x:LP.add_var(name="u_t{}".format(x),var_type="C",lb=0) for x in list(returns.index)}
""" Objective """

LP+=excess_return(returns,price,X_1,C)



""" Constarints """    

for j in range(1,returns.shape[1]):

    stock="security_{}".format(j)

    q_jT=price[stock][T]

    #Constraint from eqn. 5

    LP+=(lamda*C*y[stock]<= X_1[stock]*q_jT)

    LP+=(X_1[stock]*q_jT <=nuh*C*y[stock])

    #Constraint from eqn. 8

    LP+=(b[stock]-s[stock]==(X_1[stock]*q_jT-X_0[j-1]*q_jT))

    #Constraint from eqn. 9

    LP+=(b[stock]+s[stock]==nuh*C*w[stock])

    #LP+=(b[stock]<=(nuh*C-X_0[j-1]*q_jT)*w[stock]) #Eqn 14

    #LP+=(s[stock]<=X_0[j-1]*q_jT*w[stock]) # Eqn 15





#Constraint from eqn. 6

LP+=(mip.xsum(y.values())<=k)



stocks=["security_{}".format(j) for j in range(1,returns.shape[1])]

#Constraint from eqn. 7

LP+=(mip.xsum([X_1[stock]*price[stock][T] for stock in stocks])==C)



#Constraint from eqn. 10

LP+=(mip.xsum([c_b*b[stock]+c_s*s[stock]+f*w[stock] for stock in stocks])<=pho*C)





for t in range(1,T+1):

    #Constraint from eqn. 4

    LP+=(d[t]-u[t]==deviation(price,returns,C,X_1,t))



#Constraint from eqn. 16

LP+=(mip.xsum([d[t]+u[t] for t in range(1,T+1)])<=xii*C)
print("Optimisastion Status={}".format(str(LP.optimize())))



result=pd.DataFrame()

for stock in stocks:

    temp=pd.DataFrame()

    temp["security"]=[stock]

    temp["X_0"]=X_0[int(stock.split('_')[-1])-1]

    temp["X_1"]=[X_1[stock].x]

    temp["y"]=[y[stock].x]

    temp["w"]=[w[stock].x]

    temp["b"]=[b[stock].x]

    temp["s"]=[s[stock].x]

    result=result.append(temp,ignore_index=True)



result.to_csv("result.csv",index=False)
LP.write("EIT_mip.lp")
#Calulation

q_T=price.iloc[T][1:]

w=result["X_1"].values*q_T.values

w=(w/np.sum(w))

#Initialisation

index=[1]

tracking=[1]

portfolio_return=[]

#Looping

for t in returns.index:

    index.append((1+returns["index"][t])*index[-1])

    portfolio_return.append(sum(w*returns.loc[t][1:].values))

    tracking.append((1+portfolio_return[-1])*tracking[-1])

#Plotting

plot_df=pd.DataFrame()

plot_df["index_value"]=index

plot_df["portfolio_value"]=tracking

plot_df["time_period"]=list(price.index)

plot_df.index=price.index

fig, ax = plt.subplots()

ax.set_xlim(0,price.shape[0])

try:

    ax.set_ylim(-0.3, 1.1*max(index+tracking))

except:

    print ("Error in file={}".format(file))

ind_1=plot_df[["time_period","index_value"]][0:T].values

ind_2=plot_df[["time_period","index_value"]][T:].values

port_1=plot_df[["time_period","portfolio_value"]][0:T].values

port_2=plot_df[["time_period","portfolio_value"]][T:].values

plt.plot(ind_1[:,0],ind_1[:,1],color=(57/255,62/255,68/255,0.7),label="Index")

plt.plot(port_1[:,0],port_1[:,1],color=(255/255,87/255,86/255,0.43),label="Tracking Portfolio")

ax.axvspan(T,price.shape[0],color=(57/255,62/255,68/255),alpha=0.025,label="Outside of Time")

plt.axvline(x=T,color=(0,.20,.40))

plt.legend(frameon=False,loc=2)

cols=[(57/255,62/255,68/255,0.8),(255/255,87/255,86/255,0.8)]

lc = LineCollection([ind_2,port_2],linewidths=(2,2),colors=cols,linestyles=["solid","solid"])

ax.add_collection(lc)

plt.fill_between(x=ind_2[:,0], y1=port_2[:,1]+3*np.std(portfolio_return[T:]),

                  y2=port_2[:,1]-3*np.std(portfolio_return),

                  color=(255/255,87/255,86/255,0.2))

plt.title("Index vs Tracking Portfolio for index=1")
