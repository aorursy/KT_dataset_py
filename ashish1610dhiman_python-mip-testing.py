!pip install mip



import os

import pandas as pd

import numpy as np

import random

import mip.model as mip
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
file=1

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

T=T=returns.shape[0]

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