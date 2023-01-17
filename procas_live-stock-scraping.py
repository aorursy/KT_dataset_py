import bs4

from bs4 import BeautifulSoup

import requests
from datetime import datetime



def getDate():

    r=requests.get("https://finance.yahoo.com/quote/INTC?p=INTC")

    soup=bs4.BeautifulSoup(r.text, "html.parser")

    date=soup.find_all('div', {'id':'quote-market-notice'}, {'class': 'C($tertiaryColor) D(b) Fz(12px) Fw(n) Mstart(0)--mobpsm Mt(6px)--mobpsm'})[0].find('span').text

    #date=date+date.today()

    return datetime.today().strftime("%m--%d")



def parsePriceIntel():

    r=requests.get("https://finance.yahoo.com/quote/INTC?p=INTC")

    soup=bs4.BeautifulSoup(r.text, "html.parser")

##price=soup.find('div', {'class':'My(6px) Pos(r) smartphone_Mt(6px)'})

    soup

    price=soup.find_all('div', {'class':'My(6px) Pos(r) smartphone_Mt(6px)'})[0].find('span').text

    return price



def parsePriceBA():

    r=requests.get("https://finance.yahoo.com/quote/BA?p=BA")

    soup=bs4.BeautifulSoup(r.text, "html.parser")

##price=soup.find('div', {'class':'My(6px) Pos(r) smartphone_Mt(6px)'})

    soup

    price=soup.find_all('div', {'class':'My(6px) Pos(r) smartphone_Mt(6px)'})[0].find('span').text

    return price



def parsePriceJohnson():

    r=requests.get("https://finance.yahoo.com/quote/JNJ?p=JNJ")

    soup=bs4.BeautifulSoup(r.text, "html.parser")

##price=soup.find('div', {'class':'My(6px) Pos(r) smartphone_Mt(6px)'})

    soup

    price=soup.find_all('div', {'class':'My(6px) Pos(r) smartphone_Mt(6px)'})[0].find('span').text

    return price



def parsePriceDisney():

    r=requests.get("https://finance.yahoo.com/quote/DIS?p=DIS")

    soup=bs4.BeautifulSoup(r.text, "html.parser")

##price=soup.find('div', {'class':'My(6px) Pos(r) smartphone_Mt(6px)'})

    soup

    price=soup.find_all('div', {'class':'My(6px) Pos(r) smartphone_Mt(6px)'})[0].find('span').text

    return price



def parsePriceMicrosoft():

    r=requests.get("https://finance.yahoo.com/quote/MSFT?p=MSFT")

    soup=bs4.BeautifulSoup(r.text, "html.parser")

##price=soup.find('div', {'class':'My(6px) Pos(r) smartphone_Mt(6px)'})

    soup

    price=soup.find_all('div', {'class':'My(6px) Pos(r) smartphone_Mt(6px)'})[0].find('span').text

    return price



def parsePriceIbm():

    r=requests.get("https://finance.yahoo.com/quote/IBM?p=IBM")

    soup=bs4.BeautifulSoup(r.text, "html.parser")

##price=soup.find('div', {'class':'My(6px) Pos(r) smartphone_Mt(6px)'})

    soup

    price=soup.find_all('div', {'class':'My(6px) Pos(r) smartphone_Mt(6px)'})[0].find('span').text

    return price
dataset=pd.read_csv('file1.csv')

#dataset.drop(dataset.columns[[0]], axis=1, inplace=True)

dataset
dataset.drop(dataset.columns[[0]], axis=1, inplace=True)

dataset
##OPTIMIZED



import matplotlib.pyplot as plt

import numpy as np

l=dataset.shape[1]



beta=0.2

alpha=0.2

outcome=[1 for i in range(l)]

w=np.ones(l-2)

w=w.tolist()

probability=np.zeros(3).tolist()

expert=[[0 for i in range(l)] for i in range(l-2)]



for i in range(l):

    if((dataset.iloc[l-2,i]-dataset.iloc[l-3,i])>0):

        outcome[i]=-1

    if((dataset.iloc[l-2,i]-dataset.iloc[l-3,i])<0):

        outcome[i]=1

    if((dataset.iloc[1-2,i]-dataset.iloc[l-3,i])==0):

        outcome[i]=0



        

for i in range(0,l-2):

    for j in range(l):

        if(dataset.iloc[i+1,j]-dataset.iloc[i,j]>0):

            expert[i][j]=-1

        elif(dataset.iloc[i+1,j]-dataset.iloc[i,j]<0):

            expert[i][j]=1

        elif(dataset.iloc[i+1,j]-dataset.iloc[i,j]==0):

            expert[i][j]=0



expertdf=pd.DataFrame(expert)

##expertdf = expertdf.drop([1,0],axis=0)

expertdf



pred=np.ones(l).tolist()



## reduce weights for |S-1|

for i in range(l):

    w_minus=0

    w_zero=0

    w_one=0

    for j in range(l-3):

        if(expert[j][i] != outcome[j]):

            w[j]=w[j]-w[j]*beta

      

        

w1=list.copy(w)







for i in range(l):

    w_minus=0

    w_zero=0

    w_one=0

    for j in range(l-2):

        if(expert[j][i] != outcome[j]):

            w[j]=w[j]-w[j]*beta

       

    for j in range(l-2):

        if(expertdf.iloc[j,i]==-1):

            w_minus=w_minus+w[j]

        elif(expertdf.iloc[j,i]==1):

            w_one=w_one+w[j]

        elif(expertdf.iloc[j,i]==0):

            w_zero=w_zero+w[j]

        p_one=w_one/(w_one+w_minus+w_zero)

        p_minus=w_minus/(w_one+w_minus+w_zero)

        p_zero=w_zero/(w_one+w_minus+w_zero)

        

        probability[0]=p_one

        probability[1]=p_minus

        probability[2]=p_zero

    

    colors = ['gold', 'yellowgreen', 'lightcoral']

    plt.pie(probability, autopct='%1.0f%%', shadow=True, colors=colors)

    plt.axis('equal')

    plt.show()

    y=np.random.choice(3, 1, p=probability)

    if(y==0):

        print(1) 

    elif(y==1):

        print(-1)

    elif(y==0): 

        print(0)

    

    

        y=np.random.choice(3, 1, p=probability)

    if(y==0):

        pred[i] = 1

    elif(y==1):

        pred[i] = -1

    elif(y==0): 

        pred[i] = 0

        

print(pred)





fig = plt.figure()

ax1 = fig.add_subplot(111)

date=['31/01', '03/02', '04/02', '05/02']

ax1.scatter(x=date, y=w1, color='blue', marker="s", label='first')

ax1.scatter(x=date, y=w, color='red', marker="o", label='second')

plt.legend(loc='upper left');

plt.show()



print(w1)

print(w)

sm=difflib.SequenceMatcher(None,pred,outcome)

sm.ratio()
##RANDOMIZED 





import matplotlib.pyplot as plt

import numpy as np

l=dataset.shape[1]

p=np.zeros(l-2).tolist()

beta=0.2

def all_prob(w):

    prob=np.zeros(l-2).tolist()

    c=0

    for i in w:

        prob[c]=i/total(w)

        p[c]=prob[c]

        c+=1

def total(w):

    sum=0

    for j in w:

        sum=sum+j

    return sum

outcome=[1 for i in range(l)]

w=np.ones(l-2)

w=w.tolist()

probability=np.zeros(3).tolist()

expert=[[0 for i in range(l)] for i in range(l-2)]



for i in range(l):

    if((dataset.iloc[l-2,i]-dataset.iloc[l-3,i])>0):

        outcome[i]=-1

    if((dataset.iloc[l-2,i]-dataset.iloc[l-3,i])<0):

        outcome[i]=1

    if((dataset.iloc[1-2,i]-dataset.iloc[l-3,i])==0):

        outcome[i]=0



        

for i in range(0,l-2):

    for j in range(l):

        if(dataset.iloc[i+1,j]-dataset.iloc[i,j]>0):

            expert[i][j]=-1

        elif(dataset.iloc[i+1,j]-dataset.iloc[i,j]<0):

            expert[i][j]=1

        elif(dataset.iloc[i+1,j]-dataset.iloc[i,j]==0):

            expert[i][j]=0



expertdf=pd.DataFrame(expert)

##expertdf = expertdf.drop([1,0],axis=0)

expertdf

x=np.zeros(l).tolist()

for i in range(l):

     for j in range(l-2):

        if(expert[j][i] != outcome[j]):

            w[j]=w[j]-w[j]*beta

     colors = ['gold', 'yellowgreen', 'lightcoral','pink','brown', 'yellow']

     plt.pie(w, autopct='%1.1f%%', shadow=True, colors=colors)



     plt.axis('equal')

     plt.show()

     all_prob(w)

    ###choose the opinion of any expert based on their respective probabilities

     y=np.random.choice(l-2, 1, p=p)

     x[i]=expert[y[0]][i]

    

print(x)
## compare with outcome:

import difflib

sm=difflib.SequenceMatcher(None,x,outcome)

sm.ratio()
##REINFORCEMENT





import matplotlib.pyplot as plt

import numpy as np

l=dataset.shape[1]

p=np.zeros(l-2).tolist()

beta=0.1

alpha=0.2

def all_prob(w):

    prob=np.zeros(l-2).tolist()

    c=0

    for i in w:

        prob[c]=i/total(w)

        p[c]=prob[c]

        c+=1

def total(w):

    sum=0

    for j in w:

        sum=sum+j

    return sum

outcome=[1 for i in range(l)]

w=np.ones(l-2)

w=w.tolist()

probability=np.zeros(3).tolist()

expert=[[0 for i in range(l)] for i in range(l-2)]



for i in range(l):

    if((dataset.iloc[l-2,i]-dataset.iloc[l-3,i])>0):

        outcome[i]=-1

    if((dataset.iloc[l-2,i]-dataset.iloc[l-3,i])<0):

        outcome[i]=1

    if((dataset.iloc[1-2,i]-dataset.iloc[l-3,i])==0):

        outcome[i]=0



        

for i in range(0,l-2):

    for j in range(l):

        if(dataset.iloc[i+1,j]-dataset.iloc[i,j]>0):

            expert[i][j]=-1

        elif(dataset.iloc[i+1,j]-dataset.iloc[i,j]<0):

            expert[i][j]=1

        elif(dataset.iloc[i+1,j]-dataset.iloc[i,j]==0):

            expert[i][j]=0



expertdf=pd.DataFrame(expert)

##expertdf = expertdf.drop([1,0],axis=0)

expertdf

x=np.zeros(l).tolist()

for i in range(l):

     for j in range(l-2):

        if(expert[j][i] != outcome[j]):

            w[j]=w[j]-w[j]*beta

        else:

            w[j]=w[j]+w[j]*alpha

     colors = ['gold', 'yellowgreen', 'lightcoral','pink','brown', 'yellow']

     plt.pie(w, autopct='%1.1f%%', shadow=True, colors=colors)



     plt.axis('equal')

     plt.show()

     all_prob(w)

    ###choose the opinion of any expert based on their respective probabilities

     y=np.random.choice(l-2, 1, p=p)

     x[i]=expert[y[0]][i]

    

print(x)
sm=difflib.SequenceMatcher(None,x,outcome)

sm.ratio()
expertdf
## reduce weights for |S|

for i in range(l):

    w_minus=0

    w_zero=0

    w_one=0

    for j in range(l-2):

        if(expert[j][i] != outcome[j]):

            w[j]=w[j]-w[j]*beta

    for j in range(l-2):

        if(expertdf.iloc[j,i]==-1):

            w_minus=w_minus+w[j]

        elif(expertdf.iloc[j,i]==1):

            w_one=w_one+w[j]

        elif(expertdf.iloc[j,i]==0):

            w_zero=w_zero+w[j]

        p_one=w_one/(w_one+w_minus+w_zero)

        p_minus=w_minus/(w_one+w_minus+w_zero)

        p_zero=w_zero/(w_one+w_minus+w_zero)

        

        probability[0]=p_one

        probability[1]=p_minus

        probability[2]=p_zero

        

        y=np.random.choice(3, 1, p=probability)

    if(y==0):

        pred[i] = 1

    elif(y==1):

        pred[i] = -1

    elif(y==0): 

        pred[i] = 0

        

print(pred)
import pandas as pd

file1 = pd.read_csv("../input/file1.csv")

file1.to_csv('file1.csv',index=False)

pd.read_csv('file1.csv')
prices=[ float(parsePriceIbm()), float(parsePriceMicrosoft()), float(parsePriceDisney()), float(parsePriceJohnson()), float(parsePriceBA()), float(parsePriceIntel())]

data={'06/02/20':prices}

dff=pd.DataFrame.from_dict(data, orient='index')

with open('file1.csv', 'r') as infile:

    dff.to_csv('file1.csv', mode='a', header=False)

#pd.read_csv('/users/promamukherjee/Desktop/file1.csv')
pd.read_csv('file1.csv')
## Add 7th Feb