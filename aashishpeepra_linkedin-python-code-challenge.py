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
def checker(val):
    data=val.split()
    data.sort(lambda a,b: cmp(a.lower(),b.lower()))
    return " ".join(data)

checker("hello Ji")
def checker2(val):
    data=val.split()
    for i in range(len(data)):
        for j in range(0,len(data)-1-i):
            if data[j].lower()>data[j+1].lower():
                data[j],data[j+1]=data[j+1],data[j]
    return " ".join(data)
checker2("ORANGE apple BYe")
isinstance([],(list))
def nest(val,curr,res):
    if isinstance(val,(list)):
        for i in range(len(val)):
            nest(val[i],curr.append(i))
    else:
        res.append(current)
        curr=[]
        print(val)
data=[1,[1,2],[4,5,6]]
nest(data,[],[])
def nester(data,val):
    current=list()
    print(data,val,current)
    for i in range(len(data)):
        if data[i]==val:
            current.append([i])
        elif isinstance(data[i],(list)):
            for index in nester(data[i],val):
                current.append([i]+index)
    return current
            
data=[[1,2],1,[2,1]]
nester(data,1)
import time
def game(sec):
    print("Try to guess when {} Seconds is done.".format(sec))
    input("Press Enter if you are ready")
    for i in range(1,4):
        print("Game Starts in {}".format(i))
        time.sleep(1)
    print("----Start----")
    d1=time.time()
    d2=0
    while True:
        val=input()
        if val!="" or val=="":
            d2=time.time()
            break
    val=round((d2-d1),3)
    if val<sec:
        print("You are too fast",round((sec-val),2))
    elif val>sec:
        print("You are too slow",round((-sec+val),2))
    else:
        print("You Won!")
game(4)
time.time()
def put(file,dic):
    with open(file,"w") as f:
        for a,b in dic.items():
            f.write(str(a)+","+str(b)+"\n")
def load(file):
    temp={}
    with open(file,"r") as f:
        for each in f.readlines():
            temp.update({each[:each.index(",")]:each[each.index(",")+1]})
    return temp
data={str(k):k**2 for k in range(1,10)}
put("temp.txt",data)
load("temp.txt")
import pickle

def putter(data,file):
    with open(file,"wb") as f:
        pickle.dump(data,f)
def getter(file):
    with open(file,"rb") as f:
        return pickle.load(f)
putter(data,"temp.pickle")
getter("temp.pickle")
import winsound
!pip install playsound
!pip install schedule
from playsound import playsound
import sched
import time

def alarm(time1,file,message):
    s=sched.scheduler(time.time,time.sleep)
    s.enterabs(time1,1,print,argument=(message,))
    s.enterabs(time1,1,playsound,argument=(file,))
    s.run()
import time
alarm(time.time()+5,"/kaggle/input/musicc.wav","Alarm")
def temp(a,b,*args,**kwargs):
    print(args)
    print(kwargs)
import random
import collections
def dice(*args):
    chances=dict()
    output=[]
    for i in range(1000000):
        res=sum([random.randint(1,k) for k in args])
        output.append(res)
        chances.update({res:output.count(res)})
    chances=sorted(chances.items(), key=lambda x: x[1],reverse=True)
    for a,b in chances:
        print("{:2} : {:2} %".format(a,(b*100)/100))
