import pandas as pd
import matplotlib.pyplot as plt

import random
import operator
import copy


def enter():
    for k in temp.keys():
        coord = input(f"Please input a desired coordinate for {k}: ")
        x = int(coord[0])
        y = int(coord[2])

        temp[k] = (x, y)

def check1(k):
    important = [k]
    for key, value in temp.items():
        if temp[k][0] == value[0] and k != key and temp[k][1] < value[1]:
            important.append(key)
    return important


def check2(futurex):  # futurex=>4
    stack = 0
    for k in temp.values():
        if k[0] == futurex:
            stack += 1
    return stack


def drop(imp):
    imp.sort(key=lambda x: temp[x][1])
    for k in imp:
        temp[k] = (temp[k][0], imp.index(k) + 1)
    return imp


def move(p):
    imp = drop(check1(p))
    dice = random.randint(1, 3)
    futurex = dice + temp[p][0]
    ystack = check2(futurex)
    for k in imp:  # check for 4
        x = temp[k][0]
        y = temp[k][1]
        newx = x + dice
        newy = y + ystack
        temp[k] = (newx, newy)


# ================================================
temp = {"Orange": (0,0), "Blue": (0,0), "Purple": (0,0), "Red": (0,0), "Green": (0,0)}

enter()
dicegen = random.SystemRandom()
order_list = list(temp.keys())
biglist = []

for i in range(10000):
    random.shuffle(order_list)
    reset = copy.deepcopy(temp)
    for x in order_list:
        move(x)
    biglist.append(temp)

    temp = reset

alist = []
blist = []
clist = []
dlist = []
elist = []


def biglist1():
    for dictionary in biglist:
        alist.append(dictionary["Orange"])
        blist.append(dictionary["Blue"])
        clist.append(dictionary["Purple"])
        dlist.append(dictionary["Red"])
        elist.append(dictionary["Green"])


biglist1()

def unique(xlist):
    unique_list = []
    for x in xlist:
        if x not in unique_list:
            unique_list.append(x)
    return unique_list


newalist = unique(alist)
newblist = unique(blist)
newclist = unique(clist)
newdlist = unique(dlist)
newelist = unique(elist)

newalist.sort(key=lambda x: x[0])
newblist.sort(key=lambda x: x[0])
newclist.sort(key=lambda x: x[0])
newdlist.sort(key=lambda x: x[0])
newelist.sort(key=lambda x: x[0])



def cumua():
    jupyterlist = []
    for x in newalist:
        jupyterlist.append((x, alist.count(x)))
    return (jupyterlist)


def cumub():
    jupyterlist2 = []
    for x in newblist:
        jupyterlist2.append((x, blist.count(x)))
    return (jupyterlist2)


def cumuc():
    jupyterlist3 = []
    for x in newclist:
        jupyterlist3.append((x, clist.count(x)))
    return (jupyterlist3)


def cumud():
    jupyterlist4 = []
    for x in newdlist:
        jupyterlist4.append((x, dlist.count(x)))
    return (jupyterlist4)


def cumue():
    jupyterlist5 = []
    for x in newelist:
        jupyterlist5.append((x, elist.count(x)))
    return (jupyterlist5)


cumua()
cumub()
cumuc()
cumud()
cumue()

df1 = pd.DataFrame(cumua(), columns=['word', 'Orange'])
df2 = pd.DataFrame(cumub(), columns=['word', 'Blue'])
df3 = pd.DataFrame(cumuc(), columns=['word', 'Purple'])
df4 = pd.DataFrame(cumud(), columns=['word', 'Red'])
df5 = pd.DataFrame(cumue(), columns=['word', 'Green'])

df1.plot(kind='bar', x='word', color='Orange')
df2.plot(kind='bar', x='word', color= 'Blue')
df3.plot(kind='bar', x='word', color='Purple')
df4.plot(kind='bar', x='word', color='Red')
df5.plot(kind='bar', x='word', color='Green')


plt.show()