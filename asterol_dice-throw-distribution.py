import random

import matplotlib

import matplotlib.pyplot as plt



dict={}



def throw():

    global x    

    x+=random.randint(1,6)



def multi_throw(dice_amount):

    global x

    x=0

    for i in range(dice_amount):

        throw()

    dict[dice_amount].append(x)

    

def multi_times(time_amount, dice_amount):

    dict[dice_amount]=[]

    for i in range(time_amount):

        multi_throw(dice_amount)
x=0

th_amount=100000



for i in range(20):

    multi_times(th_amount, i)
fig, ax = plt.subplots()

plt.xlabel('score')

plt.ylabel('occurences')

plt.title('Histogram of '+str(th_amount)+' dice throws')



ax.hist(dict[1],

        bins=range(0,8),

        color="green",

        alpha=0.8,

        )



plt.show()
fig, ax = plt.subplots()

plt.xlabel('score')

plt.ylabel('occurences')

plt.title('Histogram of '+str(th_amount)+' dice throws')



ax.hist(dict[2],

        bins=range(0,20),

        color="red",

        alpha=0.8,

        )



plt.show()
fig, ax = plt.subplots()

plt.xlabel('score')

plt.ylabel('occurences')

plt.title('Histogram of '+str(th_amount)+' dice throws')



ax.hist(dict[2],

        bins=range(0,15),

        color="blue",

        alpha=0.8,

        )



plt.show()
fig, ax = plt.subplots()

plt.xlabel('score')

plt.ylabel('occurences')

plt.title('Histogram of '+str(th_amount)+' dice throws')





ax.hist(dict[3],

        bins=range(0,36),

        color="red",

        alpha=0.8,

        )



plt.show()
fig, ax = plt.subplots()

plt.xlabel('score')

plt.ylabel('occurences')

plt.title('Histogram of '+str(th_amount)+' dice throws')



histy='step'



ax.hist(dict[6],

        bins=range(0,36),

        color="red",

        alpha=0.8,

        )



plt.show()
fig, ax = plt.subplots()

plt.xlabel('score')

plt.ylabel('occurences')

plt.title('Histogram of '+str(th_amount)+' dice throws')



histy='step'



ax.hist(dict[19],

        color="green",

        bins=range(40,120),

        alpha=0.8,

        )



plt.show()
fig, ax = plt.subplots()

plt.xlabel('score')

plt.ylabel('occurences')

plt.title('Histogram of '+str(th_amount)+' dice throws')



histy='step'



ax.hist(dict[6],

        bins=range(2,36),

        color="red",

        alpha=1.0,

        )



ax.hist(dict[5],

        bins=range(2,36),

        color="blue",

        alpha=0.8,

        )



ax.hist(dict[4], 

        bins=range(2,36),

        color="green",

        alpha=0.7,

        )



ax.hist(dict[3], 

        bins=range(2,36),

        color="yellow",

        alpha=0.5,

        )



ax.legend(["6 dices", "5 dices", "4 dices", "3 dices"])



plt.show()
import seaborn as sns



sns.distplot(dict[3], kde=False)
sns.distplot(dict[3], kde=False)

sns.distplot(dict[4], kde=False)
sns.distplot(dict[3], label="3 dices", kde=False)

sns.distplot(dict[4], label="4 dices", kde=False)

sns.distplot(dict[5], label="5 dices", kde=False)

sns.distplot(dict[6], label="6 dices", kde=False)

plt.legend()
for i in range(1,14):

    sns.distplot(dict[i], kde=False)



plt.show()