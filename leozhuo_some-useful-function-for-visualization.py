import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



#Plot the histogram of each gift, also can plot joint histogram of several gifts(2 balls and 3 gloves)



# First get sample data

def get_one_gift(gift,n):

    # this function returns array of sampled gift weight, given gift name and number of it

    if gift == 'horse':

        res = np.maximum(0, np.random.normal(5,2,n)) # mean = 5, var = 4

    elif gift == 'ball':    

        res = np.maximum(0, 1 + np.random.normal(1,0.3,n)) # mean = 1, var = 0.09

    elif gift == 'bike':    

        res = np.maximum(0, np.random.normal(20,10,n)) # mean = 20, var = 100

    elif gift == 'train':    

        res = np.maximum(0, np.random.normal(10,5,n)) # # mean = 10, var = 25

    elif gift == 'coal':    

        res = 47 * np.random.beta(0.5,0.5,n) # mean = 0.5, var = 0.125

    elif gift == 'book':    

        res = np.random.chisquare(2,n) # mean = 2, var = 4

    elif gift == 'doll':    

        res = np.random.gamma(5,1,n) # mean = 5, var = 5

    elif gift == 'blocks':    

        res = np.random.triangular(5,10,20,n) # min = 5, max = 20, mode = 10

    elif gift == 'gloves':   

        gloves = []

        while len(gloves)<n:

            a = 3.0 + np.random.rand(1)[0] if np.random.rand(1) < 0.3 else np.random.rand(1)[0]

            gloves.append(a)

        res = np.array(gloves)

    else:

        res = 0   

    return res



get_one_gift('ball',20)
# Second get the whole sample of all gifts (each 10000) or some of them



# By default, the function will return 10000 sample for each of the 9 gifts as a dict()

# By providing portfolios of gifts, it return 10000 sample for each of the gifts in the portfolio

# Portfolio should be in the form of [(gift_1,num_1),...,(gift_n,num_n)], num_k means the number of the same gift in the bag

def get_sample(portfolio = None):

    # portfolio should be [(gift1,num1),(gift2,num2)]

    n = 10000

    gift_simu = dict()

    if not portfolio:

        

        gift_simu['horse'] = np.maximum(0, np.random.normal(5,2,n)) # mean = 5, var = 4, min = 0

        gift_simu['ball'] = np.maximum(0, 1 + np.random.normal(1,0.3,n)) # mean = 1, var = 0.09, min = 0

        gift_simu['bike'] = np.maximum(0, np.random.normal(20,10,n)) # mean = 20, var = 100, min = 0

        gift_simu['train'] = np.maximum(0, np.random.normal(10,5,n)) # # mean = 10, var = 25, min = 0

        gift_simu['coal'] = 47 * np.random.beta(0.5,0.5,n) # mean = 0.5, var = 0.125

        gift_simu['book'] = np.random.chisquare(2,n) # mean = 2, var = 4

        gift_simu['doll'] = np.random.gamma(5,1,n) # mean = 5, var = 5

        gift_simu['blocks'] = np.random.triangular(5,10,20,n) # min = 5, max = 20, mode = 10

        

        gloves = []

        while len(gloves)<n:

            a = 3.0 + np.random.rand(1)[0] if np.random.rand(1) < 0.3 else np.random.rand(1)[0]

            gloves.append(a)

        gift_simu['gloves'] = np.array(gloves)

    else:

        for gift,num in portfolio:            

            for i in range(num):

                gift_simu[gift] = gift_simu.get(gift,np.zeros(n))+ get_one_gift(gift,n)

    return gift_simu



SAMPLE = get_sample()

print(SAMPLE.keys())

port = [('ball',2),('bike',3)]

sample = get_sample(port)

print(sample.keys())
#Then plot the histogram

# By default, it plots the SAMPLE, thus all of the 9 gifts' histogram

# If sum_all = True, it also plot the sum of all gift weight in the sample per bag.

def plot_sample(sample,sum_all = False):

    count = 1

    if sum_all:

        num_plots = len(sample.keys())+1

        ax1 = plt.subplot(num_plots,1,1)

        for k in sample.keys():

            if count >1:

                ax = plt.subplot(num_plots,1,count,sharex=ax1)

            x = plt.hist(sample[k],bins=100,normed = True)[0]

            plt.hist(sample[k],bins=100,normed = True)    

            plt.title(k)

            plt.yticks([0.,.5 * max(x),max(x)+0.2*max(x)])

            count+=1

        ax = plt.subplot(num_plots,1,count,sharex=ax1)

        X = np.sum(sample[x] for x in sample.keys())

        x = plt.hist(X,bins=100,normed = True)[0]

        plt.hist(X,bins=100,normed = True)

        plt.title('Sum of all gifts')

        plt.yticks([0.,.5*max(x),max(x)+0.2*max(x)])

    else:

        num_plots = len(sample.keys())

        ax1 = plt.subplot(num_plots,1,1)

        for k in sample.keys():

            if count >1:

                ax = plt.subplot(num_plots,1,count,sharex=ax1)

            x = plt.hist(sample[k],bins=100,normed = True)[0]

            plt.hist(sample[k],bins=100,normed = True)    

            plt.title(k)

            plt.yticks([0.,.5 * max(x),max(x)+0.2*max(x)])

            count+=1

    plt.show()

    

plot_sample(SAMPLE)

plot_sample(sample,True)