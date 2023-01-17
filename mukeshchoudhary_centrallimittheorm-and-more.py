import numpy as np

import random

import matplotlib.pyplot as plt
shape, scale= 2., 2.,  #mean= 4, std-dev= 2*sqrt(2)

mu= shape*scale

sigma= scale*np.sqrt(shape)

s= np.random.gamma(shape, scale, 1000000)
rs= random.choices(s, k= 10000)
# set k

ks= [0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

#prob list

probs= []



for k in ks:

    #start count

    c=0

    # for each data sample

    for i in rs:

        # count if far from mean in k standard deviation

        if abs(i- mu)> k*sigma:

            c+=1

        # count divided by number of sample

    probs.append(c/10000)
# set figure size

plt.figure(figsize=(20,10))

# plot each probability

plt.plot(ks, probs, marker= 'o')

plt.show()

# print each probability

print("Probability of a sample far from mean more than k standard deviation:")

for i, prob in enumerate(probs):

    print("k:" + str(ks[i]) + ", probability: " \

          + str(prob)[0:5] + \

          " | in theory, probability should less than: " \

          + str(1/ks[i]**2)[0:5])
shape, scale = 2., 2.  # mean=4, std=2*sqrt(2)

s = np.random.gamma(shape, scale, 1000000)
samplemeanlist = [] # list of sample mean

l = [] # list of smaple size, for x-axis of box plots

numberofsample = 50 # number of sample in each sample size

    

# set sample size (i) between 100 to 8100, step by 500

for i in range(100,8101,500):

    # set x-axis

    l.append(i)

    # list of mean of each sample

    ml = []

    # sample 50 time.

    for n in range(0,numberofsample):

        # random pick from population with sample size = i

        rs = random.choices(s, k=i)

        # calculate the mean of each sample and save it in list of mean.

        ml.append(sum(rs)/i)  

    

    # save the 50 sample mean in samplemeanlist for box plots.

    samplemeanlist.append(ml)
# set figure size

plt.figure(figsize=(20,10))

# plot box plots of each sample mean

plt.boxplot(samplemeanlist,labels = l)

# show plot

plt.show()
histplot = plt.figure(figsize=(20,10))

plt.hist(samplemeanlist[0], 10, density=True)

plt.hist(samplemeanlist[16], 10, density=True)

histplot.show()
print("sample with 100 sample size," + \

      "mean:" + str(np.mean(samplemeanlist[0])) + \

      ", standard deviation: "+ str(np.std(samplemeanlist[0])))

print("sample with 8100 sample size," + \

      "mean:" + str(np.mean(samplemeanlist[16])) + \

      ", standard deviation: "+ str(np.std(samplemeanlist[16])))

# build gamma distribution as population

shape, scale = 2., 2.  # mean=4, std=2*sqrt(2)

s = np.random.gamma(shape, scale, 1000000)
## sample from population with different number of sampling

# a list of sample mean

meansample = []

# number of sample

numofsample = [1000,2500,5000,10000,25000,50000]

# sample size

samplesize = 500

# for each number of sampling (1000 to 50000)

for i in numofsample:

    # collect mean of each sample

    eachmeansample = []

    # for each sampling

    for j in range(0,i):

        # sampling 500 sample from population

        rc = random.choices(s, k=samplesize)

        # collect mean of each sample

        eachmeansample.append(sum(rc)/len(rc))

    # add mean of each sampling to the list

    meansample.append(eachmeansample)
# plot

cols = 2

rows = 3

fig, ax = plt.subplots(rows, cols, figsize=(20,15))

n = 0

for i in range(0, rows):

    for j in range(0, cols):

        ax[i, j].hist(meansample[n], 200, density=True)

        ax[i, j].set_title(label="number of sampling :" + str(numofsample[n]))

        n += 1
# use last sampling

sm = meansample[len(meansample)-1]
# calculate start deviation

std = np.std(sm)

# set population mean

mean = np.mean(sm)
# list of standarded sample

zn = []

# for each sample subtract with mean and devided by standard deviation

for i in sm:

    zn.append((i-mean)/std)
import scipy.stats as stats
# plot hist

plt.figure(figsize=(20,10))

plt.hist(zn, 200, density=True)

# compare with standard normal disrtibution line

mu = 0

sigma = 1

x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)

# draw standard normal disrtibution line

plt.plot(x, stats.norm.pdf(x, mu, sigma),linewidth = 5, color='red')

plt.show()
## sample with different sample size

# list of sample mean

meansample = []

# number of sampling

numofsample = 25000

# sample size

samplesize = [1,5,10,30,100,1000]

# for each sample size (1 to 1000)

for i in samplesize:

    # collect mean of each sample

    eachmeansample = []

    # for each sampling

    for j in range(0,numofsample):

        # sampling i sample from population

        rc = random.choices(s, k=i)

        # collect mean of each sample

        eachmeansample.append(sum(rc)/len(rc))

    # add mean of each sampling to the list

    meansample.append(eachmeansample)
# plot

cols = 2

rows = 3

fig, ax = plt.subplots(rows, cols, figsize=(20,15))

n = 0

for i in range(0, rows):

    for j in range(0, cols):

        ax[i, j].hist(meansample[n], 200, density=True)

        ax[i, j].set_title(label="sample size :" + str(samplesize[n]))

        n += 1
## expect value of sample

# use last sampling

sample = meansample[5]

# expected value of sample equal to expect value of population

print("expected value of sample:", np.mean(sample))

print("expected value of population:", shape*scale)

# standard deviation of sample equl to standard deviation of population divided by squre root of n

print("standard deviation of sample:", np.std(sample))

print("standard deviation of population:", scale*np.sqrt(shape))

print("standard deviation of population divided by squre root of sample size:", scale*np.sqrt(shape)/np.sqrt(1000))
# build gamma distribution as population

shape, scale = 2., 2.  # mean=4, std=2*sqrt(2)

s = np.random.gamma(shape, scale, 1000000)
## show that as the sample size increases the mean of sample is close to population mean

# set expected values of population

mu = shape*scale # mean

# sample size

samplesize = []

# collect difference between sample mean and mu

diflist = []

# for each sample size

for n in range(10,20000,20): 

    # sample n sample

    rs = random.choices(s, k=n)

    # start count

    c = 0

    # calculate mean

    mean = sum(rs)/len(rs)

    # collect difference between sample mean and mu

    diflist.append(mean-mu)

    samplesize.append(n)
# set figure size.

plt.figure(figsize=(20,10))

# plot each diference.

plt.scatter(samplesize,diflist, marker='o')

# show plot.

plt.show()
# build gamma distribution as population

shape, scale = 2., 2.  # mean=4, std=2*sqrt(2)  

mu = shape*scale # mean

s = np.random.gamma(shape, scale, 1000000)

# margin of error

epsilon = 0.05
# list of probability of each sample size

proberror = []

# sample size for plotting

samplesize = []

# for each sample size

for n in range(100,10101,500): 

    # start count

    c = 0

    for i in range(0,100):

        # sample n sample

        rs = random.choices(s, k=n)

        # calculate mean

        mean = sum(rs)/len(rs)

        # check if the difference is larger than error

        if abs(mean - mu) > epsilon:

            # if larger count the sampling

            c += 1

    # calculate the probability

    proberror.append(c/100)

    # save sample size for plotting

    samplesize.append(n)
# set figure size.

plt.figure(figsize=(20,10))

# plot each probability.

plt.plot(samplesize,proberror, marker='o')

# show plot.

plt.show()