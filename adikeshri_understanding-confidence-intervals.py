import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

sns.set()
n=600

no_vanilla=400

no_chocolate=200
p=no_vanilla/n

std_error=((p*(1-p))/n)**0.5

print('We estimate, with 95% confidence, that the proportion of people who prefer vanilla over chocolate '

      +'lies withing the range: ('+ str(round(p-1.96*std_error,4))+','+str(round(p+1.96*std_error,4))+')')

print('The estimation is centered at: '+str(round(p,4)))
std_error=((p*(1-p))/n)**0.5

print('We estimate, with 99% confidence, that the proportion of people who prefer vanilla over chocolate '

      +'lies withing the range: ('+ str(round(p-2.58*std_error,4))+','+str(round(p+2.58*std_error,4))+')')

print('The estimation is centered at: '+str(round(p,4)))
mu=50

sigma=25

x=np.random.normal(mu,sigma,1000)

plt.figure(figsize=(14,8))

sns.distplot(x)

plt.show()
x=np.array(x)

mu=np.mean(x)

sigma=np.std(x)

in_range=0

print('Thoery:- 68% of the values in x should lie between: ' + str(round(mu-sigma,4))+ ' to ' + str(round(mu+sigma,4)))

for i in x:

    if i>=mu-sigma and i<=mu+sigma:

        in_range+=1

print('Total percentage of values which actually lies between the above range: ' + str(in_range/10)+'%')
in_range=0

print('Thoery:- 95% of the values in x should lie between: ' + str(round(mu-2*sigma,4))+ ' to ' + str(round(mu+2*sigma,4)))

for i in x:

    if i>=mu-2*sigma and i<=mu+2*sigma:

        in_range+=1

print('Total percentage of values which actually lies between the above range: ' + str(in_range/10)+'%')
in_range=0

print('Thoery:- 99.7% of the values in x should lie between: ' + str(round(mu-3*sigma,4))+ ' to ' + str(round(mu+3*sigma,4)))

for i in x:

    if i>=mu-3*sigma and i<=mu+3*sigma:

        in_range+=1

print('Total percentage of values which actually lies between the above range: ' + str(in_range/10)+'%')
plt.figure(figsize=(14,8))

sns.distplot(x)

plt.plot([mu-sigma,mu+sigma],[0.0175,0.0175],color='maroon')

plt.plot([mu-sigma,mu-sigma],[0,0.0175],color='maroon',linestyle='dotted')

plt.plot([mu+sigma,mu+sigma],[0,0.0175],color='maroon',linestyle='dotted')

plt.text(mu-0.75*sigma,0.0177,'68% of the values lie here')



plt.plot([mu-2*sigma,mu+2*sigma],[0.0125,0.0125],color='maroon')

plt.plot([mu-2*sigma,mu-2*sigma],[0,0.0125],color='maroon',linestyle='dotted')

plt.plot([mu+2*sigma,mu+2*sigma],[0,0.0125],color='maroon',linestyle='dotted')

plt.text(mu-0.75*sigma,0.0127,'95% of the values lie here')



plt.plot([mu-3*sigma,mu+3*sigma],[0.0075,0.0075],color='maroon')

plt.plot([mu-3*sigma,mu-3*sigma],[0,0.0075],color='maroon',linestyle='dotted')

plt.plot([mu+3*sigma,mu+3*sigma],[0,0.0075],color='maroon',linestyle='dotted')

plt.text(mu-0.75*sigma,0.0077,'99.7% of the values lie here')

plt.show()

p=no_vanilla/n

std_error=1/(n**0.5)

print('The conservative confidence interval for 95% confidence level is: (' +

      str(round(p-1.96*std_error,4))+','+str(round(p+1.96*std_error,4)) + ')')

print('Difference between them: ' + str(round(2*1.96*std_error,4)))
ice_cream=[1]*400 #1->People answered 'vanilla'

ice_cream+=[0]*200 #0->People answered 'chocolate'
import scipy.stats

def calc_mean_confidence_interval(data, confidence=0.95):

    a=1.0*np.array(data)

    n=len(a)

    m,se=np.mean(a), scipy.stats.sem(a)

    h=se*scipy.stats.t.ppf((1+confidence)/2.,n-1)

    return round(m,4),round(m-h,4),round(m+h,4)
p,a,b=calc_mean_confidence_interval(ice_cream)

print('The 95% confidence interval for our ice_cream data using scipy is: ('+str(a)+','+str(b)+')'

     +', with value centered at: '+str(p))



p,a,b=calc_mean_confidence_interval(ice_cream,0.99)

print('The 99% confidence interval for our ice_cream data using scipy is: ('+str(a)+','+str(b)+')'

     +', with value centered at: '+str(p))