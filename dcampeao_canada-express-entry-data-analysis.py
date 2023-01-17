import pandas as pd

import matplotlib.pyplot as plt

import matplotlib.mlab as mlab

import numpy as np

import scipy.stats

import datetime

%matplotlib inline
candidateScore = 465

data_ori = pd.read_csv('../input/ExpressEntry.csv')

data_ori['dayOfInvitation'] = pd.to_datetime(data_ori['dayOfInvitation'])
data_ori.head()
fig=plt.figure(figsize=(12, 8), dpi= 80)

plt.plot(data_ori['dayOfInvitation'], data_ori['lowestScore'] )

plt.xlabel('Date')

plt.ylabel('Score')
data = data_ori.iloc[38:,:]

data.head()
scrMean = data['lowestScore'].mean()

scrStd = data['lowestScore'].std()



fig=plt.figure(figsize=(12, 8), dpi= 80)

x = np.linspace(scrMean - 3*scrStd, scrMean + 3*scrStd, 100)

plt.plot(x,mlab.normpdf(x, scrMean, scrStd), label='Pdf')

plt.plot([candidateScore,candidateScore],[0,scipy.stats.norm(scrMean, scrStd).pdf(candidateScore)],color='green', label='Candidate score')

plt.plot([scrMean,scrMean],[0,scipy.stats.norm(scrMean, scrStd).pdf(scrMean)],color='red', label='50.0%', linestyle='--')

plt.plot([scrMean+scrStd,scrMean+scrStd],[0,scipy.stats.norm(scrMean, scrStd).pdf(scrMean+scrStd)],color='red', label='84.1%', linestyle=':')

plt.plot([scrMean+2*scrStd,scrMean+2*scrStd],[0,scipy.stats.norm(scrMean, scrStd).pdf(scrMean+2*scrStd)],color='red', label='97.7%')

plt.grid()

plt.legend()

plt.title('Probability Density Function')

plt.xlabel('Score')



plt.show()



invitationProbability = round(scipy.stats.norm(scrMean, scrStd).cdf(candidateScore),3)

print('The probability of being invited in the next round is: {}%'.format(invitationProbability*100))
data['dayOfWeek'] = data['dayOfInvitation'].dt.weekday_name
fig=plt.figure(figsize=(12, 8), dpi= 80)

ax = data.groupby(by='dayOfWeek')['dayOfWeek'].count().plot(kind='bar')

ax.set_xlabel("Day of the Week")

ax.set_ylabel("Rounds")
data['dayOfInvitationShifted'] = data['dayOfInvitation'].shift(-1)

meanInterval=(data['dayOfInvitationShifted']-data['dayOfInvitation']).mean()



#Adding this to the last invitation date

nextinvitation = list(data['dayOfInvitation'])[-1]+meanInterval



#weekday = 0 is monday - 6 is sunday

dayofweek = nextinvitation.weekday()



#Let's take the closest wednesday (weekday=2)

nextinvitationday = nextinvitation.day + 2 - dayofweek



#Convert back to date

nextinvitation = datetime.date(nextinvitation.year, nextinvitation.month, nextinvitationday)



print('The next invitation will probably take place on {}.'.format(nextinvitation))
fig=plt.figure(figsize=(12, 8), dpi= 80)

plt.plot(data['dayOfInvitation'], data['numberOfInvited'] )

plt.xlabel('Date')

plt.ylabel('Number of People Invited')
fig=plt.figure(figsize=(12, 8), dpi= 80)

ax = data.groupby(by='numberOfInvited')['numberOfInvited'].count().plot(kind='bar')

ax.set_xlabel("Number of People Invited")

ax.set_ylabel("Rounds")