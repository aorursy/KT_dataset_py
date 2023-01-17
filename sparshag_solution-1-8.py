import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

from scipy import stats
# Generating 10000 normally-distributed random numbers with mean 5 and std deviation 2
norm_num = np.random.normal(loc=5, scale=2, size=10000)

#Plotting Density curve of the numbers generated above
ax = sns.distplot(norm_num, hist=False)
ax.set_title('Probability Density Curve of 10000 normally distributed no.')
ax.set_xlabel('x')
ax.set_ylabel('Probability Density')
#Given values
mean = 10
variance = 9
sd = np.sqrt(variance)
prob = 1-0.67 #P(greater than 67%) == P(less than 33%)

#Calculating random variable for the given probability
RV2 = stats.norm.ppf(prob, mean, sd) #Required random variable
print('The Value of Random Variable is ',np.round(RV2,2))

#Visualizing
ax = sns.distplot(np.random.normal(loc=mean, scale=sd, size=10000), hist=False)
ax.set_title('Probability Density Curve of 10000 normally distributed no.')
ax.set_xlabel('x')
ax.set_ylabel('Probability Density')

shade = np.linspace(RV2,20.5, 1000)
ax.fill_between(shade, stats.norm.pdf(shade, mean, sd), alpha=0.2, facecolor='red')
plt.show()
rateP = 1.5 #Unit - calls/min. ## Given that avg 3 calls per 2 minutes. 
print("Rate Parameter is", rateP, " which is 3 calls per 2 minute and the unit is calls/minute \n\n")


# What is the probability that at most three minutes will pass 
# before the first call is registered in the relay system. Identify the random variable. 
# Comment on the type of distribution which the random variable tends to follow
scale = 1/rateP
x = 3
RV3 = stats.expon.cdf(x=x, scale=scale)
print("Probability that at most 3 minutes will pass before first call is ",np.round(RV3*100,2), "% \n\n")
print("This is Exponential Distribution because we have to find the time until the first event occurs\n\n")

#Visualizing
x3 = np.linspace(0.0,4.0, 100)
plt.plot(x3, stats.expon.cdf(x=x3, scale=scale))

shade = np.linspace(0.0,x,100)
plt.fill_between(shade, stats.expon.cdf(x=shade, scale=scale), alpha=0.5, facecolor='red')

plt.title('Exponential CDF')
plt.xlabel('x')
plt.ylabel('Exponential Cumulative Probability')

plt.show()
# What is the probability that more than three calls will be registered 
# in the relay system in next 2-min. Identify the random variable. 
# Comment on the type of distribution which the random variable tends to follow. 

rateP = 1.5 #Unit - calls/min. ## Given that avg 3 calls per 2 minutes.
t = 2 #Unit time interval is 1 min, here we are converting it back to 2-minute unit time interval
rateP_new = rateP*t #New rate parameter with unit/2min

#P(More than 3 calls) = 1 - (P(3 calls or less))

prob3_max = stats.poisson.cdf(3, rateP_new) #Probability of receiving 3 or less calls
prob3_more = 1 - prob3_max
print("Probability of receiving more than 3 calls in next 2 mins is ", np.round(prob3_more*100,2), "%")
#given values
dfn = 4
dfd = 5

# Upper 10th Percentile == x-point r.v. at y-point of 90% on CDF F-distribution
upper_10_F = stats.f.ppf(0.9, dfn, dfd)
print('Upper 10th Percentile is ',np.round(upper_10_F,2))

#Mean of distribution
mean_F = stats.f.mean(dfn, dfd)
print('\nMean of Distribution is ',np.round(mean_F,2),'\n\n')

#Visualizing
x = np.linspace(0.0,15.0, 100)
plt.plot(x, stats.f.cdf(x, dfn, dfd))

plt.plot(upper_10_F, 0.9, 'o')
plt.text(upper_10_F+0.5, .85, 'upper 10th percentile point')
plt.title('F-distribution CDF')
plt.xlabel('x')
plt.ylabel('Cumulative Probability')
#given values
cum_prob = 0.75
sample_size = 25 
dof = sample_size-1 #Degree of Freedom is (sample_size-1)

# Chi Sq. critical value
value_chisq = stats.chi2.ppf(cum_prob, dof)
print('Chi-square Critical Value with Cum. Prob 75% and Sample size 25 is ', np.round(value_chisq,2))

#Visualizing
x = np.linspace(0.0,50.0, 100)
plt.plot(x, stats.chi2.cdf(x, dof))

plt.plot(value_chisq, 0.75, 'o')
plt.text(value_chisq+0.6, .72, 'Chi-square Value = 28.24')
plt.title('Chi. Sq. CDF')
plt.xlabel('Chi Sq. Critical Value')
plt.ylabel('Cumulative Probability')
plt.show()
#given values
num_trial = 7
prob_success = 2/6 #success only if 5 or 6 appears out of sample space [1,2,3,4,5,6]

#Distribution of Random Variable X, the no. of success out of no. of trials
#Sample space X = [0,1,2,3,4,5,6,7]
x = np.linspace(0, num_trial, num_trial+1)
plt.bar(x, stats.binom.pmf(x, num_trial, prob_success))[3].set_color('r')
plt.title('Binomial Probability Mass Function')
plt.xlabel('No. of Trial')
plt.ylabel('Probability')
plt.show()

#Probability that there are exactly 3 successes
prob_3success = stats.binom.pmf(3, num_trial, prob_success)
print('\n\nProbability of exactly 3 successes out of 7 trials is ',np.round(prob_3success*100,2),'%')
#given values
pop_mu = 32 #unit : hrs./week
sample_size = 25
sample_mu = 26 #unit : hrs./week
sample_sd = 6
# First, we have to compute t-statistic. 
t_score = (sample_mu - pop_mu) / ((sample_sd)/(np.sqrt(sample_size)))
print('Calculated t-score: ', t_score)


dof_t = sample_size - 1
# P(sample_mu >= 26) is equivalent to (1 - P(sample_mu <26))
prob_upto26 = stats.t.cdf(x=t_score, df=dof_t, loc=sample_mu, scale=sample_sd)
prob_more26 = 1-prob_upto26

print("Probability that people will spend at least 26 hours watching TV is ",np.round(prob_more26*100,4), "%")

#Visualizing
x = np.linspace(-20.0,100.0, 1000)
plt.plot(x, stats.t.pdf(x = x, df = dof_t, loc=26, scale=sample_sd))
plt.title('T-distribution PDF')
plt.xlabel('x')
plt.ylabel('Probability')
plt.show()
#Loading Excel file and saving into dataframe
df = pd.read_excel('../input/unloc.xlsx')
df_report = df[['area_name', 'Low-income countries', 'Middle-income countries', 'High-income countries']]
df_report.groupby(by='area_name').sum()
df['less or least developed country'] = 0
df.loc[(df['Less developed countries']==1) | (df['Least developed countries']==1), 'less or least developed country'] = 1
df.head()
print('Dataframe shape before removing null region_values records: ', df.shape)
df = df[pd.notnull(df['reg_name'])]
print('\nDataframe shape after removing null region_values records: ', df.shape)
df_visual1 = df_report.groupby(by='area_name').sum().plot(kind='barh')
plt.title('Region-wise Countries income-classification')
plt.xlabel('No. of countries')
plt.ylabel('Region name')
df_visual2 = df[['More developed countries', 'Less developed countries', 'Least developed countries']]
df_visual2 = df_visual2.sum()*100/len(df)
df_visual2.plot(kind='barh')

plt.title('Development Area wise % distribution of countries')
plt.xlabel('% countries')
plt.ylabel('Development Area')