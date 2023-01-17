import math as mt

import numpy as np

import pandas as pd

from scipy.stats import norm
# Place estimators into dictionary 



baseline = {'cookies': 40000,

            'clicks': 3200,

            'enrollment': 660,

            'CTP': 0.08,

            'Gconversion': 0.20625,

            'retention': 0.53,

            'Nconversion': 0.109313}



baseline
# Scaling count estimates 



baseline['cookies'] = 5000



# cookies / daily number of unique cookies per day 

baseline['clicks'] = baseline['clicks']*(5000/40000)



baseline['enrollment'] = baseline['enrollment']*(5000/40000)



baseline
# Create p-hat & n 



Gc = {}



Gc['D_min'] = 0.01



# p-hat already given 

    # Alternatively, calc from enrollments / clicks 

Gc['p'] = baseline['Gconversion']



# Sample 

Gc['n'] = baseline['clicks']



# Compute SD, round to 4 decimals

Gc['sd'] = round(mt.sqrt((Gc['p']*(1-Gc['p']))/Gc['n']), 4)

Gc['sd']
R = {}

R['D_min'] = 0.01



R['p'] = baseline['retention']



# sample size = enrolled users 

R['n'] = baseline['enrollment']



R['sd'] = round(mt.sqrt((R['p']*(1-R['p']))/R['n']), 4)

R['sd']
Nc = {}

Nc['D_min'] = 0.075



Nc['p'] = baseline['Nconversion']



# sample size = number of cookies clicked 

Nc['n'] = baseline['clicks']



Nc['sd'] = round(mt.sqrt((Nc['p']*(1-Nc['p']))/Nc['n']), 4)

Nc['sd']
def get_sd(p,d):

    sd1 = mt.sqrt(2 * p *(1 - p))

    sd2 = mt.sqrt(p * (1 - p) + (p + d) * (1-(p + d)

                                          )

                 )

    x = [sd1,sd2]

    return x
# alpha should already fit required test

    #Return z-score for alpha 

    

def get_z_score(alpha):

    return norm.ppf(alpha)
def get_sd(p,d):

    sd1 = mt.sqrt(2 * p * (1 - p))

    sd2 = mt.sqrt(p * (1 - p)+(p + d) * (1 - (p + d)

                                        )

                 )

    sd = [sd1,sd2]

    return sd

def get_sampSize(sd,alpha,beta,d):

    n = pow((get_z_score(1 - alpha / 2) * sd[0] + get_z_score(1 - beta) * sd[1]),

            2) / pow(d, 2)

    return n
# Add detectable effect parameter to each metric characteristics for all eval metrics 



Gc['d'] = 0.01

R['d'] = 0.01

Nc['d'] = 0.0075
Gc["SamSiz"]=round(get_sampSize(get_sd(Gc["p"], 

                                          Gc["d"]),

                                  0.05,0.2,

                                  Gc["d"])

                  )



Gc["SamSiz"]
Gc['SamSiz'] = round(Gc['SamSiz'] / 0.16*2)



Gc['SamSiz']
R['SamSiz'] = round(get_sampSize(get_sd(R['p'],

                                        R['d']),

                                        0.05, 0.2,

                                       R['d']

                                )

                   )

                



R['SamSiz']
# 0.08 = CTF

# 0.20625 = GConversion

    # 0.20625 = base conversion rate (probability of enrolling, given click)

    

R['SamSiz'] = R['SamSiz'] / 0.08 / 0.20625 * 2



R['SamSiz']
Nc['SamSiz'] = round(get_sampSize(get_sd(Nc['p'], 

                                         Nc['d']), 

                                  0.05, 0.2, 

                                  Nc['d']

                                 )

                    )

Nc['SamSiz']
Nc['SamSiz'] = Nc['SamSiz']/0.08*2



Nc['SamSiz']
cont = pd.read_csv('../input/control-data/control_data.csv')



exp = pd.read_csv('../input/experiment-data/experiment_data.csv')
cont.head()
pageviews_cont = cont['Pageviews'].sum()

pageviews_exp = exp['Pageviews'].sum()



pageviews_total = pageviews_cont + pageviews_exp



print('number of control pageviews: ', pageviews_cont)



print('number of experiment pageviews: ', pageviews_exp)
p = 0.5



alpha = 0.05



# Proportion of total 

p_hat = round(pageviews_cont / (pageviews_total), 4)



sd = mt.sqrt(p * (1 - p) / (pageviews_total))



ME = round(get_z_score(1 - (alpha / 2)) * sd, 4)



print('Confidence interval is between', p - ME, 'and', p + ME, '; Verify', p_hat, 'is within this range')
clicks_cont = cont['Clicks'].sum()

clicks_exp = exp['Clicks'].sum()

clicks_total = clicks_cont + clicks_exp



p_hat = round(clicks_cont / clicks_total, 4)

sd = mt.sqrt(p * (1 - p) / clicks_total)



ME = round(get_z_score(1 - (alpha / 2)) * sd, 4)



print('Confidence interval is between', p - ME, 'and', p + ME, '; Verify', p_hat, 'is within this range')
ctp_cont = clicks_cont / pageviews_cont

ctp_exp = clicks_exp / pageviews_exp



# Detecable change

d_hat = round(ctp_exp - ctp_cont, 4)



p_pooled = clicks_total / pageviews_total



sd_pooled = mt.sqrt(p_pooled * (1 - p_pooled) * (1 / pageviews_cont + 1 / pageviews_exp))



ME = round(get_z_score(1-(alpha / 2)) * sd_pooled, 4)



print('Confidence interval is between', 0-ME, 'and', 0+ME, '; Verify', d_hat, 'is within this range')
# Count total clicks from complete records 

    # i.e. where rows with pageviews and clicks have corresponding values with enrollments and payments

    

clicks_cont = cont['Clicks'].loc[cont['Enrollments'].notnull()].sum()



clicks_exp = exp['Clicks'].loc[exp['Enrollments'].notnull()].sum()



print(clicks_cont, clicks_exp)
# Gross conversion = P(enrollment | click)



enrol_cont = cont['Enrollments'].sum()

enrol_exp = exp['Enrollments'].sum()



GC_cont = enrol_cont / clicks_cont 

GC_exp = enrol_exp / clicks_exp



total_enrol = enrol_cont + enrol_exp # xcont + xexp

total_clicks = clicks_cont + clicks_exp



# p_hat pool 

GC_pooled = total_enrol / total_clicks 



GC_sd_pooled = mt.sqrt(GC_pooled * (1 - GC_pooled) * (1 / clicks_cont + 1 / clicks_exp))



GC_ME = round(get_z_score(1 - alpha / 2) * GC_sd_pooled, 4)



GC_diff = round(GC_exp - GC_cont, 4)



print('Change due to experiment is', GC_diff * 100, '%')



print('Confidence interval: [', GC_diff - GC_ME, ',', GC_diff + GC_ME, ']')



print('Statistically signfiicant if interval does not include 0')



print('Practically significant if interval does not contain', -Gc['D_min'])
# Net conversion = number of paying users after 14-day boundary / number of clicks on Start Free Trial 

    # P(payment | clicks)

    

payments_cont = cont['Payments'].sum()

payments_exp = exp['Payments'].sum()



NC_cont = payments_cont / clicks_cont 

NC_exp = payments_exp / clicks_exp 



total_payments = NC_cont + NC_exp 



NC_pooled = total_payments / total_clicks 



NC_sd_pooled =  mt.sqrt(NC_pooled * (1 - NC_pooled) * (1 / clicks_cont + 1 / clicks_exp))



NC_ME = round(get_z_score(1 - alpha / 2) * GC_sd_pooled, 4)



NC_diff = round(NC_exp - NC_cont, 4)



print('Change due to experiment is', NC_diff * 100, '%')

print('Confidence interval: [', NC_diff - NC_ME, ',', NC_diff + NC_ME, ']')

print('Statistically significant if interval does not include 0.')

print('Practically significant if interval does not include', Nc['D_min'])
# Merge datasets 



merge = cont.join(other = exp,

                  how = 'inner',

                      lsuffix = '_cont',

                      rsuffix = '_exp')



print(merge.count())

merge.head()
# Drop incomplete rows (on any col with 23 observations)



merge = merge.loc[merge['Payments_exp'].notnull()]

merge.count()
# Create binary daily col for each metric 



    # Return 0 if control > experiment 

        # Return 1 if experiment > control 



# GC

x = merge['Enrollments_cont'] / merge['Clicks_cont']



y = merge['Enrollments_exp'] / merge['Clicks_exp']



merge['GC'] = np.where(x > y, 0, 1)



# NC

a = merge['Payments_cont'] / merge['Clicks_cont']



b = merge['Payments_exp'] / merge['Clicks_exp']



merge['NC'] = np.where(a > b, 0, 1)



merge.head()
# Experiment > control

GC_x = merge.GC[merge['GC'] == 1].count()

NC_x = merge.NC[merge['NC'] == 1].count()



GC_y = merge.GC[merge['GC'] == 0].count()

NC_y = merge.NC[merge['NC'] == 0].count()



n = merge.NC.count()



print('Number of cases for GC_x:', GC_x, '\n',

      'Number of cases for NC_x:', NC_x, '\n',

      'Number of total cases:', n)



print('Number of cases for GC_y:', GC_y, '\n',

      'Number of cases for NC_y:', NC_y, '\n')
# Function for calculating probability of x i.e. number of successes 



def get_proba(x, n):

    p = round(mt.factorial(n) / (mt.factorial(x) * mt.factorial(n - x)) * 0.5 ** x * 0.5 ** (n - x), 4)

    return p 



# Function to compute p-val from probabilities of maximum x 



def get_2tail_pval(x, n):

    p = 0

    for i in range(0, x + 1):

        p = p + get_proba(i, n)

    return 2 * p 
# Significance of obsering extreme values 



print('GC change significant if', get_2tail_pval(GC_x, n), 'is lower than 0.05')

print('NC change significant if', get_2tail_pval(NC_x, n), 'is lower than 0.05')