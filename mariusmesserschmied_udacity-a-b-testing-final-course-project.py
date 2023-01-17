import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



import numpy as np 

import pandas as pd

from scipy import stats 

from statsmodels.stats.proportion import proportions_ztest

from statsmodels.stats.proportion import binom_test
#Storing baseline data

d = {"Metric Name": ["Cookies", "Clicks", "User-ids", "Click-through-probability", "Gross conversion", "Retention", "Net conversion"], 

     "Estimator": [40000, 3200, 660, 0.08, 0.20625, 0.53, 0.109313],

     "dmin": [3000, 240, -50, 0.01, -0.01, 0.01, 0.0075]}

md = pd.DataFrame(data=d, index=["C", "CL", "ID", "CTP", "CG", "R", "CN"])

md
#create new column to store scaled estimators

md.insert (2, "Scaled_Est", np.nan)



#scale count estimates

scaling_factor = 5000/md.loc["C"]["Estimator"]



for i in ["C", "CL", "ID"]:

    md.at[i, "Scaled_Est"] = md.loc[i]["Estimator"] * scaling_factor

md
def checkN (n, p, metric):

    '''Given sample size n and probability p, return whether n is large enough to pass the 3-standard deviation rule,

    i.e. whether we can assume that the distribution can be approximated by the normal distribution'''

    if n > 9*((1-p)/p) and n > 9*(p/(1-p)):

        result = print(metric,":  n =", n, "is large enough to assume normal distribution approximation")

    else:

        result = print(metric,":  n =", n, "is not large enough to assume normal distribution approximation")

    return result



#check whether n is large enough to assume normal distribution approximation

for i,j in zip(["CL", "ID", "CL"],["CG", "R", "CN"]):

    checkN (md.at[i, "Scaled_Est"], md.at[j,"Estimator"], md.at[j,"Metric Name"])
#create new column to store standard errors

md["SE"] = np.nan



#formula to calculate standard deviation

def standardError (n, p):

    '''Return the standard deviation for a given probability p and sample size n'''

    return (p*(1-p)/n)**0.5



#calculating standard errors for evaluation metrics and store them in md

for i in ["CG", "CN"]:

    md.at[i, "SE"] = standardError(md.loc["CL"]["Scaled_Est"], md.loc[i]["Estimator"]) 

    

md.at["R", "SE"] = standardError(md.loc["ID"]["Scaled_Est"], md.loc["R"]["Estimator"])

md

#storing alpha and beta in a dictionary

error_prob = {"alpha": 0.05, "beta": 0.20}

error_prob
#create new column n_c to store sample sizes

md["n_C"] = np.nan



#define function for approach B

def get_sampleSize (alpha, beta, p, dmin):

    '''Return sample size given alpha, beta, p and dmin'''

    return (pow((stats.norm.ppf(1-alpha/2)*(2*p*(1-p))**0.5+stats.norm.ppf(1-beta)*(p*(1-p)+(p+dmin)*(1-(p+dmin)))**0.5),2))/(pow(dmin,2))



#calculate sample sizes for evaluation metrics with defined adjustments and store results in md

for i in ["CG", "CN"]:

    md.at[i, "n_C"] = round((get_sampleSize(error_prob["alpha"], error_prob["beta"], md.loc[i]["Estimator"], md.loc[i]["dmin"])/md.loc["CTP"]["Estimator"])*2)



md.at["R", "n_C"] = round(((get_sampleSize(error_prob["alpha"], error_prob["beta"], md.loc["R"]["Estimator"], md.loc["R"]["dmin"])/md.loc["CTP"]["Estimator"])/md.loc["CG"]["Estimator"])*2)

md

#traffic diverted to experiment [0:1]

traffic_diverted = 1



#Days it would take to run experiment for each case

for i, j in zip(["CG", "CN", "R"],["CG", "CG+CN", "CG+CN+R"]):

   print("Days required for",j,":", round(md.loc[i]["n_C"]/(md.loc["C"]["Estimator"]*traffic_diverted),2))

#traffic diverted to experiment

traffic_diverted = 0.47



#Days it would take to run experiment if we use net conversion and gross coversion as evaluation metrics

print("Experiment duration in days, CN+CG: ",round(md.loc["CN"]["n_C"]/(md.loc["C"]["Estimator"]*traffic_diverted),2))
#loading experiment data into new dataframes

control = pd.read_csv("../input/Final Project Results - Control.csv") 

experiment = pd.read_csv("../input/Final Project Results - Experiment.csv")



#check if loaded correctly

control.head()
#check if loaded correctly

experiment.head()
#check number of entries

control.count()
#check number of entries

experiment.count()
#check sample size and store it as sample_size

sample_size_control = control["Pageviews"].sum()

sample_size_experiment = experiment["Pageviews"].sum()

sample_size = sample_size_control+sample_size_experiment

sample_size
#create empty dataframe to store sanity check results

sanity_check = pd.DataFrame(columns=["CI_left", "CI_right", "obs","passed?"], index=["C", "CL", "CTP"])



#set alpha and p_hat

p = 0.5

alpha = 0.05



#fill dataframe with results from binomial test

#for cookies and clicks do the following

for i,j in zip(["C", "CL"], ["Pageviews", "Clicks"]):

    #calculate the number of successes (n_control) and number of observations (n)

    n = control[j].sum()+experiment[j].sum()

    n_control = control[j].sum()

    

    #compute confidence interval

    sanity_check.at[i, "CI_left"] = p-(stats.norm.ppf(1-alpha/2)*standardError(n,p))

    sanity_check.at[i, "CI_right"] = p+(stats.norm.ppf(1-alpha/2)*standardError(n,p))

    

    #compute observed fraction of successes

    sanity_check.at[i, "obs"] = round(n_control/(n),4)

    

    #check if the observed fraction of successes lies within the 95% confidence interval

    if sanity_check.at[i, "CI_left"] <= sanity_check.at[i, "obs"] <= sanity_check.at[i, "CI_right"]:

        sanity_check.at[i, "passed?"] = "yes"

    else:

        sanity_check.at[i, "passed?"] = "no"



#return results

sanity_check
#calculate the number of observations

n = control["Pageviews"].sum()+experiment["Pageviews"].sum()

#calculate the number of successes

n_control = control["Pageviews"].sum()



#calculate the test-statistic Z and corresponding p_value

z_statistic, p_value = proportions_ztest(n_control, n, value=0.5, alternative="two-sided", prop_var=False)



print("z-test-statistic: ", z_statistic)

print("p-value:" , p_value)



#alternatively compute p-value using the exact binomial test

p_value_binom = binom_test(n_control, n, prop=0.5, alternative='two-sided')

print("p-value_binomial: ", p_value_binom)



#check whether p_value is smaller than alpha

alpha = 0.05



if p_value_binom > 0.05:

    print("The null hypothesis cannot be rejected and the sanity check is passed")

else:

    print("The null hypothesis is rejected and the sanity check is not passed")
#compute CTP for both groups

CTP_control = control["Clicks"].sum()/control["Pageviews"].sum()

CTP_experiment = experiment["Clicks"].sum()/experiment["Pageviews"].sum()



#compute sample standard deviations for both groups

S_control = (CTP_control*(1-CTP_control))**0.5

S_experiment = (CTP_experiment*(1-CTP_experiment))**0.5



#compute SE_pooled

SE_pooled = (S_control**2/control["Pageviews"].sum()+S_experiment**2/experiment["Pageviews"].sum())**0.5



#compute 95% confidence interval and store it in sanity check

alpha = 0.05



sanity_check.at["CTP", "CI_left"] = 0-(stats.norm.ppf(1-alpha/2)*SE_pooled)

sanity_check.at["CTP", "CI_right"] = 0+(stats.norm.ppf(1-alpha/2)*SE_pooled)



#compute observed difference d and store it in sanity check

sanity_check.at["CTP", "obs"] = round(CTP_experiment - CTP_control,4)



#check if sanity check is passed

if sanity_check.at["CTP", "CI_left"] <= sanity_check.at["CTP", "obs"] <= sanity_check.at["CTP", "CI_right"]:

    sanity_check.at["CTP", "passed?"] = "yes"

else:

    sanity_check.at["CTP", "passed?"] = "no"



#return results

sanity_check

#calculate the number of observations for each group and store results in an array

n = np.array([control["Pageviews"].sum(), experiment["Pageviews"].sum()])

#calculate the number of successes for each group and store results in an array

n_clicks = np.array([control["Clicks"].sum(), experiment["Clicks"].sum()])



#calculate the test-statistic Z and corresponding p_value

z_statistic, p_value = proportions_ztest(n_clicks, n, value=0, alternative="two-sided", prop_var=False)



print("z-test-statistic: ", z_statistic)

print("p-value:" , p_value)



#check whether p_value is smaller than alpha

alpha = 0.05



if p_value > 0.05:

    print("The null hypothesis cannot be rejected and the sanity check is passed")

else:

    print("The null hypothesis is rejected and the sanity check is not passed")
#compute true sample size

true_sample_size = control.iloc[:23]["Pageviews"].sum()+experiment.iloc[:23]["Pageviews"].sum()

true_sample_size
#create dataframe test_results

test_results = pd.DataFrame(columns=["CI_left", "CI_right", "d","stat sig?", "dmin", "pract rel?"], index=["CG", "CN"])



#set alpha

alpha = 0.05





#run two proportion z test for both metrics

for i,j in zip(["Enrollments", "Payments"],["CG", "CN"]):

    #compute sample conversion rates

    conv_control = control.iloc[:23][i].sum()/control.iloc[:23]["Clicks"].sum()

    conv_experiment = experiment.iloc[:23][i].sum()/experiment.iloc[:23]["Clicks"].sum()

    

    #compute observed difference between treatment and control conversion d

    test_results.at[j, "d"] = conv_experiment-conv_control

    

    #compute sample standard deviations

    S_control = (conv_control*(1-conv_control))**0.5

    S_experiment = (conv_experiment*(1-conv_experiment))**0.5

    

    #compute SE_pooled

    SE_pooled = (S_control**2/control.iloc[:23]["Clicks"].sum()+S_experiment**2/experiment.iloc[:23]["Clicks"].sum())**0.5

    

    #compute 95% confidence interval around observed difference d

    test_results.at[j, "CI_left"] = test_results.at[j, "d"]-(stats.norm.ppf(1-alpha/2)*SE_pooled)

    test_results.at[j, "CI_right"] = test_results.at[j, "d"]+(stats.norm.ppf(1-alpha/2)*SE_pooled)

    

    #check statistical significance

    if test_results.at[j, "CI_left"] <= 0 <= test_results.at[j, "CI_right"]:

        test_results.at[j, "stat sig?"] = "no"

    else:

        test_results.at[j, "stat sig?"] = "yes"

    

    #import dmin

    test_results.at[j, "dmin"] = md.loc[j]["dmin"]

    

    

    #check if practical relevant

    #check if dmin is positive or negative

    if test_results.at[j, "dmin"] >= 0:

        #check if d is larger than dmin and if dmin lies left of the confidence interval around d

        if test_results.at[j, "d"] > test_results.at[j, "dmin"] and test_results.at[j, "CI_left"] > test_results.at[j, "dmin"]:

                test_results.at[j, "pract rel?"] = "yes"

        else:

            test_results.at[j, "pract rel?"] = "no"

    else:

        #check if d is smaller than dmin and if dmin lies right of the confidence interval around d

        if test_results.at[j, "d"] < test_results.at[j, "dmin"] and test_results.at[j, "dmin"] > test_results.at[j, "CI_right"]:

                test_results.at[j, "pract rel?"] = "yes"

        else:

            test_results.at[j, "pract rel?"] = "no"



#return results

test_results