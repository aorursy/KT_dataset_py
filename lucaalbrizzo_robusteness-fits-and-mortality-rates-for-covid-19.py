!pip install lmfit #installing lmfit needed for fit analysis
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from lmfit.models import ExponentialModel
from lmfit.models import StepModel

def fit_and_obtain_parameters(x,y,model):
    """
    Just a wrapper of lmfit fit to then obtain the best fit values. 
    """
    
    params = model.guess(y, x=x)
    result = model.fit(y, params, x=x)
    
    values = np.array([])
    std_err = np.array([])
    
    for p in result.params:
        values = np.append(values,result.params[p].value)
        std_err = np.append(std_err,result.params[p].stderr)
        
        
    return values, std_err, result.redchi


def return_parameters_over_time(x,y,model,position_param=0,min_days=16):
    """
    A function which performs a fit per day (after min_days days)
    and returns the value and the standard error of the
    parameter in position position_param. 
    """
    
    values_param_per_day =  np.array([])
    stderr_param_per_day =  np.array([])
    rchi_per_day = np.array([])
    
    for j in range(min_days,len(y)):
        
        values, stds, red_chi = fit_and_obtain_parameters(x.head(j),
                                                 y.head(j),
                                                 model)
        
        values_param_per_day = np.append(values_param_per_day,values[position_param])
        stderr_param_per_day = np.append(stderr_param_per_day,stds[position_param])
        rchi_per_day = np.append(rchi_per_day, red_chi)
        
    
    return values_param_per_day, stderr_param_per_day, rchi_per_day

import scipy



def rate_estimate(x,n_bay_0=0,n_bay_1=0):
    """
    Estimate of the mortality rate using beta function
    """
    rate = scipy.stats.beta.ppf(0.5 , 
                                x[0]+n_bay_0+1, 
                               (x[1]+n_bay_1)-(x[0]+n_bay_0)+1)
    return rate

def confidence_beta_distribution(x, alpha=0.90, n_bay_0=0,n_bay_1=0):
    """
    Estimate of the confidence intervals of the mortality rate using beta function
    """
    lower = scipy.stats.beta.ppf((1.0 - alpha)/2.0 ,
                                x[0]+n_bay_0+1, 
                               (x[1]+n_bay_1)-(x[0]+n_bay_0)+1)
    upper = scipy.stats.beta.ppf((1.0 + alpha)/2.0 ,
                                x[0]+n_bay_0+1, 
                               (x[1]+n_bay_1)-(x[0]+n_bay_0)+1)
    return  upper - lower 


def create_df_country(df_world, country_name,n_bay_0=0,n_bay_1=0):
    """
    Create dataframe for country and order by days after first case. 
    """
    df_country = df_world[df_world["countriesandterritories"] == country_name]
    df_country["date_time_date"] = pd.to_datetime(df_country[["year", "month","day"]])
    
    df_country = df_country.sort_values("date_time_date",ascending=True)    
    df_country["Total_cumulative_Cases"] = df_country["cases"].cumsum()
    df_country["Total_cumulative_Deaths"] = df_country["deaths"].cumsum()
    df_country["mortality_rate"] = df_country[["Total_cumulative_Deaths", "Total_cumulative_Cases"]].apply(rate_estimate, axis=1)
    df_country["mortality_error"] = df_country[["Total_cumulative_Deaths", "Total_cumulative_Cases"]].apply(confidence_beta_distribution, axis=1)
    df_country["mortality_relative_error"] = df_country["mortality_error"] / df_country["mortality_rate"]
    df_country["mortality_rate_bayes"] = df_country[["Total_cumulative_Deaths", "Total_cumulative_Cases"]].apply(rate_estimate,
                                                                                                       axis=1,
                                                                                                       n_bay_0=n_bay_0,
                                                                                                       n_bay_1=n_bay_1)
    df_country["mortality_error_bayes"] = df_country[["Total_cumulative_Deaths", "Total_cumulative_Cases"]].apply(confidence_beta_distribution,
                                                                                                        axis=1,
                                                                                                        n_bay_0=n_bay_0,
                                                                                                        n_bay_1=n_bay_1)
    
    df_country = df_country[df_country["Total_cumulative_Cases"]>0]
    first_day = df_country["date_time_date"].iloc[0]
    df_country["number_days"] =  [abs((day - first_day).days) for day in df_country["date_time_date"]]
    
    return df_country
plt.rcParams['figure.figsize'] = [10, 8] #larger plots
df_world = df = pd.read_csv("/kaggle/input/uncover/ECDC/current-data-on-the-geographic-distribution-of-covid-19-cases-worldwide.csv")
df_world.head(5)
top20_countries = df_world.groupby("countriesandterritories").sum().sort_values("cases", ascending=False).reset_index().head(20)["countriesandterritories"].tolist()

n_world_top20_cases = df_world[df_world["countriesandterritories"].isin(top20_countries)]["cases"].sum()
n_world_top20_death = df_world[df_world["countriesandterritories"].isin(top20_countries)]["deaths"].sum()

dict_of_df = {}

for country in top20_countries:
    
    df = create_df_country(df_world, country, n_bay_0=n_world_top20_death,n_bay_1=n_world_top20_cases)  

    dict_of_df[country] = df   
df_world_italy = dict_of_df["Italy"] #defining dataset for Italy and China for convi
df_world_china = dict_of_df["China"] 
model_exp = ExponentialModel()
params_exp = model_exp.guess(df_world_italy["Total_cumulative_Cases"], x=df_world_italy["number_days"])
result_exp = model_exp.fit(df_world_italy["Total_cumulative_Cases"], params_exp, x=df_world_italy["number_days"])

model_log = StepModel(form='logistic')
params_log = model_log.guess(df_world_italy["Total_cumulative_Cases"], x=df_world_italy["number_days"])
result_log = model_log.fit(df_world_italy["Total_cumulative_Cases"], params_log, x=df_world_italy["number_days"])


tanti_giorni = np.array([i for i in range(1,len(df_world_italy)+20)])

plt.plot(df_world_italy["number_days"],
         result_log.best_fit,
         label="Logistic $\chi^2 = {:.2E}$".format(result_log.redchi))

plt.plot(df_world_italy["number_days"],
         result_exp.best_fit,
         label="ExponentialModel $\chi^2 = {:.2E}$".format(result_exp.redchi))

plt.scatter(df_world_italy["number_days"],
            df_world_italy["Total_cumulative_Cases"], 
            marker = 'o',color='black',
           label ="Data")

dely_log = result_log.eval_uncertainty(x=df_world_italy["number_days"],sigma=3)

plt.fill_between(df_world_italy["number_days"], 
                 result_log.best_fit-dely_log,
                 result_log.best_fit+dely_log, 
                 color='b',
                 alpha=0.5)

dely_exp = result_exp.eval_uncertainty(x=df_world_italy["number_days"],sigma=3)

plt.fill_between(df_world_italy["number_days"], 
                 result_exp.best_fit-dely_exp,
                 result_exp.best_fit+dely_exp, 
                 color='orange',
                 alpha=0.5)

params_exp = model_exp.make_params(decay = result_exp.params["decay"].value, 
                            amplitude = result_exp.params["amplitude"].value)

plt.plot(tanti_giorni, result_exp.eval(params_exp, x=tanti_giorni),color='orange')


params_log = model_log.make_params(sigma = result_log.params["sigma"].value, 
                                   amplitude = result_log.params["amplitude"].value,
                                   center = result_log.params["center"].value) 


#params_log = model_log.make_params(result_log.params) 

plt.plot(tanti_giorni, result_log.eval(params_log, x=tanti_giorni),color='b')

plt.ylim(100,result_log.params["amplitude"].value*1.1)
plt.xlim(1,tanti_giorni[-1])

plt.title("Total number of observed cases for Italy")
plt.xlabel("Days after the fist observed case")
plt.legend()
model_log = StepModel(form='logistic')
params_log = model_log.guess(df_world_italy["Total_cumulative_Cases"], x=df_world_italy["number_days"])

result_log = model_log.fit(df_world_italy["Total_cumulative_Cases"], params_log, x=df_world_italy["number_days"])

n_less = 7 

n_days = len(df_world_italy) - n_less 

params_log_5days_ago = model_log.guess(df_world_italy["Total_cumulative_Cases"].head(n_days), 
                                       x=df_world_italy["number_days"].head(n_days))


result_log_5days_ago = model_log.fit(df_world_italy["Total_cumulative_Cases"].head(n_days), 
                                     params_log, 
                                     x=df_world_italy["number_days"].head(n_days))

plt.plot(df_world_italy["number_days"],
         result_log.best_fit,
         label="Logistic total $\chi^2 = {:.2E}$".format(result_log.redchi))

plt.plot(df_world_italy["number_days"].head(n_days),
         result_log_5days_ago.best_fit,
         label="Logistic {} days less $\chi^2 = {:.2E}$".format(n_less, result_log_5days_ago.redchi))

plt.scatter(df_world_italy["number_days"],
            df_world_italy["Total_cumulative_Cases"], 
            marker = 'o',color='black',
           label = 'Data')

dely_log = result_log.eval_uncertainty(x=df_world_italy["number_days"],sigma=3)

plt.fill_between(df_world_italy["number_days"], 
                 result_log.best_fit-dely_log,
                 result_log.best_fit+dely_log, 
                 color='b',
                 alpha=0.5)


dely_log_5days_ago = result_log_5days_ago.eval_uncertainty(x=df_world_italy["number_days"].head(n_days),sigma=3)

plt.fill_between(df_world_italy["number_days"].head(n_days), 
                 result_log_5days_ago.best_fit-dely_log_5days_ago,
                 result_log_5days_ago.best_fit+dely_log_5days_ago, 
                 color='orange',
                 alpha=0.5)


params_log = model_log.make_params(sigma = result_log.params["sigma"].value, 
                                   amplitude = result_log.params["amplitude"].value,
                                   center = result_log.params["center"].value) 

plt.plot(tanti_giorni, result_log.eval(params_log, x=tanti_giorni),color='b')


params_log_5days_ago = model_log.make_params(sigma = result_log_5days_ago.params["sigma"].value, 
                                   amplitude = result_log_5days_ago.params["amplitude"].value,
                                   center = result_log_5days_ago.params["center"].value) 

plt.plot(tanti_giorni, result_log_5days_ago.eval(params_log_5days_ago, x=tanti_giorni),color='orange')

plt.ylim(0,120000)
plt.xlim(10,70)
plt.legend()

plt.title("Total number of observed cases for Italy")
plt.xlabel("Days after the fist observed case")

model_log = StepModel(form='logistic')
params_log = model_log.guess(df_world_china["Total_cumulative_Cases"], x=df_world_china["number_days"])

result_log = model_log.fit(df_world_china["Total_cumulative_Cases"], params_log, x=df_world_china["number_days"])

n_days = len(df_world_china) - 20

params_log_5days_ago = model_log.guess(df_world_china["Total_cumulative_Cases"].head(n_days), 
                                       x=df_world_china["number_days"].head(n_days))


result_log_5days_ago = model_log.fit(df_world_china["Total_cumulative_Cases"].head(n_days), 
                                     params_log, 
                                     x=df_world_china["number_days"].head(n_days))

plt.plot(df_world_china["number_days"],
         result_log.best_fit,
         label="Logistic total $\chi^2 = {:.2E}$".format(result_log.redchi))

plt.plot(df_world_china["number_days"].head(n_days),
         result_log_5days_ago.best_fit,
         label="Logistic twenty days less $\chi^2 = {:.2E}$".format(result_log_5days_ago.redchi))


plt.scatter(df_world_china["number_days"],
            df_world_china["Total_cumulative_Cases"], 
            marker = 'o',color='black')


dely_log = result_log.eval_uncertainty(x=df_world_china["number_days"],sigma=3)

plt.fill_between(df_world_china["number_days"], 
                 result_log.best_fit-dely_log,
                 result_log.best_fit+dely_log, 
                 color='b',
                 alpha=0.5)


dely_log_5days_ago = result_log_5days_ago.eval_uncertainty(x=df_world_china["number_days"].head(n_days),sigma=3)

plt.fill_between(df_world_china["number_days"].head(n_days), 
                 result_log_5days_ago.best_fit-dely_log_5days_ago,
                 result_log_5days_ago.best_fit+dely_log_5days_ago, 
                 color='orange',
                 alpha=0.5)


params_log = model_log.make_params(sigma = result_log.params["sigma"].value, 
                                   amplitude = result_log.params["amplitude"].value,
                                   center = result_log.params["center"].value) 

plt.plot(tanti_giorni, result_log.eval(params_log, x=tanti_giorni),color='b')


params_log_5days_ago = model_log.make_params(sigma = result_log_5days_ago.params["sigma"].value, 
                                   amplitude = result_log_5days_ago.params["amplitude"].value,
                                   center = result_log_5days_ago.params["center"].value) 

plt.plot(tanti_giorni, result_log_5days_ago.eval(params_log_5days_ago, x=tanti_giorni),color='orange')

plt.ylim(0,90000)
plt.xlim(0,95)
plt.title("Total number of observed cases for China")
plt.xlabel("Days after the fist observed case")
plt.legend()
countries = ['United_States_of_America', 'Italy', 
             'Spain', 'China', 
             'Germany', 'South_Korea',
             'Iran','France']

cmap = plt.cm.get_cmap("jet", len(countries)+1)
i=0 

model_log = StepModel(form='logistic')

params_log = model_log.guess(df_world_china["Total_cumulative_Cases"], 
                             x=df_world_china["number_days"]) #using China as a starting point for minization of parameters



for country in countries:
    
    i+= 1 
    
    df = dict_of_df[country]
    values_name = country + '_values_amplitude_per_day'
    stderr_name = country + '_stderr_amplitude_per_day'
    red_chi_name = country + '_red_chi_per_day'
    
    values_amplitude_per_day, stderr_amplitude_per_day, red_chi_log = return_parameters_over_time(
                                                df["number_days"],
                                                df["Total_cumulative_Cases"],
                                                model_log,
                                                position_param=0,
                                                min_days=30)
    
    dict_of_df[values_name] = values_amplitude_per_day
    dict_of_df[stderr_name] = stderr_amplitude_per_day
    dict_of_df[red_chi_name] = red_chi_log
       
    relative_error = stderr_amplitude_per_day / values_amplitude_per_day
    
    filter_error = relative_error < 1.0 #excludying values that have error that are "too high"
    
    plt.errorbar(df["number_days"][30:len(df)+1][filter_error], 
             values_amplitude_per_day[filter_error] / values_amplitude_per_day[-1],
             linestyle = '--',
             marker = 'o',
             color = cmap(i),
             yerr = stderr_amplitude_per_day[filter_error] / values_amplitude_per_day[-1] ,
             ecolor = cmap(i),
             label=country)
    
    
plt.ylim(0.4,1.3)
plt.xlim(35,80)
plt.vlines(74,-10,30, linestyles='--') #"end of quarantine" in China
plt.legend(loc="lower right")
plt.xlabel("Days after first case")
plt.ylabel("$A(t) / A(t_O)$")
countries = ['United_States_of_America', 'Italy', 
             'Spain', 'China', 
             'Germany', 'South_Korea',
             'Iran','France']

cmap = plt.cm.get_cmap("jet", len(countries)+1)
i=0 

for country in countries:
    
    i+=1
    
    df = dict_of_df[country]
    df = df[df["mortality_relative_error"]<1.5]
    
    plt.errorbar(df["number_days"],
                 df["mortality_rate"]*100,
                 linestyle = '--',
                 marker = 'o',
                 color = cmap(i),
                 yerr = df["mortality_error"]*100,
                 ecolor = cmap(i),
                 label = country)
    
plt.legend()
plt.ylim(-0.01,14.2)
plt.xlim(10,80)
plt.xlabel("Number of days after first case in the country")
plt.title("Daily mortality rate estimate")

#plt.title("$\hat{t}^i_{A}$ per day")
cmap = plt.cm.get_cmap("hsv", len(top20_countries)+1)


from random import shuffle

shuffle(top20_countries) #shuffling for better reading the colors on plots

x = np.arange(0,100) # just a straight line 

i=0
for country in top20_countries:
    
    i += 1
     
    plt.errorbar(100*dict_of_df[country].iloc[-1]["mortality_rate"], 
                 100*dict_of_df[country].iloc[-1]["mortality_rate_bayes"],
             linestyle = ' ',
             marker = ' ',
             xerr = 100*dict_of_df[country].iloc[-1]["mortality_error"],
             yerr = 100*dict_of_df[country].iloc[-1]["mortality_error_bayes"],
             label = country,
             color = cmap(i) )
    
    plt.text(100*dict_of_df[country].iloc[-1]["mortality_rate"], 
            100*dict_of_df[country].iloc[-1]["mortality_rate_bayes"],
            s=country,
            fontsize=8,
            bbox=dict(facecolor=cmap(i), alpha=0.5),
            rotation=0)
    
 
plt.xlabel('$\hat{t}^i_{A}$')
plt.ylabel('$\hat{t}^i_{T}$')
plt.fill_between(x,x,np.zeros(len(x)),alpha = 0.1)
plt.fill_between(x,x,10*np.ones(len(x)),alpha = 0.1)

plt.xlim(0.0,12.5)
plt.ylim(4.0,6.6)
#plt.legend(bbox_to_anchor=(1.0, 1.0)) 
i=0
for country in top20_countries:
    
    i += 1
     
    plt.errorbar(100*dict_of_df[country].iloc[-15]["mortality_rate"], 
                 100*dict_of_df[country].iloc[-15]["mortality_rate_bayes"],
             linestyle = ' ',
             marker = ' ',
             xerr = 100*dict_of_df[country].iloc[-15]["mortality_error"],
             yerr = 100*dict_of_df[country].iloc[-15]["mortality_error_bayes"],
             label = country,
             color = cmap(i) )
    
    plt.text(100*dict_of_df[country].iloc[-15]["mortality_rate"], 
            100*dict_of_df[country].iloc[-15]["mortality_rate_bayes"],
            s=country,
            fontsize=8,
            bbox=dict(facecolor=cmap(i), alpha=0.5),
            rotation=0)
    
 

plt.xlabel('$\hat{t}^i_{A}$')
plt.ylabel('$\hat{t}^i_{T}$')
plt.fill_between(x,x,np.zeros(len(x)),alpha = 0.1)
plt.fill_between(x,x,10*np.ones(len(x)),alpha = 0.1)

plt.xlim(0.0,8.5)
plt.ylim(4.75,5.3)
#plt.legend(bbox_to_anchor=(1.0, 1.0)) 
