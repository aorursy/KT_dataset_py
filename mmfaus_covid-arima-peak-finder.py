#this cell shows imports and calculation 

import numpy as np

import scipy as sp



import matplotlib

matplotlib.rcParams['figure.figsize'] = (12,6)

matplotlib.rcParams['figure.dpi'] = 100

matplotlib.rcParams['axes.titlesize'] = 20

matplotlib.rcParams['axes.labelsize'] = 18

matplotlib.rcParams['font.family'] = 'serif'

matplotlib.rcParams['mathtext.fontset'] = 'cm'

matplotlib.rcParams['mathtext.rm'] = 'serif'



from matplotlib.ticker import MultipleLocator

import matplotlib.pyplot as plt



from datetime import datetime

import sys

import os



import pandas as pd



from statsmodels.tsa.arima_model import ARMA



#helper function to translate date strings to day of year integers, easier for fitting 

def str_date_to_doy(date_str):

        y,m,d = date_str.split('-')

        doy = int(datetime(int(y),

                            int(m),

                            int(d)).strftime('%j'))

        return int(y),int(m),int(d),doy



    

#helper function to calculate the rolling variance of w in some window length

def get_sliding_var(n,w_length):

    w = np.ones(w_length)

    w /= w.sum()

    return np.convolve( (n - n.mean())**2 ,w,'same')

    

#load data, reformat dates

df = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/train.csv')

year, month, day, day_of_year = [],[],[],[]

for t in df['Date'].values:

    y,m,d,doy = str_date_to_doy(t)

    year.append(y)

    month.append(m)

    day.append(d)

    day_of_year.append(doy)



#extract the columns that I care the most about

day_of_year = np.array(day_of_year)

confirmed_cases = df['ConfirmedCases'].values

N_cases = []

for doy in np.unique(day_of_year):

    m = day_of_year == doy

    N_cases.append(np.sum( confirmed_cases[m] ))

doy,u_idx = np.unique(day_of_year,return_index=True)

month = np.array(month)[u_idx]

day = np.array(day)[u_idx]

N_cases = np.array(N_cases)



#rename, compute 1st/2nd difference, and compute rolling variance

t = doy

n = N_cases

diff_n  = np.diff(n)

diff2_n = np.diff(np.diff(n))

w  = np.ones(5)

w /= np.sum(w)

var = get_sliding_var( diff2_n, 5)



#this cell makes plots of these calculations for inspection

F,(ax1,ax2,ax3,ax4) = plt.subplots(4,1,sharex='col')



ax1.plot(t,n,'k.-')



ax2.plot(t[1::],diff_n,'r.-',label='1st difference')

ax2.plot(t[2::],diff2_n,'b.-',label='2nd difference')

xl,xh = ax2.get_xlim()

ax2.plot([xl,xh],[0,0],'k--')



ax3.plot(t[2::], np.sqrt(var),'m.-')



ax4.plot(t[2::], diff2_n/np.sqrt(var),'b.-')



#all the formatting follows

ax1.set_yscale('log')

xl,xh = ax3.get_xlim()

ax4.plot([xl,xh],[0,0],'k--')

ax4.set_xticks(t[::7])

ax4.set_xticklabels(['{:1.0f}/{:02.0f}'.format(z[0],z[1]) for z in zip(month[::7],day[::7])])

ax4.set_xlim([xl,xh])



ax1.set_title('Global COVID-19 cases (training)')

ax4.set_xlabel('Date')

plt.figtext(0.04, 0.75,'Reported\nCases',fontsize=16,rotation=90)

plt.figtext(0.04, 0.53,'Derivatives',fontsize=16,rotation=90)

plt.figtext(0.04, 0.38, 'Rolling\nSTD',fontsize=16,rotation=90)

plt.figtext(0.04, 0.18, '2nd_diff/\nroll_std',fontsize=16,rotation=90)



ax2.legend()



F.subplots_adjust(hspace=0)

F.set_size_inches(12,8)

def fit_second_derivative_and_predict(t,n,t_extrap, p, q, check_plots=False):

    """

    for timeseries data 'n' at timestamps t, transform the data 

    to a stationary state (second derivative normalized by rolling standard deviation),

    and fit an ARMA(p,q) model.  Reconstruct prediction of 1st derivative and original function.  

    Includes the ability to forecase some number of epochs into the future.

    

    Parameters

    ----------

    t : ndarray

        timestamp for data, should be a sequence of integers

    n : ndarray

        counts of data to fit

    t_extrap : int

        how many epochs beyond the end of the time series to forecast

    p : int

        autoreggesive order of the model (timscale of "memory")

    q : int

        moving average order of the model (response to random impulses/shocks)

    check_plots : bool

        flag to plot the data being fitted.  Useful to check if model fails

    

    

    Returns

    -------

    predb : ndarray

        model predictions for original data, has size len(n) + t_extrap

    preda :

        model predictions of 1st difference, has size len(n) + t_extrap - 1

    pred : ndarray

        model predictions of 2nd differences, has size len(n) + t_extrap - 2

    epredb : ndarray

        1sigma estimate of uncertainty on predb, only produced for forecasted (future) epochs

    res : ndarray

        residual (data - model)

    tout : ndarray

        timestamps of forecast

    

    """

    

    #transform to stationary state--take 2nd difference,

    #remove the mean, and divide by rolling standard deviation

    diff_n  = np.diff(n)

    diff2_n = np.diff(np.diff(n))

    var = get_sliding_var(diff2_n , 5)

    std_norm = (diff2_n - diff2_n.mean())/np.sqrt(var)

    



    if check_plots:

        print('n data to fit',len(std_norm))

        F,(ax1,ax2) = plt.subplots(2,1)

        ax1.plot(t[2::],diff2_n,'k.-')

        ax2.plot(t[2::],std_norm,'r.-')

 



    tout  = np.r_[t[-1]+1: t[-1] + t_extrap + 1] 

    #print(t,tout)



    #statsmodels interface:  define model, fit, 

    #get the predictions, get the forecasts and errors

    model = ARMA(std_norm, (p,q), t[2:])

    m_fit = model.fit(#solver='powell',

                      disp=0,

                      #tol=1.e-10

                       )

    #pred will have length len(std_dev) + t_extrap

    #std_norm is missing two points from differencing,

    #and the index starts at zero.  so subtract 3 from

    #len(t) to get index of last point

    pred = m_fit.predict(start = 0,

                        end=len(t) - 3 + t_extrap,

                         exog=tout,

    )

    fc,stderr,conf_int = m_fit.forecast(steps = t_extrap, exog = tout)

    

    #To reconstruct 1st diff and data, 

    #add back the mean, rescale for the non-stationary variance

    #predicted data needs some estimate of the variance.  

    #Use the most recent value from rolling variance; but this estimate

    #has edge effects.  For a window of length 5, the most recent uncorrupted value 

    #is 3 timesteps in the past.

    #print(sp.sqrt(var))

    e_data = np.sqrt(var[-3])

    pred = (pred )*np.r_[np.sqrt(var), [e_data]*t_extrap]

    epreda = stderr*np.sqrt(var[-3])



    #reconstruct first derivative, and propagate uncertainty.  

    #I believe this matches the statsmodels algorithm,

    #where data are used to ancor each successive estimate.  

    #If we integrated the derivatives (cumsum), small errors at

    #the begining compound and the model does very poorly

    preda = np.r_[diff_n[0], diff_n[0:-1] + pred[0:-t_extrap]]

    for ii in range(t_extrap):

        preda = np.r_[preda, preda[-1] + pred[-(t_extrap - ii)] ]

    #cannot have negative new cases

    preda[preda < 0] = 0

    #note that e_data term (estimate of data variance) seems to dominate over epreda,

    #error from model parameter uncertainties

    epreda = np.sqrt(np.cumsum(epreda**2 + e_data**2))

    

    #reconstruct original data and propagate uncertainty.    

    predb = np.r_[n[0], n[0:-1] + preda[0:-t_extrap]]

    for ii in range(t_extrap):

        predb = np.r_[predb, predb[-1] + preda[-(t_extrap - ii)] ] 

    epredb = np.sqrt(np.cumsum(epreda**2 + e_data**2))



    res = n - predb[0:-t_extrap]

    return predb, preda, pred, epredb, res, tout



# variables for the fitting

t_extrap = 7

p = 3

q = 1

check_std_norm_in_fits = False



#t_freeze = int(datetime(2020,4,11).strftime('%j'))

t_freeze = int(datetime(2020,4,8).strftime('%j'))



t_fit = t[ t <= t_freeze ]

n_fit = n[ t <= t_freeze ]





predb, preda, pred,\

epredb, res, tout = fit_second_derivative_and_predict(t_fit,

                                                      n_fit,

                                                      t_extrap,

                                                      p,

                                                      q,

                                                      check_plots=check_std_norm_in_fits)



#fit a second test, but based on data up to t_extrap days in the past.  

#This is a straight forward way of testing the model, and investigating its limitations

pre_predb, pre_preda,\

pre_pred, epre_predb,\

pre_res, pre_tout = fit_second_derivative_and_predict(t_fit[0:-t_extrap],

                                                       n_fit[0:-t_extrap],

                                                       2*t_extrap,

                                                       p,

                                                       q,

                                                       check_plots=check_std_norm_in_fits)



def plot_arima_results(t_fit, n_fit, t, n,

                       t_test, n_test,

                       predb, preda, pred,

                       epredb, res, tout,

                       pre_predb, pre_preda,

                       pre_pred, epre_predb,

                       pre_res, pre_tout,

                       month, day):

    """

    This functions takes all outputs from the previous cell block,

    and constructs figures to see the performance of the model.

    

    Two figures are produced.  The first shows the data, and to predictions,

    one from April 1st onward and one from april 8 onward.  

    The 90% confidence intervals are also plotted.

    

    The second figure has the pannels.  

    The first panel is identical to the first figure (above).  The second panel

    shows the residuals (data - model).  The third panel shows the first and

    second differences, and the model predictions of both.

    """

    #this is just the data and the model/predictions

    Fa,(ax1a) = plt.subplots(1,1)

    #this includes three panels to look at the

    #residuals and 1st/2nd differences

    F,(ax1,ax2,ax3) = plt.subplots(3,1,sharex='col')



    #print(pred, len(pred))

    #print(len(sp.r_[t,tout]), len(predb))





    ###################################

    #Panel 1/main plot---data and model

    ###################################

    #use data to april 8 to predict the future

    ax1.plot(np.r_[t_fit,tout],predb,'c.-')

    #use data to april 1 to predict the future

    ax1.plot(np.r_[t_fit[0:-t_extrap],pre_tout],pre_predb,'b.-')



    ax1a.plot(np.r_[t_fit,tout],predb,'c.-')

    ax1a.plot(np.r_[t_fit[0:-t_extrap], pre_tout],pre_predb,'b.-')





    #multiply 1 sigma uncertainty by 2.04 to get +/- 45% confidence interval

    ax1.fill_between(tout,

                     predb[-t_extrap::] + 2.04*epredb,

                     predb[-t_extrap::] - 2.04*epredb,

                     facecolor='c',alpha=0.3)

    ax1a.fill_between(tout,

                     predb[-t_extrap::] + 2.04*epredb,

                     predb[-t_extrap::] - 2.04*epredb,

                     facecolor='c',alpha=0.3)

    ax1.fill_between(pre_tout,

                     pre_predb[-2*t_extrap::] + 2.04*epre_predb,

                     pre_predb[-2*t_extrap::] - 2.04*epre_predb,

                     facecolor='b',alpha=0.3)

    ax1a.fill_between(pre_tout,

                     pre_predb[-2*t_extrap::] + 2.04*epre_predb,

                     pre_predb[-2*t_extrap::] - 2.04*epre_predb,

                     facecolor='b',alpha=0.3)



    #plot data last so that it appears on top

    ax1.plot(t, n,'k.')

    ax1a.plot(t,n,'k.')



    ###############################

    #Panel 2:  residuals and comparison to predictions

    ###############################

    ax2.plot(t_fit,res,'c.-')

    #would plot with residual, except want to connect dots from early times to

    #new data;  need to find common times between prediction and data

    ax2.plot(t_fit, n_fit - pre_predb[0:-t_extrap],'b.-')

    

    

    #will fail if t_test goes beyond t_extrap.  For now, let's just cut these points out

    mask_test = t_test <= max(tout)

    t_test,n_test = t_test[mask_test], n_test[mask_test]

    idx_test = np.where(np.in1d(t, t_test))[0]

    if len(idx_test) > 0:

        #small offset in time to see data points, if the overlap is too close

        ax2.plot(t_test+0.1, n_test - predb[idx_test],'co')



    ly,hy = ax2.get_ylim()

    mask_t = np.in1d(t, pre_tout)

    idx_pred = np.where(np.in1d( np.r_[t_fit[0:-t_extrap],pre_tout], t[mask_t]))[0]

    idx_epred = np.where(np.in1d( pre_tout, t))[0]    

    ax2.errorbar(t[mask_t],

                 n[mask_t] - pre_predb[idx_pred],

                 2.04*epre_predb[idx_epred],fmt='bo-',mfc='w')



    ax2.errorbar(tout, np.zeros(t_extrap), 2.04*epredb,fmt='co-',mfc='w')

    ax2.plot([t_freeze,t_freeze],[ly- 0.2*(hy - ly),hy + 0.2*(hy - ly)],'k--')

    ax2.set_ylim([ly- 0.2*(hy - ly),hy + 0.2*(hy - ly)])



    xl,xh = ax2.get_xlim()

    ax2.plot([xl - 0.1*(xh - xl),xh],[0,0],'k--')

    ax2.set_xlim([xl - 0.1*(xh - xl),xh])



    ###############################

    #Panel 3:  1st and 2nd differences, along with predictions

    ###############################

    ax3.plot(t[1:],diff_n,'r.-',label='$y^{\prime}$ data')

    ax3.plot(np.r_[t_fit[1:],tout], preda,'.-',color='m',label='$y^{\prime}$ model')

    ax3.plot(np.r_[t_fit[1:-t_extrap],pre_tout], pre_preda,'-.',color='m')



    ax3.plot(t[2:],diff2_n,'.-',color='b',label='$y^{\prime \prime}$ data')

    ax3.plot(np.r_[t_fit[2::],tout], pred,'.-',color='c',label='$y^{\prime \prime}$ model')

    ax3.plot(np.r_[t_fit[2:-t_extrap],pre_tout], pre_pred,'-.',color='c')

    #xl,xh = ax3.get_xlim()



    ax3.plot([xl - 0.1*(xh - xl), xh],[0,0],'k--')

    ax3.set_xlim([xl - 0.1*(xh - xl), xh])

    ly,hy = ax3.get_ylim()

    ax3.plot([t_freeze,t_freeze],[ly,hy],'k--')

    ax3.set_ylim([ly,hy])







    ################################

    #Formatting commands and labels

    ################################

    F.subplots_adjust(hspace=0)





    ax3.legend(loc = 'upper left',fontsize=12)

    tick_locs = np.r_[t_fit, tout]

    tick_locs = tick_locs[::7]

    month = np.r_[month, [month[-1]]*len(tout)]

    day = np.r_[day, sp.r_[0:t_extrap] + 1 + day[-1]]



    ax3.set_xticks(tick_locs)

    ax3.set_xticklabels(['{:1.0f}/{:02.0f}'.format(z[0],z[1]) for z in zip(month[::7],day[::7])])

    #print(tick_locs)

    #print(['{:1.0f}/{:02.0f}'.format(z[0],z[1]) for z in zip(month[::7],day[::7])])

    ax1a.set_xticks(tick_locs)

    ax1a.set_xticklabels(['{:1.0f}/{:02.0f}'.format(z[0],z[1]) for z in zip(month[::7],day[::7])])





    ax3.xaxis.set_minor_locator(MultipleLocator(1))

    ax1a.xaxis.set_minor_locator(MultipleLocator(1))





    ax3.set_xlabel('Date')

    #ax1.set_ylabel('Reported Cases',fontsize=16)

    #ax2.set_ylabel('Residuals',fontsize=16)

    #ax3.set_ylabel('Derivatives',fontsize=16)

    plt.figtext(0.04, 0.65,'Reported Cases',fontsize=16,rotation=90)

    plt.figtext(0.04, 0.45,'Residuals',fontsize=16,rotation=90)

    plt.figtext(0.04, 0.2,'Derivatives',fontsize=16,rotation=90)



    ax1a.set_ylabel('Reported Cases')



    ax1.set_title('World Coronavirus Cases')

    ax1a.set_title('World Coronavirus Cases')



    ax1.set_yscale('log')

    ax1a.set_yscale('log')



    ly,hy = ax1.get_ylim()

    #print(ly,hy)

    ax1.set_ylim([ly,hy*3])

    ax1a.set_ylim([ly,hy*3])



    F.set_size_inches(12,8)

    #F.savefig('plots/{}_arima_fits_derivatives.png'.format(sys.argv[1]))

    #Fa.savefig('plots/{}_arima_fits.png'.format(sys.argv[1]))



    

t_test = t[ t > t_freeze ]

n_test = n[ t > t_freeze ]



plot_arima_results(t_fit,n_fit, t, n,

                   t_test,n_test,

                   predb, preda, pred,

                   epredb, res, tout,

                   pre_predb, pre_preda,

                   pre_pred, epre_predb,

                   pre_res, pre_tout,

                   month,day)
#need to predict 4-2 to 5-14

df_test = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/test.csv')



t_freeze = int(datetime(2020,4,14).strftime('%j'))

t_extrap = 30



def get_states_with_nans(dataframe):

    """

    assumes we pass only a data frame from a given country, 

    extract the unique list of province/state names.

    

    just have to tip toe around the NaN, so it doesn't crash everything

    """

    states = dataframe['Province_State']

    state_list = np.unique(states[~states.isnull()])

    if dataframe['Province_State'].isnull().any():

        state_list = np.r_[state_list, np.nan]

    return state_list



fout = open('submission.csv','w')

fout.write('{},{},{}\n'.format('ForecastId',

                               'ConfirmedCases',

                               'Fatalities'))



fout2 = open('annotated_submission.csv','w')

fout2.write('{},{},{},{},{},{},{},{}\n'.format( 'ForecastId',

                                                'ConfirmedCases',

                                                'Fatalities',

                                                'Country_Region',

                                                'Province_State',

                                                'N_cases',

                                                'N_fatalities',

                                                'Dates'))





for country in np.unique(df_test['Country_Region']):

    df_use1 = df[ df['Country_Region'].isin([country])]        

    state_list = get_states_with_nans(df_use1)

        

    for state in state_list:

        print(country,state)

        if isinstance(state, (str)):

            state_str = state.replace(',','_')

        else:

            state_str = 'nan'

        try:



            #print(state)

            if state is np.nan:

                df_use = df_use1[df_use1['Province_State'].isnull()]

            else:

                #print(state)

                df_use = df_use1[ df_use1['Province_State'].isin([state]) ]

            

            year, month, day, day_of_year = [],[],[],[]

            dates = df_use['Date'].values            

            for t in dates:

                y,m,d,doy = str_date_to_doy(t)

                day_of_year.append(doy)

            #extract the columns that I care the most about

            #day_of_year = np.array(day_of_year)

            t = np.array(day_of_year)

            N_cases = df_use['ConfirmedCases'].values

            N_fatalities  = df_use['Fatalities'].values

            

            #print(country,state,N_cases[-10:])

            #if len(N_cases) == 0:

            #    print(df_use)

            

            

            #to look up the date and ForecastID number of the country/region in question

            df_test_use = df_test[ df_test['Country_Region'].isin([country])]

            df_test_use = df_test_use[ df_test_use['Province_State'].isin([state])]



            out_day_of_year = []                

            for date_out in df_test_use['Date']:

                y,m,d,doy = str_date_to_doy(date_out)

                out_day_of_year.append(doy)

                

            out_forecast_id = df_test_use['ForecastId'].values



            data_mask = np.in1d(t,out_day_of_year)

            out_cases_data = np.r_[N_cases[data_mask],

                                   [np.nan]*(len(out_day_of_year) - len(np.where(data_mask == True)[0]))]

            out_fatalities_data = np.r_[N_fatalities[data_mask],

                                       [np.nan]*(len(out_day_of_year) - len(np.where(data_mask == True)[0]))]





            

            

            for ii,n in enumerate([N_cases, N_fatalities]):

            

                t_fit = t[ t <= t_freeze ]

                n_fit = n[ t <= t_freeze ]                

                

                mask = (n_fit == np.nan) | (n_fit == np.inf)

                t_fit, n_fit = t_fit[~mask], n_fit[~mask]

                

                if len(n_fit) == 0:

                    #print('all data removed!')

                    raise ValueError('all data removed!')                    





                predb, preda, pred,\

                epredb, res, tout = fit_second_derivative_and_predict(t_fit,

                                                                      n_fit,

                                                                      t_extrap,

                                                                      p,

                                                                      q,

                                                                      check_plots=check_std_norm_in_fits)



                t_pred = np.r_[t_fit, tout]

                predb = np.around(predb).astype(int)

                                

                mask_pred = np.in1d(t_pred,out_day_of_year)

                                

                #print(np.c_[t_pred[mask_pred], 

                #            predb[mask_pred], 

                #            out_day_of_year,

                #            df_test_use['Date'],

                #            df_test_use['ForecastId']])

                

                #ii is either 0 (cases) or 1 (fatalities)

                if ii == 0:

                    out_cases = predb[mask_pred]

                                        

                else:

                    out_fatalities = predb[mask_pred]

                    

            

            

            #print(len(out_forecast_id))

            #print(len(out_cases))

            #print(len(out_fatalities))

            #print(np.shape(out_countries))

            #print(out_countries)

            #print(np.shape(out_states))

            #print(out_states)

            #print(len(out_cases_data))

            #print(out_cases_data)

            #print(len(out_fatalities_data))

            #print(len(out_dates))

            #input()

            

            #print(len(out_forecast_id), len(out_cases), len(out_fatalities))



                

                

            for ii in range(len(out_forecast_id)):

                fout.write('{},{},{}\n'.format(int(out_forecast_id[ii]),

                                               out_cases[ii],

                                               out_fatalities[ii]))

                

                fout2.write('{},{},{},{},{},{},{},{}\n'.format(

                    int(out_forecast_id[ii]),

                        out_cases[ii],

                        out_fatalities[ii],

                        country.replace(',','_'),

                        state_str,

                        out_cases_data[ii],

                        out_fatalities_data[ii],

                        out_day_of_year[ii]))

            

            

        except Exception as e:

            #raise

            print(country, state, e)

            #print(n_fit)

            out_forecast_id = df_test_use['ForecastId'].values

            for ii in range(len(out_forecast_id)):

                if not np.isnan(out_cases_data[ii]): 

                    fout.write('{},{},{}\n'.format(out_forecast_id[ii],

                                                   out_cases_data[ii],

                                                   out_fatalities[ii]))

                else:

                    #print(out_cases_data)

                    idx_write = np.where(~np.isnan(out_cases_data))[0]

                    #print(idx_write)

                    fout.write('{},{},{}\n'.format(out_forecast_id[ii],

                                                   out_cases_data[idx_write[-1]],

                                                   out_fatalities[idx_write[-1]]))



            

                fout2.write('{},{},{},{},{},{},{},{}\n'.format(

                            int(out_forecast_id[ii]),

                            0,

                            0,

                            country.replace(',','_'),

                            state_str,

                            out_cases_data[ii],

                            out_fatalities_data[ii],

                            out_day_of_year[ii]))

            

            

            



fout.close()

fout2.close()

#france = df_test[df_test['Country_Region'] == 'France']

#for state in france['Province_State'].values:

#    print(state)

    

#print(len(np.where(france['Province_State'].isnull())[0]))



df_annotate = pd.read_csv('annotated_submission.csv')

#df_annotate

diffs = np.diff(df_annotate['ForecastId'].values)

idx = np.where(diffs > 1)[0]

print(idx)

idx = np.r_[idx,idx + 1,idx -1]

print(np.sort(idx))



missing_countries = []

missing_state = []

for country in np.unique(df_test['Country_Region']):

    if country not in np.unique(df_annotate['Country_Region']):

        #print(country)

        missing_countries.append(country)

    

 

print(missing_countries)

print(len(np.where(diffs>1)[0]))

print(len(df_test))

print(len(df_annotate))

#df_annotate.iloc[np.sort(idx)].head(2000)
