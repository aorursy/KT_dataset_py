#!/usr/bin/env python

# coding: utf-8



# In[1]:





# Multiple Regression Variable Selection

def mr(selection=False):

    import pandas as pd

    h=pd.read_csv('HDB_Data500.csv',index_col=0)

    #print(h.head(0)) # dataset's variable names

    

    yvar='resale_price'

    modeleq = yvar + ' ~'

    for xvar in ( # Insert new 'x variable' into a row, ending with ','

        'storey_range_lower',

        'storey_range_upper',

        'floor_area_sqm',

        'lease_commence_year',

        'transaction_month',

        'town',

        'flat_model',

        'flat_type',

        'no_of_rooms',

        'block_number',

        'postal_code',

        'postal_code_2digit',

        'storey_range_lower_sq',

        'storey_range_lower_rt',

        'storey_range_upper_sq',

        'storey_range_upper_rt',

        'floor_area_sqm_sq',

        'floor_area_sqm_rt',

        'lease_commence_year_sq',

        'lease_commence_year_rt',

        'transaction_month_sq',

        'transaction_month_rt',

        'block_number_sq',

        'block_number_rt',

        'postal_code_sq',

        'postal_code_rt',

        'postal_code_2digit_sq',

        'postal_code_2digit_rt'

        ):

        if modeleq[-1] == '~':

            modeleq = modeleq + ' ' + xvar

        else:

            modeleq = modeleq + ' + ' + xvar



    #import matplotlib.pyplot as pl

    #%matplotlib inline

    #import numpy as np



    import statsmodels.api as sm

    from statsmodels.formula.api import ols

    

    bmodeleq=modeleq

    if selection :

        print('Variable Selection using p-value & AIC:')

        #minfpv = 1.0

        minaic=10000000000



        while True :

            #Specify C() for Categorical, else could be interpreted as numeric:

            #hout=ols('resale_price ~ floor_area_sqm + C(flat_type)', data=h).fit()

            hout=ols(modeleq, data=h).fit()

            if modeleq.find(' + ') == -1 :

                # 1 xvar left

                break



            #print(dir(hout)) gives all the attributes of .fit(), e.g. .fvalue & .f_pvalue

            AIC=hout.aic

            if AIC < minaic :

                minaic=AIC

                bmodeleq=modeleq

            print('\nF-statistic =',hout.fvalue,'       AIC =',hout.aic)



            prf = sm.stats.anova_lm(hout, typ=3)['PR(>F)']

            maxp=max(prf[1:])

            

            #print('\n',dict(prf))



            xdrop = prf[maxp==prf].axes[0][0] # 1st element of row-label .axes[0]

            print(xdrop)

            #if xdrop.find('Intercept') != -1 :

            #    break



            # xdrop removed from model equation:

            if (modeleq.find('~ ' + xdrop + ' + ') != -1): 

                modeleq = modeleq.replace('~ ' + xdrop + ' + ','~ ') 

            elif (modeleq.find('+ ' + xdrop + ' + ') != -1): 

                modeleq = modeleq.replace('+ ' + xdrop + ' + ','+ ')

            else:

                modeleq = modeleq.replace(' + ' + xdrop,'') 

            #print('Model equation:',modeleq,'\n')

            print('-----------',modeleq)

            print('Variable to drop:',xdrop,'       p-value =',prf[xdrop])

            #print('\nVariable left:\n'+str(prf[maxp!=prf][:-1]),'\n')

        

        print('\nF-statistic =',hout.fvalue,'       AIC =',hout.aic)

        print('Variable left:\n'+str(prf[maxp!=prf][:-1]),'\n')

        #input("found intercept")

        print('Best model equation:',bmodeleq)

        print('Minimum AIC) =',minaic,'\n')

        

    hout=ols(bmodeleq, data=h).fit()



    print(sm.stats.anova_lm(hout, typ=1))

    #print(anova) # Anova table with 'Treatment' broken up

    hsum=hout.summary()



    print('\n',hsum)

    

    last=10 #number of bottom p-values to display with more precision

    hout_anova= sm.stats.anova_lm(hout, typ=1).sort_values(by='PR(>F)',ascending=False)   

    #p-values are not in general the same as PR(>F) from ANOVA

    print("\nLast",last,"x-coefficients' PR(<F):")

    nxvar=len(hout_anova)

    for i in range(last+1,1,-1):

        print('    ',hout_anova.axes[0][nxvar-i],'    ',hout_anova['PR(>F)'][nxvar-i])





    # Output Coefficient table:

    #from IPython.core.display import HTML

    #HTML(hout.summary().tables[1].as_html()) #.tables[] from 0 to 3



mr(True) # do Variable Selection

#mr() # do multiple regression once





# In[ ]:








