#import libraries

import numpy as np

import pandas as pd #for DataFrame

from scipy import stats #for pearson correlation
list_1 = {'H': pd.Series([34,76,21,98,12,98,55,76,22,43]),

          'W': pd.Series([65,32,77,87,54,22,90,78,65,33]),

          'Marks':pd.Series([56,76,34,78,56,76,98,76,87,46]),

         }

list_1 = pd.DataFrame(list_1)

list_1
covariance = list_1.cov(min_periods=None)

covariance
list_of_H = list_1['H']
list_of_W = list_1['W']

list_of_Marks = list_1['Marks']
corr = stats.pearsonr(list_of_H,list_of_W)
print('Correlation between H and W column: ',corr)
corr_1 = stats.pearsonr(list_of_H,list_of_Marks)

print('Correlation between H and Marks column: ',corr_1)
corr_2 = stats.pearsonr(list_of_W,list_of_Marks)

print('Correlation between W and Marks column: ',corr_2)
list_1
chi_square_test = stats.chisquare(list_of_H, 

                                  list_of_W,

                                  axis=0)
print('chi-square test of given dataset is: ',chi_square_test)
obs = np.array([[71.0,154.0,398.0],[4992.0,2808.0,2737.0]])

exp = np.array([[282.6,165.4,175.0],[4780.4,2796.6,2960.0]])
print('chi-square of given set matrix: ',stats.chisquare(f_obs=obs,f_exp=exp,axis=None))