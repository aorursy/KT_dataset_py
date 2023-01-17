# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

from tqdm import tqdm

from sklearn.preprocessing import StandardScaler

from sklearn.svm import NuSVR

from sklearn.metrics import mean_absolute_error
train = pd.read_csv('../input/train.csv', dtype={'acoustic_data': np.int16, 'time_to_failure': np.float64})

                    

                    #nrows=6000000
# pandas doesn't show us all the decimals

pd.options.display.precision = 15
train.shape
train.info(memory_usage='deep')
train.rename({"acoustic_data": "signal", "time_to_failure": "time"}, axis="columns", inplace=True)
# Create a training file with simple derived features



rows = 150_000

segments = int(np.floor(train.shape[0] / rows))



X_train = pd.DataFrame(index=range(segments), dtype=np.float64,

                       columns=['ave', 'std', 'max', 'min','kurt','skew','median','var','sum','q1','q19','iqr','diff'\

                                'q2','Q1','q3','q4','Q2','q6','q7','Q3','q8','q81','q82','q9','q10','p1',\

                                'hmean','gmean','time_to_failure'])

y_train = pd.DataFrame(index=range(segments), dtype=np.float64,

                       columns=['time_to_failure'])



for segment in tqdm(range(segments)):

    seg = train.iloc[segment*rows:segment*rows+rows]

    x = seg['signal'].values

    y = seg['time'].values[-1]

    

    y_train.loc[segment, 'time_to_failure'] = y

    

    X_train.loc[segment, 'ave'] = np.mean(x)

    X_train.loc[segment, 'std'] = np.std(x)

    X_train.loc[segment, 'max'] = np.max(x)

    X_train.loc[segment, 'min'] = np.min(x)

    X_train.loc[segment, 'time_to_failure'] = y

    X_train.loc[segment, 'skew'] = ((x-x.mean())/x.std() ** 3).mean()

    X_train.loc[segment,'kurt'] =  ((x-x.mean())/x.std() ** 4).mean()

    X_train.loc[segment, 'sum'] = np.sum(x)

    X_train.loc[segment, 'median'] = np.median(x)

    X_train.loc[segment, 'var'] = np.var(x)

    #X_train.loc[segment, 'mode'] = x.mode

    #X_train.loc[segment, 'log'] = ln(x)

    X_train.loc[segment, 'q1'] = np.quantile(x,.1)

    X_train.loc[segment, 'q19'] = np.quantile(x,.19)

    X_train.loc[segment, 'q2'] = np.quantile(x,.2)

    X_train.loc[segment, 'Q1'] = np.quantile(x,.25)

    X_train.loc[segment, 'q3'] = np.quantile(x,.3)

    X_train.loc[segment, 'q4'] = np.quantile(x,.4)

    X_train.loc[segment, 'Q2'] = np.quantile(x,.5)

    X_train.loc[segment, 'q6'] = np.quantile(x,.6)

    X_train.loc[segment, 'q7'] = np.quantile(x,.7)

    X_train.loc[segment, 'Q3'] = np.quantile(x,.75)

    X_train.loc[segment, 'q8'] = np.quantile(x,.8)

    X_train.loc[segment, 'q81'] = np.quantile(x,.81)

    X_train.loc[segment, 'q82'] = np.quantile(x,.82)

    X_train.loc[segment, 'q9'] = np.quantile(x,.9)

    X_train.loc[segment, 'q10'] = np.quantile(x,1) #81,82

    X_train.loc[segment, 'iqr'] = np.quantile(x,.75) - np.quantile (x,.25)

    X_train.loc[segment, 'p1'] = np.percentile(x,1)

    X_train.loc[segment, 'p2'] = np.percentile(x,2)

    X_train.loc[segment, 'p3'] = np.percentile(x,3)

    X_train.loc[segment, 'p4'] = np.percentile(x,4)

    X_train.loc[segment, 'p5'] = np.percentile(x,5)

    X_train.loc[segment, 'p6'] = np.percentile(x,6)

    X_train.loc[segment, 'p7'] = np.percentile(x,7)

    X_train.loc[segment, 'p8'] = np.percentile(x,8)

    X_train.loc[segment, 'p9'] = np.percentile(x,9)

    X_train.loc[segment, 'p10'] = np.percentile(x,10)

    X_train.loc[segment, 'p11'] = np.percentile(x,11)

    X_train.loc[segment, 'p12'] = np.percentile(x,12)

    X_train.loc[segment, 'p13'] = np.percentile(x,13)

    X_train.loc[segment, 'p14'] = np.percentile(x,14)

    X_train.loc[segment, 'p15'] = np.percentile(x,15)

    X_train.loc[segment, 'p16'] = np.percentile(x,16)

    X_train.loc[segment, 'p17'] = np.percentile(x,17)

    X_train.loc[segment, 'p18'] = np.percentile(x,18)

    X_train.loc[segment, 'p19'] = np.percentile(x,19)

    X_train.loc[segment, 'p20'] = np.percentile(x,20)

    X_train.loc[segment, 'p21'] = np.percentile(x,21)

    X_train.loc[segment, 'p22'] = np.percentile(x,22)

    X_train.loc[segment, 'p23'] = np.percentile(x,23)

    X_train.loc[segment, 'p24'] = np.percentile(x,24)

    X_train.loc[segment, 'p25'] = np.percentile(x,25)

    X_train.loc[segment, 'p26'] = np.percentile(x,26)

    X_train.loc[segment, 'p27'] = np.percentile(x,27)

    X_train.loc[segment, 'p28'] = np.percentile(x,28)

    X_train.loc[segment, 'p29'] = np.percentile(x,29)

    X_train.loc[segment, 'p30'] = np.percentile(x,30)

    X_train.loc[segment, 'p31'] = np.percentile(x,31)

    X_train.loc[segment, 'p32'] = np.percentile(x,32)

    X_train.loc[segment, 'p33'] = np.percentile(x,33)

    X_train.loc[segment, 'p34'] = np.percentile(x,34)

    X_train.loc[segment, 'p35'] = np.percentile(x,35)

    X_train.loc[segment, 'p36'] = np.percentile(x,36)

    X_train.loc[segment, 'p37'] = np.percentile(x,37)

    X_train.loc[segment, 'p38'] = np.percentile(x,38)

    X_train.loc[segment, 'p39'] = np.percentile(x,39)

    X_train.loc[segment, 'p40'] = np.percentile(x,40)

    X_train.loc[segment, 'p41'] = np.percentile(x,41)

    X_train.loc[segment, 'p42'] = np.percentile(x,42)

    X_train.loc[segment, 'p43'] = np.percentile(x,43)

    X_train.loc[segment, 'p44'] = np.percentile(x,44)

    X_train.loc[segment, 'p45'] = np.percentile(x,45)

    X_train.loc[segment, 'p46'] = np.percentile(x,46)

    X_train.loc[segment, 'p47'] = np.percentile(x,47)

    X_train.loc[segment, 'p48'] = np.percentile(x,48)

    X_train.loc[segment, 'p49'] = np.percentile(x,49)

    X_train.loc[segment, 'p50'] = np.percentile(x,50)

    X_train.loc[segment, 'p51'] = np.percentile(x,51)

    X_train.loc[segment, 'p52'] = np.percentile(x,52)

    X_train.loc[segment, 'p53'] = np.percentile(x,53)

    X_train.loc[segment, 'p54'] = np.percentile(x,54)

    X_train.loc[segment, 'p55'] = np.percentile(x,55)

    X_train.loc[segment, 'p56'] = np.percentile(x,56)

    X_train.loc[segment, 'p57'] = np.percentile(x,57)

    X_train.loc[segment, 'p58'] = np.percentile(x,58)

    X_train.loc[segment, 'p59'] = np.percentile(x,59)

    X_train.loc[segment, 'p60'] = np.percentile(x,60)

    X_train.loc[segment, 'p61'] = np.percentile(x,61)

    X_train.loc[segment, 'p62'] = np.percentile(x,62)

    X_train.loc[segment, 'p63'] = np.percentile(x,63)

    X_train.loc[segment, 'p64'] = np.percentile(x,64)

    X_train.loc[segment, 'p65'] = np.percentile(x,65)

    X_train.loc[segment, 'p66'] = np.percentile(x,66)

    X_train.loc[segment, 'p67'] = np.percentile(x,67)

    X_train.loc[segment, 'p68'] = np.percentile(x,68)

    X_train.loc[segment, 'p69'] = np.percentile(x,69)

    X_train.loc[segment, 'p70'] = np.percentile(x,70)

    X_train.loc[segment, 'p71'] = np.percentile(x,71)

    X_train.loc[segment, 'p72'] = np.percentile(x,72)

    X_train.loc[segment, 'p73'] = np.percentile(x,73)

    X_train.loc[segment, 'p74'] = np.percentile(x,74)

    X_train.loc[segment, 'p75'] = np.percentile(x,75)

    X_train.loc[segment, 'p76'] = np.percentile(x,76)

    X_train.loc[segment, 'p77'] = np.percentile(x,77)

    X_train.loc[segment, 'p78'] = np.percentile(x,78)

    X_train.loc[segment, 'p79'] = np.percentile(x,79)

    X_train.loc[segment, 'p80'] = np.percentile(x,80)

    X_train.loc[segment, 'p81'] = np.percentile(x,81)

    X_train.loc[segment, 'p82'] = np.percentile(x,82)

    X_train.loc[segment, 'p83'] = np.percentile(x,83)

    X_train.loc[segment, 'p84'] = np.percentile(x,84)

    X_train.loc[segment, 'p85'] = np.percentile(x,85)

    X_train.loc[segment, 'p86'] = np.percentile(x,86)

    X_train.loc[segment, 'p87'] = np.percentile(x,87)

    X_train.loc[segment, 'p88'] = np.percentile(x,88)

    X_train.loc[segment, 'p89'] = np.percentile(x,89)

    X_train.loc[segment, 'p90'] = np.percentile(x,90)

    X_train.loc[segment, 'p91'] = np.percentile(x,91)

    X_train.loc[segment, 'p92'] = np.percentile(x,92)

    X_train.loc[segment, 'p93'] = np.percentile(x,93)

    X_train.loc[segment, 'p94'] = np.percentile(x,94)

    X_train.loc[segment, 'p95'] = np.percentile(x,95)

    X_train.loc[segment, 'p96'] = np.percentile(x,96)

    X_train.loc[segment, 'p97'] = np.percentile(x,97)

    X_train.loc[segment, 'p98'] = np.percentile(x,98)

    X_train.loc[segment, 'p99'] = np.percentile(x,99)

    X_train.loc[segment, 'p100'] = np.percentile(x,100)
X_train.head()
from scipy import stats

#1

pearson_coef, p_value = stats.pearsonr(X_train['p1'], X_train['time_to_failure'])

print("p1: The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)

#1

pearson_coef, p_value = stats.pearsonr(X_train['p2'], X_train['time_to_failure'])

print("p2: The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)

#1

pearson_coef, p_value = stats.pearsonr(X_train['p3'], X_train['time_to_failure'])

print("p3: The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)

#1

pearson_coef, p_value = stats.pearsonr(X_train['p4'], X_train['time_to_failure'])

print("p4: The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)

#1

pearson_coef, p_value = stats.pearsonr(X_train['p5'], X_train['time_to_failure'])

print("p5: The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)

#1

pearson_coef, p_value = stats.pearsonr(X_train['p6'], X_train['time_to_failure'])

print("p6: The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)

#1

pearson_coef, p_value = stats.pearsonr(X_train['p7'], X_train['time_to_failure'])

print("p7: The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)

#1

pearson_coef, p_value = stats.pearsonr(X_train['p8'], X_train['time_to_failure'])

print("p8: The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)

#1

pearson_coef, p_value = stats.pearsonr(X_train['p9'], X_train['time_to_failure'])

print("p9: The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)

#1

pearson_coef, p_value = stats.pearsonr(X_train['p10'], X_train['time_to_failure'])

print("p10: The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)
#1

pearson_coef, p_value = stats.pearsonr(X_train['p11'], X_train['time_to_failure'])

print("p11: The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)

#1

pearson_coef, p_value = stats.pearsonr(X_train['p12'], X_train['time_to_failure'])

print("p12: The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)

#1

pearson_coef, p_value = stats.pearsonr(X_train['p13'], X_train['time_to_failure'])

print("p13: The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)

#1

pearson_coef, p_value = stats.pearsonr(X_train['p14'], X_train['time_to_failure'])

print("p14: The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)

#1

pearson_coef, p_value = stats.pearsonr(X_train['p15'], X_train['time_to_failure'])

print("p15: The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)

#1

pearson_coef, p_value = stats.pearsonr(X_train['p16'], X_train['time_to_failure'])

print("p16: The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)

#1

pearson_coef, p_value = stats.pearsonr(X_train['p17'], X_train['time_to_failure'])

print("p17: The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)

#1

pearson_coef, p_value = stats.pearsonr(X_train['p18'], X_train['time_to_failure'])

print("p18: The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)

#1

pearson_coef, p_value = stats.pearsonr(X_train['p19'], X_train['time_to_failure'])

print("p19: The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)

#1

pearson_coef, p_value = stats.pearsonr(X_train['p20'], X_train['time_to_failure'])

print("p20: The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)
#1

pearson_coef, p_value = stats.pearsonr(X_train['p21'], X_train['time_to_failure'])

print("p21: The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)

#1

pearson_coef, p_value = stats.pearsonr(X_train['p22'], X_train['time_to_failure'])

print("p22: The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)

#1

pearson_coef, p_value = stats.pearsonr(X_train['p23'], X_train['time_to_failure'])

print("p23: The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)

#1

pearson_coef, p_value = stats.pearsonr(X_train['p24'], X_train['time_to_failure'])

print("p24: The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)

#1

pearson_coef, p_value = stats.pearsonr(X_train['p25'], X_train['time_to_failure'])

print("p25: The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)

#1

pearson_coef, p_value = stats.pearsonr(X_train['p26'], X_train['time_to_failure'])

print("p26: The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)

#1

pearson_coef, p_value = stats.pearsonr(X_train['p27'], X_train['time_to_failure'])

print("p27: The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)

#1

pearson_coef, p_value = stats.pearsonr(X_train['p28'], X_train['time_to_failure'])

print("p28: The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)

#1

pearson_coef, p_value = stats.pearsonr(X_train['p29'], X_train['time_to_failure'])

print("p29: The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)

#1

pearson_coef, p_value = stats.pearsonr(X_train['p30'], X_train['time_to_failure'])

print("p30: The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)
#1

pearson_coef, p_value = stats.pearsonr(X_train['p31'], X_train['time_to_failure'])

print("p31: The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)

#1

pearson_coef, p_value = stats.pearsonr(X_train['p32'], X_train['time_to_failure'])

print("p32: The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)

#1

pearson_coef, p_value = stats.pearsonr(X_train['p33'], X_train['time_to_failure'])

print("p33: The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)

#1

pearson_coef, p_value = stats.pearsonr(X_train['p34'], X_train['time_to_failure'])

print("p34: The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)

#1

pearson_coef, p_value = stats.pearsonr(X_train['p35'], X_train['time_to_failure'])

print("p35: The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)

#1

pearson_coef, p_value = stats.pearsonr(X_train['p36'], X_train['time_to_failure'])

print("p36: The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)

#1

pearson_coef, p_value = stats.pearsonr(X_train['p37'], X_train['time_to_failure'])

print("p37: The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)

#1

pearson_coef, p_value = stats.pearsonr(X_train['p38'], X_train['time_to_failure'])

print("p38: The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)

#1

pearson_coef, p_value = stats.pearsonr(X_train['p39'], X_train['time_to_failure'])

print("p39: The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)

#1

pearson_coef, p_value = stats.pearsonr(X_train['p40'], X_train['time_to_failure'])

print("p40: The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)
#1

pearson_coef, p_value = stats.pearsonr(X_train['p41'], X_train['time_to_failure'])

print("p41: The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)

#1

pearson_coef, p_value = stats.pearsonr(X_train['p42'], X_train['time_to_failure'])

print("p42: The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)

#1

pearson_coef, p_value = stats.pearsonr(X_train['p43'], X_train['time_to_failure'])

print("p43: The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)

#1

pearson_coef, p_value = stats.pearsonr(X_train['p44'], X_train['time_to_failure'])

print("p44: The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)

#1

pearson_coef, p_value = stats.pearsonr(X_train['p45'], X_train['time_to_failure'])

print("p45: The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)

#1

pearson_coef, p_value = stats.pearsonr(X_train['p46'], X_train['time_to_failure'])

print("p46: The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)

#1

pearson_coef, p_value = stats.pearsonr(X_train['p47'], X_train['time_to_failure'])

print("p47: The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)

#1

pearson_coef, p_value = stats.pearsonr(X_train['p48'], X_train['time_to_failure'])

print("p48: The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)

#1

pearson_coef, p_value = stats.pearsonr(X_train['p49'], X_train['time_to_failure'])

print("p49: The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)

#1

pearson_coef, p_value = stats.pearsonr(X_train['p50'], X_train['time_to_failure'])

print("p50: The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)
#1

pearson_coef, p_value = stats.pearsonr(X_train['p51'], X_train['time_to_failure'])

print("p51: The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)

#1

pearson_coef, p_value = stats.pearsonr(X_train['p52'], X_train['time_to_failure'])

print("p52: The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)

#1

pearson_coef, p_value = stats.pearsonr(X_train['p53'], X_train['time_to_failure'])

print("p53: The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)

#1

pearson_coef, p_value = stats.pearsonr(X_train['p54'], X_train['time_to_failure'])

print("p54: The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)

#1

pearson_coef, p_value = stats.pearsonr(X_train['p55'], X_train['time_to_failure'])

print("p55: The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)

#1

pearson_coef, p_value = stats.pearsonr(X_train['p56'], X_train['time_to_failure'])

print("p56: The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)

#1

pearson_coef, p_value = stats.pearsonr(X_train['p57'], X_train['time_to_failure'])

print("p57: The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)

#1

pearson_coef, p_value = stats.pearsonr(X_train['p58'], X_train['time_to_failure'])

print("p58: The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)

#1

pearson_coef, p_value = stats.pearsonr(X_train['p59'], X_train['time_to_failure'])

print("p59: The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)

#1

pearson_coef, p_value = stats.pearsonr(X_train['p60'], X_train['time_to_failure'])

print("p60: The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)

#1

pearson_coef, p_value = stats.pearsonr(X_train['p61'], X_train['time_to_failure'])

print("p61: The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)

#1

pearson_coef, p_value = stats.pearsonr(X_train['p62'], X_train['time_to_failure'])

print("p62: The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)

#1

pearson_coef, p_value = stats.pearsonr(X_train['p63'], X_train['time_to_failure'])

print("p63: The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)

#1

pearson_coef, p_value = stats.pearsonr(X_train['p64'], X_train['time_to_failure'])

print("p64: The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)

#1

pearson_coef, p_value = stats.pearsonr(X_train['p65'], X_train['time_to_failure'])

print("p65: The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)

#1

pearson_coef, p_value = stats.pearsonr(X_train['p66'], X_train['time_to_failure'])

print("p66: The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)

#1

pearson_coef, p_value = stats.pearsonr(X_train['p67'], X_train['time_to_failure'])

print("p67: The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)

#1

pearson_coef, p_value = stats.pearsonr(X_train['p68'], X_train['time_to_failure'])

print("p68: The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)

#1

pearson_coef, p_value = stats.pearsonr(X_train['p69'], X_train['time_to_failure'])

print("p69: The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)

#1

pearson_coef, p_value = stats.pearsonr(X_train['p70'], X_train['time_to_failure'])

print("p70: The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)

#1

pearson_coef, p_value = stats.pearsonr(X_train['p71'], X_train['time_to_failure'])

print("p71: The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)

#1

pearson_coef, p_value = stats.pearsonr(X_train['p72'], X_train['time_to_failure'])

print("p72: The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)

#1

pearson_coef, p_value = stats.pearsonr(X_train['p73'], X_train['time_to_failure'])

print("p73: The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)

#1

pearson_coef, p_value = stats.pearsonr(X_train['p74'], X_train['time_to_failure'])

print("p74: The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)

#1

pearson_coef, p_value = stats.pearsonr(X_train['p75'], X_train['time_to_failure'])

print("p75: The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)

#1

pearson_coef, p_value = stats.pearsonr(X_train['p76'], X_train['time_to_failure'])

print("p76: The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)

#1

pearson_coef, p_value = stats.pearsonr(X_train['p77'], X_train['time_to_failure'])

print("p77: The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)

#1

pearson_coef, p_value = stats.pearsonr(X_train['p78'], X_train['time_to_failure'])

print("p78: The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)

#1

pearson_coef, p_value = stats.pearsonr(X_train['p79'], X_train['time_to_failure'])

print("p79: The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)

#1

pearson_coef, p_value = stats.pearsonr(X_train['p80'], X_train['time_to_failure'])

print("p80: The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)

#1

pearson_coef, p_value = stats.pearsonr(X_train['p81'], X_train['time_to_failure'])

print("p81: The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)

#1

pearson_coef, p_value = stats.pearsonr(X_train['p82'], X_train['time_to_failure'])

print("p82: The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)

#1

pearson_coef, p_value = stats.pearsonr(X_train['p83'], X_train['time_to_failure'])

print("p83: The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)

#1

pearson_coef, p_value = stats.pearsonr(X_train['p84'], X_train['time_to_failure'])

print("p84: The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)

#1

pearson_coef, p_value = stats.pearsonr(X_train['p85'], X_train['time_to_failure'])

print("p85: The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)

#1

pearson_coef, p_value = stats.pearsonr(X_train['p86'], X_train['time_to_failure'])

print("p86: The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)

#1

pearson_coef, p_value = stats.pearsonr(X_train['p87'], X_train['time_to_failure'])

print("p87: The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)

#1

pearson_coef, p_value = stats.pearsonr(X_train['p88'], X_train['time_to_failure'])

print("p88: The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)

#1

pearson_coef, p_value = stats.pearsonr(X_train['p89'], X_train['time_to_failure'])

print("p89: The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)

#1

pearson_coef, p_value = stats.pearsonr(X_train['p90'], X_train['time_to_failure'])

print("p90: The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)
#1

pearson_coef, p_value = stats.pearsonr(X_train['p91'], X_train['time_to_failure'])

print("p91: The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)

#1

pearson_coef, p_value = stats.pearsonr(X_train['p92'], X_train['time_to_failure'])

print("p92: The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)

#1

pearson_coef, p_value = stats.pearsonr(X_train['p93'], X_train['time_to_failure'])

print("p93: The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)

#1

pearson_coef, p_value = stats.pearsonr(X_train['p94'], X_train['time_to_failure'])

print("p94: The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)

#1

pearson_coef, p_value = stats.pearsonr(X_train['p95'], X_train['time_to_failure'])

print("p95: The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)

#1

pearson_coef, p_value = stats.pearsonr(X_train['p96'], X_train['time_to_failure'])

print("p96: The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)

#1

pearson_coef, p_value = stats.pearsonr(X_train['p97'], X_train['time_to_failure'])

print("p97: The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)

#1

pearson_coef, p_value = stats.pearsonr(X_train['p98'], X_train['time_to_failure'])

print("p98: The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)

#1

pearson_coef, p_value = stats.pearsonr(X_train['p99'], X_train['time_to_failure'])

print("p99: The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)

#1

pearson_coef, p_value = stats.pearsonr(X_train['p100'], X_train['time_to_failure'])

print("p100: The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)

from scipy import stats



#1

pearson_coef, p_value = stats.pearsonr(X_train['ave'], X_train['time_to_failure'])

print("ave: The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)



#2

pearson_coef, p_value = stats.pearsonr(X_train['std'], X_train['time_to_failure'])

print("std: The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)



#3



pearson_coef, p_value = stats.pearsonr(X_train['kurt'], X_train['time_to_failure'])

print("kurt: The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)



#4

pearson_coef, p_value = stats.pearsonr(X_train['max'], X_train['time_to_failure'])

print("max: The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)



#5

pearson_coef, p_value = stats.pearsonr(X_train['min'], X_train['time_to_failure'])

print("min: The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)



#6

pearson_coef, p_value = stats.pearsonr(X_train['skew'], X_train['time_to_failure'])

print("skew: The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)



#7

pearson_coef, p_value = stats.pearsonr(X_train['sum'], X_train['time_to_failure'])

print("sum: The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)



#8

pearson_coef, p_value = stats.pearsonr(X_train['median'], X_train['time_to_failure'])

print("median: The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)



#9

pearson_coef, p_value = stats.pearsonr(X_train['var'], X_train['time_to_failure'])

print("var: The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)



#10

pearson_coef, p_value = stats.pearsonr(X_train['q1'], X_train['time_to_failure'])

print("q1: The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)



#11

pearson_coef, p_value = stats.pearsonr(X_train['q19'], X_train['time_to_failure'])

print("q19: The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)



#12

pearson_coef, p_value = stats.pearsonr(X_train['q2'], X_train['time_to_failure'])

print("q2: The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)



#13

pearson_coef, p_value = stats.pearsonr(X_train['Q1'], X_train['time_to_failure'])

print("Q1: The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)



#14

pearson_coef, p_value = stats.pearsonr(X_train['q3'], X_train['time_to_failure'])

print("q3: The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)



#15

pearson_coef, p_value = stats.pearsonr(X_train['q4'], X_train['time_to_failure'])

print("q4: The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)



#16

pearson_coef, p_value = stats.pearsonr(X_train['Q2'], X_train['time_to_failure'])

print("Q2: The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)



#17

pearson_coef, p_value = stats.pearsonr(X_train['q6'], X_train['time_to_failure'])

print("q6: The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)



#18

pearson_coef, p_value = stats.pearsonr(X_train['q7'], X_train['time_to_failure'])

print("q7: The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)



#19

pearson_coef, p_value = stats.pearsonr(X_train['Q3'], X_train['time_to_failure'])

print("Q3: The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)



#20

pearson_coef, p_value = stats.pearsonr(X_train['q8'], X_train['time_to_failure'])

print("q8: The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)



#21

pearson_coef, p_value = stats.pearsonr(X_train['q81'], X_train['time_to_failure'])

print("q81: The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)



#22

pearson_coef, p_value = stats.pearsonr(X_train['q82'], X_train['time_to_failure'])

print("q82: The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)



#23

pearson_coef, p_value = stats.pearsonr(X_train['q9'], X_train['time_to_failure'])

print("q9: The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)



#24

pearson_coef, p_value = stats.pearsonr(X_train['q10'], X_train['time_to_failure'])

print("q10: The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)



#25

pearson_coef, p_value = stats.pearsonr(X_train['iqr'], X_train['time_to_failure'])

print("iqr: The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)
#X_train['time_to_failure'].unique

#X_train['signal'].describe()

#train['signal'].describe()
#X_train=X_train[X_train['ave']<=5.20555]
# Creating a cloumn featuring binary values on the basis of accident risks

#X_train['outcome'] =[1 if 4.519464272770625 else 0 for x in X_train['std']]

#train['outcome'] =[1 if 4.884113333333334 else 0 for x in train['signal']]
x_train=X_train[['p1','p2','p3','p4','p5','p6','p7','p8','p9','p10','p11','p12','p13','p14','p15','p16',\

                 'p17','p18','p19','p20','p21','p22','p23','p24','p25','p26','p27','p28','p29','p30',\

                'p69','p70','p71','p72','p73','p74','p75','p76','p77','p78','p79','p80','p81',\

                'p82','p83','p84','p85','p86','p87','p88','p89','p90','p91','p92','p93','p94','p95',\

                'p96','p97','p98','p99',]] #=2.019
x_train.head()
x_train=x_train.shift(periods=3)#, fill_value=0)
#x_train=X_train [[ 'p13','p19','p80','p81','p82','p86','p87','p88', 'p89', 'p90']]
x_train.head()
x_train=x_train.fillna(0)
#numpy.corrcoef(df['C'][1:-1], df['C'][2:])
#def df_autocorr(df, lag=1, axis=0):

 #   """Compute full-sample column-wise autocorrelation for a DataFrame."""

  #  return df.apply(lambda col: col.autocorr(lag), axis=axis)

#d1 = DataFrame(np.random.randn(100, 6))
#x_train['min'].autocorr(lag=0)
x_train.shape
x_train.head()
#y_train.head()

y_train=X_train[['time_to_failure']]

y_train.head()
y_train.shape
scaler = StandardScaler()

scaler.fit(x_train)

X_train_scaled = scaler.transform(x_train)
svm = NuSVR()

svm.fit(X_train_scaled, y_train.values.flatten())

y_pred = svm.predict(X_train_scaled)
plt.figure(figsize=(6, 6))

plt.scatter(y_train.values.flatten(), y_pred)

plt.xlim(0, 20)

plt.ylim(0, 20)

plt.xlabel('actual', fontsize=12)

plt.ylabel('predicted', fontsize=12)

plt.plot([(0, 0), (20, 20)], [(0, 0), (20, 20)])

plt.show()
score = mean_absolute_error(y_train.values.flatten(), y_pred)

print(f'Score: {score:0.3f}')
submission = pd.read_csv('../input/sample_submission.csv', index_col='seg_id')
X_test = pd.DataFrame(columns=x_train.columns, dtype=np.float64, index=submission.index)
X_test.shape
X_test.head()
for seg_id in X_test.index:

    seg = pd.read_csv('../input/test/' + seg_id + '.csv')

    

    x = seg['acoustic_data'].values

    

    X_test.loc[seg_id, 'p1'] = np.percentile(x,1)

    X_test.loc[seg_id, 'p2'] = np.percentile(x,2)

    X_test.loc[seg_id, 'p3'] = np.percentile(x,3)

    X_test.loc[seg_id, 'p4'] = np.percentile(x,4)

    X_test.loc[seg_id, 'p5'] = np.percentile(x,5)

    X_test.loc[seg_id, 'p6'] = np.percentile(x,6)

    X_test.loc[seg_id, 'p7'] = np.percentile(x,7)

    X_test.loc[seg_id, 'p8'] = np.percentile(x,8)

    X_test.loc[seg_id, 'p9'] = np.percentile(x,9)

    X_test.loc[seg_id, 'p10'] = np.percentile(x,10)

    X_test.loc[seg_id, 'p11'] = np.percentile(x,11)

    X_test.loc[seg_id, 'p12'] = np.percentile(x,12)

    X_test.loc[seg_id, 'p13'] = np.percentile(x,13)

    X_test.loc[seg_id, 'p14'] = np.percentile(x,14)

    X_test.loc[seg_id, 'p15'] = np.percentile(x,15)

    X_test.loc[seg_id, 'p16'] = np.percentile(x,16)

    X_test.loc[seg_id, 'p17'] = np.percentile(x,17)

    X_test.loc[seg_id, 'p18'] = np.percentile(x,18)

    X_test.loc[seg_id, 'p19'] = np.percentile(x,19)

    X_test.loc[seg_id, 'p20'] = np.percentile(x,20)

    X_test.loc[seg_id, 'p21'] = np.percentile(x,21)

    X_test.loc[seg_id, 'p22'] = np.percentile(x,22)

    X_test.loc[seg_id, 'p23'] = np.percentile(x,23)

    X_test.loc[seg_id, 'p24'] = np.percentile(x,24)

    X_test.loc[seg_id, 'p25'] = np.percentile(x,25)

    X_test.loc[seg_id, 'p26'] = np.percentile(x,26)

    X_test.loc[seg_id, 'p27'] = np.percentile(x,27)

    X_test.loc[seg_id, 'p28'] = np.percentile(x,28)

    X_test.loc[seg_id, 'p29'] = np.percentile(x,29)

    X_test.loc[seg_id, 'p30'] = np.percentile(x,30)

    

    X_test.loc[seg_id, 'p69'] = np.percentile(x,69)

    X_test.loc[seg_id, 'p70'] = np.percentile(x,70)

    X_test.loc[seg_id, 'p71'] = np.percentile(x,71)

    X_test.loc[seg_id, 'p72'] = np.percentile(x,72)

    X_test.loc[seg_id, 'p73'] = np.percentile(x,73)

    X_test.loc[seg_id, 'p74'] = np.percentile(x,74)

    X_test.loc[seg_id, 'p75'] = np.percentile(x,75)

    X_test.loc[seg_id, 'p76'] = np.percentile(x,76)

    X_test.loc[seg_id, 'p77'] = np.percentile(x,77)

    X_test.loc[seg_id, 'p78'] = np.percentile(x,78)

    X_test.loc[seg_id, 'p79'] = np.percentile(x,79)

    X_test.loc[seg_id, 'p80'] = np.percentile(x,80)

    X_test.loc[seg_id, 'p81'] = np.percentile(x,81)

    X_test.loc[seg_id, 'p82'] = np.percentile(x,82)

    X_test.loc[seg_id, 'p83'] = np.percentile(x,83)

    X_test.loc[seg_id, 'p84'] = np.percentile(x,84)

    X_test.loc[seg_id, 'p85'] = np.percentile(x,85)

    X_test.loc[seg_id, 'p86'] = np.percentile(x,86)

    X_test.loc[seg_id, 'p87'] = np.percentile(x,87)

    X_test.loc[seg_id, 'p88'] = np.percentile(x,88)

    X_test.loc[seg_id, 'p89'] = np.percentile(x,89)

    X_test.loc[seg_id, 'p90'] = np.percentile(x,90)

    X_test.loc[seg_id, 'p91'] = np.percentile(x,91)

    X_test.loc[seg_id, 'p92'] = np.percentile(x,92)

    X_test.loc[seg_id, 'p93'] = np.percentile(x,93)

    X_test.loc[seg_id, 'p94'] = np.percentile(x,94)

    X_test.loc[seg_id, 'p95'] = np.percentile(x,95)

    X_test.loc[seg_id, 'p96'] = np.percentile(x,96)

    X_test.loc[seg_id, 'p97'] = np.percentile(x,97)

    X_test.loc[seg_id, 'p98'] = np.percentile(x,98)

    X_test.loc[seg_id, 'p99'] = np.percentile(x,99)

    
X_test.shape
X_test.isnull().sum()
X_test_scaled = scaler.transform(X_test)

submission['time_to_failure'] = svm.predict(X_test_scaled)

submission.to_csv('submission.csv')