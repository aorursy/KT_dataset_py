# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import matplotlib.pyplot as plt 

import pandas as pd 

import numpy as np 

import seaborn as sns

from scipy.stats.mstats import winsorize

import scipy.stats as stats

import warnings

warnings.filterwarnings('ignore')



pd.options.display.float_format= '{:.3f}'.format

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
starts = pd.read_csv("../input/kickstarter-projects/ks-projects-201801.csv")
starts.isnull().sum()*100/starts.shape[0]
starts.columns
starts.info()
starts.describe().T
starts['goal'].unique()
starts['usd pledged'].unique()
starts['usd pledged'].fillna(starts['usd_pledged_real'], inplace=True)
starts.isnull().sum()*100/starts.shape[0]
# Check missing values in the column "name"

starts[pd.isnull(starts['name'])].index
starts[starts.index == 166851]

starts[starts.index == 307234]
starts['name'].unique()
starts.shape
# we have a lot data so we can delete mising value of name feature but we fill with 'unknown'

starts['name'].fillna('unknown',inplace=True)
starts.isnull().sum()*100/starts.shape[0] # our data has correct values and no missing values.
starts.head()
plt.figure(figsize=(30,15))

title_font = {'family': 'arial', 'color': 'darkred','weight': 'bold','size': 13 }

curve_font  = {'family': 'arial', 'color': 'darkblue','weight': 'bold','size': 10 }



variables = ['goal', 'pledged' , 'backers',"usd pledged","usd_pledged_real","usd_goal_real"]



for i in range(6):

    plt.subplot(2, 6, i+1)

    plt.hist(starts[variables[i]])

    plt.title(variables[i], fontdict=title_font)

    

for i in range(6):

    plt.subplot(2, 6, i+7)

    plt.hist(np.log(starts[variables[i]]+1))

    plt.title(variables[i] + ' (logarithm expression)', fontdict=title_font)
plt.figure(figsize=(30,15))

title_font = {'family': 'arial', 'color': 'darkred','weight': 'bold','size': 13 }

curve_font  = {'family': 'arial', 'color': 'darkblue','weight': 'bold','size': 10 }



variables = ['goal', 'pledged' , 'backers',"usd pledged","usd_pledged_real","usd_goal_real"]



for i in range(6):

    plt.subplot(2, 6, i+1)

    plt.boxplot(starts[variables[i]])

    plt.title(variables[i], fontdict=title_font)

    

for i in range(6):

    plt.subplot(2, 6, i+7)

    plt.boxplot(np.log(starts[variables[i]]+1))

    plt.title(variables[i] + ' (logarithm expression)', fontdict=title_font)
# IQR Method 



q75,q25= np.percentile(starts['goal'],[75,25])

caa= q75-q25
threshold_variables=[]

for threshold_worth in np.arange(1,5):

    min_worth=q25 - (caa*threshold_worth)

    max_worth=q75 + (caa*threshold_worth)

    

    number_of_outliers=len((np.where((starts['goal']>max_worth) | (starts['goal']<min_worth))[0]))

    threshold_variables.append((threshold_worth,number_of_outliers))

threshold_variables
log_threshold_variables= pd.DataFrame()

variables = ['goal', 'pledged' , 'backers',"usd pledged","usd_pledged_real","usd_goal_real"]

for j in variables:

    for threshold_worth in np.arange(1,5,1):

        q75_log, q25_log = np.percentile(np.log(starts[j]), [75 ,25])

        caa_log = q75_log - q25_log

        q75, q25 = np.percentile(starts[j], [75 ,25])

        caa= q75 - q25

        min_worth_log = q25_log - (caa_log*threshold_worth) 

        max_worth_log = q75_log + (caa_log*threshold_worth)

        min_worth= q25 - (caa*threshold_worth) #logarithm expression

        max_worth = q75 + (caa*threshold_worth) #logarithm expression

        number_of_outliers_log = len((np.where((np.log(starts[j]) > max_worth_log)| 

                                               (np.log(starts[j]) < min_worth_log))[0]))

        number_of_outliers = len((np.where((starts[j] > max_worth)| 

                                               (starts[j] < min_worth))[0]))

        log_threshold_variables = log_threshold_variables.append({'threshold_worth': threshold_worth,

                                                            'number_of_outliers' : number_of_outliers, #logarithm expression

                                                            'number_of_outliers_log': number_of_outliers_log 

                                                            }, ignore_index=True)

    print("-"*10,"",j,"-"*10)

    display(log_threshold_variables)

    log_threshold_variables = pd.DataFrame()
plt.boxplot(np.log(starts['goal']),whis=4)

plt.show()
starts.head()
starts_2= starts.copy()
starts_2['goal']= np.log(starts_2['goal'])

starts_2['pledged']= np.log(starts_2['pledged']+1) # we have 0.. if there is 0 log will write -inf

starts_2['usd pledged']= np.log(starts_2['usd pledged']+1)  

starts_2['usd_pledged_real']=np.log(starts_2['usd_pledged_real']+1) 

starts_2['usd_goal_real']= np.log(starts_2['usd_goal_real'])
np.log(starts_2['goal']).isnull().index # There is no empty value 
starts_2['pledged']
plt.figure(figsize=(28,18))

log_columns=['goal','pledged','usd pledged','usd_pledged_real','usd_goal_real']

    

for i in range(5):

    plt.subplot(2, 5, i+1)

    plt.hist(starts_2[log_columns[i]])

    plt.title(log_columns[i] + ' (logarithm expression)', fontdict=title_font)

for i in range(5):

    plt.subplot(2, 5, i+6)

    plt.boxplot(starts_2[log_columns[i]])

    plt.title(log_columns[i] + ' (logarithm expression)', fontdict=title_font)
from scipy.stats.mstats import winsorize

winsorize_starts = winsorize(starts_2["goal"], (0.01, 0.01))



winsorize_starts
plt.boxplot(winsorize_starts )

plt.show()
plt.hist(winsorize_starts)

plt.show()
starts_2['goal']=winsorize(starts_2["goal"], (0.01, 0.01))

starts_2['usd_goal_real']=winsorize(starts_2["usd_goal_real"], (0.01, 0.01))
plt.boxplot(starts_2['goal'])

plt.show()
plt.hist(starts_2['goal'])

plt.show()
from scipy.stats import jarque_bera

from scipy.stats import normaltest



pd.options.display.float_format = '{:.5f}'.format



columns = ["goal", "pledged", "usd pledged","usd_pledged_real","usd_goal_real"]

dispersion_tests = pd.DataFrame(columns=['column', 'jarque_bera_stats', 'jarque_bera_p_value', 

                                         'normal_stats', 'normal_p_value'])



for column in columns:

    jb_stats = jarque_bera(starts_2[column])

    norm_stats = normaltest(starts_2[column])

    dispersion_tests = dispersion_tests.append({"column": column,

                                                "jarque_bera_stats" : jb_stats[0] ,

                                                "jarque_bera_p_value" : jb_stats[1] ,

                                                "normal_stats": norm_stats[0] , 

                                                "normal_p_value" : norm_stats[1]

                                               }, ignore_index=True)

dispersion_tests
log_threshold_variables= pd.DataFrame()

variables = ['goal', 'pledged' , 'backers',"usd pledged","usd_pledged_real","usd_goal_real"]

for j in variables:

    for threshold_worth in np.arange(1,5,1):

        q75_log, q25_log = np.percentile(np.log(starts_2[j]), [75 ,25])

        caa_log = q75_log - q25_log

        q75, q25 = np.percentile(starts_2[j], [75 ,25])

        caa= q75 - q25

        min_worth_log = q25_log - (caa_log*threshold_worth) 

        max_worth_log = q75_log + (caa_log*threshold_worth)

        min_worth= q25 - (caa*threshold_worth) #logarithm expression

        max_worth = q75 + (caa*threshold_worth) #logarithm expression

        number_of_outliers_log = len((np.where((np.log(starts_2[j]) > max_worth_log)| 

                                               (np.log(starts_2[j]) < min_worth_log))[0]))

        number_of_outliers = len((np.where((starts_2[j] > max_worth)| 

                                               (starts_2[j] < min_worth))[0]))

        log_threshold_variables = log_threshold_variables.append({'threshold_worth': threshold_worth,

                                                            'number_of_outliers' : number_of_outliers, #logarithm expression

                                                            'number_of_outliers_log': number_of_outliers_log 

                                                            }, ignore_index=True)

    print("-"*10,"",j,"-"*10)

    display(log_threshold_variables)

    log_threshold_variables = pd.DataFrame()
starts.head()
starts.isnull().sum()*100/starts.shape[0]
from sklearn.preprocessing import normalize



starts["norm_goal"] = normalize(np.array(starts["goal"]).reshape(1,-1)).reshape(-1,1)

starts["norm_pledged"] = normalize(np.array(starts["pledged"]).reshape(1,-1)).reshape(-1,1)

starts["norm_usd_pledged_real"] = normalize(np.array(starts["usd_pledged_real"]).reshape(1,-1)).reshape(-1,1)

starts["norm_usd_goal_real"] = normalize(np.array(starts["usd_goal_real"]).reshape(1,-1)).reshape(-1,1)



normal_ozellikler=["goal","norm_goal","pledged","norm_pledged",

                    "usd_pledged_real","norm_usd_pledged_real","usd_goal_real","norm_usd_goal_real"]





print('Minimum Worths\n-----------------',)

print(starts[normal_ozellikler].min())

print('\nMaksimum Worths\n-----------------',)

print(starts[normal_ozellikler].max())
plt.figure(figsize=(22,5))



for i in range(4):

    plt.subplot(1,4,i+1)

    plt.scatter(starts[normal_ozellikler[2*i]], starts[normal_ozellikler[2*i+1]])

    plt.title("Orjinal and Normalize Worths \n ({})".format(normal_ozellikler[2*i]), fontdict=title_font)

    plt.xlabel("Orjinal Worths", fontdict=curve_font)

    plt.ylabel("Normalize Worths", fontdict=curve_font)



plt.show()
plt.hist(winsorize(starts['norm_goal'], (0,.05)))

plt.show()
plt.hist(starts['norm_goal'])

plt.show()
starts_2.head()
plt.hist(starts_2['goal'])

plt.show()
starts_2['success_pleged_ration']= (starts['pledged']*100)/starts['goal']
starts_2.head()
plt.figure(figsize=(22,15))

log_columns=['goal','pledged','usd pledged','usd_pledged_real','usd_goal_real','success_pleged_ration']

    

for i in range(5):

    plt.subplot(2, 5, i+1)

    plt.hist(starts_2[log_columns[i]])

    plt.title(log_columns[i] + ' (logarithmic expression)', fontdict=title_font)

for i in range(5):

    plt.subplot(2, 5, i+6)

    plt.boxplot(starts_2[log_columns[i]])

    plt.title(log_columns[i] + ' (logarithmic expression)', fontdict=title_font)
corr_starts_2=starts_2.corr()

display(corr_starts_2)
plt.figure(figsize=(18,5))

sns.heatmap(corr_starts_2, square=True, annot=True, linewidths=.5, vmin=0, vmax=1, cmap='viridis')

plt.title("Correlation Matrix (Starts_2)", fontdict=title_font)



plt.show()
starts_2.head()
starts_2.info()
def year_cut(string):

    return string[0:4]

starts_2['year'] = starts_2['launched'].apply(year_cut)

starts_2['year'] = starts_2['year'].astype(int)
from datetime import datetime

starts_2['deadline']= pd.to_datetime(starts_2['deadline'])

starts_2['launched']= pd.to_datetime(starts_2['launched'])
print('Categories in category: ', starts_2['category'].nunique())

starts_2['category'].value_counts()[:20].plot(kind='barh', 

                                        figsize=(14,6), 

                                        title='Top 20 most popular categories')
#In which category did investors mostly donate?

df=pd.DataFrame(starts_2.groupby('category')['pledged'].sum())

df=df.reset_index()

df



df2=pd.DataFrame()

df2= df[df.pledged>=40000]

df2.head()

df2.reset_index()

df2

plt.figure(figsize=(30,20))

sns.barplot(df2['category'], y= df2['pledged'],

            palette="Blues_d", saturation = 0.5)

sns.despine(right = True, top = True)
# In which year did investors invest the most?

df=pd.DataFrame(starts_2.groupby("year")["pledged"].sum(),columns=['pledged'])

df=df.reset_index()

df

plt.figure(figsize=(20,10))

sns.barplot(df['year'], y= df['pledged'] ,

            palette="Blues_d", saturation = 0.5)

sns.despine(right = True, top = True)
#Which category has been the most successful?

df=pd.DataFrame(starts_2.groupby('category')['success_pleged_ration'].mean())

df=df.reset_index()

df



df2=pd.DataFrame()

df2= df[df.success_pleged_ration>=1000]

df2.head()

df2.reset_index()

df2





plt.figure(figsize=(30,20))

sns.barplot(df2['category'], y= df2['success_pleged_ration'],

            palette="Blues_d", saturation = 0.5)

sns.despine(right = True, top = True)
# Which year has the most successful enterprise?

adet_tablosu = pd.crosstab(starts_2["year"], starts_2["state"])

adet_tablosu
#Which year has the most successful enterprise?



plt.figure(figsize=(18,10))

sns.countplot(y="year", hue="state", data=starts_2)

plt.title("Successful Enterprise Amount by Years", fontdict = title_font)

plt.ylabel("YEARS", fontdict = curve_font)

plt.xlabel("STATE", fontdict = curve_font)

plt.show()
#How many of the startup have received the investment they targeted

df=pd.DataFrame()

df= starts_2[starts_2.success_pleged_ration<100]

df=df.reset_index()

df



df2=pd.DataFrame()

df2= starts_2[starts_2.success_pleged_ration>=100]

df2.head()

df2.reset_index()

df2



plt.figure(figsize=(15,5))

plt.subplot(1,2,1)

plt.hist(df['success_pleged_ration'])

plt.title("not receive their targeted investment",fontdict=title_font)

plt.subplot(1,2,2)

plt.hist(df2['success_pleged_ration'])

plt.title("reaching their targeted investment",fontdict=title_font)

plt.show()
plt.hist(starts_2[starts_2["success_pleged_ration"]>=100].iloc[:,16], density=True, alpha=0.6)

plt.hist(starts_2[starts_2["success_pleged_ration"]<100].iloc[:,16], density=True, alpha=0.6)

plt.title('reaching/not reaching their targeted investment',fontdict=title_font)

plt.show()
#How many ventures were successful despite collecting the money he had targeted?

df2=pd.DataFrame()

df2= starts_2[starts_2.success_pleged_ration>=100]

df2.reset_index()



plt.figure(figsize=(20,10))

sns.barplot(df2['state'], y= df2['success_pleged_ration'],

            palette="Blues_d", saturation = 0.5)

plt.title("Startups that take the investment they target",fontdict=title_font)

sns.despine(right = True, top = True)
#How many ventures were successful despite collecting the money he had not targeted?



df=pd.DataFrame()

df= starts_2[starts_2.success_pleged_ration<100]

df=df.reset_index()



plt.figure(figsize=(20,10))

sns.barplot(df['state'], y= df['success_pleged_ration'],

            palette="Blues_d", saturation = 0.5)

plt.title("startups that Not take the investment they target",fontdict=title_font)

sns.despine(right = True, top = True)
#which country has reached its targeted investment?



df2=pd.DataFrame()

df2= starts_2[starts_2.success_pleged_ration>=100]

df2.reset_index()



plt.figure(figsize=(20,10))

sns.barplot(df2['country'], y= df2['success_pleged_ration'],

            palette="Blues_d", saturation = 0.5)

plt.title("Most invested countries",fontdict=title_font)

sns.despine(right = True, top = True)
# t-test

country = df2["country"].unique()

grup_country = df2.groupby("country")
import scipy.stats as stats

for var in ["success_pleged_ration"]:

    karsilastirma = pd.DataFrame(columns=['grup_1', 'grup_2','istatistik', 'p_degeri'])

    print("{} için karşılaştırma".format(var),end='')

    for i in range(0, len(country)):

        for j in range(i+1, len(country)):

            ttest = stats.ttest_ind(df2[df2["country"]==country[i]][var], 

                                df2[df2["country"]==country[j]][var])

            grup_1 = country[i]

            grup_2 = country[j]

            istatistik = ttest[0]

            p_degeri = ttest[1]

            

            karsilastirma = karsilastirma.append({"grup_1" : grup_1 ,

                                                  "grup_2" : grup_2 ,

                                                  "istatistik": istatistik , 

                                                  "p_degeri" : p_degeri}, ignore_index=True)

    display(karsilastirma)
#which countries have been successful?

plt.figure(figsize=(20,15))

sns.catplot(y="country", hue="state", kind="count",

            palette="pastel", edgecolor=".6",

            data=starts_2)

plt.title("According to countries ; Success/Fail", fontdict = title_font)

plt.show()