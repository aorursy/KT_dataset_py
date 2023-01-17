# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#Import needed packages to process data

import sklearn

import matplotlib as mpl

import matplotlib.pyplot as plt



#Extraction of the data

data_path=os.path.join('/kaggle/input/chronic-disease','U.S._Chronic_Disease_Indicators.csv')

df_source=pd.read_csv(data_path)

df_source.head()
df_source.info()
#Let's see which features are completely filled

(df_source.count()/len(df_source))*100
def df_values(df):

    for i in range(0, len(df.columns)):

        print("*****start of feature ", df.columns[i], "*************************")

        print (df.iloc[:,i].value_counts())

        print ("*****end of feature ", df.columns[i], "************************** \n")
#Exploring the values of every feature

df_values(df_source)
# Lets get of rid those features considered meaningless to my personal understanding: StratificationCategory2, Stratification2, StratificationCategory3,Stratification3,GeoLocation,ResponseID, LocationID 

indexes=[18,19,20,21,22,23,24,30,31,32,33]

df_source=df_source.drop(df_source.columns[indexes],axis=1)
#Re-do info() and non-null percentaje distribution among features

print(df_source.info())

print((df_source.count()/len(df_source))*100)
#DataValue feature seems to be the key one to drop those rows that have null value

#Altough DataValueAlt could be complementary to DataValue, i.e. some rows with DataValue null value might

#have DataValueAlt filled

#So let's drop those rows where there are null value in DataValue && DataValueAlt features

df_source_filtered=df_source.dropna(how='all', subset=['DataValue','DataValueAlt'])

df_source_filtered.reset_index(drop=True, inplace=True)

df_source_filtered.shape

df_source_filtered.info()
# Explore values in "apparently key" features as topic

# The dataframe is divided into several 'Topic' that could be employed to be analyzed by separate

df_source_filtered['Topic'].value_counts()
#Let's try with Cardiovascular Disease due to its highest occurrence

df_cvd=df_source_filtered[df_source_filtered['Topic']=='Cardiovascular Disease']

df_cvd.info()
# extra sentences used to try instructions

df_cvd['Question'].value_counts()
# A first analysis is done about mortality:

#'Mortality from heart failure', 'Mortality from cerebrovascular disease (stroke)',

#'Mortality from diseases of the heart', 'Mortality from total cardiovascular diseases'

df_cvd_mortality=df_cvd[(df_cvd['Question']=='Mortality from heart failure')|(df_cvd['Question']=='Mortality from cerebrovascular disease (stroke)')|(df_cvd['Question']=='Mortality from diseases of the heart')|(df_cvd['Question']=='Mortality from total cardiovascular diseases')]

df_values(df_cvd_mortality)

#There are 4497 rows with 'DataValue' blank, so lets get rid of those empty values

df_cvd_mortality=df_cvd_mortality[df_cvd_mortality['DataValue']!=' ']

df_cvd_mortality.reset_index(drop=True, inplace=True)

print(df_cvd_mortality['DataValue'].value_counts())

print(df_cvd_mortality.shape)

df_cvd_mortality.info()
# Lets get of rid another features considered meaningless in the filtered df: DataSource, Response,DataValueFootnoteSymbol,DatavalueFootnote,TopicID,QuestionID,DataValueTypeID,StratificationCategoryID1,StratificationID1 

indexes=[4,7,12,13,18,19,20,21,22]

df_cvd_mortality=df_cvd_mortality.drop(df_cvd_mortality.columns[indexes],axis=1)

print(df_cvd_mortality.info())
df_cvd_mortality

#next steps, study the stratification feature and see the difference the overall value and rest


#When DataValueUnit is NaN the figure expressed in DataValue is unknown, thus let's drop those rows to carried out an precise analysis

df_cvd_mortality.dropna(subset=['DataValueUnit'], inplace=True)

df_cvd_mortality.reset_index(drop=True, inplace=True)

df_cvd_mortality
#Merge features regarding year if possible

print((df_cvd_mortality['YearStart']==df_cvd_mortality['YearEnd']).value_counts())

df_cvd_mortality['Year']=df_cvd_mortality['YearStart']

df_cvd_mortality.drop(['YearStart','YearEnd'], axis=1, inplace=True)

df_cvd_mortality.head()

df_cvd_mortality['Question'].value_counts()

df_cvd_mortality['DataValue']=pd.to_numeric(df_cvd_mortality['DataValue'])

df_cvd_mortality.info()


#Separating into different dataframes depending on 'Question' value

#df_cvd_1

df_cvd_mortality_cardio_disease=df_cvd_mortality[df_cvd_mortality['Question']=='Mortality from total cardiovascular diseases']

#df_cvd_2

df_cvd_mortality_diseases_heart=df_cvd_mortality[df_cvd_mortality['Question']=='Mortality from diseases of the heart']

#df_cvd_3

df_cvd_mortality_heart_failure=df_cvd_mortality[df_cvd_mortality['Question']=='Mortality from heart failure']

#df_cvd_4

df_cvd_mortality_cerebrovascular=df_cvd_mortality[df_cvd_mortality['Question']=='Mortality from cerebrovascular disease (stroke)']

#Show the evolution of deaths by total cardiovascular diseases trough the years

# the datavalues regarding the overall, male and female

df_cvd_1_overall=df_cvd_mortality_cardio_disease[df_cvd_mortality_cardio_disease['Stratification1']=='Overall'].groupby('Year').sum()

df_cvd_1_overall.rename(columns={'DataValue':'DV_overall'},inplace=True)

df_cvd_1_male=df_cvd_mortality_cardio_disease[df_cvd_mortality_cardio_disease['Stratification1']=='Male'].groupby('Year').sum()

df_cvd_1_male.rename(columns={'DataValue':'DV_male'},inplace=True)

df_cvd_1_female=df_cvd_mortality_cardio_disease[df_cvd_mortality_cardio_disease['Stratification1']=='Female'].groupby('Year').sum()

df_cvd_1_female.rename(columns={'DataValue':'DV_female'},inplace=True)



df_cvd_1_merged=pd.merge(df_cvd_1_overall,df_cvd_1_male,on='Year', how='inner')

df_cvd_1_merged=pd.merge(df_cvd_1_merged,df_cvd_1_female,on='Year', how='inner')

df_cvd_1_merged=df_cvd_1_merged[['DV_overall','DV_male','DV_female']]

df_cvd_1_merged



df_plot = pd.DataFrame({'Overall': df_cvd_1_merged['DV_overall'],

                    'Male': df_cvd_1_merged['DV_male'],

                       'Female': df_cvd_1_merged['DV_female']}, index= df_cvd_1_merged.index)

ax = df_plot.plot.bar(rot=0)
#It's dificult to apprecite a trend in the latter plot, so lets normalize the value df_cvd_1_merged to see the evolution of the figures 

from sklearn import preprocessing

x=df_cvd_1_merged.values 

min_max_scaler = preprocessing.StandardScaler()

x_scaled = min_max_scaler.fit_transform(x)

df_cvd_1_merged_norm = pd.DataFrame(x_scaled, columns=df_cvd_1_merged.columns,index=df_cvd_1_merged.index)

#df_cvd_1_merged_norm=(df_cvd_1_merged-df_cvd_1_merged.mean())/df_cvd_1_merged.std()

df_cvd_1_merged_norm 
df_cvd_1_merged_norm_plt = pd.DataFrame({'Overall': df_cvd_1_merged_norm['DV_overall'],

                    'Male': df_cvd_1_merged_norm['DV_male'],

                       'Female': df_cvd_1_merged_norm['DV_female']}, index= df_cvd_1_merged_norm.index)

ax1 = df_cvd_1_merged_norm_plt.plot.line(rot=0)

ax1.xaxis.set_major_locator(plt.MaxNLocator(4))
ax = df_cvd_1_merged_norm_plt.plot.bar(rot=0)
#Show the evolution of deaths by total cardiovascular diseases trough the years

df_cvd_1_overall=df_cvd_mortality_cardio_disease[df_cvd_mortality_cardio_disease['Stratification1']=='Overall'].groupby('Year').sum()

df_cvd_2_overall=df_cvd_mortality_diseases_heart[df_cvd_mortality_diseases_heart['Stratification1']=='Overall'].groupby('Year').sum()

df_cvd_3_overall=df_cvd_mortality_heart_failure[df_cvd_mortality_heart_failure['Stratification1']=='Overall'].groupby('Year').sum()

df_cvd_4_overall=df_cvd_mortality_cerebrovascular[df_cvd_mortality_cerebrovascular['Stratification1']=='Overall'].groupby('Year').sum()



df_cvd_overall = pd.DataFrame({'Mortality from total cardiovascular diseases': df_cvd_1_overall['DataValue'],

                    'Mortality from diseases of the heart': df_cvd_2_overall['DataValue'],

                       'Mortality from heart failure': df_cvd_3_overall['DataValue'],

                       'Mortality from cerebrovascular disease (stroke)':df_cvd_4_overall['DataValue']}, index= df_cvd_1_overall.index)



x=df_cvd_overall.values 

min_max_scaler = preprocessing.StandardScaler()

x_scaled = min_max_scaler.fit_transform(x)

df_cvd_overall_norm = pd.DataFrame(x_scaled, columns=df_cvd_overall.columns,index=df_cvd_overall.index)

df_cvd_overall_norm 



ax = df_cvd_overall_norm.plot.bar(rot=0)
ax1 = df_cvd_overall_norm.plot.line(rot=0)

ax1.xaxis.set_major_locator(plt.MaxNLocator(4))
#In the above plot the line regarding Mortality from heart failure has a anomalous slope up during 2012-2014

df_cvd_3_overall=df_cvd_mortality_heart_failure[df_cvd_mortality_heart_failure['Stratification1']=='Overall'].groupby('Year').sum()

df_cvd_3_overall.rename(columns={'DataValue':'DV_overall'},inplace=True)

df_cvd_3_male=df_cvd_mortality_heart_failure[df_cvd_mortality_heart_failure['Stratification1']=='Male'].groupby('Year').sum()

df_cvd_3_male.rename(columns={'DataValue':'DV_male'},inplace=True)

df_cvd_3_female=df_cvd_mortality_heart_failure[df_cvd_mortality_heart_failure['Stratification1']=='Female'].groupby('Year').sum()

df_cvd_3_female.rename(columns={'DataValue':'DV_female'},inplace=True)



df_cvd_3_merged=pd.merge(df_cvd_3_overall,df_cvd_3_male,on='Year', how='inner')

df_cvd_3_merged=pd.merge(df_cvd_3_merged,df_cvd_3_female,on='Year', how='inner')

df_cvd_3_merged=df_cvd_3_merged[['DV_overall','DV_male','DV_female']]

df_cvd_3_merged

x=df_cvd_3_merged.values 

min_max_scaler = preprocessing.StandardScaler()

x_scaled = min_max_scaler.fit_transform(x)

df_cvd_3_merged_norm = pd.DataFrame(x_scaled, columns=df_cvd_3_merged.columns,index=df_cvd_3_merged.index)

df_cvd_3_merged_norm_plt = pd.DataFrame({'Overall': df_cvd_3_merged_norm['DV_overall'],

                    'Male': df_cvd_3_merged_norm['DV_male'],

                       'Female': df_cvd_3_merged_norm['DV_female']}, index= df_cvd_3_merged_norm.index)

ax3 = df_cvd_3_merged_norm_plt.plot.line(rot=0)

ax3.xaxis.set_major_locator(plt.MaxNLocator(4))

# We should check state by state to see if any state influences highly in the last two year (see section 3)
#In the above plot the line regarding Mortality from heart failure has a anomalous slope up during 2012-2014

df_cvd_2_overall=df_cvd_mortality_diseases_heart[df_cvd_mortality_diseases_heart['Stratification1']=='Overall'].groupby('Year').sum()

df_cvd_2_overall.rename(columns={'DataValue':'DV_overall'},inplace=True)

df_cvd_2_male=df_cvd_mortality_diseases_heart[df_cvd_mortality_diseases_heart['Stratification1']=='Male'].groupby('Year').sum()

df_cvd_2_male.rename(columns={'DataValue':'DV_male'},inplace=True)

df_cvd_2_female=df_cvd_mortality_diseases_heart[df_cvd_mortality_diseases_heart['Stratification1']=='Female'].groupby('Year').sum()

df_cvd_2_female.rename(columns={'DataValue':'DV_female'},inplace=True)



df_cvd_2_merged=pd.merge(df_cvd_2_overall,df_cvd_2_male,on='Year', how='inner')

df_cvd_2_merged=pd.merge(df_cvd_2_merged,df_cvd_2_female,on='Year', how='inner')

df_cvd_2_merged=df_cvd_2_merged[['DV_overall','DV_male','DV_female']]

df_cvd_2_merged
x=df_cvd_2_merged.values 

min_max_scaler = preprocessing.StandardScaler()

x_scaled = min_max_scaler.fit_transform(x)

df_cvd_2_merged_norm = pd.DataFrame(x_scaled, columns=df_cvd_2_merged.columns,index=df_cvd_2_merged.index)

df_cvd_2_merged_norm_plt = pd.DataFrame({'Overall': df_cvd_2_merged_norm['DV_overall'],

                    'Male': df_cvd_2_merged_norm['DV_male'],

                       'Female': df_cvd_2_merged_norm['DV_female']}, index= df_cvd_2_merged_norm.index)

ax2 = df_cvd_2_merged_norm_plt.plot.line(rot=0)

ax2.xaxis.set_major_locator(plt.MaxNLocator(4))
df_cvd_4_overall=df_cvd_mortality_cerebrovascular[df_cvd_mortality_cerebrovascular['Stratification1']=='Overall'].groupby('Year').sum()

df_cvd_4_overall.rename(columns={'DataValue':'DV_overall'},inplace=True)

df_cvd_4_male=df_cvd_mortality_cerebrovascular[df_cvd_mortality_cerebrovascular['Stratification1']=='Male'].groupby('Year').sum()

df_cvd_4_male.rename(columns={'DataValue':'DV_male'},inplace=True)

df_cvd_4_female=df_cvd_mortality_cerebrovascular[df_cvd_mortality_cerebrovascular['Stratification1']=='Female'].groupby('Year').sum()

df_cvd_4_female.rename(columns={'DataValue':'DV_female'},inplace=True)



df_cvd_4_merged=pd.merge(df_cvd_4_overall,df_cvd_4_male,on='Year', how='inner')

df_cvd_4_merged=pd.merge(df_cvd_4_merged,df_cvd_4_female,on='Year', how='inner')

df_cvd_4_merged=df_cvd_4_merged[['DV_overall','DV_male','DV_female']]

df_cvd_4_merged
x=df_cvd_4_merged.values 

min_max_scaler = preprocessing.StandardScaler()

x_scaled = min_max_scaler.fit_transform(x)

df_cvd_4_merged_norm = pd.DataFrame(x_scaled, columns=df_cvd_4_merged.columns,index=df_cvd_4_merged.index)

df_cvd_4_merged_norm_plt = pd.DataFrame({'Overall': df_cvd_4_merged_norm['DV_overall'],

                    'Male': df_cvd_4_merged_norm['DV_male'],

                       'Female': df_cvd_4_merged_norm['DV_female']}, index= df_cvd_4_merged_norm.index)

ax4 = df_cvd_4_merged_norm_plt.plot.line(rot=0)

ax4.xaxis.set_major_locator(plt.MaxNLocator(4))
#Let's group the four images in a subplot 

fig,axs =plt.subplots(2,2, figsize=(15,10)) 

axs[0,0].plot(df_cvd_1_merged_norm_plt)

axs[0,0].set_title('Mortality from total cardiovascular diseases')

axs[0,1].plot(df_cvd_2_merged_norm_plt)

axs[0,1].set_title('Mortality from diseases of the heart')

axs[1,0].plot(df_cvd_3_merged_norm_plt)

axs[1,0].set_title('Mortality from heart failure')

axs[1,1].plot(df_cvd_4_merged_norm_plt)

axs[1,1].set_title('Mortality from cerebrovascular disease (stroke)')

for i in [0,1]:

    for j in [0,1]:

        axs[i,j].legend(['Overall','Male','Female'])

        axs[i,j].xaxis.set_major_locator(plt.MaxNLocator(5))

#The most anomaluos trend is mortality from heart failure. So, lets explore per each state

df_cvd_mortality_heart_failure.head()
df_cvd_3ov_2010=df_cvd_mortality_heart_failure[(df_cvd_mortality_heart_failure['Stratification1']=='Overall')&(df_cvd_mortality_heart_failure['Year']==2010)].groupby('LocationAbbr').sum()

df_cvd_3ov_2011=df_cvd_mortality_heart_failure[(df_cvd_mortality_heart_failure['Stratification1']=='Overall')&(df_cvd_mortality_heart_failure['Year']==2011)].groupby('LocationAbbr').sum()

df_cvd_3ov_2012=df_cvd_mortality_heart_failure[(df_cvd_mortality_heart_failure['Stratification1']=='Overall')&(df_cvd_mortality_heart_failure['Year']==2012)].groupby('LocationAbbr').sum()

df_cvd_3ov_2013=df_cvd_mortality_heart_failure[(df_cvd_mortality_heart_failure['Stratification1']=='Overall')&(df_cvd_mortality_heart_failure['Year']==2013)].groupby('LocationAbbr').sum()

df_cvd_3ov_2014=df_cvd_mortality_heart_failure[(df_cvd_mortality_heart_failure['Stratification1']=='Overall')&(df_cvd_mortality_heart_failure['Year']==2014)].groupby('LocationAbbr').sum()
#Depict the cases per 100,000 for each state regarding heart failure and overall stratification criteria

df_cvd_3_overall_year=pd.DataFrame({'2010': df_cvd_3ov_2010['DataValue'],

                    '2011': df_cvd_3ov_2011['DataValue'],

                    '2012': df_cvd_3ov_2012['DataValue'],

                    '2013': df_cvd_3ov_2013['DataValue'],

                    '2014': df_cvd_3ov_2014['DataValue']}, index=df_cvd_3ov_2010.index)

ax1 = df_cvd_3_overall_year.plot.bar(rot=0,figsize=(40,10))
#It's hard to see which state influence in the huge increase of the total figures per year (section 2.3)

#So, lets calculate the linear regression slope from 2010 to 2014 for each state to see if the slope is positive is negative or positive.

from sklearn.linear_model import LinearRegression

lr=LinearRegression()

df_cvd_3_overall_year['Slope']=np.nan

for i in range (df_cvd_3_overall_year.shape[0]):

    #x=[0,1,2,3,4]

    x=df_cvd_3_overall_year.columns[0:4]

    y=df_cvd_3_overall_year.iloc[i,0:4]

    lr.fit(x[:, np.newaxis],y)

    df_cvd_3_overall_year['Slope'].iloc[i]=lr.coef_





#Now we can filter to those state that have a positive slope

df_cvd_3_overall_year_positive=df_cvd_3_overall_year[df_cvd_3_overall_year['Slope']>0]

df_cvd_3_overall_year_positive.describe()

    
#Let's depict the second half of slope values >1.8

df_cvd_3_overall_year_positive_half=df_cvd_3_overall_year_positive[df_cvd_3_overall_year_positive['Slope']>1.79]

print(df_cvd_3_overall_year_positive_half)

df_cvd_3_overall_year_positive_half.drop('Slope',axis=1,inplace=True)

ax1 = df_cvd_3_overall_year_positive_half.plot.bar(rot=0,figsize=(40,10))