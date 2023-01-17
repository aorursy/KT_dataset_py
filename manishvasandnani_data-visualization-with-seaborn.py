import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

%matplotlib inline
dataDf = pd.read_csv('../input/lozpdata-assignment-week01-data-visualizaition/train.csv')
dataDf.head()
dataDf.isnull().sum()
dataDf['Province_State'] =dataDf['Province_State'].fillna('NA')

dataDf.isnull().sum()
dataDf.info()
#### Dropping the records if fatalities and ConfirmedCases is zero 

filteredDf = dataDf.loc[(dataDf.ConfirmedCases > 0) | (dataDf.Fatalities > 0)]
confirmedCf =  dataDf.loc[(dataDf.ConfirmedCases > 0)]

#### How many cases are recorderd all over the world per month

def getMonth(x):

    splitDate = x.split('-')

    if(splitDate[1] == '01'):

        return 'January'

    elif(splitDate[1] == '02'):

        return 'February'

    elif(splitDate[1] == '03'):

        return 'March'

    elif(splitDate[1] == '04'):

        return 'April'

    else:

        return 'NA'

    

confirmedCf['Month_of_Case'] = confirmedCf['Date'].apply(getMonth)



sns.countplot(data = confirmedCf, x='Month_of_Case')

plt.title('Confirmed Cases per Month')

plt.show()
fatalDf =  dataDf.loc[(dataDf.Fatalities > 0)]

fatalDf['Month_of_Death'] = fatalDf['Date'].apply(getMonth)



plt.rc('figure', figsize=(20, 5))

fig, axes =plt.subplots(1,2)



sns.catplot(x= 'Month_of_Death',y = 'Fatalities',data =fatalDf,ax=axes[0] )



plt.title('Number of Fatalities Month Wise')



sns.countplot(y= 'Month_of_Death',data =fatalDf,ax=axes[1])

### Grouping the data country wise

filteredDf_country = filteredDf.groupby(['Country_Region'])

filteredDf_country_Fatal = pd.DataFrame(filteredDf_country.Fatalities.sum())

filteredDf_country_Confirmed = pd.DataFrame(filteredDf_country.ConfirmedCases.sum())

filteredDf_country_Fatal= filteredDf_country_Fatal.reset_index()

filteredDf_country_Confirmed= filteredDf_country_Confirmed.reset_index()

filteredDf_country_Fatal.head()

mergedDf = pd.merge(left=filteredDf_country_Fatal,right=filteredDf_country_Confirmed,how='inner')

mergedDf['Death_Rate'] = mergedDf['Fatalities']/ mergedDf['ConfirmedCases'] *100

mergedDf.head()
plt.rc('figure', figsize=(10, 10))



sns.boxplot(data = mergedDf,y= 'Death_Rate')

plt.show()
merged_filtered = mergedDf.loc[(mergedDf.Death_Rate >=15)]

sns.barplot(merged_filtered.Country_Region, merged_filtered.Death_Rate)

plt.title('Countries with Death Rate more than 15 percent')
##Dividing the Countries based on death rate

def getCondition(x):

    if(x < 2):

        return 'Normal'

    elif(x>  2  and x < 5):

        return 'Mild'

    elif(x>5 and x< 10):

        return 'Critical'

    elif(x>10):

        return 'Severe Emergency'

    

mergedDf['Condition_of_Nation'] =mergedDf['Death_Rate'].apply(getCondition)



plt.pie(mergedDf['Condition_of_Nation'] .value_counts(),labels =mergedDf['Condition_of_Nation'].unique(),colors=['blue','orange','red','green'])

#### Plotting how the confirmed cases increases day on day basis , we will be plotting line graph

confirmedCf['Date'] = pd.to_datetime(confirmedCf['Date'])

firstDate = list(confirmedCf['Date'].sort_values())

firstDate =firstDate[0]

def assignFIrstDay(x):

    

    return firstDate

confirmedCf['firstDateofCase']  ='NA'

confirmedCf['firstDateofCase'] =confirmedCf['firstDateofCase'].apply(assignFIrstDay)

confirmedCf['Number_of_Day'] = confirmedCf['Date']  - confirmedCf['firstDateofCase'] 

confirmedCf['Number_of_Day'] =confirmedCf['Number_of_Day'].astype('string')

confirmedCf['Number_of_Day'] =confirmedCf['Number_of_Day'].str.replace('days 00:00:00.000000000','')

confirmedCf['Number_of_Day'] =confirmedCf['Number_of_Day'].astype('int')

confirmedCf.tail(100)
groupByConfirmCases = confirmedCf.groupby(['Number_of_Day'])

groupByConfirmCases = pd.DataFrame(groupByConfirmCases['ConfirmedCases'].count())



groupByConfirmCases = groupByConfirmCases.reset_index()

groupByConfirmCases1 = groupByConfirmCases[0:10]

groupByConfirmCases2 = groupByConfirmCases[10:20]

groupByConfirmCases3 = groupByConfirmCases[20:30]

groupByConfirmCases4 = groupByConfirmCases[30:40]

groupByConfirmCases5 = groupByConfirmCases[40:50]

groupByConfirmCases6 = groupByConfirmCases[50:60]

groupByConfirmCases7 = groupByConfirmCases[60:70]

groupByConfirmCases8 = groupByConfirmCases[70:80]





plt.figure(1)

plt.subplot(221)

plt.title('First 10 Days')

plt.plot(groupByConfirmCases1.Number_of_Day,groupByConfirmCases1.ConfirmedCases, 'b*')



plt.subplot(222)

plt.title('10 - 20 Days')

plt.plot(groupByConfirmCases2.Number_of_Day,groupByConfirmCases2.ConfirmedCases, 'b*')

plt.figure(2)

plt.subplot(221)

plt.title('20 - 30 Days')

plt.plot(groupByConfirmCases3.Number_of_Day,groupByConfirmCases3.ConfirmedCases, 'b*')

plt.subplot(222)

plt.title('30 - 40 Days')

plt.plot(groupByConfirmCases4.Number_of_Day,groupByConfirmCases4.ConfirmedCases, 'b*')



plt.figure(3)

plt.subplot(221)

plt.title('40 - 50 Days')

plt.plot(groupByConfirmCases5.Number_of_Day,groupByConfirmCases5.ConfirmedCases, 'b*')



plt.subplot(222)

plt.title('50 - 60 Days')

plt.plot(groupByConfirmCases6.Number_of_Day,groupByConfirmCases6.ConfirmedCases, 'b*')

plt.figure(4)

plt.subplot(221)

plt.title('60 - 70 Days')

plt.plot(groupByConfirmCases7.Number_of_Day,groupByConfirmCases7.ConfirmedCases, 'b*')

plt.subplot(222)

plt.title('70 - 80 Days')

plt.plot(groupByConfirmCases8.Number_of_Day,groupByConfirmCases8.ConfirmedCases, 'b*')

groupByConfirmCases1 = groupByConfirmCases[0:20]

groupByConfirmCases2 = groupByConfirmCases[20:40]

groupByConfirmCases3 = groupByConfirmCases[40:60]

groupByConfirmCases4 = groupByConfirmCases[60:80]

plt.figure(1)

plt.subplot(221)

plt.title('First 20 Days')

plt.plot(groupByConfirmCases1.Number_of_Day,groupByConfirmCases1.ConfirmedCases, 'b*')



plt.subplot(222)

plt.title('20 - 40 Days')

plt.plot(groupByConfirmCases2.Number_of_Day,groupByConfirmCases2.ConfirmedCases, 'b*')

plt.figure(2)

plt.subplot(221)

plt.title('40 - 60 Days')

plt.plot(groupByConfirmCases3.Number_of_Day,groupByConfirmCases3.ConfirmedCases, 'b*')

plt.subplot(222)

plt.title('60 - 80 Days')

plt.plot(groupByConfirmCases4.Number_of_Day,groupByConfirmCases4.ConfirmedCases, 'b*')

groupByConfirmCases1.columns