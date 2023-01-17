!pip install scipy

!pip install sklearn

!pip install seaborn
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import scipy.stats as stats

import datetime

from sklearn import linear_model

from sklearn.metrics import mean_squared_error,r2_score

import seaborn as sns

import time

from pandas.core.common import SettingWithCopyWarning

import warnings

warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)

from sklearn.model_selection import train_test_split

sns.set()


def LoadData(FileSavePath,sheetname):

    df = pd.read_excel(FileSavePath, sheet_name=sheetname)

    df.dropna(how='all', inplace=True)

    df.reset_index(drop = True,inplace = True)



    Index_Yield = int(df.loc[df[df.columns[0]] == 'Date'].index[0])

    DataInformation = df.iloc[Index_Yield:, :2]

    DataInformation.columns = DataInformation.iloc[0]

    DataInformation = DataInformation[1:]

    if sheetname == 'UST1YT Bill 02-Jan-2020 | Thom':

        DataInformation = DataInformation.append({'Date':datetime.datetime.strptime('2001/12/01',"%Y/%m/%d"),

                                                  'Bid Yield':2.08},ignore_index=True)

        DataInformation = DataInformation.append({'Date':datetime.datetime.strptime('2002/12/01',"%Y/%m/%d"),

                                                  'Bid Yield':1.56},ignore_index=True)

        DataInformation = DataInformation.append({'Date':datetime.datetime.strptime('2007/12/01',"%Y/%m/%d"),

                                                  'Bid Yield':3.15},ignore_index=True)

        DataInformation = DataInformation.append({'Date': datetime.datetime.strptime('2012/12/01',"%Y/%m/%d"),

                                                  'Bid Yield': 0.17},ignore_index=True)

        DataInformation = DataInformation.append({'Date': datetime.datetime.strptime('2013/12/01',"%Y/%m/%d"),

                                                  'Bid Yield': 0.122},ignore_index=True)

        DataInformation = DataInformation.append({'Date': datetime.datetime.strptime('2018/12/01',"%Y/%m/%d"),

                                                  'Bid Yield': 2.707},ignore_index=True)



    DataInformation = DataInformation.sort_values(by = 'Date')

    DataInformation.reset_index(drop=True, inplace=True)

    DataInformation.index = DataInformation['Date']



    print(DataInformation.head())

    return DataInformation

def LoadFRED(FileSavePath):



    df = pd.read_excel(FileSavePath)

    df.dropna(how='all', inplace=True)

    df.reset_index(drop = True,inplace = True)



    Index_Yield = int(df.loc[df[df.columns[0]] == 'Frequency: Monthly'].index[0])

    DataInformation = df.iloc[Index_Yield:, :2]

    DataInformation.columns = DataInformation.iloc[1]

    DataInformation = DataInformation[2:]

    DataInformation = DataInformation.sort_values(by='observation_date')

    DataInformation.reset_index(drop=True, inplace=True)

    DataInformation.index = DataInformation['observation_date']

    print(DataInformation.head())

    return DataInformation
def LoadInflation(FileSavePath):



    df = pd.read_excel(FileSavePath)

    df.dropna(how='all', inplace=True)

    df.reset_index(drop = True,inplace = True)



    Index_Yield = int(df.loc[df[df.columns[0]] == 'date'].index[0])

    DataInformation = df.iloc[Index_Yield:, :2]

    DataInformation.columns = DataInformation.iloc[0]

    DataInformation = DataInformation[1:]

    DataInformation = DataInformation.sort_values(by='date')

    DataInformation.reset_index(drop=True, inplace=True)

    DataInformation.index = DataInformation['date']

    DataInformation.rename(columns = {DataInformation.columns[1]:'Inflation Rate'},inplace = True)

    print(DataInformation.head())

    return DataInformation
def PearsonAtSpecificPeriod(DataInformation,Start_Time,End_Time,X1Name , X2Name):

    DataInformation['Date'] = pd.to_datetime(DataInformation.index)

    DataDiff_Rising_Period = DataInformation.loc[(DataInformation['Date'] >= Start_Time) & (DataInformation['Date'] <= End_Time)]

    DataDiff_Rising_Period[[X1Name, X2Name]].plot()

    plt.grid(True)

    plt.ylabel('Yield(%)')

    plt.xlabel('Date')

    plt.show()



    Info = ''.join(['The correlation between ', str(X1Name), ' and ', str(X2Name), ' at the given period is '])

    print('\nFrom', Start_Time.strftime('%Y-%m-%d'), 'to', End_Time.strftime('%Y-%m-%d'))

    print(Info,'%.2f' %stats.pearsonr(DataDiff_Rising_Period[X1Name], DataDiff_Rising_Period[X2Name])[0])

    print('The p-value is %.8f' % (

    stats.pearsonr(DataDiff_Rising_Period[X1Name], DataDiff_Rising_Period[X2Name])[1]))

    GetLinearRegression(DataDiff_Rising_Period,X1Name = X1Name,X2Name = X2Name)

    return DataDiff_Rising_Period

def MultipleLinearRegression(DataInformation,Start_Time,End_Time,X1Name ,X2Name,YName):

    DataInformation['Date'] = pd.to_datetime(DataInformation.index)

    DataDiff_Rising_Period = DataInformation.loc[(DataInformation['Date'] >= Start_Time) & (DataInformation['Date'] <= End_Time)]

    Multiple_Linear_Regression_Data = np.c_[DataDiff_Rising_Period[X1Name],DataDiff_Rising_Period[X2Name]]

    Multiple_Linear_Regression_Data_Y = np.c_[DataDiff_Rising_Period[YName]]



    X_Train,X_Test,Y_Tain,Y_Test = train_test_split(Multiple_Linear_Regression_Data,Multiple_Linear_Regression_Data_Y,train_size= 0.8,random_state=9)

    model = linear_model.LinearRegression()

    model.fit(X_Train,Y_Tain)

    Intercept = model.intercept_

    Coefficient = model.coef_



    print('\nFrom', Start_Time.strftime('%Y-%m-%d'), 'to', End_Time.strftime('%Y-%m-%d'))

    print('Multiple Linear Regression : ',str(YName),' = %.4f'%Intercept[0],' + %.4f'%Coefficient[0][0],'*',str(X1Name), '+ %.4f'%Coefficient[0][1],'*',str(X2Name))

    Y_Predict = model.predict(X_Test)

    print('Mean Squared Error:%.2f' % mean_squared_error(Y_Test, Y_Predict))

    print('Coefficient of determination: %.2f' % r2_score(Y_Test, Y_Predict))
def GetLinearRegression(DataInformation,X1Name,X2Name):

    Data_X, Data_Y = np.array([DataInformation[X1Name]]).T,np.array([DataInformation[X2Name]]).T

    regr = linear_model.LinearRegression()

    regr.fit(Data_X, Data_Y)

    Data_Y_Predict = regr.predict(Data_X)



    print('Linear Regression : Y = %.2f'%regr.coef_[0][0],'* X + %.2f'%regr.intercept_[0])

    print('Mean Squared Error:%.2f' % mean_squared_error(Data_Y, Data_Y_Predict))

    print('Coefficient of determination: %.2f' % r2_score(Data_Y, Data_Y_Predict))



    plt.scatter(Data_X, Data_Y)

    plt.plot(Data_X, Data_Y_Predict, linewidth=3)

    Xlabel_Name = ''.join([str(X1Name),r'(%)'])

    Ylabel_Name = ''.join([str(X2Name), r'(%)'])

    plt.xlabel(Xlabel_Name)

    plt.ylabel(Ylabel_Name)



    Title_Name = '' .join(['From ',str(DataInformation.index[0].strftime('%Y-%m-%d')),' to ',str(DataInformation.index[-1].strftime('%Y-%m-%d'))])

    plt.title(Title_Name)

    plt.grid(True)

    plt.show()
# Loading

FileSavePath = r'../input/Datathon_Treasury interest rates.xlsx'

FederalFundRateSavePath = r'../input/FEDFUNDS.xlsx'

HQMCBSavePath = r'../input/HQMCB10YR.xlsx'

AAASavePath = r'../input/AAA.xlsx'

BAASavePath = r'../input/BAA.xlsx'

InflationRate = r'../input/united-states-inflation-rate-cpi.xlsx'
DataInformation_US = LoadData(FileSavePath,'UST1YT Bill 02-Jan-2020 | Thom')

DataInformation_HK = LoadData(FileSavePath,'HKGV 22-Jan-2020 | Thom')

DataInformation_CN = LoadData(FileSavePath,'CNGV 2.4100 22-Nov-2019 | Tho')



DataInformation_IR = LoadFRED(FederalFundRateSavePath)

DataInformation_HQM = LoadFRED(HQMCBSavePath)

DataInformation_AAA = LoadFRED(AAASavePath)

DataInformation_BAA = LoadFRED(BAASavePath)



DataInformation_Inflation = LoadInflation(InflationRate)

DataInformation_US['Bid Yield'].plot()

DataInformation_HK['Bid Yield'].plot()

DataInformation_CN['Bid Yield'].plot()

DataInformation_IR['FEDFUNDS'].plot()

DataInformation_Inflation['Inflation Rate'].plot()



plt.legend(['US Yield','HK Yield','CN Yield','Fed Interest Rate','Inflation Rate'])

#plt.grid(True)

plt.title('Interest Rate & Fixed-Income Market')

plt.show()
DataInformation_HQM[DataInformation_HQM.columns[1]].plot()

DataInformation_AAA[DataInformation_AAA.columns[1]].plot()

DataInformation_BAA[DataInformation_BAA.columns[1]].plot()

plt.legend(['10Y High Quality Market Corporate Bond', 'Moody"s Seasoned Aaa Corporate Bond', 'Moody"s Seasoned BAA Corporate Bond'])

#plt.grid(True)

plt.title('Corporate Bond Tendency')

plt.show()
DataDiff = DataInformation_HQM[DataInformation_HQM.columns[1]] - DataInformation_US['Bid Yield']

DataDiff.dropna(how='all', inplace=True)

DataDiff = pd.DataFrame(DataDiff,columns=['Spread'])
fig = plt.figure()

ax1 = fig.add_subplot(111)

ax1.plot(DataInformation_US.index,DataInformation_US['Bid Yield'])

ax1.plot(DataInformation_HQM.index,DataInformation_HQM[DataInformation_HQM.columns[1]])

ax1.plot(DataInformation_Inflation.index,DataInformation_Inflation['Inflation Rate'])

ax1.set_ylabel('Yield Curve(%)')

plt.legend(['US Yield', '10Y High Quality Market Corporate Bond','Inflation Rate'])

ax2 = ax1.twinx()

ax2.plot(DataDiff.index,DataDiff['Spread'],'r')

ax2.set_ylabel('Spread')

plt.legend(['Spread'])

plt.grid(True)

plt.show()
DataDiff['US Yield'] = DataInformation_US['Bid Yield']

DataDiff['HQMCB'] = DataInformation_HQM[DataInformation_HQM.columns[1]]





DataDiff['Inflation Rate'] = DataInformation_Inflation['Inflation Rate']

DataDiff.reset_index(inplace=True)

Indexes = DataDiff[~(DataDiff['Inflation Rate'].isnull().T)].index.tolist()

for i in range(len(Indexes)-1):

    Ranges = np.arange(Indexes[i]+1,Indexes[i+1],1)

    for j in Ranges:

        k =  (DataDiff['Inflation Rate'][Indexes[i+1]] - DataDiff['Inflation Rate'][Indexes[i]])/(Indexes[i+1]-Indexes[i])

        b = DataDiff['Inflation Rate'][Indexes[i+1]] - k * Indexes[i+1]

        DataDiff['Inflation Rate'][j] = k * j + b



DataDiff.index = DataDiff['index']

print(DataDiff.head())
DataDiff['IR+USY'] = DataDiff['Inflation Rate'] + DataDiff['US Yield']

plt.scatter(DataDiff['US Yield'],DataDiff['HQMCB'])

plt.xlabel('US Yield (%)')

plt.ylabel('10Y High Quality Market Corporate Bond(%)')

plt.show()
print('From ',DataDiff.index[0].strftime('%Y-%m-%d'),' to ',DataDiff.index[-1].strftime('%Y-%m-%d'))

print('The correlation between US Yield and HQMCB at the given period is %.2f'%stats.pearsonr(DataDiff['US Yield'],DataDiff['HQMCB'])[0])

print('The p-value is %.8f'%(stats.pearsonr(DataDiff['US Yield'],DataDiff['HQMCB'])[1]))



print('From ', DataDiff.index[0].strftime('%Y-%m-%d'), ' to ', DataDiff.index[-1].strftime('%Y-%m-%d'))

print('The correlation between US Yield and Inflation Rate at the given period is %.2f' %

          stats.pearsonr(DataDiff['US Yield'], DataDiff['Inflation Rate'])[0])

print('The p-value is %.8f' % (stats.pearsonr(DataDiff['US Yield'], DataDiff['Inflation Rate'])[1]))



print('From ', DataDiff.index[0].strftime('%Y-%m-%d'), ' to ', DataDiff.index[-1].strftime('%Y-%m-%d'))

print('The correlation between Inflation Rate and HQMCB at the given period is %.2f' %

          stats.pearsonr(DataDiff['HQMCB'], DataDiff['Inflation Rate'])[0])

print('The p-value is %.8f' % (stats.pearsonr(DataDiff['HQMCB'], DataDiff['Inflation Rate'])[1]))



print('From ', DataDiff.index[0].strftime('%Y-%m-%d'), ' to ', DataDiff.index[-1].strftime('%Y-%m-%d'))

print('The correlation between IR+USY and HQMCB at the given period is %.2f' %

          stats.pearsonr(DataDiff['HQMCB'], DataDiff['IR+USY'])[0])

print('The p-value is %.8f' % (stats.pearsonr(DataDiff['HQMCB'], DataDiff['IR+USY'])[1]))
Start_Time = datetime.datetime(2004,1,1)

End_Time = datetime.datetime(2007,8,31)

DataDiff_Rising_Period_1 = PearsonAtSpecificPeriod(DataDiff,Start_Time,End_Time,X1Name='US Yield',X2Name='HQMCB')
Start_Time2 = datetime.datetime(2015, 8, 1)

End_Time2 = DataDiff.index[-1]

DataDiff_Rising_Period_2 = PearsonAtSpecificPeriod(DataDiff,Start_Time2,End_Time2,X1Name='US Yield',X2Name='HQMCB')

Start_Time = datetime.datetime(2004,1,1)

End_Time = datetime.datetime(2007,8,31)

DataDiff_Rising_Period_1 = PearsonAtSpecificPeriod(DataDiff,Start_Time,End_Time,X1Name='Inflation Rate',X2Name='HQMCB')
Start_Time2 = datetime.datetime(2015, 8, 1)

End_Time2 = DataDiff.index[-1]

DataDiff_Rising_Period_2 = PearsonAtSpecificPeriod(DataDiff,Start_Time2,End_Time2,X1Name='Inflation Rate',X2Name='HQMCB')
Start_Time = datetime.datetime(2004,1,1)

End_Time = datetime.datetime(2007,8,31)

DataDiff_Rising_Period_1 = PearsonAtSpecificPeriod(DataDiff,Start_Time,End_Time,X1Name='IR+USY',X2Name='HQMCB')
Start_Time2 = datetime.datetime(2015, 8, 1)

End_Time2 = DataDiff.index[-1]

DataDiff_Rising_Period_2 = PearsonAtSpecificPeriod(DataDiff,Start_Time2,End_Time2,X1Name='IR+USY',X2Name='HQMCB')
MultipleLinearRegression(DataDiff, DataDiff.index[0], DataDiff.index[-1], X1Name='Inflation Rate', X2Name='US Yield', YName='HQMCB')
MultipleLinearRegression(DataDiff,Start_Time,End_Time,X1Name='Inflation Rate',X2Name='US Yield',YName='HQMCB')
MultipleLinearRegression(DataDiff, Start_Time2, End_Time2, X1Name='Inflation Rate', X2Name='US Yield', YName='HQMCB')