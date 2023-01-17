# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import math as math
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data2 = pd.read_csv('../input/data2_modified.csv')
Coefficient = pd.read_csv('../input/Coefficient_modified.csv')
Coefficient
data2.head(5)
def look_for_coef(tag,content,coef):
    try:
        tag.replace(content,coef.loc[0],inplace=True,regex = True)
    except:
        tag.replace(content,0,inplace=True,regex = True)
    return data2

#Replace values with each coefficient
#ServiceType
data2.ServiceType.replace('C',Coefficient.ServiceTypeC.loc[0],inplace=True,regex = True)
data2.ServiceType.replace('E',Coefficient.ServiceTypeE.loc[0],inplace=True,regex = True)
data2.ServiceType.replace('H',Coefficient.ServiceTypeH.loc[0],inplace=True,regex = True)
data2.ServiceType.replace('I',Coefficient.ServiceTypeI.loc[0],inplace=True,regex = True)
data2.ServiceType.replace('M',Coefficient.ServiceTypeM.loc[0],inplace=True,regex = True)
data2.ServiceType.replace('N',Coefficient.ServiceTypeN.loc[0],inplace=True,regex = True)
data2.ServiceType.replace('S',Coefficient.ServiceTypeS.loc[0],inplace=True,regex = True)
data2.ServiceType.replace('B',0,inplace=True,regex = True)
#Age
for x in range(0,100):
    if 27 <x<=32:
        data2.Age.replace(x,Coefficient.Age_largerthan27_smallerthanorequal32.loc[0],inplace=True)
    if 32 <x<=39:
        data2.Age.replace(x,Coefficient.Age_largerthan32_smallerthanorequal39.loc[0],inplace=True)
    if 39<x:
        data2.Age.replace(x,Coefficient.Age_largerthan39.loc[0],inplace=True)
    else:
        data2.Age.replace(x,0,inplace=True)
data2.Age.head(3)
Coefficient.NewCustomer_Y
#Credit
for x in range(0, 100000,1000):
    if 55000 <x<=65000:
        data2.Credit.replace(x,Coefficient.Credit_largerthan55000_smallerthanorequal65000.loc[0],inplace=True)
    if 65000 <x<=72000:
        data2.Credit.replace(x,Coefficient.Credit_largerthan65000_smallerthanorequal72000.loc[0],inplace=True)
    if 72000 <x:
        data2.Credit.replace(x,Coefficient.Credit_largerthan72000.loc[0],inplace=True)
    else:
        data2.Credit.replace(x,0,inplace=True)
for x in range(15000, 41000,10):
        data2.Credit.replace(x,0,inplace=True)
#Government
data2.Government.replace('Y',Coefficient.Government_Y.loc[0],inplace=True,regex = True)
data2.Government.replace('N',0,inplace=True,regex = True)
#Market
data2.Market.replace('MountainPlains',Coefficient.MarketMountainPlains.loc[0],inplace=True,regex = True)
data2.Market.replace('Western',Coefficient.MarketWestern.loc[0],inplace=True,regex = True)
data2.Market.replace('Midwest',Coefficient.MarketMidwest.loc[0],inplace=True,regex = True)
data2.Market.replace('Southwest',Coefficient.MarketSouthwest.loc[0],inplace=True,regex = True)
data2.Market.replace('Southeast',Coefficient.MarketSoutheast.loc[0],inplace=True,regex = True)
data2.Market.replace('Northeast',Coefficient.MarketNortheast.loc[0],inplace=True,regex = True)
data2.Market.replace('Internet',0,inplace=True,regex = True)
#NewCustomer
data2.NewCustomer.replace('Y',Coefficient.MarketSoutheast.loc[0],inplace=True,regex = True)
data2.NewCustomer.replace('N',0,inplace=True,regex = True)
#PaymentMethod
data2.PaymentMethod.replace('Direct',Coefficient.PaymentMethod_Direct.loc[0],inplace=True,regex = True)
data2.PaymentMethod.replace('Other',Coefficient.PaymentMethod_Other.loc[0],inplace=True,regex = True)
data2.PaymentMethod.replace('Goverment',Coefficient.PaymentMethod_Gov.loc[0],inplace=True,regex = True)
data2.PaymentMethod.replace('Defer',0,inplace=True,regex = True)

#Gender
data2.Gender.replace('F',Coefficient.Gender_F.loc[0],inplace=True,regex = True)
data2.Gender.replace('M',Coefficient.Gender_M.loc[0],inplace=True,regex = True)

#Dependents
data2.Dependents.replace('No',Coefficient.Dependents_No.loc[0],inplace=True,regex = True)
data2.Dependents.replace('Yes',Coefficient.Dependents_Yes.loc[0],inplace=True,regex = True)
data2.Dependents.replace('Blank',0,inplace=True,regex = True)

#MaritalStatus
data2.MaritalStatus.replace('Married',Coefficient.MaritalStatus_Married.loc[0],inplace=True,regex = True)
data2.MaritalStatus.replace('Separ',Coefficient.MaritalStatus_Separ.loc[0],inplace=True,regex = True)
data2.MaritalStatus.replace('Single',Coefficient.MaritalStatus_Single.loc[0],inplace=True,regex = True)
#Classification1
data2.Classification1.replace(1,Coefficient.Classification1_1.loc[0],inplace=True,regex = True)
data2.Classification1.replace(2,Coefficient.Classification1_2.loc[0],inplace=True,regex = True)
data2.Classification1.replace(3,Coefficient.Classification1_3.loc[0],inplace=True,regex = True)
data2.Classification1.replace(4,Coefficient.Classification1_4.loc[0],inplace=True,regex = True)

#Classification2
data2.Classification2.replace(1,Coefficient.Classification2_1.loc[0],inplace=True,regex = True)
data2.Classification2.replace(2,Coefficient.Classification2_2.loc[0],inplace=True,regex = True)
data2.Classification2.replace(2,Coefficient.Classification2_3.loc[0],inplace=True,regex = True)
data2.Classification2.replace(4,Coefficient.Classification2_4.loc[0],inplace=True,regex = True)
#AnnualIncome
for x in range(0, 200000):
    if 32004 <x<=62227:
        data2.AnnualIncome.replace(x,0.59153,inplace=True)
    if 62227 <x<=130624:
        data2.AnnualIncome.replace(x,Coefficient.AnnualIncome_largerthan62227_smallerthanorequa130624.loc[0],inplace=True)
    if 72000 <x:
        data2.AnnualIncome.replace(x,Coefficient.AnnualIncome_largerthan130624.loc[0],inplace=True)
    else:
        data2.AnnualIncome.replace(x,0,inplace=True)
data2.AnnualIncome    

#Sum of all coefficients
data2.drop(['CustID','WeeksWithService','ServiceStatus'], axis=1, inplace=True)
data2.sum(axis =  1, skipna = True)
data2
Sum_of_coeffcients = pd.DataFrame({'Sum':data2.sum(axis =  1, skipna = True)})
Sum_of_coeffcients
Sum_of_coeffcients.to_csv('Sum_of_coeffcients.csv',index=False)
