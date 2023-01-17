#                                   Program:    Vix Interpolation Calculation

#                                        By:    Charles Lewson

#

#----------------------------------------------------------------------------------

#                                                   Import ToolBox

import datetime

from datetime import date

import pandas as pd

import numpy as np

import scipy as si

import scipy.stats

from scipy.stats import norm

import matplotlib.pyplot as plt

from scipy.optimize import curve_fit

#---------------------------------------------------------------------------------
df = pd.read_csv('../input/sortedoptiondata2/SortedOptionData.csv')

df.head(7)
# Check the dtype of StockDate and Exp (these should be date/time format)

df.info() 
# Create another DataFrame that contains only PCG

pcg = df.loc[df['Name'] == 'PCG']

pcg.head()
pcg.info()
#Assign our Scalar values (hopefuly one day these are passed in by your user through a fancy ui)

#                       Scalar Inputs

RiskFreeRate = [0.03]

#3% RiskFreeRate (somewhere between 0.01% and 10% is usually acceptable)

RiskFreeRate = np.array(RiskFreeRate)



InterpoRange = [30,40,50,60,70,80,90,120,150,180,210,240]

#InterpoRange wont be used until much further down

#       (I just like keeping my scalar inputs together to make it easier when we eventually recode for input from ui)

InterpoRange = np.array(InterpoRange)
# Create a copy to avoid the SettingWarning .loc issue

pcg_df = pcg.copy()

# Change to datetime datatype.

pcg_df.loc[:, 'StockDate'] = pd.to_datetime(pcg.loc[:,'StockDate'], format="%m/%d/%Y")



#Repeat for Exp - (I purposely changed the original data to have a different format type)

pcg_df.loc[:, 'Exp'] = pd.to_datetime(pcg.loc[:,'Exp'], format="%d-%b-%y")
pcg_df.info()
#Create a numpy array from Pandas df selecting 

Tte = TimeToExpire = pcg_df['TimeToExpire'].to_numpy(copy=True)

print('Time_To_Expire')

print(Tte)

#Find the Unique Values in the numpy array

UniTte = UniqueTimeToExpire = np.unique(Tte)

print('Unique_Time')

print(UniTte)

#Find the length of Unique Values as to itterate over the array

L1 = Length1 = len(UniTte)

print('Length_Of_Tte')

print(L1)

#Convert Tte to minutes

#TteMinutes = TimeToExpire

#L0 = len(TimeToExpire)

#I label this one L0 because I will use it in Final For Loop

#for i in range(0,L0):

#    TteMinutes[i] = (Tte[i] *(24*60))

#print(TteMinutes)
#Create an array to fill with the calculation

UniTteMinutes = np.linspace(0,0,L1)

print('Unique_Tte_Array')

print(UniTteMinutes)

#Calculate T for each unique TimeToExpire

for i in range(0,L1): 

    UniTteMinutes[i] = ((480+570+(UniTte[i]*1440))/(525600))

print('Unique_Tte_As_Minutes/Year_As_Minutes')    

print(UniTteMinutes)
#Create a numpy array from Pandas df selecting 'Diff'

Diff = Difference = pcg_df['Diff'].to_numpy(copy=True)

print('Difference')

print(Diff)

#Find Positional Information with regards to Unique TimeToExpire

UniTteLengths = np.linspace(0,0,L1)

for i in range(0,L1):

    UniTteLengths[i] = np.digitize(UniTte[i],Tte)

print('Unique_Tte_Position')

print(UniTteLengths)
#Create Positional Information to ensure proper iteration over dataset for input

p1 = np.array([0])

p1 = p1.astype(int)

p2 = UniTteLengths[0]

p2 = p2.astype(int)

p2 = np.array([p2])

print('Position_One')

print(p1)

print('Position_Two')

print(p2)

# We now  know that position 2 is where our first unique TimeToExpire ends(since python begins at 0) 16-1 = 15

print('Tte_at_Postion[15]')

print(Tte[15])

print('Diff@[p1:p2]')

print(Diff[p1[0]: p2[0]])
#Find Minium # in Diff column as it relates to above Positional Information

#        Showing All steps(compact code at end)

DiffPosition = Diff[p1[0]:p2[0]]

print('Diff_Position')

print(DiffPosition)

print('Min_Diff')

MinDiff = min(DiffPosition)

print(MinDiff)

#

#

MinDiffCompact=min(Diff[p1[0]:p2[0]])

print('Min_Diff_Compact')

print(MinDiffCompact)
#Find Foward Price index (labeling F1 same as White Paper)

TtePos = Tte[p1[0]:p2[0]]

print('Tte1@[p1:p2]')

#if you look below all numbers in this array should be the same (dealing with a grouping of options with same TimeToExpire {44 days left})

print(TtePos)

#creating a numpy array from our df using Strike

Strike = pcg_df['Strike'].to_numpy(copy=True)

StrikePos = Strike[p1[0]:p2[0]]

print('StrikePos@[p1:p2]')

print(StrikePos)

L2 = len(StrikePos)

LTest = len(TtePos)

#showing Strike1 and Tte1 are equal lengths for iterative process later

print('Length_Test_Check')

print(L2,LTest)

#

#Find the Strike whos position corresponds to the position where MinDiff is Equal to(==) DiffPosition

for i in range(0,L2):

    if (DiffPosition[i] == MinDiff):

            StrikeDiff = StrikePos[i]

print('StrikeDiff')

print(StrikeDiff)
#Find what e1 ='s

#     (only seperating this out for demonstration purposes, though if you were to use it elsewhere would be nice to have as a standalone variable)

e1 = np.exp(RiskFreeRate[0]*UniTteMinutes[0])

print('e^(R*T)')

print(e1)



#Final we can find what F1 ='s

F1 = (StrikeDiff) + ((e1) * (MinDiff))

print('F1')

print(F1)

#index trick here to find Ko,1

index = (np.abs(StrikePos-F1)).argmin()

print('This number is our position in Strike1 where Ko is')

print(index)

#ko,1 and F1 arent always the same (look at Vix White Paper for example)

k0 = (StrikePos[index])

print('Ko,1')

print(k0)
StrikeChange = np.linspace(0,0,L2)

for i in range(0,1):

    if (StrikeChange[0] == StrikeChange[0]) or (StrikeChange[-1] == StrikeChange[-1]) :

        StrikeChange[0] = abs(StrikePos[1] - StrikePos[0])

        StrikeChange[-1] = abs(StrikePos[-1] - StrikePos[-2])

print('First/Last_Position_In_StrikeChange')

print(StrikeChange)

#range below starts at second position and ends on second to last position

for ii in range(1,(L2-1)):

    StrikeChange[ii] = (abs(StrikePos[ii-1] - StrikePos[ii+1])/2)

        #because of how python iterates top down and the confusing wording by the Vix the above (i-1) - (i+1) is correct

print('Everything_Inbetween_In_StrikeChange')

print(StrikeChange)

#Find Mark by combining MarkC and MarkP over proper interval

MarkC = pcg_df['MarkC'].to_numpy(copy=True)

MarkP = pcg_df['MarkP'].to_numpy(copy=True)

#

#Also need StockPrice for iteration

StockPrice = pcg_df['StockPrice'].to_numpy(copy=True)

#print(Strike)

#print(StockPrice)

L3 = len(StockPrice)

Mark = np.linspace(0,0,L3)

#print(L3)

#print(Mark)

for i in range(0,L3):

        if Strike[i] >= StockPrice[i]:

            Mark[i] = MarkC[i]

        elif Strike[i] <= StockPrice[i]:

            Mark[i] = MarkP[i]

print('Mark')

print(Mark)

MarkPos = Mark[p1[0]:(p2[0])]

print('MarkPos@[p1:p2]')

print(MarkPos)
#Find Step4

Step4 = np.linspace(0,0,L2)

for i in range(0,L2):

    Step4[i] = ((StrikeChange[i] / (StrikePos[i]**2)) * (e1) * (MarkPos[i]))

print('Step4')

print(Step4)

#I seperate out the Cstrike for demonstrations purposes (makes the calculus look less daunting)

Cstrike = sum(Step4)

print('Cstrike')

print(Cstrike)
#Only need to solve this once per Unique_TimeToExpire

Step5 = ((1/UniTteMinutes[0])*(((F1/k0)-1)**2))

print('Step5')

print(Step5)
#Solve for Sigma then square for Volatility

Sigma = (2/UniTteMinutes[0])*((Cstrike)-(Step5))

print('Sigma')

print(Sigma)

Volatility = (Sigma**2)

print('Volatility')

print(Volatility)

Vol = Volatility*100

print('Volatility_As_Percentage')

print(Vol)

#PG&E and the Nevada County District Attorneys have extended their agreement, previously entered into as of October 3, 2018,

#under which PG&E agreed to waive any applicable statutes of limitation related to the October 2017 fires that started in Nevada County for a period of six months

#Remeber this is the Volatility for 44 days calculated from the end of option data on 10/3/2018
#Create target Variable-Arrays 

VixVol = np.linspace(0,0,L1)

VixSigma = np.linspace(0,0,L1)

#I changed this to p3,p4 just so we are never overwriting the above lines

p3 = np.array([0])

p3 = p3.astype(int)

p4 = UniTteLengths[0]

p4 = p4.astype(int)

p4 = np.array([p4])

print(UniTteLengths)



for i in range(0,L1):

    if i > 0:

        p3 = p4

        p4 = UniTteLengths[i]

        p4 = p4.astype(int)

        p4 = np.array([p4])

            

    MinDiffCompact = min(Diff[p3[0]:(p4[0])])

    Strike1 = Strike[p3[0]:(p4[0])]

    DiffPosition = Diff[p3[0]:(p4[0])]

    Mark1 = Mark[p3[0]:(p4[0])]

    

    L2 = len(Strike1)

    print('Option_Row_Count_Per_Iteration')

    print(L2)

    for q in range(0,L2):

        if (DiffPosition[q] == MinDiffCompact):

            StrikeDiff = Strike1[q]

    

    e1 = np.exp((RiskFreeRate[0])*(UniTteMinutes[i]))

    F1 = (StrikeDiff) + ((e1) * (MinDiffCompact))

    

    index = (np.abs(Strike1-F1)).argmin()

    k0 = (Strike1[index])

   

    Tstep = ((1/UniTteMinutes[i])*(((F1/k0)-1)**2))

    

    StrikeChange = np.linspace(0,0,L2)

    for j in range(0,1):

        if (StrikeChange[0] == StrikeChange[0]) or (StrikeChange[-1] == StrikeChange[-1]) :

            StrikeChange[0] = abs(Strike1[1] - Strike1[0])

            StrikeChange[-1] = abs(Strike1[-1] - Strike1 [-2])

        for jj in range(1,(L2-1)):

            StrikeChange[jj] = (abs(Strike1[jj-1] - Strike1[jj+1])/2)

    

    Step4 = np.linspace(0,0,L2)

    for q in range(0,L2):

        Step4[q] = ((StrikeChange[q] / (Strike1[q]**2)) * (e1) * (Mark1[q]))

    Cstrike = sum(Step4)

    

    #I left these prints in as a way to see how I check per iteration if the calculation is working

    #print(Cstrike)

    #print(Tstep)

    #print(UniTteMinutes[i])

    

    Sigma = (2/UniTteMinutes[i])*((Cstrike)-(Tstep))

    VixSigma[i] = (Sigma**2)

    VixVol[i] = ((Sigma**2) *100)

    print('VixSigma_VixVol')

    print(VixSigma)

    print(VixVol)
#Convert InterpoRange Input from days to MinutesPerDay

#Create Logic InterpoRange array - this checks where one can create an interpolation (cant interpolate over inputs where you have no orginial data)

L3 = len(InterpoRange)

InterpoRangeMpD = np.linspace(0,0,L3)

LogicInter = np.linspace(0,0,L3)

for q in range(0,L3):

    InterpoRangeMpD[q] = ((480 + 570 + (InterpoRange[q]*(1440))))

    if (UniTte[0] <= InterpoRange[q]) and (UniTte[-1] >= InterpoRange[q]):

        LogicInter[q] = 1

#print('InterpoRangeMpD', InterpoRangeMpD)

print('Interpolation_Range_Minutes_Per_Day')

print(InterpoRangeMpD)

print('Logic_Interpolation',LogicInter)
#Create an array containing Unique TimeToExpire as Minutes Per Day

UniTteMpD = np.linspace(0,0,L1)

for i in range(0,L1):

    UniTteMpD[i] = ((480 + 570 + (UniTte[i]*(1440))))

print('Unique_TimeToExpire_Minutes_Per_Day')

print(UniTteMpD)
#Generate Interpolations where LogicInterpoRange ='s 1

VixVarInterpo = np.linspace(0,0,L3)

for b in range(0,L3):       

    if LogicInter[b] == 1:

        print('InterpoRange', InterpoRange[b])

        MinTteMpD = (UniTteMpD[UniTteMpD <= InterpoRangeMpD[b]]).max()

        MaxTteMpD = (UniTteMpD[UniTteMpD >= InterpoRangeMpD[b]]).min()

#I Name these Small/Large so it wasnt confusing when comparing to Min/Max TimeToExpire Minutes per Day

#Small/Large respectivly mean minium and maxium - I believed this was easier to understand due to the confusing nature of interpolation equation

        SmallPo = np.array(np.where(UniTteMpD == MinTteMpD))

        SmallPo = SmallPo.astype(int)

        

        LargePo = np.array(np.where(UniTteMpD == MaxTteMpD))

        LargePo = LargePo.astype(int)

        

        SmallVar = VixSigma[(SmallPo[0])]

        LargeVar = VixSigma[(LargePo[0])]



        SmallTte = UniTteMinutes[(SmallPo[0])]

        LargeTte = UniTteMinutes[(LargePo[0])]



        VarInterpo = np.sqrt((((SmallTte[0]*SmallVar[0]) * (((MaxTteMpD)-(InterpoRangeMpD[b]))/((MaxTteMpD) - (MinTteMpD)))) + ((LargeTte[0]*LargeVar[0]) * (((InterpoRangeMpD[b]-(MinTteMpD)))/((MaxTteMpD) - (MinTteMpD))))) * (525600/(InterpoRangeMpD[b])))

        print(VarInterpo*100)

        VixVarInterpo[b] = VarInterpo*100

print(' ')

print('Final Solution')

print(VixVarInterpo)