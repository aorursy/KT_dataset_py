import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
%matplotlib inline
## 2008 Elections ##
NA2008 = pd.read_csv("../input/National Assembly 2008.csv", encoding = "ISO-8859-1")
Winner08 = pd.DataFrame([])

for i in range(272):
    Const = "NA-%d"%(i+1)
    NA08 = NA2008.loc[NA2008['ConstituencyTitle']==Const, ['Party','Votes','TotalVotes', 'Seat']]
    if NA08.empty == True:
        print("2008 Missing:",Const) # missing constituency
    if NA08.empty == False:
        MAX = (NA08.loc[NA08['Votes'].idxmax()])
        temp = pd.DataFrame([Const,MAX['Party'],MAX['Votes'],MAX['TotalVotes'], MAX['Seat']], ).T
        temp.index = [i]
        Winner08 = pd.concat([Winner08,temp])
Winner08 = Winner08.rename(columns = {0:'Constituency', 1:'Party', 2:'Votes', 3:'TotalVotes', 4:'Seat'}) # Winners of 2008 Elections


## 2013 Elections ##
NA2013 = pd.read_csv("../input/National Assembly 2013 - Updated.csv", encoding = "ISO-8859-1")
Winner13 = pd.DataFrame([])

for i in range(272):
    Const = "NA-%d"%(i+1)
    NA13 = NA2013.loc[NA2013['ConstituencyTitle']==Const, ['Party','Votes','TotalVotes', 'Seat']]
    Votes = NA13['Votes'].astype('int64')
    NA13['Votes'] = Votes
    if NA13.empty == True:
        print("2013 Missing:",Const) # missing constituency
    if NA13.empty == False:
        MAX = (NA13.loc[NA13['Votes'].idxmax()])
        temp = pd.DataFrame([Const,MAX['Party'],MAX['Votes'],MAX['TotalVotes'], MAX['Seat']]).T
        temp.index = [i]
        Winner13 = pd.concat([Winner13,temp])
Winner13 = Winner13.rename(columns = {0:'Constituency', 1:'Party', 2:'Votes', 3:'TotalVotes', 4:'Seat'})
#Winner08
Total_Seats = 272
Winners = pd.DataFrame([])
Con = pd.DataFrame([])

for i in range(Total_Seats):
    Const = "NA-%d"%(i+1)
    if Const != "NA-8" and Const != "NA-119" and Const != "NA-207" and Const != "NA-235" and Const != "NA-266" and Const != "NA-83" and Const != "NA-254":
        tempCon = (Winner13.loc[Winner13['Constituency']==Const,['Constituency']])
        tempCon = tempCon.values.ravel()
        tempSeat = (Winner13.loc[Winner13['Constituency']==Const,['Seat']])
        tempSeat = tempSeat.values.ravel()
        temp13 = (Winner13.loc[Winner13['Constituency']==Const,['Party']])
        temp13 = temp13.values.ravel()
        temp08 = (Winner08.loc[Winner08['Constituency']==Const,['Party']])
        temp08 = temp08.values.ravel()
        temp = pd.DataFrame([tempCon, tempSeat,temp08,temp13])
        temp.columns = [i]
#        temp = temp.rename(columns = {0:'Winner'})
        Winners = pd.concat([Winners,temp], axis = 1)
        Con = pd.concat([Con,pd.DataFrame([Const])])
Final = Winners.T
Final = Final.rename(columns = {0: 'Constituency', 1: 'Seat', 2:'2008', 3:'2013'})
Final['2008'] = Final['2008'].replace(['MUTTHIDA\xa0MAJLIS-E-AMAL\xa0PAKISTAN'], 'Muttahidda Majlis-e-Amal Pakistan')
Final['2008'] = Final['2008'].replace(['Pakistan Muslim League'], 'Pakistan Muslim League(QA)')
Final['2013'] = Final['2013'].replace(['Pakistan Muslim League'], 'Pakistan Muslim League(QA)')
Final
Total_Seats = 272
Safe = pd.DataFrame([])
for i in range(Total_Seats):
    Const = "NA-%d"%(i+1)
    if Const != "NA-8" and Const != "NA-119" and Const != "NA-207" and Const != "NA-235" and Const != "NA-266" and Const != "NA-83" and Const != "NA-254":
        tempCon = (Final.loc[Final['Constituency']==Const,['Constituency']])
        tempCon = tempCon.values[0][0]
        tempSeat = (Final.loc[Final['Constituency']==Const,['Seat']])
        tempSeat = tempSeat.values[0][0]
        Party = (Final.loc[Final['Constituency']==Const,['2008']])
        Party = Party.values[0][0]
        Num = len(np.unique(Final.loc[Final['Constituency'] == Const]))-2
#        if (Num == 2):
#            Num = 0
#            Num = 100
        temp = pd.DataFrame([tempCon, tempSeat, Party, Num]).T
        temp.index = [i]
        Safe = pd.concat([Safe,temp])

Safe_Const = Safe[Safe[3]==1]
Safe_Const
np.unique(Safe_Const[2])
#Safe
PPPP_Safe = (Safe_Const[2]=='Pakistan Peoples Party Parliamentarians').sum()
MQMP_Safe = (Safe_Const[2]=='Muttahida Qaumi Movement Pakistan').sum()
Ind_Safe =  (Safe_Const[2]=='Independent').sum()
PMLN_Safe = (Safe_Const[2]=='Pakistan Muslim League (N)').sum()
PMLF_Safe = (Safe_Const[2]=='Pakistan Muslim League (F)').sum()
ANP_Safe =  (Safe_Const[2]=='Awami National Party').sum()
NNP_Safe =  (Safe_Const[2]=='National Peoples Party').sum()

x = np.arange(len(np.unique(Safe_Const[2])))
value = [PPPP_Safe, MQMP_Safe, Ind_Safe, PMLN_Safe, PMLF_Safe, ANP_Safe, NNP_Safe]

plt.figure(figsize=(14,8))
plt.grid()
pp, mq, nd, pmn, pmf, anp, nnp = plt.bar(x,value)
plt.xticks(x,('PPPP', 'MQM-P', 'Ind', 'PML-N', 'PML-F', 'ANP', 'NNP'))
plt.ylabel('Seats')
pp.set_facecolor('r')
mq.set_facecolor('g')
nd.set_facecolor('b')
pmn.set_facecolor('y')
pmf.set_facecolor('k')
anp.set_facecolor('m')

plt.show()
Safe_Rating = pd.DataFrame([])
for i in range(len(Safe_Const)): #len(Safe_Const)
    Const = Safe_Const.iloc[i,0]
#    print(Const)
    NA = NA2013.loc[NA2013['ConstituencyTitle']==Const, ['ConstituencyTitle','Party','Votes','TotalVotes', 'Seat']]
    Votes = NA['Votes'].astype('int64')
    NA['Votes'] = Votes
    NA = NA.sort_values(by=['Votes'], ascending = False)
    NA = NA.iloc[0:2,:]
    if Const == 'NA-46':  ### Total Votes for NA-46 missing in original data
        NA['TotalVotes'] = 16857 
    for j in range(len(NA)):
        temp = (NA.iloc[j,:])
        V = temp['Votes']*100/temp['TotalVotes']
        NA.iloc[j,2] = V
    
    Win = NA.iloc[0,1]
#    print(Win)
    Diff = NA.iloc[0,2] - NA.iloc[1,2]
    temp = pd.DataFrame([Const,Diff,Win]).T
    temp.index = [i]
    Safe_Rating = pd.concat([Safe_Rating,temp])

Safe_Rating = Safe_Rating.sort_values(by=[1], ascending = True)
Safe_Rating = Safe_Rating.reset_index()
Safe_Rating = Safe_Rating.drop(['index'], axis=1)
Safe_Rating = Safe_Rating.rename(columns = {0:'Constituency', 1:'Diff(%)', 2:'Winner'})
Safe_Rating = Safe_Rating.set_index('Constituency')
Safe_Rating['Diff(%)'] = Safe_Rating['Diff(%)'].astype('float16')
#Safe_Rating
import seaborn as sns
plt.figure(figsize=(10,20))
sns.heatmap(Safe_Rating.loc[:,['Diff(%)']], annot=True, cmap='viridis')
plt.show()
plt.figure(figsize=(14,8))
sns.distplot(Safe_Rating['Diff(%)'],kde=False, rug=True);
Top_High_Diff = Safe_Rating.tail(10)
Top_High_Diff
Top_Low_Diff = Safe_Rating.head(10)
Top_Low_Diff