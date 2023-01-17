import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
%matplotlib inline
import warnings
warnings.filterwarnings("ignore")
## 2008 Elections ##
Margin = pd.DataFrame([])
NA2008 = pd.read_csv("../input/National Assembly 2008.csv", encoding = "ISO-8859-1")
Constituencies = NA2008.ConstituencyTitle.unique()
for i in range(len(Constituencies)):
    Const = Constituencies[i]
    if Const != "NA-8" and Const != "NA-119" and Const != "NA-207" and Const != "NA-235" and Const != "NA-266" and Const != "NA-83" and Const != "NA-254":
        NA = NA2008.loc[NA2008['ConstituencyTitle']==Const, ['ConstituencyTitle','Party','Votes','TotalVotes', 'Seat']]
        NA = NA.sort_values(by=['Votes'], ascending = False)
        NA = NA.iloc[0:2,:] # Only Winner and Runnerup
        NA['TotalVotes'] = NA.Votes.sum() 
        NA['Votes'] = NA['Votes']/np.unique(NA['TotalVotes']) # Changing votes to percentages
        Diff = NA.iloc[0,2] - NA.iloc[1,2]
        Win = NA.iloc[0,1] # Winner
        Run = NA.iloc[1,1] # RunnerUp
        temp = pd.DataFrame([Const,Diff,Win,Run]).T
        temp.index = [i]
        Margin = pd.concat([Margin,temp])
Margin08 = Margin.rename(columns = {0:'Constituency08', 1:'Diff08(%)', 2:'Winner08', 3:'RunnerUp08'})
## 2013 Elections ##
Margin = pd.DataFrame([])
NA2013 = pd.read_csv("../input/National Assembly 2013.csv", encoding = "ISO-8859-1")
Constituencies = NA2013.ConstituencyTitle.unique()
for i in range(len(Constituencies)):
    Const = Constituencies[i]
    if Const != "NA-8" and Const != "NA-119" and Const != "NA-207" and Const != "NA-235" and Const != "NA-266" and Const != "NA-83" and Const != "NA-254":
        NA = NA2013.loc[NA2013['ConstituencyTitle']==Const, ['ConstituencyTitle','Party','Votes','TotalVotes', 'Seat']]
        NA = NA.sort_values(by=['Votes'], ascending = False)
        NA = NA.iloc[0:2,:] # Only Winner and Runnerup
        NA['TotalVotes'] = NA.Votes.sum() 
        NA['Votes'] = NA['Votes']/np.unique(NA['TotalVotes']) # Changing votes to percentages
        Diff = NA.iloc[0,2] - NA.iloc[1,2]
        Win = NA.iloc[0,1] # Winner
        Run = NA.iloc[1,1] # RunnerUp
        temp = pd.DataFrame([Const,Diff,Win,Run]).T
        temp.index = [i]
        Margin = pd.concat([Margin,temp])
Margin13 = Margin.rename(columns = {0:'Constituency13', 1:'Diff13(%)', 2:'Winner13', 3:'RunnerUp13'})
Sorted_Margin13 = Margin13#.sort_values(by=['Diff(%)'], ascending = True)
Sorted_Margin13 = Sorted_Margin13.reset_index()
Sorted_Margin13 = Sorted_Margin13.drop(['index'], axis=1)
#Sorted_Margin13 = Sorted_Margin13.drop([269], axis=0)
Sorted_Margin13 = Sorted_Margin13.set_index('Constituency13')
Sorted_Margin13['Diff13(%)'] = (Sorted_Margin13['Diff13(%)']*100).astype('float16')

Sorted_Margin08 = Margin08#.sort_values(by=['Diff(%)'], ascending = True)
Sorted_Margin08 = Sorted_Margin08.reset_index()
Sorted_Margin08 = Sorted_Margin08.drop(['index'], axis=1)
#Sorted_Margin08 = Sorted_Margin08.drop([266], axis=0)
Sorted_Margin08 = Sorted_Margin08.set_index('Constituency08')
Sorted_Margin08['Diff08(%)'] = (Sorted_Margin08['Diff08(%)']*100).astype('float16')
Table = pd.concat([Sorted_Margin08,Sorted_Margin13],axis=1)
#Table.drop(['Constituency13'],axis=1)
Table.style.background_gradient(cmap='summer',axis=0)
Winner13 = pd.DataFrame(np.array(Sorted_Margin13['Winner13']))
Winner08  = Sorted_Margin08.iloc[:,0:2]
Winner08.reset_index(inplace=True)
Comp = pd.concat([Winner08,Winner13],axis=1)
Comp = Comp.rename(columns = {0:'Winner13'})
Comp = Comp.sort_values(by=['Diff08(%)'], ascending = True)
Comp = Comp.set_index('Constituency08')
Comp = Comp.drop(['NA-42'], axis=0)
import seaborn as sns
Comp['Retained'] = Comp['Winner08'] == Comp['Winner13']
Comp = Comp[Comp.Winner08 != 'Independent']#.tail(8)#.Retained.sum()
Comp = Comp[Comp.Winner13 != 'Independent']
Comp.style.background_gradient(cmap='summer',axis=0)

Not_Ret = Comp.loc[Comp['Retained']==False]#.style.background_gradient(cmap='summer',axis=0)
Ret = Comp.loc[Comp['Retained']==True]
print("Seats Retained:", len(Ret))
print("Seats Lost:" ,len(Not_Ret))

Ret.loc[Ret['Diff08(%)'] <= 10]#.Retained.sum()
#sns.distplot(Ret['Diff08(%)'], kde=False, rug=False);
#sns.distplot(Not_Ret['Diff08(%)'], kde=True, rug=False);
Not_Ret.loc[Not_Ret['Diff08(%)'] >= 30]