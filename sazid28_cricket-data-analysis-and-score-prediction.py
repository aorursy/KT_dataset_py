import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
#warnings.simplefilter(action = "ignore", category = FutureWarning)
matches = pd.read_csv('../input/dataset1_test.csv') 
matches.head()
Records=pd.DataFrame(matches)
Records.columns=['Batsman','Runs','Minute','Ballfaced','fours','sixes','StrikeRate','Position','DismissalAt','Innings']
#Records.head(10)
# split individual player records

batsman1 = Records[(Records.Batsman==1)]
batsman2 = Records[(Records.Batsman==2)]
batsman3 = Records[(Records.Batsman==3)]
batsman4 = Records[(Records.Batsman==4)]
batsman5 = Records[(Records.Batsman==5)]
batsman6 = Records[(Records.Batsman==6)]
  ### average run of each player

avgOfPlayersGrpBy= Records[['Batsman','Runs']].groupby('Batsman').mean() 
print(avgOfPlayersGrpBy)
                 
avgOfPlayersGrpByPlot=avgOfPlayersGrpBy.sum(axis = 1).sort_values(ascending = False)
plt.ylabel('Average Runs',fontsize=15, fontweight='bold')
plt.rcParams["figure.figsize"] = [7,4]
avgOfPlayersGrpByPlot.plot(kind ='bar',color='green' ,title = " Average run of Batsman")
#batsman run by four sixes comparison

runBysixFour= Records[['Batsman','fours','sixes','Runs']].groupby('Batsman').sum() #run from boundary from each batsman
runBysixFour['totalRunFromBoundary']=runBysixFour['fours']*4+runBysixFour['sixes']*6
runBysixFour['runByFour']=runBysixFour['fours']*4
runBysixFour['runBysix']=runBysixFour['sixes']*6
runBysixFour['totalRunFromBoundary']
print(runBysixFour)

plt.rcParams["figure.figsize"] = [10,7]
plt.show(runBysixFour.plot.bar())
#batsman 1 data analysis
batsman1_run=batsman1['Runs'] #batsman 1 runs 
print(('Average of Batsman 1: '),  batsman1_run.mean())
print(('Maximum of Batsman 1 in a Match: '),  batsman1_run.max())

batsman1_run.plot(color='red')

p1=[batsman1_run.count(),0]    #x1,x2
p2=[batsman1_run.mean(),batsman1_run.mean()]   #y1,y2
plt.rcParams["figure.figsize"] = [10,7]
plt.plot(p1,p2)
plt.xlabel('No Of Matches',fontsize=15)
plt.ylabel('Runs',fontsize=15)
plt.rcParams["figure.figsize"] = [12,7]
plt.show()
plt.gcf().clear()
plt.clf()
plt.cla()
plt.close()
#calculation of batsman1 average run of each consecutive match
batsman1_avg = []
i=0
for row in batsman1_run:
    
        if i==0:
            batsman1_avg.append(row)

        else :
            value= (batsman1_avg[i-1]*i+row)/(i+1)
            batsman1_avg.append(value)
            
        i += 1

batsman1['avgRunPerMatch'] = batsman1_avg
 #batsman 1 runs 
batsman1_avg_run=batsman1['avgRunPerMatch']       #batsman 1 average runs concecutive matches
print(('Average of Batsman 1: '),  batsman1_run.mean())
print(('Maximum of Batsman 1 in a Match: '),  batsman1_run.max())

batsman1_run.plot(color='red')
batsman1_avg_run.plot(color='blue')
p1=[batsman1_run.count(),0]    #x1,x2
p2=[batsman1_run.mean(),batsman1_run.mean()]   #y1,y2
plt.rcParams["figure.figsize"] = [10,7]
plt.plot(p1,p2)
plt.xlabel('No Of Matches',fontsize=15)
plt.ylabel('Runs',fontsize=15)
plt.rcParams["figure.figsize"] = [12,7]
plt.show()
plt.gcf().clear()
plt.clf()
plt.cla()
plt.close()
###run 
 
# Data to plot
labels = 'Fours', 'sixes', 'others( 1s,2s etc.)'
sizes = [batsman1['fours'].sum()*4,batsman1['sixes'].sum()*6,batsman1['Runs'].sum()-(batsman1['fours'].sum()*4+batsman1['sixes'].sum()*6)]

plt.pie(sizes,  labels=labels, 
        autopct='%1.1f%%', shadow=True, startangle=140)
 
plt.axis('equal')
plt.show()
#print(batsman1)
batsman1_Dismissal_lbw =len( batsman1[(batsman1.DismissalAt=='lbw')])
batsman1_Dismissal_bowled =len( batsman1[(batsman1.DismissalAt=='bowled')])
batsman1_Dismissal_runout =len( batsman1[(batsman1.DismissalAt=='run out')])
batsman1_Dismissal_caught =len( batsman1[(batsman1.DismissalAt=='caught')])
batsman1_Dismissal_notout=len( batsman1[(batsman1.DismissalAt=='not out')])
batsman1_Dismissal_stumped =len( batsman1[(batsman1.DismissalAt=='stumped')])

labels = 'lbw', 'bowled', 'run out','caught','stumped','not out'
sizes = [batsman1_Dismissal_lbw,batsman1_Dismissal_bowled,batsman1_Dismissal_runout,
          batsman1_Dismissal_caught,batsman1_Dismissal_stumped,batsman1_Dismissal_notout]

plt.pie(sizes,  labels=labels, 
        autopct='%1.1f%%', shadow=True, startangle=140)
 
plt.axis('equal')
plt.show()

batsman1_runat_3 = batsman1[(batsman1.Position==3)]
batsman1_runat_4 = batsman1[(batsman1.Position==4)]
#print(batsman1_runat_3)
batsman1_runat_3_runs=batsman1_runat_3['Runs']       #batsman 1 average runs concecutive matches
batsman1_runat_4_runs=batsman1_runat_4['Runs']
batsman1_runat_3_strikerate=batsman1_runat_3['StrikeRate']       #batsman 1 average runs concecutive matches
batsman1_runat_4_strikerate=batsman1_runat_4['StrikeRate']

print(('Total no of matches of Batsman 1 at position 3: '),  batsman1_runat_3_runs.count())
print(('Total no of matches of Batsman 1 at position 4: '),  batsman1_runat_4_runs.count(),'\n\n')

print(('Average of Batsman 1 at position 3: '),  batsman1_runat_3_runs.mean())
print(('Average of Batsman 1 at position 4: '),  batsman1_runat_4_runs.mean(),'\n\n')

print(('Total run of Batsman 1 at position 3: '),  batsman1_runat_3_runs.sum())
print(('Total run of Batsman 1 at position 4: '),  batsman1_runat_4_runs.sum(),'\n\n')

print(('average strike rate  Batsman 1 at position 3: '),  batsman1_runat_3_strikerate.mean())
print(('average strike rate of Batsman 1 at position 4: '),  batsman1_runat_4_strikerate.mean(),'\n\n')

batsman1_runat_3_runs.plot(color='black')
batsman1_runat_4_runs.plot(color='green')


plt.rcParams["figure.figsize"] = [10,7]
plt.xlabel('No Of Matches',fontsize=15)
plt.ylabel('Runs',fontsize=15)
plt.rcParams["figure.figsize"] = [12,7]
plt.show()
plt.gcf().clear()
plt.clf()
plt.cla()
plt.close()
ax=sns.swarmplot(y="Runs", x="DismissalAt",size=10, data=batsman1)

ax=sns.swarmplot(y="StrikeRate", x="Runs",size=8,data=batsman1)

ax.grid(True)
#ax=sns.swarmplot(y="StrikeRate", x="Runs",size=8, data=batsman1[(batsman1.Position==3) | (batsman1.Position==4)])


ax=sns.swarmplot(y="StrikeRate", x="Runs",size=8, color='red',data=batsman1[(batsman1.Position==3) & (batsman1.Runs>=50)&(batsman1.Runs<100) ])
ax=sns.swarmplot(y="StrikeRate", x="Runs",size=8, color='blue',data=batsman1[(batsman1.Position==4) & (batsman1.Runs>=50)& (batsman1.Runs>=50)&(batsman1.Runs<100)  ])
#red position 3   50<run<100
#blue position 4  50<run<100
ax=sns.swarmplot(y="StrikeRate", x="Runs",size=8, color='black',data=batsman1[(batsman1.Position==3) & (batsman1.Runs>=100) ])
ax=sns.swarmplot(y="StrikeRate", x="Runs",size=8, color='green',data=batsman1[(batsman1.Position==4) & (batsman1.Runs>=100)  ])
#black  position 3 more than 100 run
#green  position 4 more than 100 run
ax=sns.swarmplot(y="StrikeRate", x="DismissalAt",size=10, data=batsman1)
#batsman 2 data analysis
#calculation of batsman1 average run of each consecutive match
batsman2_avg = []
match_no = []
i=0
for row in batsman2_run:
        match_no.append(i+1)
        if i==0:
            batsman2_avg.append(row)

        else :
            value= (batsman2_avg[i-1]*i+row)/(i+1)
            batsman2_avg.append(value)
            
        i += 1

batsman2['avgRunPerMatch'] = batsman2_avg
batsman2['match_no']=match_no
 #batsman 2 runs       #batsman 2 average runs concecutive matches
print(('Average of Batsman 2: '),  batsman1_run.mean())
print(('Maximum of Batsman 2 in a Match: '),  batsman1_run.max())

batsman2['Runs'].plot(color='red')
batsman2['avgRunPerMatch'].plot(color='blue')
p1=[batsman2_run.count()+57,57]    #x1,x2
p2=[batsman2_run.mean(),batsman2_run.mean()]   #y1,y2
plt.rcParams["figure.figsize"] = [10,7]
plt.plot(p1,p2)
plt.xlabel('No Of Matches',fontsize=15)
plt.ylabel('Runs',fontsize=15)
plt.rcParams["figure.figsize"] = [12,7]
plt.show()
plt.gcf().clear()
plt.clf()
plt.cla()
plt.close()
#batsman 2 runs 
batsman2_avg_runs=batsman2['avgRunPerMatch']       #batsman 1 average runs concecutive matches

print(('Average of Batsman 2: '),  batsman1_run.mean())
print(('Maximum of Batsman 2 in a Match: '),  batsman1_run.max())
ax = batsman2.plot(kind='line',x='match_no',y='Runs', title ="Batsman 2 runs",figsize=(10,7),legend=True, fontsize=12)
plt.xlabel('No Of Matches',fontsize=15)


plt.ylabel('Runs',fontsize=15)
plt.show()

###run 
 
# Data to plot
labels = 'Fours', 'sixes', 'others( 2s,2s etc.)'
sizes = [batsman2['fours'].sum()*4,batsman2['sixes'].sum()*6,batsman2['Runs'].sum()-(batsman2['fours'].sum()*4+batsman2['sixes'].sum()*6)]

plt.pie(sizes,  labels=labels, 
        autopct='%1.1f%%', shadow=True, startangle=240)
 
plt.axis('equal')
plt.show()
#print(batsman2)
batsman2_Dismissal_lbw =len( batsman2[(batsman2.DismissalAt=='lbw')])
batsman2_Dismissal_bowled =len( batsman2[(batsman2.DismissalAt=='bowled')])
batsman2_Dismissal_runout =len( batsman2[(batsman2.DismissalAt=='run out')])
batsman2_Dismissal_caught =len( batsman2[(batsman2.DismissalAt=='caught')])
batsman2_Dismissal_notout=len( batsman2[(batsman2.DismissalAt=='not out')])
batsman2_Dismissal_stumped =len( batsman2[(batsman2.DismissalAt=='stumped')])

labels = 'lbw', 'bowled', 'run out','caught','stumped','not out'
sizes = [batsman2_Dismissal_lbw,batsman2_Dismissal_bowled,batsman2_Dismissal_runout,
          batsman2_Dismissal_caught,batsman2_Dismissal_stumped,batsman2_Dismissal_notout]

plt.pie(sizes,  labels=labels, 
        autopct='%1.1f%%', shadow=True, startangle=140)
 
plt.axis('equal')
plt.show()
ax=sns.swarmplot(y="Runs", x="DismissalAt",size=10, data=batsman2)
batsman2_runat_3 = batsman2[(batsman2.Position==3)]
batsman2_runat_4 = batsman2[(batsman2.Position==4)]
batsman2_runat_5 = batsman2[(batsman2.Position==5)]
#print(batsman2_runat_3)
batsman2_runat_3_runs=batsman2_runat_3['Runs']       #batsman 2 average runs concecutive matches
batsman2_runat_4_runs=batsman2_runat_4['Runs']
batsman2_runat_5_runs=batsman2_runat_5['Runs']
batsman2_runat_3_strikerate=batsman2_runat_3['StrikeRate']       #batsman 2 average runs concecutive matches
batsman2_runat_4_strikerate=batsman2_runat_4['StrikeRate']
batsman2_runat_5_strikerate=batsman2_runat_5['StrikeRate']

print(('Total no of matches of Batsman 2 at position 3: '),  batsman2_runat_3_runs.count())
print(('Total no of matches of Batsman 2 at position 4: '),  batsman2_runat_4_runs.count())
print(('Total no of matches of Batsman 2 at position 5: '),  batsman2_runat_5_runs.count(),'\n\n')

print(('Average of Batsman 2 at position 3: '),  batsman2_runat_3_runs.mean())
print(('Average of Batsman 2 at position 5: '),  batsman2_runat_4_runs.mean())
print(('Average of Batsman 2 at position 5: '),  batsman2_runat_5_runs.mean(),'\n\n')

print(('Total run of Batsman 2 at position 3: '),  batsman2_runat_3_runs.sum())
print(('Total run of Batsman 2 at position 4: '),  batsman2_runat_4_runs.sum())
print(('Total run of Batsman 2 at position 5: '),  batsman2_runat_5_runs.sum(),'\n\n')

print(('average strike rate  Batsman 2 at position 3: '),  batsman2_runat_3_strikerate.mean())
print(('average strike rate of Batsman 2 at position 4: '),  batsman2_runat_4_strikerate.mean())
print(('average strike rate of Batsman 2 at position 5: '),  batsman2_runat_5_strikerate.mean(),'\n\n')

batsman2_runat_3_runs.plot(color='black')
batsman2_runat_4_runs.plot(color='green')
batsman2_runat_5_runs.plot(color='blue')

plt.rcParams["figure.figsize"] = [10,7]
plt.xlabel('No Of Matches',fontsize=15)
plt.ylabel('Runs',fontsize=15)

plt.show()
plt.gcf().clear()
plt.clf()
plt.cla()
plt.close()
ax=sns.swarmplot(y="StrikeRate", x="Runs",size=8,data=batsman2)

ax.grid(True)
#ax=sns.swarmplot(y="StrikeRate", x="Runs",size=8, data=batsman1[(batsman1.Position==3) | (batsman1.Position==4)])

ax=sns.swarmplot(y="StrikeRate", x="Runs",size=8, color='blue',data=batsman2[ (batsman2.Runs>=50) &   (batsman2.Runs<=100) ])
ax=sns.swarmplot(y="StrikeRate", x="Runs",size=8, color='red',data=batsman2[ (batsman2.Runs>=100) ])
#keep batsman1 all record in data
data = batsman1
#check the data
data.head()
data.columns
#our feature variable "Runs"
X = data[['Runs']]
X.head()
#our Target variable "number of ball faced"
Y = data[['Ballfaced']]
Y.head()
#import train,test split library 
from sklearn.cross_validation import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2 ,random_state = 1)
X_train.head()
Y_train.head()
X_test.head()
Y_test.head()
data.shape
X_train.shape
Y_train.shape
X_test.shape
Y_test.shape
#import linear regression
from sklearn.linear_model import LinearRegression
#create instance
LR = LinearRegression()
LR.fit(X_train,Y_train)
# y = mx + c (coefficient , intercept
print(LR.intercept_)
print(LR.coef_)
# X_test = Runs (input) 
# we will predict number of ball faced
prediction = LR.predict(X_test)
print(prediction)
# 200 = Run
prediction = LR.predict(200)
#for 200 run batsman may b faced 331 or 332 ball
prediction
