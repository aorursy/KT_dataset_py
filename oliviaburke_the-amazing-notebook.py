# Imports

# pandas
import pandas as pd
from pandas import Series,DataFrame

# numpy, matplotlib, seaborn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
## %matplotlib inline

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
# get titanic & test csv files as a DataFrame
titanic_df = pd.read_csv("../input/train.csv")
test_df    = pd.read_csv("../input/test.csv")


# preview the data
print(titanic_df.head())
titanic_df.info()
print("----------------------------")
test_df.info()


test_df["Survived"] = -1

print("============================")
titanicANDtest_df = pd.concat([titanic_df, test_df], keys=['titanic', 'test'])
float_formatter = lambda x: "%.2f" % x
np.set_printoptions(formatter={'float_kind':float_formatter})

#$# Nov 19 edit
print('Here are the NAN counts of titanic_df')
print( titanic_df.isnull().sum(), '\n' )
print('Pclass and Sex are useful factors.')
print('Here are pivot tables for survivor sum, passenger count, and mean.\n')

table0a = pd.pivot_table(titanic_df, values = 'Survived', \
                    index = ['Pclass'], columns=['Sex'], aggfunc=np.sum)
print( table0a,'\n' )

table0b = pd.pivot_table(titanic_df, values = 'Survived', \
                    index = ['Pclass'], columns=['Sex'], aggfunc='count')
print( table0b,'\n' )

table0c = pd.pivot_table(titanic_df, values = 'Survived', \
                    index = ['Pclass'], columns=['Sex'], aggfunc=np.mean)
print( table0c,'\n' )

print('So we create new columns Female and Male for machine learning.\n')

sex_dummies_titanic  = pd.get_dummies(titanic_df['Sex'])
sex_dummies_titanic.columns = ['Female','Male']
sex_dummies_titanic.drop(['Male'], axis=1, inplace=True)
titanic_df = titanic_df.join(sex_dummies_titanic)
titanic_df['Fem'] = titanic_df['Female']
titanic_df['F'] = titanic_df['Female']
#$# titanic_df.drop(['Sex'],axis=1,inplace=True)

sex_dummies_test  = pd.get_dummies(test_df['Sex'])
sex_dummies_test.columns = ['Female','Male']
sex_dummies_test.drop(['Male'], axis=1, inplace=True)
test_df = test_df.join(sex_dummies_test)
test_df['Fem'] = test_df['Female']
test_df['F'] = test_df['Female']
#$# titanic_df.drop(['Sex'],axis=1,inplace=True)

sex_dummies_titanicANDtest  = pd.get_dummies(titanicANDtest_df['Sex'])
sex_dummies_titanicANDtest.columns = ['Female','Male']
sex_dummies_titanicANDtest.drop(['Male'], axis=1, inplace=True)
titanicANDtest_df = titanicANDtest_df.join(sex_dummies_titanicANDtest)
titanicANDtest_df['Fem'] = titanicANDtest_df['Female']
titanicANDtest_df['F'] = titanicANDtest_df['Female']
#$# titanic_df.drop(['Sex'],axis=1,inplace=True)



pclass_dummies_titanic  = pd.get_dummies(titanic_df['Pclass'])
pclass_dummies_titanic.columns = ['Class1','Class2','Class3']
titanic_df    = titanic_df.join(pclass_dummies_titanic)

pclass_dummies_test  = pd.get_dummies(test_df['Pclass'])
pclass_dummies_test.columns = ['Class1','Class2','Class3']
test_df    = test_df.join(pclass_dummies_test)

pclass_dummies_titanicANDtest  = pd.get_dummies(titanicANDtest_df['Pclass'])
pclass_dummies_titanicANDtest.columns = ['Class1','Class2','Class3']
titanicANDtest_df    = titanicANDtest_df.join(pclass_dummies_titanicANDtest)
print('Now from the Name we locate the MasterOrMiss passengers.')

def get_masterormiss(passenger):
    name = passenger
    if (   ('Master' in str(name)) \
        or ('Miss'   in str(name)) \
        or ('Mlle'   in str(name)) ):
        return 1
    else:
        return 0

titanic_df['MasterMiss'] = \
    titanic_df[['Name']].apply( get_masterormiss, axis=1 )
titanic_df['MMs'] = titanic_df['MasterMiss']
titanic_df['Ms'] = titanic_df['MasterMiss']
titanic_df['m'] = titanic_df['MasterMiss']

test_df['MasterMiss'] = \
    test_df[['Name']].apply( get_masterormiss, axis=1 )
test_df['MMs'] = test_df['MasterMiss']
test_df['Ms'] = test_df['MasterMiss']
test_df['m'] = test_df['MasterMiss']

titanicANDtest_df['MasterMiss'] = \
    titanicANDtest_df[['Name']].apply( get_masterormiss, axis=1 )
titanicANDtest_df['MMs'] = titanicANDtest_df['MasterMiss']
titanicANDtest_df['Ms'] = titanicANDtest_df['MasterMiss']
titanicANDtest_df['m'] = titanicANDtest_df['MasterMiss']

#$# print(titanicANDtest_df.head())
    
    

print('Here are pivot tables for survival by Sex and MasterMiss, by Pclass.\n')

table0d = pd.pivot_table(titanic_df, values = 'Survived', \
                    index = ['Female', 'MasterMiss'], columns=['Pclass'], \
                    aggfunc=np.sum)
print( table0d.iloc[::-1],'\n' ) #$# This hack reverses the order of the rows.

table0e = pd.pivot_table(titanic_df, values = 'Survived', \
                    index = ['Female', 'MasterMiss'], columns=['Pclass'], \
                    aggfunc='count')
print( table0e.iloc[::-1],'\n' )

table0f = pd.pivot_table(titanic_df, values = 'Survived', \
                    index = ['Female', 'MasterMiss'], columns=['Pclass'], \
                    aggfunc=np.mean)
print( table0f.iloc[::-1],'\n' ) 

print('Now Embarked.  Fill the 2 NaNs with S, as ticket-number blocks imply.')

titanic_df["Embarked"] = titanic_df["Embarked"].fillna("S")
titanicANDtest_df["Embarked"] = titanicANDtest_df["Embarked"].fillna("S")


embark_dummies_titanic  = pd.get_dummies(titanic_df['Embarked'])
#$# embark_dummies_titanic.columns = ['3','17','19']
#$# embark_dummies_titanic.drop(['S'], axis=1, inplace=True)
titanic_df = titanic_df.join(embark_dummies_titanic)

embark_dummies_test  = pd.get_dummies(test_df['Embarked'])
#$# embark_dummies_test.columns = ['3','17','19']
#$# embark_dummies_test.drop(['S'], axis=1, inplace=True)
test_df    = test_df.join(embark_dummies_test)

embark_dummies_titanicANDtest  = pd.get_dummies(titanicANDtest_df['Embarked'])
#$# embark_dummies_titanicANDtest.columns = ['3','17','19']
#$# embark_dummies_titanic.drop(['S'], axis=1, inplace=True)
titanicANDtest_df = titanicANDtest_df.join(embark_dummies_titanicANDtest)


print('Pivot tables for survival by Sex + MasterMiss, by Pclass + Embark.\n')

table0d = pd.pivot_table(titanic_df, values = 'Survived', \
                    index = ['Female', 'MasterMiss'], \
                    columns=['Embarked', 'Pclass'], \
                    aggfunc=np.sum)
print( table0d.iloc[::-1],'\n' ) #$# This hack reverses the order of the rows.

table0e = pd.pivot_table(titanic_df, values = 'Survived', \
                    index = ['Female', 'MasterMiss'], \
                    columns=['Embarked', 'Pclass'], \
                    aggfunc='count')
print( table0e.iloc[::-1],'\n' )

table0f = pd.pivot_table(titanic_df, values = 'Survived', \
                    index = ['Female', 'MasterMiss'], \
                    columns=['Embarked', 'Pclass'], \
                    aggfunc=np.mean)
print( table0f.iloc[::-1],'\n' ) 

print('Now consider Parch as a binary decision: is the value greater than 0?')

#$# def is_positive(passenger):
#$#     parch = int(passenger)
#$#     return 1 if (parch > 0) else 0

titanic_df['ParchBinary'] = \
  titanic_df[['Parch']].apply( (lambda x: int(int(x) > 0) ), axis=1)
titanic_df['Pch'] = titanic_df['ParchBinary']
titanic_df['Pc'] = titanic_df['ParchBinary']
titanic_df['p'] = titanic_df['ParchBinary']
 
test_df['ParchBinary'] = \
  test_df[['Parch']].apply( (lambda x: int(int(x) > 0) ), axis=1)
test_df['Pch'] = test_df['ParchBinary']
test_df['Pc'] = test_df['ParchBinary']
test_df['p'] = test_df['ParchBinary']
 
titanicANDtest_df['ParchBinary'] = \
  titanicANDtest_df[['Parch']].apply( (lambda x: int(int(x) > 0) ), axis=1) 
titanicANDtest_df['Pch'] = titanicANDtest_df['ParchBinary']
titanicANDtest_df['Pc'] = titanicANDtest_df['ParchBinary']
titanicANDtest_df['p'] = titanicANDtest_df['ParchBinary']


print('Pivot tables: Sex + MasterMiss + ParchBinary, by Pclass + Embark.\n')

table0d = pd.pivot_table(titanic_df, values = 'Survived', \
                    index = ['Pch', 'Female', 'MasterMiss'], \
                    columns=['Embarked', 'Pclass'], \
                    aggfunc=np.sum)
print( table0d.iloc[::-1],'\n' ) #$# This hack reverses the order of the rows.

table0e = pd.pivot_table(titanic_df, values = 'Survived', \
                    index = ['Pch', 'Female', 'MasterMiss'], \
                    columns=['Embarked', 'Pclass'], \
                    aggfunc='count')
print( table0e.iloc[::-1],'\n' )

table0f = pd.pivot_table(titanic_df, values = 'Survived', \
                    index = ['Pch', 'Female', 'MasterMiss'], \
                    columns=['Embarked', 'Pclass'], \
                    aggfunc=np.mean)
print( table0f.iloc[::-1],'\n' )

print('Now consider SibSp as a binary decision: is the value greater than 0?')

titanic_df['SibSpBinary'] = \
  titanic_df[['SibSp']].apply( (lambda x: int(int(x) > 0) ), axis=1)
titanic_df['SbS'] = titanic_df['SibSpBinary']
titanic_df['Sb'] = titanic_df['SibSpBinary']
titanic_df['s'] = titanic_df['SibSpBinary']
 
test_df['SibSpBinary'] = \
  test_df[['SibSp']].apply( (lambda x: int(int(x) > 0) ), axis=1)
test_df['SbS'] = test_df['SibSpBinary']
test_df['Sb'] = test_df['SibSpBinary']
test_df['s'] = test_df['SibSpBinary']

titanicANDtest_df['SibSpBinary'] = \
  titanicANDtest_df[['SibSp']].apply( (lambda x: int(int(x) > 0) ), axis=1)
titanicANDtest_df['SbS'] = titanicANDtest_df['SibSpBinary']
titanicANDtest_df['Sb'] = titanicANDtest_df['SibSpBinary']
titanicANDtest_df['s'] = titanicANDtest_df['SibSpBinary']


print('Pivot tables: ParchBinary + SibSpBinary + Sex + MasterMiss, \
by Embark + Pclass.\n')

table0d = pd.pivot_table(titanic_df, values = 'Survived', \
             index = ['Fem', 'MMs', 'SbS', 'Pch'], \
             columns=['Pclass', 'Embarked'], \
             aggfunc=np.sum)
print( table0d.iloc[::-1],'\n' ) #$# This hack reverses the order of the rows.

table0e = pd.pivot_table(titanic_df, values = 'Survived', \
             index = ['Fem', 'MMs', 'SbS', 'Pch'], \
             columns=['Pclass', 'Embarked'], \
             aggfunc='count')
print( table0e.iloc[::-1],'\n' )

table0f = pd.pivot_table(titanic_df, values = 'Survived', \
             index = ['Fem', 'MMs', 'SbS', 'Pch'], \
             columns=['Pclass', 'Embarked'], \
             aggfunc=np.mean )
print( table0f.iloc[::-1].round(2),'\n' )

print('Now consider Ticket for sharedTicket and sharedTicketWithSurvivor\n')


pd.options.mode.chained_assignment = None  # This turns off needless warnings.


titanicANDtest_df['SharedTicket?'] = 0
titanicANDtest_df['SharedTicket?'].loc[ \
    titanicANDtest_df['Ticket'].duplicated(keep=False) ] = 1

titanicANDtest_df['SharedCabin?'] = 0
titanicANDtest_df['SharedCabin?'].loc[ \
    titanicANDtest_df['Cabin'].duplicated(keep=False) ] = 1                 


smallExperimentA_df = titanicANDtest_df[ \
    titanicANDtest_df['SharedTicket?'] == 1 ]
SharedTicketHolders_df = titanicANDtest_df[ \
    titanicANDtest_df['SharedTicket?'] == 1 ]
SharedCabinHolders_df = titanicANDtest_df[ \
    titanicANDtest_df['SharedCabin?'] == 1 ]


smallExperimentB_df = \
    smallExperimentA_df.groupby(['Ticket', \
        'Survived']).size().to_frame('Size').reset_index()
SharedTicketHolders_TicketSurvivedSize_df = \
    SharedTicketHolders_df.groupby(['Ticket', \
        'Survived']).size().to_frame('Size').reset_index()
SharedCabinHolders_CabinSurvivedSize_df = \
    SharedCabinHolders_df.groupby(['Cabin', \
        'Survived']).size().to_frame('Size').reset_index()
    
    
smallExperimentG_df = \
    smallExperimentB_df[ smallExperimentB_df['Survived'] == 1 ]
SharedTicketHoldersSurvivors_TicketSurvivedSize_df = \
    SharedTicketHolders_TicketSurvivedSize_df[ \
        SharedTicketHolders_TicketSurvivedSize_df['Survived'] == 1 ]
SharedCabinHoldersSurvivors_CabinSurvivedSize_df = \
    SharedCabinHolders_CabinSurvivedSize_df[ \
        SharedCabinHolders_CabinSurvivedSize_df['Survived'] == 1 ]  
    
    
smallExperimentH_df = \
    smallExperimentB_df[ smallExperimentB_df['Survived'] == 0 ]
SharedTicketHoldersDeceased_TicketSurvivedSize_df = \
    SharedTicketHolders_TicketSurvivedSize_df[ \
        SharedTicketHolders_TicketSurvivedSize_df['Survived'] == 0 ]
SharedCabinHoldersDeceased_CabinSurvivedSize_df = \
    SharedCabinHolders_CabinSurvivedSize_df[ \
        SharedCabinHolders_CabinSurvivedSize_df['Survived'] == 0 ] 
    

smallExperimentHH_df = \
    smallExperimentB_df[ smallExperimentB_df['Survived'] == -1 ]
SharedTicketHoldersUnknown_TicketSurvivedSize_df = \
    SharedTicketHolders_TicketSurvivedSize_df[ \
        SharedTicketHolders_TicketSurvivedSize_df['Survived'] == -1 ]
SharedCabinHoldersUnknown_CabinSurvivedSize_df = \
    SharedCabinHolders_CabinSurvivedSize_df[ \
        SharedCabinHolders_CabinSurvivedSize_df['Survived'] == -1 ] 
    
    
titanicANDtest_df['TixWLive'] = 0
    
for tANDtindex, tANDtrow in titanicANDtest_df.iterrows():   
    thisTicket = str(tANDtrow['Ticket'])
    for myindex, myrow in \
        SharedTicketHoldersSurvivors_TicketSurvivedSize_df.iterrows():
        thatTicket = str(myrow['Ticket'])
        if thisTicket == thatTicket:
            titanicANDtest_df.at[tANDtindex, 'TixWLive'] = myrow['Size']
            
titanicANDtest_df['TixWLiveBinary'] = \
  titanicANDtest_df[['TixWLive']].apply( (lambda x: int(int(x) > 0) ), axis=1)
titanicANDtest_df['TxL'] = titanicANDtest_df['TixWLiveBinary']            
#$# titanicANDtest_df['TL'] = titanicANDtest_df['TixWLiveBinary']  
titanicANDtest_df['TL'] = titanicANDtest_df['TixWLive']
titanicANDtest_df['L'] = titanicANDtest_df['TixWLive']


titanicANDtest_df['TixWDead'] = 0

for tANDtindex, tANDtrow in titanicANDtest_df.iterrows():   
    thisTicket = str(tANDtrow['Ticket'])
    for myindex, myrow in \
        SharedTicketHoldersDeceased_TicketSurvivedSize_df.iterrows():
        thatTicket = str(myrow['Ticket'])
        if thisTicket == thatTicket:
            titanicANDtest_df.at[tANDtindex, 'TixWDead'] = myrow['Size']
            
titanicANDtest_df['TixWDeadBinary'] = \
  titanicANDtest_df[['TixWDead']].apply( (lambda x: int(int(x) > 0) ), axis=1)
titanicANDtest_df['TxD'] = titanicANDtest_df['TixWDeadBinary']             
#$# titanicANDtest_df['TD'] = titanicANDtest_df['TixWDeadBinary'] 
titanicANDtest_df['TD'] = titanicANDtest_df['TixWDead']
titanicANDtest_df['D'] = titanicANDtest_df['TixWDead']


titanicANDtest_df['TixWUnknown'] = 0
    
for tANDtindex, tANDtrow in titanicANDtest_df.iterrows():   
    thisTicket = str(tANDtrow['Ticket'])
    for myindex, myrow in \
        SharedTicketHoldersUnknown_TicketSurvivedSize_df.iterrows():
        thatTicket = str(myrow['Ticket'])
        if thisTicket == thatTicket:
            titanicANDtest_df.at[tANDtindex, 'TixWUnknown'] = myrow['Size']

titanicANDtest_df['TixWUnknownBinary'] = \
  titanicANDtest_df[['TixWUnknown']].apply((lambda x: int(int(x)>0)), axis=1)
titanicANDtest_df['TxU'] = titanicANDtest_df['TixWUnknownBinary']            
#$# titanicANDtest_df['TL'] = titanicANDtest_df['TixWLiveBinary']  
titanicANDtest_df['TU'] = titanicANDtest_df['TixWUnknown']
titanicANDtest_df['U'] = titanicANDtest_df['TixWUnknown']


titanicANDtest_df['LDU'] = \
    titanicANDtest_df['L'].astype(str) + \
    titanicANDtest_df['D'].astype(str) + \
    titanicANDtest_df['U'].astype(str)


LDUList = \
    titanicANDtest_df['LDU'].values.tolist()
    
LDUList = list( set ( LDUList ) ) 
   
LDUList = sorted(LDUList)


newtitanic_df = titanicANDtest_df.loc['titanic']

print('Pivot tables: Sex + MasMs+ SibSpB + ParchB + TixWLive + TixWDead, \
by Embark + Pclass.\n')

table0d = pd.pivot_table(newtitanic_df, values = 'Survived', \
          index = ['Fem','MMs','SbS','Pch','TxL','TxD'],\
          columns=['Pclass', 'Embarked'], \
          aggfunc=np.sum)
print( table0d.iloc[::-1],'\n' ) #$# This hack reverses the order of the rows.

table0e = pd.pivot_table(newtitanic_df, values = 'Survived', \
          index = ['Fem','MMs','SbS','Pch','TxL','TxD'],\
          columns=['Pclass', 'Embarked'], \
          aggfunc='count')
#$ print( table0e.iloc[::-1],'\n' )

table0f = pd.pivot_table(newtitanic_df, values = 'Survived', \
          index = ['Fem','MMs','SbS','Pch','TxL','TxD'],\
          columns=['Pclass', 'Embarked'], \
          aggfunc=np.mean)
#$# print( table0f.iloc[::-1],'\n' )


pd.set_option('display.max_rows', 700)

table0g = pd.pivot_table(titanicANDtest_df, values = 'Survived', \
          index = ['F','Ms','Sb','Pc','TL','TD'],\
          columns=['Pclass', 'Embarked'], \
          aggfunc=  lambda mySeries: (mySeries ==  1).sum() )
#$# print( table0g.iloc[::-1].round(2),'\n' )

table0h = pd.pivot_table(titanicANDtest_df, values = 'Survived', \
          index = ['F','Ms','Sb','Pc','TL','TD'],\
          columns=['Pclass', 'Embarked'], \
          aggfunc=  lambda mySeries: (mySeries ==  0).sum() )
#$# print( table0h.iloc[::-1].round(2),'\n' )

table0i = pd.pivot_table(titanicANDtest_df, values = 'Survived', \
          index = ['F','Ms','Sb','Pc','TL','TD'],\
          columns=['Pclass', 'Embarked'], \
          aggfunc= lambda mySeries: (mySeries==1).sum() + (mySeries==0).sum())
#$# print( table0i.iloc[::-1].round(2),'\n' )

table0j = pd.pivot_table(titanicANDtest_df, values = 'Survived', \
          index = ['F','Ms','Sb','Pc','TL','TD'],\
          columns=['Pclass', 'Embarked'], \
          aggfunc=  lambda mySeries: (mySeries == -1).sum() )
#$# print( table0j.iloc[::-1].round(2),'\n' )


def myQuotient(a,b):
    try:
        return float(a/b) 
    except TypeError:
        return float('nan')

table0k = pd.pivot_table(titanicANDtest_df, values = 'Survived', \
          index = ['F','Ms','Sb','Pc','TL','TD'],\
          columns=['Pclass', 'Embarked'], \
          aggfunc= lambda mySeries: \
             myQuotient( (mySeries==1).sum(),\
                         (mySeries==1).sum()+(mySeries==0).sum() ) 
          )                         
                         
print('Now consider TicketType, which is the first 2 characters of Ticket.')
print('This is a simple approach to the question of ticket adjacency.\n')

def getTicketType(ticket):
    myTicketString = ticket
    myTicketType = myTicketString[0]
    return myTicketType

#$# titanicANDtest_df['TicketType'] = \
#$#     titanicANDtest_df[['Ticket']].apply(getTicketType, axis=1)

titanicANDtest_df['TicketType'] = \
    titanicANDtest_df['Ticket'].astype(str).str[0:2]

tickettype_dummies_titanicANDtest = \
    pd.get_dummies(titanicANDtest_df['TicketType'])
titanicANDtest_df = titanicANDtest_df.join(tickettype_dummies_titanicANDtest)

titanicANDtest_df['TT'] = titanicANDtest_df['TicketType']


titanicANDtest_TickettypePclassSize_df = \
    titanicANDtest_df.groupby(['TT', \
        'Pclass']).size().to_frame('Size').reset_index()

titanicANDtest_TickettypePclassSize_df['PclassSizeScorer'] = \
    1000**( 3 - (titanicANDtest_TickettypePclassSize_df['Pclass']) ) \
    * (titanicANDtest_TickettypePclassSize_df['Size'])

titanicANDtest_TickettypePclassSizeScoreSum_df = \
    titanicANDtest_TickettypePclassSize_df.groupby( \
        ['TT'] )['PclassSizeScorer'].sum().to_frame('Score')
    
titanicANDtest_TickettypePclassRanked_df = \
    titanicANDtest_TickettypePclassSizeScoreSum_df.sort_values(by='Score', \
        ascending=False).reset_index()    
    
titanicANDtest_TickettypePclassRankedList = \
    titanicANDtest_TickettypePclassRanked_df['TT'].values.tolist()
    
TTlist = titanicANDtest_TickettypePclassRankedList



TTlistEdited = [ \
  'PC', '11', '12', '13', '14', '16', '17', '19', '20', '21', \
  '22', '23', '24', '25', '26', '27', '28', '29', '31', '32', \
  '33', '34', '35', '36', '37', '38', '39', '41', '45', '54', \
  '57', '65', '68', '69', '72', '75', '79', '84', '92', 'A.', \
  'A/', 'A4', 'AQ', 'C ', 'C.', 'CA', 'F.', 'Fa', 'LI', 'LP', \
  'P/', 'PP', 'S.', 'SC', 'SO', 'ST', 'SW', 'W.', 'W/', 'WE']

TTlistHelperA = [ \
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, \
   10,11,12,13,14,15,16,17,18,19, \
   20,21,22,23,24,25,26,27,28,29, \
   30,31,32,33,34,35,36,37,38,39, \
   40,41,42,43,44,45,46,47,48,49, \
   50,51,52,53,54,55,56,57,58,59  ]

TTlistHelperB = list( reversed ( TTlistHelperA ) )

TTlistHelper_df = pd.DataFrame(TTlistEdited, columns=['TTlistEdited'])
TTlistHelper_df['TTlistHelperA_df'] = TTlistHelperA
TTlistHelper_df['TTlistHelperB_df'] = TTlistHelperB

#$# print(TTlistHelper_df)

titanicANDtest_df['TTlistHelperA'] = \
    titanicANDtest_df['TT'].apply( lambda x: \
        TTlistHelper_df.loc[TTlistHelper_df['TTlistEdited'] == x].index[0] \
        )
titanicANDtest_df['TN'] = titanicANDtest_df['TTlistHelperA'] #Ticket numberer, alternative order

titanicANDtest_df['TTlistHelperB'] = 59 - titanicANDtest_df['TTlistHelperA']
titanicANDtest_df['Tn'] = titanicANDtest_df['TTlistHelperB'] # Ticket numberer


titanicANDtest_df['TicketId'] = \
    titanicANDtest_df['Ticket'].astype(str).str[-4:]
    
titanicANDtest_df['TicketId'].loc[titanicANDtest_df['SharedTicket?'] == 0] \
                 = ' 000'    

ticketId_dummies_titanicANDtest = \
    pd.get_dummies(titanicANDtest_df['TicketId'])
titanicANDtest_df = titanicANDtest_df.join(ticketId_dummies_titanicANDtest)

titanicANDtest_df['Tid'] = titanicANDtest_df['TicketId']


TidList = \
    titanicANDtest_df['Tid'].values.tolist()
    
TidList = list( set ( TidList ) ) 
   
TidList = sorted(TidList)


titanicANDtest_df['Pclass'] = titanicANDtest_df['Pclass'].astype(str)

table0l = pd.pivot_table(titanicANDtest_df, values = 'Survived', \
          index = ['TT','TL','TD','F','m','s','p'],\
          columns=['Pclass', 'Embarked'], \
          aggfunc=  lambda mySeries: (mySeries ==  1).sum() )
#$# print( table0l.iloc[::-1].round(2),'\n' )

table0m = pd.pivot_table(titanicANDtest_df, values = 'Survived', \
          index = ['TT','TL','TD','F','m','s','p'],\
          columns=['Pclass', 'Embarked'], \
          aggfunc=  lambda mySeries: (mySeries ==  0).sum() )
#$# print( table0m.iloc[::-1].round(2),'\n' )

table0n = pd.pivot_table(titanicANDtest_df, values = 'Survived', \
          index = ['TT','TL','TD','F','m','s','p'],\
          columns=['Pclass', 'Embarked'], \
          aggfunc= lambda mySeries: (mySeries==1).sum() + (mySeries==0).sum())
#$# print( table0n.iloc[::-1].round(2),'\n' )

table0o = pd.pivot_table(titanicANDtest_df, values = 'Survived', \
          index = ['TT','TL','TD','F','m','s','p'],\
          columns=['Pclass', 'Embarked'], \
          aggfunc=  lambda mySeries: (mySeries == -1).sum() )
#$# print( table0o.iloc[::-1].round(2),'\n' )

table0p = pd.pivot_table(titanicANDtest_df, values = 'Survived', \
          index = ['TT','TL','TD','F','m','s','p'],\
          columns=['Pclass', 'Embarked'], \
          aggfunc= lambda mySeries: \
             myQuotient( (mySeries==1).sum(),\
                         (mySeries==1).sum()+(mySeries==0).sum() )
          )
print( table0p.iloc[::-1].round(2),'\n' )
             
pd.set_option('display.column_space', 0) 
pd.set_option('display.expand_frame_repr', False)


pd.set_option('display.width', 72)

table0q = pd.pivot_table(titanicANDtest_df, values = 'Survived', \
          index = ['Tn','TT','LDU','Tid','F','m','s','p'],\
          columns=['Pclass', 'Embarked'], \
          aggfunc= lambda mySeries: \
             str((mySeries== 1).sum()) + str(',') + \
             str((mySeries== 0).sum()) + str(',') + \
             str((mySeries==-1).sum())
          ) 

table0q.columns = [''.join(col).strip() for col in table0q.columns.values]

table0q = table0q.iloc[::-1]


table0r = pd.DataFrame(table0q.iloc[::-1].to_records())
table0r = table0r.drop('Tn',axis=1)
table0r.set_index('TT', inplace=True)
#pandas 0.18.0 and higher
table0r = table0r.rename_axis(None)


table0s = pd.pivot_table(titanicANDtest_df, values = 'Survived', \
          index = ['Tn','TT','LDU','Tid','F','m','s','p'],\
          columns=['Pclass', 'Embarked'], \
          aggfunc= lambda mySeries: \
                         str(
                             (mySeries== 1).sum() * 10000 + \
                             (mySeries== 0).sum() * 100 + \
                             (mySeries==-1).sum() * 1 \
                            ).rjust(6, '0') \
          ) 

table0s.columns = [''.join(col).strip() for col in table0s.columns.values]

table0s = table0s.iloc[::-1]


from numpy import nan
table0q.fillna(value='0,0,0', inplace=True)
#$# print( table0q )

table0s.fillna(value='000000', inplace=True)
print( table0s )
def _backgroundcolor(val):
    bgcolor = 'white'
    living  = int(val[0])*10 + int(val[1])
    dead    = int(val[2])*10 + int(val[3])
    unknown = int(val[4])*10 + int(val[5])
    if ((living == 0) and (dead == 0) and (unknown == 0)):
        bgcolor = 'white'
    elif ((living == 0) and (dead == 0)):
        bgcolor = 'lightgrey'
    elif ((dead == 0) or (living / (living + dead) >= 0.85)):
        bgcolor = 'lightgreen'
    elif (living / (living + dead) >= 0.5):
        bgcolor = 'lightblue'
    elif (living / (living + dead >= 0.15)):
        bgcolor = 'orange'
    else:
        bgcolor = 'pink'
    return 'background-color: %s' % bgcolor


#$# ss = table0q.style.applymap(_backgroundcolor)
ss = table0s.style.applymap(_backgroundcolor)
ss
def magnify():
    return [dict(selector="th",
                 props=[("font-size", "6pt")]),
            dict(selector="td",
                 props=[('padding', "0em 0em")]),
            dict(selector="th:hover",
                 props=[("font-size", "18pt")]),
            dict(selector="tr:hover td:hover",
                 props=[('max-width', '400px'),
                        ('font-size', '18pt')])
]
#$# np.random.seed(25)
#$# cmap = cmap=sns.diverging_palette(5, 250, as_cmap=True)
#$# bigdf = pd.DataFrame(np.random.randn(40, 25)).cumsum()

"""
bigdf.style.background_gradient(cmap, axis=1)\
    .set_properties(**{'max-width': '80px', 'font-size': '1pt'})\
    .set_caption("Hover to magnify")\
    .set_precision(2)\
    .set_table_styles(magnify())
"""

sss = table0s.style.applymap(_backgroundcolor)\
    .set_properties(**{'max-width': '80px', 'font-size': '1pt'})\
    .set_caption("Hover to magnify")\
    .set_table_styles(magnify())
    
sss    
table0t = pd.pivot_table(titanicANDtest_df, values = 'Survived', \
          index = ['Tn','TT','LDU','Tid'],\
          columns=['Pclass', 'Embarked'], \
          aggfunc= lambda mySeries: \
                         str(
                             (mySeries== 1).sum() * 10000 + \
                             (mySeries== 0).sum() * 100 + \
                             (mySeries==-1).sum() * 1 \
                            ).rjust(6, '0') \
          ) 

table0t.columns = [''.join(col).strip() for col in table0t.columns.values]
table0t = table0t.iloc[::-1]

table0t.fillna(value='000000', inplace=True)
#$# print( table0t )

ttt = table0t.style.applymap(_backgroundcolor)\
    .set_properties(**{'max-width': '80px', 'font-size': '1pt'})\
    .set_caption("Hover to magnify")\
    .set_table_styles(magnify())
    
ttt    
table0u = table0t.T

uuu = table0u.style.applymap(_backgroundcolor)\
    .set_properties(**{'max-width': '80px', 'font-size': '1pt'})\
    .set_caption("Hover to magnify")\
    .set_table_styles(magnify())

uuu
table0v = pd.pivot_table(titanicANDtest_df, values = 'Survived', \
          index = ['F','m','s','p'],\
          columns=['Pclass', 'Embarked'], \
          aggfunc= lambda mySeries: \
                         str(
                             (mySeries== 1).sum() * 10000 + \
                             (mySeries== 0).sum() * 100 + \
                             (mySeries==-1).sum() * 1 \
                            ).rjust(6, '0') \
          ) 

table0v.columns = [''.join(col).strip() for col in table0v.columns.values]
table0v = table0v.iloc[::-1]

table0v.fillna(value='000000', inplace=True)
#$# print( table0t )

vvv = table0v.style.applymap(_backgroundcolor)\
    .set_properties(**{'max-width': '80px', 'font-size': '1pt'})\
    .set_caption("Hover to magnify")\
    .set_table_styles(magnify())
    
vvv    
table0w = pd.pivot_table(titanicANDtest_df, values = 'Survived', \
          index = ['F','m'],\
          columns=['Pclass', 'Embarked'], \
          aggfunc= lambda mySeries: \
                         str(
                             (mySeries== 1).sum() * 10000 + \
                             (mySeries== 0).sum() * 100 + \
                             (mySeries==-1).sum() * 1 \
                            ).rjust(6, '0') \
          ) 

table0w.columns = [''.join(col).strip() for col in table0w.columns.values]
table0w = table0w.iloc[::-1]

table0w.fillna(value='000000', inplace=True)
print( table0w )

www = table0w.style.applymap(_backgroundcolor)\
    .set_properties(**{'max-width': '80px', 'font-size': '1pt'})\
    .set_caption("Hover to magnify")\
    .set_table_styles(magnify())
    
www 
table0x = pd.pivot_table(titanicANDtest_df, values = 'Survived', \
          index = ['F'],\
          columns=['Pclass', 'Embarked'], \
          aggfunc= lambda mySeries: \
                         str(
                             (mySeries== 1).sum() * 10000 + \
                             (mySeries== 0).sum() * 100 + \
                             (mySeries==-1).sum() * 1 \
                            ).rjust(6, '0') \
          ) 

table0x.columns = [''.join(col).strip() for col in table0x.columns.values]
table0x = table0x.iloc[::-1]

table0x.fillna(value='000000', inplace=True)
print( table0x )

xxx = table0x.style.applymap(_backgroundcolor)\
    .set_properties(**{'max-width': '480px', 'font-size': '6pt'})\
    .set_caption("Hover to magnify")\
    .set_table_styles(magnify())
    
xxx 
def _backgroundcolor3(val):
    bgcolor = 'white'
    living  = int(val[0])*100 + int(val[1])*10 + int(val[2])
    dead    = int(val[4])*100 + int(val[5])*10 + int(val[6])
    unknown = int(val[8])*100 + int(val[9])*10 + int(val[10])
    if ((living == 0) and (dead == 0) and (unknown == 0)):
        bgcolor = 'white'
    elif ((living == 0) and (dead == 0)):
        bgcolor = 'lightgrey'
    elif ((dead == 0) or (living / (living + dead) >= 0.85)):
        bgcolor = 'lightgreen'
    elif (living / (living + dead) >= 0.5):
        bgcolor = 'lightblue'
    elif (living / (living + dead) >= 0.15):
        bgcolor = 'orange'
    else:
        bgcolor = 'pink'
    return 'background-color: %s' % bgcolor


table1a = pd.pivot_table(titanicANDtest_df, values = 'Survived', \
          index = ['F'],\
          columns=['Pclass', 'Embarked'], \
          aggfunc= lambda mySeries: \
                         str( (mySeries== 1).sum() ).rjust(3, '0') + ',' + \
                         str( (mySeries== 0).sum() ).rjust(3, '0') + ',' + \
                         str( (mySeries==-1).sum() ).rjust(3, '0')         \
          ) 

table1a.columns = [''.join(col).strip() for col in table1a.columns.values]
table1a = table1a.iloc[::-1]

table1a.fillna(value='000,000,000', inplace=True)
#$# print( table1a )

print('\n','Statistics of Passengers in the Titanic Data Set:')
print(' each cell has the form lllddduuu, counting those who lived, died, or unknown status.')
print(' The headers denote passenger class 1,2,3, and embarkation port Cherbourg, Queenstown, Southampton.')
print(' This table: F = 1 is female; F = 0 is male.')

aaaa = table1a.style.applymap(_backgroundcolor3)\
    .set_properties(**{'max-width': '480px', 'font-size': '6pt'})\
    .set_caption("Hover to magnify")\
    .set_table_styles(magnify())
    
aaaa 
table1b = pd.pivot_table(titanicANDtest_df, values = 'Survived', \
          index = ['F','m'],\
          columns=['Pclass', 'Embarked'], \
          aggfunc= lambda mySeries: \
                         str( (mySeries== 1).sum() ).rjust(3, '0') + ',' + \
                         str( (mySeries== 0).sum() ).rjust(3, '0') + ',' + \
                         str( (mySeries==-1).sum() ).rjust(3, '0')         \
          )
table1b.columns = [''.join(col).strip() for col in table1b.columns.values]
table1b = table1b.iloc[::-1]
table1b.fillna(value='000,000,000', inplace=True)

#$# print( table1a,'\n' )
#$# print( table1b )

print('\n','Statistics of Passengers in the Titanic Data Set:')
print(' each cell has the form lllddduuu, counting those who lived, died, or unknown status.')
print(' The headers denote passenger class 1,2,3, and embarkation port Cherbourg, Queenstown, Southampton.')
print(' This table: m = 1 is master/miss (i.e. a youth, or an unmarried woman); m = 0 is married woman or adult man.')

bbbb = table1b.style.applymap(_backgroundcolor3)\
    .set_properties(**{'max-width': '480px', 'font-size': '6pt'})\
    .set_caption("Hover to magnify")\
    .set_table_styles(magnify())
    
bbbb
table1c = pd.pivot_table(titanicANDtest_df, values = 'Survived', \
          index = ['F','m','s'],\
          columns=['Pclass', 'Embarked'], \
          aggfunc= lambda mySeries: \
                         str( (mySeries== 1).sum() ).rjust(3, '0') + ',' + \
                         str( (mySeries== 0).sum() ).rjust(3, '0') + ',' + \
                         str( (mySeries==-1).sum() ).rjust(3, '0')         \
          )
table1c.columns = [''.join(col).strip() for col in table1c.columns.values]
table1c = table1c.iloc[::-1]
table1c.fillna(value='000,000,000', inplace=True)

#$# print( table1b,'\n' )
#$# print( table1c )

print('\n','Statistics of Passengers in the Titanic Data Set:')
print(' each cell has the form lllddduuu, counting those who lived, died, or unknown status.')
print(' The headers denote passenger class 1,2,3, and embarkation port Cherbourg, Queenstown, Southampton.')
print(' This table: s = 1 is travelling with a spouse or sibling; s = 0 is travelling with no spouse/sibling.')

cccc = table1c.style.applymap(_backgroundcolor3)\
    .set_properties(**{'max-width': '480px', 'font-size': '6pt'})\
    .set_caption("Hover to magnify")\
    .set_table_styles(magnify())
    
cccc
table1d = pd.pivot_table(titanicANDtest_df, values = 'Survived', \
          index = ['F','m','s','p'],\
          columns=['Pclass', 'Embarked'], \
          aggfunc= lambda mySeries: \
                         str( (mySeries== 1).sum() ).rjust(3, '0') + ',' + \
                         str( (mySeries== 0).sum() ).rjust(3, '0') + ',' + \
                         str( (mySeries==-1).sum() ).rjust(3, '0')         \
          )
table1d.columns = [''.join(col).strip() for col in table1d.columns.values]
table1d = table1d.iloc[::-1]
table1d.fillna(value='000,000,000', inplace=True)

#$# print( table1c,'\n' )
#$# print( table1d )

print('\n','Statistics of Passengers in the Titanic Data Set:')
print(' each cell has the form lllddduuu, counting those who lived, died, or unknown status.')
print(' The headers denote passenger class 1,2,3, and embarkation port Cherbourg, Queenstown, Southampton.')
print(' This table: p = 1 is travelling with a parent or child; p = 0 is travelling with no parent/child.')

dddd = table1d.style.applymap(_backgroundcolor3)\
    .set_properties(**{'max-width': '480px', 'font-size': '6pt'})\
    .set_caption("Hover to magnify")\
    .set_table_styles(magnify())
    
dddd
table1e = pd.pivot_table(titanicANDtest_df, values = 'Survived', \
          index = ['F','m','s','p'],\
          columns=['Pclass', 'Embarked','TN','TT'], \
          aggfunc= lambda mySeries: \
                         str( (mySeries== 1).sum() ).rjust(3, '0') + ',' + \
                         str( (mySeries== 0).sum() ).rjust(3, '0') + ',' + \
                         str( (mySeries==-1).sum() ).rjust(3, '0')         \
          ) 
#$# table1e.columns = [''.join(col).strip() for col in table1e.columns.values]
table1e = table1e.iloc[::-1]
table1e.fillna(value='000,000,000', inplace=True)

#$# pd.set_option('display.width', 1000)
#$# pd.set_option('display.max_columns', 500)
#$# pd.set_option('display.expand_frame_repr', False)
#$# pd.set_option('max_colwidth', 200)


#$# print( table1d,'\n' )
#$# print( table1e )

print('\n','Statistics of Passengers in the Titanic Data Set:')
print(' each cell has the form lllddduuu, counting those who lived, died, or unknown status.')
print(' The headers denote passenger class 1,2,3, and embarkation port Cherbourg, Queenstown, Southampton.')
print(' This table: header TT is ticket-type: the first 2 letters of passenger ticket.  TN is synonymous.')

eeee = table1e.style.applymap(_backgroundcolor3)\
    .set_properties(**{'max-width': '480px', 'font-size': '6pt'})\
    .set_caption("Hover to magnify")\
    .set_table_styles(magnify())
    
eeee
table1f = pd.pivot_table(titanicANDtest_df, values = 'Survived', \
          index = ['F','m','s','p'],\
          columns=['Pclass', 'Embarked','TN','TT','LDU'], \
          aggfunc= lambda mySeries: \
                         str( (mySeries== 1).sum() ).rjust(3, '0') + ',' + \
                         str( (mySeries== 0).sum() ).rjust(3, '0') + ',' + \
                         str( (mySeries==-1).sum() ).rjust(3, '0')         \
          )
#$# table1f.columns = [''.join(col).strip() for col in table1f.columns.values]
table1f = table1f.iloc[::-1]
table1f.fillna(value='000,000,000', inplace=True)

#$# print( table1d,'\n' )
#$# print( table1f )

print('\n','Statistics of Passengers in the Titanic Data Set:')
print(' each cell has the form lllddduuu, counting those who lived, died, or unknown status.')
print(' The headers denote passenger class 1,2,3, and embarkation port Cherbourg, Queenstown, Southampton.')
print(' This table: header LDU is the count of living-dead-unknown for each ticket that has more than one passenger.')
print(' Note: within each ticket-type, the header 000 collects the passengers who travelled on an unshared ticket.')

ffff = table1f.style.applymap(_backgroundcolor3)\
    .set_properties(**{'max-width': '480px', 'font-size': '6pt'})\
    .set_caption("Hover to magnify")\
    .set_table_styles(magnify())
    
ffff
table1g = pd.pivot_table(titanicANDtest_df, values = 'Survived', \
          index = ['F','m','s','p'],\
          columns=['Pclass', 'Embarked','TN','TT','LDU','Tid'], \
          aggfunc= lambda mySeries: \
                         str( (mySeries== 1).sum() ).rjust(3, '0') + ',' + \
                         str( (mySeries== 0).sum() ).rjust(3, '0') + ',' + \
                         str( (mySeries==-1).sum() ).rjust(3, '0')         \
          ) 
#$# table1g.columns = [''.join(col).strip() for col in table1g.columns.values]
table1g = table1g.iloc[::-1]
table1g.fillna(value='000,000,000', inplace=True)

#$# print( table1d,'\n' )
#$# print( table1g )

print('\n','Statistics of Passengers in the Titanic Data Set:')
print(' each cell has the form lllddduuu, counting those who lived, died, or unknown status.')
print(' The headers denote passenger class 1,2,3, and embarkation port Cherbourg, Queenstown, Southampton.')
print(' This table: header Tid is the final 4 characters of each ticket that has more than one passenger.')
print(' Note: in each ticket-type, the Tid-header 000 collects the passengers who travelled on an unshared ticket.')

gggg = table1g.style.applymap(_backgroundcolor3)\
    .set_properties(**{'max-width': '480px', 'font-size': '6pt'})\
    .set_caption("Hover to magnify")\
    .set_table_styles(magnify())
    
gggg
