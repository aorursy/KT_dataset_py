import pandas as pd 
import numpy as np 
import csv

ifile=pd.read_csv('test.csv',header=0)

ifile['Gender']=ifile['Sex'].map({'female':0,'male':1}).astype(int)


missingages=np.zeros((2,3))

for i in xrange(0,2):
	for j in xrange(0,3):
			missingages[i,j]=ifile[(ifile['Gender']==i) & (ifile['Pclass']==j+1)]['Age'].dropna().median()


ifile['AgeN']=ifile['Age']

for i in xrange(0,2):
	for j in xrange(0,3):
		ifile.loc[(ifile['Age'].isnull())&(ifile['Gender']==i)&(ifile['Pclass']==j+1),'AgeN']=missingages[i,j]




ifile['Start']=ifile['Embarked'].dropna().map({'S':0,'C':1,'Q':2}).astype(int)

ifile.loc[ifile['Start'].isnull(),'Start']=ifile['Start'].median()





ifile.loc[ifile['Fare']<=8,'FareClass']=0
ifile.loc[(ifile['Fare']<=15) & (ifile['Fare']>8),'FareClass']=1
ifile.loc[(ifile['Fare']<=39) & (ifile['Fare']>15),'FareClass']=2
ifile.loc[ifile['Fare']>39,'FareClass']=3

ifile.loc[ifile['AgeN']<=18,'AgeClass']=0
ifile.loc[(ifile['AgeN']<=30) & (ifile['AgeN']>18),'AgeClass']=1
ifile.loc[(ifile['AgeN']<=55) & (ifile['AgeN']>30),'AgeClass']=2
ifile.loc[ifile['AgeN']>55,'AgeClass']=3






ifile.loc[ifile['FareClass'].isnull(),'FareClass']=ifile['FareClass'].median()
ifile['Status1']=ifile['FareClass']*ifile['Pclass']

ifile['Famsize']=ifile['SibSp']+ifile['Parch']
ifile['Status2']=ifile['AgeClass']*ifile['Famsize']
ifile['Status3']=ifile['Gender']*ifile['Famsize']

ifile = ifile.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked','PassengerId','Age','SibSp','Parch','Fare','AgeN'], axis=1) 
ifile=ifile.dropna()

print( ifile)
train_data=ifile.values
ofile=open('finaltest.csv',"wb")
writer=csv.writer(ofile)

for row in train_data:
	writer.writerow(row)


