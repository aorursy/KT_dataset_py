# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.

# -*- coding: utf-8 -*-  

import matplotlib.pyplot as plt

# import csv

# from sklearn.ensemble import RandomForestClassifier

    

df = pd.read_csv('../input/train.csv')

list_name=df.columns   #name of each column

data=df.values

N=len(data[0::,1])

number_survived = np.sum(data[0::,1].astype(np.float))

women_number=data[0::,4]=='female'   #sum(women_number) is the number of women

men_number=data[0::,4]!='female'

women_survived=sum(data[women_number,1])

men_survived=sum(data[men_number,1])

data_men=(men_survived,sum(men_number)-men_survived)

data_women=(women_survived,sum(women_number)-women_survived)

print("the survived and non-survived gender: %d, %d,%d,%d" % (men_survived,women_survived,sum(men_number),sum(women_number)))

# the sex survive

plt.figure(1)

group=2

opacity = 0.4

index=np.arange(group)

bar_width = 0.3

plt.bar(index,data_men,bar_width, alpha=opacity, color='g',label='Men' )

plt.bar(index+bar_width, data_women, bar_width, alpha=opacity, color='r', label='Women' )    

plt.xlabel('Survive')    

plt.ylabel('Number')    

plt.text(0.12,115,'109')

plt.text(0.4,245,'233')

plt.text(1.1,480,'468')

plt.text(1.4,90,'81')

plt.xticks(index+bar_width,('1','0'),fontsize =18)  

plt.title('The Survived with Gender')

plt.legend()

plt.grid(True)

#the age survive

plt.figure(2)

plt.subplot(211)

df['Age'].dropna().hist(bins=100)

plt.ylabel('Number')  

plt.xlim(0,80.5)     

plt.subplot(212)

hist, bin_edges =np.histogram(df['Age'].dropna(),bins=100)

# np.hstack((0,bin_edges))   #add an element 0

number=np.array([0.0 for i in range(len(hist))])

mean_age=np.array([0.0 for i in range(len(hist))])

for i in range(len(hist)):

	#survived in specific age domain, except for some blank age data in two method:  AND operation bit-by-bit or loop

	number[i]=sum(data[ (data[0::,5] >=bin_edges[i]) & (data[0::,5] <=bin_edges[i+1]), 1])

# 	for j in xrange(N):

# 		if (bin_edges[i]<=data[0::,5][j]<=bin_edges[i+1] and data[0::,1][j]==1):

# 			number[i]=number[i]+1

	temp=data[ (data[0::,5] >=bin_edges[i])&(data[0::,5] <=bin_edges[i+1]), 5]

	# temp=df['Age'].dropna().values[ (df['Age'].dropna().values>=bin_edges[i])&(df['Age'].dropna().values<=bin_edges[i+1])]

	if len(temp)!=0:

		mean_age[i]=temp.mean()

# print len(mean_age[mean_age!=0.0])

plt.plot(mean_age[mean_age!=0.0],(number/sum(hist))[mean_age!=0.0],'rx')

plt.plot(mean_age[mean_age!=0.0],((hist-number)/sum(hist))[mean_age!=0.0],'b*')

plt.legend(("survived","non-survived"),loc='best')

plt.xlabel('Age')    

plt.ylabel('Percent')   

plt.xlim(0,80.5)     

plt.ylim(0,0.035)     

#the embarked survive 

plt.figure(3)

S_number=data[0::,11]=='S'

C_number=data[0::,11]=='C'

Q_number=data[0::,11]=='Q'

S_survived=sum(data[S_number,1])

C_survived=sum(data[C_number,1])

Q_survived=sum(data[Q_number,1])

Sdata=(S_survived,sum(S_number)-S_survived)

Cdata=(C_survived,sum(C_number)-C_survived)

Qdata=(Q_survived,sum(Q_number)-Q_survived)

plt.bar(index,Sdata,bar_width, alpha=opacity, color='g',label='S' )

plt.bar(index+bar_width, Cdata, bar_width, alpha=opacity, color='r', label='C' ) 

plt.bar(index+2*bar_width, Qdata, bar_width, alpha=opacity, color='b', label='Q' ) 

plt.xlabel('Survive')    

plt.ylabel('Number')   

plt.xticks(index+1.5*bar_width,('1','0'),fontsize =18)  

plt.grid(True)

plt.title('The Survived with different embarked')

plt.legend()

#Pclass data

plt.figure(4)

Pclass1=sum((data[0::,2]==1) & (data[0::,1]==1)).astype(np.float)  #don't forget the bracket

Pclass2=sum((data[0::,2]==2) & (data[0::,1]==1)).astype(np.float)

Pclass3=sum((data[0::,2]==3) & (data[0::,1]==1)).astype(np.float)

print('survived Pclass Percent for 1,2,3:%s,%s,%s' % (Pclass1/number_survived,Pclass2/number_survived,Pclass3/number_survived))

plt.plot([1,2,3],[sum(data[0::,2]==1),sum(data[0::,2]==2),sum(data[0::,2]==3)],'ro-',linewidth=2)

plt.plot([1,2,3],[sum((data[0::,2]==1) & (data[0::,1]==1)),sum((data[0::,2]==2) & \

	(data[0::,1]==1)),sum((data[0::,2]==1) & (data[0::,1]==1))],'g*-',linewidth=2)

plt.xlabel('Pclass')    

plt.ylabel('Numbers') 

plt.legend(("Total","Survived"),loc='best')

plt.xticks([1,2,3])    

plt.xlim(0.5,3.5)     

plt.grid(True)

plt.title('The Survived with different Pclass')

#with or not with family

plt.figure(5)

with_family=sum((data[0::,6]!=0) | (data[0::,7]!=0)).astype(np.float)  #don't forget the bracket

not_with_family=sum((data[0::,6]==0) & (data[0::,7]==0)).astype(np.float)

with_family_survived=sum(((data[0::,6]==0) | (data[0::,7]==0)) & (data[0::,1]==1)).astype(np.float)

not_with_family_survived=sum((data[0::,6]==0) & (data[0::,7]==0) & (data[0::,1]==1)).astype(np.float)

print('survived with or ont with family:%s,%s' % (with_family_survived/with_family,not_with_family_survived/not_with_family))

plt.plot([0,1],[with_family,not_with_family],'ro-',linewidth=2)

plt.plot([0,1],[with_family_survived,not_with_family_survived],'gs-',linewidth=2)

plt.xlabel('Family')    

plt.ylabel('Numbers') 

plt.xticks([0,1],('With','Alone'))

plt.xlim(-0.5,1.5)     

plt.legend(("Total","Survived"),loc='best')

plt.grid(True)

plt.show() 