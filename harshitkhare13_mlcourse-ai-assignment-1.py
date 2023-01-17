# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import numpy as np
import pandas as pd
data = pd.read_csv('../input/athlete_events.csv')
data.head()
'''
1. How old were the youngest male and female participants of the 1996 Olympics?
16 and 15
14 and 12
16 and 12
13 and 11
'''
data1996 = data[data['Year']==1996][['Age','Sex']]
print(data1996[data['Sex']=='M']['Age'].min())
print(data1996[data['Sex']=='F']['Age'].min())
'''
2. What was the percentage of male gymnasts among all the male participants of the 2000 Olympics? 
Round the answer to the first decimal.
Hint: here and further if needed drop duplicated sportsmen to count only unique ones.
0.2
1.5
2.5
7.7
'''
dM = data[data.Sex == 'M'][['Name','Sport','Year']]
dMY = dM[dM.Year == 2000][['Name','Sport']]
DMYG = dMY[dMY.Sport == 'Gymnastics']['Name']
dMY = dMY['Name']
DU = set(DMYG)
dMY = set(dMY)
print(len(DU))
print(len(dMY))
print(round(len(DU)/len(dMY)*100,1))


'''
3. What are the mean and standard deviation of height for female basketball players participated
in the 2000 Olympics? Round the answer to the first decimal.
178.5 and 7.2
179.4 and 10
180.7 and 6.7
182.4 and 9.1
'''
dF = data[data.Sex == 'F'][['Name','Height','Sport','Year']]
dFY = dF[dF.Year == 2000][['Name','Sport','Height']]
DFYB = dF[dF.Sport == 'Basketball'][['Name','Height']]
print(round(np.mean(DFYB.Height),2))
print(np.std(DFYB.Height))

'''
4. Find a sportsperson participated in the 2002 Olympics, with the highest weight among 
other participants of the same Olympics. What sport did he or she do?
Judo
Bobsleigh
Weightlifting
Boxing
'''
dY = data[data.Year == 2002][['Sport','Weight']]
dY[dY.Weight == max(dY.Weight)]['Sport']
'''
5. How many times did Pawe Abratkiewicz participate in the Olympics held in different years?
0
1
2
3
'''
data[data.Name == 'Pawe Abratkiewicz'][['Name','Year']]
'''
6. How many silver medals in tennis did Australia win at the 2000 Olympics?
0
1
2
3
'''
data.head()
dY = data[data.Year == 2000][['Sport','Medal','Team']]
dYA = dY[dY.Team == 'Australia'][['Sport','Medal']]
dYAT = dYA[dYA.Sport == 'Tennis']['Medal']
dYAT == 'Silver'
'''
7. Is it true that Switzerland won fewer medals than Serbia at the 2016 Olympics? 
Do not consider NaN values in Medal column.
Yes
No
'''
dY = data[data.Year == 2016][['Medal','Team']]
dYS = dY[dY.Team == 'Switzerland']['Medal']
dYSe = dY[dY.Team == 'Serbia']['Medal']
dYS.dropna()
dYSe.dropna()
len(dYS) < len(dYSe)
'''
8. What age category did the fewest and the most participants of the 2014 Olympics belong to?
[45-55] and [25-35) correspondingly
[45-55] and [15-25) correspondingly
[35-45] and [25-35) correspondingly
[45-55] and [35-45) correspondingly
'''
dY = data[data.Year == 2014][['Age']]
dYa = dY[dY.Age >= 45]
print(len(dYa[dYa.Age <= 55]))
dYb = dY[dY.Age >= 35]
print(len(dYb[dYb.Age <= 45]))
dYc = dY[dY.Age >= 25]
print(len(dYc[dYc.Age <= 35]))
dYd = dY[dY.Age >= 15]
print(len(dYd[dYd.Age <= 25]))
'''
9. Is it true that there were Summer Olympics held in Lake Placid? 
Is it true that there were Winter Olympics held in Sankt Moritz?
Yes, Yes
Yes, No
No, Yes
No, No
'''
dS = data[data['Season']=='Summer'][['City']]
print(dS[dS.City == 'Lake Placid'])
dW = data[data['Season']=='Winter'][['City','Season']]
print(dW[dW.City == 'Sankt Moritz'].head())
'''
10. What is the absolute difference between the number of unique sports
at the 1995 Olympics and 2016 Olympics?
16
24
26
34
'''
d1995 = data[data.Year == 1995]['Sport']
d2016 = data[data.Year == 2016]['Sport']
d1995u = set(d1995)
d2016u = set(d2016)
print(abs(len(d1995u) - len(d2016u)))
