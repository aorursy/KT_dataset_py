# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
a_solar=pd.read_csv("/kaggle/input/a_solar.txt")

b_dream=pd.read_csv("/kaggle/input/b_dream.txt")

c_soup=pd.read_csv("/kaggle/input/c_soup.txt")

d_maelstrom=pd.read_csv("/kaggle/input/d_maelstrom.txt")

e_igloos=pd.read_csv("/kaggle/input/e_igloos.txt")

f_glitch=pd.read_csv("/kaggle/input/f_glitch.txt")
print(a_solar['5 3'][0:int(a_solar.columns[0].split()[1])])
print(b_dream['100 100'][0:int(b_dream.columns[0].split()[1])])
print(c_soup['200 200'][0:int(c_soup.columns[0].split()[1])])
print(d_maelstrom['300 200'][0:int(d_maelstrom.columns[0].split()[1])])
print(e_igloos['500 400'][0:int(e_igloos.columns[0].split()[1])])
print(f_glitch['500 400'][0:int(f_glitch.columns[0].split()[1])])
dev_space_a = list()

man_space_a = list()

print("free seats for developers:")

for i in range(int(a_solar.columns[0].split()[1])):

    for j in range(int(a_solar.columns[0].split()[0])):

        if (a_solar.iloc[i,0][j] == '_'):

            print(j,i)

            dev_space_a.append(str(j) +' '+ str(i))

print("free seats for managers:")            

for i in range(int(a_solar.columns[0].split()[1])):

    for j in range(int(a_solar.columns[0].split()[0])):

        if (a_solar.iloc[i,0][j] == 'M'):

            print(j,i)

            man_space_a.append(str(j) +' '+ str(i))
dev_space_b = list()

man_space_b = list()

print("free seats for developers:")

for i in range(int(b_dream.columns[0].split()[1])):

    for j in range(int(b_dream.columns[0].split()[0])):

        if (b_dream.iloc[i,0][j] == '_'):

            print(j,i)

            dev_space_b.append(str(j) +' '+ str(i))

print("free seats for managers:")            

for i in range(int(b_dream.columns[0].split()[1])):

    for j in range(int(b_dream.columns[0].split()[0])):

        if (b_dream.iloc[i,0][j] == 'M'):

            print(j,i)

            man_space_b.append(str(j) +' '+ str(i))
dev_space_c = list()

man_space_c = list()

print("free seats for developers:")

for i in range(int(c_soup.columns[0].split()[1])):

    for j in range(int(c_soup.columns[0].split()[0])):

        if (c_soup.iloc[i,0][j] == '_'):

            print(j,i)

            dev_space_c.append(str(j) +' '+ str(i))

print("free seats for managers:")            

for i in range(int(c_soup.columns[0].split()[1])):

    for j in range(int(c_soup.columns[0].split()[0])):

        if (c_soup.iloc[i,0][j] == 'M'):

            print(j,i)

            man_space_c.append(str(j) +' '+ str(i))
dev_space_d = list()

man_space_d = list()

print("free seats for developers:")

for i in range(int(d_maelstrom.columns[0].split()[1])):

    for j in range(int(d_maelstrom.columns[0].split()[0])):

        if (d_maelstrom.iloc[i,0][j] == '_'):

            print(j,i)

            dev_space_d.append(str(j) +' '+ str(i))

print("free seats for managers:")            

for i in range(int(d_maelstrom.columns[0].split()[1])):

    for j in range(int(d_maelstrom.columns[0].split()[0])):

        if (d_maelstrom.iloc[i,0][j] == 'M'):

            print(j,i)

            man_space_d.append(str(j) +' '+ str(i))
dev_space_e = list()

man_space_e = list()

print("free seats for developers:")

for i in range(int(e_igloos.columns[0].split()[1])):

    for j in range(int(e_igloos.columns[0].split()[0])):

        if (e_igloos.iloc[i,0][j] == '_'):

            print(j,i)

            dev_space_e.append(str(j) +' '+ str(i))

print("free seats for managers:")            

for i in range(int(e_igloos.columns[0].split()[1])):

    for j in range(int(e_igloos.columns[0].split()[0])):

        if (e_igloos.iloc[i,0][j] == 'M'):

            print(j,i)

            man_space_e.append(str(j) +' '+ str(i))
dev_space_f = list()

man_space_f = list()

print("free seats for developers:")

for i in range(int(f_glitch.columns[0].split()[1])):

    for j in range(int(f_glitch.columns[0].split()[0])):

        if (f_glitch.iloc[i,0][j] == '_'):

            print(j,i)

            dev_space_f.append(str(j) +' '+ str(i))

print("free seats for managers:")            

for i in range(int(f_glitch.columns[0].split()[1])):

    for j in range(int(f_glitch.columns[0].split()[0])):

        if (f_glitch.iloc[i,0][j] == 'M'):

            print(j,i)

            man_space_f.append(str(j) +' '+ str(i))
devcnt = int(a_solar.iloc[int(a_solar.columns[0].split()[1]):int(a_solar.columns[0].split()[1])+1,0])

mancnt = int(a_solar.iloc[int(a_solar.columns[0].split()[1]) + devcnt +1 :int(a_solar.columns[0].split()[1]) + devcnt +2,0])

devs = data.iloc[int(a_solar.columns[0].split()[1])+1:int(a_solar.columns[0].split()[1])+1+devcnt,0]

mans = data.iloc[int(a_solar.columns[0].split()[1]) + devcnt + 2:int(a_solar.columns[0].split()[1]) + devcnt + 2 + mancnt,0]
myfile = open('outputFile_a_solar.txt', 'w')

for a in range(len(dev_space_a)):

    myfile.write(dev_space_a[a])

    myfile.write("\n")

for b in range(devcnt-len(dev_space_a)):

    myfile.write('X')

    myfile.write("\n")

for c in range(len(man_space_a)):

    myfile.write(man_space_a[c])

    myfile.write("\n")

for d in range(mancnt-len(man_space_a)):

    myfile.write('X')

    myfile.write("\n")

myfile.close()

myfile.close()