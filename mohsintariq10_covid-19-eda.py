# 🎨 Justin Faler 

# 📆 3/12/2020

# 🦅 Mt. San Jacinto College



# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import os

import sys

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
file = open('../input/cleancovid19genome/MN908947-genome.txt', "r")

data = file.read()
# removing spaces and new line characters

data = data.replace(" ", "")

data = data.replace("\n", "")



characters = len(data)

print('Genes:', characters)
#This function will print how many times Adenine shows up



count = data.count('A')

print('Adenine:', count)
#This function will print how many times Cytosine shows up



count = data.count('C')

print('Cytosine:', count)
#This function will print how many times Guanine shows up



count = data.count('G')

print('Guanine:', count)
#This function will print how many times Thymine shows up



count = data.count('T')

print('Thymine:', count)
# This function will split up the genome in pairs of 3's



codons = [data[start:start+3]



for start in range(0,len(data),3)]

print(codons)
# This function will split up the genome in pairs of 6



codons = [data[start:start+6]



for start in range(0,len(data),6)]

print(codons)