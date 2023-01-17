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
###MADLIB
string1 = input("Noun? ")
string2 = input("Name? ")
string3 = input("Same name as string2. ")
string4 = input("place? ")
string5 = input("Transportation verb? ")
string6 = input("Mode of Transportation? ")
string7 = input("Crazy Noun? ")
string8 = input("Plural Noun? ")
string9 = input("Animal? ")
string10 = input("Smaller animal? ")
string11 = input("Vicious adjective? ")
string12 = input("Same smaller animal. ")
string13 = input("Name from beginning. ")
string14 = input("girly object? ")
string15 = input("Different vicious adjective? ")
string16 = input("Big place? ")

story = "Once upon a time, there was a %s named %s. %s went to the %s to buy some pizza. It %s its %s but upon arrival, a wild %s appeared. All of a sudden, the two of them broke out into a huge fight. They were throwing %s and things got out of control. A police %s came to break up the fight but a %s came and ate it. Now there is no cop and an eating, %s animal is on the loose. The %s was now headed towards the other two looking to do harmful things. But %s broke out his %s and slayed the %s animal. %s was saved!" 

    
print(story % (string1, string2, string3, string4, string5, string6, string7, string8, string9, string10, string11, string12, string13, string14, string15, string16))