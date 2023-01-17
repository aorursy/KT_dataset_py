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



data = pd.read_csv("PlayGolf.csv")



#Get Total Yes 

global total_yes

global total_no

global p_Total_yes

global p_Total_no



total_yes = data.loc[data['PLAY GOLF'] =='Yes'].count().mean()

print("Total 'YES' on 'Play Golf':",total_yes)



#Get Total No

total_no = data.loc[data['PLAY GOLF'] =='No'].count().mean()

print("Total 'NO' on 'Play Golf':",total_no)



print("***********Probability of 'Yes' and 'NO'************************")

p_Total_yes = total_yes / n

p_Total_no = total_no /n

print("P(Yes):", p_Total_yes )

print("P(No):", p_Total_no)



print("***********Total number of Observation************************")

print("Total number of Observation:",data.shape[0])



def findProbabilityofEvents(event,outlook,playingGolf,total):    

    outlook_yes = data.loc[(data[event] == outlook) & (data['PLAY GOLF'] == playingGolf)].shape[0]

    p_Outlook_yes = outlook_yes / total

    print("P(" + outlook + "|" + playingGolf +"):",p_Outlook_yes)

    return p_Outlook_yes

    

def findProbabilityOfPlayGolf(outlook,temperature,humidity):

    # Find Probability of Playing Golf for 'Yes'

    outlook_Yes = findProbabilityofEvents('OUTLOOK',outlook,'Yes',total_yes)

    temp_Yes = findProbabilityofEvents('TEMPERATURE',temperature,'Yes',total_yes)

    humidity_yes = findProbabilityofEvents('HUMIDITY',humidity,'Yes',total_yes)

    p_Yes_Today = outlook_Yes * temp_Yes * humidity_yes * p_Total_yes

   

    # Find Probability of Playing Golf for 'No'

    outlook_No = findProbabilityofEvents('OUTLOOK',outlook,'No',total_no)

    temp_No = findProbabilityofEvents('TEMPERATURE',temperature,'No',total_no)

    humidity_No = findProbabilityofEvents('HUMIDITY',humidity,'No',total_no)

    p_No_Today = outlook_No * temp_No * humidity_No * p_Total_no

    

    print("****************************************************")

    print("Probability of playing golf given by :")

    print("P(Yes/today):",p_Yes_Today)

    

    print("Probability of not playing golf given by :")

    print("P(No/today)",p_No_Today)

    

    if(p_Yes_Today > p_No_Today):

        print("Prediction that golf would be played is ‘Yes’.")

    else:

        print("Prediction that golf would be played is ‘No’.")



# Check for the Sample Example

findProbabilityOfPlayGolf('Rainy','Normal','High')