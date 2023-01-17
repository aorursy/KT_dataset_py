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
import pandas as pd
csv_data =pd.read_csv("/kaggle/input/covid19-in-india/covid_19_india.csv")
csv_data.head()
data = csv_data.drop(["Sno","Time","State/UnionTerritory","ConfirmedIndianNational","ConfirmedForeignNational"], axis=1)
data = data.rename(columns={'Date': 'date'})  #converting 'Date' to 'date' because 'Date' is inbuild variable use in python
data.head()
len(data)
Dates = data['date'].tolist()
Confirmed = data['Confirmed'].tolist()
Deaths = data['Deaths'].tolist()
Cured = data['Cured'].tolist()


# dates = data['date'].to_numpy()
Confirmed[0:10]
Dates[0:10]
current_date = Dates[0]
confirm = Confirmed[0]
death = Deaths[0]
cure = Cured[0]

date =[current_date,]
confirm_cases =[]
death_cases =[]
cure_cases =[]
new_cases= [Confirmed[0],]

for i in range(1,len(Dates)): 
    if Dates[i] == current_date:
        confirm += Confirmed[i]
        death += Deaths[i]
        cure += Cured[i]
    else:
        if len(confirm_cases) != 0:
            new = abs(confirm_cases[-1] - confirm)
            new_cases.append(new)
        current_date = Dates[i]
        date.append(current_date)
        confirm_cases.append(confirm)
        confirm = Confirmed[i]
        death_cases.append(death)
        death = Deaths[i]
        cure_cases.append(cure)
        cure = Cured[i]
new_cases.append(abs(confirm_cases[-1] - confirm))
confirm_cases.append(confirm)
death_cases.append(death)
cure_cases.append(cure)
len(confirm_cases)
len(date)
import os
import numpy as np
from matplotlib import pyplot as plt

%matplotlib inline
#Data visualization
plt.figure(1)
x_ticks = np.arange(0, 80, 10)
plt.xticks(x_ticks)
plt.plot(date,confirm_cases, label = "Confirmed Cases") 
plt.plot(date,death_cases, label = "Death Cases") 
plt.plot(date,cure_cases, label = "Cured Cases") 
plt.plot(date,new_cases, label = "New Cases") 
plt.tight_layout()
plt.xlabel('Date') 
plt.ylabel('Number of Cases') 
plt.title('Coronavirus in India') 
plt.legend() 
plt.show() 
csv_data
data = csv_data.drop(["Sno","Time","ConfirmedIndianNational","ConfirmedForeignNational"], axis=1)
data
data=data.sort_index(ascending=0)
data
data = data.rename(columns={'Date': 'date', 'State/UnionTerritory' : 'state'})
data
states = list(data.pop("state"))
dates = list(data.pop("date"))
cured = list(data.pop("Cured"))
deaths = list(data.pop("Deaths"))
confirmed = list(data.pop("Confirmed"))
print("total number of rows :" , len(states))
print("total number of Indian state's ehich are affected by COVID-19 :", len(set(states)))
print("Dictionnary key = state name, value = [total confirmed cases, total death cases, total cured cases]")
state_dic = {}
count = 0

for i in range(len(states)):
    if states[i] not in state_dic:
        state_dic[states[i]] = [confirmed[i],deaths[i],cured[i]] 
        count +=1
    if count == len(set(states)):
        break
        
state_dic
states = list(state_dic)
cured_cases = []
death_cases = []
confirmed_cases = [] 
for i in range(len(state_dic)):
    confirmed_cases.append(state_dic[states[i]][0])
    death_cases.append(state_dic[states[i]][1])
    cured_cases.append(state_dic[states[i]][2])
    
    
print("confirmed_cases : {} \n".format(confirmed_cases))
print("death_cases : {} \n".format(death_cases))
print("cured_cases : {} \n".format(cured_cases))
y_pos = np.arange(len(states))
performance = confirmed_cases

plt.rcParams.update({'font.size': 22})#chamge font size
plt.figure(figsize=(30,20)) #change size of plot
plt.barh(y_pos, performance, align='center', alpha=1)
plt.yticks(y_pos, states)
plt.xlabel('Usage')
plt.title('Confirmed Cases Plot')

plt.show()
y_pos = np.arange(len(states))
performance = death_cases

plt.rcParams.update({'font.size': 22})#chamge font size
plt.figure(figsize=(30,20)) #change size of plot
plt.barh(y_pos, performance, align='center', alpha=1)
plt.yticks(y_pos, states)
plt.xlabel('Usage')
plt.title('Death Cases Plot')

plt.show()
y_pos = np.arange(len(states))
performance = cured_cases

plt.rcParams.update({'font.size': 22})#chamge font size
plt.figure(figsize=(30,20)) #change size of plot
plt.barh(y_pos, performance, align='center', alpha=1)
plt.yticks(y_pos, states)
plt.xlabel('Usage')
plt.title('Cured Cases Plot')

plt.show()
from __future__ import print_function
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets
selected = ''

def graph(selected):
    index = states.index(selected)
    total_confirmed = confirmed_cases[index]
    total_death = death_cases[index]
    total_cured = cured_cases[index]
    print("currently total number of 'COVID-19' cases in '{}' state are '{}',\n total deaths are '{}' and total cured cases are '{}'".
          format(selected,total_confirmed,total_death,total_cured))
    
def f(state):
    global selected
    selected = state
    if selected in states:
        graph(selected)
#     return state
print("Select State :")
interact(f, state=widgets.Combobox(options=states, value="Maharashtra"));

