import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.linear_model import LinearRegression 
import os

print(os.listdir("../input/graduate-admissions"))
UCLA_data = pd.read_csv("../input/graduate-admissions/Admission_Predict_Ver1.1.csv")
UCLA_data.head(2)
UCLA_data.info()
UCLA_data.describe()
data = pd.DataFrame(UCLA_data)

data.columns = data.columns.str.strip().str.lower().str.replace(' ', '_')
data.head(2)
data['chance_of_admit']=data["chance_of_admit"]*100

del data["serial_no."]
data.head(2)
data.columns
x=data["gre_score"]

y=data["research"]

plt.xlabel("GRE Scores")

plt.ylabel("Research")

plt.scatter(x,y)
x=data["gre_score"]

y=data["chance_of_admit"]

plt.xlabel("GRE Scores")

plt.ylabel("Chance Of Admit")

plt.bar(x,y)
labels = 'gre_score', 'toefl_score', 'university_rating', 'sop', 'lor', 'cgpa','research'

sizes = [10,10,10,10,10,10,10]

explode = (0.1, 0, 0, 0, 0, 0, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')



fig1, ax1 = plt.subplots()

ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',

        shadow=True, startangle=90)

ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.



plt.show()
sns.distplot(x);
fig,ax = plt.subplots(figsize=(10, 10))

sns.heatmap(data.corr(), ax=ax, annot=True, linewidths=0.05, fmt= '.2f',cmap="magma")

plt.show()
grid = sns.pairplot(data)

grid
grid.savefig('PairPlot.png')
grid.savefig('PairPlot.pdf')
high = pd.DataFrame(data[data['chance_of_admit'] > 90])
high.head()
sns.pairplot(high)
x=data["research"]

y=data["chance_of_admit"]

plt.xlabel("Research")

plt.ylabel("Chance Of Admit")

plt.scatter(x,y)
x=high["research"]

y=high["chance_of_admit"]

plt.xlabel("Research")

plt.ylabel("Chance Of Admit")

plt.scatter(x,y)
print("Not Having Research:",len(data[data.research == 0]))

print("Having Research:",len(data[data.research == 1]))

y = np.array([len(data[data.research == 0]),len(data[data.research == 1])])

x = ["Not Having Research","Having Research"]

plt.bar(x,y)

plt.title("Research Experience")

plt.xlabel("Canditates")

plt.ylabel("Frequency")

plt.show()
y = np.array([data["toefl_score"].min(),data["toefl_score"].mean(),data["toefl_score"].max()])

x = ["Worst","Average","Best"]

plt.bar(x,y)

plt.title("TOEFL Scores")

plt.xlabel("Level")

plt.ylabel("TOEFL Score")

plt.show()
x = data[['gre_score', 'toefl_score', 'university_rating', 'sop', 'lor', 'cgpa','research']]

y = data[['chance_of_admit']]
x.head(2)
y.head(2)
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2)



reg=LinearRegression()
reg.fit(X_train,y_train)
reg.score(X_train,y_train)
pdt = reg.predict(X_test)

predicted = pd.DataFrame(pdt, columns = ["chance_of_admit"])

predicted

X_test
pdt_custom = reg.predict([[330,120,4,4,4,9,0]])

predicted = pd.DataFrame(pdt, columns = ["chance_of_admit"])

predicted

pdt_custom
pdt
actual = y_test

actual
plt.scatter( predicted, actual, color='red')
reg.score(X_test, y_test)
from sklearn.metrics import mean_squared_error

from math import sqrt



rms = sqrt(mean_squared_error(y_test,pdt))

rms
import tkinter as tk

from tkinter import *

from tkinter import messagebox

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()



def entry():

    window = tk.Tk()

    window.title("Admission Predictor")

    window.geometry('350x400')

    

    Label(window,label="MTech Admission Predictor", font='arial 14 bold').grid(row=0, columnspan=2)

    Label(window,label="").grid(row=1,column=0)

    Label(window,label="GRE Score").grid(row=2,column=0)

    Label(window,label="").grid(row=3,column=0)

    Label(window,label="TOEFEL Score").grid(row=4,column=0)

    Label(window,label="").grid(row=5,column=0)

    Label(window,label="University Rating").grid(row=6,column=0)

    Label(window,label="").grid(row=7,column=0)

    Label(window,label="SOP Strength").grid(row=8,column=0)

    Label(window,label="").grid(row=9,column=0)

    Label(window,label="LOR Strength").grid(row=10,column=0)

    Label(window,label="").grid(row=11,column=0)

    Label(window,label="CGPA").grid(row=12,column=0)

    Label(window,label="").grid(row=13,column=0)

    Label(window,label="Research Experience").grid(row=14,column=0)

    

    v1=IntVar()

    v2=IntVar()

    v3=IntVar()

    v4=IntVar()

    v5=IntVar()

    v6=IntVar()

    v7=IntVar()

    

    v1.set('')

    v2.set('')

    v3.set('')

    v4.set('')

    v5.set('')

    v6.set('')

    v7.set('')

    

    

    e1=Entry(window, textvariable=v1).grid(row=2,column=1)

    e2=Entry(window, textvariable=v2).grid(row=4,column=1)

    e3=Entry(window, textvariable=v3).grid(row=6,column=1)

    e4=Entry(window, textvariable=v4).grid(row=8,column=1)

    e5=Entry(window, textvariable=v5).grid(row=10,column=1)

    e6=Entry(window, textvariable=v6).grid(row=12,column=1)

    e7=Entry(window, textvariable=v7).grid(row=14,column=1)

    

    Label(window,label="").grid(row=15,column=0)

    

    

    def insert():

        newlist=[]

        gre_score=v1.get()

        toefl_score=v2.get()

        university_rating=v3.get()

        sop=v4.get()

        lor=v5.get()

        cgpa=v6.get()

        research=v7.get()

        

        newlist.append(gre_score)

        newlist.append(toefl_score)

        newlist.append(university_rating)

        newlist.append(sop)

        newlist.append(lor)

        newlist.append(cgpa)

        newlist.append(research)

        

        newlist_nn=[newlist]

        print(newlist_nn)

        

        

        pdt = reg.predict(newlist_nn)

        

        result = pdt.flatten()

        messagebox.showinfo("% Chance ", "Your Chance of Admission is %.2f %%" %(result) )

        

        print("Your Chance of Admission is %.2f %%" %(result) )

        print("\n")

        

        v1.set('')

        v2.set('')

        v3.set('')

        v4.set('')

        v5.set('')

        v6.set('')

        v7.set('')

        

        

    def clear():

        v1.set('')

        v2.set('')

        v3.set('')

        v4.set('')

        v5.set('')

        v6.set('')

        v7.set('')

        

    def close():

        window.destroy()

        

    Button(window, label="RESET", command=clear).grid(row=16,column=0)

    Button(window, label="SUBMIT", command=insert).grid(row=16,column=1)

    Button(window, label="EXIT", command=close).grid(row=16,column=2)

    

    window.mainloop()

    

entry()
