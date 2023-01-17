#1.step
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#2.step
#to retrieve our comma-separated data and read it;
ma=pd.read_csv("../input/creditcard.csv")
#3.step
ma.info()

#output; 
#1."ma" consists of 284807 data (From 0 to 284807)
#2."ma" consists of 31 non-empty columns and the name of the columns.
#3."ma" consists of 30 float type 1 int64 type columns.
#4.The total size of "ma" is 67.4 mb.
#4.step (columns)
ma.columns

#for output;
#Columns of "ma": The names of the columns of "ma" are written respectively.
#5.step: for correlation
#uses ma.corr () to find the correlation between the columns of "ma".
#(If value 1 is correct proportional, value -1 is inversely proportional, and value 0 is not any relation between them)

ma.corr()
#6.step: heatmap with subplot
f,ax=plt.subplots(figsize=(16,18))
sns.heatmap(ma.corr(),annot=True,linewidths=0.1,alpha=0.9,fmt=".2f",ax=ax)
#7.step : head()
#indicates how many lines will be written from the start. Starts from line 0.
#if there were no numbers instead of 7 then it would print from 0 to 4, ie 5 lines.
print("1.",ma.head(7))
print("2.",ma.head())
#8.step : tail()
#indicates how many lines to type. Because there are no numbers in parentheses, it prints 5 lines from the last.
ma.tail()
#9.step: line plot
#use of: "name of the folder in which we define the data"."name of selected column"."plot"

#for output;
#1.kind= The type of the plot is written.
#2.alpha= opacity (The closer the 0 is, the finer the graphics)  
#(0.6) equals (.6) 0 does not need to be written.
#3.label= The label of the plot
#4.plot legend: The plot's label is written to appear. If the location is written, loc = ("location name")
#5.grid=True: It is used to divide the data into cages (squares) so that the values in the graph can be read better.
#6.linewidth: gives the thickness of each line (the larger the number, the thicker it is).
#7.figsize=(a,b)  a: Length of the graph on the x (horizontal) axis, b: Length of the graph on the y (vertical) axis

ma.V1.plot(kind="line",alpha=.6,color="b",grid=True,label="V1 line plot",linewidth=5)
ma.V2.plot(kind="line",color="red",label="V2 line plot",linewidth=1,figsize=(10,12))
plt.legend(loc="best")
plt.show()
#10.step: histogram plot
#use of: the same line plot, "name of the folder in which we define the data"."name of selected column"."plot"
#bins: Number of bars in the figure (the greater the thickness is reduced)

ma.V3.plot(kind="hist",bins=150,linewidth=2,figsize=(9,10))
#11.step: scatter plot
#use of: "name of the folder in which we define the data"."plot"
#plt.clf(): deletes everything from the plot

ma.plot(kind="scatter",x="V7",y="V1",color="r",label="scatter plot",linewidth=0.2,figsize=(10,10),alpha=0.5,grid=True)
plt.legend(loc="upper right")
print("1.",plt.show())
print("2.",plt.clf())
#12.step: Dictionary
#use of: Dictionary name={"key1":"value1","key2":"value2",...}


#1.other use of (with list and one value (object)): Dictionary naem={"key1":["values1"],"key2":"values"}
dictionary_1={"Norway":"Stavanger","Finland":"Helsinki","France":["Nice","Montpeiller","Lyon","Pais"]}
print("1.",dictionary_1)
#2.to find the keys to the dictionary;
print("2.",dictionary_1.keys())
#3.to find the values to the dictionary;
print("3.",dictionary_1.values())
#4.If we want to change the city of Norway as an example;
dictionary_1["Norway"]="Harstad"
print("4.",dictionary_1)
#5.If we want to add another country and city;
dictionary_1["Denmark"]="Copenhagen"
print("5.",dictionary_1)
#6.If you are writing the value of its key to look for any value.
print("6.",dictionary_1.pop("Finland"))
#7.to remove any key-value pair randomly;
print("7.",dictionary_1.popitem())
print("result",dictionary_1)
#8.completely to erase;
print("8.",dictionary_1.clear())  
#12.1 dictionary example about numbers
dictionary_2={81:9,49:7,25:5,9:3,1:1}
print("1.",dictionary_2)
dictionary_2[121]=11
print("2.",dictionary_2)
print("3.",dictionary_2.popitem())
print("4.",dictionary_2.pop(25))
print("5.",dictionary_2.clear())
#13.step: pandas 
#import pandas as pd
#ma=pd.read_csv("../input/creditcard.csv")
#csv: comma - separated values 
#Type of 1 series
#Type of 2 DataFrame

print("1.",type(ma["V1"]))
print("2.",type(ma[["V1"]]))
#14.step: Comparison operator
print("1.",3>2)
print("2.",5!=3)
print("3.",5/2==2)
print("4.",5//2==2)
#15 step: while
i=0
while(i!=7):
    print("i=",i)
    i=i+1
i==7
print("i=",i)
#16.step: Collecting operation with arrays 

each=0                                 #first we matched each to 0.
count=0                                #first we matched count to 0.
list_3=[0,1,2,3,4,5,6]                
for each in list_3:                    #each is any value in the list;
    count=count+list_3[each]           #first; 0=0+list_3[0] 2.step 0=0+liste[1]...
    print(count)
    each=each+1
print("result",count)
#17.step: enumerate with for
#a and b can be any two value.
#If a and b are in list_4, print a: b.

list_4=[0,1,2,3,4,5,6]
for a, b in enumerate(list_4):
    print(a,":",b)
#18. dictionary with for

dictionary_4={"name":"martin","surname":"aniston"}
for a, b in dictionary_4.items():
    print(a,":",b)
#19. iterrows with for

for a,b in ma[["V3"]][:6].iterrows():
    print(a,",",b)
