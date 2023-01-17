#1.step
#We are writing matplotlib and pandas libraries.
import pandas as pd
import matplotlib.pyplot as plt
#2.step
#to retrieve our comma-separated data and read it;
ma=pd.read_csv("../input/creditcard.csv")
#3.step: line plot
#use of: "name of the folder in which we define the data"."name of selected column"."plot"

#for output;
#1.kind= The type of the plot is written.
#2.alpha= opacity (The closer the 0 is, the finer the graphics)  
#(0.6) equals (.6) 0 does not need to be written.
#3.label= The label of the plot
#4.plot legend: The plot's label is written to appear. If the location is written, loc = ("location name")
#(locationa name=upper right, upper left, lower right and lower left)
#5.grid=True: It is used to divide the data into cages (squares) so that the values in the graph can be read better.
#6.linewidth: gives the thickness of each line (the larger the number, the thicker it is).
#7.figsize=(a,b)  a: Length of the graph on the x (horizontal) axis, b: Length of the graph on the y (vertical) axis

ma.V1.plot(kind="line",alpha=.6,color="b",grid=True,label="V1 line plot",linewidth=5)
ma.V2.plot(kind="line",color="red",label="V2 line plot",linewidth=1,figsize=(10,12))
plt.legend(loc="lower right")
plt.show()
#4.step: scatter plot
#use of: "name of the folder in which we define the data"."plot"
#plt.clf(): deletes everything from the plot

ma.plot(kind="scatter",x="V7",y="V1",color="r",label="scatter plot",linewidth=0.2,figsize=(10,10),alpha=0.5,grid=True)
plt.legend(loc="upper right")
print("1.",plt.show())
print("2.",plt.clf())
#5.step: histogram plot
#use of: the same line plot, "name of the folder in which we define the data"."name of selected column"."plot"
#bins: Number of bars in the figure (the greater the thickness is reduced)

ma.V3.plot(kind="hist",bins=150,linewidth=2,figsize=(9,10))