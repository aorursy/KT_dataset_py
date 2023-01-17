%matplotlib inline
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
plt.plot()
plt.show()
#above lines plots an empty figure
plt.plot([1,2,3,4])
plt.show()
x=[1,2,3,4]
y=[11,22,33,44]
plt.plot(x,y,color='green')
plt.show()
#1st method
fig = plt.figure()  #creates figure
ax=fig.add_subplot() #adds some axes
fig.show()

#2nd method
fig = plt.figure()
ax = fig.add_axes([1,1,1,1])
ax.plot(x,y)
plt.show()
#3rd method-Recommended
fig,ax=plt.subplots()
ax.plot(x,y);
fig,ax=plt.subplots(figsize = (5,5))
ax.set(title="Simple plot",
      xlabel = "x axis",
      ylabel = "y axis")  # to customize the plot

ax.plot(x,y);
fig.savefig("simpleplot.png")  # to save the figure
import numpy as np
#linspace generates numbers between given range and count will also be given, returns a numpy array
x = np.linspace(0,10,100)
x[:10]
#Plot the data and create a line plot
fig,ax= plt.subplots()
ax.plot(x,x**2);

#Use same data to make scatter plot
fig,ax = plt.subplots()
ax.scatter(x,np.exp(x));
#Another Scatter Plot
#Use same data to make scatter plot
fig,ax = plt.subplots()
ax.scatter(x,np.sin(x));
#Bar plot
nut_butter_prices = {"Almond butter": 15,
                    "Peanut butter": 8,
                     "Cashew butter": 12}
ax, fig = plt.subplots()
plt.bar(nut_butter_prices.keys(), height=nut_butter_prices.values());
plt.title("Nut butter Graph");
plt.xlabel('Types of butter')
plt.ylabel("Prices");
#Horizontal Bar plot
nut_butter_prices = {"Almond butter": 15,
                    "Peanut butter": 8,
                     "Cashew butter": 12}
ax, fig = plt.subplots()
plt.barh(list(nut_butter_prices.keys()), list(nut_butter_prices.values()))
plt.title("Nut butter Graph");
plt.ylabel('Types of butter')
plt.xlabel("Prices");
#Make some data
x = np.random.randn(1000)
x[:5]
#Plot a histogram
ax,fig = plt.subplots()
plt.hist(x);

#Option1 tuple method
fig,((ax1,ax2),(ax3,ax4))= plt.subplots(nrows = 2,
                                       ncols= 2,
                                       figsize=(10,5))
ax1.plot(x,x*2)  # provide data to individual plot
ax2.bar(nut_butter_prices.keys(),nut_butter_prices.values());
ax3.scatter(np.random.random(5),np.random.random(5))
ax4.hist(np.random.randn(1000));
#Option 2 by list method
fig, ax = plt.subplots(nrows = 2,
                        ncols= 2,
                       figsize=(10,5))
ax[0,0].plot(x,x*2)  # provide data to individual plot
ax[0,1].bar(nut_butter_prices.keys(),nut_butter_prices.values());
ax[1,0].scatter(np.random.random(5),np.random.random(5))
ax[1,1].hist(np.random.randn(1000));

car_sales = pd.read_csv(r"../input/carsales/car-sales.csv",engine='python')
car_sales
fig,ax = plt.subplots()
ax.bar(car_sales.Make,car_sales.Price)
#Sample graph
ts = pd.Series(np.random.randn(1000),index=pd.date_range('1/1/2020',periods=1000))
ts= ts.cumsum()
ts.plot();
#Manipulate the Price Column
car_sales["Price"] = car_sales["Price"].str.replace('[\$\,\.]', '')
car_sales
#remove last zeros
car_sales.Price = car_sales["Price"].str[:-2] #only run once else it will keep removing
car_sales
car_sales["Sale Date"] = pd.date_range("1/1/2020",periods=len(car_sales))
car_sales
car_sales["Total sales"] =car_sales["Price"].astype(int).cumsum() # astype used as the Price column was a string,
#it was concatenating
car_sales
# Now plot the total sales
car_sales.plot(x= "Sale Date",y= "Total sales");
car_sales

r1 = np.random.rand(10,4)

df= pd.DataFrame(r1, columns=["a","b","c","d"])
df
df.plot(kind="bar");
#Plot a heart disease data graph
heart_disease = pd.read_csv(r"../input/heartdisease/heart-disease.csv")
heart_disease.head()
#create a histogram on age column

heart_disease["age"].plot(kind="hist",bins=60);
heart_disease.plot.hist(figsize=(10,10),subplots = True);
over_50 = heart_disease[heart_disease["age"]>50]
over_50.head()
over_50.plot(kind="scatter",
             x= "age",
            y="chol",
            c='target');
fig,ax=plt.subplots(figsize=(10,6))
over_50.plot(kind="scatter",
             x= "age",
            y="chol",
            c='target',
            ax=ax);
# To plot by OO method
fig,ax= plt.subplots(figsize=(5,5))
scatter = ax.scatter(x=over_50['age'],
                   y=over_50['chol'],
                    c= over_50['target'])

#customize
ax.set(title="Heart and Cholesterol Levels",
      xlabel="Age",
      ylabel="Cholesterol");

#add a legend-> it will unpack the c from scatter
ax.legend(*scatter.legend_elements(),title='target');

#add a horizontal line
ax.axhline(over_50["chol"].mean(),
          linestyle="--");
# Subplot of colesterol and thalach
fig, (ax0,ax1)=plt.subplots(figsize=(10,10),
                           nrows=2,
                           ncols=1,
                           sharex=True)
scatter = ax0.scatter(x=over_50["age"],
                    y=over_50["chol"],
                    c=over_50["target"],
                     )

scatter =  ax1.scatter(x=over_50["age"],
                      y=over_50["thalach"],
                      c=over_50["target"],
                       )
#customizing ax0
ax0.set(title="Age vs cholesterol levels",
       
       ylabel="Cholesterol Level");

ax0.legend(*scatter.legend_elements(),title="target");
ax0.axhline(over_50["chol"].mean())

#customizing ax1
ax1.set(title="Age vs max heart rate levels",
       xlabel = "Age",
       ylabel="Max heart rate level");

ax1.legend(over_50["target"]);
ax1.legend(*scatter.legend_elements(),title="target");
ax1.axhline(over_50["thalach"].mean());

fig.suptitle("Heart Disease",
            fontsize= 20,
            fontweight="bold");
plt.style.available
plt.style.use("grayscale")
car_sales["Price"].astype(int).plot();
x1= np.random.rand(10,4)
x1
x1_d= pd.DataFrame(x1, columns=["a","b","c","d"])
x1_d
ax= x1_d.plot(kind='bar')
ax.set(title="Random Graph",
      xlabel= 'Column name',
      ylabel='Ranges');
# set style
plt.style.use('seaborn')

# Subplot of colesterol and thalach
fig, (ax0,ax1)=plt.subplots(figsize=(10,10),
                           nrows=2,
                           ncols=1,
                           sharex=True)
scatter = ax0.scatter(x=over_50["age"],
                    y=over_50["chol"],
                    c=over_50["target"],
                     cmap= 'winter')

scatter =  ax1.scatter(x=over_50["age"],
                      y=over_50["thalach"],
                      c=over_50["target"],
                       cmap='winter')
#customizing ax0
ax0.set(title="Age vs cholesterol levels",
       
       ylabel="Cholesterol Level");

ax0.legend(*scatter.legend_elements(),title="target");
ax0.axhline(over_50["chol"].mean(),
           linestyle='--')

#customizing ax1
ax1.set(title="Age vs max heart rate levels",
       xlabel = "Age",
       ylabel="Max heart rate level");

ax1.legend(over_50["target"]);
ax1.legend(*scatter.legend_elements(),title="target");
ax1.axhline(over_50["thalach"].mean(),
           linestyle='--');

fig.suptitle("Heart Disease",
            fontsize= 20,
            fontweight="bold");
#to save the plotted graph locally
fig.savefig("heart_disease_plot.png")
