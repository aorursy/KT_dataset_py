#Import MatplotLib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
df= pd.DataFrame({'A':range(10),'B':range(5,15)})
#Way to plot aline fro our dataframe.
df.plot.line(title = 'Info')
#Doing it using matplotlob
plt.plot(df.A,label='A')
plt.plot(df.B,label='B')
plt.xlabel('Index')
plt.ylabel('Values')
plt.title('Info')
plt.legend()
#ScatterPlot
#s determines the size of scatters.
plt.scatter(x=df.A,y=df.B,s=10)
#Create a random DataFrame to visualize
df = pd.DataFrame({'a1' : range(10)})
df['a2'] = df.a1 * df.a1
df['a3'] = df.a2 * df.a1
df
df.plot(linewidth = 5,colormap = 'Set1',title='Degree')
#Colormap can be set to Set1,Set2,Set3 for different colors.
from sklearn.datasets import load_iris
iris = load_iris()
iris.feature_names
iris.target_names
#Let's scatter different targets using sepal length and sepal width
plt.scatter(x=iris.data[:,0],y=iris.data[:,1],c = iris.target, s = iris.data[:,2]*10)
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.title('Iris')
#Let's scatter different targets using Petal length and petal width
plt.scatter(x=iris.data[:,2],y=iris.data[:,3],c = iris.target, s = iris.data[:,0]*10)
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')
plt.title('Iris')
#Now convert this Iris data into DataFrame
iris_df = pd.DataFrame(data = iris.data,columns = iris.feature_names)
iris_df
iris_df['Type'] = iris.target
iris_df
