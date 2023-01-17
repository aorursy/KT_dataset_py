import seaborn as sb

df=sb.load_dataset('titanic')

df.shape
df.head()
sb.distplot(df['fare']) #distribution plot
sb.distplot(df['fare'],kde=False) #kernel density equation
sb.rugplot(df['fare']) #no sharpe value
sb.boxplot(df['fare'])
import matplotlib.pyplot as plt

for i in df.mean().index:

    sb.boxplot(df[i])

    plt.show()
plt.figure(figsize=(10,10)) #used to resize graph

sb.heatmap(df['fare'].sample(100).values.reshape(10,10),annot=True,cmap='Pastel1') # annot to print values also, # cmap gives color *matplotlib cmap
plt.figure(figsize=(10,10)) #used to resize graph

sb.heatmap(df['fare'].sample(100).values.reshape(10,10),annot=True,cmap='Pastel1',linecolor='w') # annot to print values also, # cmap gives color *matplotlib cmap
sb.jointplot(x='fare',y='age',data=df)
sb.jointplot(x='fare',y='age',data=df,kind='reg') #best fitted line for regression
sb.jointplot(x='fare',y='age',data=df,kind='hex')
sb.jointplot(x='fare',y='age',data=df,kind='kde') #kernel density equ
df.describe()
import pandas as pd

for i in df.select_dtypes(include=['int']).columns:

    for j in df.select_dtypes(include=['int']).columns:

        if i!=j:

            sb.jointplot(i,j,data=df,kind='reg')
sb.boxplot('pclass','fare',data=df) # drawing outlayers by grouping all values and amking max,min , mean and median value by making outlayer 
sb.violinplot('pclass','fare',data=df)
sb.swarmplot('pclass','fare',data=df)
sb.stripplot('pclass','fare',data=df)
sb.pairplot(df) #create graph as per no. of col