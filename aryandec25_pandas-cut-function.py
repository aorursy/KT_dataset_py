import pandas as pd
data=pd.read_csv('http://bit.ly/kaggletrain')

data.head()
bins=[0,5,10,17,60,99]  #Creating the bins for the Age group.

labels=['Toddler','Child','Teenagers','Adult','Elderly'] #Labeling the bin range

catergory=pd.cut(data['Age'], bins=bins, labels=labels) #Using cut function to create group labels for a continues variable. 

data.insert(6,'Age Group',catergory) #Creating a extra column as Age Group

data.head()
data['Age Group'].value_counts() #Total number of Age group by category.