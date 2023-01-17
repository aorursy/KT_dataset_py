#Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
%matplotlib inline
data = pd.read_csv("../input/timesData.csv")
data.head()
country = list(data["country"].unique()) # We create a new list that includes unique country names.

x = "Unisted States of America"
if x in country: country.remove(x) # there is no country like "Unisted States of america". This might have written wrongly ,anyway we delete this.


research_rate = [] 
for i in country:
    x = data[data["country"] == i] # We look each of the Countries. 
    research_ratio = sum(x.research) / len(x) # Average of research values for each country.
    research_rate.append(research_ratio) # Adding it to research_rate.
    
dataframe = pd.DataFrame({"country":country , "research":research_rate }) # We create a new dataframe that includes country and research. That's what we want.
new_index = dataframe["research"].sort_values(ascending=False).index.values # We sort our reseach values.
sorted_data = dataframe.reindex(new_index) # Our sorted research values and countries are in the sorted_data now.

#Visualization
plt.figure(figsize = (17,12))
sns.barplot(x = sorted_data["country"] , y = sorted_data["research"] , palette = sns.cubehelix_palette(len(sorted_data)))
plt.xticks(rotation = 90)
plt.title("COUNTRY VS RESEARCH")
plt.xlabel("Research")
plt.ylabel("Country")
plt.show()
teaching_rate = []
for i in country:
    y = data[data["country"] == i] # We look each of the Countries. 
    teaching_ratio = sum(y.teaching) / len(y) # Average of teaching values for each country.
    teaching_rate.append(teaching_ratio) # Adding it to teaching_rate.

dataframe2 = pd.DataFrame({"country":country , "teaching":teaching_rate }) # We create a new dataframe that includes country and teaching. That's what we want.
new_index2 = dataframe2["teaching"].sort_values(ascending=False).index.values # We sort our teaching values.
sorted_data2 = dataframe2.reindex(new_index2) # Our sorted teaching values and countries are in the sorted_data2 now.
#Visualization
plt.figure(figsize = (17,12))
sns.barplot(x = sorted_data2["country"] , y = sorted_data2["teaching"] , palette = sns.cubehelix_palette(len(sorted_data)))
plt.xticks(rotation = 90)
plt.title("COUNTRY VS TEACHING")
plt.xlabel("Teaching")
plt.ylabel("Country")
plt.show()
#Visualization
f , ax1 = plt.subplots(figsize = (20,10))
sns.pointplot(x = "country" , y = "research" , data=sorted_data , color = "lime" , alpha=0.8)
sns.pointplot(x = "country" , y = "teaching" , data=sorted_data2 , color = "red" , alpha=0.8)
plt.text(62,35,'Teaching ratio',color='red',fontsize = 17,style = 'italic')
plt.text(62,30,'Research ratio',color='lime',fontsize = 18,style = 'italic')
plt.xlabel('Countries',fontsize = 15,color='blue')
plt.ylabel('Values',fontsize = 15,color='blue')
plt.xticks(rotation = 90)
plt.title('TEACHING VS RESEARCH ',fontsize = 20,color='blue')
plt.grid() 
