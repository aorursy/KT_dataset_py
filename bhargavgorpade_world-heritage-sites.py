import pandas as pd



#Read the CSV File into a dataframe 

#I have used 'r' before the File path to avoid UniCode error caused by escape character '\'

h_df = pd.read_csv('../input/unesco-world-heritage-sites/whc-sites-2019.csv')



#Print the top few rows of the df

h_df.head()
#Selecting the first 20 countries with the most World Heritage Sites



#h_df['states_name_en'] will give the values in states_name_en only



#h_df['states_name_en'].value_counts()

#The value_counts() function is used to get a Series containing counts of unique values.

#The resulting object will be in descending order so that the first element is the most frequently-occurring element. 

#Excludes NA values by default.



#h_df['states_name_en'].value_counts()[:10] - Implies first 10 values. 



h_df['states_name_en'].value_counts()[:10]
import matplotlib.pyplot as plt



#To set the size of the output visual

plt.figure(figsize=(15,5))



# Taking the already available value 'h_df['states_name_en'].value_counts()[:10]' and plotting using it

#kind=bar indicates that the visual is going to be a bar chart

#edgecolor='black' indicates that the edges of the bars will be black in color

#alpha=0.1 indicates how dark or light the bar colour should be, Values to be within 0.0-1.0

h_df['states_name_en'].value_counts()[:10].plot(kind='bar',edgecolor='black', alpha=0.8)



#A lot of times when dealing with iterators, we also get a need to keep a count of iterations. 

#Python eases the programmers’ task by providing a built-in function enumerate() for this task.  

#plt.text() is to show the individual values on top of the respective bars, converting value to string

for index, value in enumerate(h_df['states_name_en'].value_counts()[:10]):

    plt.text(index, value, str(value))

    

#Setting x label, y Label and Title along with setting their font size

plt.xlabel("Countries", fontsize=14)

plt.ylabel("Count of sites", fontsize=14)

plt.title("World Heritage Sites by Country", fontsize=15)



#To display the plot 

plt.show()
h_df['region_en'].value_counts()
region_ser = h_df['region_en'].value_counts()



#To initialise the pie-chart figure

fig = plt.figure()



#Use the series name 'region_ser' to plot the pie-chart  

region_ser.plot.pie(label="", title="Percentage of World Heritage Sites according to Region",autopct='%1.2f%%');

plt.show()

h_df['category'].value_counts()
fig = plt.figure()

ax = fig.add_axes([0,0,1,1])



ser = h_df['category'].value_counts()



ser.plot.pie(label="", title="Percentage of World Heritage Sites according to Category",autopct='%1.2f%%');

plt.show()
h_df['danger'].value_counts()
fig = plt.figure()

ax = fig.add_axes([0,0,1,1])



ser = h_df['danger'].value_counts()



ser.plot.pie(label="", title="Endangered World Heritage Sites",autopct='%1.2f%%');

plt.show()