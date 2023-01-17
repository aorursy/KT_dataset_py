# Importing the necessary libraries
# Supress Warnings

import warnings
warnings.filterwarnings('ignore')

import sys
import glob,os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# Path where all the files are stored
path = r'Downloads/Data Sets Part 1B' # use your path
# Reading all the paths
all_files = glob.glob(path + "/*.csv")

# Empty list to add all the dataframes
li = []

# Loop to read each file with the path given in the filename variable.
for filename in all_files:
    # Reading each file as a pandas dataframe
    df = pd.read_csv(filename, index_col=None, header=0,encoding='cp1252')
    # Selecting only required columns
    df = df[['Item','2014-15']]
    # Transpose the data
    df = df.T
    # Taking the header row
    new_header = df.iloc[0] #grab the first row for the header
    df = df[1:] #take the data less the header row
    # Assign the new header
    df.columns = new_header 
    # Add the section name from the filename
    df['States'] = filename.split('\\')[-1].split('NAD-')[0]+filename.split('-')[1]
    # Append the final DF to the list
    li.append(df)
    
# Creating a final DF by joining all the df's in the list li
df = pd.concat(li, axis=0, ignore_index=True)
pd.options.display.float_format='{:.0f}'.format
all_files # Checking the .csv names in the file 
df.info()
# Clean the data and remove the unwated coloumns

df=df.drop(['Crops', 'Livestock','Forestry and logging','Fishing and aquaculture','Trade & repair services','Hotels & restaurants',
              'Railways', 'Road transport','Water transport','Air transport','Services incidental to transport','Storage',
              'Communication & services related to broadcasting','TOTAL GSVA at basic prices','Taxes on Products','Subsidies on products'
              ],axis=1)
df
df.info() # find the data frame information

df.isnull().sum()
# from the bavoe we can see still unwanted coloumns are there. Let us remove the coloumns
df=df.drop(['Road transport*','Road transport**','Services incidental to transport*','Trade & repair services*',"Population ('00)"],axis=1)
df

# Now the data is cleaned completely the needed rows and coloumns.

df.shape[0] # there are 30
df.isnull().sum() > 2
round(100*(df.isnull().sum()/len(df.index)), 2)# find the NAN values find the data sets
# We can found one coloumn is having one NAN values. We we impute by replace it by zero.

df.loc[pd.isnull(df['Mining and quarrying']), ['Mining and quarrying']] = 0
round(100*(df.isnull().sum()/len(df.index)), 2)
# Now we can see that the data is cleaned fully from the above functions.
# We need to filter out the Union Teritories. drop the union teritories

Union_teritories = {'Delhi','Puducherry','Chandigarh','Andaman & Nicobar Island','Dadar & Nagar Haveli','Daman & Diu','Lakshadeep'}
# Out of this Union teritories, we need to remove filter the states alone. we can use iloc method to drop the coloumns.
df=df[df.States != 'Delhi']
df=df[df.States != 'Puducherry']
df=df[df.States != 'Chandigarh']
df

# Filter the States and Per capita first 
df4 = df[['States','Per Capita GSDP (Rs.)']] # extracting the state and the all India GDP
df4

# now sort the percapita in rupees 
df6= df4.sort_values(by='Per Capita GSDP (Rs.)',ascending = False)
df6

# Import the necessary libraries for plotting the graph for statewise GDP per capita.
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt; plt.rcdefaults()
import seaborn as sns
import sys

# Ploting the bar graph
plt.figure(num=None, figsize=(18, 6), dpi=80, facecolor='w', edgecolor='k')

sns.barplot(x='States', y='Per Capita GSDP (Rs.)', data=df6)
# The tile name indicated using the function below

plt.title(" Statewise GDP Per Capita ")
plt.xlabel('States')
plt.ylabel('Per Capita GSDP (Rs.)')
plt.xticks(rotation=45)

plt.grid(True, linewidth= 1, linestyle="--",color ="r")
plt.show()

df7=df6.iloc[0:5,] 
df7   # Top 5 states are given below in the data starting Goa, Sikkim, Haryana, Kerla,Uttarakhand
df5 = df6.iloc[[-1,-2,-3,-4,-5],:]  # We can see the Bottom 5 states having low capita such as Bhihar, Uttar pradesh, Manipur, Assam & Jharkhand
df5

Ratio = df6['Per Capita GSDP (Rs.)'].max()/df6['Per Capita GSDP (Rs.)'].min()

Ratio
df # Get the original datasets which we have already merged after dropping the union teritories
df['Per_Primary']=(df['Primary']/df['Gross State Domestic Product'])*100# Calcuating the Primary % by dividing it by Per capita GDP
df
df['Per_Secondary']=(df['Secondary']/df['Gross State Domestic Product'])*100 # Clacuating the Secondary divided by Gross state 
df
df['Per_Tertiary']=(df['Tertiary']/df['Gross State Domestic Product'])*100
df
# Filter the States and Percentage Primary,Secondary and Tertiary and stored in df2

df_P = df[['States','Per Capita GSDP (Rs.)','Per_Primary','Per_Secondary','Per_Tertiary']] # extracting the state and the all India GDP
df_P


# Before plotting the graph, we can sort the data sets from the highest to the lowset GDP per capita

Sorted_df= df_P.sort_values(by='Per Capita GSDP (Rs.)',ascending = True)
Sorted_df

# Ploting the bar graph


Sorted_df.plot(x='States',y=['Per_Primary','Per_Secondary','Per_Tertiary'],stacked=True, kind='bar')
# The tile name indicated using the function below

plt.title(" Percentage of Total GDP Across of States across Sectors")
plt.xlabel('States')
plt.ylabel('Percentage of Total GDP across Sectors')
plt.xticks(rotation=90)

plt.grid(True, linewidth= 1, linestyle="--",color ="r")
plt.show()
Category_GDP = Sorted_df[['States','Per Capita GSDP (Rs.)']] # filter the States and Per Capita GSDP
Category_GDP
# Calculate the Quantiles and lables of each coloumns 

df["Quantile"] = pd.qcut(df["Per Capita GSDP (Rs.)"], q=[0,.2,.5,.85,1])
df["Labels"] = pd.qcut(df["Per Capita GSDP (Rs.)"], q=[0,.2,.5,.85,1], labels=['C4','C3','C2','C1'])

df
df['Gross State Domestic Product'] = df['Gross State Domestic Product'].astype(float)
df['Per Capita GSDP (Rs.)'] = df['Per Capita GSDP (Rs.)'].astype(float)
df['States']=df['States'].astype(str)
df.info()
## We can see that Bihar is the lowest Per capita with C4 caterogy and Goa being the highest per capita as C1 category.
# Let us see the top 3/4/5 - Sub cateogry impact to the overall Per capita GDP contributing 80% of GDP.
# We can use the Pareto chart to draw this graph and make it clear about the GDP contributing factors.
# filter out the needed data frame from the above subsectors 
df.set_index('States',inplace=True,)
df# Checking the coloumns in the cleaned data now. 
# Quantiles coloumn is not needed. so we can drop it 
Newdf = df.drop(['Quantile','Per_Primary','Per_Secondary','Per_Tertiary','Primary','Secondary','Tertiary'],axis=1)

Newdf.reset_index(level=0, inplace=True)
Newdf


Newdf.info()
# Define the function and do the 
def findtop(cla):
    Category = Newdf.loc[(Newdf['Labels']==cla),:]  # Labels to be called here
    Category=Category.transpose() # Transpose the Data
    new_header=Category.iloc[0]
    Category=Category.iloc[1:]
    Category.columns=new_header # move to the new header 
    Category=Category.loc[(Category.index != 'Labels'),:]
    Category=Category.fillna(0)
    Category[new_header]=Category.loc[:,(new_header)].astype(int)
    GSDP = Category.loc[(Category.index == 'Gross State Domestic Product'),:]
    GSDP = GSDP.transpose()
    A= GSDP.sum()
    B=int(A)
    Category.loc[:,'Total Subsector'] = Category.sum(axis=1)
    Category.loc[:,'Per%'] = round((Category.loc[:,'Total Subsector']/B)*100,2)       
    Subcat=['Agriculture, forestry and fishing','Construction','Electricity, gas, water supply & other utility services','Manufacturing','Financial services','Mining and quarrying','Real estate, ownership of dwelling & professional services','Public administration','Trade, repair, hotels and restaurants','Transport, storage, communication & services related to broadcasting','Other services']
    k = Category[Category.index.isin(Subcat)]
    k = k.sort_values(by='Per%', axis=0,ascending = True)
    k.loc[:,'Cumm%']=k['Per%'].cumsum(axis = 0,skipna = True)
    k = k.loc[(k ['Cumm%'] <= 80),:]
    return k

C1States =findtop('C1') # find the C1 states only contributing to 80% 
C1States

C2States = findtop('C2') #find the C2 only contributing to 80% 
C2States
C3States = findtop('C3') # filter the C3 states
C3States
C4States = findtop('C4') # filter the C4 states
C4States
# Plot the Graph on the individual values 

#  Now plot the Category of C1, C2, C3, C4

x1 = C1States.index
x2 = C2States.index
x3 = C3States.index
x4 = C4States.index

y1 = C1States['Per%']
y2 = C2States['Per%']
y3 = C3States['Per%']
y4 = C4States['Per%']
# Plot the graph and size the graph and legend outside the graph

## C1States 

plt.title('Top Sub Sector % of GSDP of C1 Sates')
plt.ylabel('Sub Sector')
plt.xlabel('% Per')
a=sns.barplot(x=x1,y=y1)
b = a.set_yticklabels(a.get_yticklabels(),rotation =45)
plt.grid(True, linewidth= 1, linestyle="--",color="g")
plt.xticks(rotation=90)
plt.show()


# Plot the graph and size the graph and legend outside the graph

## C2States - 

plt.title('Top Sub Sector % of GSDP of C2 Sates')
plt.ylabel('Sub Sector')
plt.xlabel('% Per')
a=sns.barplot(x=x2,y=y2)
b = a.set_yticklabels(a.get_yticklabels(),rotation =45)
plt.grid(True, linewidth= 1, linestyle="--",color="g")
plt.xticks(rotation=90)
plt.show()
# Plot the graph and size the graph and legend outside the graph

## C3States - 

plt.title('Top Sub Sector % of GSDP of C3 Sates')
plt.ylabel('Sub Sector')
plt.xlabel('% Per')
a=sns.barplot(x=x3,y=y3)
b = a.set_yticklabels(a.get_yticklabels(),rotation =45)
plt.grid(True, linewidth= 1, linestyle="--",color="g")
plt.xticks(rotation=90)
plt.show()
# Plot the graph and size the graph and legend outside the graph

## C4States - 

plt.title('Top Sub Sector % of GSDP of C4 Sates')
plt.ylabel('Sub Sector')
plt.xlabel('% Per')
a=sns.barplot(x=x4,y=y4)
b = a.set_yticklabels(a.get_yticklabels(),rotation =45)
plt.xticks(rotation=90)
plt.show()


Contribution = Newdf.sort_values(by = 'Per Capita GSDP (Rs.)',ascending = False)  # GSDP per captia sort the C1 first .
Contribution = Contribution.drop(['Gross State Domestic Product','Labels'],axis =1)
Contribution
# let see statewise, Sub category wise most contrbuting factor 
Contribution.plot(stacked=True, kind='bar',figsize=(15,8))
# The tile name indicated using the function below

plt.title(" Percentage contribution of Total GDP Across of States across sub Sectors")
plt.xlabel('States')
plt.ylabel('Percentage of Total GDP across Sectors')
plt.xticks(rotation=90)

plt.legend(bbox_to_anchor=(0, 1), loc='upper left', ncol=3)

plt.grid(True, linewidth= 1, linestyle="--",color ="b")
plt.show()
## Which sub-sectors seem to be correlated with high GDP?
##1. Let us see the agriculure Vs per capita

plt.scatter(x='Agriculture, forestry and fishing', y='Per Capita GSDP (Rs.)',data = Contribution,alpha=0.8)
plt.title('GDP Per Capita Vs Agriculture,forsest & fishing')
plt.xlabel('Agriculture, forestry and fishing')
plt.ylabel('Per Capita GSDP(Rs.)')
plt.show()

## We can clearly see there is not much impact on the Per Capita GDP - Relatively correlated.
## let us see the Per Capita Vs the below sub sectors 
## 1. Manufactring 
## 2. Real Estate, Ownership & Dwelling & Professional Services 
## 3. Trade, Repair, hotel & restaurants
plt.scatter(x='Real estate, ownership of dwelling & professional services', y='Per Capita GSDP (Rs.)',data = Contribution,alpha=0.8)
plt.title('Per Capita Vs Real Estate, Ownership & Dwelling & Professional Services ')
plt.xlabel('Real Estate, Ownership & Dwelling & Professional Services ')
plt.ylabel('Per Capita GSDP(Rs.)')
plt.show()

# From the above graph it is strongly correlated with the percapita GSDP(Rs.)of the states.

plt.scatter(x='Trade, repair, hotels and restaurants', y='Per Capita GSDP (Rs.)',data = Contribution,alpha=0.8)
plt.title('Per Capita Vs Trade, Repair, hotel & restaurants')
plt.xlabel('Trade, Repair, hotel & restaurants')
plt.ylabel('Per Capita GSDP(Rs.)')
plt.show()
# He is relatively correlated. the inference from the graph

plt.scatter(x='Mining and quarrying', y='Per Capita GSDP (Rs.)',data = Contribution,alpha=0.8)
plt.title('Per Capita Vs Mining and quarrying')
plt.xlabel('Mining and quarrying')
plt.ylabel('Per Capita GSDP(Rs.)')
plt.show()
# Importing the necessary libraries
# Supress Warnings

import warnings
warnings.filterwarnings('ignore')

import sys
import glob,os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# Read the Data of Data Part I-B: This dataset consists of the GSDP (Gross State Domestic Product) data for the Level of Education

Edu1 = pd.read_csv("C:/Users/103591/Downloads/Data Sets Part 2B/rs_session243_au570_1.1.csv")
Edu2 = pd.read_csv("C:/Users/103591/Downloads/Data Sets Part 1A/ab40c054-5031-4376-b52e-9813e776f65e.csv")
Edu1.head(10) # Read the first 10 rows across the data set to view the data /.
Edu2.head()
Edu1.info() # inspect the data 
Edu2.info() # inspceting the file
# Before merging let us transpose the file name We need to transpose and change the states name same for each merging purpose.

# As we dont worry about the union territories we can drop it out in both the data sets

Edu3 = Edu2.transpose()
Edu3
# Lets clean the data of Edu1 first 
Edu1

# Drop the unwated coloumns:
Edu1 = Edu1[['Level of Education - State','Primary - 2014-2015','Upper Primary - 2014-2015','Secondary - 2014-2015']]
Edu1

# let us rename the coloumn as States

Edu1.rename(columns={'Level of Education - State':'States'},inplace=True)
Edu1


# index the State 
Edu1.set_index(['States'])

new_header = Edu3.iloc[0] #grab the first row for the header
Edu3 = Edu3[1:] #take the data less the header row
                # Assign the new header
Edu3.columns = new_header 

Edu3
Edu3 = Edu3.drop(['GSDP - CURRENT PRICES (` in Crore)'],axis=1)

Edu3
new_header = Edu3.iloc[0] #grab the first row for the header
Edu3 = Edu3[1:] #take the data less the header row
                # Assign the new header
Edu3.columns = new_header 

Edu3
Edu3 = Edu3.drop(['2012-13','2013-14','2015-16','2016-17'],axis=1) # dropped the unwanted coloumns
Edu3
# Rename the coloumn in this data sets

Edu1.rename(columns={'States':'Duration'},inplace=True)

Edu1
Edu3.reset_index(level=0,inplace=True)
Edu1.rename(columns={'States':'Duration'},inplace=True)

Edu3

Edu3.rename(columns={'index':'States'},inplace=True)
Edu3
Edu1.rename(columns={'Duration':'States'},inplace=True)
Edu1
# Rename all the states proerly
Edu1.rename({'A & N Islands':'Andaman & Nicobar Islands',},inplace=True)
Merged = pd.merge(Edu1,Edu3,how='outer',on='States')
Merged
Merged.isnull().sum()
Merged.isnull().sum() > 2
round(100*(Merged.isnull().sum()/len(Merged.index)), 2)
Merged
Merged.sort_values(by='2014-15',ascending=False)

# dropped all the union territories
Merged=Merged[Merged.States != 'Delhi']
Merged=Merged[Merged.States != 'Puducherry']
Merged=Merged[Merged.States != 'Chandigarh']
Merged= Merged[Merged.States != 'A & N Islands']
Merged = Merged[Merged.States != 'Lakshadweep']
Merged = Merged[Merged.States != 'Daman & Diu']
Merged = Merged[Merged.States!='Dadra & Nagar Haveli']
Merged = Merged[Merged.States!='Andaman & Nicobar Islands']


Merged
Merged.rename(columns={'2014-15':'GDP'},inplace=True)
Merged

Merged.set_index(['States'])
# as the GDP is NAN vlaues of no use, so remove those states as well
Merged = Merged[Merged.States!='Andhra Pradesh']
Merged = Merged[Merged.States!='Uttrakhand']
Merged = Merged[Merged.States!= 'Jammu and Kashmir']
Merged = Merged[Merged.States!= 'West Bengal1']
Merged = Merged[Merged.States!= 'Chhattisgarh']
Merged
Merged.set_index(['States'])
# let us see the scatter plot of these correlations
plt.scatter(x='Primary - 2014-2015', y='GDP',data = Merged,alpha=0.8)
plt.title('Level of Primary Education Vs GDP')
plt.xlabel('Primary')
plt.ylabel('GDP')
plt.show()
# From the above graph it is saying that there is relatively positive but not strong.
plt.scatter(x='Upper Primary - 2014-2015', y='GDP',data = Merged,alpha=0.8)
plt.title('Level of Upper Primary Education Vs GDP')
plt.xlabel('Upper Primary')
plt.ylabel('GDP')
plt.show()
# This seems also be the relatively postivie side for the upper primary education Vs GDP
# From the above graph it is saying that there is relatively positive but not strong.
plt.scatter(x='Secondary - 2014-2015', y='GDP',data = Merged,alpha=0.8)
plt.title('Level of Secondary - 2014-15 Vs GDP')
plt.xlabel('Secondary')
plt.ylabel('GDP')
plt.show()
# this seems to be the postive in terms of level of Seconday education.

#Let us see across the states 

Merged.plot.bar(x='States',figsize=(18,6))
# The tile name indicated using the function below

plt.title(" Level of Education Statewise GDP Per Capita ")
plt.xlabel('States')
plt.ylabel('Education level')
plt.xticks(rotation=45)

plt.grid(True, linewidth= 1, linestyle="--",color ="r")
plt.show()

# The secondary Education level of 2014-15 is high escpcially in Nagaland, then Tripura, andhra pradesh and karnataka, The goverments needs to focus on the primary and upper secondary areas in Educaiton such as Scholarships, more government aided schools and regularise and revisit the existing policy.