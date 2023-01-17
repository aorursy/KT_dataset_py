import numpy as np

import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns                         # seaborn is based on Matplotlib  

sns.set(color_codes = True)                   # adds lot of color 

%matplotlib inline                           



# tells to print the plot not just store







auto_df = pd.read_csv("../input/automobiles-dataset/Automobile.csv")

auto_df.head()
auto_df.head().T
auto_df.hist(figsize=(10,10))
auto_df['make'].nunique()
auto_df['horsepower'].max()
auto_df.columns
#auto_df.loc[auto_df["horsepower"] == 262, : ]





# I want only specific columns 





auto_df.loc[auto_df["horsepower"] == 262,  ['horsepower','make','drive_wheels','engine_location','price','city_mpg'] ]
# 2 categorical columns 
auto_df.columns
pd.crosstab(auto_df['engine_location'],auto_df['fuel_type'])
#pd.pivot_table(auto_df,index=['price'],columns=['number_of_doors'] ,aggfunc=len)



#pd.pivot_table(auto_df,index=['price'])#,columns=['number_of_doors'] ,aggfunc=len)



#pd.pivot_table(auto_df,index=['price','make']).head() #,columns=['make'] ,aggfunc=len)



#pd.pivot_table( auto_df, index=['price','make']), columns=['horsepower'] , aggfunc=len )



pd.pivot_table(auto_df,index=['horsepower','make','price']).tail()#,columns=['number_of_doors'] ,aggfunc=len).tail()
pd.pivot_table(auto_df,index=['price','horsepower','make'],columns=['number_of_doors'] ,aggfunc=len).tail()
#pd.pivot_table(auto_df, 'horsepower', index=['price','make'],columns=['number_of_doors'] ,aggfunc=len).tail()



pd.pivot_table(auto_df, 'horsepower', index=['price','make'],columns=['number_of_doors']).tail()  # ,aggfunc=len)





pd.pivot_table(auto_df, 'price', index=['make']).tail() #,columns=['number_of_doors']).tail()

pd.pivot_table(auto_df, 'horsepower', index=['make']).tail()
auto_df['make'].value_counts()
auto_df.columns
plt.figure(figsize=(14,7))



sns.distplot(auto_df['price'])







# The wave is the Kernel Density Estimator 



# If density is High it will go up and vice-versa



# The peak denotes that Cars in the price range of around $10,ooo are high in number. 
plt.figure(figsize=(14,7))



sns.distplot(auto_df['peak_rpm'])





import warnings



warnings.filterwarnings('ignore')



auto_df.columns
#plt.figure(figsize=(5,10))



auto_df.hist(by='number_of_doors',column = 'horsepower',figsize=(15,7) )






auto_df.hist(by='make',column = 'horsepower',figsize=(12,14))
sns.distplot(auto_df['highway_mpg'], kde = False ); #  , rug = True,vertical=False,norm_hist=True  );



#kde is important

# graph is looking like actual graph 







    #vertical=False,

    #norm_hist=False



#  mpg - miles per gallon



#putting the ;    is very important
sns.jointplot(auto_df["engine_size"],auto_df["horsepower"]);
sns.jointplot(auto_df["engine_size"],auto_df["horsepower"],kind ="hex" );
sns.jointplot(auto_df["engine_size"],auto_df["horsepower"],kind ="kde" );
auto_df['fuel_type'].unique()
auto_df['horsepower'].unique()
sns.stripplot(auto_df["fuel_type"],auto_df["horsepower"], jitter = True);





# jitter is important 



plt.figure(figsize=(12,6))



sns.swarmplot(auto_df["fuel_type"],auto_df["horsepower"]);







plt.figure(figsize=(12,6))



sns.boxplot(auto_df["fuel_type"],auto_df["horsepower"]);





# what does this say  ?





# The line denotes the median - 50% of Gas cars less than 90 Horse Power (below the line)



#  50% of Gas cars More than 90 Horse Power (above the line)





#   Each line from the box is 





#   Upper Line  - 75% percentile 

#   Middle line - 50% percentile

#   Base Line   - 25% percentile                      - refer to the vertical axis and understand that 



# below that 25% percentile line we have that much Horsepower cars and lesser 



sns.barplot(auto_df["body_style"],auto_df["horsepower"], hue = auto_df["fuel_type"] );
sns.barplot(auto_df['body_style'],auto_df['horsepower'],  hue = auto_df["number_of_doors"] );





#   by adding -  hue =  adf[]





#   We are able to add important condition 





#   to the Vizualizations - which makes more sense 



sns.lmplot(y ='engine_size', x = 'horsepower', data = auto_df);
import seaborn as sns
#sns.pairplot(auto_df)
correlation = auto_df.corr()



correlation
plt.figure(figsize=(14,11))





sns.heatmap(correlation,annot = True)