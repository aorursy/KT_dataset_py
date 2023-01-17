import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline



import seaborn as sns

sns.set(color_codes=True)
mcd_df = pd.read_csv("../input/mcdonald-data-analysis/McD.csv")



#  ../input/automobiles-dataset/Automobile.csv



# Transpose - so that we can have a look at all the columns of the Dataset 



mcd_df.head()

mcd_df.describe().T   
mcd_df.info()  
mcd_df.shape   
mcd_df.isnull().sum()

mcd_df.head()
mcd_df['Category'].value_counts()
mcd_df['Category'].nunique()
mcd_df['Item'].value_counts().head(10)
mcd_df['Item'].nunique()
plt.figure(figsize=(15,7))





mcd_df["Category"].hist()
mcd_df.describe()



plt.figure(figsize=(15,10))



sns.boxplot(data=mcd_df[['Calories','Calories from Fat','Total Fat','Saturated Fat']]);



plt.figure(figsize=(15,10))



sns.boxplot(data=mcd_df[['Saturated Fat (% Daily Value)' ,'Trans Fat','Cholesterol','Cholesterol (% Daily Value)']]);



plt.figure(figsize=(3,5))

sns.boxplot(y = mcd_df["Trans Fat"]); 
mcd_df["Trans Fat"].head(10)
mcd_df["Trans Fat"].unique()
plt.figure(figsize=(9,11))



sns.boxplot(data=mcd_df[['Cholesterol','Cholesterol (% Daily Value)','Calories from Fat']]);



plt.figure(figsize=(7,5))



sns.boxplot(data=mcd_df[['Total Fat','Total Fat (% Daily Value)','Sodium (% Daily Value)']]);
plt.figure(figsize=(3,5))

sns.boxplot(y = mcd_df["Sodium"]); 
plt.figure(figsize=(15,10))



sns.boxplot(data=mcd_df[['Carbohydrates', 'Carbohydrates (% Daily Value)','Dietary Fiber', 'Dietary Fiber (% Daily Value)']]);
plt.figure(figsize=(7,5))



sns.boxplot(data=mcd_df[['Sugars', 'Protein']]);
plt.figure(figsize=(15,10))



sns.boxplot(data=mcd_df[['Vitamin A (% Daily Value)', 'Vitamin C (% Daily Value)', 'Calcium (% Daily Value)', 'Iron (% Daily Value)']]);
correlation = mcd_df.corr()



correlation
plt.figure(figsize=(15,11))





sns.heatmap(correlation,annot = True)
mcd_df['Category'].nunique()



print('The number of Categories is -', mcd_df['Category'].nunique())
mcd_df['Category'].unique()



from pandas import DataFrame



Category_List = sorted(mcd_df['Category'].unique())



type(sorted(mcd_df['Category'].unique()))



a = pd.DataFrame(Category_List,columns=[''] )  



a.index += 1



print('\n \n  Let us view the different food categories that are available to us :- \n \n  ',a )



b = pd.pivot_table(mcd_df, 'Cholesterol (% Daily Value)', index=['Category'])





Result = b.sort_values(('Cholesterol (% Daily Value)'), ascending=False)



Result
mcd_df['Item'].nunique()



print('\n \n The different type of Food items available at McD are -', mcd_df['Item'].nunique())
mcd_df['Item'].head(16)

c = pd.pivot_table(mcd_df, 'Sodium', index=['Item'])





Result_1 = c.sort_values(('Sodium'), ascending=False)





Result_1.head(15)
c = pd.pivot_table(mcd_df, 'Saturated Fat', index=['Item'])





Result_1 = c.sort_values(('Saturated Fat'), ascending=False)





Result_1.head(15)