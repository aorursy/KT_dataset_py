!pip install jovian -q --upgrade
import jovian
import pandas as pd

import numpy as np
df=pd.read_csv('./zomato.csv')

df.head(2)

print(len(df))
#df1 = pd.DataFrame(df) 

#df1.drop(['url'], axis = 1)

#print(len(df1))

df1 = df.drop(['url','address','name','phone','location','dish_liked','menu_item','listed_in(type)','reviews_list'], axis = 1)

df1.head(5)

print(len(df1))


df1['rate'].unique()

df1 = df1.loc[df1.rate !='NEW']

df1 = df1.loc[df1.rate !='-'].reset_index(drop=True)

remove_slash = lambda x: x.replace('/5', '') if type(x) == np.str else x

df1.rate = df1.rate.apply(remove_slash).str.strip().astype('float')

df1['rate'].head()



#print(len(df1))
print(len(df1))
# Optimising the dropping of null values

ctr=1

l=[]

for i in df1:

    if ctr==1:

        l.append(i)

    else:

        break

print(l)

new_data = df1.dropna(axis = 0, how ='any')

print("Old data frame length:", len(df1), "\nNew data frame length:",  

       len(new_data), "\nNumber of rows with at least 1 NA value: ", 

       (len(df1)-len(new_data))) 



#df1.head()
mean_rate = df1['rate'].mean(skipna = True)

round(mean_rate , 2)



df1['rate'].fillna(value = mean_rate, inplace = True)

df1.head(57000)

#print(len(df1))
print(len(df1))
null_columns=df1.columns[df1.isnull().any()]

df1[null_columns].isnull().sum()
df2 = df1.dropna(axis = 0, how = 'any')



# comparing sizes of data frames 

print("Old data frame length:", len(df1), "\nNew data frame length:",  

       len(df2), "\nNumber of rows with at least 1 NA value: ", 

       (len(df1)-len(df2))) 
jovian.commit()