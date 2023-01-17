import numpy as np
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt
%matplotlib inline 
bf=pd.read_csv('../input/BlackFriday.csv')
bf.head()
bf.info()  
bf.columns
bf.isnull().head()
sns.heatmap(bf.isnull(),yticklabels=False,cbar=False,cmap='viridis')
#  fill the missing data with zeros

bf=bf.fillna(value='0')
# we can to observe that our DataFrame is clean

bf.head() 
sns.heatmap(bf.isnull(),yticklabels=False,cbar=False,cmap='viridis')
len(bf['Product_ID'].value_counts()) # products offered
bf['Product_ID'].value_counts().head(10) # 10 most sold
sns.set_style('darkgrid')
bf['Product_ID'].value_counts().head(10).plot(kind='bar',color='green')

# Sum of purchases made by user

Buy_by_User=bf[['User_ID','Purchase']].groupby('User_ID').sum() 

# Top 10 of the users who make more purchases

Buy_by_User.sort_values(by=['Purchase'],ascending=False).head(10)

bf['User_ID'].nunique() # We note that have 5891 users
Age_by_User=bf[['User_ID','Age']].groupby(['User_ID', 'Age']).count() 
test=pd.DataFrame(Age_by_User)
# New dataframe to evaluate the number of clients by age category
New_Age_by_User=test.reset_index(inplace=False) 
New_Age_by_User.groupby('Age').count()
sns.set_style('darkgrid')
New_Age_by_User.groupby('Age').count().plot(kind='bar',color='green')
Gender_by_User=bf[['User_ID','Gender']].groupby(['User_ID', 'Gender']).count() 
test_Gender=pd.DataFrame(Gender_by_User)
# New dataframe to evaluate the number of clients by age category
New_Gender_by_User=test_Gender.reset_index(inplace=False) 
New_Gender_by_User.groupby('Gender').count()
bf_gender=pd.DataFrame(index=['F','M']
                       ,columns=['# Users'],data=[1666,4225]) # we create a dataframe
plt.pie(New_Gender_by_User.groupby('Gender').count()
        ,autopct='%1.1f%%',labels=bf_gender.index,shadow=True, startangle=90)
plt.legend()
bf[['Gender','Purchase']].groupby([ 'Gender']).sum()
bf_gender_pur=pd.DataFrame(index=['F','M'],columns=['Purchase']
                           ,data=[1164624021,3853044357]) # we create a dataframe
plt.pie(bf[['Gender','Purchase']].groupby([ 'Gender']).sum()
        ,autopct='%1.1f%%',labels=bf_gender_pur.index,shadow=True, startangle=90)
plt.legend()
bf[['Marital_Status','Purchase']].groupby([ 'Marital_Status']).sum()
bf_marital_sta=pd.DataFrame(index=['singles','married'],columns=['Purchase']
                           ,data=[2966289500,2051378878]) # we create a dataframe
plt.pie(bf[['Marital_Status','Purchase']].groupby([ 'Marital_Status']).sum()
        ,autopct='%1.1f%%',labels=bf_marital_sta.index,shadow=True, startangle=90,radius=0.8)
plt.legend(loc=1)
Stay_City=bf[['Stay_In_Current_City_Years','Purchase']].groupby([ 'Stay_In_Current_City_Years']).sum()
Stay_City.sort_values(by=['Purchase'],ascending=False)
sns.set_style('darkgrid')
Stay_City.plot(kind='bar',color='g')
Prod_By_Gender=bf[['Product_ID','Purchase','Gender']].groupby([ 'Product_ID','Gender']).sum()
test_prod=pd.DataFrame(Prod_By_Gender)
New_product_by_gender=test_prod.reset_index(inplace=False)
New_product_by_gender[New_product_by_gender['Gender']=='M'].sort_values(by=['Purchase']
                                                        ,ascending=False).head(10)
New_product_by_gender[New_product_by_gender['Gender']=='F'].sort_values(by=['Purchase']
                                                        ,ascending=False).head(10)
left=New_product_by_gender[New_product_by_gender['Gender']=='M'].sort_values(by=['Purchase']
                                                        ,ascending=False).head(10)
right=New_product_by_gender[New_product_by_gender['Gender']=='F'].sort_values(by=['Purchase']
                                                    ,ascending=False).head(10)
pd.merge(left, right, how='inner',on='Product_ID')
left_less=New_product_by_gender[New_product_by_gender['Gender']=='M'].sort_values(by=['Purchase'],ascending=False).tail(10)
left_less    
right_less=New_product_by_gender[New_product_by_gender['Gender']=='F'].sort_values(by=['Purchase'],ascending=False).tail(10)
right_less
pd.merge(left_less, right_less, how='inner',on='Product_ID')
#  Number of occupations
bf['Occupation'].nunique() 
# Sum of purchases made by user

Buy_by_User=bf[['Occupation','Purchase']].groupby('Occupation').sum() 

# Top 10 of the users who make more purchases

Buy_by_User.sort_values(by=['Purchase'],ascending=False)
sns.set_style('darkgrid')
Buy_by_User.plot(kind='bar',color='g')
