# This data set contains data about purchases made during a Black Friday along with demographic data of the customers.



# Our goal in this exploratory analysis is to find correlations between the purchases made and demographic data to 

# deliver our findings to the marketing department. These findings should assist in a better targeting those customers 

# who are already more loyal to the store AS WELL AS as finding any missed (if any) opportunities.  



# Analyzing data: 

# 1) What were the top products sold on BF? 

# 2) Who is our most loyal customer on BF? 

# 3) Which cities are driving most sales? 





import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib as mlp

import matplotlib.pyplot as plt

import os

datafile = "../input/BlackFriday.csv"

df=pd.read_csv(datafile)

df.info();
df.shape
df.head()
df.info()
# Filling missing values



df = df.fillna(0)
df.describe()
# Before starting to dive into different parameters let's look at the distribution of spendings during the Black Friday



plt.hist(df['Purchase'],bins=20, alpha=0.6, color='g');



# The average purchase size was around $9,333 while the median purchase was at $8,062
# 1) Let's look at the top 10 products that were most popular during the Black Friday



df['Product_ID'].value_counts().sort_values(ascending=False).head(10)



# If we could compare this data to the regular day sales during the rest of the year we could find out 

# which products are the most in demand on the BF specifically
# 2) let's look at the data by gender 

# How many women vs men were shopping on Black Froday 



plt.pie(df.groupby(['Gender']).size(), radius=1, labels=['F', 'M'], wedgeprops={'alpha':0.6},colors=['r','b'] );

plt.axis('equal');



# Who could have thought that men are such a bargain hunters?  
# How the amount of $ spent differed between men and women 



grouped = df.groupby('Gender').agg({'Purchase':'mean'})

print (grouped)

grouped.plot(kind='bar', legend=None, alpha = 0.6, color = ['r', 'b']);



# Not only men are more in numbers but they also spend on average $700 more during the BF
# And now let's see if single people spend more in our stores 



grouped = df.groupby('Marital_Status').agg({'Purchase':'mean'})

print (grouped)

grouped.plot(kind='bar', legend=None, alpha = 0.7);



# It looks like on average there is no difference in spendings
# How about the age groups? Is it a younger or older group that visits our stores more often?  



grouped = df.groupby('Age').agg({'Purchase':'count'})

grouped.plot(kind='bar', legend=None, alpha = 0.6, color = 'y');



# It looks like our stores are popular among the mid aged people in their early 30s
# Let's now see men's and women's behavior in various age groups  



age_order = ['0-17','18-25','26-35','36-45','46-50','51-55','55+']



plt.subplots(figsize=(10,5))

plt.subplot(1, 2, 1)

sns.countplot('Age',order=age_order,hue='Gender',data=df, alpha = 0.7)

plt.xlabel('Age')

plt.ylabel('')

plt.xticks(rotation=90)

plt.title('Number of customers')

plt.legend(['Female','Male'])



plt.subplot(1,2,2)

df_by_Age = df.groupby(['Age','Gender']).agg({'Purchase':np.sum}).reset_index()

sns.barplot('Age','Purchase',hue='Gender',data=df_by_Age, alpha = 0.7)

plt.xlabel('Age')

plt.ylabel('')

plt.xticks(rotation=90)

plt.title('Total purchase')

plt.legend().set_visible(False)



# These graphs show that our primary clientelle were men in their early 30s and they tend to spend most. 
# And now let's dive deeper to help our marketing folks to paint the portrait of our most loyal customers



df['combined_G_M'] = df.apply(lambda x:'%s_%s' % (x['Gender'],x['Marital_Status']),axis=1)

sns.countplot(df['Age'],hue=df['combined_G_M'], order=age_order, alpha = 0.7)



# Here we can conclude that single men in 26-35 years old are our primary clientelle. On the second place would be married men

# in the same age group. And the third place goes to single men 18-25 years old. 
# Does the time lived in a city play a role in shopping during the BF? 



grouped = df.groupby('Stay_In_Current_City_Years').agg({'Purchase':'sum'})

grouped.plot(kind='bar', legend=None, alpha = 0.7);



# We can clearly see that the those who lived at least one year but no more that 2 years in the city tend to visit our store during the BF.
# How the amount of $ spent differed between various occupations 



grouped = df.groupby('Occupation').agg({'Purchase':'sum'})

grouped.plot(kind='bar', legend=None, alpha = 0.7);



# Our marketing team will be happy to know that people in occupations 4, 0, and 7 are the biggest spenders. 

# This could considerably help them narrow down and target those potential customers based on their occupation 
# Let's now see our top customers based on age, marital status, and their occupation



df['combined_G_M_O'] = df.apply(lambda x:'%s_%s_%s' % (x['Gender'],x['Marital_Status'],x['Occupation']),axis=1)

print (df['combined_G_M_O'].value_counts().sort_values(ascending=False).head(15))



# We see the already clearly established trend that single men in their early 30s who are in occupations 4, 0, and 7 

# are our target audience 
# 3) Let's look at the sales generated by the stores in each city 



grouped = df.groupby('City_Category').agg({'Purchase':'sum'})

grouped.plot(kind='bar', legend=None, alpha = 0.7);





#Let's look at the number of visitors per store 



grouped = df.groupby('City_Category').agg({'User_ID':'count'})

grouped.plot(kind='bar', legend=None, alpha = 0.7);



# The distribution is tha same as with the sales generated, thus we can conlcude that the city B drives more sales 

# through more customers rather than through bigger size of purchases 
# CONCLUSION

# 1) We have identified top 10 most popular products on BF

# 2) Our most loyal customer on BF are single men in their early 30s who are in occupations 4, 0, and 7 

# 3) The city B did the best during this season in attracting the most customers and thus selling more  
# Let's look at correlation matrix



df.corr()
from sklearn.feature_selection import SelectPercentile

from sklearn.feature_selection import f_regression



data=df.drop(['Product_ID','User_ID','combined_G_M_O','combined_G_M'],axis=1)



data['Gender']=data['Gender'].map( {'M': 1, 'F': 0} ).astype(int)

data['City_Category']=data['City_Category'].map( {'A': 0, 'B': 1, 'C':2} ).astype(int)

data['Age']=data['Age'].map( {'0-17': 0, '18-25': 1, '26-35': 2,'36-45':3,'46-50':4,'51-55':5,'55+':6} ).astype(int)

data['Stay_In_Current_City_Years']=data['Stay_In_Current_City_Years'].map( {'0': 0, '1': 1, '2': 2,'3':3,'4+':4}).astype(int)

df['Product_Category_1']=df['Product_Category_1'].astype(int)

df['Product_Category_2']=df['Product_Category_2'].astype(int)

df['Product_Category_3']=df['Product_Category_3'].astype(int)



X=data.drop(['Purchase'],axis=1).values

y=data['Purchase'].values



# Chosing the features based on the percentile scores 

Selector_f = SelectPercentile(f_regression, percentile=25)

Selector_f.fit(X,y)

name_score=list(zip(data.drop(['Purchase'],axis=1).columns.tolist(),Selector_f.scores_))

name_score_df=pd.DataFrame(data=name_score,columns=['Feat_names','F_scores'])

name_score_df.sort_values('F_scores',ascending=False)

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_absolute_error

from sklearn.metrics import mean_squared_error



data=df.copy()

data=data[['City_Category','Gender','Purchase', 'Product_Category_1','Product_Category_3']]



data=pd.get_dummies(data=data,columns=['City_Category','Gender', 'Product_Category_1', 'Product_Category_3'])



data.drop(['City_Category_A','Gender_F','Product_Category_1_1', 'Product_Category_3_0'],axis=1,inplace=True)



X=data.drop(['Purchase'],axis=1).values

y=data['Purchase'].values



X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)



sc_X=StandardScaler()

X_train=sc_X.fit_transform(X_train)

X_test=sc_X.transform(X_test) 



regressor=LinearRegression()

regressor.fit(X_train,y_train)



y_pred=regressor.predict(X_test)

print("Prediction\n",y_pred)

print("Actual\n",y_test)

print("R_squared Score:",regressor.score(X_test,y_test))

mae = mean_absolute_error(y_test,y_pred)

print("MAE:",mae)

print("RMSE:",mean_squared_error(y_test,y_pred)**0.5)