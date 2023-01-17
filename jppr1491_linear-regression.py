#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


df = pd.read_excel (r'your_path_to_the_file.xlsx') #I save it as EXCEL File
# we want to create a linear regression model that predicts the "selling price" for a car given the data


# In[4]:


df.head()


# In[5]:


# check the if theres any association between the selling price and the other variables
plt.figure(figsize=(15,10))
plt.xlabel("Present_Price")
plt.ylabel("Selling_Price")
plt.scatter(df.Present_Price, df.Selling_Price,marker='+', color='blue')


# In[6]:


plt.figure(figsize=(20,10))
plt.xlabel("Kms_Driven")
plt.ylabel("Selling_Price")
plt.scatter(df.Kms_Driven, df.Selling_Price,marker='+', color='blue')


# In[7]:


plt.figure(figsize=(20,10))
plt.xlabel("Year")
plt.ylabel("Selling_Price")
plt.scatter(df.Year, df.Selling_Price,marker='+', color='blue')


# In[8]:


# turn "string" type data as a factor

df["fact_Car_Name"] = pd.factorize(df.Car_Name)[0]
df["fact_Fuel_Type"] = pd.factorize(df.Fuel_Type)[0]
df["fact_Seller_Type"] = pd.factorize(df.Seller_Type)[0]
df["fact_Transmission"] = pd.factorize(df.Transmission)[0]

df.head()


# In[9]:


plt.scatter(df.fact_Transmission, df.Selling_Price, marker = "+")


# In[10]:


plt.scatter(df.fact_Seller_Type, df.Selling_Price, marker = "+")


# In[11]:


plt.scatter(df.fact_Car_Name, df.Selling_Price, marker = "+")


# In[76]:


plt.scatter(df.fact_Fuel_Type, df.Selling_Price, marker = "+")


# In[12]:


plt.scatter(df.Owner, df.Selling_Price, marker = "+")


# In[ ]:


# These graphs lead me to believe that all variable may have influence on the selling price
# Some evident conclusions are that a more recent car has a higher selling price and the same can be said about Kms_driven


# In[13]:


# Modeling

import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


# In[48]:


# A backward method is used to select what variables will take part in the model 

X = df[["fact_Car_Name", "Year", "Kms_Driven", "Present_Price", 
                   "fact_Fuel_Type", "fact_Seller_Type", "fact_Transmission", "Owner"]]

# including a line like Y = df.Selling_Price would make the code cleaner, but this way is more clear to me

X_train, X_test, Y_train, Y_test = train_test_split(X, df.Selling_Price, test_size=0.25, random_state=69)


# In[49]:


regr1 = linear_model.LinearRegression()
regr1.fit(X_train,Y_train)


# In[50]:


# If we are interested on checking the coeficients for the model
#print('Intercept: \n', regr1.intercept_)
#print('Coefficients: \n', regr1.coef_)


# In[51]:


regr1.predict(X_test)


# In[47]:


regr1.score(X_test,Y_test)


# In[35]:


# 88% of the models variance is explained by the current model
# but is it good enough? can we make it better? So what is next?
# from all the variables the name may or may not influenciate the Selling_Price, so lets see if the model gets better without it


# In[54]:


X2 = df[["Year", "Kms_Driven", "Present_Price", "fact_Fuel_Type", "fact_Seller_Type", "fact_Transmission", "Owner"]]

# including a line like Y = df.Selling_Price would make the code cleaner, but this way is more clear to me

X2_train, X2_test, Y2_train, Y2_test = train_test_split(X2, df.Selling_Price, test_size=0.25, random_state=69)


# In[55]:


regr2 = linear_model.LinearRegression()
regr2.fit(X2_train,Y2_train)


# In[59]:


regr2.predict(X2_test)


# In[ ]:


# like before checking the coeficients is an option


# In[60]:


regr2.score(X2_test,Y2_test)


# In[ ]:


# the model score almost didnt change which means excluding the Car_name just saves sometime on the calculations
# for the next model we will remove the Owner variable and see if the model gets better
# of course that a car with zero owners to date, a brand new car, will be more expensive than a car that had 1 or more owners


# In[61]:


X3 = df[["Year", "Kms_Driven", "Present_Price", "fact_Fuel_Type", "fact_Seller_Type", "fact_Transmission"]]

# including a line like Y = df.Selling_Price would make the code cleaner, but this way is more clear to me

X3_train, X3_test, Y3_train, Y3_test = train_test_split(X3, df.Selling_Price, test_size=0.25, random_state=69)


# In[62]:


regr3 = linear_model.LinearRegression()
regr3.fit(X3_train,Y3_train)


# In[63]:


regr3.predict(X3_test)


# In[64]:


regr3.score(X3_test,Y3_test)


# In[ ]:


# the model improved in quality 
# so at this point I'm thinking that the variables that should be considered for the model are Year, Kms_driven, Fuel_Type
# will do one with Seller_Type and later another with Transmission as last variable


# In[67]:


X4 = df[["Year", "Kms_Driven", "Present_Price", "fact_Fuel_Type", "fact_Seller_Type"]]

# including a line like Y = df.Selling_Price would make the code cleaner, but this way is more clear to me

X4_train, X4_test, Y4_train, Y4_test = train_test_split(X4, df.Selling_Price, test_size=0.25, random_state=69)


# In[69]:


regr4 = linear_model.LinearRegression()
regr4.fit(X4_train,Y4_train)


# In[70]:


regr4.predict(X4_test)


# In[71]:


regr4.score(X4_test,Y4_test)


# In[ ]:


# this model is almost at a score of 90% which leads me to believe that Transmission has less impact on Selling_Price than 
# Seller_Type  


# In[68]:


X5 = df[["Year", "Kms_Driven", "Present_Price", "fact_Fuel_Type", "fact_Transmission"]]

# including a line like Y = df.Selling_Price would make the code cleaner, but this way is more clear to me

X5_train, X5_test, Y5_train, Y5_test = train_test_split(X5, df.Selling_Price, test_size=0.25, random_state=69)


# In[72]:


regr5 = linear_model.LinearRegression()
regr5.fit(X5_train,Y5_train)


# In[74]:


regr5.predict(X5_test)


# In[75]:


regr5.score(X5_test,Y5_test)


# In[ ]:


# As stated above we have better predictiong with Seller_Type 
# So at this point if I was to use a linear model to make predictions on Selling_Price I would use model 4 
# So the challenge now is to work with model 4 variables and decide what to variable to remove
# This time I think we just have to see what happens to the model when you remove Seller_Type
# if it gets a better score then Seller_Type isnt needed 


# In[77]:


X6 = df[["Year", "Kms_Driven", "Present_Price", "fact_Fuel_Type"]]

# including a line like Y = df.Selling_Price would make the code cleaner, but this way is more clear to me

X6_train, X6_test, Y6_train, Y6_test = train_test_split(X6, df.Selling_Price, test_size=0.25, random_state=69)


# In[78]:


regr6 = linear_model.LinearRegression()
regr6.fit(X6_train,Y6_train)


# In[79]:


regr6.predict(X6_test)


# In[80]:


regr6.score(X6_test,Y6_test)


# In[ ]:


# so it seems with the current data and if we want to use a linear model to make predictions, we have to use model 4 

# X4 = df[["Year", "Kms_Driven", "Present_Price", "fact_Fuel_Type", "fact_Seller_Type"]]

# X4_train, X4_test, Y4_train, Y4_test = train_test_split(X4, df.Selling_Price, test_size=0.25, random_state=69)