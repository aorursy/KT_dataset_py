



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

get_ipython().magic('matplotlib inline')







df=pd.read_csv("../input/mushrooms.csv")







df.head()



df. describe()



sns.countplot(data=df,x='odor',hue='class')





# In[35]:



from sklearn.preprocessing import LabelEncoder

le=LabelEncoder()

for i in df.columns:

    df[i]=le.fit_transform(df[i])

df.head()





# In[36]:



from sklearn.model_selection import train_test_split





# In[37]:



X=df.drop('class',axis=1)

y=df['class']





# In[53]:



X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=101)





# In[54]:



from sklearn.tree import DecisionTreeClassifier





# In[55]:



dtree=DecisionTreeClassifier()





# In[56]:



dtree.fit(X_train,y_train)





# In[57]:



predictions=dtree.predict(X_test)





# In[58]:



from sklearn.metrics import classification_report,confusion_matrix





# In[59]:



print(classification_report(y_test,predictions))





# In[60]:



print(confusion_matrix(y_test,predictions))





# In[62]:



from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators=100)

rfc.fit(X_train, y_train)





# In[63]:



rfc_predict=rfc.predict(X_test)





# In[64]:



print(confusion_matrix(y_test,rfc_predict))





print(classification_report(y_test,rfc_predict))








