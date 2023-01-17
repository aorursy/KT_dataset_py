# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#importing libraries for data visualisation 
import matplotlib.pyplot as plt 
import seaborn as sns 
import plotly.express as px  
import plotly.graph_objects as go  
from wordcloud import WordCloud 
%matplotlib inline 

from sklearn.model_selection import train_test_split 
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.naive_bayes import MultinomialNB 
from sklearn.metrics import accuracy_score 
df=pd.read_csv("../input/amazon-alexa-reviews/amazon_alexa.tsv",sep="\t")
#checking if any null records present in data frame 
df.isnull().sum()/len(df)*100
#checking each every data type in data set 
df.dtypes
rating=df.groupby("rating")["rating"].sum() 
rating
labels=["rating1","rating2","rating3","rating4","rating5"]
fig=px.pie(df,values=rating,names=labels[0:5])  
fig.update_traces(textposition="inside",textinfo="percent+label") 
fig.show()
fig = go.Figure(data=[go.Bar(
            x=labels, y=rating[0:10],
            text=rating[0:10],
            textposition='auto',
            marker_color=['green',"red","yellow","black"]

        )]) 
fig.show()
#checking how much  more variations are good with customers
plt.figure(figsize=(20,20))
wc=WordCloud().generate(" ".join(df["variation"])) 
plt.imshow(wc)
# As we can see most of the persons are playing with their devices are Alexa,Love.....
plt.figure(figsize=(20,200))
wc=WordCloud().generate(" ".join(df["verified_reviews"])) 
plt.imshow(wc)
#checking feed back of their devices i.e 1 means positive 0 means negative 
feed_back=df["feedback"].value_counts() 
feed_back
#representation of feed_back in terms of pie chart i.e positive and negative feed_back 
reviews=["positive_feed back","negative_feedback"]
fig=px.pie(df,values=feed_back,names=reviews,hole=0.4)
fig.update_traces(textposition="inside",textinfo="percent+label")  
fig.update_layout(template="plotly_dark")
#representation feedback in terms of bar graph  
fig=go.Figure(data=[go.Bar(x=reviews,y=feed_back[0:10],text=feed_back[0:10],textposition="auto",marker_color=["green","pink"])]) 
fig.update_layout(title="Number of positive and negative responses", template="plotly_dark")
fig.show()
diff_devices=df["variation"].value_counts()  
df1=pd.DataFrame(data=diff_devices)  
df1["devices"]=df1.index 
df1.reset_index(inplace=True) 
col=df1.columns 
f=list(col) 
f[0]="Devices" 
df1.columns=f 
df1.drop("Devices",axis=1,inplace=True) 
df1
uni=df["variation"].unique().tolist() 
uni
fig=px.pie(df,values=diff_devices[0:16],names=uni,hole=0.2) 
fig.update_traces(textposition="inside",textinfo="percent+label") 
fig.update_layout(title="Percentage of Each device usage",template="plotly_dark")
fig=px.treemap(df1,path=["devices"],values=df1["variation"],title="Different types of device and their number of users",height=700) 
fig.data[0].textinfo="label+text+value" 
fig.update_layout(template="plotly_dark")
fig.show()
df1=pd.DataFrame(data=df.groupby(["variation"])["rating"].value_counts()) 
df1.unstack() 
df1=df1["rating"].astype(int) 
df1=df1.unstack() 
df1=df1.fillna(30) 
df1.reset_index(inplace=True) 
df1
df1.columns=['variation',"rating1","rating2", "rating3","rating4", "rating5"]
#As we can see in the below black dot variation receiving highest rating when comparing to other devices
rating1=go.Bar(x=df1["variation"],y=df1.rating1,name ='rating1') 
rating2=go.Bar(x=df1["variation"],y=df1.rating2,name ='rating2') 
rating3=go.Bar(x=df1["variation"],y=df1.rating3,name ='rating3') 
rating4=go.Bar(x=df1["variation"],y=df1.rating4,name ='rating4') 
rating5=go.Bar(x=df1["variation"],y=df1.rating5,name ='rating5',marker=dict(color = 'rgba(255, 128, 255, 0.8)')) 
data=[rating1,rating2,rating3,rating4,rating5] 
layout={"title":"Variations between devices and Ratings",
         "barmode":"relative", 
         "xaxis":{"title":"Variations"} , 
         "yaxis":{"title":"Ratings"}, 
          "template":"plotly_dark"}
fig=go.Figure(data,layout)   
fig.update_traces()
fig.show()
df2=pd.DataFrame(data=df.groupby("variation")["feedback"].value_counts()) 
df2=df2.unstack() 
df2["variation"]=df2.index 
df2=pd.DataFrame(data=df2)  
np.random.seed(1)
df2.index=np.random.randint(0,17,16) 
df2
#as we can seee now most of the positive feedback are receiving blackdot amazon alexa device 
negative=go.Bar(x=df2['variation'],y=df2["feedback"][0],name="negative feed back")  
positive=go.Bar(x=df2['variation'],y=df2["feedback"][1],name="positive feed back")  
data=[positive,negative]  
layout={"title":"Number of members giving feed back "}
fig=go.Figure(data,layout) 
fig.show()
cv=CountVectorizer() 
x=cv.fit_transform(df["verified_reviews"]).toarray() 
x
y=df["feedback"]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.50)
mn=MultinomialNB()
mn.fit(x_train,y_train)
print("Training score :"+str(mn.score(x_train,y_train)))
print("Testing score :"+str(mn.score(x_test,y_test))) 
x_test
