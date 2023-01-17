import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
app_data=pd.read_csv("../input/googleplaystore.csv")
app_review=pd.read_csv("../input/googleplaystore_user_reviews.csv")
app_data.columns
app_data["Installs"]=app_data["Installs"].str.replace(",","",regex=True)
app_data["Installs"]=app_data["Installs"].str.replace(".$","",regex=True)
app_data=app_data[app_data["Installs"]!="Fre"]
app_data=app_data[app_data["Installs"]!=""]
app_data["Installs"]=pd.to_numeric(app_data["Installs"])
app_data=app_data[app_data["Installs"]>10000]
app_data.drop(columns="Installs",inplace=True)
app_data["Reviews"]=pd.to_numeric(app_data["Reviews"])
app_data=app_data[app_data["Reviews"]>1000]
app_data.drop(columns="Reviews",inplace=True)
app_data["Content Rating"].value_counts()
mask1=app_data["Content Rating"].isin(["Everyone","Teens"])
app_data=app_data[mask1]

mask2=app_data["Type"]=="Free"
app_data=app_data[mask2]
app_data.head()
app_data.shape
app_data.drop(columns=["Size","Type","Content Rating","Genres"],inplace=True)
x=app_data["Category"].value_counts()>50
x=x[x]
listx=x.index.tolist()
app_data=app_data[app_data["Category"].isin(listx)]
#app_data.drop(columns="Category",inplace=True)
app_data.replace("Varies with device",np.NaN,inplace=True)
app_data["Current Ver"].str.replace("[!.]", "", regex=True)

app_data["Current Ver"]=pd.to_numeric(app_data["Current Ver"],errors="coerce")
app_data["Android Ver"]=app_data["Android Ver"].str.replace(" and up","",regex=True)
app_data=app_data[~app_data["Android Ver"].isin(["4.0.3 - 7.1.1","5.0 - 8.0","4.1 - 7.1.1"])]
app_data["Android Ver"].replace({"4.0.3":"4.0","2.3.3":"2.3","2.0.1":"2.0"},inplace=True)
app_data["Android Ver"].fillna(4.0,inplace=True)
app_data["Current Ver"].fillna(app_data["Current Ver"].mean(),inplace=True)

app_data.dropna(inplace=True)
app_data.shape
app_review.shape
app_review.drop(columns=["Sentiment_Polarity","Sentiment_Subjectivity"],inplace=True)
app_review.dropna(inplace=True)
app_review.shape
app_data=app_data.merge(app_review,on="App")
app_review.columns
app_data["Price"]=pd.to_numeric(app_data["Price"].str.replace("$","",regex=True))

app_data.shape
app_data.drop(columns="App",inplace=True)
app_data.columns
app_data.drop(columns="Last Updated",inplace=True)
app_data.columns
app_data["Android Ver"]=pd.to_numeric(app_data["Android Ver"])
y=app_data.iloc[:,1].values
y
words=[]
for i in app_data.iloc[:,5]:
    words+=i.split()
len(words)
preplist=["about","above","across","after","against","among","around","at","before","behind","below","beside","between","by","down","during","except","for","from","in","inside","into","near","of","off","on","out","over","through","to","toward","under","up","with"]
prepcent=[i.capitalize() for i in preplist]
prepupper=[i.upper() for i in preplist]

pronouns=["I","we","you","he","she","it","they","me","us","you","her","him","it","them"]
procent=[i.capitalize() for i in pronouns]
proupper=[i.upper() for i in pronouns]

conjunctions=["for", "and", "nor", "but", "or", "yet", "so", "after", "although", "as", "because", "before", "if", "than", "that", "though", "till", "unless", "until", "when", "whenever", "where", "wherever", "while", "neither", "nor", "either"]
concent=[i.capitalize() for i in conjunctions]
conupper=[i.upper() for i in conjunctions]

determiners=["the","a","an","this", "that", "these", "those", "my", "your", "his", "her", "its", "our", "their", "much", "many", "most", "some", "any", "enough","all", "both", "half", "either", "neither", "each", "every", "other", "another", "such", "what", "rather", "quite"]
detcent=[i.capitalize() for i in determiners]
detupper=[i.upper() for i in determiners]

for i in range(len(words)):
    if not words[i].isalpha():
        words[i]=""
    if ((words[i] in preplist) or (words[i] in prepcent) or(words[i] in prepupper)):
        words[i]=""
    if ((words[i] in pronouns) or (words[i] in procent) or (words[i] in proupper)):
        words[i]=""
    if ((words[i] in conjunctions) or (words[i] in concent) or (words[i] in conupper)):
        words[i]=""
    if ((words[i] in determiners) or (words[i] in detcent) or (words[i] in detupper)):
        words[i]=""

len(words)
from collections import Counter
word_dict=Counter(words)
del word_dict[""]
len(word_dict)
word_dict=word_dict.most_common(3000)
len(word_dict)
features=[]
for i in app_data.iloc[:,5]:
    blob=i.split()
    data=[]
    
    for j in word_dict:
        data.append(blob.count(j[0]))
    features.append(data)
len(features)
app_data.columns
from sklearn.preprocessing import LabelEncoder
encoder=LabelEncoder()
app_data["Category"]=encoder.fit_transform(app_data.iloc[:,0].values)
from sklearn.preprocessing import OneHotEncoder
oencoder=OneHotEncoder(categorical_features=[0])
labelled=oencoder.fit_transform(app_data.iloc[:,0:1].values)
labelled.shape
labelled=labelled.toarray()
labelled=labelled[:,1:-1]
labelled.shape

features=np.array(features)
app_data.drop(columns="Translated_Review",inplace=True)
app_data.columns
app_data.drop(columns=["Sentiment","Category"],inplace=True)
app_data.iloc[:,1:4].columns
labelled.shape
features=np.append(features,app_data.iloc[:,1:4].values,axis=1)
features=np.append(features,labelled,axis=1)
app_data.iloc[:,1:4].values.shape
features.shape
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(features,y,test_size=0.2,random_state=9)
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)
from sklearn.ensemble import RandomForestRegressor
y_train

regressor=RandomForestRegressor(n_estimators=10,verbose=3,n_jobs=-1)
regressor.fit(X_train,y_train)


y_pred=regressor.predict(X_test)
from sklearn.metrics import r2_score
r2_score(y_test,y_pred)




