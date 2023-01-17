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
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

test_df=pd.read_csv("/kaggle/input/consumer/Edureka_Consumer_Complaints_test.csv")
train_df=pd.read_csv("/kaggle/input/consumer/Edureka_Consumer_Complaints_train.csv")
text_test_df=test_df[["Consumer complaint narrative","Product"]]
text_train_df=train_df[["Consumer complaint narrative","Product"]]
##Checking for Test
print("------Test------\n")
print("Shape:",text_test_df.shape)
print("Null values per column:")
print(text_test_df.isnull().sum())
##Checking for Train
print("------Train-------\n")
print("Shape:",text_train_df.shape)
print("Null values per column:")
print(text_train_df.isnull().sum())
new_test_df=text_test_df[pd.notnull(text_test_df["Consumer complaint narrative"])].reset_index().drop("index",axis=1)
new_train_df=text_train_df[pd.notnull(text_train_df["Consumer complaint narrative"])].reset_index().drop("index",axis=1)
print("Test:",new_test_df.isnull().sum().sum())
print("Train:",new_train_df.isnull().sum().sum())
new_train_df.head()["Consumer complaint narrative"]
new_test_df.head()["Consumer complaint narrative"]
new_train_df["Consumer complaint narrative"]=new_train_df["Consumer complaint narrative"].apply(lambda x: str(x).lower())
new_test_df["Consumer complaint narrative"]=new_test_df["Consumer complaint narrative"].apply(lambda x: str(x).lower())
print("------Train-------\n")
print(new_train_df.head(3)["Consumer complaint narrative"])
print("------Test------\n")
print(new_test_df.head(3)["Consumer complaint narrative"])
import re
new_train_df["Consumer complaint narrative"]=new_train_df["Consumer complaint narrative"].apply(lambda x: re.sub(r"(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?","",x)) 
new_test_df["Consumer complaint narrative"]=new_test_df["Consumer complaint narrative"].apply(lambda x: re.sub(r"(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?","",x))
print("------Train-------\n")
print(new_train_df.iloc[1]["Consumer complaint narrative"])
print("------Test------\n")
print(new_test_df.iloc[1]["Consumer complaint narrative"])
new_train_df["Consumer complaint narrative"]=new_train_df["Consumer complaint narrative"].apply(lambda l: re.sub(r"xx*","",l)) 
new_test_df["Consumer complaint narrative"]=new_test_df["Consumer complaint narrative"].apply(lambda l: re.sub(r"xx*","",l))
print("------Train-------\n")
print(new_train_df.iloc[1]["Consumer complaint narrative"])
print("------Test------\n")
print(new_test_df.iloc[1]["Consumer complaint narrative"])
new_train_df["Consumer complaint narrative"]=new_train_df["Consumer complaint narrative"].apply(lambda l: re.sub(r"\s+"," ",l)) 
new_test_df["Consumer complaint narrative"]=new_test_df["Consumer complaint narrative"].apply(lambda l: re.sub(r"\s+"," ",l))
print("------Train-------\n")
print(new_train_df.iloc[1]["Consumer complaint narrative"])
print("------Test------\n")
print(new_test_df.iloc[1]["Consumer complaint narrative"])
print("------Train-------\n")
print(new_train_df.head())
print("------Test------\n")
print(new_test_df.head())
print("------Train-------\n")
print("No of Categories in Product :",new_train_df["Product"].nunique())
print("------Test------\n")
print("No of Categories in Product :",new_test_df["Product"].nunique())
train_product=new_train_df["Product"].unique().tolist()
test_product=new_test_df["Product"].unique().tolist()

not_available_in_test=[]
for prod in train_product:
    if prod in test_product:
        pass
    else:
        not_available_in_test.append(prod)

not_available_in_train=[]
for prod in test_product:
    if prod in train_product:
        pass
    else:
        not_available_in_train.append(prod)
        
if(len(not_available_in_test)==0 and len(not_available_in_train)==0):
    print("All categories of the test dataset is there in the train dataset")
elif(len(not_available_in_test)>0 and len(not_available_in_train)==0):
    print("All test categories are present in train but there are few categories of train unavailable in test ")
    print(not_available_in_test)
elif(len(not_available_in_test)==0 and len(not_available_in_train)>0):
    print("All train categories are present in test but there are few categories of test unavailable in train ")
    print(not_available_in_train)
elif(len(not_available_in_test)>0 and len(not_available_in_train)>0):
    print("There are few categories unavailable in both")
    print("Extra categories of test",not_available_in_train)
    print("Extra categories of train",not_available_in_test)
    
new_train_df["Product"].value_counts()
x_train=new_train_df["Consumer complaint narrative"]
y_train=new_train_df["Product"]
x_test=new_test_df["Consumer complaint narrative"]
y_test=new_test_df["Product"]
from nltk.tokenize import word_tokenize

def tokenizer_word(text):
    tokens=word_tokenize(text)
    return(tokens)

from sklearn.feature_extraction.text import TfidfVectorizer
tf=TfidfVectorizer(analyzer="word",tokenizer=tokenizer_word,lowercase=True,stop_words={'english'})
x_train_tfidf=tf.fit_transform(x_train)
x_test_tfidf=tf.transform(x_test)
print("train",x_train_tfidf.shape)
print("test",x_test_tfidf.shape)
##Findout which model performs better
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

models=[LogisticRegression(max_iter=2000),RandomForestClassifier(),MultinomialNB(),LinearSVC(max_iter=3000),XGBClassifier()]
for i in models:
    obj=i
    obj.fit(x_train_tfidf,y_train)
    pred=obj.predict(x_test_tfidf)
    result=accuracy_score(y_test,pred)
    print("Accuracy of",i,"is : ",result)
##Implementing LinearSVC
svc=LinearSVC(C=1,penalty="l2",max_iter=3000)
svc.fit(x_train_tfidf,y_train)
prediction=svc.predict(x_test_tfidf)
print("Accuracy of LinearSVC model is",np.round(accuracy_score(y_test,prediction),3))

from sklearn.metrics import classification_report
print(classification_report(y_test,prediction,target_names=new_test_df["Product"].unique()))
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

cm=confusion_matrix(y_test,prediction)
fig, ax = plt.subplots(figsize=(10,9))
sns.heatmap(cm,annot=True, cmap="Reds",fmt='d',xticklabels=new_test_df["Product"].unique(),yticklabels=new_test_df["Product"].unique())
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title("CONFUSION MATRIX - LinearSVC\n", size=10);
#Storing LinearSVC model's prediction to a file
res_dec=pd.Series(prediction)
res_dec.to_csv("/kaggle/working/Consumer_Complaints_prediction.csv")
