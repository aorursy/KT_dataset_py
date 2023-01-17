from IPython.display import YouTubeVideo
YouTubeVideo(id="hdtQPawYm2k",width=1000,height=500)
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
import joblib 
import operator
import seaborn as sns
import plotly.express as px 
from plotly.subplots import make_subplots 
import plotly.graph_objects as go  
from matplotlib import pyplot as plt
import warnings 
warnings.filterwarnings("ignore")
from sklearn.metrics import precision_score 
from sklearn.metrics import recall_score 
from sklearn.metrics import f1_score 
from sklearn.metrics import roc_auc_score
# For metrics Calculation 
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import mean_squared_error 
from sklearn.metrics import classification_report  
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve 
from sklearn.metrics import auc
from sklearn.model_selection import train_test_split #splitting train and testing data 
from sklearn.model_selection import cross_val_score 
from sklearn.linear_model import LogisticRegression # to build a model that will be able to classify the response is 1 or 0 based on input attributes
from sklearn.model_selection import  StratifiedKFold,KFold # for splitting the data into five parts  
from sklearn.preprocessing import LabelEncoder # for labeling some of the categorical Variables i.e 0,1,2,3.....
df=pd.read_csv("../input/health-insurance-cross-sell-prediction/train.csv") 
df1=pd.read_csv("../input/health-insurance-cross-sell-prediction/test.csv")
print("Shape of training dataset") 
print("=================================================")
print("Number of rows :"+str(df.shape[0])) 
print("Number of columns : "+str(df.shape[1]))
print("shape of testing data set")  
print("=======================================")
print("Number of rows :"+str(df1.shape[0])) 
print("Number of columns : "+str(df1.shape[1]))
Gender=df["Gender"].value_counts().to_frame().reset_index().rename(columns={'index':'Gender',"Gender":'Count'})
fig=go.Figure(go.Bar(x=Gender["Gender"],y=Gender["Count"],text=Gender["Count"],textposition="outside")) 
fig.update_layout(title="Gender") 
fig.show()
fig=px.pie(df,names="Gender",title="Gender",hole=0.4,color_discrete_map={'Male':'royalblue','Female':'blue'}) 
fig.show()
fig=px.histogram(df,x="Age",nbins=50,title="Distribution of Ages") 
fig.show()
fig=px.pie(df,names="Driving_License",title="Driving_License",hole=0.3) 
fig.show()
fig=px.pie(df,names="Previously_Insured",title="Previously_insured",hole=0.8) 
fig.show()
fig=px.pie(df,names="Vehicle_Damage",title="Vehicle Damage",hole=0.3,color_discrete_sequence=["springgreen","aqua"])
fig.show()
fig=px.pie(df,names="Vehicle_Age",title="Vehicle_Age",hole=0.7,color_discrete_sequence=["royalblue","blue","green"]) 
fig.show()
fig=px.pie(df,names="Response",title="Response",hole=0.5,color_discrete_sequence=["skyblue","yellow"]) 
fig.show()
plt.figure(figsize=(20,10))
sns.violinplot(df["Annual_Premium"])
fig=px.histogram(df,"Vintage",color="Response") 
fig.show()
Driving=df.groupby(["Gender","Driving_License"])["Driving_License"].count().unstack("Driving_License") 
Driving
#plotting the graph betweeen gender and driving_license i.e having male and female having license or not 
Driving.plot(kind="barh",stacked=True,figsize=(10,5),colormap="Spectral") 
plt.title("Difference having Gender whose having Driving License or not",color="blue",fontsize=10,loc="center")
Vehicle_Age=df.groupby(["Gender","Vehicle_Age"])["Vehicle_Age"].count().unstack("Vehicle_Age")
Vehicle_Age
#Here We can see the bar chart between gender with respective Vehicle Age 
Vehicle_Age.plot(kind="barh",stacked=True,figsize=(10,5),fontsize=10,colormap="rainbow") 
plt.title("Gender comparision With Vehicle Age",color="green")
Vehicle_damage=df.groupby(["Gender","Vehicle_Damage"])["Vehicle_Damage"].count().unstack("Vehicle_Damage")
Vehicle_damage.plot(kind="barh",stacked=True,figsize=(10,5),colormap="cool") 
plt.title("Gender comparision with Vehicle_damage i.e caused by accident or not ")
Response=df.groupby(["Gender","Response"])["Response"].count().unstack("Response")
Response.plot(kind="barh",stacked=True,figsize=(10,5),colormap="Set1")
Previously_insured=df.groupby(["Gender","Previously_Insured"])["Previously_Insured"].count().unstack("Previously_Insured") 
Previously_insured
Previously_insured.plot(kind="barh",stacked=True,figsize=(10,5),colormap="Oranges_r")
plt.figure(figsize=(12,8)) 
sns.heatmap(df.drop("id",axis=1).corr(),annot=True,cmap="cool")
#converting Categorical columns into numerical columns by Label Encoder in training dataset
le=LabelEncoder() 
df["Gender"]=le.fit_transform(df["Gender"]) 
df["Vehicle_Damage"]=le.fit_transform(df["Vehicle_Damage"]) 
df["Vehicle_Age"]=le.fit_transform(df["Vehicle_Age"])
#converting Categorical columns into numerical columns by lable encoder in Testing dataset
le=LabelEncoder() 
df1["Gender"]=le.fit_transform(df1["Gender"]) 
df1["Vehicle_Damage"]=le.fit_transform(df1["Vehicle_Damage"]) 
df1["Vehicle_Age"]=le.fit_transform(df1["Vehicle_Age"])
X=df.drop("Response",axis=1) 
y=df["Response"]
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.30,random_state=12)
lr=LogisticRegression()
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
accuracy=accuracy_score(y_pred,y_test) 
print("accuracy Score : "+str(accuracy))
from sklearn.model_selection import cross_val_score 
score=cross_val_score(lr,X,y,cv=10) 
score
print(f'Accuracy Score : {score.mean()}')
from sklearn.model_selection import KFold
kfold=KFold(n_splits=5) 
results=cross_val_score(lr,X,y,cv=kfold)  
results
print("Accuracy Score : "+str(results.mean()*100))
y_pred=lr.predict(x_test)
sns.heatmap(confusion_matrix(y_pred,y_test),annot=True,cmap="cool") 
plt.ylabel("Actual Values") 
plt.xlabel("Predicted Values")
print("Precision Score : "+str(precision_score(y_pred,y_test))) 
print("Recall Score : "+str(recall_score(y_pred,y_test))) 
print("F1_ Score : "+str(f1_score(y_pred,y_test)))
print("Classification metrics :") 
print(classification_report(y_pred,y_test))
testing=lr.predict(df1) 
testing.shape
roc_auc_score_list=[]
for i in range(10):
    x_train,x_test,y_train,y_test = train_test_split(df.drop('Response', axis=1), 
                                              df['Response'], test_size=.3)
    roc_auc_score_list.append(roc_auc_score(y_test, lr.predict_proba(x_test)[:, 1]))
    fpr, tpr, _ = roc_curve(y_test, lr.predict_proba(x_test)[:, 1])
    plt.plot(fpr, tpr)
print(f'Mean roc_auc_score: {np.mean(roc_auc_score_list)}')
print("submission Task") 
print("="*50)
test_submission = pd.DataFrame() 
test_submission["testid"]=x_test["id"] 
test_submission["Response"]=y_pred 
test_submission.reset_index(inplace=True) 
test_submission.drop("index",axis=1,inplace=True)
test_submission.rename(columns={"testid":"id"})  
test_submission.head()
