#importing files

import numpy as np

import pandas as pd

%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import MinMaxScaler

from sklearn.decomposition import PCA

from sklearn.manifold import TSNE

from sklearn.model_selection import train_test_split

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

from sklearn import svm

from sklearn.cluster import KMeans

from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
#COMMENT OUT THESE FOR UR MACHINE RUN



#df = pd.read_csv("C:/Users/Sarthak Goel/Downloads/train.csv", sep=',')

#df2 = pd.read_csv("C:/Users/Sarthak Goel/Downloads/test.csv", sep=',')



#USE THE BELOW ONES FOR UR MACHINE



df = pd.read_csv("../input/data-mining-assignment-2/train.csv", sep=',')

df2 = pd.read_csv("../input/data-mining-assignment-2/test.csv", sep=',')



#df.info()

#df.head()

#df.tail()

#df.isnull().any()

#null_columns = df.columns[df.isnull().any()]

#print(null_columns)

#df.dropna(inplace=True)

#df.drop_duplicates()

#df2.duplicated().sum()

df2["Class"] = np.nan

df3 = df.append(df2)

df3=df3.set_index(["ID"])

#df3
#df3=df3.drop(["ID"],axis=1)

y=df3["Class"]

x=df3.drop(['Class'],axis=1)

x.head()
#x_test=df2.drop(["ID"],axis=1)

x_test = pd.get_dummies(x,columns=['col2','col11','col37','col44','col56'])

x_test.head()
corr_matrix= x_test.corr().abs()

upper= corr_matrix.where(np.triu(np.ones(corr_matrix.shape),k=1).astype(np.bool))

to_drop = [column for column in upper.columns if any(upper[column]>0.95)]

to_drop
x_test2 = pd.get_dummies(df3,columns=['col2','col11','col37','col44','col56'])

#x_test2.head()

corr=x_test2.corr()

abs(corr['Class']).sort_values()
x_test2=x_test2.drop(['col18',

 'col19',

 'col22',

 'col30',

 'col35',

 'col38',

 'col39',

 'col46',

 'col47',

 'col53',

 'col59',

 'col60'],axis=1)
x_test2=x_test2.drop(['col2_Diamond','col2_Platinum'    ,'col56_Low'        ,

'col56_High'      ,

'col13'          ,

'col37_Male'    ,'col37_Female' ,

'col56_Medium',

'col32',

'col44_No',

'col44_Yes',

'col2_Gold',

'col2_Silver',

'col11_Yes',

'col11_No',

'col26' ,

'col49',

'col0',

'col50',

'col63','col43',"Class"],axis=1)
x_test2
#df=df.drop(['Class'],axis=1)

scaler=StandardScaler()

scaled_data=scaler.fit(x_test2).transform(x_test2)

X_N=pd.DataFrame(scaled_data,columns=x_test2.columns)

X_N.head()
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_N[:700], y[:700], test_size=0.20,random_state=10)
np.random.seed(42)
from sklearn.ensemble import RandomForestClassifier



score_train_RF = []

score_test_RF = []



for i in range(1,16,1):

    rf = RandomForestClassifier( max_depth=i,random_state=7) 

    rf.fit(X_train, y_train)

    sc_train = rf.score(X_train,y_train)

    score_train_RF.append(sc_train)

    sc_test = rf.score(X_test,y_test)

    score_test_RF.append(sc_test)
plt.figure(figsize=(10,6))

#train_score,=plt.plot(range(1,16,1),score_train_RF,color='blue', linestyle='dashed', marker='o',

#         markerfacecolor='green', markersize=5)

test_score,=plt.plot(range(1,16,1),score_test_RF,color='red',linestyle='dashed',  marker='o',

         markerfacecolor='blue', markersize=5)

plt.legend( [train_score,test_score],["Train Score","Test Score"])

plt.title('Fig4. Score vs. No. of Trees')

plt.xlabel('No. of Trees')

plt.ylabel('Score')
score_test_RF = []

score_max = []

for i in range(1,15):

    for j in range(1,15):

        rf = RandomForestClassifier(max_depth=i,random_state=j) 

        rf.fit(X_train, y_train)

        #sc_train = rf.score(X_train,y_train)

        #score_train_RF.append(sc_train)

        sc_test = rf.score(X_test,y_test)

        score_test_RF.append(sc_test)

    print("depth", i)

    print("state",score_test_RF.index(max(score_test_RF)))

    print(max(score_test_RF))

    score_max.append(max(score_test_RF))

    score_test_RF = []
rf = RandomForestClassifier(max_depth =5,random_state=11)

rf.fit(X_train, y_train)

rf.score(X_test,y_test)
y_pred_RF = rf.predict(X_test)

confusion_matrix(y_test, y_pred_RF)
print(classification_report(y_test, y_pred_RF))
y2_pred_RF = rf.predict(X_N[700:])

y2_pred_RF
df_ans=pd.DataFrame(df2["ID"])

df_ans["Class"]=0

for i in range(300):

    df_ans["Class"].iloc[i]=y2_pred_RF[i]

df_ans.Class = df_ans.Class.astype(int)

df_ans
from IPython.display import HTML

import pandas as pd

import numpy as np

import base64

def create_download_link(df, title = "Download CSV file", filename = "DataRF12.csv"):

    csv = df.to_csv(index=False)

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'

    html = html.format(payload=payload,title=title,filename=filename)

    return HTML(html)

create_download_link(df_ans)