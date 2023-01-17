# from google.colab import drive



# drive.mount('/content/drive') 
# cd drive/My Drive/DM_Assignment_2

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
df= pd.read_csv("../input/train.csv", sep=',')
df = df.replace({'?': np.nan})
df.fillna(df.mode().iloc[0], inplace=True)
y=df['Class']
important_cols=['ID','Age','IC','OC','Timely Income', 'Gain', 'Loss', 'Stock','Weight', 'NOP', 'Own/Self', 'Vet_Benefits', 'Weaks', 'WorkingPeriod','Worker Class','Schooling','Married_Life','MIC','MOC','Cast','Hispanic','Sex','Full/Part','Tax Status','Detailed','Summary','COB FATHER','COB MOTHER','Citizen']

data1=df[important_cols]
mylist = list(data1.select_dtypes(include=['object']).columns)

mylist
data1 = pd.get_dummies(data1, columns=mylist)
imp_col=[ 'Age', 'IC', 'OC', 'Timely Income', 'Gain', 'Loss', 'Stock',

       'Weight', 'NOP', 'Own/Self', 'Vet_Benefits', 'Weaks', 'WorkingPeriod',

       'Worker Class_WC1', 'Worker Class_WC2', 'Worker Class_WC3',

       'Worker Class_WC4', 'Worker Class_WC5', 'Worker Class_WC6',

       'Schooling_Edu1', 'Schooling_Edu11', 'Schooling_Edu12',

       'Schooling_Edu15', 'Schooling_Edu2', 'Schooling_Edu5', 'Schooling_Edu6',

       'Schooling_Edu8', 'Married_Life_MS2', 'Married_Life_MS3',

       'Married_Life_MS4', 'MIC_MIC_A', 'MIC_MIC_C', 'MIC_MIC_D', 'MIC_MIC_E',

       'MIC_MIC_F', 'MIC_MIC_H', 'MIC_MIC_I', 'MIC_MIC_K', 'MIC_MIC_L',

       'MIC_MIC_M', 'MIC_MIC_N', 'MIC_MIC_S', 'MIC_MIC_T', 'MOC_MOC_A',

       'MOC_MOC_B', 'MOC_MOC_C', 'MOC_MOC_E', 'MOC_MOC_H', 'Cast_TypeA',

       'Cast_TypeD', 'Hispanic_HA', 'Sex_F', 'Sex_M', 'Full/Part_FB',

       'Full/Part_FC', 'Full/Part_FF', 'Tax Status_J1', 'Tax Status_J3',

       'Tax Status_J4', 'Detailed_D2', 'Detailed_D5', 'Detailed_D8',

       'Summary_sum2', 'Summary_sum5', 'COB FATHER_c24', 'COB MOTHER_c24',

       'Citizen_Case1']
data2=data1[imp_col]
data3=data2
import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA

from scipy.spatial.distance import cdist

from sklearn.cluster import KMeans

from collections import Counter

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_validate

from sklearn.metrics import make_scorer

from sklearn.metrics import f1_score

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.ensemble import RandomForestClassifier

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import Imputer

from sklearn.model_selection import cross_val_score

from sklearn.ensemble import AdaBoostClassifier
scalar=StandardScaler()

scaled_data = scalar.fit_transform(data3)

scaled_df=pd.DataFrame(scaled_data,columns=data3.columns)

scaled_df.head()
pca = PCA().fit(scaled_df)

plt.figure()

plt.plot(np.cumsum(pca.explained_variance_ratio_))

plt.xlabel('Number of Components')

plt.ylabel('Variance (%)') #for each component

plt.title('Dataset Explained Variance')

plt.show()

pca = PCA(n_components=60)

pca.fit(scaled_df)

T1 = pca.transform(scaled_df)

pca.explained_variance_ratio_.sum()
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(T1,y,random_state=10,test_size=0.2)
rf=RandomForestClassifier(n_estimators=150,bootstrap=False,class_weight='balanced',max_leaf_nodes=256,criterion='entropy',min_samples_leaf=37,max_depth=25,min_samples_split=4,n_jobs=-1)
rf.fit(X_train ,y_train)
from sklearn.metrics import roc_auc_score

y_pred=rf.predict(X_test)

print(roc_auc_score(y_test,y_pred))
from sklearn.metrics import accuracy_score

accuracy_score(y_test, y_pred)
data22 = pd.read_csv('../input/test.csv')

df2= data22.replace({'?': np.nan})
df2.fillna(df.mode().iloc[0], inplace=True)
data23=df2[important_cols]
data23 = pd.get_dummies(data23, columns=mylist)
data23=data23[imp_col]
scaled_data1 = scalar.transform(data23)

scaled_df1=pd.DataFrame(scaled_data1,columns=data23.columns)

scaled_df1.head()
x_test=pca.transform(scaled_df1)
val = rf.predict(x_test)
dict = {"ID" : df2["ID"], "Class" : val}

df3 = pd.DataFrame(dict, columns = ["ID","Class"])

# df3.to_csv("submission.csv", index = False)
from IPython.display import HTML 

import pandas as pd 

import numpy as np 

import base64 

def create_download_link(df, title = "Download CSV file", filename = "data.csv"):

    csv = df.to_csv(index=False)     

    b64 = base64.b64encode(csv.encode())     

    payload = b64.decode()     

    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'    

    html = html.format(payload=payload,title=title,filename=filename)     

    return HTML(html) 

create_download_link(df3)