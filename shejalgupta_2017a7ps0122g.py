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
import os





df =pd.read_csv("/kaggle/input/dmassign1/data.csv")

data = pd.read_csv("/kaggle/input/dmassign1/data.csv")
df
df = df.filter(["Col42","Col43","Col45",'Col69',"Col70","Col71","Col84","Col85","Col86",'Col150',"Col98","Col99"],axis=1)
df = df.replace(to_replace='?',value=np.nan)

data = data.replace(to_replace='?',value=np.nan)
for column in df.columns:

    df[column].fillna(df[column].mode()[0], inplace=True)





    
for col in df.columns:

    for x in range(len(df[col])):

        if type(df[col][x]) == str:

            df[col].replace({df[col][x]: float(df[col][x])},inplace=True)
from sklearn.preprocessing import StandardScaler

scaler=StandardScaler()

scaled_data=scaler.fit(df).transform(df)

scaled_df=pd.DataFrame(scaled_data,columns=df.columns)

scaled_df.tail()
df=scaled_df
#model=TSNE(learning_rate=1000)

from sklearn.manifold import TSNE



model=TSNE(n_iter=500,n_components=2,perplexity=50)

model_data=model.fit_transform(df)

model_data.shape
df = model_data
from sklearn.cluster import KMeans

plt.figure(figsize=(30, 8))

kmean = KMeans(n_clusters = 30, random_state = 50)

kmean.fit(df)

pred = kmean.predict(df)

pred_pd = pd.DataFrame(pred)

arr = pred_pd[0].unique()







pred
len(pred)
dict0={1:0,2:0,3:0,4:0,5:0}

dict1={1:0,2:0,3:0,4:0,5:0}

dict2={1:0,2:0,3:0,4:0,5:0}

dict3={1:0,2:0,3:0,4:0,5:0}

dict4={1:0,2:0,3:0,4:0,5:0}

dict5={1:0,2:0,3:0,4:0,5:0}

dict6={1:0,2:0,3:0,4:0,5:0}

dict7={1:0,2:0,3:0,4:0,5:0}

dict8={1:0,2:0,3:0,4:0,5:0}

dict9={1:0,2:0,3:0,4:0,5:0}

dict10={1:0,2:0,3:0,4:0,5:0}

dict11={1:0,2:0,3:0,4:0,5:0}

dict12={1:0,2:0,3:0,4:0,5:0}

dict13={1:0,2:0,3:0,4:0,5:0}

dict14={1:0,2:0,3:0,4:0,5:0}

dict15={1:0,2:0,3:0,4:0,5:0}

dict16={1:0,2:0,3:0,4:0,5:0}

dict17={1:0,2:0,3:0,4:0,5:0}

dict18={1:0,2:0,3:0,4:0,5:0}

dict19={1:0,2:0,3:0,4:0,5:0}

dict20={1:0,2:0,3:0,4:0,5:0}

dict21={1:0,2:0,3:0,4:0,5:0}

dict22={1:0,2:0,3:0,4:0,5:0}

dict23={1:0,2:0,3:0,4:0,5:0}

dict24={1:0,2:0,3:0,4:0,5:0}

dict25={1:0,2:0,3:0,4:0,5:0}

dict26={1:0,2:0,3:0,4:0,5:0}

dict27={1:0,2:0,3:0,4:0,5:0}

dict28={1:0,2:0,3:0,4:0,5:0}

dict29={1:0,2:0,3:0,4:0,5:0}







for i in range(1300):

    if pred[i] == 0:

           dict0[data["Class"][i]] += 1

    if pred[i] == 1:

           dict1[data["Class"][i]] += 1

    if pred[i] == 2:

           dict2[data["Class"][i]] += 1

    if pred[i] == 3:

           dict3[data["Class"][i]] += 1

    if pred[i] == 4:

           dict4[data["Class"][i]] += 1

    if pred[i] == 5:

           dict5[data["Class"][i]] += 1

    if pred[i] == 6:

           dict6[data["Class"][i]] += 1

    if pred[i] == 7:

           dict7[data["Class"][i]] += 1

    if pred[i] == 8:

           dict8[data["Class"][i]] += 1

    if pred[i] == 9:

           dict9[data["Class"][i]] += 1

    if pred[i] == 10:

           dict10[data["Class"][i]] += 1

    if pred[i] == 11:

           dict11[data["Class"][i]] += 1

    if pred[i] == 13:

           dict13[data["Class"][i]] += 1

    if pred[i] == 14:

           dict14[data["Class"][i]] += 1

    if pred[i] == 15:

           dict15[data["Class"][i]] += 1

    if pred[i] == 12:

           dict12[data["Class"][i]] += 1

    if pred[i] == 16:

           dict16[data["Class"][i]] += 1

    if pred[i] == 17:

           dict17[data["Class"][i]] += 1

    if pred[i] == 18:

           dict18[data["Class"][i]] += 1

    if pred[i] == 19:

           dict19[data["Class"][i]] += 1

    if pred[i] == 20:

           dict20[data["Class"][i]] += 1

    if pred[i] == 21:

           dict21[data["Class"][i]] += 1

    if pred[i] == 22:

           dict22[data["Class"][i]] += 1

    if pred[i] == 23:

           dict23[data["Class"][i]] += 1

    if pred[i] == 24:

           dict24[data["Class"][i]] += 1

    if pred[i] == 25:

           dict25[data["Class"][i]] += 1

    if pred[i] == 26:

           dict26[data["Class"][i]] += 1

    if pred[i] == 27:

           dict27[data["Class"][i]] += 1

    if pred[i] == 28:

           dict28[data["Class"][i]] += 1

    if pred[i] == 29:

           dict29[data["Class"][i]] += 1

   





            

        

    
print(0,dict0)

print(1,dict1)

print(2,dict2)

print(3,dict3)

print(4,dict4)

print(5,dict5)

print(6,dict6)

print(7,dict7)

print(8,dict8)

print(9,dict9)

print(10,dict10)

print(11,dict11)

print(12,dict12)

print(13,dict13)

print(14,dict14)

print(15,dict15)

print(16,dict16)

print(17,dict17)

print(18,dict18)

print(19,dict19)

print(20,dict20)

print(21,dict21)

print(22,dict22)

print(23,dict23)

print(24,dict24)

print(25,dict25)

print(26,dict26)

print(27,dict27)

print(28,dict28)

print(29,dict29)



res=[]


for i in range(len(pred)):

    if pred[i] == 0:

        res.append(1)

    if pred[i] == 1:

        res.append(3)

    if pred[i] == 2:

        res.append(5)

    if pred[i] == 3:

        res.append(2)

    if pred[i] == 4:

        res.append(1);

    if pred[i] == 5:

        res.append(1)

    if pred[i] == 6:

        res.append(3)

    if pred[i] == 7:

        res.append(3)

    if pred[i] == 8:

        res.append(1)

    if pred[i] == 9:

        res.append(4)

    if pred[i] == 10:

        res.append(4)

    if pred[i] == 11:

        res.append(4)

    if pred[i] == 12:

        res.append(1)

    if pred[i] == 13:

        res.append(4)

    if pred[i] == 14:

        res.append(5)

    if pred[i] == 15:

        res.append(3)

    if pred[i] == 16:

        res.append(5)

    if pred[i] == 17:

        res.append(3)

    if pred[i] == 18:

        res.append(1)

    if pred[i] == 19:

        res.append(5)

    if pred[i] == 20:

        res.append(5)

    if pred[i] == 21:

        res.append(5)

    if pred[i] == 22:

        res.append(1)

    if pred[i] == 23:

        res.append(5)

    if pred[i] == 24:

        res.append(4)

    if pred[i] == 25:

        res.append(2)

    if pred[i] == 26:

        res.append(1)

    if pred[i] == 27:

        res.append(4)

    if pred[i] == 28:

        res.append(4)

    if pred[i] == 29:

        res.append(5)

    

        

   

len(res)
ans = res
ans1 = pd.DataFrame(ans)
final = pd.concat([data["ID"], ans1], axis=1).reindex()

final = final.rename(columns={0: "Class"})

final=final[1300:]

final.head()
final.to_csv('submissionSF3T.csv', index = False)
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

create_download_link(final)