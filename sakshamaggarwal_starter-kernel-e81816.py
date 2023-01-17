import pandas as pd

import numpy as np
data0=pd.read_csv("../input/submissions-csv/submission0.csv")

data1=pd.read_csv("../input/submissions-csv/submission1.csv")

data2=pd.read_csv("../input/submissions-csv/submission2.csv")

data3=pd.read_csv("../input/submissions-csv/submission3.csv")

data4=pd.read_csv("../input/submissions-csv/submission4.csv")

data5=pd.read_csv("../input/submissions-csv/submission5.csv")

data6=pd.read_csv("../input/submissions-csv/submission6.csv")

data7=pd.read_csv("../input/submissions-csv/submission7.csv")

data8=pd.read_csv("../input/submissions-csv/submission8.csv")

data9=pd.read_csv("../input/submissions-csv/submission9.csv")

data10=pd.read_csv("../input/submissions-csv/submission10.csv")

data11=pd.read_csv("../input/submissions-csv/submission11.csv")

data12=pd.read_csv("../input/submissions-csv/submission12.csv")

data13=pd.read_csv("../input/submissions-csv/submission13.csv")

data14=pd.read_csv("../input/submissions-csv/submission14.csv")

data15=pd.read_csv("../input/submissions-csv/submission15.csv")

data16=pd.read_csv("../input/submissions-csv/submission16.csv")

from collections import Counter
a={}

for i in range(0,29996):

  data=Counter([data4.genres[i],data0.genres[i],data7.genres[i],data11.genres[i],data16.genres[i]])

  a[i]=data.most_common(1)[0][0]
list = [(v) for k, v in a.items()] 

list
dataarray=np.asarray(list)

dataarray=dataarray.reshape(-1,1)

id=np.asarray(data0.id)

id=id.reshape(-1,1)
g=np.concatenate([id,dataarray],axis=1)
df=pd.DataFrame(g,columns=['id','genres'])
df.to_csv('final-submissions2.csv',index = False)