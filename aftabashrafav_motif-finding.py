import os

os.getcwd()

os.chdir("../input/")

os.listdir()

os.getcwd()
os.getcwd()
import pandas as pd




d1= pd.read_csv("tomtom (2).tsv", delimiter="\t")
d2=pd.read_csv("tomtom (3).tsv", delimiter="\t")
d3=pd.read_csv("tomtom (4).tsv", delimiter="\t")

d4=pd.read_csv("tomtom (5).tsv", delimiter="\t")

d5=pd.read_csv("tomtom (6).tsv", delimiter="\t")

d6=pd.read_csv("tomtom (7).tsv", delimiter="\t")

d2.Target_ID
d6
frame1=[d1,d2,d3]

frame2=[d4,d5,d6]

result1 = pd.concat(frame1)

result2= pd.concat(frame2)
df1=result1.dropna()

df1.shape

df1.tail()
result2= pd.concat(frame2)

df2=result2.dropna()

result2.shape

df1 = df1[df1['Target_ID'].str.contains("HUMAN")] 

df2=df2[df2['Target_ID'].str.contains("HUMAN")] 

dfcommon= df1.merge(df2, on="Target_ID", how = 'inner')
dfcommon.Target_ID.unique()