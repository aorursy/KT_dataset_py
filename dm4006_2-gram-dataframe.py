import pandas as pd
file="../input/wp_2gram.txt"
data=pd.read_csv(file,header=None,sep='\t')
data.head()
data[1]=data[1]+" "+data[2]
data.head() 
data.drop(2,axis=1)
data.to_csv("2_gram.csv")
