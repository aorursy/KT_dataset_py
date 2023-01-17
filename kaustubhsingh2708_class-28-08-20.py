import pandas as pd
file=r'../input/s-pivot/studentr.csv'
df=pd.read_csv(file)
df[['Name','Subject']]


df.loc[[0,1,7]]#particular location
df.loc[1:5:2]#alternate
df.loc[1:5:2,['Name','Subject']]#particular row and column
print('\n\nAll Rows\n\n',df.loc[:,['Name','Subject']])#all rows particular cols
print('\n\nAll Column\n\n',df.loc[1:5,:])#all col particular rows