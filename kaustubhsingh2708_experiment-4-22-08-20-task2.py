#Task 2: Performing data manipulation on given dataset
import pandas as pd
file=r'../input/s-pivot/studentr.csv'
df=pd.read_csv(file)
print(df)
#Task 2: Perfoming melt
import pandas as pd
file=r'../input/s-pivot/studentr.csv'
df=pd.read_csv(file)
df1=pd.melt(df)
print(df1)
#Task 2: Perfoming pivot
import pandas as pd
file=r'../input/s-pivot/studentr.csv'
df=pd.read_csv(file)
df1=pd.pivot_table(df,index=['Exam','Subject'],aggfunc='mean')
print(df1)
#Task 2: Perfoming concat rowwise
import pandas as pd
file=r'../input/s-pivot/studentr.csv'
df=pd.read_csv(file)
df1=df.sample(4)
print('ROW 1\n',df1)
df2=df.sample(4)
print('ROW 2\n',df2)
pd.concat([df1,df2])
#Task 2: Perfoming concat Column wise
import pandas as pd
file=r'../input/s-pivot/studentr.csv'
df=pd.read_csv(file)
df1=df.sample(4)
print('ROW 1\n',df1)
df2=df.sample(4)
print('ROW 2\n',df2)
pd.concat([df1,df2],axis=1)