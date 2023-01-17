import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df=pd.read_csv('../input/crime.csv')
df.head()
df.describe()
df.isnull().sum()
df['Text_General_Code'].dropna(inplace=True)
list=['Dispatch_Date','Dispatch_Time']
df.sort_values(by=list,inplace=True)
df['Night']= df['Hour']<7
df['Night2']=df['Hour']>19
df['Night']=df['Night']|df['Night2']
df.drop('Night2',1,inplace=True)
df['Night']=df['Night'].astype(int)

sns.countplot(x=df.Hour)
plt.show()
sns.countplot(x=df.Night)
plt.show()
df['Freq']=1
list=['Month','Text_General_Code']
new_df=df.groupby(list)['Freq'].sum().reset_index()

list=new_df.Text_General_Code.unique()
for i in list:
    plt.plot_date(new_df['Month'][new_df.Text_General_Code==i],new_df['Freq'][new_df.Text_General_Code==i],'k')
    plt.title(i)
    plt.show()
df.Dispatch_Date_Time=pd.to_datetime(df.Dispatch_Date_Time)
new=df.groupby(df['Dispatch_Date_Time'].dt.year)['Freq'].sum().reset_index()
plt.plot(new['Dispatch_Date_Time'],new['Freq'],'k')
plt.title('Year vs #Crime')
plt.show()