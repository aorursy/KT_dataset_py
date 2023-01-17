import pandas as pd
import datetime
#gets Deltas
def Marty(string,output,Name_separator):
    df= Name_separator[string]
    df['delta'] = (df['Data']-df['Data'].shift()).fillna(pd.Timedelta(seconds=0))
    df.drop(df.iloc[::2].index, inplace=True)
    df['delta'].sum()
    
    output.at[string,'Delta'] = df['delta'].sum()
    
    return output.fillna(pd.Timedelta(seconds=0))
#sets up datafreames
def plutonium(Day,Dataframe,Day_separator):
    Name_separator = dict(tuple(Day_separator[Day].groupby(Dataframe['Nome'])))
    
    output= Dataframe['Nome'].drop_duplicates()
    output = output.reset_index()
    output.drop(columns='index',inplace=True)
    output['Delta'] = pd.Timedelta('nat')
    output.set_index('Nome', inplace=True)
    
    docs_phone_book = []
    docs_phone_book.extend(output.index.tolist())
    
    for Nome in docs_phone_book:
        
        if Nome in Name_separator.keys():
            Marty(Nome,output,Name_separator)
            
    return output
#gets date input
def get_input(N_dias):
    print('forneça %d datas no formato YYYY-MM-DD separadas por espaços...' % (N_dias))
    
    datas = str(input())
    
    date_list = [str(x) for x in datas.split()]
    
    return date_list
#gets date data
def get_days_data(df, Day_separator, N_dias):
    days_data=[]
    days= get_input(N_dias) 
    
    for i in range(N_dias):
        day = days[i]
        day_data = plutonium(day,df,Day_separator).fillna(pd.Timedelta(seconds=0))
        days_data.append(day_data)
        
    final_data = pd.concat(days_data, axis=1, keys=days)
    final_data = pd.concat([final_data,final_data.sum(axis=1)],axis=1)
    
    return final_data
#reads database and cleans up the data
df= pd.read_excel('TimeMachine.xlsx',sheet_name='Log')

df.drop('ID',axis=1,inplace=True)
df = df[df.Nome != 'Daniel S']
df['Data'] = df['hora'].apply(lambda hora: hora.strftime("%D  %H:%M"))
df['Data'] = pd.to_datetime(df['Data'])
df.drop('hora',axis=1,inplace=True)
df.drop_duplicates(inplace=True)
#stores data in a dict
Day_separator= dict(tuple(df.groupby(df['Data'].apply(lambda Data: Data.strftime('%Y-%m-%d')))))
#gets N of days the user wants
tracker = 0
while tracker ==0:
    N_dias = int(input('Digite o Número de dias que você deseja acessar: '))
    if N_dias in range(100):
        tracker = 1
tracker = 0
#gets data
final_data = get_days_data(df,Day_separator,N_dias)
#test
final_data
#export data
final_data.to_excel('Fire_Trail.xlsx',sheet_name='1985') 
