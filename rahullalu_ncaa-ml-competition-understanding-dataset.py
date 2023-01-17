import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import os
print(os.listdir("../input"))

#CREATING 2 LISTS AND POPULATED WITH FILE NAMES
#FIRST LIST TO HOLD FILE NAMES AND LIST FOR HOLDING DATAFRAME OBJECTS
file_list=os.listdir("../input")
file_lists=os.listdir("../input")

#SORTED FILES NAMES SO FILES HOLDING SIMILAR DATA OCCUR ADJACENT TO EACH OTHER
file_list.sort()
# Any results you write to the current directory are saved as output.
%%time
#READING ALL THE FILES IN ONE GO
#USING EXCEPTION HANDLING TO IDENTIFY FILES GIVING ERROR WHILE READING 
for i in range(0,len(file_list)):
    try:
        file_lists[i]=pd.read_csv('../input/'+file_list[i])
    except:
        print(file_list[i])

print('Reading complete...')
#READING FAILED FILES
print('Reading failed file: Bracket_SampleTourney2018')
file_lists[0]=mpimg.imread('../input/'+file_list[0])
print('Reading failed file: TeamSpellings')
file_lists[51]=pd.read_csv('../input/TeamSpellings.csv',engine='python')
print('Reading complete....')
#CREATING A DATAFRAME WITH FILE NAMES
df_file_list=pd.DataFrame(file_list,columns=['File_names'])
df_file_list['File_names']=df_file_list.File_names.apply(lambda x: x.split('.')[0])
#FUNCTION FOR PROVIDING SUMMARY ON EACH FILE
def feature_summary(df_fa):
    col_list=['Non_Null','Null','Unique_Count','Data_type','Max','Min','Mean','Std','Sample_values']
    df=pd.DataFrame(index=df_fa.columns,columns=col_list)
    df['Non_Null']=list([len(df_fa[col][df_fa[col].notnull()]) for i,col in enumerate(df_fa.columns)])
    df['Null']=list([len(df_fa[col][df_fa[col].isnull()]) for i,col in enumerate(df_fa.columns)])
    df['Unique_Count']=list([len(df_fa[col].unique()) for i,col in enumerate(df_fa.columns)])
    df['Data_type']=list([df_fa[col].dtype for i,col in enumerate(df_fa.columns)])
    for i,col in enumerate(df_fa.columns):
        if 'float' in str(df_fa[col].dtype) or 'int' in str(df_fa[col].dtype):
            df.at[col,'Max']=df_fa[col].max()
            df.at[col,'Min']=df_fa[col].min()
            df.at[col,'Mean']=df_fa[col].mean()
            df.at[col,'Max']=df_fa[col].max()
            df.at[col,'Std']=df_fa[col].std()
        df.at[col,'Sample_values']=list(df_fa[col].unique())
    return(df)
#CODE FOR GENERATING FILE BY FILE SUMMARY
print('Total Files:',len(file_list))
for i in range(0,len(file_list)):
    try:
        print(i+1,'-',df_file_list.File_names[i],' Feature Summary')
        if i==0:
            plt.figure(figsize=(20,20))
            plt.imshow(file_lists[i])
            plt.show()
        else:
            display(feature_summary(file_lists[i]))
    except:
        print("Unable to read the file")
        continue
    