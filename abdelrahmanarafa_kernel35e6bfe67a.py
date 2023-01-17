import pandas as pd

import numpy as np
df_Batting = pd.read_csv("b.csv") 

# Preview the first 5 lines of the loaded data 

df_Batting.head()
df_Batting.info()
df_Batting['Wkts'].unique()

df_Batting.replace(to_replace='-',value=0,inplace=True)
df_Batting['Wkts']=pd.DataFrame(df_Batting['Wkts'].astype(str).astype(int))
df_Batting.info()
df_Batting['output1']=np.where(df_Batting['Wkts']<=1, 1, 0) 

df_Batting['output2']=np.where((df_Batting['Wkts']>=2) & (df_Batting['Wkts']<=3), 2, 0) 

df_Batting['output3']=np.where(df_Batting['Wkts']>=4 , 5, 0) 
df_Batting['output']=df_Batting.output1 | df_Batting.output2 | df_Batting.output3 
# Import Drive API and authenticate.

from google.colab import drive



# Mount your Drive to the Colab VM.

drive.mount('/gdrive')

#df

df=df_Batting

# Write the DataFrame to CSV file.

with open('/gdrive/My Drive/outputB.csv', 'w') as f:

  df.to_csv(f)