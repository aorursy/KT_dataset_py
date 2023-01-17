import pandas as pd
df_data = pd.read_csv('../input/Data.csv',encoding = "ISO-8859-1",low_memory=False)
#which tournaments are in the data set

df_data['Tournament'].value_counts()
#which locations are in the data set

df_data['Location'].value_counts()