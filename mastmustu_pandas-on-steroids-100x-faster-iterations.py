import pandas as  pd
df= pd.DataFrame()

x= list(range(500000))

y= list(range(500000))



df['X'] = x

df['Y'] = y
# check the data 

df.head(5)
%%time 

sum_x = 0

sum_y = 0

for i in range(len(df)):

    sum_x = sum_x  + (df.iloc[i]['X'] +1)    

    sum_y = sum_y  + (df.iloc[i]['Y'] +2)

    

print(sum_x)

print(sum_y)
%%time 

sum_x = 0

sum_y = 0

for index, rows in df.iterrows():

    sum_x = sum_x  + (rows.X +1)    

    sum_y = sum_y  + (rows.Y +2)

        

print(sum_x)

print(sum_y)
%%time 

sum_x = 0

sum_y = 0

for rows in df.itertuples():

    sum_x = sum_x  + (rows.X +1)    

    sum_y = sum_y  + (rows.Y +2)

    

print(sum_x)

print(sum_y)