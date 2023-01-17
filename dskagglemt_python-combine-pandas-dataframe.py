import numpy as np 

import pandas as pd 
#df1 = pd.DataFrame({'A': ['A0', 'A1', 'A2', 'A3'], 

#                    'B': ['B0', 'B1', 'B2', 'B3'], 

#                    'C': ['C0', 'C1', 'C2', 'C3'], 

#                    'D': ['D0', 'D1', 'D2', 'D3']}, 

#                    index = [0, 1, 2, 3])

# We can provide the Index of our own as well.



df1 = pd.DataFrame(

                   {

                    'A': ['A0', 'A1', 'A2', 'A3'], 

                    'B': ['B0', 'B1', 'B2', 'B3']

                   }

                  ) 

  

# Creating second dataframe 

df2 = pd.DataFrame(

                    {

                    'A': ['A4', 'A5', 'A6', 'A7'], 

                    'B': ['B4', 'B5', 'B6', 'B7']

                    }

                  ) 

print(df1)

print("-"*10)

print(df2)

print("-"*10)

print(pd.concat([df1,df2], ignore_index = True)) # Note : ignore_index is set to True, This is to create new index. If set to false, will keep the original index.
print(pd.concat([df1,df2], ignore_index = False))
# Combining the data side-by-side

print(pd.concat([df1,df2], axis = 1))
# Dataframe created 

left = pd.DataFrame(

                    {

                    'Key': ['K0', 'K1', 'K2', 'K5'], 

                    'A': ['A0', 'A1', 'A2', 'A3'], 

                    'B': ['B0', 'B1', 'B2', 'B3']

                    }

                   ) 

  

right = pd.DataFrame(

                     {

                      'Key': ['K0', 'K4', 'K2', 'K3'], 

                      'C': ['C0', 'C1', 'C2', 'C3'], 

                      'D': ['D0', 'D1', 'D2', 'D3']

                     }

                    )  
# Option 1 : Keep just Intercation of both

print(pd.merge(left, right, how ='inner', on ='Key'))

print('*'*25)

print(pd.merge(left, right, how ='inner' )) # Here we have remved on argument.
# Option 2 : Keep full content of one, and the matching of another DF.

print(pd.merge(left, right, how ='left')) # Full content from DF "left", and just matching from DF "right"

print('*'*15)

print(pd.merge(right, left, how ='left' )) # Full content from DF "right", and just matching from DF "left"  
# Option 3 : Keep full content of both the DF.

print(pd.merge(left, right, how ='outer')) # Full content from DF "left", and from DF "right". Non-Matching will be set to NaN.

 
# result = left.join(right)