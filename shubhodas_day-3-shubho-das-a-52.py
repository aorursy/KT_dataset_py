import pandas as pd
df = pd.DataFrame( {'A':[1,2,3,4,5] , 'B':[6,7,8,9,10], 'C':[11,12,13,14,15]    } );
print(df)
import pandas as pd

data = {'Name' : ['AB','CD','EF','GH','IJ','KL','MN','OP','QR','ST'] , 'Marks':[20,40,60,30,50,70,50,20,37,60]}
labels = ['a','b','c','d','e','f','g','h','i','j']

df = pd.DataFrame(data , index=labels)
print(df)
import pandas as pd

df = pd.read_csv("../input/titanic/train_and_test2.csv")
df.head(3)
import pandas as pd

df = pd.read_csv("../input/titanic/train_and_test2.csv")
print("The selected specific data is:")
print(df[['Fare','Age','Sex']])
import pandas as pd
import numpy as np

data = {'Names':['AB','CD','EF','GH','IJ','KL','MN','OP','QR','ST'],'Marks':[20,40,np.nan,30,np.nan,70,np.nan,20,37,60]}
labels = ['a','b','c','d','e','f','g','h','i','j']

df = pd.DataFrame(data , index = labels)
print(df[df['Marks'].isnull()])
