import pandas as pd
import seaborn as sns
df= sns.load_dataset('planets')
df.head()
df.head()
df.info()
df.isnull().sum()
df.notnull().sum()
df.isnull().sum().sum()
df[df.isnull().any(axis=1)]
df[df.notnull().all(axis=1)]
df.dropna(inplace=True)
#(inplace=True parametresi işlemin veri setinde kalıcı olmasını sağlar)
df.isnull().sum()
import numpy as np
import pandas as pd
V1 = np.array([1,3,6,np.NaN,7,1,np.NaN,9,15])
V2 = np.array([7,np.NaN,5,8,12,np.NaN,np.NaN,2,3])
V3 = np.array([np.NaN,12,5,6,14,7,np.NaN,2,31])
V4=np.nan
yeni_df = pd.DataFrame(
        {"V1" : V1,
         "V2" : V2,
         "V3" : V3,
         "V4":V4}        
)

yeni_df
yeni_df.dropna(how = "all")
yeni_df.dropna(axis = 1, how = "all")
!pip install missingno#kütüphaneyi yükleme
import missingno as msno
import seaborn as sns
planets= sns.load_dataset('planets')
planets.head()
msno.bar(planets);
msno.matrix(planets);
import numpy as np
import pandas as pd
V1 = np.array([1,3,6,np.NaN,7,1,np.NaN,9,15])
V2 = np.array([7,np.NaN,5,8,12,np.NaN,np.NaN,2,3])
V3 = np.array([np.NaN,12,5,6,14,7,np.NaN,2,31])

df = pd.DataFrame(
        {"V1" : V1,
         "V2" : V2,
         "V3" : V3}        
)

df
df["V1"].fillna(0,inplace=True)
df
df["V1"].fillna(df["V1"].mean(),inplace=True)#sadece V1 değişkeni için 
df
df.fillna(df.mean()[:])#tüm değişkenler için
df.fillna(df.median(),inplace=True)
df
import numpy as np
import pandas as pd
maas= np.array([1,3,6,np.NaN,7,1,np.NaN,9,15])
unvan= np.array(["MÜHENDİS",np.nan,"TEKNİKER","TEKNİKER","TEKNİKER","TEKNİKER","TEKNİKER","MÜHENDİS","MÜHENDİS"]
              , dtype=object)
df=pd.DataFrame(
{"maas":maas,
"unvan":unvan})
df
df.groupby("unvan")["maas"].mean()
df["maas"].fillna(df.groupby("unvan")["maas"].transform("mean"))