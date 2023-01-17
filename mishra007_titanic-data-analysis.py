import pandas as pd
df=pd.read_csv('../input/titanicdataset-traincsv/train.csv')
df.head()
df.shape
df.info()
df.isnull().sum()
import seaborn as sns
sns.heatmap(df.isnull())
sns.boxplot(x='Pclass',y='Age',data=df)

    
    
    
    