import pandas as pd

import numpy as np
pd.__version__
np.random.seed(100)

df=pd.DataFrame({"Date":pd.Index(pd.date_range(start='2/2/2019',periods=3)).repeat(3),

             "Class":["1A","2B","3C","1A","2B","3C","1A","2B","3C"],

             "Numbers":np.random.randn(9)})



df['Numbers2'] = df['Numbers'] * 2



df
df.pivot(index='Date', columns='Class', values='Numbers')
df.pivot(index='Date', columns='Class')
df.pivot(index='Date', columns='Class')['Numbers']
df.pivot(index='Date', columns='Class')['Numbers'].reset_index()
np.random.seed(100)

df1=pd.DataFrame({"Date":pd.Index(pd.date_range(start='2/2/2019',periods=3)).repeat(3),

             "Class":["1A","1A","1A","2B","2B","2B","3C","3C","3C"],

             "Numbers":np.random.randn(9)})



df1
df1.pivot(index='Date', columns='Class')
df
df.melt(id_vars=['Date','Class'])
df.melt(id_vars=['Date','Class'],value_vars=['Numbers'])
df.melt(id_vars=['Date','Class'],value_vars=['Numbers'],value_name="Numbers_Value",var_name="Num_Var")
df1
df1.melt(id_vars=['Date','Class'],value_vars=['Numbers'],value_name="Numbers_Value",var_name="Num_Var")
df
df.set_index(["Date","Class"]).stack()
df.set_index(["Date","Class"]).stack(0)
df.set_index(["Date","Class"]).stack(-1)
df.set_index(["Date","Class"]).stack().unstack()
df.set_index(["Date","Class"]).stack().unstack(-1)
df.set_index(["Date","Class"]).stack().unstack(0)
df.set_index(["Date","Class"]).stack().unstack(1)
df.set_index(["Date","Class"]).stack().unstack([1,-1])
df=pd.DataFrame({"Date":pd.Index(pd.date_range(start='2/2/2019',periods=2)).repeat(4),

             "Class":["1A","2B","3C","1A","2B","3C","1A","2B"],

             "Numbers":np.random.randn(8)})



df['Numbers2'] = df['Numbers'] * 2



df
df.groupby('Date')["Numbers"].mean()
df.groupby('Date',as_index=False)["Numbers"].mean()
df.groupby(['Date','Class'],as_index=False)["Numbers"].mean()
df.groupby(['Date','Class'],as_index=False)[["Numbers","Numbers2"]].mean()
df.groupby(['Date'],as_index=False).aggregate({"Numbers":"sum","Numbers2":"mean"})
df.pivot(index="Date",columns="Class")
df.pivot_table(index="Date",columns="Class")
df.pivot_table(index="Date",columns="Class",aggfunc="sum")
df
pd.crosstab(df.Date,df.Class)
pd.crosstab(df.Date,df.Class,values=df.Numbers,aggfunc='sum')
pd.crosstab(df.Date,df.Class,values=df.Numbers,aggfunc='mean')