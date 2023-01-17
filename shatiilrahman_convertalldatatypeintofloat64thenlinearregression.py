import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input"))
d = pd.read_csv("../input/Cgpa_Gre_Ielts_Toefil.csv")
d.head()
d.dtypes
d['gre_total'] = d.gre_total.astype(float)
d.dtypes
import statsmodels.api as sm
x = d['bd_cgpa']
y = d['gre_total']
x = sm.add_constant(x)
m = sm.OLS(y,x)
mf = m.fit()
print(mf.params)
mf.summary()