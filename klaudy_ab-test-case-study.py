import pandas as pd

import numpy as np
#read the data we have. 



df=pd.read_csv("/kaggle/input/ab-test-practice-case/ab_data.csv")
#take a look at the first 9 rows.

df.head(9)
#to see how many rows we have

df.info()
#to see how many unique rows we have

df.nunique()
#to see duplicated values in user id

duplicated_users_id=df[df.duplicated(['user_id'], keep = False)].sort_values("user_id",ascending=True)

duplicated_users_id
print("The rate of duplicated user id is",duplicated_users_id.user_id.nunique()/df.user_id.nunique())
df.drop_duplicates("user_id",keep=False,inplace=True)
df.nunique()
df.info()
df.isnull().sum()
print("The percentage of experiment group is",df[df.landing_page=="new_page"].shape[0]/df.shape[0])
df.groupby("landing_page")["converted"].mean()
# calculate the Ncont, Nexp, Xcont, Xexp

Ncont= df[df.landing_page=="old_page"].shape[0]

Nexp= df[df.landing_page=="new_page"].shape[0]

Xcont=df[(df["landing_page"]=="old_page") & (df["converted"] ==1)].shape[0]

Xexp=df[(df["landing_page"]=="new_page") & (df["converted"]==1)].shape[0]
Ncont,Nexp,Xcont,Xexp
p_pool = (Xcont+Xexp)/(Ncont+Nexp)

p_pool
SE_pool = np.sqrt(p_pool*(1-p_pool)*(1/Ncont+1/Nexp))

SE_pool
pexp=Xexp/Nexp

pcont=Xcont/Ncont



pexp,pcont
d_hat = pexp-pcont

z_score=d_hat/SE_pool

z_score
from scipy.stats import norm

z_alpha=norm.ppf(0.95)

z_alpha
import statsmodels.stats.proportion as sp

z_score, p_value = sp.proportions_ztest([Xexp,Xcont],[Nexp,Ncont],alternative="larger")

print('z_score:', z_score, '，p-value:', p_value)