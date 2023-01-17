%matplotlib inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3
from sklearn import linear_model
from sklearn import preprocessing
from sklearn.metrics import r2_score
sqlite_file = '../input/database.sqlite'
con = sqlite3.connect(sqlite_file)
df = pd.read_sql_query("SELECT INSTNM, ZIP, LATITUDE, LONGITUDE, TUITFTE, INEXPFTE, AVGFACSAL, TUITIONFEE_IN, TUITIONFEE_OUT, CONTROL,\
                        mn_earn_wne_p10, md_earn_wne_p10, sd_earn_wne_p10, pct25_earn_wne_p10, pct75_earn_wne_p10,\
                        ADM_RATE, ADM_RATE_ALL, SATVRMID, SATMTMID, SATWRMID\
                        PCIP14, PCIP15\
                        FROM Scorecard" , con)
con.close()
plt.figure(figsize=(10,10))
sns.regplot(df.TUITIONFEE_IN, df.TUITIONFEE_OUT, scatter = True, fit_reg = False)
print(df.CONTROL[df.TUITIONFEE_IN == df.TUITIONFEE_OUT].value_counts()/df.CONTROL.value_counts())
sns.lmplot(x="TUITIONFEE_IN", y="TUITIONFEE_OUT", col="CONTROL", data=df, size=4)
plt.show()
df_TUITcomplete = df[['TUITIONFEE_IN','TUITIONFEE_OUT', 'CONTROL']].dropna(how='any')
# aggegating results 
df_TUITcomplete.groupby('CONTROL').mean()