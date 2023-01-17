import numpy as np 
import pandas as pd 
from IPython.core.display import display, HTML
df = pd.DataFrame(pd.read_csv("../input/ks-projects-201801.csv"))
all = df['main_category'].value_counts()
successful = df['main_category'][df['state'] == 'successful'].value_counts()
df2 = pd.DataFrame({'all': all, 'successful': successful})
df2.plot.bar()
success_rate = successful / all.astype(float)
df3 = pd.DataFrame({'success_rate': success_rate})
output = df3.to_html(formatters={'success_rate': '{:,.2%}'.format})

display(HTML(output))
is_USA = df['currency'] == 'USD'
was_successful = df['state'] == 'successful'
was_failed = df['state'] == 'failed'

df4 = df[['main_category', 'usd pledged', 'usd_pledged_real', 'usd_goal_real']]

print(df4[is_USA & was_successful].groupby('main_category').median())
print('\n')
print(df4[is_USA & was_failed].groupby('main_category').median())
df.head()