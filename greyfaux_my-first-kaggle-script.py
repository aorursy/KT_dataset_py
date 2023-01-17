import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

from matplotlib import style
plt.style.use('ggplot')
df = pd.read_csv('../input/Sentiment.csv')
tt = df.candidate.value_counts()
tt.plot(kind='bar', title = 'Total Tweets for each Candidate', figsize=(14,6))
st = df.groupby(['candidate','sentiment']).size()
st.plot(kind="barh", figsize=(12,8), color=['lightgreen', 'lightskyblue', 'lightcoral'])
dt_df = df[df['candidate'] == 'Donald Trump']
sentiment_by_candidate = dt_df.groupby('sentiment').size()

sentiment_by_candidate.plot(kind = "pie" , colors=['lightgreen', 'lightskyblue', 'lightcoral'], explode=[0,0,0.05], title = "Donald Trump Tweet Sentiment" , autopct='%1.1f%%', shadow=True, figsize=[6,6])