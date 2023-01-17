import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
plt_kwargs = {'figsize': (10, 6)}
meets = pd.read_csv("../input/meets.csv", index_col=0)
meets.head()
meets['Federation'].value_counts().head(10).plot.bar(
    title='Top 10 Meet-Organizing Federations', **plt_kwargs
)
(pd.to_datetime(meets['Date'])
     .to_frame()
     .assign(n=0)
     .set_index('Date')
     .resample('AS')
     .count()
     .plot.line(title='Meets Included in the Dataset by Year', **plt_kwargs))
meets['MeetCountry'].value_counts().head(10).plot.bar(**plt_kwargs, title="Meets by Country")
meets['MeetState'].value_counts().head(10).plot.bar(**plt_kwargs, title="Powerlifting Meet State-by-State Representation")
len(" ".join(meets['MeetName'].values))
from wordcloud import WordCloud
wordcloud = WordCloud().generate(" ".join(meets['MeetName'].values))
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, figsize=(12, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
competitors = pd.read_csv("../input/openpowerlifting.csv", index_col=0)
competitors.head()
competitors['Sex'].value_counts() / len(competitors)
competitors['Equipment'].value_counts().plot.bar(**plt_kwargs, title='Assistive Equipment Used')
import seaborn as sns
fig, ax = plt.subplots(1, figsize=(10, 6))
sns.kdeplot(competitors['Age'])
plt.suptitle("Lifter Age Distribution")
competitors.query('Sex == "M"')['WeightClassKg'].str.replace("+", "").astype(float).dropna().value_counts().sort_index().plot.line(**plt_kwargs)
competitors.query('Sex == "F"')['WeightClassKg'].str.replace("+", "").astype(float).dropna().value_counts().sort_index().plot.line()
plt.suptitle("Male (Blue) and Female (Orange) Weight Classes")
plt.gca().set_xlabel("Weight Class (kg)")
plt.gca().set_ylabel("N")
(competitors
     .query('Sex == "M"')
     .loc[:, ['WeightClassKg', 'BestSquatKg']]
     .dropna()
     .pipe(lambda df: df.assign(WeightClassKg=df.WeightClassKg.map(lambda v: np.nan if "+" in v else np.nan if float(v) < 0 else v)))
     .dropna()
     .astype(float)
     .groupby("WeightClassKg")
     .agg([np.max, np.median])
     .plot.line(**plt_kwargs)
)
plt.gca().set_xlabel("Competitor Weight")
plt.gca().set_ylabel("Weight Lifted")
plt.suptitle("Male Powerlifting Competitor Median and Maximum Squats")
plt.plot(70, 70, 'go')