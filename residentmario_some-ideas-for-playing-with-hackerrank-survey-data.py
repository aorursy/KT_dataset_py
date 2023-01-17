import numpy as np
import pandas as pd

responses = pd.read_csv("../input/HackerRank-Developer-Survey-2018-Values.csv", parse_dates=['StartDate', 'EndDate'])
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
plt_kwargs = {'figsize': (12, 6), 'color': '#00b760'}
((responses.EndDate - responses.StartDate)
     .dropna()
     .map(lambda v: v.seconds // 60)
     .value_counts()
     .sort_index()
     .head(50)).plot.bar(title='Time Spent Filling Out the Survey', **plt_kwargs)
fig = plt.gcf(); fig.set_facecolor('white')
ax = plt.gca(); ax.set_facecolor('white')
((responses.EndDate - responses.StartDate)
     .dropna()
     .map(lambda v: v.seconds // 60)
     .pipe(lambda srs: pd.Series({'>30': (srs > 30).sum(), '<=30': (srs <= 30).sum()}))
     .pipe(lambda srs: srs / srs.sum())
)
responses.CountryNumeric.value_counts(dropna=False).head(10).plot.bar(title='Most Common Respondant Countries', **plt_kwargs)
fig = plt.gcf(); fig.set_facecolor('white')
ax = plt.gca(); ax.set_facecolor('white')
import numpy as np
responses_numeric = pd.read_csv("../input/HackerRank-Developer-Survey-2018-Numeric.csv", parse_dates=['StartDate', 'EndDate'])
codebook = pd.read_csv("../input/HackerRank-Developer-Survey-2018-Codebook.csv")
countries = pd.read_csv('../input/Country-Code-Mapping.csv')
ghana_code = countries.query("Label == 'Ghana'").Value.iloc[0]
india_code = countries.query("Label == 'India'").Value.iloc[0]
usa_code = countries.query("Label == 'United States'").Value.iloc[0]
def style_sd():
    fig = plt.gcf(); fig.set_facecolor('white')
    ax = plt.gca(); ax.set_facecolor('white')
    ax.set_xticks([], [])
    ax.set_ylim([-0.5, 0.5])
    pass

sd = (responses_numeric.loc[responses_numeric.CountryNumeric.isin([ghana_code, india_code, usa_code])]
     .groupby('CountryNumeric')
     .mean()
     .assign(Country=['Ghana', 'India', 'United States'])
     .set_index('Country')
)
(sd.loc['India'] - sd.loc['Ghana']).iloc[1:].plot.bar(title='India/Ghana Difference Chart', **plt_kwargs)
style_sd()
(sd.loc['United States'] - sd.loc['India']).iloc[1:].plot.bar(title='United States/India Difference Chart', **plt_kwargs)
style_sd()
usa_code = countries.query("Label == 'United States'").Value.iloc[0]
south_korea_code = countries.query("Label == 'South Korea'").Value.iloc[0]
sd = (responses_numeric.loc[responses_numeric.CountryNumeric.isin([usa_code, south_korea_code])]
     .groupby('CountryNumeric')
     .mean()
     .assign(Country=['United States', 'South Korea'])
     .set_index('Country')
)

(sd.loc['South Korea'] - sd.loc['United States']).iloc[1:].plot.bar(title='South Korea/United States Difference Chart', **plt_kwargs)
style_sd()
fig = plt.subplots(1, figsize=(12, 6))
ax = plt.gca()
df = (responses_numeric
      .groupby('CountryNumeric')
      .mean()
      .pipe(lambda df: df.assign(Country=countries.set_index('Value').Label[df.index]))
      .set_index('Country')
      .iloc[:, 1:]
      .dropna()
      .pipe(lambda df: df[df.index.isin(responses.CountryNumeric.value_counts().pipe(lambda srs: srs[srs > 20]).index)])
      .apply(lambda srs: srs.mean(), axis='columns')
      .sort_values()
     )
sns.kdeplot(df, ax=ax, color='#00b760')
fig, ax = plt.gcf(), plt.gca()
sns.rugplot(df.values, color='black', ax=ax)
plt.suptitle("Mean Question Score by Country Surveyed ($n \geq 20$)")
fig.set_facecolor('white'), ax.set_facecolor('white')
pass
responses.q4Education.value_counts().plot.bar(**plt_kwargs)
fig = plt.gcf(); fig.set_facecolor('white')
ax = plt.gca(); ax.set_facecolor('white')
ax.set_xticklabels(ax.get_xticklabels(), rotation=25, ha='right')
plt.suptitle("Educational Levels of Respondants")
(responses
     .groupby(['CountryNumeric'])
     .apply(lambda df: np.nan if len(df) < 20 else df.q4Education.value_counts().index[0])
     .value_counts()
)