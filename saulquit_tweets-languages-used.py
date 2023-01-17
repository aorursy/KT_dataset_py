import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
plt.style.use('ggplot')

import langid
df = pd.read_csv('../input/languagestweets/languages-tweets.csv')
pred_language = [langid.classify(tweet) for tweet in df['tweets']]
lang_df = pd.DataFrame(pred_language, columns = ['language', 'value'])
print(lang_df['language'].value_counts().head(10))
colors = sns.color_palette('hls', 10)
pd.Series(lang_df['language']).value_counts().head(10).plot(kind='bar',
                                                           figsize = (12,6),
                                                           color= colors,
                                                           fontsize= 14,
                                                           rot =45,
                            title='Top 10 most common tweets languages')
plt.xlabel('Language')
plt.ylabel('Count')
