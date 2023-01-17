import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
plt.style.use('ggplot')

import langid # identify languages based on tweets

df = pd.read_csv('../input/tweets.csv')

# identify languages
predicted_languages = [langid.classify(tweet) for tweet in df['tweets']]

lang_df = pd.DataFrame(predicted_languages, columns=['language','value'])

# show the top ten languages & their counts
print(lang_df['language'].value_counts().head(10))

# plot the counts for the top ten most commonly used languages
colors=sns.color_palette('hls', 10) 
pd.Series(lang_df['language']).value_counts().head(10).plot(kind = "bar",
                                                        figsize=(12,9),
                                                        color=colors,
                                                        fontsize=14,
                                                        rot=45,
                                                        title = "Top 10 most common languages")