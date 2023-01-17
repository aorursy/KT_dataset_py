import warnings
warnings.filterwarnings('ignore')

import pandas as pd

df = pd.read_csv("../input/countries of the world.csv", decimal=',')
df.head()
import seaborn as sns
sns.set(style="whitegrid", font_scale=1.3)
sns.barplot(x="GDP ($ per capita)", y="Country", data=df.sort_values(['GDP ($ per capita)'], ascending=False).reset_index(drop=True)[:10]);
sns.barplot(x="Pop. Density (per sq. mi.)", y="Country", data=df.sort_values(['Pop. Density (per sq. mi.)'], ascending=False).reset_index(drop=True)[:10]);
sns.barplot(x="Area (sq. mi.)", y="Country", data=df.sort_values(['Area (sq. mi.)'], ascending=False).reset_index(drop=True)[:10]);
sns.barplot(x="Population", y="Country", data=df.sort_values(['Population'], ascending=False).reset_index(drop=True)[:10]);
sns.barplot(x="Literacy (%)", y="Country", data=df.sort_values(['Literacy (%)'], ascending=False).reset_index(drop=True)[:10]);
sns.barplot(x="Coastline (coast/area ratio)", y="Country", data=df.sort_values(['Coastline (coast/area ratio)'], ascending=False).reset_index(drop=True)[:10]);
sns.barplot(x="Birthrate", y="Country", data=df.sort_values(['Birthrate'], ascending=False).reset_index(drop=True)[:10]);
sns.barplot(x="Deathrate", y="Country", data=df.sort_values(['Deathrate'], ascending=False).reset_index(drop=True)[:10]);
sns.barplot(x="Infant mortality (per 1000 births)", y="Country", data=df.sort_values(['Infant mortality (per 1000 births)'], ascending=False).reset_index(drop=True)[:10]);
import matplotlib.pyplot as plt
df = df.dropna()
print(df.shape)
sns.set(font_scale=1.7)
sns.heatmap(df.corr(), annot=True, fmt=".2f", linewidths=1, cmap='viridis', ax=plt.subplots(figsize=(30,30))[1]);
sns.set(font_scale=1.5)
sns.regplot(x='GDP ($ per capita)', y='Birthrate', data=df, logx=True, truncate=True, line_kws={"color": "red"});
sns.regplot(x='GDP ($ per capita)', y='Deathrate', data=df, logx=True, truncate=True, line_kws={"color": "red"});
sns.scatterplot(x='GDP ($ per capita)', y='Climate', data=df);
sns.regplot(x='GDP ($ per capita)', y='Literacy (%)', data=df, logx=True, truncate=True, line_kws={"color": "red"});
sns.regplot(x='GDP ($ per capita)', y='Phones (per 1000)', data=df, truncate=True, line_kws={"color": "red"});
sns.regplot(x='GDP ($ per capita)', y='Net migration', data=df, truncate=True, line_kws={"color": "red"});
sns.regplot(x='Infant mortality (per 1000 births)', y='GDP ($ per capita)', data=df, logx=True, truncate=True, line_kws={"color": "red"});
sns.regplot(x='Infant mortality (per 1000 births)', y='Literacy (%)', data=df, truncate=True, line_kws={"color": "red"});
sns.regplot(x='Infant mortality (per 1000 births)', y='Phones (per 1000)', data=df, logx=True, truncate=True, line_kws={"color": "red"});
sns.regplot(x='Infant mortality (per 1000 births)', y='Birthrate', data=df, truncate=True, line_kws={"color": "red"});
sns.regplot(x='Infant mortality (per 1000 births)', y='Agriculture', data=df,  truncate=True, line_kws={"color": "red"});
sns.regplot(x='Literacy (%)', y='Birthrate', data=df, truncate=True, line_kws={"color": "red"});
sns.regplot(x='Literacy (%)', y='Agriculture', data=df, truncate=True, line_kws={"color": "red"});
sns.regplot(x='Coastline (coast/area ratio)', y='Net migration', data=df, truncate=True, line_kws={"color": "red"});
sns.regplot(x='Agriculture', y='Birthrate', data=df, truncate=True, line_kws={"color": "red"});
