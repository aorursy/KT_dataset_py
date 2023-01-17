import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df =pd.read_csv('../input/ramen-ratings.csv')
df.head()
len(df)
country_count = df['Country'].value_counts()

sns.set_context('talk')

fig = plt.figure(figsize=(20, 10))
ax = sns.barplot(x=country_count.index, y=country_count.values)
ax.set_title('# of Country')
ax.set_xticklabels(country_count.index, rotation=90);
brand_count =  df['Brand'].value_counts()
brand_count_20 = brand_count[:20]
fig = plt.figure(figsize=(20, 10))
ax = sns.barplot(x=brand_count_20.index, y=[count/sum(brand_count) for count in brand_count_20])
ax.set_title('Top 20 Brand')
ax.set_xticklabels(brand_count_20.index, rotation=90);
df_stars = df[df['Stars'] != 'Unrated']

fig = plt.figure(figsize=(20, 10))
ax = sns.distplot([float(star) for star in df_stars['Stars']], norm_hist=True, bins=20)
ax.set_title('Distribution of Stars');
def plot_top_brand(country):
    brand_count = df[df['Country'] == country]['Brand'].value_counts()
    brand_count_20 = df[df['Country'] == country]['Brand'].value_counts()[:20]
    fig = plt.figure(figsize=(20, 10))
    ax = sns.barplot(x=brand_count_20.index, y=[count/sum(brand_count) for count in brand_count_20])
    ax.set_title('Top 20 Brand in {}'.format(country))
    ax.set_xticklabels(brand_count_20.index, rotation=90);
plot_top_brand('Japan')
plot_top_brand('USA')
plot_top_brand('South Korea')
plot_top_brand('Taiwan')
plot_top_brand('Vietnam')
plot_top_brand('UK')
