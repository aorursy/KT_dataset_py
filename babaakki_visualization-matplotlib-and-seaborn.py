#Impporting the pyplot module from matplotlib
import matplotlib.pyplot as plt
import numpy as np
X = np.arange(0, 10, 0.5)
Y = X ** 2
Z = X ** 3
plt.figure()
plt.plot(X, Y)
plt.figure()
plt.plot(X, Y)
#adding labels for x-axis and y-axis
plt.xlabel('Number')
plt.ylabel('Squares')
plt.figure()
plt.plot(X, Y)
plt.xlabel('Number')
plt.ylabel('Squares')
#Adding title to the plot
plt.title('Number-Square Plot')
plt.figure()
plt.plot(X, Y)
#adding another line to a plot
plt.plot(X, Z)
plt.xlabel('Number')
plt.ylabel('Values')
plt.title('Number-Square-Cube Plot')
plt.figure()
#adding label to a line
plt.plot(X, Y, label='Y=X^2')
plt.plot(X, Z, label='Z=X^3')
plt.xlabel('Number')
plt.ylabel('Squares')
plt.title('Number-Square Plot')
#adding legend
plt.legend()
plt.figure()
#adding colors to a line
plt.plot(X, X, label='Y=X', color='blue')
plt.plot(X, Y, label='Y=X^2', color='red')
plt.plot(X, Z, label='Y=X^3', color='green')
plt.xlabel('Number')
plt.ylabel('Squares')
plt.title('Number-Square Plot')
plt.legend()
plt.figure()
#adding linestyle to a line
plt.plot(X, X, label='Y=X', color='blue', linestyle='-')
plt.plot(X, Y, label='Y=X^2', color='red', linestyle='--')
plt.plot(X, Z, label='Y=X^3', color='green', linestyle='-.')
plt.xlabel('Number')
plt.ylabel('Squares')
plt.title('Number-Square Plot')
plt.legend()
plt.figure()
plt.plot(X, X, label='Y=X', color='blue', linestyle='-')
plt.plot(X, Y, label='Y=X^2', color='red', linestyle='--')
plt.plot(X, Z, label='Y=X^3', color='green', linestyle='-.')
plt.xlabel('Number')
plt.ylabel('Squares')
plt.title('Number-Square Plot')
plt.legend()
#Adding Grid to a plot
plt.grid(True)
fig, ax = plt.subplots(figsize=(6,5))

ax.plot(X, X+1, color="red", alpha=0.5) # half-transparant red
ax.plot(X, X+2, color="#1155dd")        # RGB hex code for a bluish color
ax.plot(X, X+3, color="#15cc55")        # RGB hex code for a greenish color
#Plotting 2 plots together using subplot
fig, axes = plt.subplots(1, 2, figsize=(10,3))

# default grid appearance
axes[0].plot(X, Y, X, Z)
axes[0].grid(True)

# custom grid appearance
axes[1].plot(X, Y, X, Z)
axes[1].grid(color='b', alpha=0.5, linestyle='dashed', linewidth=0.5)
fig = plt.figure(figsize=(12,8))

#2 rows 2 cols index 1
ax0 = plt.subplot(221)
ax0.plot(X, Y, X, Z)
#2 rows 2 cols index 2
ax1 = plt.subplot(222)
ax1.plot(X, Y, X, Z)
#2 rows 2 cols index 3
ax3 = plt.subplot(223)
ax3.plot(X, Y, X, Z)
#2 rows 2 cols index 4
ax4 = plt.subplot(224)
ax4.plot(X, Y, X, Z)
n = np.arange(10)
x = np.arange(100)
m = np.random.randn(100)
fig, axes = plt.subplots(2, 2, figsize=(20,16))
#Scatter Plot
axes[0][0].scatter(x, m)
axes[0][0].set_title("scatter")
#Step Plot
axes[0][1].step(n, n**2, lw=2)
axes[0][1].set_title("step")
#Bar Plot
axes[1][0].bar(n, n**2, align="center", width=0.5, alpha=0.5)
axes[1][0].set_title("bar")
#Histogram
axes[1][1].hist(m)
axes[1][1].set_title("histogram")
import pandas as pd
df = pd.read_csv('../input/imdb1000/imdb_data.csv', sep='\t', header=0, index_col='Seq')
df.head()
df.describe()
rating = np.array(df['Imdb Rating'])
votes = np.array(df.iloc[:,8])
earnings = np.array(df.loc[:,'Gross(in Million Dollars)'])
def bins_labels(bins, **kwargs):
    bin_w = (max(bins) - min(bins)) / (len(bins) - 1)
    plt.xticks(np.arange(min(bins)+bin_w/2, max(bins), bin_w), bins, **kwargs)
    plt.xlim(bins[0], bins[-1])

plt.figure(figsize=(15, 5))

bins = range(11)
ax1 = plt.subplot(1,3,1)
ax1.hist(rating, bins=bins)
ax1.set_xlabel('User Ratings')
bins_labels(bins, fontsize=20)


ax2 = plt.subplot(1,3,2)
ax2.hist(votes)
ax2.set_xlabel('User Votes')

ax3 = plt.subplot(1,3,3)
ax3.hist(earnings)
ax3.set_xlabel('Gross Earning (in Million Dollars)')

plt.show()
import seaborn as sns
num_df = df.loc[:,['Imdb Rating', 'User Votes', 'Gross(in Million Dollars)']]
num_df['User Votes'] /= 100000 
num_df['Gross(in Million Dollars)'] /= 100
num_df = num_df.rename(columns={'Gross(in Million Dollars)':'Gross(in 100 Million Dollars)'})
num_df.describe()
#Setting figure size
plt.figure(figsize=(25,5))

#Adding title
plt.title('Top 1000 movies with IMDB rating, user votes, and Gross Earnings')

#Setting xticks
plt.xticks(range(0, 1000, 99))

# Plotting the entire data
sns.lineplot(data=num_df)
#Plotting a subset of the data
plt.figure(figsize=(20,5))
plt.title('Imdb rating and User Votes Plot')
sns.lineplot(data=num_df['Imdb Rating'], label='IMDB Rating')
sns.lineplot(data=num_df['User Votes'], label='User Votes')
plt.xticks(range(0, 1000, 50))
plt.xlabel('Sequence')
def create_column_freq_dataframe(df, col_name):
    dct = {}
    series = list(df[col_name])
    for lst in series:
        #print(lst)
        lst = lst.replace('[','').replace(']','').replace("'",'').split(',')
        for item in lst:
            item = item.strip()
            dct[item] = dct.get(item, 0) + 1
    dtf = pd.DataFrame(list(dct.items()), columns=[col_name, 'freq']).set_index(col_name)
    return dtf
df.columns
df_genre = create_column_freq_dataframe(df, 'Genre')
plt.figure(figsize=(25,10))
plt.title('Number of films in each Genre')
sns.barplot(x=df_genre.index, y=df_genre['freq'])
#Checking if there is any null entry
df[df.Directors.isnull()]
df.Directors = df.Directors.fillna('[]')
gd_dct = {}
for i, j in df.iterrows():
    gen = [s.strip() for s in j['Genre'].replace('[','').replace(']','').replace("'",'').split(',')]
    dire = [s.strip() for s in j['Directors'].replace('[','').replace(']','').replace("'",'').split(',')]
    for g in gen:
        for d in dire:
            gd_dct[(g,d)] = gd_dct.get((g,d), 0) + 1

indexes = []
col_names = []
for key in gd_dct.keys():
    indexes.append(key[1])
    col_names.append(key[0])
indexes = list(set(indexes))
col_names = list(set(col_names))

gd_df = pd.DataFrame(index=indexes, columns=col_names)

for key, val in gd_dct.items():
    gd_df.loc[key[1], key[0]] = val
gd_df = gd_df.fillna(0)
gd_df
plt.figure(figsize=(25,20))
plt.title('Number of films in each Genre')
sns.heatmap(data=gd_df.iloc[1:51,:], annot=True)
#Reuse numerical entries dataframe
num_df
plt.figure(figsize=(12,4))
ax0 = plt.subplot(121)
ax0 = sns.scatterplot(x=num_df['Imdb Rating'], y=num_df['Gross(in 100 Million Dollars)'])
ax0.set_title('IMDB Rating Vs Gross earnings')

ax1 = plt.subplot(122)
ax1 = sns.scatterplot(x=num_df['Imdb Rating'], y=num_df['User Votes'])
ax1.set_title('IMDB Rating Vs User Votes')
plt.figure(figsize=(12,4))
ax0 = plt.subplot(121)
ax0 = sns.regplot(x=num_df['Imdb Rating'], y=num_df['Gross(in 100 Million Dollars)'])
ax0.set_title('IMDB Rating Vs Gross earnings')

ax1 = plt.subplot(122)
ax1 = sns.regplot(x=num_df['Imdb Rating'], y=num_df['User Votes'])
ax1.set_title('IMDB Rating Vs User Votes')
# Change the style of the figure to the "dark" theme
sns.set_style("dark")
# Other themes:
#    darkgrid
#    white
#    whitegrid
#    ticks


plt.figure(figsize=(8, 6))
plt.title('Relation Between User Votes and Imdb Rating divided by Certificate')
sns.scatterplot(x=df['Imdb Rating'], y=df['User Votes'], hue=df['Certificate'])
plt.figure(figsize=(10, 6))
plt.title('Data categorised over different Certificates scaled over IMDB Ratings')
sns.swarmplot(x=df['Certificate'], y=df['Imdb Rating'])
num_df[num_df['Gross(in 100 Million Dollars)'].isnull()]
num_df['Gross(in 100 Million Dollars)'] = num_df['Gross(in 100 Million Dollars)'].fillna(0.0)
#Generating 3 subplots, one each for Rating, Votes, and Earning distributions.
plt.figure(figsize=(15, 5))

ax1 = plt.subplot(131)
ax1 = sns.distplot(a=num_df['Imdb Rating'], kde=False)
ax1.set_title('Imdb rating Distribution')

ax2 = plt.subplot(132)
ax2 = sns.distplot(a=num_df['User Votes'], kde=False)
ax2.set_title('User Votes Distribution')

ax3 = plt.subplot(133)
ax3 = sns.distplot(a=num_df['Gross(in 100 Million Dollars)'], kde=False)
ax3.set_title('Gross Earning Distribution')

plt.plot()
# Generating color coded plot, where different color represents different density plots.
plt.figure(figsize=(5, 5))
plt.title('Ratings, Votes, and Earnings distribution')
sns.kdeplot(data=num_df['Imdb Rating'], label='Rating', shade=True)
sns.kdeplot(data=num_df['User Votes'], label='Votes', shade=True)
sns.kdeplot(data=num_df['Gross(in 100 Million Dollars)'], label='Earnings', shade=True)
