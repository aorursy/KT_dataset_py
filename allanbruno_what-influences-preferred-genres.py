#Importing necessary modules
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

filepath = '../input/videogamesales/vgsales.csv'
columns_to_drop = ['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales']
df = pd.read_csv(filepath)
df = df.drop(columns=columns_to_drop)
df_bar = df.groupby(by=df['Platform']).sum()
df_bar = df_bar.sort_values(by='Global_Sales', ascending=False)
df_bar = df_bar.iloc[0:10] #Amount of videogames to plot(Bigger to smaller)
plt.figure(figsize=(12,8))
plt.title('Global sales by platform')
sns.barplot(x=df_bar.index, y=df_bar['Global_Sales'])

fig, axs=plt.subplots(5,2, figsize=(16,25))
axes_split = [axs[0, 0], axs[0,1], axs[1,0], axs[1,1], axs[2,0], axs[2,1], axs[3,0], axs[3,1], axs[4,0], axs[4,1]]
top_platforms = ['PS','PS2', 'PS3','PS4','PSP', 'Wii','DS','X360',  'GBA', 'PC']
#ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8,ax9,ax10
fig.suptitle('Global sales of Genres')

for i in range(10):
    df_mask = df['Platform'] == top_platforms[i]
    df_general = df[df_mask]
    df_general = df_general.groupby(by=df['Genre']).sum().sort_values(by='Global_Sales', ascending=False)
    df_general = df_general.iloc[0:10]
    sns.barplot(x=df_general.index, y=df_general['Global_Sales'], ax=axes_split[i])
    axes_split[i].set_title(top_platforms[i])
    axes_split[i].xaxis.set_tick_params(rotation=20)
    axes_split[i].set_xlabel('')