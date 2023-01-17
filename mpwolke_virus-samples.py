# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv("/kaggle/input/ai4all-project/data/viral_calls/sample_overviews.csv")

df.head()
print(f"data shape: {df.shape}")
df.describe()
df.isnull().sum()
def plot_count(feature, title, df, size=1):

    f, ax = plt.subplots(1,1, figsize=(4*size,4))

    total = float(len(df))

    g = sns.countplot(df[feature], order = df[feature].value_counts().index[:20], palette='Set3')

    g.set_title("Number and percentage of {}".format(title))

    if(size > 2):

        plt.xticks(rotation=90, size=8)

    for p in ax.patches:

        height = p.get_height()

        ax.text(p.get_x()+p.get_width()/2.,

                height + 3,

                '{:1.2f}%'.format(100*height/total),

                ha="center") 

    plt.show()
plot_count("total_reads", "total_reads", df,4)
plot_count("nonhost_reads_percent", "nonhost_reads_percent", df,4)
# Distribution of different type of amount

fig , ax = plt.subplots(1,3,figsize = (12,5))



total_reads = df.total_reads.values

nonhost_reads = df.nonhost_reads.values

total_ercc_reads = df.total_ercc_reads.values



sns.distplot(total_reads , ax = ax[0] , color = 'blue').set_title('Viral Samples Total Reads' , fontsize = 14)

sns.distplot(nonhost_reads , ax = ax[1] , color = 'cyan').set_title('Viral Samples Nonhost Reads' , fontsize = 14)

sns.distplot(total_ercc_reads , ax = ax[2] , color = 'purple').set_title('Viral Samples Total ERCC Reads' , fontsize = 14)



plt.show()
df['water_control'].value_counts()
f,ax = plt.subplots(1,2,figsize = (16,8))



colors = ['blue','red']

labels = ['Yes', 'No']

plt.suptitle('Water Condition & Reads after Trimmomatic',fontsize = 20)



df['water_control'].value_counts().plot.pie(explode = [0,0.25], autopct = "%1.2f%%" , ax = ax[0],

                                                 labels = labels , colors = colors ,fontsize = 12 , startangle = 70)



ax[0].set_ylabel('% of condition of Water')



palette = ["Blue", "Red"]



sns.barplot(x = 'upload_date', y = 'reads_after_trimmomatic',hue = 'quality_control',data = df,palette = palette,

           estimator = lambda x: len(x)/len(df) * 100)



ax[1].set(ylabel='%')
cmap = plt.cm.Set2

df.groupby(['insert_size_standard_deviation','subsampled_fraction'])['total_ercc_reads'].sum().unstack().plot(figsize = (15,6))

plt.title('Subsampled Fraction Standard Deviation by Total ERCC Reads')
f , ((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2)

sns.set_style('whitegrid')

cmap = plt.cm.inferno

total_reads = df.groupby(['upload_date','insert_size_mean']).total_reads.mean()

total_reads.unstack().plot(kind = 'area',ax = ax1 , figsize = (16,12) , colormap = cmap , grid = False)

ax1.set_title('Average Total Reads by Insert Size Mean')

ax1.set_xlabel('Upload Date')



nonhost_reads = df.groupby(['upload_date','insert_size_mean']).nonhost_reads.mean().unstack().plot(kind = 'area',ax = ax2 ,colormap = cmap, figsize = (16,12),grid = False)

ax2.set_title('Average Nonhost Reads by Insert Size Mean')

ax2.set_xlabel('Upload Date')

insert_size_read_pairs = df.groupby(['upload_date','insert_size_mean'])['insert_size_read_pairs'].mean().unstack().plot(kind = 'area',ax = ax3 , figsize = (16,12) ,colormap = cmap, grid = False)

ax3.set_title('Average Size Read Pairs by Insert Size Mean ')

ax3.set_xlabel('Upload Date')



total_ercc_reads = df.groupby(['upload_date','insert_size_mean']).total_ercc_reads.mean().unstack().plot(kind = 'area',ax = ax4 , figsize = (16,12),colormap = cmap,grid = False)

ax4.set_title('Total ERCC Reads by Insert Size Mean')

ax4.set_xlabel('Upload Date')

plt.show()
fig , ((ax1,ax2),(ax3,ax4)) = plt.subplots(nrows = 2, ncols = 2, figsize = (14,6))



sns.violinplot(x = 'compression_ratio' , y = 'reads_after_star' , data = df , ax = ax1 , palette = 'Set2')

sns.violinplot(x = 'compression_ratio' , y = 'reads_after_trimmomatic' , data = df , ax = ax2 , palette = 'Set2')

sns.boxplot(x = 'compression_ratio' , y = 'reads_after_priceseq', data = df, ax = ax3 , palette = 'Set2')

sns.boxplot(x = 'compression_ratio',y = 'reads_after_cdhitdup', data = df, ax = ax4, palette = 'Set2')
f , (ax1,ax2) = plt.subplots(1,2,figsize = (15,6))

cmap = plt.cm.coolwarm



by_subsampled_fraction = df.groupby(['upload_date','insert_size_mode']).subsampled_fraction.mean()

by_subsampled_fraction.unstack().plot(ax = ax1 , colormap = cmap)

ax1.set_title('Subsampled Fraction Insert Size Mode')



by_insert_size_standard_deviation = df.groupby(['upload_date','insert_size_mode']).insert_size_standard_deviation.mean()

by_insert_size_standard_deviation.unstack().plot( ax = ax2 , colormap = cmap)

ax2.set_title('Insert Size Standard Deviation & Mode')

#ax2.legend(bbox_to_anchor=(-1.0, -0.3, 1.7, 0.1), loc=5, prop={'size':12},

 #          ncol=7, mode="expand", borderaxespad=0.)
fig = plt.figure(figsize = (16,12))



ax1 = fig.add_subplot(221)

ax2 = fig.add_subplot(222)

ax3 = fig.add_subplot(212)



cmap = plt.cm.coolwarm_r



reads_after_trimmomatic = df.groupby(['insert_size_mode','reads_after_trimmomatic']).size()

reads_after_trimmomatic.unstack().plot(kind = 'bar', ax = ax1 , stacked = True , colormap = cmap , grid = False)

ax1.set_title('Reads After Trimmomatic Insert Size Mode',fontsize = 14)



reads_after_priceseq = df.groupby(['insert_size_standard_deviation','reads_after_priceseq']).size().unstack().plot(kind = 'bar',ax = ax2, stacked = True, colormap = cmap, grid = False)

ax2.set_title('Reads After Priceseq Insert Standard Deviation', fontsize = 14)



total_ercc_reads = df.groupby(['upload_date', 'reads_after_cdhitdup']).total_ercc_reads.mean().unstack().plot(ax = ax3, colormap = cmap)

ax3.set_title('Reads After Cdhitdup', fontsize = 14)

ax3.set_ylabel('Total ERCC Reads',fontsize = 12)
fig = plt.figure(figsize = (20,10))

ax1 = fig.add_subplot(221)

ax2 = fig.add_subplot(222)

ax3 = fig.add_subplot(212)

sns.countplot(x = 'quality_control', hue = 'subsampled_fraction',data = df , ax = ax1 )

ax1.set_title('Quality control of Subsampled Fraction')



sns.countplot(x = 'quality_control', hue = 'insert_size_min',data = df , ax = ax2 )

ax2.set_title('Quality Control of Insert Size Min')



sns.distplot(df[df.compression_ratio], ax = ax3 , label = 'Compression Ration',color = 'blue')

sns.distplot(df[df.reads_after_trimmomatic] , ax = ax3 , label = 'Reads After Trimmomatic' , color = 'red')



plt.legend()

plt.show()