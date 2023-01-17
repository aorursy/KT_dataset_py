# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import Imputer #tool to replace missing value
import matplotlib.pyplot as plt #plotting charts
import seaborn as sns #plotting good-looking charts
import cufflinks as cf #plotting interative charts
from plotly.offline import download_plotlyjs,init_notebook_mode,plot,iplot 
init_notebook_mode(connected=True) #connect the javescript to the notebook
cf.go_offline() #allow using cufflinks offline


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


#Load Cean Data
df = pd.read_csv('../input/Clean_googleplaystore.csv')
del df['Unnamed: 0']
#display the first five records
df.head()
#delete duplicates
df.drop_duplicates(subset ="App", 
                keep = 'first', inplace = True) 
#check out number of records and attributes
df.shape
#draw a boxplot map to observe app's price among different categories
sns.set_style('whitegrid')

f, ax = plt.subplots(figsize=(17, 13))
sns.boxplot(x="Price", y="Category", data=df,palette="vlag")
plt.title("App's Price by Category", fontsize = '17')
plt.ylabel('Category',fontsize = '14')
plt.xlabel('Price',fontsize = '14')
#identify apps with extremely high price
df[df.Price>250][['App','Category','Price']]
#remove outliners
df = df[df.Price<=250]
#check out number of records and attributes
df.shape
#draw a boxplot map to observe app's price among different categories
sns.set_style('whitegrid')

f, ax = plt.subplots(figsize=(17, 13))
sns.boxplot(x="Price", y="Category", data=df,palette="vlag")
plt.title("App's Price by Category", fontsize = '17')
plt.ylabel('Category',fontsize = '14')
plt.xlabel('Price',fontsize = '14')
#identify these comparably expensive apps
df[df.Price>50]
#remove outliners
df = df[df.Price<=100]
#draw a boxplot map to observe app's price among different categories
sns.set_style('whitegrid')

f, ax = plt.subplots(figsize=(17, 13))
sns.boxplot(x="Price", y="Category", data=df[df.Type =='Paid'],palette="vlag")
plt.title("Non-free App's Price by Category", fontsize = '17')
plt.ylabel('Category',fontsize = '14')
plt.xlabel('Price',fontsize = '14')
#display the Price stats by Category for all paid apps
df[df.Type == 'Paid'].groupby('Category').Price.describe()
#draw a bar chart showing the number of apps by each category
f, ax = plt.subplots(figsize=(17, 13))
sns.countplot(y="Category", hue ='Type',data=df,palette="Set2")
plt.title("Non-free App's Price by Category", fontsize = '17')
plt.ylabel('Category',fontsize = '14')
plt.xlabel('Price',fontsize = '14')
df[df.Type == 'Paid'].describe()
df.describe()
'''labels = df.groupby('Category').App.count().index

values = df.groupby('Category').App.count().values

test = pd.DataFrame(values, index=labels, columns=['x'])
 
# make the plot
test.plot(kind='pie', subplots=True, figsize=(8, 8))'''
#draw a histogram showing number of apps per category
sns.set_style('whitegrid')
sns.set(font_scale=1)
fig, ax = plt.subplots(figsize=(12,8))
ax = sns.countplot(y=df['Category'],order = df['Category'].value_counts().index)
plt.title("Number of Apps by Category", fontsize = '17')
plt.ylabel('Category',fontsize = '14')
plt.xlabel('Number of Apps',fontsize = '14')
#draw a boxplot map to observe app's ratings among different categories
f, ax = plt.subplots(figsize=(17, 13))
sns.boxplot(x="Rating", y="Category", data=df,palette="vlag",order = df['Category'].value_counts().index)
plt.title("Boxplot of Ratings by Category", fontsize = '17')
plt.ylabel('Category',fontsize = '14')
plt.xlabel('Rating',fontsize = '14')
#draw a boxplot map to observe app's review counts among different categories
f, ax = plt.subplots(figsize=(17, 13))
sns.boxplot(x="Reviews", y="Category", data=df,palette="vlag",order = df['Category'].value_counts().index)
plt.title("Boxplot of Review Counts by Category", fontsize = '17')
plt.ylabel('Category',fontsize = '14')
plt.xlabel('Number of Reviews',fontsize = '14')
#display Reviews stats
df.Reviews.describe()
#create a new attribute recording number of reviews after log-scaling
df['Reviews_Log'] = df.Reviews.apply(lambda x:np.log(x+1))
#draw a boxplot map to observe app's review counts (log-scale) among different categories
f, ax = plt.subplots(figsize=(17, 13))
sns.boxplot(x="Reviews_Log", y="Category", data=df,palette="vlag",order = df['Category'].value_counts().index)
plt.title("Boxplot of Review (log-scale) Counts by Category", fontsize = '17')
plt.ylabel('Category',fontsize = '14')
plt.xlabel('Number of Reviews (log-scale)',fontsize = '14')
#create a dictionary to record Installs in ascending order
INSTALL = {
    0: '0',1: '0+', 2: '1+',3: '5+',4: '10+', 5: '50+',6: '100+',
    7: '500+',8: '1,000+', 9: '5,000+',10: '10,000+',11: '50,000+', 
    12: '100,000+',13: '500,000+',14: '1,000,000+',15: '5,000,000+', 
    16: '10,000,000+',17: '50,000,000+',18: '100,000,000+', 
    19: '500,000,000+',20: '1,000,000,000+'
}

#create a table contains intall frequency and cumulative frequency for plotting purpose
installs_cum = pd.DataFrame(data={'Install': df.groupby('Installs_Num').App.count().index, 
                                  'Freq': df.groupby('Installs_Num').App.count().values})
installs_cum['CumFreq'] = installs_cum['Freq'].cumsum()

installs_cum_paid = pd.DataFrame(data={'Install': df[df.Type == 'Paid'].groupby('Installs_Num').App.count().index, 
                                  'Freq': df[df.Type == 'Paid'].groupby('Installs_Num').App.count().values})
installs_cum_paid['CumFreq'] = installs_cum_paid['Freq'].cumsum()

installs_cum_free = pd.DataFrame(data={'Install': df[df.Type == 'Free'].groupby('Installs_Num').App.count().index, 
                                  'Freq': df[df.Type == 'Free'].groupby('Installs_Num').App.count().values})
installs_cum_free['CumFreq'] = installs_cum_free['Freq'].cumsum()
#plot the cumulative counts of intalls
sns.set_style('dark')
fig, ax = plt.subplots(figsize=(12,8))
ax = sns.lineplot(x="Install", y="CumFreq", linewidth = '2', color = 'orange', data=installs_cum, label = 'Total')
ax = sns.lineplot(x="Install", y="CumFreq", linewidth = '2', color = 'green', data=installs_cum_paid, label = 'Paid')
ax = sns.lineplot(x="Install", y="CumFreq", linewidth = '2', color = 'red', data=installs_cum_free, label = 'Free')

bars = ['0', '1', '2', '5', '10', '50', '100', 
         '500', '1,000', '5,000', '10,000', '50,000', 
         '100,000', '500,000', '1,000,000', '5,000,000', 
         '10,000,000', '50,000,000', '100,000,000', '500,000,000', 
         '1,000,000,000']

y_pos = np.arange(len(bars))
plt.xticks(y_pos, bars, rotation=90, fontsize='13', horizontalalignment='center')
plt.title("Cumulative Counts of APP's Installs", fontsize = '17')
plt.ylabel('Cumulative Frequency',fontsize = '14')
plt.xlabel('Number of Installs',fontsize = '14')
ax.grid(b=True, which='major')

plt.show()
#draw a boxplot map to observe app's install counts (log-scale) among different categories
f, ax = plt.subplots(figsize=(17, 13))
sns.boxplot(x="Installs_Num", y="Category", data=df,palette="vlag",order = df['Category'].value_counts().index)
plt.title("Boxplot of Intalls Counts (Log-Scale) by Category", fontsize = '17')
plt.ylabel('Category',fontsize = '14')
plt.xlabel('Number of Installs (Log-Scale)',fontsize = '14')
#count how many entertainment apps there are
df[df.Category == 'ENTERTAINMENT'].shape[0]
#Check out the app with no installs
df[df.Installs_Num == 0]
#draw a boxplot map to observe app's size among different categories
f, ax = plt.subplots(figsize=(17, 13))
sns.boxplot(x="Size", y="Category", data=df,palette="vlag",order = df['Category'].value_counts().index)
plt.title("Boxplot of App's Size by Category", fontsize = '17')
plt.ylabel('Category',fontsize = '14')
plt.xlabel('Size',fontsize = '14')
#draw a boxplot map to observe app's word counts among different categories
f, ax = plt.subplots(figsize=(17, 13))
sns.boxplot(x="Name_Word_Counts", y="Category", data=df,palette="vlag",order = df['Category'].value_counts().index)

plt.title("Boxplot of Number of Words in App's Name by Category", fontsize = '17')
plt.ylabel('Category',fontsize = '14')
plt.xlabel('Number of Words',fontsize = '14')
#display the apps with more than 10 words in their names but with less than 50 reviews. 
#check if there's any outliners
df[(df.Name_Word_Counts > 10) & (df.Reviews < 2)]
#remove the two outliners
df = df[(df.Name_Word_Counts <= 10) | ((df.Name_Word_Counts > 10) & (df.Reviews >= 2))]
#check out the number of records
df.shape
#draw a boxplot map to observe app's character counts among different categories
f, ax = plt.subplots(figsize=(17, 13))
sns.boxplot(x="Name_Length", y="Category", data=df,palette="vlag",order = df['Category'].value_counts().index)

plt.title("Boxplot of Number of Characters in App's Name by Category", fontsize = '17')
plt.ylabel('Category',fontsize = '14')
plt.xlabel('Number of Characters (including spaces & symbols)',fontsize = '14')
#check out the 3 apps with extremely long names. Nothing looks odd.
df[df.Name_Length > 110]
#draw a heatmaps showing correlation among all numerical variables
#I chose the spearman correlation method so that the correlation value won't change under any scaling transformation
f, ax = plt.subplots(figsize=(10, 10))
corr = df.corr(method='spearman')
sns.heatmap(corr, cmap="coolwarm",
            square=True, ax=ax,annot=True)
plt.title("Correlation Matrix", fontsize = '17')
#set a threhold as 0.1 and only display the correlation beyond the bars.
fig = plt.figure(figsize = (20,40))
ax1 = fig.add_subplot(1, 2, 1) # row, column, position
ax2 = fig.add_subplot(1, 2, 2)
corr = df[['Rating','Size','Price','Name_Word_Counts','Name_Length','Installs_Num','Reviews_Log']].corr(method='spearman')

sns.heatmap(corr[abs(corr)>0.1], ax =ax1,cmap="coolwarm",cbar_kws={'shrink': .2},
            square=True,annot=True)
ax1.set_title("Correlation Matrix", fontsize = '17')

corr1 = df[df.Installs_Num>=17].corr(method='spearman')
sns.heatmap(corr1[abs(corr)>0.1], ax = ax2, cmap="coolwarm",cbar_kws={'shrink': .2},
            square=True, annot=True)
ax2.set_title("Correlation Matrix (installs>50k)", fontsize = '17')
#plot a table showing correlation among all numerical variables
#cmap = "coolwarm"
#corr.style.background_gradient(cmap, axis=1)
#check out the correlation matrix for data set with more than x intalls.
#f, ax = plt.subplots(figsize=(10, 10))
#corr1 = df[df.Installs_Num>10].corr(method='spearman')
#ax1 = sns.heatmap(corr1[abs(corr)>0.1], cmap="coolwarm",
#            square=True, ax=ax,annot=True)
#ax1.set_title("Correlation Matrix", fontsize = '17')
#plotting the correlation matrix for mapps in 8 categories.
fig = plt.figure(figsize = (20,40))
CAT = ['FAMILY', 'GAME','TOOLS','BUSINESS','MEDICAL','COMMUNICATION','DATING','FOOD_AND_DRINK']
df_corr_Install = pd.DataFrame(corr.Installs_Num)
df_corr_Reviews = pd.DataFrame(corr.Reviews_Log)
df_corr_Rating = pd.DataFrame(corr.Rating)
for i in range(8):
    ax = fig.add_subplot(4, 2, i+1)
    corr = df[df['Category'] == CAT[i]][['Rating','Size','Price','Name_Word_Counts','Name_Length','Installs_Num','Reviews_Log']].corr(method='spearman')
    sns.heatmap(corr[abs(corr)>0.1], ax =ax,cmap="coolwarm",cbar_kws={'shrink': .5},square=True,annot=True)
    ax.set_title("Correlation Matrix - " + CAT[i], fontsize = '17')
    df_corr_Install[CAT[i]] = corr.Installs_Num
    df_corr_Reviews[CAT[i]] = corr.Reviews_Log
    df_corr_Rating[CAT[i]] = corr.Rating
del df_corr_Install['Installs_Num']
del df_corr_Reviews['Reviews_Log']
del df_corr_Rating['Rating']
#correlation table in terms of Number of Installs
df_corr_Install.style.background_gradient('coolwarm', axis=1)
#correlation table in terms of Rating
df_corr_Rating.style.background_gradient('coolwarm', axis=1)
df_paid = df[df.Type == 'Paid']
df_paid['Category'].value_counts()
#draw a histogram showing number of paid apps per category
sns.set_style('whitegrid')
sns.set(font_scale=1)
fig, ax = plt.subplots(figsize=(12,8))
ax = sns.countplot(y=df_paid['Category'],order = df_paid['Category'].value_counts().index)
plt.title("Number of Paid Apps by Category", fontsize = '17')
plt.ylabel('Category',fontsize = '14')
plt.xlabel('Number of Paid Apps',fontsize = '14')
list_paid = df_paid['Category'].value_counts()>=10
list_paid
#create a list for category with more than 10 paid apps
list_paid = ['FAMILY', 'GAME', 'MEDICAL', 'PERSONALIZATION', 'TOOLS',
       'BOOKS_AND_REFERENCE', 'COMMUNICATION', 'PRODUCTIVITY', 'SPORTS',
       'PHOTOGRAPHY', 'HEALTH_AND_FITNESS', 'LIFESTYLE', 'BUSINESS',
       'TRAVEL_AND_LOCAL', 'FINANCE']
df_price = df[df['Category'].isin(list_paid)]
#remove catogories with less than 10 paid apps
df_paid = df_paid[df_paid['Category'].isin(list_paid)]
df_free = df[df.Type == 'Free']
#build correlation table for paid and free apps
df_paid_Install = pd.DataFrame(corr.Installs_Num)
for i in range(15):
    corr_paid = df_paid[df_paid['Category'] == list_paid[i]][['Rating','Size','Price','Name_Word_Counts','Name_Length','Installs_Num','Reviews_Log',]].corr(method='spearman')
    df_paid_Install[list_paid[i]] = corr_paid.Installs_Num
del df_paid_Install['Installs_Num']
df_paid_Install = df_paid_Install.T
del df_paid_Install['Installs_Num']

df_free_Install = pd.DataFrame(corr.Installs_Num)
for i in range(15):
    corr_free = df_free[df_free['Category'] == list_paid[i]][['Rating','Size','Price','Name_Word_Counts','Name_Length','Installs_Num','Reviews_Log',]].corr(method='spearman')
    df_free_Install[list_paid[i]] = corr_free.Installs_Num
del df_free_Install['Installs_Num']
df_free_Install = df_free_Install.T
del df_free_Install['Installs_Num']
del df_free_Install['Price']
df_paid_Install.style.background_gradient('coolwarm', axis=1)
#draw a histogram showing number of paid apps per category
sns.set_style('whitegrid')
sns.set(font_scale=1)
fig, ax = plt.subplots(figsize=(12,8))
df_game= df[(df.Category == 'GAME') | (df.Category == 'FAMILY')]
ax = sns.countplot(y=df_game['Genres'],order = df_game['Genres'].value_counts().index)
plt.title("Number of Family/Game Apps by Genre", fontsize = '17')
plt.ylabel('Category',fontsize = '14')
plt.xlabel('Number of Family/Game Apps',fontsize = '14')
df_game.Installs_Num.mean()
#draw a boxplot map to observe family/game app's install counts (log-scale) among different genres
f, ax = plt.subplots(figsize=(17, 13))
sns.boxplot(x="Installs_Num", y="Genres", data=df_game,palette="vlag",order = df_game['Genres'].value_counts().index)
plt.title("Boxplot of Intalls Counts (Log-Scale) by Genres for Family/game apps", fontsize = '17')
plt.ylabel('Category',fontsize = '14')
plt.xlabel('Number of Installs (Log-Scale)',fontsize = '14')
#draw a boxplot map to observe family/game app's size among different genres
f, ax = plt.subplots(figsize=(17, 13))
sns.boxplot(x="Size", y="Genres", data=df_game,palette="vlag",order = df_game['Genres'].value_counts().index)
plt.title("Boxplot of App's Size by Genres for Family/game apps", fontsize = '17')
plt.ylabel('Category',fontsize = '14')
plt.xlabel('Size',fontsize = '14')
df_game.Size.mean()
#df_paid_Install.style.background_gradient('coolwarm', axis=1)
#df_free_Install.style.background_gradient('coolwarm', axis=1)
df.to_csv('GooglePlayStoreApp_EDA.csv')
df_price.to_csv('GooglePlayStoreApp_EDA_PaidvsFree.csv')
