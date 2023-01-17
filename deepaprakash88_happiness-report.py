#import the necessary libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

from sklearn.preprocessing import StandardScaler

import missingno as msno
#Set style
plt.style.use('fivethirtyeight')
#Read the csv file
df = pd.read_csv('../input/2015.csv')
#Check the contents
df.head()

df.columns
df.info()
df.describe()
msno.bar(df,color= sns.color_palette('muted'))
happy_cntry = df.groupby(['Happiness Rank','Country','Economy (GDP per Capita)'])['Economy (GDP per Capita)'].agg(['mean'],index = False).sort_values(by= 'mean',ascending = False)[:20]
happy_cntry.style.set_table_styles([{'selector': 'tr:hover', 'props': [('background-color', '')]}])
happy_cntry.style.highlight_max(color = 'green')
fig = plt.figure(figsize = (14,5))
cnt = df.groupby(['Happiness Rank'],as_index=False).max()['Economy (GDP per Capita)'].sort_values(ascending = False).to_frame()[:50]
sns.pointplot(y = 'Economy (GDP per Capita)', x = cnt.index,data = cnt,markers=['*'],scale= 0.4,color='green',linestyles='--')
plt.ylabel("Average Economy",rotation = 'horizontal',horizontalalignment = 'right')
plt.xlabel("Happiness Rank")

fig = plt.figure(figsize = (12,4))
cnt = df.groupby(['Happiness Rank'],as_index=False).max()['Family'].sort_values(ascending = False).to_frame()[:50]
sns.pointplot(y = 'Family', x = cnt.index,data = cnt,markers='*',scale = 0.5,color='blue')
plt.xlabel("Happiness Rank")
plt.ylabel("Family",rotation = 'horizontal',horizontalalignment = 'right',fontsize = 14)

happy_cntry = df.groupby(['Happiness Rank','Country','Family'])['Family'].agg(['max'],index = False).sort_values(by= 'max',ascending = False)[:20]
happy_cntry.style.set_table_styles([{'selector': 'tr:hover', 'props': [('background-color', '')]}])
happy_cntry.style.highlight_max(color = 'green')
fig = plt.figure(figsize = (12,4))
cnt = df.groupby(['Happiness Rank'],as_index=False).max()['Freedom'].sort_values(ascending = False).to_frame()[:30]
sns.pointplot(y = 'Freedom', x = cnt.index,data = cnt,markers=['*'],scale=0.7,palette='Blues_d')
plt.xlabel("Happiness Rank")
plt.ylabel("Freedom",rotation = 'horizontal',horizontalalignment = 'right',fontsize = 14)
free_cntry = df.groupby(['Happiness Rank','Country','Freedom'])['Freedom'].agg(['max']).sort_values(by= 'max',ascending = False)[:20]
free_cntry.style.highlight_max()
trust_cntry = df.groupby(['Happiness Rank','Country','Trust (Government Corruption)'])['Trust (Government Corruption)'].agg(['max']).sort_values(by= 'max',ascending = False)[:30]
trust_cntry.style.highlight_max()
df[df['Country'] == 'Rwanda']
fig = plt.figure(figsize = (12,4))
trust_cntry = df.groupby(['Happiness Rank'],as_index=False).max()['Trust (Government Corruption)'].sort_values(ascending = False).to_frame()[:30]
sns.pointplot(y = 'Trust (Government Corruption)', x = trust_cntry.index,data = trust_cntry,scale = 0.6,markers = 'v',palette='inferno')
plt.xlabel("Happiness Rank")
plt.ylabel("Trust (Government Corruption)",rotation = 'horizontal',horizontalalignment = 'right',fontsize = 14)
generous_cntry = df.groupby(['Happiness Rank','Country','Generosity'])['Generosity'].agg(['max']).sort_values(by= 'max',ascending = False)[:30]
generous_cntry.style.highlight_max()
fig = plt.figure(figsize = (12,4))
gen = df.groupby(['Happiness Rank'],as_index=False).max()['Generosity'].sort_values(ascending = False).to_frame()[:30]
sns.pointplot(y = 'Generosity', x = gen.index,data = gen,scale = 0.5,markers = '.',palette='inferno')
plt.xlabel("Happiness Rank")
plt.ylabel("Generosity",rotation = 'horizontal',horizontalalignment = 'right',fontsize = 14)
fig = plt.figure(figsize = (12,4))
trust_cntry = df.groupby(['Happiness Rank'],as_index=False).max()['Health (Life Expectancy)'].sort_values(ascending = False).to_frame()[:30]
sns.pointplot(y = 'Health (Life Expectancy)', x = trust_cntry.index,data = trust_cntry,scale = 0.5,markers = '*',palette='inferno')
plt.xlabel("Happiness Rank")
plt.ylabel("Health (Life Expectancy)",rotation = 'horizontal',horizontalalignment = 'right',fontsize = 14)
fig = plt.figure(figsize=(8,6))
region_famft = df.groupby(['Region']).mean()['Happiness Score'].sort_values(ascending = False).to_frame()[:15]
sns.barplot(x = region_famft['Happiness Score'],y = region_famft.index,data = region_famft,palette= 'rainbow')
plt.xlabel('Happiness Score',fontsize = 14)
plt.ylabel('Region',rotation = 'horizontal',fontsize = 14)

fig = plt.figure(figsize = (12,4))
trust_cntry = df.groupby(['Happiness Rank'],as_index=False).max()['Dystopia Residual'].sort_values(ascending = False).to_frame()[:30]
sns.pointplot(y = 'Dystopia Residual', x = trust_cntry.index,data = trust_cntry,scale = .5,markers = '*',palette='inferno')
plt.xlabel("Happiness Rank")
plt.ylabel("Dystopia Residual",rotation = 'horizontal',horizontalalignment = 'right',fontsize = 14)
X = df.drop(['Country','Region','Happiness Rank'],axis = 1)
y = df['Happiness Rank']
#Lets normalise the values 
ss = StandardScaler()
ss.fit_transform(X)
X.head(5)


from sklearn.cluster import KMeans
model = KMeans(n_clusters=4)
pred_val = model.fit_predict(X)
X['Res'] = pd.DataFrame(pred_val)
plt.scatter(X['Family'],X['Health (Life Expectancy)'] , c=pred_val,cmap='rainbow')












































































































































































































































































