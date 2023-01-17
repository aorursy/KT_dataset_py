#importing required packages
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from wordcloud import *
import wordcloud
import os
#reading files
F1=pd.read_csv("../input/AppleStore.csv")
F2=pd.read_csv("../input/appleStore_description.csv")
F1.head()
#change the column name to the convenient form
F1.rename(columns=({'id':'App ID','track_name':'App Name','size_bytes':'Size(Byt)','currency':'Curr Typ',
                    'price':'Price amt','rating_count_tot':'Usr Ratg cnt(all ver)',
                    'rating_count_ver':'Usr Ratg count(cur ver)',"user_rating":'Avg Usr Ratg(all ver)',
                    "user_rating_ver":'Avg Usr Ratg value(cur ver)',"ver":'Latest ver code',
                   "cont_rating":'Content Ratg',"prime_genre":'Primary Genre',"sup_devices.num":'No of supp devices',
                    "ipadSc_urls.num":'No of scrsht showed for display',"lang.num":'No of supp lang',"vpp_lic":'Dev License'}),
          inplace=True)
F1.columns
F1.info()
F2.head()
F2.rename(columns=({'id':'App ID','track_name':'App name','size_bytes':'Size(Byt)','app_desc':'App desc'}),inplace=True)
F2.columns
F2.info()
#merging files F1 and F2
F=pd.merge(F1,F2,how="left",on='App ID')
F.head()
#dropping the unused variable and duplicate variable
F=F.drop(['Unnamed: 0','Size(Byt)_y','App name'],axis=1)
#seperating category data and number data
Fchar=F.select_dtypes(include=['object'])
Fnum=F.select_dtypes(include=['number'])
Fnum.head()
Fnum.describe()
# the Fnum data also contains some factor variables so drop it.
Fnum1=Fnum.drop(['App ID','No of scrsht showed for display','Avg Usr Ratg(all ver)',
                 'Avg Usr Ratg value(cur ver)','Dev License'],axis=1)
#generating histogram of all num variable
Fnum1hist=Fnum1.hist(color='purple', edgecolor='red',figsize=(10,10))
#finding skewness n kurtosis for each variable
s=Fnum1.skew()
K=Fnum1.kurt()
sk=pd.DataFrame({'skewness':s,'kurtosis':K})
sk
#generating pairplot
sns.pairplot(Fnum1)
Fchar[['No of scrsht showed for display','Avg Usr Ratg(all ver)','Avg Usr Ratg value(cur ver)',
       'Dev License']]=Fnum[['No of scrsht showed for display','Avg Usr Ratg(all ver)',
                             'Avg Usr Ratg value(cur ver)','Dev License']]
Fchar.head()
#generating value counts for each variable
A=Fchar['App Name'].value_counts()
B=Fchar['Curr Typ'].value_counts()
C=Fchar['Latest ver code'].value_counts()
D=Fchar['Content Ratg'].value_counts()
E=Fchar['Primary Genre'].value_counts()
G=Fchar['No of scrsht showed for display'].value_counts()
H=Fchar['Avg Usr Ratg(all ver)'].value_counts()
I=Fchar['Avg Usr Ratg value(cur ver)'].value_counts()
J=Fchar['Dev License'].value_counts()
# plot for cont_rating....
Dplt=D.plot.bar(title='Content Rating')
Dplt.set_xlabel('Age categy',size=15)
Dplt.set_ylabel('count',size=15)
#plot for different genres
Eplt=E[:5].plot.bar(title='top 5 apps genre')
Eplt.set_xlabel('genre',size=15)
Eplt.set_ylabel('count',size=15)
# avg user rating for all version apps
Hplt=H.plot.bar(title='user rating for curr ver',figsize=(10,5))
Hplt.set_xlabel('rating',size=15)
Hplt.set_ylabel('count',size=15)
# avg user rating for current version apps
Gplt=G.plot.bar(title='user rating for all ver',figsize=(10,5))
Gplt.set_xlabel('rating',size=15)
Gplt.set_ylabel('count',size=15)
#now combining char and num data
All=pd.merge(Fnum,Fchar)
All.head()
#subsetting char data belongs to gaming
Games=All.loc[All['Primary Genre'] == 'Games']
# wordcloud for games
A=Games['App Name'].str.cat(sep=' ')
# Create the wordcloud object
wordcloud = WordCloud(width=800, height=480, margin=0).generate(A)
 
# Display the generated image:
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.margins(x=0, y=0)
plt.show()