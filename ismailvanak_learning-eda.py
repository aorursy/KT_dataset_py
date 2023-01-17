%matplotlib inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import os
os.chdir('../input/')
google_file = pd.read_csv("googleplaystore.csv")
google_user_review = pd.read_csv("googleplaystore_user_reviews.csv")
#google_file.info()
#google_file.head()

ndup = google_file.drop_duplicates(subset = 'App')

ndup = ndup[ndup['Android Ver'] != np.nan]
ndup = ndup[ndup['Android Ver'] != 'NaN']
ndup = ndup[ndup.Installs != 'Free']
ndup = ndup[ndup.Installs != 'Paid']

#print('Number of apps in the dataset : ' , len(ndup))
#ndup.sample(7)

ndup.reset_index(inplace=True)
ndup.Installs = ndup.Installs.apply(lambda x: x.replace('+','') if '+' in str(x) else x) 
ndup.Installs = ndup.Installs.apply(lambda x: x.replace(',','') if ',' in str(x) else x) 
ndup.Installs = ndup.Installs.apply(lambda x: int(x)) 

ndup.Size = ndup.Size.apply(lambda x: x.replace('Varies with device','NaN') if 'Varies with device' in str(x) else x)  
ndup.Size = ndup.Size.apply(lambda x: x.replace(',','') if ',' in str(x) else x)  
ndup.Size = ndup.Size.apply(lambda x: float(x.replace('k',''))/1024 if 'k' in str(x) else x)
ndup.Size = ndup.Size.apply(lambda x: str(x).replace('M',''))

ndup.Size = ndup.Size.apply(lambda i: float(i))

ndup['Installs'] = ndup['Installs'].apply(lambda x: float(x))

ndup['Price'] = ndup['Price'].apply(lambda x: str(x).replace('$', '') if '$' in str(x) else str(x))
ndup['Price'] = ndup['Price'].apply(lambda x: float(x))

ndup['Reviews'] = ndup['Reviews'].apply(lambda x: int(x))
datatesting = ndup[['Rating','Size','Installs','Reviews','Type']].dropna()
datatesting['Installs'] = datatesting['Installs'].apply(lambda x: np.log(x))
datatesting['Reviews'] = datatesting['Reviews'].apply(lambda x: np.log(x))
#datatesting.sample(15)
#plt.style.use('ggplot')
import seaborn as sns # for making plots with seaborn
p = sns.pairplot(datatesting, hue='Type')
num_app_per_cat = ndup['Category'].value_counts().sort_values(ascending=True)

#patches, texts = plt.pie(list(num_app_per_cat.values),labels=list(num_app_per_cat.index))

patches, texts = plt.pie(num_app_per_cat.values)
plt.legend(patches, num_app_per_cat.index, loc="center left",bbox_to_anchor=(1, 0, 0.5, 1))
plt.show()
ax = sns.jointplot(ndup['Size'], ndup['Rating'])
corrmat = datatesting.corr()
#f, ax = plt.subplots()
sns.set(rc={'figure.figsize':(25,15)})

p =sns.heatmap(corrmat, annot=True, cmap=sns.diverging_palette(220, 20, as_cmap=True))
import scipy.stats as stats
f = stats.f_oneway(ndup.loc[ndup.Category == 'BUSINESS']['Rating'].dropna(), 
               ndup.loc[ndup.Category == 'FAMILY']['Rating'].dropna(),
               ndup.loc[ndup.Category == 'GAME']['Rating'].dropna(),
               ndup.loc[ndup.Category == 'PERSONALIZATION']['Rating'].dropna(),
               ndup.loc[ndup.Category == 'LIFESTYLE']['Rating'].dropna(),
               ndup.loc[ndup.Category == 'FINANCE']['Rating'].dropna(),
               ndup.loc[ndup.Category == 'EDUCATION']['Rating'].dropna(),
               ndup.loc[ndup.Category == 'MEDICAL']['Rating'].dropna(),
               ndup.loc[ndup.Category == 'TOOLS']['Rating'].dropna(),
               ndup.loc[ndup.Category == 'PRODUCTIVITY']['Rating'].dropna()
              )

print(f)
print('\nThe p-value is extremely small, hence we reject the null hypothesis in favor of the alternate hypothesis.\n')
