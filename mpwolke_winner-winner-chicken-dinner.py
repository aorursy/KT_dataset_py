#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcQqFDgSMuSpz3Oz7nDCc8k5vSUhRFMhoHEcUhP9DTuaU7P2WF4k&usqp=CAU',width=400,height=400)
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

import plotly.graph_objs as go

import plotly.offline as py

import plotly.express as px

#plt.style.use('fivethirtyeight')



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_excel('/kaggle/input/pmpl-south-asia-2020-pubg/PMPL South Asia.xlsx')

df.head()
plt.style.use('fivethirtyeight')

df.plot(subplots=True, figsize=(10, 10), sharex=False, sharey=False)

plt.show()
#word cloud

from wordcloud import WordCloud, ImageColorGenerator

text = " ".join(str(each) for each in df.Teams)

# Create and generate a word cloud image:

wordcloud = WordCloud(max_words=200,colormap='Set2', background_color="black").generate(text)

plt.figure(figsize=(10,6))

plt.figure(figsize=(15,10))

# Display the generated image:

plt.imshow(wordcloud, interpolation='Bilinear')

plt.axis("off")

plt.figure(1,figsize=(12, 12))

plt.show()
sns.countplot(df['Placement'],linewidth=3,palette="Set2",edgecolor='black')

plt.show()
from category_encoders import OneHotEncoder

from sklearn.linear_model import LinearRegression, Ridge

from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler



cols_selected = ['Teams']

ohe = OneHotEncoder(cols=cols_selected, use_cat_names=True)

df_t = ohe.fit_transform(df[cols_selected+['Placement']])



#scaler = MaxAbsScaler()

X = df_t.iloc[:,:-1]

y = df_t.iloc[:, -1].fillna(df_t.iloc[:, -1].mean()) / df_t.iloc[:, -1].max()



mdl = Ridge(alpha=0.1)

mdl.fit(X,y)



pd.Series(mdl.coef_, index=X.columns).sort_values().head(10).plot.barh()
df['Total'].hist(figsize=(10,5), bins=20)
sns.countplot(x="Total",data=df,palette="GnBu_d",edgecolor="black")

plt.xticks(rotation=45)

plt.yticks(rotation=45)

# changing the font size

sns.set(font_scale=1)
ax = df['Match Played'].value_counts().plot.barh(figsize=(14, 6))

ax.set_title('Match Played Distribution', size=18)

ax.set_ylabel('Match Played', size=14)

ax.set_xlabel('Total', size=14)
from scipy.stats import norm, skew #for some statistics

import seaborn as sb

from scipy import stats #qqplot

#Lets check the ditribution of the target variable (Placement?)

from matplotlib import rcParams

# figure size in inches

rcParams['figure.figsize'] = 4,2



sb.distplot(df['Placement'], fit=norm)



#Get the QQ-plot

fig = plt.figure()

res = stats.probplot(df['Placement'], plot=plt)

plt.show()
from statsmodels.tsa.statespace.sarimax import SARIMAX

#The data is highly skewed, but since we'll be applying ARIMA, it's fine.

df['Placement'].skew()
#Just in case if there needs to be some transformation, it can be done by either taking log values or using box cox.



## In case you need to normalize data, use Box Cox. Pick the one that looks MOST like a normal distribution.

for i in [1,2,3,4,5,6,7,8]:

    plt.hist(df['Placement']**(1/i), bins= 40, normed=False)

plt.title("Box Cox transformation: 1/{}". format(str(i)))

plt.show()
#Match Playe by order.

df['Match Played'].value_counts().sort_values(ascending = False)
df.groupby('Match Played').sum().sort_values('Placement', ascending = False)
print (len(df['Kills'].value_counts()))



rcParams['figure.figsize'] = 50,14

sb.countplot(df['Kills'].sort_values(ascending = True))



#There's a lot of kills? on beginning of Kills?
#Lets check the orders by warehouse.



#Checking with Boxplots

from matplotlib import rcParams

# figure size in inches

rcParams['figure.figsize'] = 16,4

f, axes = plt.subplots(1, 2)

#Regular Data

fig3 = sb.boxplot( df['Match Played'],df['Placement'], ax = axes[0])

#Data with Log Transformation

fig4 = sb.boxplot( df['Match Played'], np.log1p(df['Placement']),ax = axes[1])



del fig3, fig4
#Lets check the Orders by Product Category.

rcParams['figure.figsize'] = 50,12

#Taking subset of data temporarily for in memory compute.

df_temp = df.sample(n=1000).reset_index()

fig5 = sb.boxplot( df_temp['Kills'].sort_values(),np.log1p(df_temp['Placement']))

del df_temp, fig5
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcQE7yx-KglVbC9ZKWX138nxkqSGpc71cQ6tBfo5Rjgwx3ogzm_P&usqp=CAU',width=400,height=400)