# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from wordcloud import WordCloud, STOPWORDS 

import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns
df=pd.read_csv('../input/HRDataset_v9.csv')



df.head(5)
df.rename(columns=({'Days Employed':'Days_Employed'}),inplace=True)



def GetYears(Days_Employed):

    if ( Days_Employed >= 0 and Days_Employed <= 365  ):

        return "1 Year"

    if ( Days_Employed >= 365 and Days_Employed <= 730 ):

        return "2 Year"

    if ( Days_Employed >= 730 and Days_Employed <= 1095 ):

        return "3 Year"

    if ( Days_Employed >= 1095 and Days_Employed <= 1460 ):

        return "4 Year"

    if ( Days_Employed >= 1460 and Days_Employed <= 1825 ):

        return "5 Year"

    else: 

        return 'More than 5 Year'



df['Years_With_Company'] = df.apply(lambda x : GetYears(x['Days_Employed']), axis=1)



df.Years_With_Company.value_counts()



df.head()
plt.figure(figsize=(15,8))

sns.stripplot(x='Age',y='Days_Employed',hue='Years_With_Company',data=df)

plt.xlabel('Age')

plt.ylabel('Years_With_Company')

plt.title('Age vs Years_With_Company')

plt.show()
# to see if there are no Nulls in the dataset

df.isnull().sum()
# to check the type of dataset and to make sure that it makes sense

df.dtypes
#I have Groupby the Position to analyze the number of employees in the each respective Position

count=df.groupby(df["Position"]).count()

count = pd.DataFrame(count.to_records())

#count = count.sort_values(by= 'left', ascending = False)

count = count['Position']

count
plt.figure(figsize=(15,10))

sns.countplot(y='Position', data=df,order=count )
#A large number of People meets the Annual Performance Expectation

plt.figure(figsize=(15,5))

sns.countplot(x="Performance Score",data=df)

plt.xticks(rotation=45)
#It seems that Pay rate is equal between  genders and it appears to be no discrimnation 

sns.catplot(x="GenderID",y="Pay Rate",hue="Sex",data=df ,legend_out =True)

plt.legend(loc=0)

plt.figure(figsize=(25,20))



dfcorr=df[['GenderID','Age','Pay Rate']]

corr=dfcorr.corr()

plt.figure(figsize=(5,5))

sns.heatmap(corr,vmax=0.6,annot=True)

plt.xticks(rotation=45)

plt.yticks(rotation=90)
#Count the frquency of gender 

ax=sns.barplot(x=df['Sex'].value_counts().index,y=df['Sex'].value_counts().values,palette="Blues_d",hue=['Female','Male'])

plt.legend(loc=8)

plt.xlabel('Gender')

plt.ylabel('Frequency')

plt.title('Show of Employees by Gender')

plt.xticks(rotation=45)

plt.show()

#This graph shows the Pay rate per Gender and Race

plt.figure(figsize=(20,10))

ax = sns.violinplot(x="Sex", y="Pay Rate",data=df,hue="RaceDesc")





plt.legend(loc=8)

plt.xlabel('Gender')

plt.ylabel('Pay Rate')

plt.title('Show of PayRate by Race')

plt.xticks(rotation=45)

plt.show()
plt.figure(figsize=(5,10))

sns.stripplot(data=df, x="Pay Rate",y="Position",hue="Sex",jitter=True,dodge=True,palette="Set2",)

plt.legend(loc=0)

plt.xlabel('Pay Rate')

plt.ylabel('Position')

plt.title('Pay Rate per Position')

plt.yticks(rotation=45)

plt.show()

from wordcloud import WordCloud, STOPWORDS

all_locations = ','.join(df['Employee Name'].values)



wordcloud = WordCloud(width=1500, height=800).generate(all_locations)



plt.figure(figsize=(20, 8))

# Display the generated image:

plt.imshow(wordcloud, interpolation='bilinear')

plt.axis("off")

plt.show()