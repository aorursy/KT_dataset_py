# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

#import os

#print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.


data2=pd.read_csv('../input/google-play-store-apps/googleplaystore_user_reviews.csv')

data=pd.read_csv('../input/google-play-store-apps/googleplaystore.csv')
from IPython.display import Image

import os

!ls ../input/



Image("../input/images1/rating-system.jpg")
data.info()
print( data.Type.unique() )



print( data.Category.unique() )



data.describe()
print( data2.info() )



print( data2.Sentiment.unique())



data2.describe()
data.corr()

data2.corr()


f,ax=plt.subplots( figsize=(5,5))

sns.heatmap( data.corr(),annot=True,linewidths=.5,fmt='.1f',ax=ax)



plt.show()
f,ax=plt.subplots( figsize=(5,5))

sns.heatmap( data2.corr(),annot=True,linewidths=.5,fmt='.1f',ax=ax)

plt.show()
data.head(10)
data2.head(10)
print( data.columns+'\n' )

data2.columns
plt.figure( figsize=( 10,10))

data.Rating.plot( kind='line' ,color='green' ,label='Rating' ,linewidth=1,alpha=0.5,grid=True,linestyle=':')

plt.legend(loc='upper right',labelspacing=0.5)

plt.xlabel('x axis')

plt.ylabel('y axis')

plt.title('Line plot of Ratings')

plt.figure( figsize=( 10,10))

data2.Sentiment_Polarity.plot( kind='line' ,color='pink' ,label='Sentiment_Polarity' ,linewidth=1,grid=True,linestyle=':')

data2.Sentiment_Subjectivity.plot( kind='line' ,color='purple' ,label='Sentiment_Subjectivity' ,alpha=0.5,linewidth=1,grid=True,linestyle=':')

plt.legend(loc='upper right',labelspacing=0.5)

plt.xlabel('x axis')

plt.ylabel('y axis')

plt.title('Line plot of Ratings')


#plt.figure( figsize=( 20,20)) also we can use this before plotting the barh

fig, ax = plt.subplots(figsize=(20, 10)) 



for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +

              ax.get_xticklabels() + ax.get_yticklabels()):

    item.set_fontsize(10)



plt.barh( data.App[:15],data.Rating[:15],align='center',orientation='horizontal',color='pink',edgecolor='purple',linewidth=2)



plt.title("Application-Rating")

plt.xlabel("Ratings")

plt.ylabel("Applications")

plt.legend()

plt.show()
data_updated=data.rename( index=str ,columns={"Content Rating":"Content_Rating"})



print( data_updated.Content_Rating.unique())

plt.figure( figsize=( 10,10))





content_rating_Everyone=data_updated[ data_updated.Content_Rating	=='Everyone' ]

content_rating_Teen=data_updated[ data_updated.Content_Rating=='Teen' ]

content_rating_Every10=data_updated[ data_updated.Content_Rating=='Everyone 10+' ]

content_rating_Mature17=data_updated[ data_updated.Content_Rating=='Mature 17+' ]

content_rating_Adults=data_updated[ data_updated.Content_Rating=='Adults only 18+' ]

content_rating_Unrated=data_updated[ data_updated.Content_Rating=='Unrated' ]



plt.scatter( content_rating_Everyone.Rating, pd.to_numeric(content_rating_Everyone.Reviews),color='red',label='EveryOne')

plt.scatter( content_rating_Teen.Rating, pd.to_numeric(content_rating_Teen.Reviews),color='yellow',label='Teen')

plt.scatter( content_rating_Every10.Rating, pd.to_numeric(content_rating_Every10.Reviews),color='pink',label='Everyone 10+')

plt.scatter( content_rating_Mature17.Rating, pd.to_numeric(content_rating_Mature17.Reviews),color='green',label='Matue 17+')

plt.scatter( content_rating_Adults.Rating, pd.to_numeric(content_rating_Adults.Reviews),color='blue',label='Adults only 1+')

plt.scatter( content_rating_Unrated.Rating, pd.to_numeric(content_rating_Unrated.Reviews),color='grey',label='Unrated')

plt.legend()

plt.show()
plt.figure( figsize=( 10,10) )





x=data['Type']



f=0

p=0

n=0

z=0



types=['Free', 'Paid', 'nan', '0']





for each in x:

    if( each=='Free'):

       f+=1

    if( each=='Paid'):

       p+=1

    if( each=='\0' ):

       n+=1

    if( each=='0' ):

       z+=1

numbers=np.array([f,p,n,z])    



plt.bar( types,numbers,color='pink',edgecolor='purple' )

plt.title("bar plot")

plt.xlabel("Types")

plt.ylabel("Numbers")

plt.show()



plt.figure( figsize=( 10,10) )



positive=data2[ data2.Sentiment=='Positive' ]

neutral=data2[ data2.Sentiment=='Neutral' ]

nan=data2[ data2.Sentiment=='NaN' ]



plt.scatter( positive.Sentiment_Polarity,positive.Sentiment_Subjectivity,color='green',label='positive',alpha=0.5,linewidths=0.01,norm=0.5 )

plt.scatter( neutral.Sentiment_Polarity,neutral.Sentiment_Subjectivity,color='grey' ,label='neutral',alpha=0.2,linewidths=0.001,norm=0.5)

plt.scatter( nan.Sentiment_Polarity,nan.Sentiment_Subjectivity,color='black' ,label='Nan')



plt.legend()

plt.xlabel('Sentiment Polarity')

plt.ylabel('Sentiment Subjectivity')



plt.title("Classification of Positive-Neutral-Nan Sentiments")

plt.show()





data_new=data[np.logical_and(data['Rating']==5.0 , data['Category']=='MEDICAL')]

data_new2=data_new[ np.logical_and( data_new['Type']=='Paid',data_new['Reviews']=='2')]

data_new2



       
data2[ (data2['Sentiment']=='Neutral')  & (data2['Sentiment_Subjectivity']>=1 ) & ( data2['Translated_Review']=='I downloaded only, well I know')]