import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
df=pd.read_csv("../input/appendix.csv")
df.head()
#Histpgram plots
df['Year'].plot(kind='hist')
plt.xlabel("Year Launched")
plt.show()
#Courses categories
set(df['Course Subject'])
df.groupby("Institution").count()
#MIT has more courses than Harvard
Ct = df['Institution'].value_counts().plot(kind='bar',
                                    figsize=(8,6),
                                    title="Number for each institution")
Ct.set_xlabel("Names")
Ct.set_ylabel("Frequency")
plt.show()
df.groupby('Year').count()
#Increasing number of courses
df.groupby(["Institution",'Year']).median()
#Get the course with oldest learners 
#df.iloc[df['Median Age'].idxmax()]
#Or
df.sort_values('Median Age').iloc[-1]
#Men and women's favorites
pd.Series(df['Course Title'],index=[df['% Male'].idxmax(),df['% Female'].idxmax()])
#Set default age = 25 for displaying purpose. Or: age=input("Catch up with your peer of age:")
age=30
#Peers are defined as people within 3 years' age difference.
res=df[(df['Median Age']<=(int(age)+3))&(df['Median Age']>=(int(age)-3))]                                                                                                
wc = WordCloud(stopwords=STOPWORDS,background_color='white',
               width=2000,height=1800).generate(" ".join(res['Course Title']))
plt.imshow(wc)
plt.axis('off')
plt.show()
print("Match with your interests too?")