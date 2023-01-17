import os
print(os.listdir("../input"))
import pandas as pd
df=pd.read_csv("../input/Suicides in India 2001-2012.csv")
df=df[df['Total']>0] 
df=df[df['Type_code']=='Causes']


tab1=pd.crosstab(index=df['Gender'],columns=df['Type'])
tab1
tab1.sum(axis=1)
tab3=pd.crosstab(index=df['Age_group'],columns=df['Type'])
tab3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
plt.subplots(figsize=(15,10))
sns.set_style(style='whitegrid', 
              rc={'font.sans-setif':'Helvetica'}) # axis parameters can be passed in argument rc
sns.boxplot(data=tab3.T)

plt.pie(tab1.sum(axis=1),labels=['male','female'],autopct='%.2f%%',shadow=True)
# Set aspect ratio to be equal so that 
# pie is drawn as a circle.
plt.axis('equal')
# Set aspect ratio to be equal so that 
# pie is drawn as a circle.
plt.axis('equal')



plt.subplots(figsize=(15,10))
plt.pie(tab1.sum(axis=0),labels=list(set((df['Type'].values).tolist())),autopct='%.2f%%',shadow=True)
# Set aspect ratio to be equal so that 
# pie is drawn as a circle.
plt.axis('equal')
# Set aspect ratio to be equal so that 
# pie is drawn as a circle.
plt.axis('equal')
tab2=pd.crosstab(index=df['State'],columns=df['Year'])
tab2
plt.subplots(figsize=(15,10))
sns.heatmap(tab2,linewidths=.5, cmap="YlGnBu")
df2=pd.read_csv("../input/Suicides in India 2001-2012.csv")

df2=df2[df2['Type_code']=="Means_adopted"]

from wordcloud import WordCloud
import matplotlib.pyplot as plt
ls1=df2['Type'].tolist()
text= ' '.join(ls1)
# Create a list of word
# Create the wordcloud object
wordcloud = WordCloud(width=480, height=480, margin=0).generate(text)
plt.subplots(figsize=(15,10))
# Display the generated image:
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.margins(x=0, y=0)
plt.show()


df3=pd.read_csv("../input/Suicides in India 2001-2012.csv")
df3=df3[df3['Total']>0] 
df3=df3[df3['Type_code']=="Social_Status"]

sns.countplot('Type',data=df3,palette='cubehelix')




