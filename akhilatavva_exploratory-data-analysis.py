import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
data=pd.read_csv("../input/amazon-fine-food-reviews/Reviews.csv")
print(data.shape)
data.head()
data1= data.groupby('Score').size()
data1
data1.plot(kind='bar',title='Label Distribution',color=["blue","red","orange","green","yellow"])
plt.xlabel('rating')
plt.ylabel('values')
# plt.legend()
plt.show()
data["dislikes"]=data["HelpfulnessDenominator"]-data["HelpfulnessNumerator"]
data
data1=data[["HelpfulnessNumerator","dislikes","Score"]]
data2= data1.groupby('Score').sum()
data2
data2.plot(kind='bar',title='Label Distribution')
plt.xlabel('no ')
plt.ylabel('values')
# plt.legend()
plt.show()
import pickle
pickle_in = open("../input/textpreprocessednew/textpreprocessednew","rb")
list1 = pickle.load(pickle_in)
list1[0:100]
type(list1[0])
str1=""
for i in list1:
    str1=str1+i
len(str1)
from wordcloud import WordCloud
wordcloud_spam = WordCloud(background_color="white").generate(str1[0:1000000])
plt.figure(figsize = (20,20))
plt.imshow(wordcloud_spam, interpolation='bilinear')
plt.axis("off")
plt.show()
