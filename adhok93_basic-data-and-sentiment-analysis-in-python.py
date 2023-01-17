import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
data_review = pd.read_csv('../input/Womens Clothing E-Commerce Reviews.csv')
data_review.shape
data_see[['Age']].hist(bins=10)
plt.show()
np.mean(data_see.Age)
def age_group(data_frame):
    '''Function to define the age groups for a dataframe'''
    data_frame.loc[data_frame['Age']<25,'Age Group']   = '18-25'
    data_frame.loc[(data_frame['Age']>=25) & (data_frame['Age']<30),'Age Group']   = '25-30'
    data_frame.loc[(data_frame['Age']>=30) & (data_frame['Age']<35),'Age Group']   = '30-35'
    data_frame.loc[(data_frame['Age']>=35) & (data_frame['Age']<40),'Age Group']   = '35-40'
    data_frame.loc[(data_frame['Age']>=40) & (data_frame['Age']<50),'Age Group']   = '40-50'
    data_frame.loc[(data_frame['Age']>=50) & (data_frame['Age']<60),'Age Group']   = '50-60'
    data_frame.loc[(data_frame['Age']>=60) & (data_frame['Age']<70),'Age Group']   = '60-70'
    data_frame.loc[(data_frame['Age']>=70) & (data_frame['Age']<80),'Age Group']   = '70-80'
    data_frame.loc[(data_frame['Age']>=80) & (data_frame['Age']<90),'Age Group']   = '80-90'
    data_frame.loc[(data_frame['Age']>=90),'Age Group']   = '>=90'
    return data_frame

division_name_rating = data_review.groupby(['Division Name','Rating']).size().reset_index()
division_name_rating.columns = ['Division Name','Rating','Occurence']
division_name_rating.pivot(index='Division Name',columns='Rating',values='Occurence').reset_index().plot(kind='bar',x='Division Name')
plt.show()
# generally how does each age group vote?

data_age_rating = data_review[['Age','Rating']].groupby(['Age','Rating']).size().reset_index()
data_age_rating.columns 
data_age_rating = age_group(data_age_rating)

data_age_rating.columns = ['Age','Rating','Freq','Age Group'] 
data_age_group_freq= data_age_rating.groupby('Age Group').agg({'Freq':np.sum}).reset_index()
data_age_group_freq.columns = ['Age Group','Freq']
data_age_group_freq.plot(kind='bar',x='Age Group',y='Freq',title='Number of Individuals in each age group')
plt.legend().remove()

plt.show()
# How did each age group vote on average?
data_age_rating['Total Rating'] = data_age_rating['Rating']*data_age_rating['Freq']
data_age_group_rating=data_age_rating.groupby('Age Group').agg({'Total Rating':np.sum,'Freq':np.sum}).reset_index()
data_age_group_rating['avg']= data_age_group_rating['Total Rating']/data_age_group_rating['Freq']
data_age_group_rating.plot(kind='bar',x='Age Group',y='avg')
plt.legend().remove()
plt.show()

    
    
# Variation in the ratings across ages
data_age_rating_variation = data_review[['Age','Rating']]
data_age_rating_variation = age_group(data_age_rating_variation)

data_age_rating_variation = data_age_rating_variation.drop('Age',axis=1)
data_age_rating_variation.boxplot(by='Age Group',figsize=(20,10))
plt.xticks(rotation=90)
plt.show()
division_name_rating= data_review[['Rating','Class Name']]
division_name_rating.boxplot(by='Class Name',figsize=(20,10))
plt.xticks(rotation=90)

plt.show()

division_name_rating.groupby('Class Name').agg({'Rating':np.mean}).reset_index().plot(kind='bar',x='Class Name',y='Rating')
plt.show()
# How do ratings vary for each age group across Classes of products?

data_product_rating_box = data_see[['Class Name','Age','Rating']]

from pylab import plot, show, savefig, xlim, figure,hold, ylim, legend, boxplot, setp, axes

##creating the age group variable

data_product_rating_box = age_group(data_product_rating_box)


for i in list(pd.unique(data_product_rating_box[['Age Group']].values.ravel('K'))):
    data_product_rating_box_i = data_product_rating_box.loc[data_product_rating_box['Age Group']==i,]
    data_product_rating_box_i.drop('Age',axis=1,inplace=True)
    data_product_rating_box_i.boxplot(by='Class Name',figsize=(10,10))
    plt.title('Ratings Distribution by Product for Age Groups '+i)
    plt.xticks(rotation=90)
    plt.suptitle('')
    plt.show()
    

    
from afinn import Afinn
afinn = Afinn()
data_review_text_product = data_review[['Review Text','Class Name','Age']]
data_review_text_product['Review Text']= data_review_text_product['Review Text'].astype(str)
data_review_text_product = age_group(data_review_text_product)


data_review_text_product['sent_score'] = data_review_text_product.apply(lambda row: afinn.score(row['Review Text']), axis=1)

#data_review_text_product.drop('Age',axis=1,inplace=True)
for i in list(pd.unique(data_review_text_product['Age Group'].values.ravel('K'))):
    data_review_text_product_i = data_review_text_product.loc[data_review_text_product['Age Group']==i,]
    data_review_text_product_i.drop('Age',axis=1,inplace=True)
    data_review_text_product_i.boxplot(by='Class Name',figsize=(10,10))
    plt.title('Sentiment Distribution by Product for Age Groups '+i)
    plt.xticks(rotation=90)
    plt.suptitle('')
    plt.show()
