import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
food=pd.read_csv('../input/en.openfoodfacts.org.products.tsv',delimiter='\t',nrows=10000,encoding='utf-8-sig',low_memory=False)
#reading
food.head(10)
food.describe()
food.info()
null_vals=food.isnull().sum()
total_val_sum=np.product(food.shape) #Total values in the dataset
null_val_sum=null_vals.sum() #All null values in the data set
print(null_val_sum/total_val_sum*100,'% of null values exist in the dataset.')
#CLEANING (BY DROPPING):
#method:1 (dropping the rows in which there is atleast 1 field NA
#one option is to drop the fields with NA values:
food_d=food
food_d.dropna(axis=0)
#almost all rows have atleast 1 cell with value: NA, so we get 0 rows as the result. This methods is not efficient.
#method:2 (dropping the column in which there is atleast 1 field NA
food_c=food.dropna(axis=1)
food_c.head(10)
#CLEANING BY SETTING THE VALUE OF NA THE UPPER AND LOWER VALUES IN THAT COLUMN- An efficient method
food_1=food.fillna(method='bfill').fillna(method='ffill').dropna(axis=1)
food_1.head(10)
#even after filling the null cells, some columns remains null. So, we drop them.
#After cleaning the dataset
food_1.describe()
print('Null values after cleaning the dataset (just for confirmation):',food_1.isnull().sum().sum())
#VISUALIZATION:
plt.figure(figsize=(15,7))
plt.xticks(rotation=90)
sb.barplot(x=food_1['countries'].value_counts().index,y=food_1['countries'].value_counts().values)

plt.figure(figsize=(15,7))
plt.xticks(rotation=90)
sb.barplot(x=food_1['origins'].value_counts().index,y=food_1['origins'].value_counts().values)

vitamins=food_1.loc[:,'vitamin-a_100g':'vitamin-b12_100g'].sum(axis=1)
vitamin_rich=(vitamins>0.1)
vitamin_rich['VitaminRich']=(vitamins>0.1)
sb.catplot('VitaminRich',kind='count',data=vitamin_rich)

calcium=food_1.loc[:,'calcium_100g']
high_calcium=(calcium>0.1)
plt.pie(high_calcium.value_counts().values,labels=high_calcium.value_counts().index,autopct='%1.1f%%')
plt.title('% Calcium Rich Items:')
iron=food_1.loc[:,'iron_100g']
high_iron=(iron>0.005)
plt.pie(high_iron.value_counts().values,labels=high_iron.value_counts().index,autopct='%1.1f%%')
plt.title('% Iron Rich Items:')
food_1['proteins_100g'].hist()
food_1['carbohydrates_100g'].hist()
food_1['sugars_100g'].hist()
food_1['cholesterol_100g'].hist()
#food_1['countries_tags'].hist()
#print("EAF")
#sb.swarmplot(x='code',data=food_1)
sb.swarmplot(x='nutrition-score-uk_100g',data=food_1)
#food_1['nutrition-score-uk_100g']
#sb.swarmplot(data=xx)