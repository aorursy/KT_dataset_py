

#import necessary packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#Read input files
train_data = pd.read_json('../input/train.json')
test_data = pd.read_json('../input/test.json')

#TrainData
train_data.head()
train_data.info()
#testData
test_data.head()
test_data.info()
all_ingredients_list = []
all_ingredients_list = train_data['ingredients']
in_list = []
for each in all_ingredients_list:
    for i in each:
        in_list.append(i)
unique_ing = []
for each1 in in_list:
    if each1 not in unique_ing:
        unique_ing.append(each1)
print("Total number of ingredients",len(unique_ing))
from collections import Counter
Counter(in_list).most_common(50)
total = train_data.isnull().sum().sort_values(ascending = False)
percent = (train_data.isnull().sum()/train_data.isnull().count()*100).sort_values(ascending = False)
missing_train_data  = pd.concat([total, percent], axis=1, keys=['Total missing', 'Percent missing'])
missing_train_data.head(20)
import seaborn as sns
plt.figure(figsize=(12,8))
sns.countplot(x="cuisine", data=train_data)
plt.ylabel('Count', fontsize=12)
plt.xlabel('Cuisine', fontsize=12)
plt.xticks(rotation='vertical')
plt.title("Frequency of Column Name", fontsize=15)
plt.show()

for each in train_data['ingredients']:
    print(each)
    break
# Label Encoding of y - the target kinds of cuisine
from sklearn.preprocessing import LabelEncoder


y_value = train_data['cuisine'].copy()
labelencoder = LabelEncoder()
y_value = labelencoder.fit_transform(y_value)
print(y_value)
print("Total length", len(y_value))
labelencoder.inverse_transform(y_value)
train_data_without_id = train_data.copy()
train_data_without_id.drop(['id','cuisine'],axis=1,inplace=True)
test_data_without_id = test_data.copy()
test_data_without_id.drop(['id'],axis=1,inplace=True)
for each in train_data['ingredients']:
    s = str(''.join(each).lower().strip())
    print(s)
    break
train_data_without_id['ingredients'] = [''.join(each).lower().strip() for each in train_data['ingredients']]
train_data_without_id['ingredients'].value_counts()
test_data_without_id['ingredients'] = [''.join(each).lower().strip() for each in test_data['ingredients']]
test_data_without_id['ingredients'].value_counts()
from sklearn.feature_extraction.text import TfidfVectorizer
# TFIDF statiscic applying to the data - resulting in sparse matrix
tfidf = TfidfVectorizer()
X_train = tfidf.fit_transform(train_data_without_id['ingredients'])
X_train
X_test = tfidf.transform(test_data_without_id['ingredients'])
X_test
from sklearn.tree import DecisionTreeClassifier
from sklearn.multiclass import OneVsRestClassifier
dt = DecisionTreeClassifier()
model_dt = OneVsRestClassifier(dt, n_jobs=-1)

model_dt.fit(X_train, y_value)
acc_dt = round(model_dt.score(X_train, y_value) * 100, 2)
print(acc_dt)
