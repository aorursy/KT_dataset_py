import pandas as pd

import json
with open("../input/test.json") as datatest:

    data_test = json.load(datatest)

with open("../input/train.json") as datatrain:

    data_train = json.load(datatrain)
data_test = pd.DataFrame(data_test)

data_train = pd.DataFrame(data_train)
data_train

from collections import Counter
Counter(data_train['cuisine']).most_common()
data_train['ingredients'].tolist()
from sklearn.feature_extraction.text import CountVectorizer
ingredient_list = data_train['ingredients'].tolist()
vectorizer = CountVectorizer()
ingredient_list1 = [' '.join(j for j in i) for i in ingredient_list]
ingredient_list1
x = vectorizer.fit_transform(ingredient_list1)
ingredient_list2 = data_test['ingredients'].tolist()
ingredient_list2 = [' '.join(j for j in i) for i in ingredient_list2]
x_test = vectorizer.transform(ingredient_list2)
x_test
y = data_train.cuisine

y
#from sklearn.model_selection import train_test_split 
#X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=0.3, random_state=42)

from sklearn import svm
clg = svm.SVC(kernel='linear',C=0.4)
final = clg.fit(x,y)
output = pd.Series(final.predict(x_test))
ids = data_test.id
submit = pd.concat([ids,output],axis=1)
submit.head()
submit.columns=['id','cuisine']
submit.to_csv('Final Submission1.csv',index=False)