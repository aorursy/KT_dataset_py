import pandas as pd

import warnings; warnings.simplefilter('ignore')



dataset= pd.read_csv('../input/sms-spam-collection-dataset/spam.csv',encoding='ISO-8859-1')

to_drop=['Unnamed: 2','Unnamed: 3','Unnamed: 4']

dataset.drop(columns=to_drop,inplace=True)



dataset.head()

dataset['encoded_labels']=dataset['v1'].map({'spam':0,'ham':1})

dataset.head()
from sklearn.model_selection import train_test_split as split_data



labels=dataset.pop('encoded_labels')



train_data,test_data,train_label,test_label=split_data(dataset,labels, test_size=0.3)

from sklearn.feature_extraction.text import CountVectorizer



c_v = CountVectorizer(decode_error='ignore')

train_data = c_v.fit_transform(train_data['v2'])

test_data = c_v.transform(test_data['v2'])
from sklearn import naive_bayes as nb

from sklearn.metrics import accuracy_score







clf=nb.MultinomialNB()

model=clf.fit(train_data, train_label)

predicted_label=model.predict(test_data)

print("train score:", clf.score(train_data, train_label))

print("test score:", clf.score(test_data, test_label))

print("Classifier Accuracy",accuracy_score(test_label, predicted_label))