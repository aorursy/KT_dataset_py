import numpy as np 

import pandas as pd



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
training_variants_df = pd.read_csv("../input/training_variants")
training_variants_df.head(5)
training_text_df = pd.read_csv("../input/training_text",sep="\|\|", engine='python', header=None, skiprows=1, names=["ID","Text"])
training_text_df.head(5)
training_text_df["Text"][0]
training_merge_df = training_variants_df.merge(training_text_df,left_on="ID",right_on="ID")
training_merge_df.head(5)
training_merge_df.columns
testing_variants_df = pd.read_csv("../input/test_variants")
testing_variants_df.head(5)
testing_text_df = pd.read_csv("../input/test_text", sep="\|\|", engine='python', header=None, skiprows=1, names=["ID","Text"])
testing_text_df.head(5)
testing_merge_df = testing_variants_df.merge(testing_text_df,left_on="ID",right_on="ID")
testing_merge_df.head(5)
training_merge_df["Class"].unique()
training_merge_df.describe()
testing_merge_df.describe()
import missingno as msno

%matplotlib inline

msno.bar(training_merge_df)
msno.bar(testing_merge_df)
from sklearn.model_selection import train_test_split



train ,test = train_test_split(training_merge_df,test_size=0.2) 

np.random.seed(0)

train
X_train = train['Text'].values

X_test = test['Text'].values

y_train = train['Class'].values

y_test = test['Class'].values
from sklearn.pipeline import Pipeline

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfTransformer

from sklearn import svm
text_clf = Pipeline([('vect', CountVectorizer()),

                     ('tfidf', TfidfTransformer()),

                     ('clf', svm.LinearSVC())

])

text_clf = text_clf.fit(X_train,y_train)
y_test_predicted = text_clf.predict(X_test)

np.mean(y_test_predicted == y_test)
X_test_final = testing_merge_df['Text'].values
predicted_class = text_clf.predict(X_test_final)
testing_merge_df['predicted_class'] = predicted_class
testing_merge_df.head(5)
onehot = pd.get_dummies(testing_merge_df['predicted_class'])

testing_merge_df = testing_merge_df.join(onehot)
testing_merge_df.head(5)
submission_df = testing_merge_df[["ID",1,2,3,4,5,6,7,8,9]]

submission_df.columns = ['ID', 'class1','class2','class3','class4','class5','class6','class7','class8','class9']

submission_df.head(5)
submission_df.to_csv('submission.csv', index=False)