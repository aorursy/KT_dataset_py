import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_predict

from sklearn.model_selection import cross_val_score

from collections import Counter

from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier

%matplotlib inline

df=pd.read_json('../input/train.json')

df.head()
df.info()
import pandas_profiling

pandas_profiling.ProfileReport(df)
f, ax = plt.subplots(figsize=(5,6))

sns.countplot(y = 'cuisine', 

                   data = df,

                  order = df.cuisine.value_counts(ascending=False).index)
ingredients_individual = Counter([ingredient for ingredient_list in df.ingredients for ingredient in ingredient_list])

ingredients_individual = pd.DataFrame.from_dict(ingredients_individual,orient='index').reset_index()





ingredients_individual = ingredients_individual.rename(columns={'index':'Ingredient', 0:'Count'})



#Most common ingredients

sns.barplot(x = 'Count', 

            y = 'Ingredient',

            data = ingredients_individual.sort_values('Count', ascending=False).head(20))
df.ingredients
label = df.cuisine



features = df.drop(['cuisine'], axis = 1)
X_train, X_test, y_train, y_test = train_test_split(features, label, test_size = 0.20, random_state =0) 
train_ingredients_text = X_train.ingredients.apply(lambda s: ' '.join(w.lower() for w in s)).str.replace('[^\w\s]','')

test_ingredients_text = X_test.ingredients.apply(lambda s: ' '.join(w.lower() for w in s)).str.replace('[^\w\s]','')  
tfidf = TfidfVectorizer(

    min_df = 3,

    max_df = 0.95,

    stop_words = 'english'

)



tfidf.fit(train_ingredients_text)

text = tfidf.transform(train_ingredients_text)

text
traintext = tfidf.transform(test_ingredients_text)
clf = RandomForestClassifier(n_estimators=100, max_depth=16,random_state=0)

clf.fit(text, y_train)
y_pred= clf.predict(traintext)
accuracy_score(y_test,y_pred)*100 
y_pred=clf.predict(text)

accuracy_score(y_train,y_pred)*100 
clf.score(text,y_train)
clf.score(traintext,y_test)
clf_gini = DecisionTreeClassifier(criterion = "gini", random_state = 100,max_depth=16, min_samples_leaf=5) 

clf_gini.fit(text, y_train)

clf_gini.score(text,y_train)
clf_gini.score(traintext,y_test)
crossvalidation = df.ingredients.apply(lambda s: ' '.join(w.lower() for w in s)).str.replace('[^\w\s]','')

text1 = tfidf.transform(crossvalidation)

scores = cross_val_score(clf_gini, text1, label, cv=5)

scores
scores = cross_val_score(clf, text1, label, cv=5)

scores
from sklearn.multiclass import OneVsRestClassifier

from sklearn.svm import LinearSVC, SVC

from sklearn.linear_model import LogisticRegression

#from sklearn.ensemble import RandomForestClassifier

# parameters = {'C': np.arange(1, 100, 5)}

model = LinearSVC()

# model = LogisticRegression(multi_class='multinomial')

# model = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)

# model = SVC()



model = OneVsRestClassifier(model)

# model = BaggingRegressor(model, n_estimators=100)

# model = GridSearchCV(model, parameters, n_jobs=-1, verbose=2, cv=3)



print(cross_val_score(model, text, y_train, cv=3)) 



model.fit(text, y_train)

model.score(traintext, y_test)
df1=pd.read_json("../input/test.json")

df1.head()
predicting = df1.ingredients.apply(lambda s: ' '.join(w.lower() for w in s)).str.replace('[^\w\s]','')

textpre = tfidf.transform(predicting)

predicted= model.predict(textpre)
print(predicted)
sub=pd.read_csv("../input/sample_submission.csv")

sub.head()

del sub['cuisine']

sub.head()
sub['cuisine']=predicted

sub.head()
sub.to_csv("Submission.csv",index=False)
