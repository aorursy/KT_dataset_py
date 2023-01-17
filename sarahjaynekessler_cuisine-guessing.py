#Math and DataFrame stuff
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

#basic computer stuff
import os
print(os.listdir("../input"))

#plotting stuff
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
plt.style.use('ggplot')

# machine learning
from sklearn.feature_extraction.text import CountVectorizer


train_df = pd.read_json('../input/train.json')
test_df = pd.read_json('../input/test.json')
print(train_df.columns.values)
print(test_df.columns.values)

train_df.head(10)
train_df['ingredients'] = train_df['ingredients'].apply(', '.join)
test_df['ingredients'] = test_df['ingredients'].apply(', '.join)
combine = [train_df,test_df]
train_df.head(10)
test_df.head(10)
sns.countplot(y = 'cuisine',data = train_df)
sns.set(rc = {'figure.figsize' : (8,5)})

common_ing = []
for x in np.arange(len(train_df['cuisine'])):
    for i in train_df['ingredients'][x].split(','):
        common_ing.append(i.strip())
common_ing = pd.DataFrame(common_ing, columns=['common_ing'])
          
common_ing['common_ing'].value_counts().head(10)

count_vec = CountVectorizer(tokenizer=lambda x: [i.strip() for i in x.split(',')], lowercase=False)
X_train_counts = count_vec.fit_transform(train_df['ingredients']) 
X_train_counts.shape
from sklearn.feature_extraction.text import TfidfTransformer
tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
X_train_tf = tf_transformer.transform(X_train_counts)
X_train_tf.shape
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline

text_clf = Pipeline([('vect', CountVectorizer(tokenizer=lambda x: [i.strip() for i in x.split(',')], lowercase=False)),('clf', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=42,max_iter=5, tol=None)),])
text_tdif_clf = Pipeline([('vect', CountVectorizer(tokenizer=lambda x: [i.strip() for i in x.split(',')], lowercase=False)),('tfidf', TfidfTransformer()),('clf', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=42,max_iter=5, tol=None)),])

text_tdif_clf.fit(train_df['ingredients'], train_df['cuisine']) 
predicted = text_tdif_clf.predict(train_df['ingredients'])
np.mean(predicted == train_df['cuisine'])
from sklearn import metrics
print(metrics.classification_report(train_df['cuisine'],predicted))
cm = metrics.confusion_matrix(train_df['cuisine'],predicted)
cm.shape
legend = ['brazilian','british','cajun_creole','chinese','filipino','french','greek','indian','irish','italian','jamaican','japanese','korean','mexican','moroccan','russian','southern_us','spanish','thai','vietnamese']
df_cm = pd.DataFrame(cm,index = legend,columns=legend)
plt.figure
sns.set(font_scale= 1.4,rc = {'figure.figsize' : (15,15)})
sns.heatmap(df_cm,annot = True, linewidths=.5,fmt = 'd',cmap = 'viridis',cbar = False).set_title('Confusion Matrix With Tdif')

text_clf.fit(train_df['ingredients'], train_df['cuisine']) 
predicted = text_clf.predict(train_df['ingredients'])
np.mean(predicted == train_df['cuisine'])
print(metrics.classification_report(train_df['cuisine'],predicted))
cm = metrics.confusion_matrix(train_df['cuisine'],predicted)
df_cm = pd.DataFrame(cm,index = legend,columns=legend)
plt.figure
sns.set(font_scale= 1.4,rc = {'figure.figsize' : (15,15)})
sns.heatmap(df_cm,annot = True, linewidths=.5,fmt = 'd',cmap = 'viridis',cbar = False).set_title('Confusion Matrix Without Tdif')

sub = pd.read_csv('../input/sample_submission.csv')
sub.head()
final_predicted = text_clf.predict(test_df['ingredients'])

predictions = pd.DataFrame({'cuisine' : final_predicted , 'id' : test_df.id })
predictions = predictions[[ 'id' , 'cuisine']]
predictions.to_csv('submit.csv', index = False)
