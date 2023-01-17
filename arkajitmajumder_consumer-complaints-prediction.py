#import the libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
import seaborn as sns
#load the data
df = pd.read_csv("../input/us-consumer-finance-complaints/consumer_complaints.csv")
df.head()
#looking at the attributes and null values if any
df.info()
#slicing out the necessary columns
df = df[['product', 'consumer_complaint_narrative']]
df = df[pd.notnull(df['consumer_complaint_narrative'])]
df.head()
#adding a new column for product (text to categorical values)
df['category_id']=df['product'].factorize()[0]
df.head()
df.groupby('product').consumer_complaint_narrative.count()
#making a bar plot
fig = plt.figure(figsize=(8,6))
df.groupby('product').consumer_complaint_narrative.count().plot.bar(ylim=0)
plt.show()
#splitting the datas
X_train, X_test, y_train, y_test = train_test_split(df['consumer_complaint_narrative'], df['product'])
#converting object to numeric values
lb = LabelEncoder()
y_train = lb.fit_transform(y_train)
y_test = lb.fit_transform(y_test)
#transforming text into features with Tfidf
tfidf = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=5000)
tfidf.fit(df['consumer_complaint_narrative'])
X_train_tfidf = tfidf.transform(X_train)
X_test_tfidf = tfidf.transform(X_test)
#creating and fitting a model
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)
#Lets check the accuracy score
y_predict = model.predict(X_test_tfidf)
print(accuracy_score(y_test, y_predict))
conf_mat = confusion_matrix(y_test, y_predict)
#creating a dictionary for products
category_id_df = df[['product', 'category_id']].drop_duplicates().sort_values('category_id')
category_to_id = dict(category_id_df.values)
id_to_category = dict(category_id_df[['category_id','product']].values)
id_to_category
fig, ax = plt.subplots(figsize=(8,6))
sns.heatmap(conf_mat, annot=True, fmt='d', cmap="BuPu", xticklabels=category_id_df[['product']].values, yticklabels=category_id_df[['product']].values)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()
#lets test our prediction with a random text
texts = ["This company refuses to provide me verification and validation of debt"+ "per my right under the FDCPA. I do not believe this debt is mine."]
text_features = tfidf.transform(texts)
predictions = model.predict(text_features)
print(texts)
print(" - Predicted as: '{}'".format(id_to_category[predictions[0]]))

