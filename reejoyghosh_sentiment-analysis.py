import pandas as pd
df = pd.read_csv('../input/sentiment-train/sentiment_train.csv')
df.head()
df.shape
df['label'].value_counts()
import matplotlib.pyplot as plt 
import seaborn as sns
sns.countplot(df['label'])
from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(lowercase=True,stop_words='english')
bow = cv.fit_transform(df['sentence'])
df_bow = pd.DataFrame(bow.A,columns=cv.get_feature_names())
df_bow.head()
df_bow.shape
def count_value(df, column):
    x = 0
    for i in range(df.shape[0]):
        if df[column][i] != 0:
            x += 1
    return x
for col in df_bow.columns:
    if count_value(df_bow,col) <= 2:
        df_bow = df_bow.drop(col,axis=1)
df_bow.shape
y = df.drop(['sentence'],axis=1)
X = df_bow
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=20)
model = GaussianNB()
model.fit(X_train,y_train)
model.score(X_test,y_test)
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
predictions = model.predict(X_test)
cm = confusion_matrix(y_test,predictions)
sns.heatmap(cm,annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')
print(cm)