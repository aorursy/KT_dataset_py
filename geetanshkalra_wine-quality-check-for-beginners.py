import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
df = pd.read_csv('../input/winequality-red.csv')
df.head()
df.tail()
df.isnull().sum()
df.shape
df.info()
df.head()
plt.figure(figsize=(15,8))
plt.title("Correlation")
sns.heatmap(df.corr(),annot=True)
df.drop('volatile acidity',axis=1,inplace=True)
df.drop('chlorides',axis=1,inplace=True)
df.drop('free sulfur dioxide',axis=1,inplace=True)
df.drop('total sulfur dioxide',axis=1,inplace=True)
df.drop('density',axis=1,inplace=True)
df.drop('pH',axis=1,inplace=True)
df.head()
sns.barplot(x = 'quality', y = 'fixed acidity', data = df)
bins = (2, 7, 8)
group_names = ['bad', 'good']
df['quality'] = pd.cut(df['quality'], bins = bins, labels = group_names)
label_quality = LabelEncoder()
df['quality'] = label_quality.fit_transform(df['quality'])
sns.countplot(df['quality'])
X = df.drop('quality',axis=1)
y = df['quality']
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.2)
from sklearn.linear_model import LogisticRegression
logistic = LogisticRegression()
logistic.fit(x_train,y_train)
prediction = logistic.predict(x_test)
from sklearn.metrics import accuracy_score
accuracy_score(prediction,y_test)
