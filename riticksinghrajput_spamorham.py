import pandas as pd



import string

import nltk

from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer



from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression



from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt
data = pd.read_csv("../input/sms-spam-collection-dataset/spam.csv",encoding = "latin")

data = data.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis = 1)

data = data.rename(columns = {'v1':'Spam/Ham','v2':'message'})

data.head()
def preprocessing_func(var):

    text = var.translate(str.maketrans('', '', string.punctuation))

    text = [word for word in text.split() if word.lower() not in stopwords.words('english')]

    return " ".join(text)



data_2 = data['message'].copy()

data_2 = data_2.apply(preprocessing_func)
vectorizer = TfidfVectorizer("english")

data_matrix = vectorizer.fit_transform(data_2)
X_train, X_test, Y_train, Y_test = train_test_split(data_matrix, data['Spam/Ham'], test_size=0.3)
logistic = LogisticRegression()

logistic.fit(X_train, Y_train)
predictions = logistic.predict(X_test)



# Accuracy score metrics

acc = accuracy_score(Y_test, predictions)

print("Accuracy score : ",acc,"\nAccuracy %ge = ",acc*100,"%")



# Scatter Plot

plt.scatter(Y_test, predictions)

plt.xlabel("True Values",color='red')

plt.ylabel("Predictions",color='blue')

plt.title("Predicted vs Actual value")

plt.grid(True)

plt.show()