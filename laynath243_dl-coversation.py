import pandas as pd

# df = pd.read_csv("../input/my_data - Sheet1.csv")

# x = df["Questions"]

# y = df["Answers"]

# from sklearn.model_selection import train_test_split

# x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0,random_state=0)



df = pd.read_csv("../input/Telegrambot.csv")

x = df['Keyword']

y = df['Bot Reply']

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.svm import LinearSVC

from sklearn.pipeline import Pipeline

model = Pipeline([("tfidf",TfidfVectorizer()),("scv",LinearSVC())])

clf = model.fit(x,y)

from joblib import dump , load

dump(clf, 'koompi_bot.joblib')
model = load('koompi_bot.joblib') 
model.predict([""])