import pandas as pd

heart = pd.read_csv("../input/heart-disease-uci/heart.csv")

print(heart.keys)

print(heart.shape)

X_Data=pd.DataFrame(heart.iloc[:303,:13])

Y_Data=pd.DataFrame(heart.iloc[:, 13:14])

print(X_Data)

print(Y_Data)



from sklearn.model_selection import train_test_split



x_train, x_test, y_train, y_test = train_test_split(X_Data,Y_Data, test_size=0.2, train_size=0.8, random_state=0)

print(x_train.shape)

print(x_test.shape)

print(y_test.shape)

print(y_train.shape)



from sklearn.linear_model import LogisticRegression



lg_clf=LogisticRegression()

a=lg_clf.fit(x_train,y_train)

print(a)

b=lg_clf.predict(x_test)

print(b)

score=lg_clf.score(x_test,y_test)

print(score)