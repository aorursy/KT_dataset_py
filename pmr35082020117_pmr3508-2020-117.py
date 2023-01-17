import pandas as pd
import matplotlib.pyplot as plt
adult = pd.read_csv("../input/adult-pmr3508/train_data.csv",
        names=[
        "Id","Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status",
        "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",
        "Hours per week", "Country", "Target"],
         engine='python',skiprows = 1,na_values="?")
adult = adult.drop(["Id"], axis = 1)
adult 
adult.info()
adult.describe()
plt.figure(figsize = (7,7))
adult["Workclass"].value_counts().plot(kind="bar")
plt.title("Workclass")
plt.figure(figsize = (7,7))
adult["Education"].value_counts().plot(kind="bar" )
plt.title("Education")
plt.figure(figsize = (7,7))
adult["Occupation"].value_counts().plot(kind="bar" )
plt.title("Occupation")
plt.figure(figsize = (7,7))
adult["Race"].value_counts().plot(kind="bar")
plt.title("Race")
plt.figure(figsize = (7,7))
adult["Sex"].value_counts().plot(kind="bar")
plt.title("Sex")
plt.figure(figsize = (7,7))
adult["Country"].value_counts().plot(kind="bar")
plt.title("Country")
plt.figure(figsize = (7,7))
adult["Martial Status"].value_counts().plot(kind="bar")
plt.title("Martial Status")
plt.figure(figsize = (7,7))
adult["Capital Gain"].value_counts().plot(kind="hist")
plt.title("Capital Gain")
plt.figure(figsize = (7,7))
adult["Capital Loss"].value_counts().plot(kind="hist")
plt.title("Capital Loss")
plt.figure(figsize = (7,7))
adult["Target"].value_counts().plot(kind="pie")
plt.title("Target")
plt.figure(figsize = (7,7))
adult["Education-Num"].value_counts().plot(kind="bar")
plt.title("Education-Num")
plt.figure(figsize = (7,7))
adult["Hours per week"].value_counts().plot(kind="hist")
plt.title("Hours per week")
adult.drop_duplicates(keep = 'first', inplace = True)
adult = adult.drop(['fnlwgt', 'Country', 'Education'], axis=1)
adult
categoric = adult[["Workclass","Martial Status","Occupation","Relationship", "Race", "Sex"]]
numeric = adult.drop(["Workclass","Martial Status","Occupation", "Relationship","Race", "Sex"], axis = 1)
categoric
numeric
categoric = pd.get_dummies(categoric)
new_adult = pd.concat([categoric, numeric], axis = 1)
new_adult
Y_train = new_adult.pop("Target")
X_train = new_adult
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import numpy as np
ks = [15, 20, 25, 30, 35]

k_scores = {}

for k in ks:
    knn = KNeighborsClassifier(n_neighbors= k)
    score = cross_val_score(knn, X_train, Y_train, cv = 5, scoring = "accuracy").mean()
    
    k_scores[k] = score

    print(f"k = {k} , score =  {score.mean()}")
    
melhor_k = max(k_scores, key=k_scores.get)

print()
print("Melhor hiperparâmetro = ", melhor_k)
print("Melhor acurácia = ", k_scores[melhor_k])
adult_test = pd.read_csv("../input/adult-pmr3508/test_data.csv",
        names=[
        "Id","Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status",
        "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",
        "Hours per week", "Country"],
         engine='python',skiprows = 1,na_values="?")
adult_test = adult_test.drop(["Id"], axis = 1)
adult_test 
adult_test = adult_test.drop(["fnlwgt", "Country", "Education"], axis=1)
adult_test
categoric_test = adult_test[["Workclass","Martial Status","Occupation","Relationship", "Race", "Sex"]]
numeric_test = adult_test.drop(["Workclass","Martial Status","Occupation", "Relationship","Race", "Sex"], axis = 1)
categoric_test
numeric_test
categoric_test = pd.get_dummies(categoric_test)
new_adult_test = pd.concat([categoric_test, numeric_test], axis = 1)
X_test = new_adult_test
X_test
knn = KNeighborsClassifier(n_neighbors= 20)
knn.fit(X_train, Y_train)
predict_adult = knn.predict(X_test)
predict_adult
submission = pd.DataFrame()
submission["ID"] = adult_test.index
submission["Income"] = predict_adult
submission
submission.to_csv('submission.csv',index = False)