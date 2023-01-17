import pandas as pd

from sklearn.model_selection import train_test_split 

from sklearn.tree import DecisionTreeClassifier 

from sklearn import metrics 



df_female = pd.read_csv("../input/ansur-ii/ANSUR II FEMALE Public.csv", 

                               encoding='latin-1') 

df_male = pd.read_csv("../input/ansur-ii/ANSUR II MALE Public.csv",

                             encoding='latin-1') 

df_female = df_female.rename(

                columns = {"SubjectId":"subjectid"}) # Fixing a column name

df_all = pd.concat([df_female,df_male])
# Collect only the required data for training

X = df_all[df_all.columns[1:94]]

X.insert(93, "Heightin", df_all['Heightin'], True)

X.insert(94, "Weightlbs", df_all['Weightlbs'], True)



Y = df_all['Gender']



X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=1) # 80% training and 20% test
clf = DecisionTreeClassifier(criterion="entropy")



# Train Decision Tree Classifer

clf = clf.fit(X_train,y_train)



#Predict the gender for test dataset

y_pred = clf.predict(X_test)
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))