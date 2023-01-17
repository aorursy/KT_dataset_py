import pandas as pd

pd.set_option('display.max_columns', 500)

pd.set_option('display.width', 1000)
df = pd.read_csv("../input/breast-cancer-wisconsin-data/data.csv")



df.head(2)
df.isnull().sum()
df = df.drop(['Unnamed: 32', 'id'], axis=1)



df.head()
from sklearn.preprocessing import LabelEncoder

enc = LabelEncoder()



features_to_convert = ["diagnosis"]



for i in features_to_convert:

    df[i] = enc.fit_transform(df[i].astype('str'))



df.head(5)



#1 Mal

#0 Ben
df['target'] = df['diagnosis']



df = df.drop(['diagnosis'], axis=1)



df.head(2)
df.shape
X = df.iloc[:,0:30]  #independent columns

y = df.iloc[:,30]    #target column 
from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import chi2



#apply SelectKBest class to extract top k best features

bestfeatures = SelectKBest(score_func=chi2, k=5)

fit = bestfeatures.fit(X,y)

dfscores = pd.DataFrame(fit.scores_)

dfcolumns = pd.DataFrame(X.columns)



#concat two dataframes for better visualization 

featureScores = pd.concat([dfcolumns,dfscores],axis=1)

featureScores.columns = ['Specs','Score']  #naming the dataframe columns

print(featureScores.nlargest(5,'Score'))  #print k best features
from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import f_classif



#apply SelectKBest class to extract top k best features

bestfeatures = SelectKBest(score_func=f_classif, k=5)

fit = bestfeatures.fit(X,y)

dfscores = pd.DataFrame(fit.scores_)

dfcolumns = pd.DataFrame(X.columns)



#concat two dataframes for better visualization 

featureScores = pd.concat([dfcolumns,dfscores],axis=1)

featureScores.columns = ['Specs','Score']  #naming the dataframe columns

print(featureScores.nlargest(5,'Score'))  #print k best features
dfnew = pd.DataFrame(df[['area_worst', 'area_mean', 'area_se', 'perimeter_worst', 'perimeter_mean',

                        'concave points_worst', 'concave points_mean', 'radius_worst']])

dfnew.head()
from sklearn import preprocessing



# Get column names first

names = dfnew.columns



# Create the Scaler object

scaler = preprocessing.MinMaxScaler()



# Fit your data on the scaler object

scaled_df = scaler.fit_transform(dfnew)

scaled_df = pd.DataFrame(scaled_df, columns=names)



scaled_df
scaled_df['target'] = df['target']



scaled_df.head(2)
X = scaled_df.drop("target", axis=1)

Y = scaled_df["target"]
from sklearn.model_selection import train_test_split



X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=101)



print(X_train.shape, X_test.shape)
from sklearn.tree import DecisionTreeClassifier



model = DecisionTreeClassifier(max_depth=2)

model.fit(X_train, Y_train)



y_predict = model.predict(X_test)



from sklearn.metrics import accuracy_score



a = accuracy_score(Y_test, y_predict)

print("Accuracy: ", a.round(4))



from sklearn.model_selection import cross_val_score

import statistics



kfoldscore = cross_val_score(model, X_train, Y_train, cv=5)



print("kFold Scores: {}".format(kfoldscore.round(4)))

print("kFold Score Mean: ", statistics.mean(kfoldscore).round(4))
from sklearn.ensemble import RandomForestClassifier



model = RandomForestClassifier(n_estimators=10)

model.fit(X_train, Y_train)



y_predict = model.predict(X_test)



from sklearn.metrics import accuracy_score



a = accuracy_score(Y_test, y_predict)

print("Accuracy: ", a.round(4))



from sklearn.model_selection import cross_val_score

import statistics



kfoldscore = cross_val_score(model, X_train, Y_train, cv=5)



print("kFold Scores: {}".format(kfoldscore.round(4)))

print("kFold Score Mean: ", statistics.mean(kfoldscore).round(4))
from sklearn.svm import SVC



model = SVC(gamma='auto')

model.fit(X_train, Y_train)



y_predict = model.predict(X_test)



from sklearn.metrics import accuracy_score



a = accuracy_score(Y_test, y_predict)

print("Accuracy: ", a.round(4))



from sklearn.model_selection import cross_val_score

import statistics



kfoldscore = cross_val_score(model, X_train, Y_train, cv=5)



print("kFold Scores: {}".format(kfoldscore.round(4)))

print("kFold Score Mean: ", statistics.mean(kfoldscore).round(4))
from sklearn.neighbors import KNeighborsClassifier



model = KNeighborsClassifier(n_neighbors=6)

model.fit(X_train, Y_train)



y_predict = model.predict(X_test)



from sklearn.metrics import accuracy_score



a = accuracy_score(Y_test, y_predict)

print("Accuracy: ", a.round(4))



from sklearn.model_selection import cross_val_score

import statistics



kfoldscore = cross_val_score(model, X_train, Y_train, cv=5)



print("kFold Scores: {}".format(kfoldscore.round(4)))

print("kFold Score Mean: ", statistics.mean(kfoldscore).round(4))
from sklearn.neural_network import MLPClassifier



model = MLPClassifier(max_iter=100, solver='lbfgs', learning_rate='adaptive')

model.fit(X_train, Y_train)



y_predict = model.predict(X_test)



from sklearn.metrics import accuracy_score



a = accuracy_score(Y_test, y_predict)

print("Accuracy: ", a.round(4))



from sklearn.model_selection import cross_val_score

import statistics



kfoldscore = cross_val_score(model, X_train, Y_train, cv=5)



print("kFold Scores: {}".format(kfoldscore.round(4)))

print("kFold Score Mean: ", statistics.mean(kfoldscore).round(4))