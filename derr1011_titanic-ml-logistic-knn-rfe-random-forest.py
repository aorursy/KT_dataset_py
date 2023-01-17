#work in progress
#OBJECTIVE:Predict if a passenger survived or not in the sinking of Titanic 

#METRIC: % of accuracy 

#SUBMISSION: csv with 2 columns (passenger ID and predicted class)
import pandas as pd 

import numpy as np



import matplotlib.pyplot as plt

import seaborn as sns



from pandas.plotting import scatter_matrix

from sklearn.cluster import KMeans

from sklearn import preprocessing

from sklearn.preprocessing import LabelEncoder

from sklearn.feature_selection import RFE, RFECV

from sklearn.linear_model import LinearRegression, LogisticRegression

from sklearn import tree

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix, accuracy_score
df = pd.read_csv("../input/train.csv")

df.drop(['Name'], 1, inplace=True)

df.convert_objects(convert_numeric=True)

df.fillna(0, inplace=True)

print(df.info(), df.head())
#method 1: convert to numeric for gender variable 

def sex_to_numeric(x):

    if x=='male':

        return 1

    if x=='female':

        return 2

df['Sex'] = df['Sex'].apply(sex_to_numeric)
def embarked_to_numeric(y):

    if y=="S":

        return 1

    if y=='C':

        return 2

    if y=='Q':

        return 3

    

df['Embarked'] = df['Embarked'].apply(embarked_to_numeric)
#method 2: convert to numeric using label encoder (but not applicable for str/int instances e.g. cabin and embarked )

label = LabelEncoder()

df['Ticket'] = label.fit_transform(df['Ticket'])
#method 3: convert into numeric 

def non_numeric_data(df):

    columns = df.columns.values

    for column in columns:

        numeric_data = {}

        

        def transform_to_numeric(x):

            return numeric_data[x]

        

        if df[column].dtype != np.int64 and df[column].dtype != np.float64: 

            col_contents = df[column].values.tolist()

            col_elements = set(col_contents)

            i = 0

            for element in col_elements: 

                if element not in numeric_data:

                    numeric_data[element] = i

                    i += 1

            df[column] = list(map(transform_to_numeric, df[column])) 

    return df     
df = non_numeric_data(df)
#double check - all converted to int or float

df.info(), df.head()
df.describe()
#data visualization 

df.plot(kind='box', subplots=True, sharex=False, sharey=False)

scatter_matrix(df)

plt.show()
#missing values in Embarked

df.isnull().any()
#replace all NAN elements with 0

df.fillna(0, inplace=True)
X = np.array(df.drop(['Survived', 'PassengerId'],1).astype(float))

Y = np.array(df['Survived'])
#Perform feature selections using RFE - select top 5 features

model = LogisticRegression()

rfe = RFE(model, 5)

fit = rfe.fit(X,Y)

print(fit.n_features_)

print(np.asarray(df.columns.drop(['Survived', 'PassengerId'])))

print(rfe.ranking_)

print(rfe.support_)
#automatically select best number of features after cross-validation using RFECV

rfecv = RFECV(estimator=model, step=1, cv=10)

fit = rfecv.fit(X,Y)

print(fit.n_features_)

print(np.asarray(df.columns.drop(['Survived', 'PassengerId'])))

print(rfecv.ranking_)

print(rfecv.support_)
#conduct logistic regression with all features

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=1)



model = LogisticRegression()

logreg = model.fit(X_train, Y_train)

predictions = model.predict(X_test)

print('accuracy score', accuracy_score(Y_test, predictions))
#conduct logistic regression with selected features

X2 = np.array(df.drop(['Survived', 'PassengerId', 'Age', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'],1).astype(float))

X2_train, X2_test, Y_train, Y_test = train_test_split(X2, Y, test_size=0.20, random_state=1)



model = LogisticRegression()

logreg = model.fit(X2_train, Y_train)

predictions = model.predict(X2_test)

print('accuracy score', accuracy_score(Y_test, predictions))
#KNN

knn = KMeans(n_clusters=2)

knn.fit(X_train, Y_train)



correct = 0 

for i in range(len(X)):

    predict_me = np.array(X[i].astype(float))

    predict_me = predict_me.reshape(-1, len(predict_me))

    prediction = knn.predict(predict_me)

    if prediction[0] == Y[i]:

        correct += 1 



print(correct/len(X))