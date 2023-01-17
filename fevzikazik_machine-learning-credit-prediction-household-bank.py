import pandas as pd, numpy as np

import datetime



from sklearn.model_selection import train_test_split

from sklearn.preprocessing import OneHotEncoder, LabelBinarizer

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import roc_curve, auc, confusion_matrix

from sklearn.model_selection import GridSearchCV



import sklearn

from sklearn_pandas import DataFrameMapper



import matplotlib.pyplot as plt
loans = pd.read_csv("../input/kredi-veri-seti/krediVeriseti.csv", sep=";")

loans.head(10)
features = ['krediMiktari',

            'yas',

            'evDurumu',

            'aldigi_kredi_sayi',

            'telefonDurumu',

           ]

result='KrediDurumu'
clean_data=loans[features+[result]].dropna()

clean_data.head()
numerical_cols=['krediMiktari', 'yas', 'aldigi_kredi_sayi']



categorical_cols=['evDurumu', 'telefonDurumu']



mapper_features = DataFrameMapper([

('evDurumu',sklearn.preprocessing.LabelBinarizer()),

('telefonDurumu', sklearn.preprocessing.LabelBinarizer()),

    ])



X1=mapper_features.fit_transform(clean_data)





X2=np.array(clean_data[numerical_cols])





X = np.hstack((X1,X2))



y=np.array(sklearn.preprocessing.LabelBinarizer().fit_transform(clean_data[result]))
X
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=100, stratify=y)
# Sınıflandırma Modellerine Ait Kütüphaneler

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVR

from sklearn.tree import DecisionTreeClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier, BaggingClassifier



# Modelleri Hazırlayalım

models = []

models.append(('Logistic Regression', LogisticRegression()))

models.append(('GradientBoosting', GradientBoostingClassifier(n_estimators=100)))

models.append(('Naive Bayes', GaussianNB()))

models.append(('Decision Tree (NoParam)', DecisionTreeClassifier())) 

models.append(('Decision Tree (GridSearch)', GridSearchCV(DecisionTreeClassifier(), {'max_depth':[5, 10, 15, 20, 25, 32]}, cv=5)))

models.append(('RandomForestClassifier (GridSearch)', GridSearchCV(RandomForestClassifier(), {'max_depth':[5, 15], 'n_estimators':[10,30]})))

models.append(('RandomForestClassifier (2 Param)', RandomForestClassifier(n_estimators=10, criterion='entropy')))

models.append(('KNeighborsClassifier', KNeighborsClassifier(n_neighbors=5, metric='minkowski')))

# models.append(('Support Vector Regression', SVR(kernel='rbf')))

models.append(('LinearDiscriminantAnalysis', LinearDiscriminantAnalysis()))

models.append(('AdaBoostClassifier', AdaBoostClassifier(learning_rate=0.5)))

models.append(('BaggingClassifier', BaggingClassifier()))



model_name = []

acc_score = []

from sklearn.metrics import classification_report

# Modelleri test edelim

for name, model in models:

    model = model.fit(X_train, y_train.ravel())

    y_pred = model.predict(X_test)

    from sklearn import metrics

    # print("%s -> ACC: %%%.2f" % (name,metrics.accuracy_score(y_test, y_pred)*100))

    model_name.append(name)

    acc_score.append((metrics.accuracy_score(y_test, y_pred)*100))

    print('Model: ', name)

    print('Confusion Matrix: \n', metrics.confusion_matrix(y_test, y_pred))

    #Detaylı sınıflandırma raporu...

    report = classification_report(y_test, y_pred)

    print('\n ',report)

    

col={'Model':model_name,'Oran':acc_score}

comp = pd.DataFrame(data=col)

comp
import pickle

with open('models.pickle', 'wb') as output:

    pickle.dump(models, output)



with open('X_train.pickle', 'wb') as output:

    pickle.dump(X_train, output)



with open('y_train.pickle', 'wb') as output:

    pickle.dump(y_train, output)



with open('mapper_features.pickle', 'wb') as output:

    pickle.dump(mapper_features, output)
def preProcess(a):

    data=list(a.values())

    colz=list(a.keys())

    dfx=pd.DataFrame(data=[data], columns=colz)



    XX1=mapper_features.transform(dfx)

    XX2=dfx[numerical_cols]

    XX = np.hstack((XX1,XX2))

    return XX
sample_data={ 'krediMiktari': 5000,

 'yas': 50,

 'aldigi_kredi_sayi': 5,

 'evDurumu': 'evsahibi',

 'telefonDurumu': 'var'}



sample_result=preProcess(sample_data)

sample_result
model_fitted_name = []

acc_score_model = []

from sklearn.metrics import classification_report

# Modelleri test edelim

for name, model in models:

    model = model.fit(X_train, y_train.ravel())

    from sklearn import metrics

    model_fitted_name.append(name)

    acc_score_model.append(((model.predict_proba(sample_result)[:,0][0])*100))

    

columns = {'Model':model_fitted_name,'Oran':acc_score_model}

results = pd.DataFrame(data=columns)





def karar_ver(oran):

    return (oran >= 75)



results['Karar'] = results['Oran'].apply(karar_ver)

results

#results.sort_values('Oran', ascending=False)