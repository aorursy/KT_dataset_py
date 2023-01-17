import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
heart_df = pd.read_csv("../input/heart.csv")

heart_df.info()

heart_df.head()

heart_df.describe()
fig, ax = plt.subplots(1,2,figsize=(20,10))

sns.boxplot(x='sex', y='age', hue='target', data=heart_df, orient='v', ax=ax[0])

sns.boxplot(x='target', y='thalach', data=heart_df, ax=ax[1])



sns.pairplot(heart_df,hue='target')
sns.distplot(heart_df['target'], kde=False)
plt.figure(figsize=(60,40))

sns.heatmap(heart_df.corr(),annot=True)

fig, ax = plt.subplots(len(heart_df.drop('target',axis=1).columns),figsize=(30,100))

for i,column in enumerate(heart_df.drop('target',axis=1).columns):

    sns.countplot(x=column,data=heart_df,hue='target',ax=ax[i])

    ax[i].set_title(column, size=18)
sns.heatmap(heart_df.isnull(), yticklabels=False, cmap="viridis")
sns.countplot(x='target',data=heart_df)
X = heart_df.drop('target',axis=1)

y = heart_df['target']

from sklearn.linear_model import LogisticRegression



def perform_logistic_regression(X_train, X_test, C_param,  y_train):

    lr = LogisticRegression(C=C_param)

    lr.fit(X_train, y_train)

    return lr.predict(X_test)



def Normalize_dataframe(raw_df):

    from sklearn.preprocessing import StandardScaler

    return pd.DataFrame(StandardScaler().fit_transform(raw_df))
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25, random_state=133)

norm_X_train, norm_X_test,y_train, y_test = train_test_split(Normalize_dataframe(X),y,test_size=0.25,random_state=133)



y_predicted = perform_logistic_regression(X_train, X_test, 0.5, y_train)

y_predicted_normalized = perform_logistic_regression(norm_X_train, norm_X_test, 0.5, y_train)

from sklearn.metrics import classification_report, confusion_matrix

print('Confusion matrix for unscaled X \n')

print(confusion_matrix(y_predicted,y_test))

print('\n')



print('Confusion matrix for scaled X \n')

print(confusion_matrix(y_predicted_normalized,y_test))

print('\n')

list_of_C = [0.01, 0.1, 1, 10, 100]

from sklearn.model_selection import KFold

from sklearn.metrics import accuracy_score

five_fold = KFold(n_splits=5)

i = 0

y_pred_c = []

accuracies_with_C = []

for C in list_of_C:

    accuracy_score_list = []

    for fold_number,indices in enumerate(five_fold.split(norm_X_train, y_train)):

        y_pred = (perform_logistic_regression(Normalize_dataframe(X).iloc[indices[0],:], Normalize_dataframe(X).iloc[indices[1],:], C, y[indices[0]]))

        accuracy_sc = accuracy_score(y_pred,y[indices[1]])

        accuracy_score_list.append(accuracy_sc)

        print('Iteration: ', fold_number, ' Accuracy Score: ', accuracy_sc)

    accuracies_with_C.append(np.mean(accuracy_score_list))

pd.DataFrame(accuracies_with_C, index=list_of_C, columns=['Accuracy'])



    

    
from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.svm import SVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score



models = {  'Logistic Regression': LogisticRegression(C=0.01),

            'RandomForestClassifier':RandomForestClassifier(n_estimators=500), 

            'DecisionTreeClassifier': DecisionTreeClassifier(), 

            'SVM Classifier': SVC(), 

            'KNN Classifier': KNeighborsClassifier()}
tenFold = KFold(n_splits=10)



def runModels(X,y):

    model_accuracies = {}

    for model in models:

        accuracy_score_list = []

        for fold_number,indices in enumerate(tenFold.split(norm_X_train, y_train)):

            models[model].fit(X=X.iloc[indices[0],:],y=y[indices[0]])

            y_pred = models[model].predict(X.iloc[indices[1],:])

            accuracy_sc = accuracy_score(y_pred,y[indices[1]])

            accuracy_score_list.append(accuracy_sc)

        model_accuracies[model] = np.mean(accuracy_score_list)

    return model_accuracies

normalized_accuracies = pd.DataFrame.from_dict(runModels(Normalize_dataframe(X),y), orient='index', columns=['Scaled inputs'])

regular_accuracies = pd.DataFrame.from_dict(runModels(X,y), orient='index', columns=['Unscaled Inputs'])

pd.concat([normalized_accuracies,regular_accuracies],axis=1)