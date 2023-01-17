# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



import matplotlib.pyplot as plt

%matplotlib inline

import matplotlib 

import seaborn as sns







# Any results you write to the current directory are saved as output.
gen_df = pd.read_csv("../input/gender_submission.csv")

test_df = pd.read_csv("../input/test.csv")

train_df = pd.read_csv("../input/train.csv")



print(gen_df.head())

print(train_df.head())

print(test_df.columns)

print(train_df.columns)

X_raw=pd.concat([test_df,train_df])



Y_raw=X_raw['Survived']



print(X_raw.groupby('Parch').Parch.count())



hist=train_df[train_df['Age'].notnull()]['Age'].hist

train_df[train_df['Age'].notnull()]['Age'].plot.hist()
def normalizeData(df):

    

    

    #df[['Age', 'Fare']]=df[['Age', 'Fare']].fillna(0)

    df['dr']=df['Name'].str.contains('DR.', case=False).astype(int) #get some info from name titles

    df['rev']=df['Name'].str.contains('rev.', case=False).astype(int)

    df['mil']=df['Name'].str.contains('capt.|maj.|gen.|lt.|col.', case=False).astype(int)

    df['Cabin_n']=(df['Cabin'].astype(str).str[0]=='n').astype(int) #n is the only cabin correlated with survival (inversely)

    df=df.drop('Cabin', axis=1)

    df=df.drop('Fare', axis=1) #this is covered by class

    df=df.drop("Ticket", axis=1).drop("Name", axis=1) #also covered by class/cabin

    

    df_dums=pd.get_dummies(df) #create dummy variables for non-continuous class variables

    

    df_dums['age_1']=[1 if (ele <= 5.) & (ele > 0) else 0 for ele in df_dums['Age']]

    df_dums['age_2']=[1 if (ele <= 10.) & (ele > 5.) else 0 for ele in df_dums['Age']]

    df_dums['age_3']=[1 if (ele <= 18.) & (ele > 10.) else 0 for ele in df_dums['Age']]

    df_dums['age_4']=[1 if (ele <= 50.) & (ele > 18.) else 0 for ele in df_dums['Age']]

    df_dums['age_5']=[1 if ele > 50. else 0 for ele in df_dums['Age']]

    df_dums['Class_1']=[1 if ele == 1 else 0 for ele in df_dums['Pclass']]

    df_dums['Class_2']=[1 if ele == 2 else 0 for ele in df_dums['Pclass']]

    df_dums['Class_3']=[1 if ele == 3 else 0 for ele in df_dums['Pclass']]

    df_dums['Parch_0']=[1 if ele == 0 else 0 for ele in df_dums['Parch']]

    df_dums['Parch_1']=[1 if ele == 1 else 0 for ele in df_dums['Parch']]

    df_dums['Parch_2p']=[1 if ele >= 2 else 0 for ele in df_dums['Parch']]

    

    df_dums=df_dums.drop("Pclass", axis=1)

    df_dums=df_dums.drop("Parch", axis=1)

    df_dums=df_dums.drop("Age", axis=1)

        

    f, ax = plt.subplots(figsize=(10, 8))

    corr = df_dums.drop('Sex_male', axis=1).corr()

    sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),

            square=True, ax=ax)

    

    X=df_dums.drop('Sex_male', axis=1)

    #X['Fare']=(X_raw_dums['Fare']-X_raw_dums['Fare'].mean())/X_raw_dums['Fare'].std()

    #X['Age']=(df_dums['Age']-df_dums['Age'].mean())/df_dums['Age'].std()

    

    passID=X['PassengerId']

    X=X.drop('PassengerId',axis=1)

    print(X.head())

    #X=X.drop('Embarked_S',axis=1).drop('Embarked_Q',axis=1) #don't seem predictive

    return X, passID
X_val, pass_test = normalizeData(test_df)

X, pass_train  = normalizeData(train_df)



Y=X['Survived']

X=X.drop('Survived', axis=1)
from sklearn.model_selection import train_test_split



X_train, X_test, Y_train, Y_test = train_test_split(X, Y)

print(X_train.shape)

print(X_test.shape)

print(X_train.columns)
from sklearn.linear_model import LogisticRegression as lr

from sklearn.metrics import classification_report,confusion_matrix

logreg = lr()



f=logreg.fit(X_train,Y_train) #fit a logistic regression model

print(f)

preds=logreg.predict(X_test) #predict on test set





cm=confusion_matrix(Y_test,preds)

print(cm)

print('Accuracy = ', ((cm[0,0]+cm[1,1])/np.sum(cm)))

print('train\n',classification_report(Y_train,logreg.predict(X_train)))

print('test\n',classification_report(Y_test,preds)) #logistic regression is able to predict all of the test examples



ho_predictions = logreg.predict(X_val)

output=pd.DataFrame(pass_test)

output['Survived']=pd.DataFrame(ho_predictions)



output.to_csv('csv_to_submit_lr.csv', index = False)



from sklearn.neural_network import MLPClassifier

classifier = MLPClassifier(solver="lbfgs")



classifier.hidden_layer_sizes = (5)  # Remember funny notation for tuple with single element

classifier.activation = "relu"

classifier.learning_rate_init = 1

classifier.max_iter=100

f=classifier.fit(X_train,Y_train)

print(f)

predictions = classifier.predict(X_test)

cm=confusion_matrix(Y_test,predictions)

print(cm)

print('Accuracy = ', ((cm[0,0]+cm[1,1])/np.sum(cm)))

print('train\n',classification_report(Y_train,classifier.predict(X_train)))

print('test\n',classification_report(Y_test,predictions))



ho_predictions = classifier.predict(X_val)

output=pd.DataFrame(pass_test)

output['Survived']=pd.DataFrame(ho_predictions)

print(output.columns)



output.to_csv('csv_to_submit_mlp.csv', index = False)


from sklearn.ensemble import RandomForestClassifier

from sklearn.datasets import make_classification



clf = RandomForestClassifier(max_depth=3, random_state=0)

f=clf.fit(X_train, Y_train)

print(f,'\n')

print(X_train.columns)

print(clf.feature_importances_)

predictions = clf.predict(X_test)

cm=confusion_matrix(Y_test,predictions)

print(cm)

print('Accuracy = ', ((cm[0,0]+cm[1,1])/np.sum(cm)))

print('train\n',classification_report(Y_train,clf.predict(X_train)))

print('test\n',classification_report(Y_test,predictions))



ho_predictions = clf.predict(X_val)

output=pd.DataFrame(pass_test)

output['Survived']=pd.DataFrame(ho_predictions)



output.to_csv('csv_to_submit_rf.csv', index = False)
from sklearn.cluster import KMeans





kmeans = KMeans(n_clusters=2, random_state=0).fit(X_train)

kmeans.labels_



predictions=kmeans.predict(X_test)



print(confusion_matrix(Y_test,predictions))

print('train\n',classification_report(Y_train,kmeans.predict(X_train)))

print('test\n',classification_report(Y_test,predictions))