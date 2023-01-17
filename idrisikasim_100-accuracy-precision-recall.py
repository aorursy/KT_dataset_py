import pandas as pd , numpy as np , matplotlib.pyplot as plt , seaborn as sns , warnings

%matplotlib inline



warnings.filterwarnings('ignore')



df = pd.read_csv('../input/creditcardfraud/creditcard.csv')



df.head()
df.tail()
df.shape
df.info()
df.Class.value_counts()
df.describe()
tdf = pd.DataFrame(df.isnull().sum(),columns = ['A'])

tdf[tdf.A > 0]    # No null values Found
plt.figure(figsize=(15,5))       # Expanding size

sns.countplot(df['Class'])
adf = df[df['Class'] == 0].iloc[:10000,:]

bdf = df[df['Class'] == 1]

adf = adf.reset_index().drop('index',axis=1)

bdf = bdf.reset_index().drop('index',axis=1)



data = pd.concat([adf,bdf])



print(adf.shape,bdf.shape,data.shape)
plt.figure(figsize=(15,5))       # Expanding size

sns.countplot(data['Class'])     # Reduced but still unbalanced but can be handled
c= data.corr()    # Finding correlation



i = 0



# replacing diogonal corr() which is 1 to NaN for finding

#i.e manipulating and getting the informative features.



while True:    

    try:

        c.iloc[i,i] = np.nan

        i += 1

    except:

        break
# Getting high corr. values w.r.t output because it supports the output...



features = c[(c['Class'] > 0.1) | (c['Class'] < -0.1)].dropna(how = 'all')['Class']     

features_col = list(features.index)

print(features.shape, len(features_col))     # Exactly what i want....
features   # Most Informative features w.r.t correlation...:)
sns.pairplot(data[features_col])    
# Preparing  pipeline for all the models

# Here RandomForestClassifier wins the race...



from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score,roc_auc_score

from sklearn.svm import SVC

from sklearn.ensemble import IsolationForest

from sklearn.metrics import confusion_matrix,classification_report





X = data[features_col]

y = data.Class

X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=3)





def model(X,y):

    models = [LogisticRegression(penalty='l2'),DecisionTreeClassifier()

              ,RandomForestClassifier(),KNeighborsClassifier(),SVC(),IsolationForest()]

    

    for model in models:

        model.fit(X_train,y_train)

        y_pred = model.predict(X_test)

        print(accuracy_score(y_test,y_pred),print(confusion_matrix(y_test,y_pred)) 

             ,print(classification_report(y_test,y_pred)),type(model).__name__)



model(X,y)
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score,roc_auc_score

from sklearn.metrics import confusion_matrix,classification_report





X = data[features_col]

y = data.Class

X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=3)



model = RandomForestClassifier()

model.fit(X_train,y_train)

y_pred = model.predict(X_test)

print(accuracy_score(y_test,y_pred),type(model).__name__)
y_test.value_counts()
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
len(y_pred)
X.columns
len(X.columns)