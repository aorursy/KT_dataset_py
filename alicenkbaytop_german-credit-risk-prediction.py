import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import pandas_profiling 

import seaborn as sns # visualization

import matplotlib.pyplot as plt # visualization

from sklearn.preprocessing import LabelEncoder # label encoding

from sklearn.model_selection import train_test_split # train, test split

from sklearn.preprocessing import StandardScaler # normalization

from sklearn.neighbors import KNeighborsClassifier # KNN model

from sklearn.svm import SVC # SVC model

from xgboost import XGBClassifier # XGBoost model

from sklearn.model_selection import GridSearchCV, cross_val_score # Gridsearch 

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve# results



import warnings # ignore warning

warnings.filterwarnings("ignore")



%matplotlib inline



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# reading and copying data

data = pd.read_csv("/kaggle/input/german-credit-data-with-risk/german_credit_data.csv", index_col = "Unnamed: 0")

df = data.copy()
# overview of data

df.head()
df.info()
df.describe().T
columns = ["Age","Sex","Job","Housing","Saving accounts","Checking account","Credit amount","Duration","Purpose","Risk"]



def unique_value(data_set, column_name):

    return data_set[column_name].nunique()



print("Number of the Unique Values:\n",unique_value(df, columns))    
# Missing Value Table

def missing_value_table(df):

    missing_value = df.isna().sum().sort_values(ascending=False)

    missing_value_percent = 100 * df.isna().sum()//len(df)

    missing_value_table = pd.concat([missing_value, missing_value_percent], axis=1)

    missing_value_table_return = missing_value_table.rename(columns = {0 : 'Missing Values', 1 : '% Value'})

    cm = sns.light_palette("lightgreen", as_cmap=True)

    missing_value_table_return = missing_value_table_return.style.background_gradient(cmap=cm)

    return missing_value_table_return

  

missing_value_table(df)
date_int = ["Purpose", 'Sex']

cm = sns.light_palette("lightgreen", as_cmap=True)

pd.crosstab(df[date_int[0]], df[date_int[1]]).style.background_gradient(cmap = cm)
fig, ax = plt.subplots(1,2,figsize=(15,5))



sns.countplot(df['Sex'], ax=ax[0]).set_title('Male - Female Ratio');

sns.countplot(df.Risk, ax=ax[1]).set_title('Good - Bad Risk Ratio');
fig, ax = plt.subplots(2,1,figsize=(15,5))

plt.tight_layout(2)

sns.lineplot(data=df, x='Age', y='Credit amount', hue='Sex', lw=2, ax=ax[0]).set_title("Credit Amount Graph Depending on Age and Duration by Sex", fontsize=15);

sns.lineplot(data=df, x='Duration', y='Credit amount', hue='Sex', lw=2, ax=ax[1]);
sns.countplot(x="Housing", hue="Risk", data=df).set_title("Housing and Frequency Graph by Risk", fontsize=15);

plt.show()
fig, (ax1, ax2) = plt.subplots(1,2, figsize=(10,6))

sns.countplot(x="Saving accounts", hue="Risk", data=df, ax=ax1);

sns.countplot(x="Checking account", hue="Risk", data=df, ax=ax2);

ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45)

ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45)

fig.show()
fig, ax = plt.subplots(1,3,figsize=(20,5))

plt.suptitle('Box Plots of Age, Duration and Credit amount.',fontsize = 15)

sns.boxplot(df['Credit amount'], ax=ax[0]);

sns.boxplot(df['Duration'], ax=ax[1]);

sns.boxplot(df['Age'], ax=ax[2]);

plt.show()
cor = df.corr()

sns.heatmap(cor, annot=True).set_title("Correlation Graph of Data Set",fontsize=15);

plt.show()
# Label Encoding

columns_label = ["Sex","Risk"]

labelencoder = LabelEncoder()

for i in columns_label:

    df[i] = labelencoder.fit_transform(df[i])
Cat_Age = []

for i in df["Age"]:

    if i<25:

        Cat_Age.append("0-25")

    elif (i>=25) and (i<30):

        Cat_Age.append("25-30")

    elif (i>=30) and (i<35):

        Cat_Age.append("30-35")

    elif (i>=35) and (i<40):

        Cat_Age.append("35-40")

    elif (i>=40) and (i<50):

        Cat_Age.append("40-50")

    elif (i>=50) and (i<76):

        Cat_Age.append("50-75")

        

df["Cat Age"] = Cat_Age        
# Get Dummies

columns_dummy = ['Housing','Saving accounts','Checking account',"Purpose","Cat Age"]

for i in columns_dummy:

    df = pd.concat([df, pd.get_dummies(df[i])], axis=1)
df.drop(['Housing','Saving accounts','Checking account',"Purpose","Age","Cat Age"], axis = 1, inplace=True)
y = df.Risk

X = df.drop("Risk", axis = 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.fit_transform(X_test)
knn_model = KNeighborsClassifier(n_neighbors = 3)

knn_model.fit(X_train, y_train)

y_pred = knn_model.predict(X_test) 

print('With KNN (K=3) accuracy is: ',knn_model.score(X_test,y_test))

print(confusion_matrix(y_test, y_pred))

print(classification_report(y_test, y_pred))
neig = np.arange(1, 25)

train_accuracy = []

test_accuracy = []

# Loop over different values of k

for i, k in enumerate(neig):

    # k from 1 to 25(exclude)

    knn_model = KNeighborsClassifier(n_neighbors=k)

    # Fit with knn

    knn_model.fit(X_train,y_train)

    #train accuracy

    train_accuracy.append(knn_model.score(X_train, y_train))

    # test accuracy

    test_accuracy.append(knn_model.score(X_test, y_test))



# Plot

plt.figure(figsize=[12,6])

plt.plot(neig, test_accuracy, label = 'Testing Accuracy')

plt.plot(neig, train_accuracy, label = 'Training Accuracy')

plt.legend()

plt.title('-value VS Accuracy')

plt.xlabel('Number of Neighbors')

plt.ylabel('Accuracy')

plt.xticks(neig)

plt.savefig('graph.png')

plt.show()

print("Best accuracy is {} with K = {}".format(np.max(test_accuracy),1+test_accuracy.index(np.max(test_accuracy))))
knn_model = KNeighborsClassifier(n_neighbors = 23)

knn_model.fit(X_train, y_train)

y_pred = knn_model.predict(X_test) 

print('With KNN (K=23) accuracy is: ',knn_model.score(X_test,y_test))

print(confusion_matrix(y_test, y_pred))

print(classification_report(y_test, y_pred))
#Predicting proba

y_pred_prob = knn_model.predict_proba(X_test)[:,1]



# Generate ROC curve values: fpr, tpr, thresholds

fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)



# Plot ROC curve

plt.plot([0, 1], [0, 1], 'k--')

plt.plot(fpr, tpr)

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('ROC Curve')

plt.show()
#Predicting proba

y_pred_prob = model.predict_proba(X_test)[:,1]



# Generate ROC curve values: fpr, tpr, thresholds

fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)



# Plot ROC curve

plt.plot([0, 1], [0, 1], 'k--')

plt.plot(fpr, tpr)

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('ROC Curve')

plt.show()
svc_model = SVC(kernel = "rbf").fit(X_train, y_train)

y_pred = svc_model.predict(X_test)

print("Accuracy Score:", accuracy_score(y_test, y_pred))
svc_params ={"C": [0.0001, 0.001, 0.01, 0.1, 1, 5, 10, 50, 100]

             ,"gamma": [0.0001, 0.001, 0.01, 0.1, 1, 5, 10, 50, 100]}

svc = SVC()

svc_cv_model = GridSearchCV(svc, svc_params, cv = 10, n_jobs = -1, verbose = 2)

svc_cv_model.fit(X_train, y_train)
print("Best Parameters: "+ str(svc_cv_model.best_params_))
svc_tuned = SVC(C = 10, gamma = 0.01).fit(X_train, y_train)

y_pred = svc_model.predict(X_test)

print("Accuracy Score:", accuracy_score(y_test, y_pred))
xgb = XGBClassifier()

xgb.fit(X_train, y_train)

y_pred_xgb = xgb.predict(X_test)

print(accuracy_score(y_pred_xgb, y_test))
xgb_params = {"n_estimators": [100, 500, 1000, 2000],

             "subsample": [0.6, 0.8, 1.0],

             "max_depth": [3, 4, 5, 6],

             "learning_rate": [0.1, 0.01, 0.02, 0.05],

             "min_samples_split": [2,5,10]}

xgb = XGBClassifier()

xgb_cv_model = GridSearchCV(xgb, xgb_params, cv = 10, n_jobs = -1, verbose = 2)

xgb_cv_model.fit(X_train, y_train)
print("Best Parameters: "+ str(xgb_cv_model.best_params_))
xgb = XGBClassifier(learning_rate = 0.05, max_depth = 5, min_samples_split=2,n_estimators=100,subsample=0.8 )

xgb_tuned = xgb.fit(X_train,y_train)

y_pred = xgb_tuned.predict(X_test)

print("Accuracy Score:", accuracy_score(y_test, y_pred))