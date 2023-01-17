# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

# Load in the train and test datasets

def readData():

    train = pd.read_csv('../input/titanic/train.csv')

    test = pd.read_csv('../input/titanic/test.csv')

    print('Training size: ' , train.shape)

    print('Test size: ' , test.shape)

    

    return train, test



def create_df(df,df2):

    df = df.append(df2, ignore_index=True, sort=False)

    print('df size: ' , df.shape)

    return df



train, test = readData()

# Get some useful info 

pas_id = test['PassengerId']

y = train['Survived']



# apand train and test

df = create_df(train,test)

df.head()
# Explore Target

def target_info(y):

    sns.catplot(data=train,kind='count',x='Survived',hue='Survived')

    print('Percentance of Survivors: ' , (sum(y) / len(y)) *100 , '%') 

    

target_info(train.Survived)
# Missing values by columns

def missing_values_info(df,which_dataset):

    print("Percentance of Missing Values Per Columns in " + which_dataset + " dataset.")

    print("------------------------------------------")

    print((df.isnull().sum() / df.shape[0]) * 100)

missing_values_info(train,"train")

missing_values_info(test,"test")

missing_values_info(df,"all_data")
# What about Survived with non nan Cabin values

def cabin_process1(df): # Feature Engineering

    # Create a new feature with only the first letter of Cabin

    df['Cabin'] = df['Cabin'].astype(str).str[0]

    return df

    

def plot_Cabin1(df):

    # Plot new Cabin

    sns.catplot(data=df,kind='count',x='Cabin',hue='Survived')

    print('Cabin equal to "n" are the nan values of Cabin')



    

df = cabin_process1(df)

plot_Cabin1(df)

def cabin_process2(df): # Feature Engineering

    # Create a new feature Cobin / Not Cabin

    df['Cabin_or_not'] = np.where(df['Cabin'] == 'n', 0, 1)

    return df



def plot_Cabin2(df):

    # Plot new Cabin

    sns.catplot(data=df,kind='count',x='Cabin_or_not',hue='Survived')



df = cabin_process2(df)

plot_Cabin2(df)

del df['Cabin']
# Hanndle Age Missing values with 2 different ways:

# 1. Replace by mean/median of the same class

# 2. Regression

def process_age(df):



    df['Age'] = df['Age'].fillna(100)

    return df

df = process_age(df)

df.groupby("Survived").Age.hist(alpha=0.6)

df.Embarked.describe()
# fill Nan with most important values

def Embarked_process(df):

    df.Embarked = df.Embarked.fillna(df.Embarked.value_counts().idxmax())

    return df

df = Embarked_process(df)

# See Embarked based on target

sns.catplot(data=df,kind='count',x='Embarked',hue='Survived')
# Same tickets

def create_same_tickets(df):

    df['same_ticket'] = df.groupby(['Ticket'])['PassengerId'].transform("count")

    df['same_ticket'] = np.where( df['same_ticket'] == 1 , 0, 1)

    return df

df = create_same_tickets(df)

sns.catplot(data=df,kind='count',x='same_ticket',hue='Survived')




def process_Ticket1(train):



    train['Ticket_as_str'] = train.Ticket.str.replace('\d+', '')





    train.Ticket_as_str = np.where( (train.Ticket_as_str == 'SOTON/O.Q. ') | 

                                   (train.Ticket_as_str == 'STON/O. ') | 

                                   (train.Ticket_as_str == 'STON/O . ') | 

                                   (train.Ticket_as_str == 'S.O.C. ') |

                                   (train.Ticket_as_str == 'SO/C ') |

                                   (train.Ticket_as_str == 'STON/OQ. ') |

                                   (train.Ticket_as_str == 'CA ') |

                                    (train.Ticket_as_str == 'CA. ') | 

                                   (train.Ticket_as_str == 'C.A./SOTON ') |

                                   (train.Ticket_as_str == 'C ') |

                                   (train.Ticket_as_str == 'C.A. ') |

                                   (train.Ticket_as_str == 'SOTON/O '), 'SOTON/OQ ', train.Ticket_as_str)



    train.Ticket_as_str = np.where( (train.Ticket_as_str == 'A/. ') | 

                                   (train.Ticket_as_str == 'A./. ') | 

                                   (train.Ticket_as_str == 'A.. ') | 

                                   (train.Ticket_as_str == 'A/S ') | 

                                   (train.Ticket_as_str == 'AQ/. ') | 

                                   (train.Ticket_as_str == 'AQ/ ') |

                                   (train.Ticket_as_str == 'A. . ') | 

                                   (train.Ticket_as_str == 'A/ '), 'A. ', train.Ticket_as_str)





    train.Ticket_as_str = np.where((train.Ticket_as_str == 'PC ') | (train.Ticket_as_str == 'F.C.C. ') | (train.Ticket_as_str == 'Fa '), 'F.C. ', train.Ticket_as_str)



    train.Ticket_as_str = np.where((train.Ticket_as_str == 'W./C. ') 

                                   | (train.Ticket_as_str == 'WE/P ') |

                                   (train.Ticket_as_str == 'W.E.P. '), 'W/C ', train.Ticket_as_str)



    train.Ticket_as_str = np.where( (train.Ticket_as_str == 'S.C./A.. ') | 

                                   (train.Ticket_as_str == 'SC/AH Basle ') |

                                   (train.Ticket_as_str == 'SC/AH ') |

                                   (train.Ticket_as_str == 'S.C./PARIS ') | 

                                   (train.Ticket_as_str == 'SC/Paris ') |

                                   (train.Ticket_as_str == 'SC/PARIS ') |

                                   (train.Ticket_as_str == 'SC/A ') |

                                   (train.Ticket_as_str == 'SC/A. ') |

                                   (train.Ticket_as_str == 'SCO/W ') |

                                    (train.Ticket_as_str == 'SC' ) |

                                   (train.Ticket_as_str == 'S.O./P.P. ') | 

                                   (train.Ticket_as_str == 'SW/PP ') |

                                   (train.Ticket_as_str == 'S.W./PP ') |

                                   (train.Ticket_as_str == 'S.P. ') | 

                                    (train.Ticket_as_str == 'SC ') | 

                                   (train.Ticket_as_str == 'P/PP ') |

                                   (train.Ticket_as_str == 'LP ') |

                                   (train.Ticket_as_str == 'S.O.P. '), 'PP ', train.Ticket_as_str)





    return df



df = process_Ticket1(df)

print(df.Ticket_as_str.unique())

ax = sns.catplot(data=df,kind='count',x='Ticket_as_str',hue='Survived', height=5.27, aspect=11.7/8.27)

def process_Ticket2(train):





    train.Ticket_as_str = np.where( (train.Ticket_as_str == 'A. ') | 

                                   (train.Ticket_as_str == 'PP ') | 

                                   (train.Ticket_as_str == 'W/C ') | 

                                   (train.Ticket_as_str == 'LINE'), 'SOTON/OQ ', train.Ticket_as_str)

    

    return df



df = process_Ticket2(df)

print(df.Ticket_as_str.unique())

ax = sns.catplot(data=df,kind='count',x='Ticket_as_str',hue='Survived', height=5.27, aspect=11.7/8.27)
def process_Ticket3(train):

    train.Ticket_as_str = np.where((train.Ticket_as_str == ''), 'SOTON/OQ ', train.Ticket_as_str)

    return df



df = process_Ticket3(df)

print(df.Ticket_as_str.unique())

ax = sns.catplot(data=df,kind='count',x='Ticket_as_str',hue='Survived', height=5.27, aspect=11.7/8.27)
#Name 

def process_name(df):

    #Keep only surname

    df['newName'] = df.Name.str.split(',').str[0]

    # fit with ticket in order to find the family members

    df['family_members'] = df.groupby(['newName','Ticket'])['PassengerId'].transform("count")

    del df['newName']

    del df['Name']

    del df['Ticket']

    return df

df = process_name(df)

# SibSp

sns.catplot(data=df ,kind='count',x='family_members',hue='Survived')
#Name 

def process_name(df):

    #Keep only surname

    # fit with ticket in order to find the family members

    df['family_members'] = np.where(df['family_members'] == 1, 0,df['family_members'])

    df['family_members'] = np.where(df['family_members'] > 4, 5,df['family_members'])

    df['family_members'] = np.where((df['family_members'] > 0) & (df['family_members'] < 5), 1,df['family_members'])

    df['family_members'] = df['family_members'].map({0:'Alone', 1:'Family', 5:'Big_Family'})

    return df

df = process_name(df)

# SibSp

sns.catplot(data=df ,kind='count',x='family_members',hue='Survived')
def process_SibSp(df):    

    df['SibSp_more_survivors'] = np.where(df['SibSp'] == 1 , 1 , 0)

    df['SibSp_more_survivors'].describe()

    del df['SibSp']

    return df

df = process_SibSp(df)

# SibSp

sns.catplot(data=df ,kind='count',x='SibSp_more_survivors',hue='Survived')
def process_parch(df):

    df['Parch_more_survivors'] = np.where((df['Parch'] == 1) | (df['Parch']  == 2 ), 1 , 0)

    df['Parch_more_survivors'].describe()

    del df['Parch']

    return df



df = process_parch(df)

# Parch

sns.catplot(data=df ,kind='count',x='SibSp_more_survivors',hue='Survived')
#Map Sex

def process_sex(df):

    df['Sex'] = df['Sex'].map({'male':1,'female':0})

    return df

df = process_sex(df)

sns.catplot(data=df ,kind='count',x='Sex',hue='Survived')
# Replace Missing Ages with mean

def replace_with_avg_fare(df):

    df['Fare'] = df['Fare'].fillna(0)

    return df



df = replace_with_avg_fare(df)

df.groupby("Survived").Fare.hist(alpha=0.6)

del df['PassengerId']

df = pd.get_dummies(df)
def normalize(df):

    features = df.columns.values

    for feature in features:

        mean, std = df[feature].mean(), df[feature].std()

        df[feature] = (df[feature] - mean) / std 

    return df



df = normalize(df)
#Data Model

train = df[df.Survived.notnull()]

test = df[df.Survived.isnull()]

del train['Survived']

del test['Survived']
#Using Pearson Correlation

plt.figure(figsize=(12,10))

cor_df = df

cor_df['y'] = y

cor = cor_df.corr()

sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)

plt.show()
from sklearn.decomposition import PCA



pca = PCA(n_components=2)

principalComponents = pca.fit_transform(train)

principalDf = pd.DataFrame(data = principalComponents

             , columns = ['principal component 1', 'principal component 2'])



finalDf = pd.concat([principalDf, y], axis = 1)



fig = plt.figure(figsize = (8,8))

ax = fig.add_subplot(1,1,1) 

ax.set_xlabel('Principal Component 1', fontsize = 19)

ax.set_ylabel('Principal Component 2', fontsize = 19)

ax.set_title('2 component PCA', fontsize = 20)

targets = [1, 0]

colors = ['r', 'g']

for target, color in zip(targets,colors):

    indicesToKeep = finalDf['Survived'] == target

    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']

               , finalDf.loc[indicesToKeep, 'principal component 2']

               , c = color

               , s = 50)

ax.legend(targets)

ax.grid()
from sklearn.linear_model import LassoCV



reg = LassoCV()

reg.fit(train, y)

print("Best alpha using built-in LassoCV: %f" % reg.alpha_)

print("Best score using built-in LassoCV: %f" %reg.score(train,y))

coef = pd.Series(reg.coef_, index = train.columns)



imp_coef = coef.sort_values()

import matplotlib

matplotlib.rcParams['figure.figsize'] = (8.0, 10.0)

imp_coef.plot(kind = "barh")

plt.title("Feature importance using Lasso Model")
best_features = ['Sex', 'Pclass','family_members_Big_Family','Age','Embarked_S','Parch_more_survivors','Fare','Cabin_or_not','family_members_Family']

train = train[best_features]

test = test[best_features]


from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split



depth = []

preds = []

preds_tr = []

# create training and testing vars

X_train, X_test, y_train, y_test = train_test_split(train, y, test_size=0.4)

min_depth = 1

max_depth = 30

for i in range(min_depth,max_depth):

    clf = DecisionTreeClassifier(max_depth=i)

    clf = clf.fit(X_train,y_train)

    depth.append((i,clf.score(X_test,y_test)))

    preds.append(clf.predict(X_test).astype(int))

    preds_tr.append(clf.predict(X_train).astype(int))



# Error Analysis

from sklearn.metrics import f1_score



x_axis_cv = []

x_axis_tr = []

y_axis_cv = []

y_axis_tr = []

max_score = 0

d = -1

for i in range(len(preds)):

    f1_cv = f1_score(y_test, preds[i], average='macro')

    f1_tr = f1_score(y_train, preds_tr[i], average='macro')

    if (f1_cv > max_score):

        d = depth[i][0]

        max_score = f1_cv

    x_axis_cv.append(depth[i][0])

    x_axis_tr.append(depth[i][0])

    y_axis_cv.append(f1_cv)

    y_axis_tr.append(f1_tr)

 

plt.plot(x_axis_cv,y_axis_cv, 'g-',label='cross validation')

plt.plot(x_axis_tr,y_axis_tr, 'r-',label='training')

plt.xlabel ('depth')

plt.ylabel ('f1 Score')

plt.title('F1 score based on "max_depth" hyperparameter')

plt.legend()



# Final Run

dtree = DecisionTreeClassifier(max_depth=d)

dtree.fit(train,y)

dt_predictions = dtree.predict(test).astype(int)



#Create a  DataFrame with the passengers ids and our prediction regarding whether they survived or not

submission = pd.DataFrame({'PassengerId':pas_id,'Survived':dt_predictions})



#Visualize the first 5 rows

submission.head()



filename = 'dt_predictions.csv'



submission.to_csv(filename,index=False)
#SVM

from sklearn.svm import SVC



c_hyper = []

preds = []

preds_tr = []



# create training and testing vars

X_train, X_test, y_train, y_test = train_test_split(train, y, test_size=0.4)



min_c = 1

max_c = 100

for i in range(min_c,max_c):

    clf = SVC(gamma=0.05,C = i/20)

    clf = clf.fit(X_train,y_train)

    c_hyper.append((i/20,clf.score(X_test,y_test)))

    preds.append(clf.predict(X_test).astype(int))

    preds_tr.append(clf.predict(X_train).astype(int))



# Error Analysis

from sklearn.metrics import f1_score



x_axis_cv = []

y_axis_cv = []

x_axis_tr = []

y_axis_tr = []

max_score = 0

c = -1

for i in range(len(preds)):

    f1_cv = f1_score(y_test, preds[i], average='macro')

    f1_tr = f1_score(y_train, preds_tr[i], average='macro')

    if (f1_cv > max_score):

        c = c_hyper[i][0]

        max_score = f1_cv

    x_axis_cv.append(c_hyper[i][0])

    y_axis_cv.append(f1_cv)

    x_axis_tr.append(c_hyper[i][0])

    y_axis_tr.append(f1_tr)

    

 

plt.plot(x_axis_cv,y_axis_cv, 'g-',label='cross validation')

plt.plot(x_axis_tr,y_axis_tr, 'r-',label='training')

plt.xlabel ('C')

plt.ylabel ('f1 Score')

plt.title('F1 score based on "C " hyperparameter')

plt.legend()







# Final Run

clf = SVC(gamma=0.05,C = c)

clf = clf.fit(train,y)

svm_predictions = clf.predict(test).astype(int)



#Create a  DataFrame with the passengers ids and our prediction regarding whether they survived or not

submission = pd.DataFrame({'PassengerId':pas_id,'Survived':svm_predictions})



#Visualize the first 5 rows

submission.head()



filename = 'svm_predictions.csv'



submission.to_csv(filename,index=False)
#XGBoost

import xgboost as xgb



# create training and testing vars

X_train, X_test, y_train, y_test = train_test_split(train, y, test_size=0.4)

l_hyper = []

preds = []

preds_tr = []



min_c = 1

max_c = 100

for i in range(min_c,max_c):

    xgb_model = xgb.XGBClassifier(objective="binary:logistic", random_state=42,learning_rate=i/100, n_estimators=500)

    xgb_model.fit(X_train, y_train)

    l_hyper.append((i/100,xgb_model.score(X_test,y_test)))

    preds.append(xgb_model.predict(X_test).astype(int))

    preds_tr.append(xgb_model.predict(X_train).astype(int))



# Error Analysis

from sklearn.metrics import f1_score



x_axis_cv = []

y_axis_cv = []

x_axis_tr = []

y_axis_tr = []

max_score = 0

l = -1

for i in range(len(preds)):

    f1_cv = f1_score(y_test, preds[i], average='macro')

    f1_tr = f1_score(y_train, preds_tr[i], average='macro')

    if (f1_cv > max_score):

        l = l_hyper[i][0]

        max_score = f1_cv

    x_axis_cv.append(l_hyper[i][0])

    y_axis_cv.append(f1_cv)

    x_axis_tr.append(l_hyper[i][0])

    y_axis_tr.append(f1_tr)

    

 

plt.plot(x_axis_cv,y_axis_cv, 'g-',label='cross validation')

plt.plot(x_axis_tr,y_axis_tr, 'r-',label='training')

plt.xlabel ('learning rate')

plt.ylabel ('f1 Score')

plt.title('F1 score based on "lambda regularization " hyperparameter')

plt.legend()







# Final Run

xgb_model = xgb.XGBClassifier(objective="binary:logistic", random_state=42,learning_rate=l, n_estimators=500)

xgb_model.fit(train, y)

xgboost_predictions = clf.predict(test).astype(int)



#Create a  DataFrame with the passengers ids and our prediction regarding whether they survived or not

submission = pd.DataFrame({'PassengerId':pas_id,'Survived':xgboost_predictions})



#Visualize the first 5 rows

submission.head()



filename = 'xgboost_predictions.csv'



submission.to_csv(filename,index=False)