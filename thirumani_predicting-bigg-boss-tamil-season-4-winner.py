import numpy as np

import pandas as pd

pd.set_option('display.max_columns', 50)



import matplotlib.pyplot as plt

import seaborn as sns



import warnings

warnings.filterwarnings('ignore')



from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score,f1_score,roc_auc_score,confusion_matrix,roc_curve,auc

from sklearn.preprocessing import StandardScaler,PolynomialFeatures,QuantileTransformer

from sklearn.decomposition import PCA

from sklearn.neural_network import MLPClassifier
!cd /kaggle/input/bigg-boss-india-hindi-telugu-tamil-kannada; ls -tlr
bigg_boss = pd.read_csv('/kaggle/input/bigg-boss-india-hindi-telugu-tamil-kannada/Bigg_Boss_India.csv', encoding = "ISO-8859-1")

nRow, nCol = bigg_boss.shape

print(f'There are {nRow} rows and {nCol} columns')
bigg_boss.head(5)
bigg_boss.tail(10).T
bigg_boss.sample(10)
bigg_boss.info()
bigg_boss.describe()
# Unique values in each column

for col in bigg_boss.columns:

    print("Number of unique values in", col,"-", bigg_boss[col].nunique())
# Number of seasons in all Indian languages

print(bigg_boss.groupby('Language')['Season Number'].nunique().sum())



# 35 seasons happened (including current seasons)
# Number of seasons in each Indian language

print(bigg_boss.groupby('Language')['Season Number'].nunique().nlargest(10))
# Total number of Bigg Boss housemates

fig = plt.figure(figsize=(10,4))

ax = sns.countplot(x='Language', data=bigg_boss)

ax.set_title('Bigg Boss Series - Indian Language')

for t in ax.patches:

    if (np.isnan(float(t.get_height()))):

        ax.annotate(0, (t.get_x(), 0))

    else:

        ax.annotate(str(format(int(t.get_height()), ',d')), (t.get_x(), t.get_height()*1.01))
# Number of normal entries and wild card entries

print(bigg_boss['Wild Card'].value_counts(), "\n")

print(round(bigg_boss['Wild Card'].value_counts(normalize=True)*100))

sns.countplot(x='Wild Card', data=bigg_boss)
# Common people has many professions, so clubbing them into one category

bigg_boss.loc[bigg_boss['Profession'].str.contains('Commoner'),'Profession']='Commoner'
# Participant's Profession

print(bigg_boss['Profession'].value_counts())

fig = plt.figure(figsize=(20,5))

sns.countplot(x='Profession', data=bigg_boss)

plt.xticks(rotation=90)
# Broadcastor

fig = plt.figure(figsize=(20,5))

ax = sns.countplot(x='Broadcasted By', data=bigg_boss, palette='RdBu')

ax.set_title('Bigg Boss Series - Indian Broadcastor & Total Number of Housemates')

for t in ax.patches:

    if (np.isnan(float(t.get_height()))):

        ax.annotate(0, (t.get_x(), 0))

    else:

        ax.annotate(str(format(int(t.get_height()), ',d')), (t.get_x(), t.get_height()*1.01))
bigg_boss.groupby('Host Name')['Season Number'].nunique().nlargest(25)
# Housemate's Gender

print(bigg_boss['Gender'].value_counts())

# Maximum TRP of Bigg Boss Hindi/India seasons

print("Maximum TRP",bigg_boss['Average TRP'].max(), "\n")

print(bigg_boss.loc[bigg_boss['Average TRP']==bigg_boss['Average TRP'].max()][["Language","Season Number"]].head(1).to_string(index=False))
# Longest season of Bigg Boss Hindi/India seasons

print("Longest season",bigg_boss['Season Length'].max(), "days \n")

print(bigg_boss.loc[bigg_boss['Season Length']==bigg_boss['Season Length'].max()][["Language","Season Number"]].head(1).to_string(index=False))
# All BB Winners

bigg_boss.loc[bigg_boss.Winner==1]
# Profession of BB Season Winners

bigg_boss.loc[bigg_boss.Winner==1,'Profession'].value_counts()
# Gender of Season title Winners

print(bigg_boss.loc[bigg_boss.Winner==1,'Gender'].value_counts(),'\n')



# In percentage

print(round(bigg_boss.loc[bigg_boss.Winner==1,'Gender'].value_counts(normalize=True)*100))



# Male      22  (71%)

# Female     9  (29%)
# Entry type of the Season Winners

bigg_boss.loc[bigg_boss.Winner==1,'Wild Card'].value_counts()
# No re-entered contestant won Bigg Boss title

bigg_boss.loc[bigg_boss.Winner==1,'Number of re-entries'].value_counts()
# Number of eliminations or evictions faced by the Bigg Boss competition winners

bigg_boss.loc[bigg_boss.Winner==1,'Number of Evictions Faced'].value_counts().sort_index()



# Number of eliminations faced - Number of Winners
# Bigg Boss title winner number of times elected as Captain

bigg_boss.loc[bigg_boss.Winner==1,'Number of times elected as Captain'].value_counts().sort_index()



# Number of times elected as Captain   - Number of winners
lang='Tamil'



# All Bigg Boss Tamil Participants

bigg_boss.loc[(bigg_boss['Language']==lang)]
# Bigg Boss Tamil Winners

bigg_boss.loc[(bigg_boss['Language']==lang) & (bigg_boss['Winner']==1), :]
# Bigg Boss Tamil current season participants

bigg_boss.loc[(bigg_boss['Language']==lang) & (bigg_boss['Winner'].isnull()), :]
# Handling NULL values

bigg_boss.isnull().sum()
# Removing records where Name field is empty

bigg_boss = bigg_boss.loc[bigg_boss.Name.notnull()]

bigg_boss.reset_index(drop=True,inplace=True)
# Contestant might have faced at least one eviction, so filling NaN with 'Number of Evictions Faced' with 1

bigg_boss['Number of Evictions Faced'] = bigg_boss['Number of Evictions Faced'].fillna(1)



# Number of re-entries are very less, so filling NULLs in 'Number of re-entries' with 0

bigg_boss['Number of re-entries'] = bigg_boss['Number of re-entries'].fillna(0)



# Filling blank values in 'Average TRP' column with average

bigg_boss['Average TRP'] = bigg_boss['Average TRP'].fillna(bigg_boss['Average TRP'].mean())
bigg_boss['Season Start Date'] = pd.to_datetime(bigg_boss['Season Start Date'])

bigg_boss['Season End Date'] = pd.to_datetime(bigg_boss['Season End Date'])

bigg_boss['Entry Date'] = pd.to_datetime(bigg_boss['Entry Date'])

bigg_boss['Elimination Date'] = pd.to_datetime(bigg_boss['Elimination Date'])
bigg_boss.head()
bigg_boss.tail()
# Updating last week-end elimination/entries manually

#bigg_boss.iloc[510,23] = 0
train = bigg_boss.loc[(bigg_boss['Winner'].notnull()), :]

train.sample(10)
test = bigg_boss.loc[(bigg_boss['Language']==lang) & (bigg_boss['Winner'].isnull()), :]

test



# Participants who are still in current Bigg Boss Tamil season
BB_tamil_participant = test[['Name']]

BB_tamil_participant.reset_index(drop=True, inplace=True)

BB_tamil_participant
train.drop(["Name","Entry Date","Elimination Date","Season Start Date","Season End Date","Elimination Week Number"], axis=1, inplace=True)

test.drop(["Name","Entry Date","Elimination Date","Season Start Date","Season End Date","Elimination Week Number","Winner"], axis=1, inplace=True)
train.head()
test.head()
# Spread of target variable

print(train['Winner'].value_counts(normalize=True)*100)
# One Hot Encoding



target = train.pop('Winner')

data = pd.concat([train, test])

dummies = pd.get_dummies(data, columns=data.columns, drop_first=True, sparse=True)

train2 = dummies.iloc[:train.shape[0], :]

test = dummies.iloc[train.shape[0]:, :]
print(train2.shape)

print(test.shape)
train2.head()
target.values
x_train, x_val, y_train, y_val = train_test_split(train2, target, test_size=0.3, random_state=2020)

print(x_train.shape, x_val.shape)
def plot_confusion_matrix():

    cm = confusion_matrix(y_val, y_predicted_val).T

    cm = cm.astype('float')/cm.sum(axis=0)

    ax = sns.heatmap(cm, annot=True, cmap='Blues');

    ax.set_xlabel('True Label',size=12)

    ax.set_ylabel('Predicted Label',size=12)
# Logistic Regression

for c in [0.01, 1, 10, 100, 1000]:

    lr = LogisticRegression(random_state=2020, C=c).fit(x_train, y_train)

    print ("F1 score for C=%s: %s" % (c, f1_score(y_val, lr.predict(x_val), average='weighted')*100))
logi = LogisticRegression(random_state=2020,C=100).fit(x_train, y_train)

logi
predicted_val_logi = logi.predict_proba(x_val)[:, 1]

y_predicted_val = (predicted_val_logi > 0.3).astype("int").ravel()

print('F1 Score -',f1_score(y_val, y_predicted_val, average='weighted')*100)

print('Accuracy Score -',accuracy_score(y_val, y_predicted_val)*100)
# Confusion Matrix

plot_confusion_matrix()



# TP 1 TN 1
predicted_val_logi = logi.predict_proba(test)[:, 1]

winner_lg = pd.concat([BB_tamil_participant, pd.DataFrame(predicted_val_logi, columns=['Predicted_Winner'])],axis=1)

winner_lg.sort_values('Predicted_Winner', ascending=False)
# Predicted Winner for Bigg Boss Tamil Season 4, as per LogisticRegression

winner_lg.iloc[np.argwhere(winner_lg.Predicted_Winner == np.amax(winner_lg.Predicted_Winner)).flatten().tolist()]
# RandomForest

rf = RandomForestClassifier(n_estimators=500, random_state=2020).fit(x_train, y_train)

rf
predicted_val_rf = rf.predict_proba(x_val)[:, 1]

y_predicted_val = (predicted_val_rf > 0.3).astype("int").ravel()

print('F1 Score -',f1_score(y_val, y_predicted_val, average='weighted')*100)

print('Accuracy Score -',accuracy_score(y_val, y_predicted_val)*100)



# n_estimators=100 accuracy 99.4

# n_estimators=200 accuracy 100
# Confusion Matrix

plot_confusion_matrix()



# TP 1 TN 1
predicted_val_rf = rf.predict_proba(test)[:,1]

winner_rf = pd.concat([BB_tamil_participant, pd.DataFrame(predicted_val_rf, columns=['Predicted_Winner'])],axis=1)

winner_rf.sort_values('Predicted_Winner', ascending=False)
# Predicted Winner for Bigg Boss Tamil Season 4, as per RandomForest

winner_rf.iloc[np.argwhere(winner_rf.Predicted_Winner == np.amax(winner_rf.Predicted_Winner)).flatten().tolist()]
NN = MLPClassifier(random_state=2020)

#NN = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(500, 20), random_state=2020)

NN.fit(x_train, y_train)
predicted_val_nn = NN.predict(x_val)

# predicted_val_nn = NN.predict_proba(x_val)[:,1]

# y_predicted_val = (predicted_val_nn > 0.03).astype("int").ravel()

print('F1 Score -',f1_score(y_val, y_predicted_val, average='weighted')*100)

print('Accuracy Score -',accuracy_score(y_val, y_predicted_val)*100)
predicted_val_nn
# Confusion Matrix

plot_confusion_matrix()
predicted_val_nn = NN.predict(test)

winner = pd.concat([BB_tamil_participant, pd.DataFrame(predicted_val_nn, columns=['Predicted_Winner'])],axis=1)

winner[['Name','Predicted_Winner']]
# Predicted Winner for Bigg Boss Tamil Season 4, as per Neaural Networks (Multi Layer Perceptron)

# winner.iloc[winner.Predicted_Winner.argmax()]['Name']