# Right, first some modules 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math

sns.set(style="whitegrid")
sns.set_palette('Blues')

# Some magic
%matplotlib inline

# Now let's load some data 
rawdata_df = pd.read_csv('../input/titanic_data.csv')
rawdata_df.head()
rawdata_df.tail()
rawdata_df.describe()
df = rawdata_df.set_index('PassengerId')
df.head()
print(set(rawdata_df['Pclass'].values))
print(set(rawdata_df['Ticket'].values[0:50]))
print(set(rawdata_df['Fare'].values[0:50]))
print(set(rawdata_df['Embarked'].values))
print(set(rawdata_df['Sex'].values))
print(rawdata_df['Age'].isnull().sum())

df["Cabin"] = df["Cabin"].fillna("Unknown")
df['Embarked'] = df['Embarked'].fillna('Unknown')
print ("Percentage of people with unknown room in 1st class: {0:.1f}%"
    .format(((df["Cabin"] == "Unknown") & (df["Pclass"] == 1)).sum()*100 / float((df["Pclass"] == 1).sum())))
print ("Percentage of people with unknown room in 2nd class: {0:.1f}%"
    .format(((df["Cabin"] == "Unknown") & (df["Pclass"] == 2)).sum()*100 / float((df["Pclass"] == 2).sum())))
print ("Percentage of people with unknown room in 3rd class: {0:.1f}%"
    .format(((df["Cabin"] == "Unknown") & (df["Pclass"] == 3)).sum()*100 / float((df["Pclass"] == 3).sum())))
ports = {'C': 'Cherbourg', 'Q': 'Queenstown', 'S' : 'Southampton'}
classes = {1: 'First', 2: 'Second', 3: 'Third'}

def set_port(port):
    if port == 'Unknown':
        return 'Unknown'
    else:
        return ports[port]

def set_class(classn):
    if math.isnan(classn):
        return 'Unknown'
    else:
        return classes[classn]

df['Embarked'] = df['Embarked'].apply(set_port)
df['Pclass'] = df['Pclass'].apply(set_class)
print(df.groupby('Embarked').size())
df.rename(columns = {'Survived' : 'Survival rate', 'Pclass' : 'Class', 'Embarked' : 'Port of embarkment'}, inplace=True)
df.head()
print("Survivalrate: {}%".format(round(df['Survival rate'].mean()*100)))
df_byclass = df.groupby(['Class','Port of embarkment'])

graph_data = pd.DataFrame(df_byclass.size(), columns={'Passengers'}).reset_index()
print(graph_data)
ax = graph_data.pivot(index='Class', columns='Port of embarkment', values='Passengers').plot(kind='bar', stacked=True)
ax.set_ylabel('Passengers');
df.groupby('Class')['Fare'].describe()
df.loc[df.Fare == 0]
graph_data = pd.DataFrame(df.groupby(['Class', 'Sex']).size(), columns = {'Passengers'}).reset_index()
ax3 = sns.barplot(x="Class", y="Passengers", hue="Sex", data=graph_data)
ax2 = sns.boxplot(x="Class", y="Age", 
                  hue="Sex", order = classes.values(), hue_order=['male', 'female'], 
                  data=df[np.isfinite(df['Age'])])
ax5 = sns.barplot(x="Class", y="Survival rate", order = classes.values(), data=df, ci=None)
ax5.set_ylim([0,1]);
ax4 = sns.barplot(x="Class", y="Survival rate", hue="Sex"
                  , order = list(classes.values()), hue_order=['male', 'female'], data=df, ci=None)
def is_child(age): 
    if age < 18:
        return "Child"
    else:
        return "Adult"

df['Agegroup'] = df['Age'].apply(is_child)
df_knownage = df[np.isfinite(df['Age'])]
ax7 = sns.barplot(x='Class', y='Survival rate', 
                  order = list(classes.values()), hue='Agegroup', hue_order = ['Child', 'Adult'], data=df_knownage, ci=None)
agegroups = ['Infant', 'Child', 'Teenager', 'Adolescent', 'Adult', 'Senior']

def agegroup(age): 
    if age < 4.:
        return agegroups[0]
    elif age < 10.: 
        return agegroups[1]
    elif age < 20.: 
        return agegroups[2]
    elif age < 30.: 
        return agegroups[3]
    elif age < 65.: 
        return agegroups[4]
    else:
        return agegroups[5]
    
df['Agegroup'] = df['Age'].apply(agegroup)
df_knownage = df[np.isfinite(df['Age'])]
ax = df_knownage.groupby(['Class','Agegroup']).size().unstack()[agegroups].plot(kind='bar', stacked=True);
ax.set_ylabel('Passengers');
ax8 = sns.barplot(x='Class', y='Survival rate', 
                  hue='Agegroup', order = list(classes.values()), hue_order = agegroups, data=df_knownage, ci=None)
df.loc[(df['Class'] == 'First') & (df['Survival rate'] == 0) & (df['Agegroup'] == 'Infant')]
from sklearn.ensemble import RandomForestClassifier

# Make train data
feature_cols = ['Class', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare','Port of embarkment' ]
bool_cols = ['Class', 'Sex','Port of embarkment' ]
X = df[feature_cols]
y = df['Survival rate']

print(X.isnull().sum())
print (X.shape)
print (y.shape)
rows = np.isfinite(X.Age)
X = X.loc[rows]
y = y.loc[rows]

print (X.isnull().sum())
print (X.shape)
print (y.shape)
print (sum(X.Fare == 0))
rows = X.Fare != 0
X = X.loc[rows]
y = y.loc[rows]
# Normalize string labels 
from sklearn.preprocessing import LabelEncoder

labelencoders = {}
for column in bool_cols:
    labelencoders[column] = LabelEncoder().fit(X[column])

X[bool_cols] = X[bool_cols].apply(LabelEncoder().fit_transform)

# Train classifier
clf = RandomForestClassifier()
clf.fit(X,y)

# Get importance 
output = pd.DataFrame(clf.feature_importances_, index=feature_cols, columns=['Feature importance'])
output = output.sort_values(by='Feature importance', ascending=False)
print (output)
ax8 = sns.boxplot(x="Class", y="Fare", 
                  hue="Survival rate", order = classes.values(), data=df_knownage)
ax8.set_ylim([0,180]);


Rose = dict(zip(feature_cols, ['First', 'female', 17, 0, 1, 87.961582, 'Southampton']))
Jack = dict(zip(feature_cols, ['Third', 'male', 20, 0, 0, 13.229435, 'Southampton']))

for column in bool_cols:
    Rose[column] = labelencoders[column].transform([Rose[column]])
    Jack[column] = labelencoders[column].transform([Jack[column]])

Rose_norm = []; Jack_norm = []
for column in feature_cols:
    Rose_norm.append(Rose[column])
    Jack_norm.append(Jack[column])

predict = clf.predict(np.array([Rose_norm, Jack_norm]))
predict_proba = clf.predict_proba(np.array([Rose_norm, Jack_norm]))

print ('Likelihood of Rose surviving Titanic disaster: {}%'.format(predict_proba[0,1]*100))
print ('Likelihood of Jack surviving Titanic disaster: {}%'.format(predict_proba[1,1]*100))

strings = ['perishes!!', 'survives!!']
print ('\nCLASSIFIER PREDICTIONS:')
print ('Rose ' + strings[predict[0]])
print ('Jack ' + strings[predict[1]])