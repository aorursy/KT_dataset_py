import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
#dropping unnecessary columns
df = pd.read_csv("../input/global-shark-attacks/attacks.csv", encoding = "ISO-8859-1")
df.drop(['Investigator or Source', 'pdf', 'href formula', 'href',
       'Case Number.1', 'Case Number.2', 'original order', 'Unnamed: 22',
       'Unnamed: 23', 'Case Number'],axis=1,inplace=True)
#dropping cases where shark attacks were fatal, just because it's quite insensitive to include these cases
df.drop(df[df['Fatal (Y/N)']=='Y'].index, inplace=True)
df.drop(['Injury','Fatal (Y/N)'], axis=1, inplace=True)
#delete all columns where gender is Nan
df.drop(df[(df['Sex ']!='M')&(df['Sex ']!='F')].index,inplace = True)
#delete all cases before 1990, to make life easier
df.drop(df[(df['Year']<1990)].index,inplace = True)
#Changing datatypes to make upcoming label-encoding easier
df = df.astype({'Date': 'string', 'Year': 'float64', 'Type': 'string', 'Country': 'string', 'Area': 'string', 'Location': 'string', 'Activity': 'string', 'Name': 'string', 'Sex ': 'string', 'Age': 'string', 'Time':'string','Species ':'string'})
#Getting rid of rows with NA values, also to make life easier. 
#Still around 800 rows which should be enough to experiment with the effects of outliers
df = df[(df.notna().all(axis=1))==True]

def clean_age(string):
    #function which returns first two digits of age
    output = ''
    for char in string:
        if char.isdigit():
            output += char
    if output!= '':
        return int(output)
    else:
        return -1

df['Age'] = df['Age'].apply(clean_age).astype(int)

df.head()
middle_95_df = df[(df['Age']>df['Age'].quantile(0.05))&(df['Age']<df['Age'].quantile(0.95))]
df['Age'].hist()
removed_visable_outliers = df.drop(df[df['Age']>120].index)
removed_visable_outliers['Age'].hist()
#label-encoding all the columns apart from age for the two new dataframes 
from sklearn.preprocessing import LabelEncoder
middle_95_df_le = middle_95_df[['Date','Type','Country','Area','Location','Activity','Name','Sex ','Time','Species ']].apply(LabelEncoder().fit_transform)
middle_95_df_le['Age'] = middle_95_df['Age'].copy(deep=True) 
removed_visable_outliers_le = removed_visable_outliers[['Date','Type','Country','Area','Location','Activity','Name','Sex ','Time','Species ']].apply(LabelEncoder().fit_transform)
removed_visable_outliers_le['Age'] = removed_visable_outliers['Age'].copy(deep=True)
middle_95_df_le.head()
removed_visable_outliers_le.head()
def RF_accuracy(dataframe):
    X = dataframe.drop('Sex ', axis=1)
    y = dataframe['Sex ']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    RF = RandomForestClassifier(max_depth=2, random_state=0)
    RF.fit(X_train,y_train)
    preds = RF.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(acc)
RF_accuracy(middle_95_df_le)
RF_accuracy(removed_visable_outliers_le)