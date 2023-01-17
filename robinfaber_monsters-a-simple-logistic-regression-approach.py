import pandas as pd # data processing, CSV file I/O

import seaborn as sns # plotting

from sklearn.linear_model import LogisticRegression # Logistic regression
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")



train.isnull().any()
train.head()
train.describe()
ax = sns.countplot(x='type', data=train, palette='Set3')
ax = sns.countplot(x='color', data=train, palette='Set3')
id_list = list(train['id']) # Create list of 'id' column in case we need it later

train = train.drop('id', 1) # Drop 'id' column



# Create dictionaries for 'type' and 'color' variables



type_dict = {'Ghoul': 0,

            'Goblin': 1,

            'Ghost': 2}



color_dict = {'clear': 0,

             'green': 1,

             'black': 2,

             'white': 3,

             'blue': 4,

             'blood': 5}



# Use dictionaries to re-map values categorical variables



train['type'] = train['type'].map(type_dict).astype(float)

train['color'] = train['color'].map(color_dict).astype(float)



test['color'] = test['color'].map(color_dict).astype(float)
sns.pairplot(train, hue='type', palette='Set3') 



#Legend label text shows (0, 1, 2); any tips on how to change legend text are more than welcome!
train.corr(method='pearson')
train = pd.concat([train, pd.get_dummies(train['color'], prefix = 'color')], axis=1) # Create dummies

train = train.drop('color', 1) # Drop 'color' column



test = pd.concat([test, pd.get_dummies(test['color'], prefix = 'color')], axis=1) # Create dummies

test = test.drop('color', 1) # Drop 'color' column
# List of columns we are using in the model



feature_cols = ['bone_length', 

                'rotting_flesh', 

                'hair_length', 

                'has_soul', 

                'color_0.0',

                'color_1.0',

                'color_2.0',

                'color_3.0',

                'color_4.0',

                'color_5.0']
X = train.loc[:, feature_cols] # Set independent variables

y = train.type # Set outcome variable



logreg = LogisticRegression()

logreg.fit(X, y) # Fit model
X_test = test.loc[:, feature_cols]

new_type_pred = logreg.predict(X_test) # Use fitted model to predict outcome in test df
# Create submission df



submission = pd.DataFrame({'id': test.id, 'type': new_type_pred})



# Convert 'type' variable back to string variable



type_dict_sub = {0: 'Ghoul',

            1: 'Goblin',

            2: 'Ghost'}



submission['type'] = submission['type'].map(type_dict_sub).astype(object)



# Write submission file to CSV



submission.to_csv('submission.csv', index=False)