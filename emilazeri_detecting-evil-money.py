# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns # visualization library alternative to matplotlib.pyplot





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
# first, loading our data

data = pd.read_csv("../input/PS_20174392719_1491204439457_log.csv")
data.head()
data.describe()
data.info()
data['isFlaggedFraud'].sum()
data.drop(['nameOrig','nameDest','isFlaggedFraud'], axis = 1, inplace=True)
sns.countplot(data['type'], hue = data['isFraud'])
data[data['isFraud']==1].groupby('type').count()
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_toScale = data[['amount', 'oldbalanceOrg', 'newbalanceOrig',

      'oldbalanceDest', 'newbalanceDest'

       ]]

new_X = sc.fit(X_toScale)

X_scaled = new_X.transform(X_toScale)
#creating our dataframe with scaled values



scaled_df = pd.DataFrame(X_scaled, columns=['amount', 'oldbalanceOrg', 'newbalanceOrig',

      'oldbalanceDest', 'newbalanceDest'

       ])

# we have also some categorical variable, called Type. Let's convert it to dummies, and then add to our final dataframe

dummy_df = pd.DataFrame(pd.get_dummies(data['type']))

#now, final dataframe

final_df = scaled_df.join(dummy_df, how = 'outer')
final_df.head(5)
from sklearn.ensemble import RandomForestClassifier

from sklearn.cross_validation import train_test_split

# for future, train test split will be moved into model selection

# from sklearn.model_selection import train_test_split

rfc = RandomForestClassifier() #using default values

#splitting our dataset

X = final_df #dataset that we scaled and preprocessed

y = data['isFraud'] #the column from our original dataset will be our label

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42) #use this random state to match my results only

#training our model

model = rfc.fit(X_train,y_train)

#predicting our labels

predictions = model.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix

print(classification_report(y_test,predictions))

print(confusion_matrix(y_test, predictions))