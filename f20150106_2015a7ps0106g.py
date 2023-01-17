import pandas as pd 

import os

import numpy as np



print(os.listdir("../input"))
import matplotlib.pyplot as plt

import seaborn as sns





from imblearn.ensemble import BalancedBaggingClassifier

from sklearn.decomposition import PCA

from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier, VotingClassifier

from sklearn.model_selection import train_test_split, KFold, GridSearchCV

from sklearn.metrics import roc_auc_score

from imblearn.over_sampling import RandomOverSampler

from sklearn.preprocessing import StandardScaler, LabelEncoder

from collections import Counter

training_data = pd.read_csv('../input/train.csv')

training_data = training_data.replace('?',np.NaN)

training_data.head()
variables_missing = ['PREV', 'Teen','Enrolled','Worker Class','Enrolled','Area','State','MSA','REG','MOVE','Live',

                 'COB FATHER','COB MOTHER','COB SELF','Fill','MIC','MOC','Hispanic','MLU', 'Reason']



for iter in missing_variables:

    training_data[iter].fillna(training_data[iter].mode()[0] ,inplace= True)

    

features1 = training_data.drop(['ID', 'Class'], axis=1)

#target1 = training_data['Class']

target1 = training_data['Class']
# features = pd.get_dummies(features, columns=list(features.select_dtypes(include=['object']).columns))

le1 = LabelEncoder()



for column in features1.select_dtypes(include=['object']).columns:

    le1.fit(features1[column])

    features1[column] = le1.transform(features1[column])
ros = RandomOverSampler(random_state=42)

# check

features1, target1 = ros.fit_resample(features1, target1)

unscaled_data = pd.DataFrame(features1)

# unscaled data and unscaled data

scaled_d = StandardScaler().fit_transform(unscaled_data)

scaled=pd.DataFrame(scaled_d,columns=unscaled_data.columns)
#Train/Validation Set Split

X_train, X_test, y_train, y_test = train_test_split(scaled, target, test_size=0.2, random_state=42)



#model = AdaBoostClassifier(n_estimators=100)

# model = RandomForestClassifier(n_estimators=100)



model = RandomForestClassifier(n_estimators=100, max_depth=12, criterion='gini')



model.fit(X_train, y_train)

y_pred = model.predict(X_test)



print('Accuracy : %f' % roc_auc_score(y_test, y_pred))
model.fit(scaled, target1)
testing_data = pd.read_csv('../input/test.csv')

testing_data = testing_data.replace('?',np.NaN)

testing_data.head()
missing_variables = ['PREV', 'Teen','Enrolled','Worker Class','Enrolled','Area','State','MSA','REG','MOVE','Live',

                 'COB FATHER','COB MOTHER','COB SELF','Fill','MIC','MOC','Hispanic','MLU', 'Reason']

for iter in missing_variables:

    testing_data[iter].fillna(testing_data[iter].mode()[0] ,inplace= True)

    

features1 = testing_data.drop(['ID'], axis=1)
# features = pd.get_dummies(features, columns=list(features.select_dtypes(include=['object']).columns))

le1 = LabelEncoder()



for column in features1.select_dtypes(include=['object']).columns:

    le1.fit(features1[column])

    features1[column] = le1.transform(features1[column])

        

features1.head()
scaled_data = StandardScaler().fit_transform(features1)

scaled_df=pd.DataFrame(scaled_data,columns=features1.columns)

scaled_df.head()
y_pred = model.predict(scaled_df)
s = pd.Series(y_pred)



df = pd.concat([test_data['ID'], s], axis=1)

df.columns = ['ID', 'Class']



df['Class'].value_counts()
from IPython.display import HTML

import pandas as pd

import numpy as np

import base64



# function that takes in a dataframe and creates a text link to  

# download it (will only work for files < 2MB or so)

def create_download_link(df, title = "Download CSV file", filename = "data.csv"):  

    csv = df.to_csv(index=False)

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'

    html = html.format(payload=payload,title=title,filename=filename)

    return HTML(html)





# create a link to download the dataframe

create_download_link(df)