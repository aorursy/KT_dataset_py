# Import libraries we'll need

import numpy as np 
import pandas as pd

#visualization
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

df = pd.read_csv('../input/WA_Fn-UseC_-Telco-Customer-Churn.csv')

# Convert to lower string
for item in df.columns:
    try:
        df[item] = df[item].str.lower()
    except:
        print("")
#         print(item, "couldn't convert")

        
columns_to_convert = (['Partner', 
                      'Dependents', 
                      'PhoneService', 
                      'PaperlessBilling', 
                      'Churn'])

for item in columns_to_convert:
    df[item].replace(to_replace='yes', value=1, inplace=True)
    df[item].replace(to_replace='no',  value=0, inplace=True)
    
   
df['TotalCharges'] = df['TotalCharges'].replace(r'\s+', np.nan, regex=True)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'])

df = df.fillna(value=0)

churners_number = len(df[df['Churn'] == 1])
# print("Number of churners", churners_number)

churners = (df[df['Churn'] == 1])

non_churners = df[df['Churn'] == 0].sample(n=churners_number)
# print("Number of non-churners", len(non_churners))
df2 = churners.append(non_churners)

try:
    customer_id = df2['customerID'] # Store this as customer_id variable
    del df2['customerID'] # Don't need in ML DF
except:
    print("already removed customerID")
    
ml_dummies = pd.get_dummies(df2)
ml_dummies.fillna(value=0, inplace=True)
ml_dummies.head()

# Add a random column to the dataframe
ml_dummies['---randomColumn---'] = np.random.randint(0,1000, size=len(ml_dummies))

try:
    label = ml_dummies['Churn'] # Remove the label before training the model
    del ml_dummies['Churn']
except:
    print("label already removed.")

from sklearn.model_selection import train_test_split
feature_train, feature_test, label_train, label_test = train_test_split(ml_dummies, label, test_size=0.3)

# Classifiers
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

classifiers = [
    DecisionTreeClassifier(max_depth=5)
]
    
# iterate over classifiers
for item in classifiers:
    classifier_name = ((str(item)[:(str(item).find("("))]))
    print (classifier_name)
    
    # Create classifier, train it and test it.
    clf = item
    clf.fit(feature_train, label_train)
    pred = clf.predict(feature_test)
    score = clf.score(feature_test, label_test)
    print (round(score,3),"\n", "- - - - - ", "\n")
    
feature_df = pd.DataFrame()
feature_df['features'] = ml_dummies.columns
feature_df['importance'] = clf.feature_importances_
feature_df.sort_values(by='importance', ascending=False)    
feature_df.set_index(keys='features').sort_values(by='importance', ascending=True).plot(kind='barh', figsize=(20, 15))
# Preprocessing original dataframe
def preprocess_df(dataframe):
    x = dataframe.copy()
    try:
        customer_id = x['customerID']
        del x['customerID'] # Don't need in ML DF
    except:
        print("already removed customerID")
    ml_dummies = pd.get_dummies(x)
    ml_dummies.fillna(value=0, inplace=True)

    # import random done above
    ml_dummies['---randomColumn---'] = np.random.randint(0,1000, size=len(ml_dummies))

    try:
        label = ml_dummies['Churn']
        del ml_dummies['Churn']
    except:
        print("label already removed.")
    return ml_dummies, customer_id, label

original_df = preprocess_df(df)

output_df = original_df[0].copy()
output_df['---randomColumn---']
output_df['prediction'] = clf.predict_proba(output_df)[:,1]
output_df['churn'] = original_df[2]
output_df['customerID'] = original_df[1]


### Media Activation

# activate = output_df[output_df['churn'] == 0]
activate = output_df.copy()
activate = activate[['customerID','churn','prediction']]
activate.head()
def create_audience_groups(dataframe):
    temp = dataframe.copy()

    purchased = temp[temp['churn'] == 1]
    purchased['group'] = 'churned'

    not_purchased = temp[temp['churn'] == 0]

    high = not_purchased.copy()
    high = high[high['prediction'] > 0.8]
    high['group'] = 'high'

    low = not_purchased.copy()
    low = low[low['prediction'] < 0.3]
    low['group'] = 'low'

    med = not_purchased.copy()
    med = med[(med['prediction'] >= 0.3) & (med['prediction'] <= 0.8)]
    med['group'] = 'med'

    out_df = pd.DataFrame()

    for item in [purchased, high, med, low]:
        out_df = out_df.append(item)

    return out_df

out = create_audience_groups(activate)
out[['customerID','prediction','group']].sample(frac=1)
# out[['customerID','group']].sort_values(by=['churn', 'group'])
