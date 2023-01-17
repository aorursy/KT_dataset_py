# make sure to have the latest sns version

!pip install seaborn --upgrade
import seaborn as sns

sns.__version__
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.compose import ColumnTransformer

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier





import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df = pd.read_csv('/kaggle/input/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv')
df
df.info()
for feat in df.columns:

    print(feat)

    print(df[feat].dtype)

    print(df[feat].unique())

    print('#'*30)
# change dtype

df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')



# fill NaNs with mean

df['TotalCharges'].fillna(df['TotalCharges'].mean(), inplace=True)
sns.countplot(data=df, x=df['Churn']);
# Step 1: remove features

df_cross = df.drop(columns=['customerID', 'tenure', 'MonthlyCharges', 'TotalCharges'])



# Step 2: create empty list

crosstabs = []

# Step 3: For loop to create the crosstabs

for feat in df_cross.columns:

    crosstab = pd.crosstab(df_cross[feat], df_cross['Churn'])

    # Step 4: append them to the list

    crosstabs.append(crosstab)



# Step 5: concate each of them with the column names as index

crosstab_count = pd.concat(crosstabs, keys=df_cross.columns[:-1])

crosstab_count





# this can be done in one line of code:

# pd.concat([pd.crosstab(df_cross[x], df_cross['Churn']) for x in df_cross.columns[:-1]], keys=df_cross.columns[:-1])

# code from udemy course: Data Science & Deep Learning for Business
crosstab_count['percentage'] = (crosstab_count['Yes'] / (crosstab_count['Yes'] + crosstab_count['No'])* 100)

crosstab_count.sort_values('percentage', ascending=False)
df_ratio = df[['tenure', 'MonthlyCharges', 'TotalCharges', 'Churn']]



fig, axes = plt.subplots(1, 3, figsize=[12,6])

# space between the plots 

fig.subplots_adjust(wspace=0.5)

for feat, ax in zip(df_ratio.columns[:-1], axes.flatten()):

    sns.violinplot(data=df_ratio, y=feat, x=df_ratio['Churn'], ax=ax)
df.drop(columns='customerID', inplace=True)
df_bin = df.select_dtypes('object')



# select only features with two unique values

list_bin = [df_bin[x].name for x in df_bin.columns if df_bin[x].nunique() == 2]

# create dict with values

dict_bin = {'Yes': 1, 'No': 0, 'Female': 0, 'Male': 1}



# for loop to go through each binary feature and map the values

for feat in list_bin:

    df_bin[feat] = df_bin[feat].map(dict_bin)
# dtype object only includes categorical variables now so i can just filter them

df_cat = df_bin.select_dtypes('object')



df_dummy = pd.get_dummies(df_cat, drop_first=True)
# cobine Data Frames

df_final = pd.concat([df.select_dtypes(exclude='object'), df_bin.select_dtypes(exclude='object'), df_dummy], axis=1)

df_final
X = df_final.drop(columns='Churn')

y = df_final['Churn']



# we will use the stratify argument to account for the imbalance

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42, stratify=y)



print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
# use Column Transformer to apply scaling only to the specified columns

ct = ColumnTransformer([('scaler', StandardScaler(), ['tenure', 'MonthlyCharges', 'TotalCharges'])], remainder='passthrough')

# fit and transform X_train

X_train_sc = ct.fit_transform(X_train)

# transform X_test 

X_test_sc = ct.transform(X_test)
# initiate models 

logreg = LogisticRegression()

rf = RandomForestClassifier()

adac = AdaBoostClassifier()

gbc = GradientBoostingClassifier()



model_list = [logreg, rf, adac, gbc]



for model in model_list:

    model.fit(X_train_sc, y_train)

    preds = model.predict(X_test_sc)

    print('accuracy: ', accuracy_score(y_test, preds))

    print(confusion_matrix(y_test, preds))

    print(classification_report(y_test, preds))

    print('#'*60)
col_list = list(df_final.drop(columns = 'Churn').columns)

importance = logreg.coef_.flatten()



feat_importance = pd.DataFrame({'feature': col_list, 'importance': importance})

feat_importance.sort_values('importance')