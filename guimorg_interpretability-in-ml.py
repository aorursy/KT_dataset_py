!pip install pydotplus lime
import lime

import lime.lime_tabular



from sklearn.tree import (

    ExtraTreeClassifier,

    DecisionTreeClassifier,

    export_graphviz

)

from sklearn.model_selection import train_test_split

from sklearn.metrics import (

    classification_report,

    confusion_matrix

)

from sklearn.externals.six import StringIO

from sklearn.preprocessing import (

    OneHotEncoder,

    LabelEncoder

)

from sklearn.compose import ColumnTransformer



import matplotlib.pyplot as plt



from IPython.display import Image



import pydotplus



import seaborn as sns

import pandas as pd 

import numpy as np
df = pd.read_csv('/kaggle/input/heart-disease-uci/heart.csv')
df.columns = [

    'age',

    'sex',

    'chest_pain_type',

    'resting_blood_pressure',

    'cholesterol',

    'fasting_blood_sugar',

    'rest_ecg',

    'max_heart_rate_achieved',

    'exercise_induced_angina',

    'st_depression',

    'st_slope',

    'num_major_vessels',

    'thalassemia',

    'target'

]
df.head(5)
df['sex'][df['sex'] == 0] = 'female'

df['sex'][df['sex'] == 1] = 'male'



df['chest_pain_type'][df['chest_pain_type'] == 1] = 'typical angina'

df['chest_pain_type'][df['chest_pain_type'] == 2] = 'atypical angina'

df['chest_pain_type'][df['chest_pain_type'] == 3] = 'non-anginal pain'

df['chest_pain_type'][df['chest_pain_type'] == 4] = 'asymptomatic'



df['fasting_blood_sugar'][df['fasting_blood_sugar'] == 0] = 'lower than 120mg/ml'

df['fasting_blood_sugar'][df['fasting_blood_sugar'] == 1] = 'greater than 120mg/ml'



df['rest_ecg'][df['rest_ecg'] == 0] = 'normal'

df['rest_ecg'][df['rest_ecg'] == 1] = 'ST-T wave abnormality'

df['rest_ecg'][df['rest_ecg'] == 2] = 'left ventricular hypertrophy'



df['exercise_induced_angina'][df['exercise_induced_angina'] == 0] = 'no'

df['exercise_induced_angina'][df['exercise_induced_angina'] == 1] = 'yes'



df['st_slope'][df['st_slope'] == 1] = 'upsloping'

df['st_slope'][df['st_slope'] == 2] = 'flat'

df['st_slope'][df['st_slope'] == 3] = 'downsloping'



df['thalassemia'][df['thalassemia'] == 1] = 'normal'

df['thalassemia'][df['thalassemia'] == 2] = 'fixed defect'

df['thalassemia'][df['thalassemia'] == 3] = 'reversable defect'
df.head(5)
print(f"{df['target'].sum()/df.shape[0]:.2f} is the proportion of healthy and unhealthy patients")
sns.set(style="whitegrid")
sns.barplot(x='sex', y='age', data=df, hue='target')
sns.barplot(x='sex', y='st_depression', data=df, hue='target')
df = pd.get_dummies(df, drop_first=True)
X = df.drop('target', axis=1)

y = df['target']
seed = 42

X_train, X_test, y_train, y_test = train_test_split(

    df.drop('target', axis=1), 

    df['target'], 

    test_size=0.2, 

    stratify=df['target'], 

    random_state=seed

)
tree = DecisionTreeClassifier(random_state=seed, max_depth=3)  # we set max_depth to 3 using a rule of thumb for visualization

tree.fit(X_train, y_train)
y_pred = tree.predict(X_test)
target_labels = ['healthy', 'unhealthy']



cm = confusion_matrix(y_test, y_pred)

df_cm = pd.DataFrame(cm, columns=target_labels, index=target_labels)

df_cm.index.name = 'Actual'

df_cm.columns.name = 'Predicted'

sns.set(font_scale=1.4)

sns.heatmap(df_cm, cmap="Blues", annot=True,annot_kws={"size": 16})
print(classification_report(y_test, y_pred))
dot_data = StringIO()



export_graphviz(

    tree,

    out_file=dot_data,

    filled=True,

    rounded=True, 

    class_names=target_labels, 

    special_characters=True, 

    feature_names=df.drop('target', axis=1).columns

)



graph = pydotplus.graph_from_dot_data(dot_data.getvalue())



Image(graph.create_png())
xtra_tree = ExtraTreeClassifier(random_state=seed, max_depth=3)

xtra_tree.fit(X_train, y_train)
y_pred = xtra_tree.predict(X_test)
target_labels = ['healthy', 'unhealthy']



cm = confusion_matrix(y_test, y_pred)

df_cm = pd.DataFrame(cm, columns=target_labels, index=target_labels)

df_cm.index.name = 'Actual'

df_cm.columns.name = 'Predicted'

sns.set(font_scale=1.4)

sns.heatmap(df_cm, cmap="Blues", annot=True,annot_kws={"size": 16})
print(classification_report(y_test, y_pred))
df_feature_importance = pd.DataFrame({"Feature": X_train.columns, "Feature importance": xtra_tree.feature_importances_})
plt.figure(figsize=(16, 6))

sns.barplot(

    x='Feature importance',

    y='Feature',

    data=df_feature_importance.sort_values(by='Feature importance', ascending=False)

)
explainer = lime.lime_tabular.LimeTabularExplainer(

    X_train, 

    feature_names=X_train.columns, 

    discretize_continuous=False,

    mode='classification'

)
instance = X_test.iloc[0]

exp = explainer.explain_instance(instance.to_numpy(), xtra_tree.predict_proba)
print(f"True value: {y_test.iloc[0]}")
exp.show_in_notebook(show_table=True, show_all=True)