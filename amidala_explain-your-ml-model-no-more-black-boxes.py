import warnings

warnings.filterwarnings('ignore')



import pandas as pd

import numpy as np

import pdpbox, lime, shap, eli5

from matplotlib import pyplot as plt



from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, recall_score

from sklearn.model_selection import train_test_split

from imblearn.combine import SMOTETomek



%matplotlib inline
data = pd.read_csv('../input/adult-census-income/adult.csv')

data.shape
data.columns
data.head()
data['target']=data['income'].map({'<=50K':0,'>50K':1})

data.drop("income",axis=1,inplace=True)

data['target'].value_counts()
# Let's drop "education.num" feature. We will use one-hot encoding instead.

data.drop("education.num",axis=1,inplace=True)
# Since this example is for educational purposes, we'll also drop 'native-country' feature to decrease our data dimensionality.

data.drop('native.country',axis=1,inplace=True)
# Now we will encode categorical features using one-hot encoding, i.e. each category will now be represented by a separate column

# containing only 0 and 1, depending on whether this category is relevant in a sample (row in our data) 

data=pd.get_dummies(data, drop_first = True)
data.head()
y = data['target'].values

features = [col for col in data.columns if col not in ['target']]

X = data[features]



X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, test_size=0.3, stratify=y)

print(X_train.shape, y_train.shape)

print(X_test.shape, y_test.shape)
model = RandomForestClassifier(random_state=1).fit(X_train, y_train)

y_pred = model.predict(X_test)



print("Accuracy: %.2f" %accuracy_score(y_test, y_pred))

print("Recall: %.2f" %recall_score(y_test, y_pred))
import eli5

from eli5.sklearn import PermutationImportance



imp = PermutationImportance(model, random_state=1).fit(X_test, y_test)

eli5.show_weights(imp, feature_names = X_test.columns.tolist())
from pdpbox import pdp, get_dataset, info_plots



feat_name = 'capital.gain'

capital_gain_pdp = pdp.pdp_isolate(model=model, dataset=X_test, 

                                   model_features=X_test.columns, feature=feat_name)



pdp.pdp_plot(capital_gain_pdp, feat_name)

plt.show()
feat_name = 'hours.per.week'



hours_per_week_pdp = pdp.pdp_isolate(model=model, dataset=X_test, 

                                   model_features=X_test.columns, feature=feat_name)



pdp.pdp_plot(hours_per_week_pdp, feat_name)

plt.show()
# check the target. 1? perfect!

y_test[69]
# taking a quick look on a sample

pd.DataFrame(X_test.iloc[69]).T
# First, create a prediction on this sample

row = X_test.iloc[69]

to_predict = row.values.reshape(1, -1)



model.predict_proba(to_predict)
import shap 

# create object that can calculate shap values

explainer = shap.TreeExplainer(model)



# calculate Shap values

shap_values = explainer.shap_values(row)
# draw a plot

shap.initjs()

shap.force_plot(explainer.expected_value[1], shap_values[1], row)
import lime.lime_tabular



explainer = lime.lime_tabular.LimeTabularExplainer(X_train.values, feature_names=X_test.columns,

                                                    discretize_continuous=True)



exp = explainer.explain_instance(row, model.predict_proba, num_features=8)

exp.show_in_notebook(show_table=True)