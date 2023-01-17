import re
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score
import eli5
from eli5.sklearn import PermutationImportance
from pdpbox import pdp
import shap

shap.initjs()
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
def pre_process_data(df):
    # Add "title" feature
    df['title'] = df.apply(lambda row: re.split('[,.]+', row['Name'])[1], axis=1)
    
    # Add "family" feature
    df['family'] = df['SibSp'] + df['Parch'] + 1
    
    # One-hot encode categorical values.
    df = pd.get_dummies(df, columns=['Sex', 'Embarked', 'title'])
    
    # Drop columns unwanted columns
    # I'm dropping "Cabin" because it has too much missing data.
    df = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)

    return df


# Get labels
# labels = train['Survived'].values
labels = train['Survived']
train.drop(['Survived'], axis=1, inplace=True)
# Get test ids
test_ids = test['PassengerId'].values

train = pre_process_data(train)
test = pre_process_data(test)

# align both data sets (by outer join), to make they have the same amount of features,
# this is required because of the mismatched categorical values in train and test sets.
train, test = train.align(test, join='outer', axis=1)

# replace the nan values added by align for 0
train.replace(to_replace=np.nan, value=0, inplace=True)
test.replace(to_replace=np.nan, value=0, inplace=True)
train.head()
train.describe()
X_train, X_val, Y_train, Y_val = train_test_split(train, labels, test_size=0.2, random_state=1)
model = xgb.XGBClassifier(objective ='binary:logistic', colsample_bytree=0.3, learning_rate=0.1,max_depth=8, n_estimators=50)
model.fit(X_train, Y_train)
predictions = model.predict(X_val)
# Basic metrics from the model
accuracy = accuracy_score(predictions, Y_val)
recall = recall_score(predictions, Y_val)
precision = precision_score(predictions, Y_val)

print('Model metrics')
print('Accuracy: %.2f' % accuracy)
print('Recall: %.2f' % recall)
print('Precision: %.2f' % precision)
plt.rcParams["figure.figsize"] = (15, 6)
xgb.plot_importance(model)
plt.show()
# Fitting the model again so it can work with PermutationImportance.
model_PI = xgb.XGBClassifier(objective ='binary:logistic', colsample_bytree=0.3, learning_rate=0.1,max_depth=8, n_estimators=50)
model_PI.fit(X_train.values, Y_train.values)
perm = PermutationImportance(model_PI, random_state=1).fit(X_train, Y_train)
eli5.show_weights(perm, feature_names=X_val.columns.tolist())
feature_names = train.columns.tolist()
pdp_fare = pdp.pdp_isolate(model=model, dataset=X_val, model_features=feature_names, feature='Fare')
pdp.pdp_plot(pdp_fare, 'Fare')
plt.show()
pdp_fare = pdp.pdp_isolate(model=model, dataset=X_val, model_features=feature_names, feature='Pclass')
pdp.pdp_plot(pdp_fare, 'Pclass')
plt.show()
features_to_plot1 = ['family', 'Pclass']
pdp_inter = pdp.pdp_interact(model=model, dataset=X_val, model_features=feature_names, features=features_to_plot1)
pdp.pdp_interact_plot(pdp_interact_out=pdp_inter, feature_names=features_to_plot1, plot_type='contour')
plt.show()
features_to_plot2 = ['Sex_female', 'Pclass']
pdp_inter = pdp.pdp_interact(model=model, dataset=X_val, model_features=feature_names, features=features_to_plot2)
pdp.pdp_interact_plot(pdp_interact_out=pdp_inter, feature_names=features_to_plot2, plot_type='contour')
plt.show()
row_to_show = 14
data_for_prediction = X_val.iloc[[row_to_show]]
explainer = shap.TreeExplainer(model)
shap_values_single = explainer.shap_values(data_for_prediction)
shap.force_plot(explainer.expected_value, shap_values_single, data_for_prediction)
shap_values = explainer.shap_values(X_val)
shap.summary_plot(shap_values, X_val)
shap.dependence_plot('Fare', shap_values, X_val, interaction_index="Pclass")
shap.dependence_plot('Pclass', shap_values, X_val, interaction_index="Sex_female")
test_predictions = model.predict(test)
submission = pd.DataFrame({"PassengerId":test_ids})
submission["Survived"] = test_predictions
submission.to_csv("submission.csv", index=False)