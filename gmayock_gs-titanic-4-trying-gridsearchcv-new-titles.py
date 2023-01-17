import numpy as np
import pandas as pd 
import os
print(os.listdir("../input"))
# Step 1 is to import both data sets
training_data = pd.read_csv("../input/train.csv")
testing_data = pd.read_csv("../input/test.csv")

# Step two is to create columns which I will add to the respective datasets, in order to know which row came from which dataset when I combine the datasets
training_column = pd.Series([1] * len(training_data))
testing_column = pd.Series([0] * len(testing_data))

# Now we append them by creating new columns in the original data. We use the same column name
training_data['is_training_data'] = training_column
testing_data['is_training_data'] = testing_column
# Now we can merge the datasets while retaining the key to split them later
combined_data = training_data.append(testing_data, ignore_index=True, sort=False)

# Encode gender (if == female, True)
combined_data['female'] = combined_data.Sex == 'female'

# Split out Title
title = []
for i in combined_data['Name']:
    period = i.find(".")
    comma = i.find(",")
    title_value = i[comma+2:period]
    title.append(title_value)
combined_data['title'] = title

# Replace the title values with an aliased dictionary
title_arr = pd.Series(title)
title_dict = {
    "Capt" : "Officer",
    "Col" : "Officer",
    "Major" : "Officer",
    "Jonkheer" : "Royalty",
    "Don" : "Royalty",
    "Sir" : "Royalty",
    "Dr" : "Officer",
    "Rev" : "Officer",
    "the Countess" : "Royalty",
    "Dona" : "Royalty",
    "Mme" : "Mrs",
    "Mlle" : "Miss",
    "Ms" : "Mrs",
    "Mr" : "Mr",
    "Mrs" : "Mrs",
    "Miss" : "Miss",
    "Master" : "Master",
    "Lady" : "Royalty"
}
cleaned_title = title_arr.map(title_dict)
combined_data['cleaned_title'] = cleaned_title

# Fill NaN of Age - first create groups to find better medians than just the overall median. 
grouped = combined_data.groupby(['female','Pclass', 'cleaned_title'])  

# And now fill NaN with the grouped medians
combined_data['Age'] = grouped.Age.apply(lambda x: x.fillna(x.median()))

# Fill NaN of Embarked
combined_data['Embarked'] = combined_data['Embarked'].fillna("S") 

# Fill NaN of Fare
combined_data['Fare'] = combined_data['Fare'].fillna(combined_data['Fare'].mode()[0]) 

# Fill NaN of Cabin with a U for unknown. Not sure cabin will help.
combined_data['Cabin'] = combined_data['Cabin'].fillna("U") 

# Finding cabin group
cabin_group = []
for i in combined_data['Cabin']:
    cabin_group.append(i[0])
combined_data['cabin_group'] = cabin_group

# Adding a family_size feature as it may have an inverse relationship to either of its parts
combined_data['family_size'] = combined_data.Parch + combined_data.SibSp + 1

# Mapping ports to passenger pickup order
port = {
    'S' : 1,
    'C' : 2,
    'Q' : 3
}
combined_data['pickup_order'] = combined_data['Embarked'].map(port)

# Encode childhood
combined_data['child'] = combined_data.Age < 16

# One-Hot Encoding the titles
combined_data = pd.concat([combined_data, pd.get_dummies(combined_data['cleaned_title'], prefix="C_T")], axis = 1)

# One-Hot Encoding the Pclass
combined_data = pd.concat([combined_data, pd.get_dummies(combined_data['Pclass'], prefix="PClass")], axis = 1)

# One-Hot Encoding the  cabin group
combined_data = pd.concat([combined_data, pd.get_dummies(combined_data['cabin_group'], prefix="C_G")], axis = 1)

# One-Hot Encoding the ports
combined_data = pd.concat([combined_data, pd.get_dummies(combined_data['Embarked'], prefix="Embarked")], axis = 1)
combined_data.dtypes.index
new_train_data=combined_data.loc[combined_data['is_training_data']==1]
new_test_data=combined_data.loc[combined_data['is_training_data']==0]
# here is the expanded model set and metric tools
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
k_fold = KFold(n_splits = 10, shuffle=True, random_state=0)
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
# Here are the features
features = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'female', 'child', 'Embarked_S', 'Embarked_C', 
            'Embarked_Q', 'pickup_order', 'C_T_Master', 'C_T_Miss', 'C_T_Mr', 'C_T_Mrs',
            'C_T_Officer', 'C_T_Royalty', 'C_G_A', 'C_G_B', 'C_G_C', 'C_G_D', 'C_G_E', 'C_G_F', 'C_G_G', 
            'C_G_T', 'C_G_U', 'family_size', 'PClass_1', 'PClass_2', 'PClass_3']
target = 'Survived'
cvs_train_data = new_train_data[features]
cvs_test_data = new_test_data[features]
cvs_target = new_train_data['Survived']
# Define the models
################################################################################################### NOTE - I cannot justify setting random_state=0 for any reason other than reproducability of results
model01 = RandomForestClassifier(n_estimators=10, random_state=0);
model02 = DecisionTreeClassifier(random_state=0);
model03 = LogisticRegression(solver='liblinear');
model04 = KNeighborsClassifier(n_neighbors=15)
model05 = GaussianNB()
model06 = SVC(gamma='auto')

# Define a function to make reading recall score easier
def printCVPRAF(model_number):
    print("CrossVal Precision: ", round(np.mean(cross_val_score(model_number, cvs_train_data, cvs_target, cv=k_fold, n_jobs=1, scoring='precision'))*100,2), "%")
    print("CrossVal Recall: ", round(np.mean(cross_val_score(model_number, cvs_train_data, cvs_target, cv=k_fold, n_jobs=1, scoring='recall'))*100,2), "%")
    print("CrossVal Accuracy: ", round(np.mean(cross_val_score(model_number, cvs_train_data, cvs_target, cv=k_fold, n_jobs=1, scoring='accuracy'))*100,2), "%")
    print("CrossVal F1-Score: ", round(np.mean(cross_val_score(model_number, cvs_train_data, cvs_target, cv=k_fold, n_jobs=1, scoring='f1_macro'))*100,2), "%")
    y_pred = cross_val_predict(model_number, cvs_train_data, cvs_target, cv=k_fold)
    conf_mat = confusion_matrix(cvs_target, y_pred)
    rows = ['Actually Died', 'Actually Lived']
    cols = ['Predicted Dead','Predicted Lived']
    print("\n",pd.DataFrame(conf_mat, rows, cols))
    
# Print results
print("Random Forest Classifier")
printCVPRAF(model01);
print("\n\nDecision Tree Classifier")
printCVPRAF(model02);
print("\n\nLogistic Regression")
printCVPRAF(model03);
print("\n\nKNeighborsClassifier")
printCVPRAF(model04);
print("\n\nGaussianNB")
printCVPRAF(model05);
print("\n\nSVC")
printCVPRAF(model06);
# First we fit the model
model03.fit(cvs_train_data, cvs_target)
prediction = model03.predict(cvs_test_data)
submission = pd.DataFrame({
    "PassengerId" : new_test_data['PassengerId'],
    "Survived" : prediction.astype(int)
})
submission.to_csv('submission.csv', index=False)
from sklearn.model_selection import GridSearchCV
model07 = RandomForestClassifier(random_state=0);
# The first thing to do is to create a dictionary of parameters
forest_params = dict(     
    max_depth = [n for n in range(5, 20)],     
    min_samples_split = [n for n in range(3, 13)], 
    min_samples_leaf = [n for n in range(1, 7)],     
    n_estimators = [n for n in range(10, 60, 2)],
)

# # Going to reduce the number of attempts for a while to see if this works better
# forest_params = dict(     
#     max_depth = [n for n in range(8, 12)],     
#     min_samples_split = [n for n in range(5, 9)], 
#     min_samples_leaf = [n for n in range(2, 4)],     
#     n_estimators = [n for n in range(10, 60, 25)],
# )


forest_gscv = GridSearchCV(estimator=model07, param_grid=forest_params, cv=k_fold) 
forest_gscv.fit(cvs_train_data, cvs_target)
print("Best score: {}".format(forest_gscv.best_score_))
print("Optimal params: {}".format(forest_gscv.best_estimator_))
forest_gscv_predictions = forest_gscv.predict(cvs_test_data)
forest_gscv_predictions.shape
# Export 
submission3 = pd.DataFrame({
    "PassengerId" : new_test_data['PassengerId'],
    "Survived" : forest_gscv_predictions.astype(int)
})
submission3.to_csv('submission3.csv', index=False)
