import pandas  as pd



#===========================================================================

# read in the data

#===========================================================================

train_data = pd.read_csv('../input/titanic/train.csv')



#===========================================================================

# features to rank

#===========================================================================

features = ["Pclass", "Sex", "SibSp", "Parch", "Embarked", "Fare"]



#===========================================================================

# for the features that are categorical we use pd.get_dummies:

# "Convert categorical variable into dummy/indicator variables."

#===========================================================================

X_train       = pd.get_dummies(train_data[features])

y_train       = train_data["Survived"]



#===========================================================================

# perform the classification

#===========================================================================

from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(max_features='auto', 

                                    min_samples_leaf=10, 

                                    random_state=42)



classifier.fit(X_train, y_train)
import warnings

warnings.filterwarnings("ignore")
#===========================================================================

# perform the PermutationImportance

#===========================================================================

import eli5

from   eli5.sklearn import PermutationImportance



perm_import = PermutationImportance(classifier, random_state=1).fit(X_train, y_train)



# now visualize the results

eli5.show_weights(perm_import, top=None, feature_names = X_train.columns.tolist())