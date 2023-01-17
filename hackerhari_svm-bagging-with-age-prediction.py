import re

import pandas as pd
from sklearn import linear_model, model_selection, svm, ensemble, preprocessing, pipeline
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import BaggingClassifier, GradientBoostingClassifier
%matplotlib inline
train_raw_data = pd.read_csv('../input/train.csv')
test_raw_data = pd.read_csv('../input/test.csv')

n_train = train_raw_data.shape[0]
n_test = test_raw_data.shape[0]
pd.isnull(train_raw_data).sum()
class PreprocessData:
    def __init__(self, to_predict_col, train_cols):
        self.to_predict_col = to_predict_col
        self.train_cols = train_cols
        self.model = None
        
    # just put the null columns as they are without transforming them
    def transformNonNullData(self, raw_data):
        processed_data = pd.DataFrame()

        # more preference to 1st class by assigning them higher value
        processed_data['pclass'] = pd.Series(raw_data['Pclass'], dtype=np.float64)

        processed_data['male'] = pd.Series(raw_data['Sex'] == 'male', dtype=np.float64)
        processed_data['female'] = pd.Series(raw_data['Sex'] == 'female', dtype=np.float64)

        processed_data['fare'] = pd.Series(raw_data['Fare'])    
        processed_data['sibsp'] = pd.Series(raw_data['SibSp'], dtype=np.float64)
        processed_data['parch'] = pd.Series(raw_data['Parch'], dtype=np.float64)

        # higher is the family size, lower is the chance to survive
        processed_data['family_size'] = pd.Series(raw_data['SibSp'] + raw_data['Parch'] + 1, dtype=np.float64)
        processed_data.loc[pd.isnull(processed_data['family_size']), 'family_size'] = 1
        processed_data['is_alone'] = pd.Series(processed_data['family_size'] == 1, dtype=np.float64)

        processed_data['has_cabin'] = pd.Series(raw_data['Cabin'].notna(), dtype=int)

        processed_data['embarked_s'] = pd.Series(raw_data['Embarked'] == 'S', dtype=np.float64)
        processed_data['embarked_c'] = pd.Series(raw_data['Embarked'] == 'C', dtype=np.float64)
        processed_data['embarked_q'] = pd.Series(raw_data['Embarked'] == 'Q', dtype=np.float64)

        def get_title(name):
            title_search = re.search(' ([A-Za-z]+)\.', name)
            # If the title exists, extract and return it.
            if title_search:
                return title_search.group(1)
            return ""

        processed_data['title'] = raw_data['Name'].apply(get_title)
        processed_data['title'] = processed_data['title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
        processed_data['title'] = processed_data['title'].replace('Mlle', 'Miss')
        processed_data['title'] = processed_data['title'].replace('Ms', 'Miss')
        processed_data['title'] = processed_data['title'].replace('Mme', 'Mrs')

        title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
        processed_data['title'] = processed_data['title'].map(title_mapping)
        processed_data['title'] = processed_data['title'].fillna(0)

        processed_data['age'] = pd.Series(raw_data['Age'])   

        one_hot = pd.get_dummies(processed_data['title']).reset_index()
        processed_data.drop(['title'], axis=1, inplace=True)
        processed_data = processed_data.reset_index().merge(one_hot, left_index=True, right_index=True, how='outer')

        return processed_data
    
    def predictNull(self, data):
        predict_data = data.loc[pd.isna(data[self.to_predict_col[0]]), self.train_cols]
        data.loc[pd.isna(data[self.to_predict_col[0]]), self.to_predict_col] = self.model.predict(predict_data)
        return data

    def fit(self, data, y=None):
        data = self.transformNonNullData(data)
        data['fare'] = data['fare'].fillna(data['fare'].median())
        train_data = data.loc[data[self.to_predict_col[0]].notna(), np.append(self.train_cols, self.to_predict_col)]

        y_train_data = train_data[self.to_predict_col].values.ravel()
        x_train_data = train_data.drop(self.to_predict_col, inplace=False, axis=1)

        self.model = pipeline.make_pipeline(preprocessing.StandardScaler(), svm.SVR(C=0.7))

        self.model.fit(x_train_data, y_train_data)
    
    def transform(self, raw_data):
        data = self.transformNonNullData(raw_data)
        data['fare'] = data['fare'].fillna(data['fare'].median())
        processed_data = self.predictNull(data)

        def cutAge(age):
            if age < 18:
                return 0
            elif age < 30:
                return 1
            elif age < 50:
                return 2
            else:
                return 3

        processed_data['age'] = processed_data['age'].apply(cutAge)    
        return processed_data
    
    def fit_transform(self, data, y=None):
        self.fit(data)
        return self.transform(data)
# all_data = pd.concat([train_raw_data.drop(['Survived'], axis=1, inplace=False), test_raw_data], axis=0)
# all_data = transformData(all_data)
# all_data.head()
y_train_data = train_raw_data['Survived']
x_train_data = train_raw_data.drop(['Survived'], axis=1, inplace=False)
x_test_data = test_raw_data
svm_params = {
    'C':0.7,
    'gamma':'auto',
    'kernel':'poly',
    'degree': 3
}
to_predict_col = ['age']
train_cols = ['pclass', 'male', 'female', 'sibsp', 'parch', 'fare', 'is_alone', 'has_cabin']
support_vector_model = pipeline.make_pipeline(PreprocessData(to_predict_col, train_cols), preprocessing.StandardScaler(), svm.SVC(**svm_params))
support_vector_bag = BaggingClassifier(svm.SVC(**svm_params), n_estimators=20)
svm_bag_model = pipeline.make_pipeline(PreprocessData(to_predict_col, train_cols), preprocessing.StandardScaler(), support_vector_bag)
random_forest_bag = BaggingClassifier(ensemble.RandomForestClassifier(), n_estimators=30)
rf_bag_model = pipeline.make_pipeline(PreprocessData(to_predict_col, train_cols), random_forest_bag)
clf = support_vector_model
# clf = svm_bag_model
train_sizes, train_scores, valid_scores = model_selection.learning_curve(clf, x_train_data, y_train_data, train_sizes=np.linspace(200, 712, 20, dtype=int), cv=5)
train_scores_avg = np.average(train_scores, axis=1)
valid_scores_avg = np.average(valid_scores, axis=1)
plt.plot(train_sizes, train_scores_avg, 'b-')
plt.plot(train_sizes, valid_scores_avg, 'g-')
plt.show()
clf.fit(x_train_data, y_train_data)
test_predict = clf.predict(x_test_data)
output_data_frame = pd.DataFrame()
output_data_frame['PassengerId'] = test_raw_data['PassengerId']
output_data_frame['Survived'] = test_predict

output_file_name = 'test_predict.csv'
output_data_frame.to_csv(output_file_name, index=False)
