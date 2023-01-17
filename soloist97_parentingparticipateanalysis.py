# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd 



edu_df = pd.read_csv('/kaggle/input/xAPI-Edu-Data/xAPI-Edu-Data.csv')

edu_df.head()
edu_df.keys()
pd.crosstab(edu_df['ParentAnsweringSurvey'],edu_df['Class'])
cross_table = pd.crosstab(edu_df['ParentAnsweringSurvey'],edu_df['StudentAbsenceDays'])

cross_table
# 计算ParentAnsweringSurvey和StudentAbsenceDays的lift值

total = cross_table.sum().sum()



lift_score = cross_table['Above-7']['Yes']/total / (cross_table['Above-7'].sum()/total * (cross_table['Above-7']['Yes']+cross_table['Under-7']['Yes']) / total)

print('lift({})={:.3f}'.format('Above-7, Yes', lift_score))
pd.crosstab(edu_df['ParentschoolSatisfaction'],edu_df['Class'])
pd.crosstab(edu_df['ParentschoolSatisfaction'],edu_df['StudentAbsenceDays'])
pd.crosstab(edu_df['Relation'],edu_df['Class'])
pd.crosstab(edu_df['Relation'],edu_df['StudentAbsenceDays'])
pd.crosstab(edu_df['ParentAnsweringSurvey'],edu_df['ParentschoolSatisfaction'])
pd.crosstab(edu_df['Relation'],edu_df['ParentAnsweringSurvey'])
pd.crosstab(edu_df['Relation'],edu_df['ParentschoolSatisfaction'])
# 仅包含父亲的子表

father_sub_df = edu_df.loc[edu_df['Relation']=='Father']
pd.crosstab(father_sub_df['ParentAnsweringSurvey'],father_sub_df['Class'])
# 仅包含母亲的子表

mother_sub_df = edu_df.loc[edu_df['Relation']=='Mum']
pd.crosstab(mother_sub_df['ParentAnsweringSurvey'],mother_sub_df['Class'])
from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split, KFold

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

from sklearn.model_selection import cross_val_score



RANDOM_STATE = 42
def get_features_and_target(df, features_to_use, target_key='Class'):

    

    assert target_key not in features_to_use, 'target labels should not be a part of features'

    

    target = df[target_key]

    features = df[features_to_use]

    

    label_encoder = LabelEncoder()

    cat_colums = features.dtypes.pipe(lambda features: features[features=='object']).index

    for col in cat_colums:

        label_encoder.fit(features[col])

        print('>>> Mapping')

        print({k:v for k, v in zip(label_encoder.classes_, range(len(label_encoder.classes_)))})

        features.loc[:, col] = label_encoder.transform(features[col])

    

    label_encoder.fit(target)

    print('>>> Mapping')

    print({k:v for k, v in zip(label_encoder.classes_, range(len(label_encoder.classes_)))})

    target = label_encoder.transform(target)

    

    return features, target    
def k_fold_logistic(features, target, total_exp_num, k_fold, max_iter, rand_state=42):

    # 10 fold validation



    all_scores = list()



    for i in range(total_exp_num):

        kf = KFold(n_splits=k_fold, shuffle=True)

        for j, (train_idx, test_idx) in enumerate(kf.split(features, target)):

            k_X_train = features.values[train_idx]

            k_y_train = target[train_idx]

            k_X_test = features.values[test_idx]

            k_y_test = target[test_idx]



            clf = LogisticRegression(solver='lbfgs', multi_class='multinomial',max_iter=max_iter, random_state=rand_state+i)

            clf.fit(k_X_train,k_y_train)

            k_prediction = clf.predict(k_X_test)

            k_score = accuracy_score(k_y_test, k_prediction)



            all_scores.append(k_score) 

    print("Accuracy: {:.2f} (+/- {:.2f})".format(np.mean(all_scores), np.std(all_scores) * 2))
# 特征选择

LOGISTIC_FEATURE_TO_USE = ['gender', 'Relation', 'StageID', 'Topic', 'StudentAbsenceDays', 'ParentAnsweringSurvey', 'ParentschoolSatisfaction']

features, target = get_features_and_target(edu_df, LOGISTIC_FEATURE_TO_USE, target_key='Class')
print(LOGISTIC_FEATURE_TO_USE)

k_fold_logistic(features, target, total_exp_num=10, k_fold=10, max_iter=5000, rand_state=RANDOM_STATE)
LOGISTIC_FEATURE_TO_USE = ['gender', 'Relation', 'StageID', 'Topic', 'StudentAbsenceDays', 'ParentschoolSatisfaction']



features, target = get_features_and_target(edu_df, LOGISTIC_FEATURE_TO_USE, target_key='Class')
print(LOGISTIC_FEATURE_TO_USE)

k_fold_logistic(features, target, total_exp_num=10, k_fold=10, max_iter=5000, rand_state=RANDOM_STATE)
LOGISTIC_FEATURE_TO_USE = ['gender', 'Relation', 'StageID', 'Topic', 'StudentAbsenceDays', 'ParentAnsweringSurvey']



features, target = get_features_and_target(edu_df, LOGISTIC_FEATURE_TO_USE, target_key='Class')
print(LOGISTIC_FEATURE_TO_USE)

k_fold_logistic(features, target, total_exp_num=10, k_fold=10, max_iter=5000, rand_state=RANDOM_STATE)
LOGISTIC_FEATURE_TO_USE = ['gender', 'StageID', 'Topic', 'StudentAbsenceDays', 'ParentAnsweringSurvey', 'ParentschoolSatisfaction']



features, target = get_features_and_target(edu_df, LOGISTIC_FEATURE_TO_USE, target_key='Class')
print(LOGISTIC_FEATURE_TO_USE)

k_fold_logistic(features, target, total_exp_num=10, k_fold=10, max_iter=5000, rand_state=RANDOM_STATE)
from sklearn import tree



RAND_STATE = 42
DT_FEATURE_TO_USE = ['gender', 'Relation', 'StageID', 'Topic', 'StudentAbsenceDays', 'ParentAnsweringSurvey', 'ParentschoolSatisfaction']

features, target = get_features_and_target(edu_df, DT_FEATURE_TO_USE, target_key='Class')



X_train, X_test, y_train, y_test = train_test_split(features, target, random_state=RAND_STATE)



dt = tree.DecisionTreeClassifier(max_depth=4)

dt = dt.fit(X_train, y_train)
tree.plot_tree(dt) 
import graphviz



dot_data = tree.export_graphviz(dt, out_file=None) 

graph = graphviz.Source(dot_data)

graph.render("Edu Dataset")



dot_data = tree.export_graphviz(dt, out_file=None, 

                                feature_names=DT_FEATURE_TO_USE,  

                                class_names=['H', 'L', 'M'],  

                                filled=True, rounded=True,  

                                special_characters=True)

graph = graphviz.Source(dot_data) 
graph
from sklearn.tree.export import export_text



r = export_text(dt, feature_names=DT_FEATURE_TO_USE)

print(r)
pred = dt.predict(X_test)

score = accuracy_score(y_test,pred)

report = classification_report(y_test,pred)

print('==== acc ====')

print(score)

print('==== report ====')

print(report)