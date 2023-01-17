import numpy as np

import pandas as pd

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import roc_auc_score

from sklearn.preprocessing import LabelEncoder, StandardScaler 

from sklearn.compose import ColumnTransformer,make_column_transformer

from sklearn.pipeline import make_pipeline

from lightgbm import LGBMClassifier

from category_encoders import OneHotEncoder

from sklearn.model_selection import cross_val_predict

from warnings import filterwarnings

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import StandardScaler

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import LabelEncoder

import matplotlib.pyplot as plt

import seaborn as sns

filterwarnings('ignore')
train = pd.read_csv("../input/train_2.csv") 

test = pd.read_csv("../input/test_2.csv") 

train_input = train.drop(['id','target','B_15'],axis = 1)

train_labels = train['target']

app_train = pd.get_dummies(train_input)

imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')

imp_mean.fit(app_train)

train_imputed = imp_mean.transform(app_train)

scaler = StandardScaler()

scaler.fit(train_imputed)

train_imputed = scaler.transform(train_imputed)

features = list(app_train.columns)

random_forest = RandomForestClassifier(n_estimators = 100, random_state = 50, verbose = 1, n_jobs = -1)

random_forest.fit(train_imputed,train_labels)

feature_importance_values = random_forest.feature_importances_

feature_importances = pd.DataFrame({'feature': features, 'importance':feature_importance_values})
def plot_feature_importances(df):

    #Sort features according to importance

    df = df.sort_values('importance', ascending = False).reset_index()

    

    #Normalise the feature importances to add up to one

    df['importance_normalized'] = df['importance'] / df['importance'].sum()

    

    #Make a horizontal bar chart of feature importances

    plt.figure(figsize = (10,6))

    ax = plt.subplot()

    

    #Need to reverse the index to plot most important on top

    ax.barh(list(reversed(list(df.index[:15]))),

           df['importance_normalized'].head(15),

           align = 'center', edgecolor = 'k')

    #Set the yticks and labels

    ax.set_yticks(list(reversed(list(df.index[:15]))))

    ax.set_yticklabels(df['feature'].head(15))

    

    #Plot labeling

    plt.xlabel('Normalized Importance'); plt.title('Feature Importance')

    plt.show()

    

    return df

feature_importances_sorted = plot_feature_importances(feature_importances)
train = pd.read_csv("../input/train_2.csv")

train=train[['target','B_15','B_10','B_3','B_12','B_8','B_7','B_4','D_121','D_26','D_17','B_11','D_56','D_138','D_1','D_40','D_166',

            'C_10','D_102','D_132','D_99','C_14','C_3','C_2','D_13','D_34','D_66','D_2','D_142','D_143','D_21','D_156','D_158','D_37','B_9',

            'D_14','C_12','D_28','D_6','D_29','D_54','D_117','C_5','D_86','D_107','D_30']]

test = pd.read_csv("../input/test_2.csv")

test=test[['B_15','B_10','B_3','B_12','B_8','B_7','B_4','D_121','D_26','D_17','B_11','D_56','D_138','D_1','D_40','D_166',

            'C_10','D_102','D_132','D_99','C_14','C_3','C_2','D_13','D_34','D_66','D_2','D_142','D_143','D_21','D_156','D_158','D_37','B_9',

            'D_14','C_12','D_28','D_6','D_29','D_54','D_117','C_5','D_86','D_107','D_30']]

target_column = "target"

id_column = "id"

categorical_cols = [c for c in test.columns if test[c].dtype in [np.object]]

numerical_cols = [c for c in test.columns if test[c].dtype in [np.float, np.int] and c not in [target_column, id_column]]

preprocess = make_column_transformer(

    (numerical_cols, make_pipeline(SimpleImputer(), StandardScaler())),

    (categorical_cols, OneHotEncoder()))

classifier = make_pipeline(preprocess,LGBMClassifier(n_jobs=-1,eta=0.01,max_depth=4))



oof_pred = cross_val_predict(classifier, 

                             train, 

                             train[target_column], 

                             cv=5,

                             method="predict_proba")

                  

print("LGBMClassifier Cross validation AUC {:.4f}".format(roc_auc_score(train[target_column], oof_pred[:,1])))
sub = pd.read_csv("../input/sample_submission.csv")

classifier.fit(train, train[target_column])

test_preds = classifier.predict_proba(test)[:,1]

sub[target_column] = test_preds

sub.to_csv("submissionWith42Predictors.csv", index=False)