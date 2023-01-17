#Author: Francesco Schena
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



from sklearn.metrics import log_loss

# Any results you write to the current directory are saved as output.
!pip install pydotplus

!pip install category_encoders



import pydotplus

from sklearn.tree import export_graphviz



def tree_graph_to_png(tree, feature_names, png_file_to_save):

    tree_str = export_graphviz(tree, feature_names=feature_names, 

                                     filled=True, out_file=None)

    graph = pydotplus.graph_from_dot_data(tree_str)  

    graph.write_png(png_file_to_save)
df_train = pd.read_csv('/kaggle/input/ods-mlclass-dubai-2019-03-lecture3-hw/train.csv')

print(df_train.shape)



#Remove missing values

#df_train = df_train[(df_train.astype(str) != ' ?').all(axis=1)]



df_test = pd.read_csv('/kaggle/input/ods-mlclass-dubai-2019-03-lecture3-hw/test.csv')

print(df_test.shape)



df_test['target'] = np.nan

#df = pd.concat([df_train_dedupped, df_test])

df = pd.concat([df_train, df_test])

print('df has',df.shape[0],'rows')
df.loc[(df['occupation'] == ' ?') & (df['workclass'] != ' ?')]['occupation'] = 'Never-worked'

south_american_countries = [' Columbia',' Cuba',' Dominican-Republic',' Ecuador',' El-Salvador',' Guatemala',' Haiti',' Honduras',' Jamaica',' Mexico',' Nicaragua',' Peru',' Puerto-Rico',' Trinadad&Tobago']

df['native-country-group'] = df['native-country'].map(lambda x: 'United-States' if x == ' United-States' else('South-America' if x in south_american_countries  else 'overseas'))

df['native-country-group'].value_counts(normalize=True)
df['occupation-group'] = df['occupation'].map(lambda x: 'Other' if x in [' Other-service',' Priv-house-serv',' Handlers-cleaners'] else x)

df['occupation-group'].value_counts(normalize=True)
#df['marital-status'].value_counts(normalize=True)

df['marital-status-group'] = df['marital-status'].map(lambda x: 'Married' if x in [' Married-civ-spouse',' Married-spouse-absent',' Married-AF-spouse'] else ('Divorced/Separated/Widowed' if x in [' Divorced',' Separated',' Widowed'] else x))

df['marital-status-group'].value_counts(normalize = True)
columns_to_drop = ['education',

                   'relationship',

                   #'capital-gain',

                   #'capital-loss',

                   #'workclass',

                   'fnlwgt'

                   #,'native-country'#,'occupation','marital-status'

                  ]

df = df.drop(columns_to_drop,axis=1)
df.info()
from sklearn import preprocessing

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import StandardScaler, FunctionTransformer, MinMaxScaler

from sklearn.compose import ColumnTransformer

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.feature_selection import SelectKBest, chi2, f_classif, mutual_info_classif

from sklearn.preprocessing import PolynomialFeatures

import category_encoders as ce
# CV bootstrapping

from sklearn.model_selection import GridSearchCV, cross_val_score



numeric_features = df.select_dtypes(include=['int64', 'float64']).drop(['target','uid'],axis=1).columns

categorical_features = df.select_dtypes(include=['object']).columns

X = df.loc[df['target'].notna()].drop(['target'], axis=1)

y = df.loc[df['target'].notna()]['target']

le = preprocessing.LabelEncoder()

label_encoder = le.fit(y)

y = label_encoder.transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 15)



encoder_list = [#ce.backward_difference.BackwardDifferenceEncoder, 

               #ce.basen.BaseNEncoder,

               #ce.binary.BinaryEncoder,

                #ce.cat_boost.CatBoostEncoder,

                #ce.hashing.HashingEncoder,

                #ce.helmert.HelmertEncoder,

                ce.james_stein.JamesSteinEncoder,

                #ce.one_hot.OneHotEncoder,

                #ce.leave_one_out.LeaveOneOutEncoder,

                ce.m_estimate.MEstimateEncoder,

                ce.ordinal.OrdinalEncoder,

                #ce.polynomial.PolynomialEncoder,

                #ce.sum_coding.SumEncoder,

                ce.target_encoder.TargetEncoder,

                ce.woe.WOEEncoder

                ]

encoder_array = []; score_array = []; clipped_score_array = []; model_array = []; pipe_array = []





for encoder in encoder_list:

    

    numeric_transformer = Pipeline(steps=[

    ('imputer', SimpleImputer(strategy='median')),

    #('scaler', StandardScaler()),

        #('scaler2',MinMaxScaler())

        ])

    categorical_transformer = Pipeline(steps=[

    ('imputer', SimpleImputer(missing_values=' ?'#,strategy='most_frequent'

                              ,strategy='constant', fill_value='missing'

                             )),

    ('woe', encoder()),

    #('scaler2',MinMaxScaler())

    ])

    

    preprocessor = ColumnTransformer(transformers=[

        ('num', numeric_transformer, numeric_features),

        ('cat', categorical_transformer, categorical_features)])

    

    pipe = Pipeline(steps=[('preprocessor', preprocessor)

                           ,('interactions', PolynomialFeatures(interaction_only=True))

                           #,('feature_selector',SelectKBest(chi2))

                           

                      ,('classifier', DecisionTreeClassifier(criterion='gini',splitter='best',max_depth=10,max_features=13,min_samples_split=550,min_samples_leaf=30,random_state = 10))])

    

    #model2 = pipe.fit(X, y)

    tree_params = {'classifier__max_depth': range(8,10,1),

        #           ,

        'classifier__min_samples_split': [200,450,550,700]#,'classifier__min_samples_leaf': range(1,71,10)

                  ,'classifier__max_features': range(8,70,4)

                  }

    tree_grid2 = GridSearchCV(pipe, tree_params, scoring = 'neg_log_loss',cv=3, n_jobs=-1, verbose=True)

    tree_grid2.fit(X, y)

    

    

    print(encoder)

    print(tree_grid2.best_params_)

    print(tree_grid2.best_score_)
numeric_features = df.select_dtypes(include=['int64', 'float64']).drop(['target','uid'],axis=1).columns

categorical_features = df.select_dtypes(include=['object']).columns

X = df.loc[df['target'].notna()].drop(['target'], axis=1)

y = df.loc[df['target'].notna()]['target']

le = preprocessing.LabelEncoder()

label_encoder = le.fit(y)

y = label_encoder.transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 15)



encoder_list = [#ce.backward_difference.BackwardDifferenceEncoder, 

               #ce.basen.BaseNEncoder,

               #ce.binary.BinaryEncoder,

                #ce.cat_boost.CatBoostEncoder,

                #ce.hashing.HashingEncoder,

                #ce.helmert.HelmertEncoder,

                #ce.james_stein.JamesSteinEncoder,

                #ce.one_hot.OneHotEncoder,

                #ce.leave_one_out.LeaveOneOutEncoder,

                ce.m_estimate.MEstimateEncoder,

                ce.ordinal.OrdinalEncoder,

                ce.polynomial.PolynomialEncoder,

                ce.sum_coding.SumEncoder,

                ce.target_encoder.TargetEncoder,

                ce.woe.WOEEncoder

                ]

encoder_array = []; score_array = []; clipped_score_array = []; model_array = []; pipe_array = []



for encoder in encoder_list:

    

    numeric_transformer = Pipeline(steps=[

    ('imputer', SimpleImputer(strategy='median')),

    ('scaler', StandardScaler()),

        #('scaler2',MinMaxScaler())

        ])

    categorical_transformer = Pipeline(steps=[

    ('imputer', SimpleImputer(missing_values=' ?'#,strategy='most_frequent'

                              ,strategy='constant', fill_value='missing'

                             )),

    ('woe', encoder()),

    #('scaler2',MinMaxScaler())

    ])

    

    preprocessor = ColumnTransformer(transformers=[

        ('num', numeric_transformer, numeric_features),

        ('cat', categorical_transformer, categorical_features)])

    

    pipe = Pipeline(steps=[('preprocessor', preprocessor)

                           ,('interactions', PolynomialFeatures(interaction_only=True))

                           #,('feature_selector',SelectKBest(chi2,k=14))

                           #,('classifier1', LogisticRegression(lr=0.1,num_iter=3000))

                      ,('classifier2', DecisionTreeClassifier(criterion='gini',splitter='best',max_depth=11,max_features =56, min_samples_split=750,min_samples_leaf=20,random_state = 10))

                          ])

    model2 = pipe.fit(X_train, y_train)

    

    y_pred_test = model2.predict_proba(X_test)[:, 1]

    score = log_loss(y_test, y_pred_test)

    clipped_score = log_loss(y_test, np.clip(y_pred_test,0.0025,0.9975))

    encoder_array.append(encoder);score_array.append(score);clipped_score_array.append(clipped_score); pipe_array.append(pipe);model_array.append(model2)

    models = pd.DataFrame({'encoder':encoder_array,'score':score_array,'clipped_score':clipped_score_array,'pipe':pipe_array,'model':model_array})

    print(encoder)

    print(score)

    print(clipped_score)

    
best_encoder = models.loc[models['clipped_score'].idxmin()]['encoder']

best_pipe = models.loc[models['clipped_score'].idxmin()]['pipe']

best_model = models.loc[models['clipped_score'].idxmin()]['model']

best_score = models.loc[models['clipped_score'].idxmin()]['score']

best_clipped_score = models.loc[models['clipped_score'].idxmin()]['clipped_score']



best_model2 = best_pipe.fit(df[df['target'].notna()].drop(['target'],axis=1), df[df['target'].notna()]['target'])

y_pred = best_model2.predict_proba(df.loc[df['target'].isna()].drop(['target'],axis=1))[:, 1]

y_pred_clip = np.clip(y_pred,0.0025,0.9975)

print('The encoder used was',best_encoder,'original score:',round(best_score,6),'clipped score:',round(best_clipped_score,6))
df_submit = pd.DataFrame({

    'uid': df.loc[df['target'].isna()]['uid'],

    'target': y_pred_clip

    #y_pred

})



df_submit.to_csv('/kaggle/working/submit.csv', index=False)



print('csv file created. No. rows:',len(df_submit))
!head /kaggle/working/submit.csv