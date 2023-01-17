import numpy as np

import pandas as pd



import xgboost as xgb

from sklearn.metrics import accuracy_score

from sklearn.datasets import load_iris
train= pd.read_csv('../input/siim-isic-melanoma-classification/train.csv')

test= pd.read_csv('../input/siim-isic-melanoma-classification/test.csv')
train.head()
train.target.value_counts()
train['sex'] = train['sex'].fillna('na')

train['age_approx'] = train['age_approx'].fillna(0)

train['anatom_site_general_challenge'] = train['anatom_site_general_challenge'].fillna('na')
test['sex'] = test['sex'].fillna('na')

test['age_approx'] = test['age_approx'].fillna(0)

test['anatom_site_general_challenge'] = test['anatom_site_general_challenge'].fillna('na')
train['sex'] = train['sex'].astype("category").cat.codes +1

train['anatom_site_general_challenge'] = train['anatom_site_general_challenge'].astype("category").cat.codes +1

train.head()
test['sex'] = test['sex'].astype("category").cat.codes +1

test['anatom_site_general_challenge'] = test['anatom_site_general_challenge'].astype("category").cat.codes +1

test.head()
x_train = train[['sex', 'age_approx','anatom_site_general_challenge']]

y_train = train['target']
x_test = test[['sex', 'age_approx','anatom_site_general_challenge']]
train_DMatrix = xgb.DMatrix(x_train, label= y_train)

test_DMatrix = xgb.DMatrix(x_test)
xgb_model = xgb.XGBClassifier(n_estimators=2000, 

                        max_depth=8, 

                        objective='multi:softprob',

                        seed=0,  

                        nthread=-1, 

                        learning_rate=0.15, 

                        num_class = 2, 

                        scale_pos_weight = (32542/584))
xgb_model.fit(x_train, y_train)
xgb_pred_result = xgb_model.predict_proba(x_test)[:,1]

print(xgb_pred_result)
xgb_df = pd.DataFrame({

        "image_name": test["image_name"],

        "target": xgb_pred_result

    })



xgb_df.to_csv('tuned_XGBClassifier_submission.csv', index=False)
main_submission = pd.read_csv('../input/my-siim-isic-submissions/my_siim_isic_main_submission.csv')

efficient_b7 = pd.read_csv('../input/my-siim-isic-submissions/EfficientNetB7_submission.csv')

efficient_b7_blend_6 = pd.read_csv('../input/my-siim-isic-submissions/EfficientNetB7_submission_Blend_6.csv')

model_blend_0_6 = pd.read_csv('../input/my-siim-isic-submissions/submission_models_blended_0-6.csv')
final_target =  main_submission.target *0.85 + efficient_b7.target *0.05 + xgb_df.target *0.10
result = pd.DataFrame({

        "image_name": test["image_name"],

        "target": final_target

    })



result.to_csv('final_submission_blend.csv', index=False)