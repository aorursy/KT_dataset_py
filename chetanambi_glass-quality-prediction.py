!pip install pycaret
import numpy as np
import pandas as pd  
import matplotlib.pyplot as plt
import seaborn as sns
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
train = pd.read_csv('/kaggle/input/Glass_Quality_Participants_Data/Train.csv')
test = pd.read_csv('/kaggle/input/Glass_Quality_Participants_Data/Test.csv')
sub = pd.read_excel('/kaggle/input/Glass_Quality_Participants_Data/Sample_Submission.xlsx')
train.shape, test.shape, sub.shape
train.head(3)
train['class'].value_counts()
train.nunique()
train.dtypes
from pycaret.classification import *
exp_clf101 = setup(data = train, target = 'class', session_id=123,
                   normalize = True, 
                   transformation = True,
                   ignore_low_variance = True,
                   remove_multicollinearity = True,
                   feature_interaction=True                   
                  )
#help(create_model)
compare_models()
cat = create_model('catboost')
lgbm = create_model('lightgbm')
et = create_model('et')
tuned_catboost = tune_model('catboost')
tuned_lgbm = tune_model('lightgbm')
tuned_et = tune_model('et')
#tuned_et = tune_model('et', optimize = 'AUC')
plot_model(tuned_lgbm, plot='auc')
plot_model(tuned_et, plot='auc')
plot_model(tuned_lgbm, plot='feature')
plot_model(tuned_xgb, plot='feature')
plot_model(tuned_lgbm, plot = 'confusion_matrix')
plot_model(tuned_xgb, plot = 'confusion_matrix')
evaluate_model(tuned_et)
interpret_model(tuned_catboost)
predict_model(tuned_et);
predict_model(tuned_lgbm);
predict_model(tuned_catboost);
final_catboost = finalize_model(tuned_catboost)
predict_model(final_catboost);
final_lgbm = finalize_model(tuned_lgbm)
predict_model(final_lgbm);
final_et = finalize_model(tuned_et)
predict_model(final_et);
bagged_et = ensemble_model(lgbm)
#print(bagged_et)
boosted_et = ensemble_model(lgbm, method = 'Boosting')
boosted_et2 = ensemble_model(lgbm, method = 'Boosting', n_estimators=50)
tuned_boosted_et = tune_model('lightgbm', ensemble=True, method='Boosting')
blend_hard = blend_models(method = 'hard')
blend_soft = blend_models(method = 'soft')
gbc = create_model('gbc', verbose = False)
et = create_model('et', verbose = False)
lgbm = create_model('lightgbm', verbose = False)
blend_specific_soft = blend_models(estimator_list = [gbc, et, lgbm], method = 'soft')
blend_specific_hard = blend_models(estimator_list = [gbc, et, lgbm], method = 'hard')
stack_soft = stack_models([gbc, et, lgbm])
stack_hard = stack_models([gbc, et, lgbm], method='hard')
predict_model(bagged_et);
predict_model(boosted_et); 
predict_model(tuned_boosted_et); 
predict_model(blend_hard); 
predict_model(blend_soft); 
predict_model(blend_specific_soft); 
predict_model(blend_specific_hard); 
predict_model(stack_soft); 
predict_model(stack_hard); 
final_model = finalize_model(blend_soft)
predict_model(final_model);
predictions = predict_model(final_model, data=test)
predictions.head()
y_pred = pd.DataFrame(predictions['Score']).rename(columns={'Score':'2'})
y_pred['1'] = 1-predictions['Score']
y_pred = y_pred[['1','2']]
y_pred.head()
y_pred.to_excel('Output.xlsx', index=False)
from IPython.display import HTML
import pandas as pd
import numpy as np
import base64

def create_download_link(df, title = "Download CSV file", filename = "Output.csv"):  
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode())
    payload = b64.decode()
    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'
    html = html.format(payload=payload,title=title,filename=filename)
    return HTML(html)

create_download_link(y_pred)
