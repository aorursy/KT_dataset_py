import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from catboost import CatBoostClassifier

from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import classification_report, roc_curve, precision_recall_curve, plot_precision_recall_curve

from scipy.stats import chi2_contingency, shapiro, probplot, mannwhitneyu

import warnings
warnings.simplefilter('ignore')

%matplotlib inline
DATASET_PATH = '../input/credit-default/train.csv'
VALID_DATASET_PATH = '../input/credit-default/test.csv'
df_train_base = pd.read_csv(DATASET_PATH)
df_valid_base = pd.read_csv(VALID_DATASET_PATH)
target_name = 'Credit Default'
feature_names = df_train_base.columns.drop(target_name).tolist()
feature_names_cat = ['Home Ownership', 'Years in current job', 'Tax Liens', 'Term',  
                     'Purpose', 'Bankruptcies', 'Number of Credit Problems']
df_train_base.isna().sum()
Current_Loan_Amount_max = 800000
Maximum_Open_Credit_max = df_train_base['Maximum Open Credit'].quantile(.95)
medians = df_train_base[['Annual Income', 'Credit Score']].median()
df_train_base['Years in current job'].fillna('nan', inplace=True)
df_train_base['Months since last delinquent'].fillna(0, inplace=True)
df_train_base['Bankruptcies'].fillna(0, inplace=True)
df_train_base['Annual Income'].fillna(medians['Annual Income'], inplace=True)
df_train_base['Credit Score'].fillna(medians['Credit Score'], inplace=True)
df_train_base[feature_names].hist(figsize=(16,16), bins=20, grid=False);
df_train_base.loc[df_train_base['Current Loan Amount'] > Current_Loan_Amount_max, 'Current Loan Amount'] = Current_Loan_Amount_max
df_train_base.loc[df_train_base['Maximum Open Credit'] > Maximum_Open_Credit_max, 'Maximum Open Credit'] = Maximum_Open_Credit_max
        
df_train_base[['Tax Liens', 'Bankruptcies', 'Number of Credit Problems']] = df_train_base[['Tax Liens', 'Bankruptcies', 'Number of Credit Problems']].astype(int)
df_train_base.loc[df_train_base['Number of Credit Problems'] == 4,'Number of Credit Problems'] = 1
df_train_base.loc[df_train_base['Tax Liens'] == 2,'Tax Liens'] = 1
X = df_train_base[feature_names]
y = df_train_base[target_name]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    shuffle=True,
    test_size=0.1,
    random_state=42,
    stratify=y
)
params = {
    'eval_metric': 'F1',
    'auto_class_weights': 'Balanced',
    'silent': True,
    'cat_features': feature_names_cat,
    'one_hot_max_size': 20,
    'early_stopping_rounds': 50,
    'boosting_type': 'Ordered',
    'allow_writing_files': False
}
%%time

cbr_final_model = CatBoostClassifier(
    **params,
    depth=8,
    iterations=100,
    learning_rate=0.5
)

cbr_final_model.fit(X_train, y_train, eval_set=(X_test, y_test))
def evaluate_preds(model, X_train, X_test, y_train, y_test):
    
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    cv_score = cross_val_score(
        model,
        X_train,
        y_train,
        scoring='f1',
        cv=StratifiedKFold(
            n_splits=5,
            random_state=42,
            shuffle=True
        )
    )
    get_classification_report(y_train, y_train_pred, y_test, y_test_pred, cv_score)
    
    fpr, tpr, thresholds = roc_curve(y_test, y_test_pred, pos_label=1)
    plt.rcParams['figure.figsize'] = 5, 5
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], color='grey', linestyle='dashed')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve')
    
    disp = plot_precision_recall_curve(model, X_test, y_test)
    disp.ax_.set_title('Precision-Recall curve')
def get_classification_report(y_train_true, y_train_pred, y_test_true, y_test_pred, cv_score):
    
    print('Train\n\n' + classification_report(y_train_true, y_train_pred))
    print('Test\n\n' + classification_report(y_test_true, y_test_pred))
    print('Confusion Matrix\n')
    print(pd.crosstab(y_test_true, y_test_pred))
    print('\nCross Validation Score: ' + str(round(cv_score.mean(),3)))
evaluate_preds(cbr_final_model, X_train, X_test, y_train, y_test)
feature_importances = pd.DataFrame(
    zip(X.columns, cbr_final_model.get_feature_importance()),
    columns=['feature_name', 'importance']
)

feature_importances.sort_values(by='importance', ascending=False, inplace=True)
feature_importances
Current_Loan_Amount_max = 800000
Maximum_Open_Credit_max = df_valid_base['Maximum Open Credit'].quantile(.95)
medians = df_valid_base[['Annual Income', 'Credit Score']].median()
df_valid_base['Years in current job'].fillna('nan', inplace=True)
df_valid_base['Months since last delinquent'].fillna(0, inplace=True)
df_valid_base['Bankruptcies'].fillna(0, inplace=True)
df_valid_base['Annual Income'].fillna(medians['Annual Income'], inplace=True)
df_valid_base['Credit Score'].fillna(medians['Credit Score'], inplace=True)
df_valid_base.loc[df_valid_base['Current Loan Amount'] > Current_Loan_Amount_max, 'Current Loan Amount'] = Current_Loan_Amount_max
df_valid_base.loc[df_valid_base['Maximum Open Credit'] > Maximum_Open_Credit_max, 'Maximum Open Credit'] = Maximum_Open_Credit_max
        
df_valid_base[['Tax Liens', 'Bankruptcies', 'Number of Credit Problems']] = df_valid_base[['Tax Liens', 'Bankruptcies', 'Number of Credit Problems']].astype(int)
df_valid_base.loc[df_valid_base['Number of Credit Problems'] == 4,'Number of Credit Problems'] = 1
df_valid_base.loc[df_valid_base['Tax Liens'] == 2,'Tax Liens'] = 1
y_test_pred = cbr_final_model.predict(df_valid_base)

preds_final = pd.DataFrame()
preds_final = pd.DataFrame({'Id': np.arange(0,y_test_pred.shape[0]), 'Credit Default': y_test_pred})
preds_final.to_csv('./predictions.csv', index=False, encoding='utf-8', sep=',')
preds_final.head(10)