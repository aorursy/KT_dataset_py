import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
import catboost
from catboost import Pool
import shap
train = pd.read_csv('/kaggle/input/hse-practical-ml-1/car_loan_train.csv')
test = pd.read_csv('/kaggle/input/hse-practical-ml-1//car_loan_test.csv')
cat_features = ['branch_id', 'supplier_id', 'manufacturer_id', 'Current_pincode_ID', 'Employment.Type',
'State_ID', 'Employee_code_ID',   'PERFORM_CNS.SCORE.DESCRIPTION']

drop_features = ['MobileNo_Avl_Flag', 'Passport_flag', 'Driving_flag', 'SEC.NO.OF.ACCTS', 'PAN_flag',
                'SEC.DISBURSED.AMOUNT', 'Aadhar_flag', 'branch_id_PERFORM_CNS.SCORE']
text_features = ['text_feature']
def pri_features(df):
#    df['ltv_less_90'] = df.ltv < 90
#    df['ltv_less_80'] = df.ltv < 80
    
    df['def_perc_6m'] = df['NEW.ACCTS.IN.LAST.SIX.MONTHS']/df['DELINQUENT.ACCTS.IN.LAST.SIX.MONTHS']
    df['disbursedamount_assetcosts'] = df['disbursed_amount']/df['asset_cost']
    df['credit_cost'] = df['asset_cost'] * df['ltv']
    df['credit_cost_minus_disbursed'] = df['credit_cost'] - df['disbursed_amount']
    
    
    for key in ['PRI']:        
        df[f'perc_act/nof_{key}'] = df[f'{key}.ACTIVE.ACCTS']/df[f'{key}.NO.OF.ACCTS']
        df[f'perc_overdue/noof_{key}'] = df[f'{key}.OVERDUE.ACCTS']/df[f'{key}.NO.OF.ACCTS']
        df[f'perc_act/overdue_{key}'] = df[f'{key}.ACTIVE.ACCTS']/df[f'{key}.OVERDUE.ACCTS']

        df[f'perc_act/curb_{key}'] = df[f'{key}.ACTIVE.ACCTS']/df[f'{key}.CURRENT.BALANCE']
        df[f'perc_act/sanc_amount_{key}'] = df[f'{key}.ACTIVE.ACCTS']/df[f'{key}.SANCTIONED.AMOUNT']
        df[f'perc_act_disb/amount_{key}'] = df[f'{key}.ACTIVE.ACCTS']/df[f'{key}.DISBURSED.AMOUNT']

        
        df[f'sanc_amount/disb_amount_{key}'] = df[f'{key}.SANCTIONED.AMOUNT']/df[f'{key}.DISBURSED.AMOUNT']
        df[f'sanc_amount/curr_b_{key}'] = df[f'{key}.SANCTIONED.AMOUNT']/df[f'{key}.CURRENT.BALANCE']
        
        
        df[f'perc_overdue/curb_{key}'] = df[f'{key}.OVERDUE.ACCTS']/df[f'{key}.CURRENT.BALANCE']
        df[f'perc_overdue/sanc_amount_{key}'] = df[f'{key}.OVERDUE.ACCTS']/df[f'{key}.SANCTIONED.AMOUNT']
        df[f'perc_overdue/disbursed_{key}'] = df[f'{key}.OVERDUE.ACCTS']/df[f'{key}.DISBURSED.AMOUNT']
        
    df['perc_act_all'] = df['NEW.ACCTS.IN.LAST.SIX.MONTHS']/df['PRI.NO.OF.ACCTS']
    df['credit_cost'] = df['asset_cost'] * df['ltv']
    df['credit_cost_minus_disbursed'] = df['credit_cost'] - df['disbursed_amount']
    
    df['ACTIVE.ACCTS']=df['PRI.ACTIVE.ACCTS']+df['SEC.ACTIVE.ACCTS']
    df['CURRENT.BALANCE']=df['PRI.CURRENT.BALANCE']+df['SEC.CURRENT.BALANCE']
    df['DISBURSED.AMOUNT']=df['PRI.DISBURSED.AMOUNT']+df['SEC.DISBURSED.AMOUNT']
    df['NO.OF.ACCTS']=df['SEC.NO.OF.ACCTS']+df['PRI.NO.OF.ACCTS']
    df['SANCTIONED.AMOUNT']=df['PRI.SANCTIONED.AMOUNT']+df['SEC.SANCTIONED.AMOUNT']
    df['INSTAL.AMT']=df['PRIMARY.INSTAL.AMT']+df['SEC.INSTAL.AMT']
    df['SANCTION_DISBURSED']=df['SANCTIONED.AMOUNT']-df['DISBURSED.AMOUNT']
    
    df['ACTIVE.ACCTS_minus']=df['PRI.ACTIVE.ACCTS']-df['SEC.ACTIVE.ACCTS']
    df['CURRENT.BALANCE_minus']=df['PRI.CURRENT.BALANCE']-df['SEC.CURRENT.BALANCE']
    df['DISBURSED.AMOUNT_minus']=df['PRI.DISBURSED.AMOUNT']-df['SEC.DISBURSED.AMOUNT']
    df['NO.OF.ACCTS_minus']=df['SEC.NO.OF.ACCTS']-df['PRI.NO.OF.ACCTS']
    df['SANCTIONED.AMOUNT_minus']=df['PRI.SANCTIONED.AMOUNT']-df['SEC.SANCTIONED.AMOUNT']
    df['INSTAL.AMT_minus']=df['PRIMARY.INSTAL.AMT']-df['SEC.INSTAL.AMT']
    return df


def flags_features(df):
    df['count_flags'] = df['Aadhar_flag'] + df['PAN_flag'] + df['VoterID_flag'] \
    + df['Driving_flag'] + df['Passport_flag']
#    df['driving_and_passport_flags'] = df['Driving_flag'] + df['Passport_flag']
#     df['all_flags'] = df['Aadhar_flag'] * df['PAN_flag'] * df['VoterID_flag'] \
#     * df['Driving_flag'] * df['Passport_flag']
#    df['vouter_and_aadhar'] = df.VoterID_flag + df.Aadhar_flag
    df['voter_aadhar_driving'] = df.VoterID_flag + df.Aadhar_flag + df.Driving_flag
    return df


def burea_score_decoding(df):
#    df['burea_no_history'] = df['PERFORM_CNS.SCORE'] == 0
    df['burea_14'] = df['PERFORM_CNS.SCORE'] == 14
    df['burea_18'] = df['PERFORM_CNS.SCORE'] == 18
    df['burea_17'] = df['PERFORM_CNS.SCORE'] == 17
    df['burea_16'] = df['PERFORM_CNS.SCORE'] == 16
#    df['burea_over_300'] = df['PERFORM_CNS.SCORE'] > 299
#    df['burea_less_300'] = df['PERFORM_CNS.SCORE'] <= 299
    return df


def common_categories(df):
    df['common_cats'] = pd_train['PERFORM_CNS.SCORE.DESCRIPTION'].replace({'C-Very Low Risk':'Low', 'A-Very Low Risk':'Low',
                                                       'B-Very Low Risk':'Low', 'D-Very Low Risk':'Low',
                                                       'F-Low Risk':'Low', 'E-Low Risk':'Low', 'G-Low Risk':'Low',
                                                       'H-Medium Risk': 'Medium', 'I-Medium Risk': 'Medium',
                                                       'J-High Risk':'High', 'K-High Risk':'High','L-Very High Risk':'Very High',
                                                       'M-Very High Risk':'Very High','Not Scored: More than 50 active Accounts found':'Not Scored',
                                                       'Not Scored: Only a Guarantor':'Not Scored','Not Scored: Not Enough Info available on the customer':'Not Scored',
                                                        'Not Scored: No Activity seen on the customer (Inactive)':'Not Scored','Not Scored: No Updates available in last 36 months':'Not Scored',
                                                       'Not Scored: Sufficient History Not Available':'Not Scored', 'No Bureau History Available':'Not Scored'
                                                       })
    return df


def months_to_years(df: pd.DataFrame, column_new : str, column_old : str) -> pd.DataFrame:
    df[column_new] = df[column_old].str.split('y').str.get(0).astype(int)*12 + \
    df[column_old].str.split(' ').str.get(1).str.split('m').str.get(0).astype(int)
    df.drop(column_old, inplace=True, axis=1)
    return df
    
def date_to_days(df: pd.DataFrame, column_new : str, column_old : str) -> pd.DataFrame:
    df[column_old] = df[column_old].astype('datetime64[ns]')
    now = pd.Timestamp('now')
    df[column_old] = pd.to_datetime(df[column_old], format='%m%d%y')
    df[column_old] = df[column_old].where(df[column_old] < now, 
                                                      df[column_old] -  np.timedelta64(100, 'Y'))
    df[column_new] = (now - df[column_old]).dt.days
    df.drop(column_old, inplace=True, axis=1)
    return df


def fillna_cats(df, cat_features):
    for cat_feature in cat_features:
        df[cat_feature] = df[cat_feature].fillna('NaN')
    return df

def employment_branch_manufacter_state_cnsscore(df):
    df['text_feature'] = df['Employment.Type'].astype(str) + ' ' \
    + df['branch_id'].astype(str) + ' ' + df['manufacturer_id'].astype(str) + df['State_ID'].astype(str) + \
    df['PERFORM_CNS.SCORE.DESCRIPTION'].astype(str)
    return df

def text_flags(df):
    df['text_feature'] = df['Employment.Type'].astype(str) + ' ' \
    + df['branch_id'].astype(str) + ' ' + df['manufacturer_id'].astype(str) + ' ' + df['State_ID'].astype(str) + \
    df['PERFORM_CNS.SCORE.DESCRIPTION'].astype(str)
    return df


def add_smooth_mean(df, alpha):
    for group_column in ['Employee_code_ID']:
        for target in ['ltv', 'PERFORM_CNS.SCORE']:
            mean = df[target].mean()
            agg = df.groupby(group_column)[target].agg(['count', 'mean'])
            counts = agg['count']
            means = agg['mean']
            smooth = (counts * means + alpha * mean) / (counts + alpha)
            df[f'{group_column}_{target}_me'] = df[group_column].map(smooth)
    return df


def add_percentiles(df):
    for group_column in ['Current_pincode_ID', 'State_ID', 'Employment.Type', 'branch_id']:
        df = pd.merge(df, 
              df.groupby(group_column).size().reset_index().rename({0 : f'{group_column}_size'}, 
                                                                   axis=1), on=group_column,
                     how='left').fillna(0.5)
        for target in ['ltv', 'PERFORM_CNS.SCORE']:
            df[f'{group_column}_rank'] = df.groupby(group_column)[target].rank(ascending=False)
            df[f'{group_column}_{target}'] = df[f'{group_column}_rank'] / df[f'{group_column}_size']
            
    return df


def df_preprocessing(df):
    df = pri_features(df)
    df = flags_features(df)
    df = burea_score_decoding(df)
#    df = employment_branch_manufacter_state_cnsscore(df)
    df = months_to_years(df, 'AVERAGE_ACCT_AGE_EM_MESES', 'AVERAGE.ACCT.AGE') 
    df = months_to_years(df, 'CREDIT_HISTORY_LENGTH_EM_MESES', 'CREDIT.HISTORY.LENGTH')
    df = date_to_days(df, 'Idade', 'Date.of.Birth')
    df = date_to_days(df, 'DIAS_DESEMBOLSO', 'DisbursalDate')
  #  df = common_categories(df)    
    df = fillna_cats(df, cat_features)
    df = add_percentiles(df)
#    df = add_smooth_mean(df, 5)
    return df
test['target'] = 0
test['flag'] = 0
train['flag'] = 1
df = pd.concat([train, test], axis=0)
df2 = df_preprocessing(df)
df2.drop(drop_features, axis=1, inplace=True)
model = catboost.CatBoostClassifier(cat_features=cat_features, class_weights=[1.0, 3.56758],
                                random_state=42, depth=4, l2_leaf_reg=20, learning_rate=0.05,
                                random_strength = 5,  bagging_temperature=5)
model.fit(df2[df2.flag == 1].drop(['target'], axis=1), y=df2[df2.flag == 1].target, verbose=False)
X = df2[df2.flag == 1].drop(['target'], axis=1)
y = df2[df2.flag == 1].target
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(Pool(X, y, cat_features=cat_features))
shap.summary_plot(shap_values, X, plot_size = (20, 20), max_display = 40)
preds = []

for random_state in [42, 282, 123, 1234, 12345, 1, 1337]:
    model = catboost.CatBoostClassifier(cat_features=cat_features, class_weights=[1.0, 3.56758],
                                random_state=random_state, depth=4, l2_leaf_reg=20, learning_rate=0.05,
                                random_strength = 5,  bagging_temperature=5)
    model.fit(df2[df2.flag == 1].drop(['target'], axis=1), y=df2[df2.flag == 1].target, verbose=False)
    predicts = model.predict_proba(df2[df2.flag == 0].drop(['target'], axis=1))
    preds.append(predicts[:, 1])
results = pd.DataFrame(preds).transpose()
pred_proba = results.mean(axis=1)
results = pd.DataFrame([df[df.flag == 0].UniqueID.values, pred_proba.values]).transpose()
res = pd.merge(test.reset_index()[['UniqueID', 'index']], 
         results,
         left_on ='UniqueID',
         right_on = 0)
res = res[['index', 1]]
res.columns = ['ID', 'Predicted']
res.to_csv('submission.csv', index=False)