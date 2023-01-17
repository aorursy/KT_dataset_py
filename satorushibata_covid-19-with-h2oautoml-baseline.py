import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import h2o
from h2o.automl import H2OAutoML
h2o.init(max_mem_size='16G')
# Read the data
path = '/kaggle/input/stanford-covid-vaccine'
train_df = pd.read_json(f'{path}/train.json',lines=True)
test_df = pd.read_json(f'{path}/test.json', lines=True)
sample_sub_df = pd.read_csv(f'{path}/sample_submission.csv')
train_df.head()
test_df.head()
sample_sub_df.head()
# Calculate Means of targets
train_df['reactivity'] = train_df['reactivity'].apply(lambda x: np.mean(x))
train_df['deg_Mg_pH10'] = train_df['deg_Mg_pH10'].apply(lambda x: np.mean(x))
train_df['deg_pH10'] = train_df['deg_pH10'].apply(lambda x: np.mean(x))
train_df['deg_Mg_50C'] = train_df['deg_Mg_50C'].apply(lambda x: np.mean(x))
train_df['deg_50C'] = train_df['deg_50C'].apply(lambda x: np.mean(x))
train_df.head()
# Drop unnecessary columns for now
train_df = train_df.drop(['id', 'index', 'reactivity_error', 'deg_error_Mg_pH10', 'deg_error_pH10', 'deg_error_Mg_50C', 'deg_error_50C', 'SN_filter', 'signal_to_noise', 'deg_pH10', 'deg_50C'], axis=1)
train_df.head()
# Split data in features and labels
X_train = train_df.drop(['reactivity', 'deg_Mg_pH10', 'deg_Mg_50C'], axis=1)
Y_train = train_df[['reactivity', 'deg_Mg_pH10', 'deg_Mg_50C']]
X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, test_size=0.15)
def featurize(df):
    
    df['total_A_count'] = df['sequence'].apply(lambda s: s.count('A'))
    df['total_G_count'] = df['sequence'].apply(lambda s: s.count('G'))
    df['total_U_count'] = df['sequence'].apply(lambda s: s.count('U'))
    df['total_C_count'] = df['sequence'].apply(lambda s: s.count('C'))
    
    df['total_dot_count'] = df['structure'].apply(lambda s: s.count('.'))
    df['total_ob_count'] = df['structure'].apply(lambda s: s.count('('))
    df['total_cb_count'] = df['structure'].apply(lambda s: s.count(')'))
    
    return df
X_train = featurize(X_train)
X_test = featurize(X_test)
X_train = X_train.drop(['sequence', 'structure', 'predicted_loop_type'], axis=1)
X_test = X_test.drop(['sequence', 'structure', 'predicted_loop_type'], axis=1)
X_train.head()
X_test.head()
scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
htrain = pd.concat([Y_train, pd.DataFrame(X_train)], axis=1)
htrain = h2o.H2OFrame(htrain)
htrain['reactivity'].asfactor
htrain['deg_Mg_pH10'].asfactor
htrain['deg_Mg_50C'].asfactor
htrain
htest = pd.concat([Y_test, pd.DataFrame(X_test)], axis=1)
htest = h2o.H2OFrame(htest)
htest['reactivity'].asfactor
htest['deg_Mg_pH10'].asfactor
htest['deg_Mg_50C'].asfactor
htest
# Convert test data to predict
test = featurize(test_df.drop(['index', 'id'], axis=1))
test = test.drop(['sequence', 'structure', 'predicted_loop_type'], axis=1)
test = scaler.transform(test)
test = h2o.H2OFrame(test, column_names=['0','1','2','3','4','5','6','7','8'])
# train & predict an each
def h2o_ML(htrain, htest, test):
    Y1 = 'reactivity'
    Y2 = 'deg_Mg_pH10'
    Y3 = 'deg_Mg_50C'
    X1 = htrain.columns.remove(Y1)
    X2 = htrain.columns.remove(Y2)
    X3 = htrain.columns.remove(Y3)
    
    def h2o_train_test(X, Y, htrain, htest, test):       
        aml = H2OAutoML(max_runtime_secs=(3600 * 2),  # 2 hours for all Y
                        max_models=30, # None, # no limt
                        seed=2000,
                        nfolds=3,
                        keep_cross_validation_predictions=True
                       )
        aml.train(x=X, y=Y, training_frame=htrain, leaderboard_frame=htest)
        
        # View the AutoML Leaderboard
        lb = aml.leaderboard
        print(lb.head(rows=lb.nrows))  # Print all rows instead of default (10 rows)
        
        # predict
        preds = aml.predict(test).as_data_frame().values
        preds = pd.DataFrame(preds, columns=[Y])
        
        return preds
        
    h2o_1 = h2o_train_test(X1, Y1, htrain, htest, test)
    h2o_2 = h2o_train_test(X2, Y2, htrain, htest, test)
    h2o_3 = h2o_train_test(X3, Y3, htrain, htest, test)
    
    result = pd.concat([h2o_1, h2o_2, h2o_3], axis=1)
    
    return result
# Run functions
model = h2o_ML(htrain, htest, test)
model
h2o.shutdown(prompt=False)
# Create submission csv
submission_df = model.loc[model.index.repeat(list(test_df['seq_length']))].reset_index(drop=True)
submission_df = submission_df.rename(columns={0: 'reactivity', 1: 'deg_Mg_pH10', 2: 'deg_Mg_50C'})
submission_df['id_seqpos'] = sample_sub_df['id_seqpos']
submission_df['deg_pH10'] = 0.0
submission_df['deg_50C'] = 0.0
submission_df = submission_df[['id_seqpos', 'reactivity', 'deg_Mg_pH10', 'deg_pH10', 'deg_Mg_50C', 'deg_50C']]
submission_df
# Save that CSV for submission
submission_df.to_csv('submission.csv', index=False)