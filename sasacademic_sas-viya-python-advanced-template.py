import sys
import os
import pandas as pd
import numpy as np

### Import swat
import swat
###

# Set Graphing Options
from matplotlib import pyplot as plt
%matplotlib inline

# Set column/row display options to be unlimited
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
#Connect to CAS
s = swat.CAS(os.environ['CASHOST'], os.environ['CASPORT'], None, os.environ.get("SAS_VIYA_TOKEN"))
s
s.sessionProp.setsessopt(caslib='DLUS34')

## We can also enable session metrics to view how long our procedures take to run
s.session.metrics(on=True)
# Load actionsets for analysis (for data prep, modelling, assessing)
actionsets = ['cardinality', 'sampling', 'decisionTree', 'astore','autotune']
[s.loadactionset(i) for i in actionsets]
s.help(actionSet='cardinality')
table_name = "looking_glass_v4"

castbl = s.load_path(
  f'{table_name}.sas7bdat',
  casOut=dict(name='looking_glass_v4', replace=True)
  )

castbl.head()
# Create table of summary statistics in SAS
castbl.cardinality.summarize(
    cardinality=dict(name = 'full_data_card', replace = True)
)
full_data_card = s.CASTable('full_data_card').to_frame() # bring the summary data locally

# Modify SAS output table using Python to present summary statistics
full_data_card['_PCTMISS_'] = (full_data_card['_NMISS_']/full_data_card['_NOBS_'])*100
print('\n', 'Summary Statistics'.center(90, ' '))
full_data_card[['_VARNAME_','_TYPE_','_PCTMISS_','_MIN_','_MAX_','_MEAN_','_STDDEV_','_SKEWNESS_','_KURTOSIS_']].round(2)
#Declare input variables
target =  'upsell_xsell'
input_vars = ['avg_days_susp', 'handset_age_grp', 'Plan_Code_M00', 
              'Curr_Times_Susp', 'curr_days_susp', 'calls_in_pk', 
              'curr_sec_incl_orig', 'bill_data_usg_m02']
variables = [target] + input_vars

select_castbl = castbl[variables]
select_castbl.head(20)
# Create table of summary statistics in SAS
select_castbl.cardinality.summarize(
    varList=[
        {'vars': input_vars}
    ],
    cardinality=dict(name = 'data_card', replace = True)
)

df_data_card = s.CASTable('data_card').to_frame() # bring the summary data locally

# Modify SAS output table using Python to present summary statistics
df_data_card['_PCTMISS_'] = (df_data_card['_NMISS_']/df_data_card['_NOBS_'])*100
print('\n', 'Summary Statistics'.center(90, ' '))
df_data_card[['_VARNAME_','_TYPE_','_PCTMISS_','_MIN_','_MAX_','_MEAN_','_STDDEV_','_SKEWNESS_','_KURTOSIS_']].round(2)
## Note, you can set the following option to fetch more rows of data out Viya memory - this defaults to 10000 rows.
swat.options.cas.dataset.max_rows_fetched=60000
select_castbl.hist(figsize = (15, 10));
# Plot missing values in matplotlib
df_data_miss = df_data_card[df_data_card['_PCTMISS_'] > 0]
tbl_forplot  = pd.Series(list(df_data_miss['_PCTMISS_']), index = list(df_data_miss['_VARNAME_']))
missing_val  = tbl_forplot.plot(kind  = 'bar', title = 'Percentage of Missing Values', color = 'c', figsize = (10, 6));
missing_val.set_ylabel('Percent Missing')
missing_val.set_xlabel('Variable Names');
missingInputs = ['Acct_Plan_Type', 'Plan_Code_M00', 'Curr_Times_Susp', 'bill_data_usg_m02']

s.dataPreprocess.impute(
    table = select_castbl,
    outVarsNamePrefix = 'IMP',
    methodContinuous  = 'MEDIAN',
    inputs            = missingInputs,
    copyAllVars       = True,
    casOut            = dict(caslib= 'DLUS34', name=table_name, replace=True)
)

# Print the first five rows with imputations
imp_input_vars = ['IMP_' + s for s in ['Plan_Code_M00', 'Curr_Times_Susp', 'bill_data_usg_m02']]
total_inputs = input_vars + imp_input_vars

total_inputs.remove('Plan_Code_M00')
total_inputs.remove('Curr_Times_Susp')
total_inputs.remove('bill_data_usg_m02')

# select_castbl = s.CASTable(table_name)[total_inputs]
#select_castbl.head(5)

select_castbl = s.CASTable(table_name)
select_castbl.head(5)
# Create a 70/30 simple random sample split
select_castbl.sampling.srs(
    samppct = 70,
    partind = True,
    seed    = 1,
    output  = dict(
        casOut = dict(
            name=f'{table_name}',
            replace=True), 
        copyVars = 'ALL'
    ),
    outputTables=dict(replace=True)
)
# Set key-word argument shortcuts (common model inputs)
## For models that can handle missing values (decision tree, gradient boosting)
import collections 

params = dict(
    table    = dict(name = table_name, where = '_partind_ = 1'), 
    target   = target, 
    inputs   = total_inputs, 
    nominals = target,
)

# Algorithms to be trained
models = collections.OrderedDict()
models['DT'] = 'Decision Tree'
models['GB'] = 'Gradient Boosting'
models['GBTune'] = 'Tuned Gradient Boosting'
s.decisionTree.dtreeTrain(
    **params,
    casOut = dict(name='DT_model', replace=True),
    code = dict(casout=dict(name='DT_model_code', replace=True)),
    encodeName=True
)
s.CASTable('DT_model').head()
s.decisionTree.gbtreeTrain(
    **params, 
    seed = 1, 
    casOut = dict(name = 'GB_model', replace = True),
    savestate=dict(name='save_gb', replace=True),
    encodeName=True,
)
tuneparams = dict(
    tunerOptions=dict(
    maxTime=60
    ),
    trainOptions = dict(
        table = dict(name = table_name, where = '_partind_= 1'),
        target = target,
        inputs = total_inputs,
        nominals = target,
        savestate=dict(name = 'save_gb_tuned', replace = True),
        seed=1,
        casOut = dict(name='GBTune_model', replace=True)
        )
)
s.autotune.tuneGradientBoostTree(**tuneparams)
def score_model(model):
    score = dict(
        encodeName=True,
        table      = table_name,
        modelTable = model + '_model',
        copyVars   = [target, '_partind_'],
        casOut     = dict(name = '_scored_' + model, replace = True)
    )
    return score

### Gradient Boosting
s.decisionTree.dtreeScore(**score_model('DT'))
s.decisionTree.gbtreeScore(**score_model('GB'))
s.decisionTree.gbtreeScore(**score_model('GBTune'))
s.CASTable('_scored_GB').head()
# Model assessment function
def assess_model(model):
    assess = s.percentile.assess(
        table    = dict(name = '_scored_' + model, where = '_partind_ = 0'),
        inputs   = 'P_' + target + '1',      
        response = target,
        event    = '1',   
    )
    return assess

# Loop through the models and append to the roc_df dataframe
roc_df  = pd.DataFrame()
for i in range(len(models)):
    tmp = assess_model(list(models)[i])
    tmp.ROCInfo['Model'] = list(models.values())[i]
    roc_df = pd.concat([roc_df, tmp.ROCInfo])

# Display stacked confusion matrix using Python
print('\n', 'Confusion Matrix Information'.center(42, ' '))
roc_df[round(roc_df['CutOff'], 2) == 0.5][['Model', 'TP', 'FP', 'FN', 'TN']].reset_index(drop = True)
# Display assessment statistics
#roc_df
# Add misclassification rate calculation
roc_df['Misclassification'] = 1 - roc_df['ACC']

print('\n', 'Misclassification Rate Comparison'.center(37, ' '))
miss = roc_df[round(roc_df['CutOff'], 2) == 0.5][['Model', 'Misclassification']].reset_index(drop = True)
miss.sort_values('Misclassification')
plt.figure(figsize = (7, 6))
for key, grp in roc_df.groupby(['Model']):
    plt.plot(grp['FPR'], grp['Sensitivity'], label = key + ' (C = %0.2f)' % grp['C'].mean())
plt.plot([0,1], [0,1], 'k--')
plt.xlabel('False Postive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.title('ROC Curve (using validation data)');
## Describe ASTORE file
m=s.describe(
     rstore='save_gb_tuned',
     epcode=True
    )

# Load into memory 
castbls = s.load_path(
  'KAGGLETEST_LOOKING_GLASS_1_V3.sas7bdat',
  caslib='ACADCOMP',
  casOut=dict(name='testset', replace=True)
)

# Score
eval_model = castbls.score(
    ds2code = m.epcode,
    table='testset',
    copyVars = ['Customer_ID'],
    out=dict(name=f'{table_name}_out', replace=True),
    rstore='save_gb_tuned'
)
eval_model
s.table.fetch(table=dict(name=f'{table_name}_out'))
keepcolumns = ['I_upsell_xsell']

evaluation = s.CASTable('looking_glass_v4_out').loc[:,keepcolumns]
evaluation.head()
## Output column as CSV - make sure you set the data download limit to be greater than 10000 rows since the test set has more than 10k samples.
swat.options.cas.dataset.max_rows_fetched=60000
evaluation.to_csv('predictionColumn_Python.csv', index=False, float_format='%.12g')
m=s.describe(
     rstore='save_gb_tuned',
     epcode=True
    )
m
# Score code file 
file=open('GB_score.sas', 'w') 
file.write(m.epcode)
file.close()
# ASTORE file
astore_file = s.download(rstore='save_gb_tuned')
with open(m.Key.Key[0].strip()+'.sasast','wb') as file:
    file.write(astore_file['blob'])
#Run this cell to view the score code for the gradient boosting model
#sys.stdout.write(m.epcode)