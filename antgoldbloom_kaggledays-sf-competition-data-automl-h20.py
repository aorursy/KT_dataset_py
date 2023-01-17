

import numpy as np 

import pandas as pd 

import glob #library for navigating file structure

import re #regular expressions



#path to submission files

fps = "../input/kaggle-days-top-ten-submission-files/multiplesubmissions/multiplesubmissions/**/*.csv"

#path to solution file

fs = "../input/hackathon-solution/derived.csv"



#iterate through submission files and add to dataframe

need_to_setup_dataframe = True

for filename in glob.iglob(fps, recursive=True):



    df_tmp = pd.read_csv(filename,index_col='id' )

        

    if need_to_setup_dataframe == True:

        df_submission_files = df_tmp

        need_to_setup_dataframe = False

        

    else:

        df_submission_files = df_submission_files.join(df_tmp)   

    

    df_submission_files[re.search('000[0-9].*\/',filename)[0][5:-1]] = df_submission_files['target']

    del(df_submission_files['target'])

    



#Read in solution file and 

df_solution = pd.read_csv(fs,index_col='id')



df_submission_files = df_submission_files.reindex(sorted(df_submission_files.columns), axis=1)

df_submission_files= df_submission_files.join(df_solution['Usage'])



df_submission_files = df_submission_files[df_submission_files['Usage'] == 'Private']

df_solution = df_solution[df_solution['Usage'] == 'Private']



del(df_submission_files['Usage'])

df_submission_files.columns = ['9hr Overfitness','ALDAPOP','Arno Candel @ H2O.ai','Erin (H2O AutoML 100 mins)','Erkut & Mark','Google AutoML','Shlandryn','Sweet Deal']
#calculate AUC to replicate leaderboard and try a few ensembles

from sklearn import metrics



df_ensemble_results = pd.DataFrame(columns=['team','score'])



for team in df_submission_files.columns:

    fpr, tpr, thresholds = metrics.roc_curve(df_solution['target'],df_submission_files[team])

    df_ensemble_results = df_ensemble_results.append({'team':team,"score":metrics.auc(fpr,tpr)},ignore_index=True)



fpr, tpr, thresholds = metrics.roc_curve(df_solution['target'],df_submission_files[['Erkut & Mark','Google AutoML','Sweet Deal']].mean(axis=1)

)

df_ensemble_results = df_ensemble_results.append({'team':'Erkut & Mark,Google AutoML,Sweet Deal',"score":metrics.auc(fpr,tpr)},ignore_index=True)



fpr, tpr, thresholds = metrics.roc_curve(df_solution['target'],df_submission_files[['Erkut & Mark','Google AutoML']].mean(axis=1)

)

df_ensemble_results = df_ensemble_results.append({'team':'Erkut & Mark,Google AutoML',"score":metrics.auc(fpr,tpr)},ignore_index=True)



fpr, tpr, thresholds = metrics.roc_curve(df_solution['target'],df_submission_files[['Erin (H2O AutoML 100 mins)','Google AutoML']].mean(axis=1)

)

df_ensemble_results = df_ensemble_results.append({'team':'Erin (H2O AutoML 100 mins),Google AutoML',"score":metrics.auc(fpr,tpr)},ignore_index=True)

#Display leaderboard

df_ensemble_results.index = df_ensemble_results['team'] 

del(df_ensemble_results['team'] )

df_ensemble_results.sort_values(by='score',ascending=False)