model_id = 'ICN2247013813647982800'

score_threshold = 0.000001



gcp_service_account_json = '/kaggle/input/gcloudserviceaccountkey/kaggle-playground-170215-4ece6a076f22.json'

gcp_project_id = 'kaggle-playground-170215'

#AutoML package

!pip install google-cloud-automl
from concurrent.futures import ProcessPoolExecutor as PoolExecutor, as_completed

        

from google.cloud import automl_v1beta1

from tqdm import tqdm

import operator

import os

import pandas as pd

import sys

import time

#generates predictions for all sirnas above the prediction threshold

def get_prediction(file_path, project_id, model_id):



    name = 'projects/{}/locations/us-central1/models/{}'.format(project_id, model_id)

    

    with open(file_path, 'rb') as ff:

        content = ff.read()

        payload = {'image': {'image_bytes': content }}



    params = {'score_threshold':str(score_threshold)}

    request = prediction_client.predict(name, payload, params)

        

    return request

#convert sirnas to integrater

def make_int(s):

        try: 

            int(s)

            return int(s) 

        except ValueError:

            return 1109 #1109 is invalid so it's skipped over by the process function. That's why I set the label to 1109 for an error. 

def process(i,df_sample_submission,project_id):

    id_code = df_sample_submission.index[i]

    if id_code in df_solution.index:

        return None



    exp_len = id_code.find('_')

    experiment = id_code[0:exp_len]

    plate = id_code[(exp_len+1):(exp_len+2)]

    well = id_code[exp_len+3:]

    pred_dict = {}



    res = []



    for site in range(1,3):

        file_path = '../input/recursion_rgb_512/testrgb512/testRGB512/{}_{}_{}_s{}.png'.format(experiment,plate,well,site)

      

        try:

            prediction_request = get_prediction(file_path, project_id,  model_id)

        except Exception as e:

            print('Something got wrong: ', e)

            return None







        for prediction in prediction_request.payload:

            label = make_int(prediction.display_name)

            if label <= 1108:

                pred_dict[label] = float(prediction.classification.score)





        sirna_prediction = max(pred_dict.items(), key=operator.itemgetter(1))[0] 

        confidence = pred_dict[sirna_prediction]



        res.append({

            'id_code': id_code, 

            'site': site, 

            'sirna_prediction': sirna_prediction, 

            'confidence': confidence

        })



    return res

def generated_predictions_with_pool_executor(max_workers,gcp_project_id):

    results = []



    df_sample_submission = pd.read_csv('../input/recursion-cellular-image-classification/sample_submission.csv',index_col=[0])

    

    with PoolExecutor(max_workers=max_workers) as executor:

        futures_list = [executor.submit(process, i,df_sample_submission,gcp_project_id) for i in range(len(df_sample_submission))]

        for f in tqdm(as_completed(futures_list), total=len(futures_list)):

            results.append(f.result())



    nb_escaped = 0

    for r in results:

        if r is None:

            nb_escaped += 1

            continue

        for site in r:

            df_solution.loc[site['id_code'], ['site{}_sirna'.format(site['site']),'site{}_confidence'.format(site['site'])]] = [site['sirna_prediction'], site['confidence']]



    #df_solution.to_csv('./submissions/submission_{}.csv'.format(model_id))









solution_file_path = ',/submissions/submission_{}.csv'.format(model_id)

if os.path.exists(solution_file_path):

    df_solution = pd.read_csv(solution_file_path,index_col=[0])

else:

    df_solution = pd.DataFrame(columns=['site1_sirna','site1_confidence','site2_sirna','site2_confidence'])

    df_solution.index.name = 'id_code'



prediction_client = automl_v1beta1.PredictionServiceClient.from_service_account_json(gcp_service_account_json)



generated_predictions_with_pool_executor(20,gcp_project_id)

#run the command a few more times just to pick up any rows that reported errors

generated_predictions_with_pool_executor(5,gcp_project_id)

generated_predictions_with_pool_executor(5,gcp_project_id)



site1_wins = df_solution['site1_confidence'] >= df_solution['site2_confidence']

site2_wins = df_solution['site2_confidence'] > df_solution['site1_confidence']



df_solution.loc[site1_wins,'sirna'] = (df_solution[site1_wins])['site1_sirna']

df_solution.loc[site2_wins,'sirna'] = (df_solution[site2_wins])['site2_sirna']

df_solution['sirna'] = df_solution['sirna'].astype(int)

df_solution['sirna'].sort_index().to_csv('submit_{}.csv'.format(model_id),header=True)
