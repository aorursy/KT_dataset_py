import os
import torch
!git clone https://github.com/AIStream-Peelout/flow-forecast.git -b metadata_v5
os.chdir('flow-forecast')
!wget -O "weight.pth" https://storage.googleapis.com/coronaviruspublicdata/experiments/09_September_202006_03AM_model.pth
!wget -O "config.json" https://storage.googleapis.com/coronaviruspublicdata/experiments/09_September_202006_03AM.json
!pip install -r requirements.txt
!python setup.py develop  
import json
meta_embedding_config = {                 
    "model_name": "BasicAE",
    "model_type": "PyTorch",
    "model_params": {
       "input_shape":33,
       "out_features":128
     }, 
    "dataset_params":
    {  "class": "AutoEncoder",
       "training_path": "acs2015_census_tract_data.csv",
       "validation_path": "acs2015_census_tract_data.csv",
       "test_path": "acs2017_county_data.csv",
       "batch_size":4,
       "train_end": 6000,
       "valid_start":6001,
       "valid_end": 7000,
       "test_start":0,
       "relevant_cols": ['TotalPop', 'Men', 'Women',
       'Hispanic', 'White', 'Black', 'Native', 'Asian', 'Pacific',
       'Income', 'IncomeErr', 'IncomePerCap', 'IncomePerCapErr', 'Poverty',
       'ChildPoverty', 'Professional', 'Service', 'Office', 'Construction',
       'Production', 'Drive', 'Carpool', 'Transit', 'Walk', 'OtherTransp',
       'WorkAtHome', 'MeanCommute', 'Employed', 'PrivateWork', 'PublicWork',
       'SelfEmployed', 'FamilyWork', 'Unemployment'],
       "Scaling":"StandardScaler",
       "interpolate": False
    },
    "training_params":
    {
       "criterion":"MSE",
       "optimizer": "Adam",
       "lr": 0.3,
       "epochs": 4,
       "batch_size":4,
       "optim_params":
       {
       }
    
    },
    "GCS": False,
    
    "wandb": {
       "name": "flood_forecast_circleci",
       "project": "repo-flood_forecast",
       "tags": ["auto_encoder", "circleci"]
    },
   "metrics":["MSE"],

   "inference_params":{
      "hours_to_forecast":1

   },
    "weight_path":"weight.pth"
}

with open("meta_config.json", "w+") as j:
    json.dump(meta_embedding_config, j)
def make_config(weight_path=None):
    run = wandb.init(project="covid_forecast", entity="covid")
    print(wandb.config)
    wandb_config = wandb.config
    full_config = {                 
    "model_name": "CustomTransformerDecoder",
    "model_type": "PyTorch",
    "model_params": {
      "n_time_series":9,
      "seq_length":wandb_config["forecast_history"],
      "output_seq_length": wandb_config["out_seq_length"], 
      "n_layers_encoder": wandb_config["number_encoder_layers"],
      "dropout": wandb_config["dropout"],
      "meta_data": {
          "embedding_size":128,
          "output_size":10
      }
     }, 
      "metadata":{
        "path":"meta_config.json",
        "column_id": "full_county",
        "uuid": "Middlesex County_Massachusetts",
        "use": wandb_config["meta_data"]

    }
            
    ,
    "dataset_params":
    {  "class": "default",
       "training_path": "ma_test_data.csv",
       "validation_path": "ma_test_data.csv",
       "test_path": "ma_test_data.csv",
       "batch_size":4,
       "forecast_history":wandb_config["forecast_history"],
       "forecast_length":wandb_config["out_seq_length"],
       "train_end": 100,
       "valid_start":101,
       "valid_end": 126,
       "test_start":101,
       "target_col": ["rolling_7"],
       "relevant_cols": ["rolling_7", "month", "weekday", "mobility_retail_recreation", "mobility_grocery_pharmacy", "mobility_parks", "mobility_transit_stations", "mobility_workplaces", "mobility_residential"],
       "scaler": "StandardScaler", 
       "interpolate": False
    },
    "training_params":
    {
       "criterion":"RMSE",
       "optimizer": "Adam",
       "optim_params":
       {

       },
       "lr": wandb_config["lr"],
       "epochs": 10,
       "batch_size":wandb_config["batch_size"]

    },
    "GCS": False,
    "sweep":True,
    "wandb":False,
    "forward_params":{},
    "metrics":["MSE"],
    "inference_params":
    {     
         "datetime_start":"2020-06-18",
          "hours_to_forecast":16, 
          "test_csv_path":"ma_test_data.csv",
          "decoder_params":{
            "decoder_function": "simple_decode", 
            "unsqueeze_dim": 1},
          "dataset_params":{
             "file_path": "ma_test_data.csv",
             "forecast_history":wandb_config["forecast_history"],
             "forecast_length":wandb_config["out_seq_length"],
             "target_col": ["rolling_7"],
             "relevant_cols": ["rolling_7", "month", "weekday", "mobility_retail_recreation", "mobility_grocery_pharmacy", "mobility_parks", "mobility_transit_stations", "mobility_workplaces", "mobility_residential"],
             "scaling": "StandardScaler",
             "interpolate_param": False
          }
    } 
    }
    wandb.config.update(full_config)
    return full_config

from flood_forecast.trainer import train_function 
import wandb
sweep_config_full = {
  "name": "Default sweep",
  "method": "grid",
  "parameters": {
        "meta_data":{
            "values":[True, False]
        },
        "batch_size": {
            "values": [10, 25, 30]
        },
        "lr":{
            "values":[0.001, 0.0001, .01, .1]
        },
        "forecast_history":{
            "values":[5, 10, 11, 15]
        },
        "out_seq_length":{
            "values":[1, 2, 5]
        },
        "number_encoder_layers":
        {
            "values":[1, 2, 4, 5, 6]
        }, 
        "dropout":
        {
            "values":[0.1, 0.2]
        },
        "use_mask":{
            "values":[True, False]
        }

        
}}
def run_sweep_shit():
    sweep_id = wandb.sweep(sweep_config_full, project="covid-forecast")
    wandb.agent(sweep_id, lambda:train_function("PyTorch", make_config()))

!wget -O ma_test_data.csv https://gist.githubusercontent.com/isaacmg/2a4cda7b72555ddd2aa75a978ef15648/raw/ee70bb4181f69a7cece6a6c916726eaa74c2fed8/mass.csv
!pip install pandas --upgrade
import pandas as pd
def format_pd(file_path="../../input/us-census-demographic-data/acs2015_census_tract_data.csv"):
    df = pd.read_csv(file_path)
    df["full_county"] = df["County"] + "_" + df["State"]
    df.dropna().to_csv(file_path.split("/")[-1])
format_pd()
format_pd("../../input/us-census-demographic-data/acs2017_county_data.csv")
run_sweep_shit()





