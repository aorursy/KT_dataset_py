import os

import torch

!git clone https://github.com/AIStream-Peelout/flow-forecast.git -b metadata_v5

os.chdir('flow-forecast')

!pip install -r requirements.txt

!python setup.py develop
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



   }   

}
import json

with open("meta_config.json", "w+") as j:

    json.dump(meta_embedding_config, j)
import wandb 

import torch

# Uncomment this to login manually

wandb.login()
!pip install pandas --upgrade

import pandas as pd

def format_pd(file_path="../../input/us-census-demographic-data/acs2015_census_tract_data.csv"):

    df = pd.read_csv(file_path)

    df["full_county"] = df["County"] + "_" + df["State"]

    df.dropna().to_csv(file_path.split("/")[-1])

format_pd()

format_pd("../../input/us-census-demographic-data/acs2017_county_data.csv")

from flood_forecast.meta_train import train_function 

trained_model = train_function("PyTorch", meta_embedding_config)
from torch.utils.data import DataLoader

def get_add_embeddings_df(model, relevant_cols, original_file_path="acs2017_county_data.csv"):

    """

    Function to return a given data-frame

    with all the embeddings.

    """

    df = pd.read_csv(original_file_path)

    model.test_data.df["simple_embeddings"] = model.test_data.df[relevant_cols].apply(lambda x: torch.from_numpy(x[relevant_cols].to_numpy()).float().unsqueeze(0), axis=1)

    model.test_data.df["model_embeddings"] = model.test_data.df["simple_embeddings"].map(lambda x: model.model.generate_representation(x))

    model.test_data.df["county"] = df["County"] + "_" + df["State"]

    model.test_data.df["State"] = df["State"]

    return model

trained_model.model.eval()

trained_model = get_add_embeddings_df(trained_model, trained_model.params["dataset_params"]["relevant_cols"])
def get_most_similar_counties(df, county_name, column="model_embeddings"):

    embedding = df[df["county"]==county_name].iloc[0][column]

    cos = torch.nn.CosineSimilarity()

    df["similarity_to_" + county_name] = df[column].map(lambda x: cosine_similarity(embedding.detach(), x.detach()))

    return df.sort_values(by="similarity_to_"+county_name, ascending=False)
from sklearn.metrics.pairwise import cosine_similarity

get_most_similar_counties(trained_model.test_data.df, "New York County_New York")
get_most_similar_counties(trained_model.test_data.df, "Chase County_Kansas")
!pip install umap-learn

import umap

reducer = umap.UMAP()

from bokeh.plotting import figure, show, output_notebook

from bokeh.models import HoverTool, ColumnDataSource, CategoricalColorMapper

from bokeh.palettes import Spectral10, Category20c

from bokeh.palettes import magma

import pandas as pd

output_notebook()
def make_and_fit_reducer(df, reducer, embeddings_col, sorting_column, number_to_fit, name_col="county", ascending=False):

    df = df.sort_values(sorting_column, ascending=ascending)

    the_embeddings = df[embeddings_col].map(lambda x: x.squeeze(0).detach().numpy())[:number_to_fit].to_list()

    plottable_embeddings = reducer.fit_transform(the_embeddings)

    return plottable_embeddings, df[name_col][:number_to_fit], df[:number_to_fit]
largest_counties_embeddings, largest_counties_names, df = make_and_fit_reducer(trained_model.test_data.df, reducer, 

                                                                          "model_embeddings", "TotalPop", 150)
def make_plot(red, title_list, number=200, color = True, df=None, other_cols=None, color_mapping_cat=None, color_cats = None, bg_color="white"):   

    digits_df = pd.DataFrame(red, columns=('x', 'y'))

    if color_mapping_cat:

        digits_df['colors'] = color_mapping_cat

    digits_df['digit'] = title_list

    for col in other_cols:

        digits_df[col] = list(df[col])

    datasource = ColumnDataSource(digits_df)

    plot_figure = figure(

    title='County Embedding Map',

    plot_width=890,

    plot_height=600,

    tools=('pan, wheel_zoom, reset'),

    background_fill_color = bg_color

    )

    plot_figure.legend.location = "top_left",

    plot_figure.add_tools(HoverTool(tooltips="""

    <div>

    <div>

        <img src='@image' style='float: left; margin: 5px 5px 5px 5px'/>

    </div>

    <div>

        <span style='font-size: 10px; color: #224499'></span>

        <span style='font-size: 10px'>@digit</span>

        <p></p>

        <span style='font-size: 10px'> Perecentage White: @White</span>

        <p></p>

        <span style='font-size: 10px'> Perecentage Black: @Black</span>

        <p></p>

        <span style='font-size: 10px'> Median Income: @Income</span>

        

                        

    </div>

    </div>

    """))

    if color:   

        color_mapping = CategoricalColorMapper(factors=title_list, palette=magma(number))

        plot_figure.circle(

            'x',

            'y',

            source=datasource,

            color=dict(field='digit', transform=color_mapping),

            line_alpha=0.6,

            fill_alpha=0.6,

            size=7

        )

        show(plot_figure)

    elif color_mapping_cat:

        color_mapping = CategoricalColorMapper(factors=color_cats, palette=magma(len(color_cats)+2)[2:])

        plot_figure.circle(

            'x',

            'y',

            source=datasource,

            color=dict(field='colors', transform=color_mapping),

            line_alpha=0.6,

            fill_alpha=0.6,

            size=8,

            legend_field='colors'

        )

        show(plot_figure)

    else:

        

        plot_figure.circle(

            'x',

            'y',

            source=datasource,

            color=dict(field='digit'),

            line_alpha=0.6,

            fill_alpha=0.6,

            size=7

        )

        show(plot_figure)



make_plot(largest_counties_embeddings, list(largest_counties_names), number=150, df=df, other_cols=["White", "Black", "Hispanic", "Income"], color_mapping_cat=True, color_cats=df["State"])
smallest_county_embeddings, smallest_counties_names, df = make_and_fit_reducer(trained_model.test_data.df, reducer, 

                                                                          "model_embeddings", "TotalPop", 100, ascending=True)

make_plot(smallest_county_embeddings, list(smallest_counties_names), number=100, df=df, other_cols=["White", "Black", "Hispanic", "Income"], color_mapping_cat=True, color_cats=df["State"])
# Define synthesis configuration file

def make_config(weight_path=None):

    run = wandb.init(project="covid_forecast", entity="covid")

    print(wandb.config)

    wandb_config = wandb.config

    full_config = {                 

    "model_name": "CustomTransformerDecoder",

    "model_type": "PyTorch",

    "model_params": {

      "n_time_series":9,

      "seq_length":5,

      "output_seq_length": 1, 

      "n_layers_encoder": 1,

      "meta_data": {

          "embedding_size":128,

          "output_size":10

      }

     }, 

      "metadata":{

        "path":"meta_config.json",

        "column_id": "full_county",

        "uuid": "Chase County_Kansas",

        "use": wandb_config["meta_data"]



    }

            

    ,

    "dataset_params":

    {  "class": "default",

       "training_path": "ma_test_data.csv",

       "validation_path": "ma_test_data.csv",

       "test_path": "ma_test_data.csv",

       "batch_size":4,

       "forecast_history":5,

       "forecast_length":1,

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

       "lr": 0.3,

       "epochs": 1,

       "batch_size":4



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

             "forecast_history":5,

             "forecast_length":1,

             "target_col": ["rolling_7"],

             "relevant_cols": ["rolling_7", "month", "weekday", "mobility_retail_recreation", "mobility_grocery_pharmacy", "mobility_parks", "mobility_transit_stations", "mobility_workplaces", "mobility_residential"],

             "scaling": "StandardScaler",

             "interpolate_param": False

          }

    } 

    }

    return full_config

#with open("full_config.json", "w+") as f:

#json.dump(full_config, f)
from flood_forecast.trainer import train_function 

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

    sweep_id = wandb.sweep(sweep_config_full)

    wandb.agent(sweep_id, lambda:train_function("Python", make_config()))

!wget -O ma_test_data.csv https://gist.githubusercontent.com/isaacmg/2a4cda7b72555ddd2aa75a978ef15648/raw/ee70bb4181f69a7cece6a6c916726eaa74c2fed8/mass.csv

run_sweep_shit()
import argparse

from typing import Dict

import json

import plotly.graph_objects as go

import wandb

from flood_forecast.pytorch_training import train_transformer_style

from flood_forecast.time_model import PyTorchForecast

from flood_forecast.evaluator import evaluate_model

from flood_forecast.pre_dict import scaler_dict

from flood_forecast.plot_functions import plot_df_test_with_confidence_interval

from flood_forecast.trainer import train_function

full_config = make_config()

train_function("PyTorch", make_config())
from flood_forecast.time_model import PyTorchForecast

def get_meta_representation(column_id: str, uuid: str, meta_model):

    return meta_model.test_data.__getitem__(0, uuid, column_id)[0]



with open("meta_config.json") as f:

    json_data = json.load(f)

    dataset_params2 = json_data["dataset_params"]

    training_path = dataset_params2["training_path"]

    valid_path = dataset_params2["validation_path"]

    name = json_data["model_name"]

    meta_model = PyTorchForecast(name, training_path, valid_path, dataset_params2["test_path"], json_data)

    meta_representation = get_meta_representation("County",

                                                  "Kansas", meta_model)

    
full_config