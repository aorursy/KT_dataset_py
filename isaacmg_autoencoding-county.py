# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from torch import nn
import torch
from sklearn.preprocessing import RobustScaler
#from torch import Dataset
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
from math import sqrt
import os

# This code is based on https://medium.com/pytorch/implementing-an-autoencoder-in-pytorch-19baa22647d1
class AE(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.encoder_hidden_layer = nn.Linear(
            in_features=kwargs["input_shape"], out_features=128
        )
        self.encoder_output_layer = nn.Linear(
            in_features=128, out_features=128
        )
        self.decoder_hidden_layer = nn.Linear(
            in_features=128, out_features=128
        )
        self.decoder_output_layer = nn.Linear(
            in_features=128, out_features=kwargs["input_shape"]
        )

    def forward(self, features):
        activation = self.encoder_hidden_layer(features)
        activation = torch.relu(activation)
        code = self.encoder_output_layer(activation)
        code = torch.relu(code)
        activation = self.decoder_hidden_layer(code)
        activation = torch.relu(activation)
        activation = self.decoder_output_layer(activation)
        reconstructed = torch.relu(activation)
        return reconstructed
    
    def encoder_forward(self, features):
        activation = self.encoder_hidden_layer(features)
        activation = torch.relu(activation)
        code = self.encoder_output_layer(activation)
        code = torch.relu(code)
        return code
    
def train_loop(model, batch_size, df, relevant_columns, epochs=10):
    for epoch in range(epochs):
        loss = 0
        i = 0
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
        while i<len(df)/batch_size:
            relevant_rows = df.iloc[i:i+batch_size][relevant_columns].values
            the_tensor = torch.tensor(relevant_rows).float()

            # reset the gradients back to zero
            # PyTorch accumulates gradients on subsequent backward passes
            optimizer.zero_grad()

            # compute reconstructions
            outputs = model(the_tensor)

            # compute training reconstruction loss
            train_loss = criterion(outputs, the_tensor)
            
            # compute accumulated gradients
            train_loss.backward()

            # perform parameter update based on current gradients
            optimizer.step()

            # add the mini-batch training loss to epoch loss
            loss += train_loss.item()
            i+=batch_size
            print(loss)
        # compute the epoch training loss
        print(loss)
        loss = loss / len(df) 
    
    # display the epoch training loss
    print("epoch : {}/{}, loss = {:.6f}".format(epoch + 1, epochs, loss))
relevant_cols = ['TotalPop', 'Men', 'Women', 'Hispanic',
       'White', 'Black', 'Native', 'Asian', 'Pacific',
       'Income', 'IncomeErr', 'IncomePerCap', 'IncomePerCapErr', 'Poverty',
       'ChildPoverty', 'Professional', 'Service', 'Office', 'Construction',
       'Production', 'Drive', 'Carpool', 'Transit', 'Walk', 'OtherTransp',
       'WorkAtHome', 'MeanCommute', 'Employed', 'PrivateWork', 'PublicWork',
       'SelfEmployed', 'FamilyWork', 'Unemployment']
train_df = pd.read_csv("../input/us-census-demographic-data/acs2015_census_tract_data.csv")
test_df = pd.read_csv("../input/us-census-demographic-data/acs2017_county_data.csv")
model = AE(**{"input_shape":33})
# Train the model. Since this is an autoencoder it doesn't matter if we use test set for training.
# We will evaluate the utility of the autoencoder based on embeddings produced. Clustering analysis will be below.
r_train = RobustScaler()
df_train = pd.DataFrame(r_train.fit_transform(train_df[relevant_cols]))
df_train = df_train.dropna()
df_train.columns = relevant_cols
r_test = RobustScaler()
df_test = pd.DataFrame(r_test.fit_transform(test_df[relevant_cols]))
df_test.columns = relevant_cols
train_loop(model, 10, df_train, relevant_columns=['TotalPop', 'Men', 'Women', 'Hispanic', 
                                                    'White', 'Black', 'Native', 'Asian', 'Pacific',
                                                    'Income', 'IncomeErr', 'IncomePerCap', 'IncomePerCapErr', 'Poverty',
                                                    'ChildPoverty', 'Professional', 'Service', 'Office', 'Construction',
                                                    'Production', 'Drive', 'Carpool', 'Transit', 'Walk', 'OtherTransp',
                                                    'WorkAtHome', 'MeanCommute', 'Employed', 'PrivateWork', 'PublicWork',
                                                    'SelfEmployed', 'FamilyWork', 'Unemployment'])


train_loop(model, 10, df_test, relevant_columns=['TotalPop', 'Men', 'Women', 'Hispanic', 
                                                    'White', 'Black', 'Native', 'Asian', 'Pacific',
                                                    'Income', 'IncomeErr', 'IncomePerCap', 'IncomePerCapErr', 'Poverty',
                                                    'ChildPoverty', 'Professional', 'Service', 'Office', 'Construction',
                                                    'Production', 'Drive', 'Carpool', 'Transit', 'Walk', 'OtherTransp',
                                                    'WorkAtHome', 'MeanCommute', 'Employed', 'PrivateWork', 'PublicWork',
                                                    'SelfEmployed', 'FamilyWork', 'Unemployment'])
model.eval()
def preprocess_prod(df, index, relevant_cols, batch_size=1):
    d = torch.tensor(df.iloc[index:index+batch_size][relevant_cols].values)
    return d
relevant_cols = ['TotalPop', 'Men', 'Women', 'Hispanic',
       'White', 'Black', 'Native', 'Asian', 'Pacific',
       'Income', 'IncomeErr', 'IncomePerCap', 'IncomePerCapErr', 'Poverty',
       'ChildPoverty', 'Professional', 'Service', 'Office', 'Construction',
       'Production', 'Drive', 'Carpool', 'Transit', 'Walk', 'OtherTransp',
       'WorkAtHome', 'MeanCommute', 'Employed', 'PrivateWork', 'PublicWork',
       'SelfEmployed', 'FamilyWork', 'Unemployment']
test_df["embeddings"] = list(df_test[relevant_cols].values)
test_df["embeddings"] = test_df["embeddings"].map(lambda x: model.encoder_forward(torch.tensor(x).unsqueeze(0).float()).detach().numpy())
def get_county_from_df(county, state, df):
    return df.query("County=='" + county + "' and State=='" +state+"'")
get_county_from_df("Queens County", "New York", test_df)

from sklearn.metrics.pairwise import cosine_similarity
cosine_similarity(get_county_from_df("Aroostook County", "Maine", test_df).iloc[0]["embeddings"], get_county_from_df("Broward County", "Florida", test_df).iloc[0]["embeddings"])

!pip install umap-learn
import umap
reducer = umap.UMAP()
from bokeh.plotting import figure, show, output_notebook
from bokeh.models import HoverTool, ColumnDataSource, CategoricalColorMapper
from bokeh.palettes import Spectral10, Category20c
from bokeh.palettes import magma
import pandas as pd
output_notebook()
val_arr = [ ]
name_list = [ ] 
for embed in test_df[["embeddings", "County", "State"]].values[0:300]: 
    val_arr.append(embed[0])
    name_list.append(embed[1] +"_" +embed[2])
res = np.vstack(val_arr)
#res = np.ndarray(res)
def make_plot(red, title_list, number=200, color = True, color_mapping_cat=None, color_cats = None, bg_color="white"):   
    digits_df = pd.DataFrame(red, columns=('x', 'y'))
    if color_mapping_cat:
        digits_df['colors'] = color_mapping_cat
    digits_df['digit'] = title_list
    datasource = ColumnDataSource(digits_df)
    plot_figure = figure(
    title='UMAP projection Counties',
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
red = reducer.fit_transform(res)   
make_plot(red, name_list, number=199)
new_york_ny = get_county_from_df("New York County", "New York", test_df).iloc[0]["embeddings"]
def get_most_similar_counties(target_embed, test_df, target_name):
    test_df = test_df.dropna()
    test_df[target_name] = test_df['embeddings'].map(lambda x: cosine_similarity(x, target_embed))
    return test_df.sort_values(by=target_name)
get_most_similar_counties(new_york_ny, test_df, "new_yor_sim")
kansas_chase = get_county_from_df("Chase County", "Kansas", test_df).iloc[0]["embeddings"]
get_most_similar_counties(kansas_chase, test_df, "kansas")
denver_county = get_county_from_df("Denver County", "Colorado", test_df).iloc[0]["embeddings"]
get_most_similar_counties(denver_county, test_df, "d_colorado")