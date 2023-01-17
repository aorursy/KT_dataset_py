import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import json # For processing the JSON data type
from IPython.display import display, Javascript #for displaying the Javascripts inside the notebook

import seaborn as sns #for generating the colors
def ChartJS(chartType, data, options={}, width="700px", height="700px"):
    """ Custom iphython extension allowing chartjs visualizations
    
    Usage:
        chartjs(chartType, data, options, width=1000, height=400)
    
    Args:
        chartType: one of the supported chart type options (line, bar, radar, polarArea, pie, doughnut)
        data: a python dictionary with datasets to be rapresented and related visualization settings, as expected 
              by chart js (see data parameter in http://www.chartjs.org/docs/)
        options: defaults {}; a python dictionary with additional graph options, as expected 
              by chart js (see options parameter in http://www.chartjs.org/docs/)
        width: default 700px
        height: default 400px
        
        NB. data and options structure depends on the chartType
    """
    display(
        Javascript("""
            require(['https://cdnjs.cloudflare.com/ajax/libs/Chart.js/2.7.2/Chart.js'], function(chartjs){
                var chartType="%s";
                var data=%s;
                var options=%s;
                var width="%s";
                var height="%s";

                element.append('<canvas width="' + width + '" height="' + height + '">s</canvas>');                
                var ctx = element.children()[0].getContext("2d");
                
                var myChart = new Chart(ctx, {type: chartType, data:data, options:options});
            });
            """ % (chartType, json.dumps(data), json.dumps(options), width, height)
        )
    )
orig_dataset_path = "../input/"
print(os.listdir(orig_dataset_path))
kiva_loans_df = pd.read_csv(orig_dataset_path + "kiva_loans.csv")
kiva_loans_df.head(1)
cnt_srs = kiva_loans_df['country'].value_counts().head(50)
print("Count of Countries = ", len(cnt_srs))
cnt_srs.describe()
countryNames = cnt_srs.index.tolist()
countryValues = cnt_srs.values.tolist()
colors = sns.color_palette("coolwarm", n_colors=len(cnt_srs)).as_hex()
data = {
    "labels":countryNames,
    "datasets": [
        {
            "label": "Loan Size",
            "backgroundColor":colors,
            "fill":"true",
            "pointColor": "rgba(151,187,205,1)",
            "pointStrokeColor": "#fff",
            "pointHighlightFill": "#fff",
            "pointHighlightStroke": "rgba(151,187,205,1)",
            "data": countryValues
        }
]}

options= {"scales":
            {"yAxes":[{"ticks":{"beginAtZero":"true"}}]
            }
        }
ChartJS('horizontalBar', data, options, "700px", "600px")
