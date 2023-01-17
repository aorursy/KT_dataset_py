!pip install pycaret -q
!wget -q https://www.flaticon.com/svg/static/icons/svg/3313/3313446.svg

!wget -q https://www.flaticon.com/svg/static/icons/svg/89/89977.svg
import pandas as pd

import numpy  as np



import plotly.express as px

import matplotlib.pyplot as plt



from pycaret.nlp import create_model, setup, plot_model, assign_model, evaluate_model, tune_model
raw_data = pd.read_csv('../input/trip-advisor-hotel-reviews/tripadvisor_hotel_reviews.csv')

raw_data['Review'] = raw_data['Review'].str.replace('*',' stars')

raw_data
# Check Rating Distribution

fig = px.histogram(raw_data,

             x = 'Rating',

             title = 'Histogram of Review Rating',

             template = 'plotly_dark',

             color = 'Rating',

             color_discrete_sequence= px.colors.sequential.Blues_r,

             opacity = 0.8,

             height = 525,

             width = 835,

            )



fig.update_yaxes(title='Count')

fig.show()
# Configure the first experiment - General Topic Modeling

experiment_1 = setup(raw_data,

                          target = 'Review',

                          session_id = 451,

                          log_experiment = True,

                          log_plots = True,

                          experiment_name='General Topic Modeling'

                         )
lda_grid = {'num_topics':list(range(2,5))}
# Create an Latent Dirilech Allocation model

#lda_model = create_model('lda')

tuned_lda = tune_model(model = 'lda',

                       multi_core = True,

                       supervised_target = 'Rating',

                       custom_grid = list(range(2,5)),                      

                      )



# Assign model results

lda_results = assign_model(tuned_lda)

lda_results.head()
#evaluate_model(lda_model)
plot_model(lda_model, plot = 'topic_model')