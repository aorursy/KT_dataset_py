## Importing Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
%matplotlib inline
## function to add data to plot
def annot_plot(ax,w,h):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for p in ax.patches:
        ax.annotate('{}'.format(p.get_height()), (p.get_x()+w, p.get_height()+h))
survey_data=pd.read_csv('../input/survey_results_public.csv')
survey_data.head()
survey_data.shape

survey_data.groupby('AIFuture')['AIFuture'].agg(['count']).sort_values('count').reset_index().plot(x='AIFuture',y='count',kind='barh');
survey_data['AIFuture'].value_counts()
survey_data.groupby('AIDangerous')['AIDangerous'].agg(['count']).sort_values('count').reset_index().plot(x='AIDangerous',y='count',kind='barh');
survey_data['AIDangerous'].value_counts()
survey_data.groupby('AIInteresting')['AIInteresting'].agg(['count']).sort_values('count').reset_index().plot(x='AIInteresting',y='count',kind='barh');
survey_data['AIInteresting'].value_counts()
survey_data.groupby('AIResponsible')['AIResponsible'].agg(['count']).sort_values('count').reset_index().plot(x='AIResponsible',y='count',kind='barh');
survey_data['AIResponsible'].value_counts()
