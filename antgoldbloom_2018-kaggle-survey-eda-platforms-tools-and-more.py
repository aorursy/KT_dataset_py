import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import warnings
import seaborn as sns
from matplotlib_venn import venn3
warnings.filterwarnings("ignore")
import matplotlib.pylab as plt
plt.style.use('fivethirtyeight')
%matplotlib inline

# Read Multiple Choice
mc = pd.read_csv('../input/multipleChoiceResponses.csv')

def create_plot(question_number,parts,drop_parts):

    list_question_parts = []
    for part in range(1,parts+1):
        if part not in drop_parts:
            list_question_parts.append('Q' + str(question_number) + '_Part_' + str(part))
    
    ide_qs = mc[list_question_parts].drop(0)

    ide_qs.columns=ide_qs.mode().values[0]
    ide_qs_binary = ide_qs.fillna(0).replace('[^\\d]',1, regex=True)

    color_pal = sns.color_palette("Blues", parts)

    (ide_qs_binary.sum() / ide_qs_binary.count()).sort_values().plot(kind='barh', figsize=(10, 10),
         color=color_pal)

    plt.show()
    
def create_plot_from_single_column(question_number,parts):
    color_pal = sns.color_palette("Blues", parts)
    mc['Q' + str(question_number)].value_counts()[0:parts].sort_values().plot(kind='barh', figsize=(10, 10),color=color_pal)

create_plot_from_single_column(6,10)
create_plot_from_single_column(7,10)
create_plot_from_single_column(3,10)
create_plot(15,7,[None])
create_plot(27,20,[19])
create_plot(28,43,[42])
create_plot(30,25,[24])
create_plot(29,28,[27])
create_plot(13,15,[None])
create_plot(14,11,[10])
create_plot(16,18,[None])
create_plot_from_single_column(17,17)
create_plot(19,19,[None])
create_plot_from_single_column(20,18)
create_plot(21,13,[None])
create_plot(31,12,[None])
color_pal = sns.color_palette("Blues", 6)
df_pct_of_time = mc[['Q34_Part_1','Q34_Part_2','Q34_Part_3','Q34_Part_4','Q34_Part_5','Q34_Part_6']][1:].dropna().astype(float).mean()
df_pct_of_time.index = ['Gathering data','Cleaning data','Visualizing data','Model building/selection','Putting models into production','Finding inights and communicating results']
df_pct_of_time.sort_values().plot(kind='barh', figsize=(10, 10),color=color_pal)


create_plot_from_single_column(10,6)
create_plot(11,7,[None])
create_plot(36,12,[12])
create_plot(38,18,[None])
