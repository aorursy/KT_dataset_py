!pip install matplotlib-venn
!pip install venn
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import warnings
import seaborn as sns
from matplotlib_venn import venn3, venn2
from sklearn.preprocessing import OneHotEncoder 
from sklearn import preprocessing
from kmodes.kmodes import KModes
#from venn import venn, pseudovenn
warnings.filterwarnings("ignore")
import matplotlib.pylab as plt
plt.style.use('fivethirtyeight')
%matplotlib inline

# Read Multiple Choice Responses
mc = pd.read_csv('../input/kaggle-survey-2019/multiple_choice_responses.csv')

# Load other text responses
text = pd.read_csv('../input/kaggle-survey-2019/other_text_responses.csv')
# Data Prep for IDE

# Pull just IDE Questions
ide_qs = mc[['Q16_Part_1','Q16_Part_2','Q16_Part_3','Q16_Part_4','Q16_Part_5',
             'Q16_Part_6','Q16_Part_7','Q16_Part_8','Q16_Part_9','Q16_Part_10',
             'Q16_Part_11','Q16_Part_12']].drop(0)


# Rename Columns for IDE Type
column_rename = {'Q16_Part_1': 'Jupyter (JupyterLab, Jupyter Notebooks, etc)',
                 'Q16_Part_2': 'RStudio',
                'Q16_Part_3': 'PyCharm',
                'Q16_Part_4': 'Atom',
                'Q16_Part_5': 'MATLAB',
                'Q16_Part_6': 'Visual Studio / Visual Studio Code',
                'Q16_Part_7': 'Spyder',
                'Q16_Part_8': 'Vim / Emacs',
                'Q16_Part_9': 'Notepad++',
                'Q16_Part_10': 'Sublime Text',
                'Q16_Part_11': 'None',
                'Q16_Part_12': 'Other'
                }

# Make binary columns from IDE answers.
ide_qs_binary = ide_qs.rename(columns=column_rename).fillna(0).replace('[^\\d]',1, regex=True)
mc_and_ide = pd.concat([mc.drop(0), ide_qs_binary], axis=1)
#IDE Survey Results
color_pal = sns.color_palette("hls", 16)
ide_qs_binary_drop_noresponse = ide_qs_binary.copy()
ide_qs_binary_drop_noresponse['no reponse'] = ide_qs_binary_drop_noresponse.sum(axis=1).apply(lambda x: 1 if x == 0 else 0)
ide_qs_binary_drop_noresponse = ide_qs_binary_drop_noresponse.loc[ide_qs_binary_drop_noresponse['no reponse'] == 0].drop('no reponse', axis=1).copy()

plot_df = ((ide_qs_binary_drop_noresponse.sum() / ide_qs_binary_drop_noresponse.count()).sort_values() * 100 ).round(2)
ax = plot_df.plot(kind='barh', figsize=(10, 10),
          title='2019 Kaggle Survey IDE Preference (Excluding Non-Respondents)',
          color=color_pal)
for i, (p, pr) in enumerate(zip(plot_df.index, plot_df.values)):
    #plt.text(s=p, x=1, y=i, color="w", verticalalignment="center", size=18)
    plt.text(s=str(pr)+"%", x=pr-5, y=i, color="w",
             verticalalignment="center", horizontalalignment="left", size=10)
ax.set_xlabel("% of Respondents")
plt.show()
#Combinations of IDEs (Jupyter, Visual Studio, MATLAB)
plt.figure(figsize=(15, 8))

venn3(subsets=(len(ide_qs_binary.loc[(ide_qs_binary['Jupyter (JupyterLab, Jupyter Notebooks, etc)'] == 1) & (ide_qs_binary['Visual Studio / Visual Studio Code'] == 0) & (ide_qs_binary['MATLAB'] == 0)]),
               len(ide_qs_binary.loc[(ide_qs_binary['Jupyter (JupyterLab, Jupyter Notebooks, etc)'] == 0) & (ide_qs_binary['Visual Studio / Visual Studio Code'] == 1) & (ide_qs_binary['MATLAB'] == 0)]),
               len(ide_qs_binary.loc[(ide_qs_binary['Jupyter (JupyterLab, Jupyter Notebooks, etc)'] == 1) & (ide_qs_binary['Visual Studio / Visual Studio Code'] == 1) & (ide_qs_binary['MATLAB'] == 0)]),
               len(ide_qs_binary.loc[(ide_qs_binary['Jupyter (JupyterLab, Jupyter Notebooks, etc)'] == 0) & (ide_qs_binary['Visual Studio / Visual Studio Code'] == 0) & (ide_qs_binary['MATLAB'] == 1)]),
               len(ide_qs_binary.loc[(ide_qs_binary['Jupyter (JupyterLab, Jupyter Notebooks, etc)'] == 1) & (ide_qs_binary['Visual Studio / Visual Studio Code'] == 0) & (ide_qs_binary['MATLAB'] == 1)]),
               len(ide_qs_binary.loc[(ide_qs_binary['Jupyter (JupyterLab, Jupyter Notebooks, etc)'] == 0) & (ide_qs_binary['Visual Studio / Visual Studio Code'] == 1) & (ide_qs_binary['MATLAB'] == 1)]),
               len(ide_qs_binary.loc[(ide_qs_binary['Jupyter (JupyterLab, Jupyter Notebooks, etc)'] == 1) & (ide_qs_binary['Visual Studio / Visual Studio Code'] == 1) & (ide_qs_binary['MATLAB'] == 1)])),
      set_labels=('Jupyter', 'Visual Studio', 'MATLAB'))
plt.title('Jupyter vs Visual Studio vs MATLAB (All users)')
plt.tight_layout()
plt.show()
#Combinations of IDEs (Jupyter, PyCharm, Spyder)
plt.figure(figsize=(15, 8))

venn3(subsets=(len(ide_qs_binary.loc[(ide_qs_binary['Jupyter (JupyterLab, Jupyter Notebooks, etc)'] == 1) & (ide_qs_binary['PyCharm'] == 0) & (ide_qs_binary['Spyder'] == 0)]),
               len(ide_qs_binary.loc[(ide_qs_binary['Jupyter (JupyterLab, Jupyter Notebooks, etc)'] == 0) & (ide_qs_binary['PyCharm'] == 1) & (ide_qs_binary['Spyder'] == 0)]),
               len(ide_qs_binary.loc[(ide_qs_binary['Jupyter (JupyterLab, Jupyter Notebooks, etc)'] == 1) & (ide_qs_binary['PyCharm'] == 1) & (ide_qs_binary['Spyder'] == 0)]),
               len(ide_qs_binary.loc[(ide_qs_binary['Jupyter (JupyterLab, Jupyter Notebooks, etc)'] == 0) & (ide_qs_binary['PyCharm'] == 0) & (ide_qs_binary['Spyder'] == 1)]),
               len(ide_qs_binary.loc[(ide_qs_binary['Jupyter (JupyterLab, Jupyter Notebooks, etc)'] == 1) & (ide_qs_binary['PyCharm'] == 0) & (ide_qs_binary['Spyder'] == 1)]),
               len(ide_qs_binary.loc[(ide_qs_binary['Jupyter (JupyterLab, Jupyter Notebooks, etc)'] == 0) & (ide_qs_binary['PyCharm'] == 1) & (ide_qs_binary['Spyder'] == 1)]),
               len(ide_qs_binary.loc[(ide_qs_binary['Jupyter (JupyterLab, Jupyter Notebooks, etc)'] == 1) & (ide_qs_binary['PyCharm'] == 1) & (ide_qs_binary['Spyder'] == 1)])),
      set_labels=('Jupyter', 'PyCharm', 'Spyder'))
plt.title('Jupyter vs PyCharm vs Spyder (All users)')
plt.tight_layout()
plt.show()
#Combinations of IDEs (Jupyter, RStudio, Notepad++)
plt.figure(figsize=(15, 8))

venn3(subsets=(len(ide_qs_binary.loc[(ide_qs_binary['Jupyter (JupyterLab, Jupyter Notebooks, etc)'] == 1) & (ide_qs_binary['RStudio'] == 0) & (ide_qs_binary['Notepad++'] == 0)]),
               len(ide_qs_binary.loc[(ide_qs_binary['Jupyter (JupyterLab, Jupyter Notebooks, etc)'] == 0) & (ide_qs_binary['RStudio'] == 1) & (ide_qs_binary['Notepad++'] == 0)]),
               len(ide_qs_binary.loc[(ide_qs_binary['Jupyter (JupyterLab, Jupyter Notebooks, etc)'] == 1) & (ide_qs_binary['RStudio'] == 1) & (ide_qs_binary['Notepad++'] == 0)]),
               len(ide_qs_binary.loc[(ide_qs_binary['Jupyter (JupyterLab, Jupyter Notebooks, etc)'] == 0) & (ide_qs_binary['RStudio'] == 0) & (ide_qs_binary['Notepad++'] == 1)]),
               len(ide_qs_binary.loc[(ide_qs_binary['Jupyter (JupyterLab, Jupyter Notebooks, etc)'] == 1) & (ide_qs_binary['RStudio'] == 0) & (ide_qs_binary['Notepad++'] == 1)]),
               len(ide_qs_binary.loc[(ide_qs_binary['Jupyter (JupyterLab, Jupyter Notebooks, etc)'] == 0) & (ide_qs_binary['RStudio'] == 1) & (ide_qs_binary['Notepad++'] == 1)]),
               len(ide_qs_binary.loc[(ide_qs_binary['Jupyter (JupyterLab, Jupyter Notebooks, etc)'] == 1) & (ide_qs_binary['RStudio'] == 1) & (ide_qs_binary['Notepad++'] == 1)])),
      set_labels=('Jupyter', 'RStudio', 'Notepad++'))
plt.title('Jupyter vs RStudio vs Notepad++ (All users)')
plt.tight_layout()
plt.show()
#Colour map of IDEs prefered by different professions
# Pull just question5
ide_by_q5 = mc_and_ide \
    .rename(columns={'Q5':'Job Title'}) \
    .groupby('Job Title')['Jupyter (JupyterLab, Jupyter Notebooks, etc)','RStudio','PyCharm','Atom','MATLAB','Visual Studio / Visual Studio Code', 'Spyder','Vim / Emacs','Notepad++','Sublime Text','None','Other'] \
    .mean()

def magnify():
    return [dict(selector="th",
                 props=[("font-size", "8pt")]),
            dict(selector="td",
                 props=[('padding', "0em 0em")]),
            dict(selector="th:hover",
                 props=[("font-size", "9pt")]),
            dict(selector="tr:hover td:hover",
                 props=[('max-width', '200px'),
                        ('font-size', '9pt')])
]
np.random.seed(25)
cmap = cmap=sns.diverging_palette(5, 250, as_cmap=True)
#bigdf = pd.DataFrame(np.random.randn(20, 25)).cumsum()
ide_by_q5.T.sort_values('Data Analyst', ascending=False).T \
    .sort_values('RStudio', ascending=False) \
    .rename(columns={'Jupyter (JupyterLab, Jupyter Notebooks, etc)': 'Jupyter',
                     'Visual Studio / Visual Studio Code':'VStudio',
                     'Sublime Text': 'Sublime'}) \
    [['Jupyter','RStudio','VStudio','Notepad++','PyCharm','Spyder','Sublime','MATLAB']] \
    .sort_index() \
    .style.background_gradient(cmap, axis=1)\
    .set_precision(2)\
    .format("{:.0%}")
# Venn Diagram of comparative IDE combinations of Jupyter, RStudio, Notepad++ prefered by different professions
plt.figure(figsize=(15, 10))
plt.subplot(2, 2, 1)
venn3(subsets=(len(ide_qs_binary.loc[(ide_qs_binary['Jupyter (JupyterLab, Jupyter Notebooks, etc)'] == 1) & (ide_qs_binary['RStudio'] == 0) & (ide_qs_binary['Notepad++'] == 0 & (mc_and_ide['Q5'] == 'Student'))]),
               len(ide_qs_binary.loc[(ide_qs_binary['Jupyter (JupyterLab, Jupyter Notebooks, etc)'] == 0) & (ide_qs_binary['RStudio'] == 1) & (ide_qs_binary['Notepad++'] == 0)& (mc_and_ide['Q5'] == 'Student')]),
               len(ide_qs_binary.loc[(ide_qs_binary['Jupyter (JupyterLab, Jupyter Notebooks, etc)'] == 1) & (ide_qs_binary['RStudio'] == 1) & (ide_qs_binary['Notepad++'] == 0)& (mc_and_ide['Q5'] == 'Student')]),
               len(ide_qs_binary.loc[(ide_qs_binary['Jupyter (JupyterLab, Jupyter Notebooks, etc)'] == 0) & (ide_qs_binary['RStudio'] == 0) & (ide_qs_binary['Notepad++'] == 1)& (mc_and_ide['Q5'] == 'Student')]),
               len(ide_qs_binary.loc[(ide_qs_binary['Jupyter (JupyterLab, Jupyter Notebooks, etc)'] == 1) & (ide_qs_binary['RStudio'] == 0) & (ide_qs_binary['Notepad++'] == 1)& (mc_and_ide['Q5'] == 'Student')]),
               len(ide_qs_binary.loc[(ide_qs_binary['Jupyter (JupyterLab, Jupyter Notebooks, etc)'] == 0) & (ide_qs_binary['RStudio'] == 1) & (ide_qs_binary['Notepad++'] == 1)& (mc_and_ide['Q5'] == 'Student')]),
               len(ide_qs_binary.loc[(ide_qs_binary['Jupyter (JupyterLab, Jupyter Notebooks, etc)'] == 1) & (ide_qs_binary['RStudio'] == 1) & (ide_qs_binary['Notepad++'] == 1& (mc_and_ide['Q5'] == 'Student'))])),
      set_labels=('Jupyter', 'RStudio', 'Notepad++'))
plt.title('Students IDE Use')
plt.subplot(2, 2, 2)
venn3(subsets=(len(ide_qs_binary.loc[(ide_qs_binary['Jupyter (JupyterLab, Jupyter Notebooks, etc)'] == 1) & (ide_qs_binary['RStudio'] == 0) & (ide_qs_binary['Notepad++'] == 0) & (mc_and_ide['Q5'] == 'Data Scientist')]),
               len(ide_qs_binary.loc[(ide_qs_binary['Jupyter (JupyterLab, Jupyter Notebooks, etc)'] == 0) & (ide_qs_binary['RStudio'] == 1) & (ide_qs_binary['Notepad++'] == 0)& (mc_and_ide['Q5'] == 'Data Scientist')]),
               len(ide_qs_binary.loc[(ide_qs_binary['Jupyter (JupyterLab, Jupyter Notebooks, etc)'] == 1) & (ide_qs_binary['RStudio'] == 1) & (ide_qs_binary['Notepad++'] == 0)& (mc_and_ide['Q5'] == 'Data Scientist')]),
               len(ide_qs_binary.loc[(ide_qs_binary['Jupyter (JupyterLab, Jupyter Notebooks, etc)'] == 0) & (ide_qs_binary['RStudio'] == 0) & (ide_qs_binary['Notepad++'] == 1)& (mc_and_ide['Q5'] == 'Data Scientist')]),
               len(ide_qs_binary.loc[(ide_qs_binary['Jupyter (JupyterLab, Jupyter Notebooks, etc)'] == 1) & (ide_qs_binary['RStudio'] == 0) & (ide_qs_binary['Notepad++'] == 1)& (mc_and_ide['Q5'] == 'Data Scientist')]),
               len(ide_qs_binary.loc[(ide_qs_binary['Jupyter (JupyterLab, Jupyter Notebooks, etc)'] == 0) & (ide_qs_binary['RStudio'] == 1) & (ide_qs_binary['Notepad++'] == 1)& (mc_and_ide['Q5'] == 'Data Scientist')]),
               len(ide_qs_binary.loc[(ide_qs_binary['Jupyter (JupyterLab, Jupyter Notebooks, etc)'] == 1) & (ide_qs_binary['RStudio'] == 1) & (ide_qs_binary['Notepad++'] == 1& (mc_and_ide['Q5'] == 'Data Scientist'))])),
      set_labels=('Jupyter', 'RStudio', 'Notepad++'))
plt.title('Data Scientists IDE Use')
plt.subplot(2, 2, 3)
venn3(subsets=(len(ide_qs_binary.loc[(ide_qs_binary['Jupyter (JupyterLab, Jupyter Notebooks, etc)'] == 1) & (ide_qs_binary['RStudio'] == 0) & (ide_qs_binary['Notepad++'] == 0 & (mc_and_ide['Q5'] == 'Software Engineer'))]),
               len(ide_qs_binary.loc[(ide_qs_binary['Jupyter (JupyterLab, Jupyter Notebooks, etc)'] == 0) & (ide_qs_binary['RStudio'] == 1) & (ide_qs_binary['Notepad++'] == 0)& (mc_and_ide['Q5'] == 'Software Engineer')]),
               len(ide_qs_binary.loc[(ide_qs_binary['Jupyter (JupyterLab, Jupyter Notebooks, etc)'] == 1) & (ide_qs_binary['RStudio'] == 1) & (ide_qs_binary['Notepad++'] == 0)& (mc_and_ide['Q5'] == 'Software Engineer')]),
               len(ide_qs_binary.loc[(ide_qs_binary['Jupyter (JupyterLab, Jupyter Notebooks, etc)'] == 0) & (ide_qs_binary['RStudio'] == 0) & (ide_qs_binary['Notepad++'] == 1)& (mc_and_ide['Q5'] == 'Software Engineer')]),
               len(ide_qs_binary.loc[(ide_qs_binary['Jupyter (JupyterLab, Jupyter Notebooks, etc)'] == 1) & (ide_qs_binary['RStudio'] == 0) & (ide_qs_binary['Notepad++'] == 1)& (mc_and_ide['Q5'] == 'Software Engineer')]),
               len(ide_qs_binary.loc[(ide_qs_binary['Jupyter (JupyterLab, Jupyter Notebooks, etc)'] == 0) & (ide_qs_binary['RStudio'] == 1) & (ide_qs_binary['Notepad++'] == 1)& (mc_and_ide['Q5'] == 'Software Engineer')]),
               len(ide_qs_binary.loc[(ide_qs_binary['Jupyter (JupyterLab, Jupyter Notebooks, etc)'] == 1) & (ide_qs_binary['RStudio'] == 1) & (ide_qs_binary['Notepad++'] == 1& (mc_and_ide['Q5'] == 'Software Engineer'))])),
      set_labels=('Jupyter', 'RStudio', 'Notepad++'))
plt.title('Software Engineer IDE Use')
plt.subplot(2, 2, 4)
venn3(subsets=(len(ide_qs_binary.loc[(ide_qs_binary['Jupyter (JupyterLab, Jupyter Notebooks, etc)'] == 1) & (ide_qs_binary['RStudio'] == 0) & (ide_qs_binary['Notepad++'] == 0) & (mc_and_ide['Q5'] == 'Statistician')]),
               len(ide_qs_binary.loc[(ide_qs_binary['Jupyter (JupyterLab, Jupyter Notebooks, etc)'] == 0) & (ide_qs_binary['RStudio'] == 1) & (ide_qs_binary['Notepad++'] == 0)& (mc_and_ide['Q5'] == 'Statistician')]),
               len(ide_qs_binary.loc[(ide_qs_binary['Jupyter (JupyterLab, Jupyter Notebooks, etc)'] == 1) & (ide_qs_binary['RStudio'] == 1) & (ide_qs_binary['Notepad++'] == 0)& (mc_and_ide['Q5'] == 'Statistician')]),
               len(ide_qs_binary.loc[(ide_qs_binary['Jupyter (JupyterLab, Jupyter Notebooks, etc)'] == 0) & (ide_qs_binary['RStudio'] == 0) & (ide_qs_binary['Notepad++'] == 1)& (mc_and_ide['Q5'] == 'Statistician')]),
               len(ide_qs_binary.loc[(ide_qs_binary['Jupyter (JupyterLab, Jupyter Notebooks, etc)'] == 1) & (ide_qs_binary['RStudio'] == 0) & (ide_qs_binary['Notepad++'] == 1)& (mc_and_ide['Q5'] == 'Statistician')]),
               len(ide_qs_binary.loc[(ide_qs_binary['Jupyter (JupyterLab, Jupyter Notebooks, etc)'] == 0) & (ide_qs_binary['RStudio'] == 1) & (ide_qs_binary['Notepad++'] == 1)& (mc_and_ide['Q5'] == 'Statistician')]),
               len(ide_qs_binary.loc[(ide_qs_binary['Jupyter (JupyterLab, Jupyter Notebooks, etc)'] == 1) & (ide_qs_binary['RStudio'] == 1) & (ide_qs_binary['Notepad++'] == 1& (mc_and_ide['Q5'] == 'Statistician'))])),
      set_labels=('Jupyter', 'RStudio', 'Notepad++'))
plt.title('Statistician IDE Use')
plt.tight_layout()
plt.show()
# Venn Diagram of comparative IDE combinations of Jupyter, RStudio, MATLAB prefered by different professions
plt.figure(figsize=(15, 10))
plt.subplot(2, 2, 1)
venn3(subsets=(len(ide_qs_binary.loc[(ide_qs_binary['Jupyter (JupyterLab, Jupyter Notebooks, etc)'] == 1) & (ide_qs_binary['RStudio'] == 0) & (ide_qs_binary['MATLAB'] == 0 & (mc_and_ide['Q5'] == 'Student'))]),
               len(ide_qs_binary.loc[(ide_qs_binary['Jupyter (JupyterLab, Jupyter Notebooks, etc)'] == 0) & (ide_qs_binary['RStudio'] == 1) & (ide_qs_binary['MATLAB'] == 0)& (mc_and_ide['Q5'] == 'Student')]),
               len(ide_qs_binary.loc[(ide_qs_binary['Jupyter (JupyterLab, Jupyter Notebooks, etc)'] == 1) & (ide_qs_binary['RStudio'] == 1) & (ide_qs_binary['MATLAB'] == 0)& (mc_and_ide['Q5'] == 'Student')]),
               len(ide_qs_binary.loc[(ide_qs_binary['Jupyter (JupyterLab, Jupyter Notebooks, etc)'] == 0) & (ide_qs_binary['RStudio'] == 0) & (ide_qs_binary['MATLAB'] == 1)& (mc_and_ide['Q5'] == 'Student')]),
               len(ide_qs_binary.loc[(ide_qs_binary['Jupyter (JupyterLab, Jupyter Notebooks, etc)'] == 1) & (ide_qs_binary['RStudio'] == 0) & (ide_qs_binary['MATLAB'] == 1)& (mc_and_ide['Q5'] == 'Student')]),
               len(ide_qs_binary.loc[(ide_qs_binary['Jupyter (JupyterLab, Jupyter Notebooks, etc)'] == 0) & (ide_qs_binary['RStudio'] == 1) & (ide_qs_binary['MATLAB'] == 1)& (mc_and_ide['Q5'] == 'Student')]),
               len(ide_qs_binary.loc[(ide_qs_binary['Jupyter (JupyterLab, Jupyter Notebooks, etc)'] == 1) & (ide_qs_binary['RStudio'] == 1) & (ide_qs_binary['MATLAB'] == 1& (mc_and_ide['Q5'] == 'Student'))])),
      set_labels=('Jupyter', 'RStudio', 'MATLAB'))
plt.title('Students IDE Use')
plt.subplot(2, 2, 2)
venn3(subsets=(len(ide_qs_binary.loc[(ide_qs_binary['Jupyter (JupyterLab, Jupyter Notebooks, etc)'] == 1) & (ide_qs_binary['RStudio'] == 0) & (ide_qs_binary['MATLAB'] == 0) & (mc_and_ide['Q5'] == 'Data Scientist')]),
               len(ide_qs_binary.loc[(ide_qs_binary['Jupyter (JupyterLab, Jupyter Notebooks, etc)'] == 0) & (ide_qs_binary['RStudio'] == 1) & (ide_qs_binary['MATLAB'] == 0)& (mc_and_ide['Q5'] == 'Data Scientist')]),
               len(ide_qs_binary.loc[(ide_qs_binary['Jupyter (JupyterLab, Jupyter Notebooks, etc)'] == 1) & (ide_qs_binary['RStudio'] == 1) & (ide_qs_binary['MATLAB'] == 0)& (mc_and_ide['Q5'] == 'Data Scientist')]),
               len(ide_qs_binary.loc[(ide_qs_binary['Jupyter (JupyterLab, Jupyter Notebooks, etc)'] == 0) & (ide_qs_binary['RStudio'] == 0) & (ide_qs_binary['MATLAB'] == 1)& (mc_and_ide['Q5'] == 'Data Scientist')]),
               len(ide_qs_binary.loc[(ide_qs_binary['Jupyter (JupyterLab, Jupyter Notebooks, etc)'] == 1) & (ide_qs_binary['RStudio'] == 0) & (ide_qs_binary['MATLAB'] == 1)& (mc_and_ide['Q5'] == 'Data Scientist')]),
               len(ide_qs_binary.loc[(ide_qs_binary['Jupyter (JupyterLab, Jupyter Notebooks, etc)'] == 0) & (ide_qs_binary['RStudio'] == 1) & (ide_qs_binary['MATLAB'] == 1)& (mc_and_ide['Q5'] == 'Data Scientist')]),
               len(ide_qs_binary.loc[(ide_qs_binary['Jupyter (JupyterLab, Jupyter Notebooks, etc)'] == 1) & (ide_qs_binary['RStudio'] == 1) & (ide_qs_binary['MATLAB'] == 1& (mc_and_ide['Q5'] == 'Data Scientist'))])),
      set_labels=('Jupyter', 'RStudio', 'MATLAB'))
plt.title('Data Scientists IDE Use')
plt.subplot(2, 2, 3)
venn3(subsets=(len(ide_qs_binary.loc[(ide_qs_binary['Jupyter (JupyterLab, Jupyter Notebooks, etc)'] == 1) & (ide_qs_binary['RStudio'] == 0) & (ide_qs_binary['MATLAB'] == 0 & (mc_and_ide['Q5'] == 'Software Engineer'))]),
               len(ide_qs_binary.loc[(ide_qs_binary['Jupyter (JupyterLab, Jupyter Notebooks, etc)'] == 0) & (ide_qs_binary['RStudio'] == 1) & (ide_qs_binary['MATLAB'] == 0)& (mc_and_ide['Q5'] == 'Software Engineer')]),
               len(ide_qs_binary.loc[(ide_qs_binary['Jupyter (JupyterLab, Jupyter Notebooks, etc)'] == 1) & (ide_qs_binary['RStudio'] == 1) & (ide_qs_binary['MATLAB'] == 0)& (mc_and_ide['Q5'] == 'Software Engineer')]),
               len(ide_qs_binary.loc[(ide_qs_binary['Jupyter (JupyterLab, Jupyter Notebooks, etc)'] == 0) & (ide_qs_binary['RStudio'] == 0) & (ide_qs_binary['MATLAB'] == 1)& (mc_and_ide['Q5'] == 'Software Engineer')]),
               len(ide_qs_binary.loc[(ide_qs_binary['Jupyter (JupyterLab, Jupyter Notebooks, etc)'] == 1) & (ide_qs_binary['RStudio'] == 0) & (ide_qs_binary['MATLAB'] == 1)& (mc_and_ide['Q5'] == 'Software Engineer')]),
               len(ide_qs_binary.loc[(ide_qs_binary['Jupyter (JupyterLab, Jupyter Notebooks, etc)'] == 0) & (ide_qs_binary['RStudio'] == 1) & (ide_qs_binary['MATLAB'] == 1)& (mc_and_ide['Q5'] == 'Software Engineer')]),
               len(ide_qs_binary.loc[(ide_qs_binary['Jupyter (JupyterLab, Jupyter Notebooks, etc)'] == 1) & (ide_qs_binary['RStudio'] == 1) & (ide_qs_binary['MATLAB'] == 1& (mc_and_ide['Q5'] == 'Software Engineer'))])),
      set_labels=('Jupyter', 'RStudio', 'MATLAB'))
plt.title('Software Engineer IDE Use')
plt.subplot(2, 2, 4)
venn3(subsets=(len(ide_qs_binary.loc[(ide_qs_binary['Jupyter (JupyterLab, Jupyter Notebooks, etc)'] == 1) & (ide_qs_binary['RStudio'] == 0) & (ide_qs_binary['MATLAB'] == 0) & (mc_and_ide['Q5'] == 'Statistician')]),
               len(ide_qs_binary.loc[(ide_qs_binary['Jupyter (JupyterLab, Jupyter Notebooks, etc)'] == 0) & (ide_qs_binary['RStudio'] == 1) & (ide_qs_binary['MATLAB'] == 0)& (mc_and_ide['Q5'] == 'Statistician')]),
               len(ide_qs_binary.loc[(ide_qs_binary['Jupyter (JupyterLab, Jupyter Notebooks, etc)'] == 1) & (ide_qs_binary['RStudio'] == 1) & (ide_qs_binary['MATLAB'] == 0)& (mc_and_ide['Q5'] == 'Statistician')]),
               len(ide_qs_binary.loc[(ide_qs_binary['Jupyter (JupyterLab, Jupyter Notebooks, etc)'] == 0) & (ide_qs_binary['RStudio'] == 0) & (ide_qs_binary['MATLAB'] == 1)& (mc_and_ide['Q5'] == 'Statistician')]),
               len(ide_qs_binary.loc[(ide_qs_binary['Jupyter (JupyterLab, Jupyter Notebooks, etc)'] == 1) & (ide_qs_binary['RStudio'] == 0) & (ide_qs_binary['MATLAB'] == 1)& (mc_and_ide['Q5'] == 'Statistician')]),
               len(ide_qs_binary.loc[(ide_qs_binary['Jupyter (JupyterLab, Jupyter Notebooks, etc)'] == 0) & (ide_qs_binary['RStudio'] == 1) & (ide_qs_binary['MATLAB'] == 1)& (mc_and_ide['Q5'] == 'Statistician')]),
               len(ide_qs_binary.loc[(ide_qs_binary['Jupyter (JupyterLab, Jupyter Notebooks, etc)'] == 1) & (ide_qs_binary['RStudio'] == 1) & (ide_qs_binary['MATLAB'] == 1& (mc_and_ide['Q5'] == 'Statistician'))])),
      set_labels=('Jupyter', 'RStudio', 'MATLAB'))
plt.title('Statistician IDE Use')
plt.tight_layout()
plt.show()
# Clusters
# Create IDE binary dataset, dropping no responses
ide_qs_binary = ide_qs.rename(columns=column_rename).fillna(0).replace('[^\\d]',1, regex=True)
ide_qs_binary['no reponse'] = ide_qs_binary.sum(axis=1).apply(lambda x: 1 if x == 0 else 0)
ide_qs_binary = ide_qs_binary.loc[ide_qs_binary['no reponse'] == 0].drop('no reponse', axis=1).copy()

# Make the clusters using sklean's KMeans
from sklearn.cluster import KMeans
y_pred = KMeans(n_clusters=4, random_state=1).fit_predict(ide_qs_binary)
ide_qs_binary['cluster'] = y_pred

# Name the clusters
y_pred_named = ['Cluster1' if x == 0 else \
                'Cluster2' if x == 1 else \
                'Cluster3' if x == 2 else \
                'Cluster4' for x in y_pred]

ide_qs_binary['cluster_name'] = y_pred_named

cluster1 = ide_qs_binary.loc[ide_qs_binary['cluster'] == 0]
cluster2 = ide_qs_binary.loc[ide_qs_binary['cluster'] == 1]
cluster3 = ide_qs_binary.loc[ide_qs_binary['cluster'] == 2]
cluster4 = ide_qs_binary.loc[ide_qs_binary['cluster'] == 3]

ide_qs_binary = ide_qs_binary.replace({ide_qs_binary.groupby('cluster_name').sum().sort_values('Jupyter (JupyterLab, Jupyter Notebooks, etc)', ascending=False).iloc[0].name: 'Pro Jupyter',
                     ide_qs_binary.groupby('cluster_name').sum().sort_values('Jupyter (JupyterLab, Jupyter Notebooks, etc)', ascending=True).iloc[0].name: 'Non-Jupyters',
                     ide_qs_binary.groupby('cluster_name').sum().sort_values('RStudio', ascending=False).iloc[0].name: 'RStudio and Jupyter',
                     ide_qs_binary.groupby('cluster_name').sum().sort_values('PyCharm', ascending=False).iloc[0].name: 'All IDEs'}).copy()

mc_and_ide['cluster_name'] = ide_qs_binary['cluster_name']
mc_and_ide['cluster_name'] = mc_and_ide['cluster_name'].fillna('No Response')
mc_and_ide['count'] = 1
# "Non-Jupyters" refers to people who use IDEs except Jupyter
# "Everyone Else" refers to people who use different combination of IDEs including Jupyter, but excluding people who are "Non-Jupyters"
anti_jupyters = mc_and_ide.loc[mc_and_ide['cluster_name'] == 'Non-Jupyters']
everyone_else = mc_and_ide.loc[mc_and_ide['cluster_name'] != 'Non-Jupyters']
df = pd.DataFrame([everyone_else.groupby('Q23').count()['count'] / everyone_else.groupby('Q23').count().dropna()['count'].sum() * 100,
              anti_jupyters.groupby('Q23').count()['count'] / anti_jupyters.groupby('Q23').count().dropna()['count'].sum() * 100]).T
df.columns = ['Everyone Else','Non-Jupyters']

# Order the columns
df.index = pd.Categorical(df.index, ['I have never studied machine learning but plan to learn in the future',
                                     '< 1 year',
                                     '1-2 years',
                                     '2-3 years',
                                     '3-4 years',
                                     '4-5 years',
                                     '5-10 years',
                                     '10-15 years',
                                     '20+ years',
                                     'I have never studied machine learning and I do not plan to'])
df = df.sort_index(ascending=False)
df = df.rename({'I have never studied machine learning but plan to learn in the future' : 'Never but plan to',
           'I have never studied machine learning and I do not plan to': 'Dont plan to'})
plt.subplot(1, 2, 1)
plt.ylim(0,50)
plt.ylabel('% of Group')
ax = df['Non-Jupyters'].plot(kind='bar',
                         color=color_pal[0],
                         figsize=(15, 3),
                         title='Non-Jupyters')
for p in ax.patches:
    ax.annotate(str(p.get_height().round(1)) + '%', (p.get_x() * 1, p.get_height() + 1.05))
plt.subplot(1, 2, 2)
plt.ylim(0,50)
ax2 = df['Everyone Else'].plot(kind='bar',
                         color=color_pal[1],
                         figsize=(15, 3),
                         title='Everyone Else')
for p in ax2.patches:
    ax2.annotate(str(p.get_height().round(1)) + '%', (p.get_x() * 1, p.get_height() + 1.05))
plt.ylabel('% of Group')
plt.suptitle('How many years have you used machine learning methods?', fontsize=15, y=1.05)
plt.show()
# People using IDEs excluding all options in the survey


# Format into lower strings
text['count'] = 1
text['IDE_lower'] = text['Q16_OTHER_TEXT'].str.lower()
text.drop(0)[['IDE_lower','count']].groupby('IDE_lower').sum()[['count']].sort_values('count', ascending=False)

# Create wordcloud
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
plt.figure(figsize=[15,8])

# Create and generate a word cloud image:
ide_words = ' '.join(text['IDE_lower'].drop(0).dropna().values)
wordcloud = WordCloud(colormap="tab10",
                      width=1200,
                      height=480,
                      normalize_plurals=False,
                      background_color="white",
                      random_state=5).generate(ide_words)

# Display the generated image:
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()
#ML Frameworks
#Data Prep

# Pull just ML framework Questions
framework_qs = mc[['Q28_Part_1','Q28_Part_2','Q28_Part_3','Q28_Part_4','Q28_Part_5',
             'Q28_Part_6','Q28_Part_7','Q28_Part_8','Q28_Part_9','Q28_Part_10',
             'Q28_Part_11','Q28_Part_12']].drop(0)


# Rename Columns for ML framework Type
framework_column_rename = {'Q28_Part_1': 'Scikit-learn',
                 'Q28_Part_2': 'TensorFlow',
                'Q28_Part_3': 'Keras',
                'Q28_Part_4': 'RandomForest',
                'Q28_Part_5': 'Xgboost',
                'Q28_Part_6': 'PyTorch',
                'Q28_Part_7': 'Caret',
                'Q28_Part_8': 'LightGBM',
                'Q28_Part_9': 'Spark MLib',
                'Q28_Part_10': 'Fast.ai',
                'Q28_Part_11': 'None',
                'Q28_Part_12': 'Other'
                }

# Make binary columns from ML framework answers.
framework_qs_binary = framework_qs.rename(columns=framework_column_rename).fillna(0).replace('[^\\d]',1, regex=True)
mc_and_framework = pd.concat([mc.drop(0), ide_qs_binary], axis=1)
#ML Frameworks Survey Results
color_pal = sns.color_palette("hls", 16)
framework_qs_binary_drop_noresponse = framework_qs_binary.copy()
framework_qs_binary_drop_noresponse['no reponse'] = framework_qs_binary_drop_noresponse.sum(axis=1).apply(lambda x: 1 if x == 0 else 0)
framework_qs_binary_drop_noresponse = framework_qs_binary_drop_noresponse.loc[framework_qs_binary_drop_noresponse['no reponse'] == 0].drop('no reponse', axis=1).copy()

plot_df = ((framework_qs_binary_drop_noresponse.sum() / framework_qs_binary_drop_noresponse.count()).sort_values() * 100 ).round(2)
ax = plot_df.plot(kind='barh', figsize=(10, 10),
          title='2019 Kaggle Survey ML Framework Preference (Excluding Non-Respondents)',
          color=color_pal)
for i, (p, pr) in enumerate(zip(plot_df.index, plot_df.values)):
    #plt.text(s=p, x=1, y=i, color="w", verticalalignment="center", size=18)
    plt.text(s=str(pr)+"%", x=pr-5, y=i, color="w",
             verticalalignment="center", horizontalalignment="left", size=10)
ax.set_xlabel("% of Respondents")
plt.show()
#Combinations of ML Frameworks
plt.figure(figsize=(15, 8))

venn3(subsets=(len(framework_qs_binary.loc[(framework_qs_binary['Scikit-learn'] == 1) & (framework_qs_binary['TensorFlow'] == 0) & (framework_qs_binary['Keras'] == 0)]),
               len(framework_qs_binary.loc[(framework_qs_binary['Scikit-learn'] == 0) & (framework_qs_binary['TensorFlow'] == 1) & (framework_qs_binary['Keras'] == 0)]),
               len(framework_qs_binary.loc[(framework_qs_binary['Scikit-learn'] == 1) & (framework_qs_binary['TensorFlow'] == 1) & (framework_qs_binary['Keras'] == 0)]),
               len(framework_qs_binary.loc[(framework_qs_binary['Scikit-learn'] == 0) & (framework_qs_binary['TensorFlow'] == 0) & (framework_qs_binary['Keras'] == 1)]),
               len(framework_qs_binary.loc[(framework_qs_binary['Scikit-learn'] == 1) & (framework_qs_binary['TensorFlow'] == 0) & (framework_qs_binary['Keras'] == 1)]),
               len(framework_qs_binary.loc[(framework_qs_binary['Scikit-learn'] == 0) & (framework_qs_binary['TensorFlow'] == 1) & (framework_qs_binary['Keras'] == 1)]),
               len(framework_qs_binary.loc[(framework_qs_binary['Scikit-learn'] == 1) & (framework_qs_binary['TensorFlow'] == 1) & (framework_qs_binary['Keras'] == 1)])),
      set_labels=('Scikit-learn', 'TensorFlow', 'Keras'))
plt.title('Scikit-learn vs TensorFlow vs Keras (All users)')
plt.tight_layout()
plt.show()
from venn import venn, pseudovenn
venn_dict_framework4 = {'Scikit-learn': set(framework_qs_binary.index[framework_qs_binary['Scikit-learn']==True]),
       'Tensorflow': set(framework_qs_binary.index[framework_qs_binary['TensorFlow']==True]),
       'Keras': set(framework_qs_binary.index[framework_qs_binary['Keras']==True]),
       'PyTorch': set(framework_qs_binary.index[framework_qs_binary['PyTorch']==True]),}

venn(venn_dict_framework4,fontsize=11, legend_loc="upper left")
venn_dict_framework5 = {'Scikit-learn': set(framework_qs_binary.index[framework_qs_binary['Scikit-learn']==True]),
       'Tensorflow': set(framework_qs_binary.index[framework_qs_binary['TensorFlow']==True]),
       'Keras': set(framework_qs_binary.index[framework_qs_binary['Keras']==True]),
       'PyTorch': set(framework_qs_binary.index[framework_qs_binary['PyTorch']==True]),
       'RandomForest': set(framework_qs_binary.index[framework_qs_binary['RandomForest']==True])     }

venn(venn_dict_framework5,fontsize=11, legend_loc="upper left")

venn_dict_framework6 = {'Scikit-learn': set(framework_qs_binary.index[framework_qs_binary['Scikit-learn']==True]),
       'Tensorflow': set(framework_qs_binary.index[framework_qs_binary['TensorFlow']==True]),
       'Keras': set(framework_qs_binary.index[framework_qs_binary['Keras']==True]),
       'PyTorch': set(framework_qs_binary.index[framework_qs_binary['PyTorch']==True]),
       'RandomForest': set(framework_qs_binary.index[framework_qs_binary['RandomForest']==True]),
       'Xgboost': set(framework_qs_binary.index[framework_qs_binary['Xgboost']==True])                }

venn(venn_dict_framework6,cmap="plasma",fontsize=11, legend_loc="upper left")
pseudovenn(venn_dict_framework6,cmap="plasma",fontsize=11, legend_loc="upper left")
#plt.legend(bbox_to_anchor=(1.0, 0.5))
#Hosted notebooks
color_pal = sns.color_palette("Set1", 11)

notebook_cols = []
for x in mc.columns:
    if x[:3] == 'Q17':
        notebook_cols.append(x)

notebook_qs = mc[notebook_cols]
colname_replace = {}
for x in notebook_qs.columns:
    col_newname = notebook_qs[x][0].replace('Which of the following hosted notebook products do you use on a regular basis?  (Select all that apply) - Selected Choice -','')
    colname_replace[x] = col_newname
colname_replace['Q17_OTHER_TEXT'] = 'Text'
notebook_qs = notebook_qs.rename(columns=colname_replace).drop(0).fillna(0).replace('[^\\d]',1, regex=True)

plot_df = notebook_qs.mean().sort_values().copy() * 100
plot_df = plot_df.round(2)
plot_df.plot.barh(title = 'Which of the following hosted notebook products do you use on a regular basis?',
                                          figsize=(10, 8),
                 color=color_pal)

for i, (p, pr) in enumerate(zip(plot_df.index, plot_df.values)):
    if pr > 2:
        plt.text(s=str(pr)+"%", x=pr-0.3, y=i, color="w",
                 verticalalignment="center", horizontalalignment="right", size=12)
ax.set_xlabel("% of Respondents")
plt.tight_layout()
plt.show()
# Huge percentage of responses under "None"
# Hosted notebooks excluding above options

text['count'] = 1
text['NB_lower'] = text['Q17_OTHER_TEXT'].str.lower()
text.drop(0)[['NB_lower','count']].groupby('NB_lower').sum()[['count']].sort_values('count', ascending=False)

# Create wordcloud
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
plt.figure(figsize=[15,8])

# Create and generate a word cloud image:
nb_words = ' '.join(text['NB_lower'].drop(0).dropna().values)
wordcloud_nb = WordCloud(colormap="tab10",
                      width=1200,
                      height=480,
                      normalize_plurals=False,
                      background_color="white",
                      random_state=5).generate(nb_words)

# Display the generated image:
plt.imshow(wordcloud_nb, interpolation='bilinear')
plt.axis("off")
plt.show()
#clustering data
ML_DF = pd.read_csv('../input/kaggle-survey-2019/multiple_choice_responses.csv',sep=',', low_memory=False,header=[0,1])
questions = pd.DataFrame(list(zip(ML_DF.columns.get_level_values(0),ML_DF.columns.get_level_values(1))))
ML_DF.columns = ML_DF.columns.droplevel(1)
DF_data = ML_DF[['Time from Start to Finish (seconds)','Q1', 'Q2', 'Q2_OTHER_TEXT', 'Q3','Q4','Q5','Q5_OTHER_TEXT','Q6','Q7','Q8','Q9_Part_1','Q9_Part_2','Q9_Part_3','Q9_Part_4','Q9_Part_5','Q9_Part_6','Q9_Part_7','Q9_Part_8','Q9_OTHER_TEXT','Q10','Q11','Q12_Part_1','Q12_Part_2','Q12_Part_3','Q12_Part_4','Q12_Part_5','Q12_Part_6','Q12_Part_7','Q12_Part_8','Q12_Part_9','Q12_Part_10','Q12_Part_11','Q12_Part_12','Q12_OTHER_TEXT','Q13_Part_1','Q13_Part_2','Q13_Part_3','Q13_Part_4','Q13_Part_5','Q13_Part_6','Q13_Part_7','Q13_Part_8','Q13_Part_9','Q13_Part_10','Q13_Part_11','Q13_Part_12','Q13_OTHER_TEXT','Q14','Q14_Part_1_TEXT','Q14_Part_2_TEXT','Q14_Part_3_TEXT','Q14_Part_4_TEXT','Q14_Part_5_TEXT','Q14_OTHER_TEXT','Q15','Q16_Part_1','Q16_Part_2','Q16_Part_3','Q16_Part_4','Q16_Part_5','Q16_Part_6','Q16_Part_7','Q16_Part_8','Q16_Part_9','Q16_Part_10','Q16_Part_11','Q16_Part_12','Q16_OTHER_TEXT','Q17_Part_1','Q17_Part_2','Q17_Part_3','Q17_Part_4','Q17_Part_5','Q17_Part_6','Q17_Part_7','Q17_Part_8','Q17_Part_9','Q17_Part_10','Q17_Part_11','Q17_Part_12','Q17_OTHER_TEXT','Q18_Part_1','Q18_Part_2','Q18_Part_3','Q18_Part_4','Q18_Part_5','Q18_Part_6','Q18_Part_7','Q18_Part_8','Q18_Part_9','Q18_Part_10','Q18_Part_11','Q18_Part_12','Q18_OTHER_TEXT','Q19','Q19_OTHER_TEXT','Q20_Part_1','Q20_Part_2','Q20_Part_3','Q20_Part_4','Q20_Part_5','Q20_Part_6','Q20_Part_7','Q20_Part_8','Q20_Part_9','Q20_Part_10','Q20_Part_11','Q20_Part_12','Q20_OTHER_TEXT','Q21_Part_1','Q21_Part_2','Q21_Part_3','Q21_Part_4','Q21_Part_5','Q21_OTHER_TEXT','Q22','Q23','Q24_Part_1','Q24_Part_2','Q24_Part_3','Q24_Part_4','Q24_Part_5','Q24_Part_6','Q24_Part_7','Q24_Part_8','Q24_Part_9','Q24_Part_10','Q24_Part_11','Q24_Part_12','Q24_OTHER_TEXT','Q25_Part_1','Q25_Part_2','Q25_Part_3','Q25_Part_4','Q25_Part_5','Q25_Part_6','Q25_Part_7','Q25_Part_8','Q25_OTHER_TEXT','Q26_Part_1','Q26_Part_2','Q26_Part_3','Q26_Part_4','Q26_Part_5','Q26_Part_6','Q26_Part_7','Q26_OTHER_TEXT','Q27_Part_1','Q27_Part_2','Q27_Part_3','Q27_Part_4','Q27_Part_5','Q27_Part_6','Q27_OTHER_TEXT','Q28_Part_1','Q28_Part_2','Q28_Part_3','Q28_Part_4','Q28_Part_5','Q28_Part_6','Q28_Part_7','Q28_Part_8','Q28_Part_9','Q28_Part_10','Q28_Part_11','Q28_Part_12','Q28_OTHER_TEXT','Q29_Part_1','Q29_Part_2','Q29_Part_3','Q29_Part_4','Q29_Part_5','Q29_Part_6','Q29_Part_7','Q29_Part_8','Q29_Part_9','Q29_Part_10','Q29_Part_11','Q29_Part_12','Q29_OTHER_TEXT','Q30_Part_1','Q30_Part_2','Q30_Part_3','Q30_Part_4','Q30_Part_5','Q30_Part_6','Q30_Part_7','Q30_Part_8','Q30_Part_9','Q30_Part_10','Q30_Part_11','Q30_Part_12','Q30_OTHER_TEXT','Q31_Part_1','Q31_Part_2','Q31_Part_3','Q31_Part_4','Q31_Part_5','Q31_Part_6','Q31_Part_7','Q31_Part_8','Q31_Part_9','Q31_Part_10','Q31_Part_11','Q31_Part_12','Q31_OTHER_TEXT','Q32_Part_1','Q32_Part_2','Q32_Part_3','Q32_Part_4','Q32_Part_5','Q32_Part_6','Q32_Part_7','Q32_Part_8','Q32_Part_9','Q32_Part_10','Q32_Part_11','Q32_Part_12','Q32_OTHER_TEXT','Q33_Part_1','Q33_Part_2','Q33_Part_3','Q33_Part_4','Q33_Part_5','Q33_Part_6','Q33_Part_7','Q33_Part_8','Q33_Part_9','Q33_Part_10','Q33_Part_11','Q33_Part_12','Q33_OTHER_TEXT','Q34_Part_1','Q34_Part_2','Q34_Part_3','Q34_Part_4','Q34_Part_5','Q34_Part_6','Q34_Part_7','Q34_Part_8','Q34_Part_9','Q34_Part_10','Q34_Part_11','Q34_Part_12','Q34_OTHER_TEXT']]
#Replacing the blank columns with  value as 'space'
columns=['Time from Start to Finish (seconds)','Q1', 'Q2', 'Q2_OTHER_TEXT', 'Q3','Q4','Q5','Q5_OTHER_TEXT','Q6','Q7','Q8','Q9_Part_1','Q9_Part_2','Q9_Part_3','Q9_Part_4','Q9_Part_5','Q9_Part_6','Q9_Part_7','Q9_Part_8','Q9_OTHER_TEXT','Q10','Q11','Q12_Part_1','Q12_Part_2','Q12_Part_3','Q12_Part_4','Q12_Part_5','Q12_Part_6','Q12_Part_7','Q12_Part_8','Q12_Part_9','Q12_Part_10','Q12_Part_11','Q12_Part_12','Q12_OTHER_TEXT','Q13_Part_1','Q13_Part_2','Q13_Part_3','Q13_Part_4','Q13_Part_5','Q13_Part_6','Q13_Part_7','Q13_Part_8','Q13_Part_9','Q13_Part_10','Q13_Part_11','Q13_Part_12','Q13_OTHER_TEXT','Q14','Q14_Part_1_TEXT','Q14_Part_2_TEXT','Q14_Part_3_TEXT','Q14_Part_4_TEXT','Q14_Part_5_TEXT','Q14_OTHER_TEXT','Q15','Q16_Part_1','Q16_Part_2','Q16_Part_3','Q16_Part_4','Q16_Part_5','Q16_Part_6','Q16_Part_7','Q16_Part_8','Q16_Part_9','Q16_Part_10','Q16_Part_11','Q16_Part_12','Q16_OTHER_TEXT','Q17_Part_1','Q17_Part_2','Q17_Part_3','Q17_Part_4','Q17_Part_5','Q17_Part_6','Q17_Part_7','Q17_Part_8','Q17_Part_9','Q17_Part_10','Q17_Part_11','Q17_Part_12','Q17_OTHER_TEXT','Q18_Part_1','Q18_Part_2','Q18_Part_3','Q18_Part_4','Q18_Part_5','Q18_Part_6','Q18_Part_7','Q18_Part_8','Q18_Part_9','Q18_Part_10','Q18_Part_11','Q18_Part_12','Q18_OTHER_TEXT','Q19','Q19_OTHER_TEXT','Q20_Part_1','Q20_Part_2','Q20_Part_3','Q20_Part_4','Q20_Part_5','Q20_Part_6','Q20_Part_7','Q20_Part_8','Q20_Part_9','Q20_Part_10','Q20_Part_11','Q20_Part_12','Q20_OTHER_TEXT','Q21_Part_1','Q21_Part_2','Q21_Part_3','Q21_Part_4','Q21_Part_5','Q21_OTHER_TEXT','Q22','Q23','Q24_Part_1','Q24_Part_2','Q24_Part_3','Q24_Part_4','Q24_Part_5','Q24_Part_6','Q24_Part_7','Q24_Part_8','Q24_Part_9','Q24_Part_10','Q24_Part_11','Q24_Part_12','Q24_OTHER_TEXT','Q25_Part_1','Q25_Part_2','Q25_Part_3','Q25_Part_4','Q25_Part_5','Q25_Part_6','Q25_Part_7','Q25_Part_8','Q25_OTHER_TEXT','Q26_Part_1','Q26_Part_2','Q26_Part_3','Q26_Part_4','Q26_Part_5','Q26_Part_6','Q26_Part_7','Q26_OTHER_TEXT','Q27_Part_1','Q27_Part_2','Q27_Part_3','Q27_Part_4','Q27_Part_5','Q27_Part_6','Q27_OTHER_TEXT','Q28_Part_1','Q28_Part_2','Q28_Part_3','Q28_Part_4','Q28_Part_5','Q28_Part_6','Q28_Part_7','Q28_Part_8','Q28_Part_9','Q28_Part_10','Q28_Part_11','Q28_Part_12','Q28_OTHER_TEXT','Q29_Part_1','Q29_Part_2','Q29_Part_3','Q29_Part_4','Q29_Part_5','Q29_Part_6','Q29_Part_7','Q29_Part_8','Q29_Part_9','Q29_Part_10','Q29_Part_11','Q29_Part_12','Q29_OTHER_TEXT','Q30_Part_1','Q30_Part_2','Q30_Part_3','Q30_Part_4','Q30_Part_5','Q30_Part_6','Q30_Part_7','Q30_Part_8','Q30_Part_9','Q30_Part_10','Q30_Part_11','Q30_Part_12','Q30_OTHER_TEXT','Q31_Part_1','Q31_Part_2','Q31_Part_3','Q31_Part_4','Q31_Part_5','Q31_Part_6','Q31_Part_7','Q31_Part_8','Q31_Part_9','Q31_Part_10','Q31_Part_11','Q31_Part_12','Q31_OTHER_TEXT','Q32_Part_1','Q32_Part_2','Q32_Part_3','Q32_Part_4','Q32_Part_5','Q32_Part_6','Q32_Part_7','Q32_Part_8','Q32_Part_9','Q32_Part_10','Q32_Part_11','Q32_Part_12','Q32_OTHER_TEXT','Q33_Part_1','Q33_Part_2','Q33_Part_3','Q33_Part_4','Q33_Part_5','Q33_Part_6','Q33_Part_7','Q33_Part_8','Q33_Part_9','Q33_Part_10','Q33_Part_11','Q33_Part_12','Q33_OTHER_TEXT','Q34_Part_1','Q34_Part_2','Q34_Part_3','Q34_Part_4','Q34_Part_5','Q34_Part_6','Q34_Part_7','Q34_Part_8','Q34_Part_9','Q34_Part_10','Q34_Part_11','Q34_Part_12','Q34_OTHER_TEXT']
for column in columns:
  DF_data[column].fillna("Not Answered", inplace=True)
DF_data.rename(columns={'Time from Start to Finish (seconds)':'Duration (in seconds)',
'Q1':'Age',
'Q2':'Gender',
'Q2_OTHER_TEXT':'Gender_Other',
'Q3':'Resident Country',
'Q4':'Highest level of Education',
'Q5':'Title/Profession',
'Q5_OTHER_TEXT':'Current/Most Recent title_Other',
'Q6':'Size of the company',
'Q7':'No of individuals  for data science work',
'Q8':'Any ML methods used in Business',
'Q9_Part_1':'Role at work_Analyze Data',
'Q9_Part_2':'Role at work_Build and/or Run Data',
'Q9_Part_3':'Role at work_Build prototypes',
'Q9_Part_4':'Role at work_ Build and/or run a machine learning service',
'Q9_Part_5':'Role at work_ Experimentation and iteration(ML models)',
'Q9_Part_6':'Role at work_Researche in machine learning',
'Q9_Part_7':'Role at work_None of these activities ',
'Q9_Part_8':'Role at work_Other',
'Q9_OTHER_TEXT':'Role at work_ Other _ Text',
'Q10':'Yearly Compensation ($USD)',
'Q11':'Amount spent on machine learning ,cloud computing products',
'Q12_Part_1':'Twitter',
'Q12_Part_2':'Hacker News',
'Q12_Part_3':'Reddit',
'Q12_Part_4':'Kaggle',
'Q12_Part_5':'Course Forums',
'Q12_Part_6':'YouTube',
'Q12_Part_7':'Podcasts',
'Q12_Part_8':'Blogs',
'Q12_Part_9':'Journal Publications',
'Q12_Part_10':'Slack Communities',
'Q12_Part_11':'No Media Sources',
'Q12_Part_12':'Other Media Sources',
'Q12_OTHER_TEXT':'Other Media sources_Text',
'Q13_Part_1':'Udacity',
'Q13_Part_2':'Coursera',
'Q13_Part_3':'edX',
'Q13_Part_4':'DataCamp',
'Q13_Part_5':'DataQuest',
'Q13_Part_6':'Kaggle Courses',
'Q13_Part_7':'Fast.ai',
'Q13_Part_8':'Udemy',
'Q13_Part_9':'LinkedIn Learning',
'Q13_Part_10':'University Courses',
'Q13_Part_11':'No Platform',
'Q13_Part_12':'Other Platform',
'Q13_OTHER_TEXT':'Other Platform_text',
'Q14':'Primary Tool for analyse data_ Selected Choice',
'Q14_Part_1_TEXT':'Basic statistical software (Microsoft Excel, Google Sheets, etc.)',
'Q14_Part_2_TEXT':'Advanced statistical software (SPSS, SAS, etc.)',
'Q14_Part_3_TEXT':'Business intelligence software (Salesforce, Tableau, Spotfire, etc.)',
'Q14_Part_4_TEXT':'Local development environments (RStudio, JupyterLab, etc.)',
'Q14_Part_5_TEXT':'Cloud-based data software & APIs (AWS, GCP, Azure, etc.) ',
'Q14_OTHER_TEXT':'Primary Tool for analyse data_ Other - Text',
'Q15':'Experience writing code to analyze data',
'Q16_Part_1':'IDE_Jupyter',
'Q16_Part_2':'IDE_RStudio',
'Q16_Part_3':'IDE_PyCharm',
'Q16_Part_4':'IDE_Atom',
'Q16_Part_5':'IDE_MATLAB',
'Q16_Part_6':'IDE_Visual Studio / Visual Studio Code',
'Q16_Part_7':'IDE_Spyder',
'Q16_Part_8':'IDE_Vim / Emacs',
'Q16_Part_9':'IDE_Notepad++',
'Q16_Part_10':'IDE_Sublime Text',
'Q16_Part_11':'IDE_None',
'Q16_Part_12':'IDE_Other',
'Q16_OTHER_TEXT':'IDE_Other_Text',
'Q17_Part_1':'Kaggle Notebooks',
'Q17_Part_2':'Google Colab',
'Q17_Part_3':'Microsoft Azure Notebooks',
'Q17_Part_4':'Google Cloud Notebook Products',
'Q17_Part_5':'Paperspace/Gradient',
'Q17_Part_6':'FloydHub',
'Q17_Part_7':'Binder/JupyterHub',
'Q17_Part_8':'IBM Watson Studio',
'Q17_Part_9':'Code Ocean',
'Q17_Part_10':'AWS Notebook Products',
'Q17_Part_11':'Notebook_None',
'Q17_Part_12':'Notebook_Other',
'Q17_OTHER_TEXT':'Notebook_Other_Text',
'Q18_Part_1':'Python',
'Q18_Part_2':'R',
'Q18_Part_3':'SQL',
'Q18_Part_4':'C',
'Q18_Part_5':'C++',
'Q18_Part_6':'Java',
'Q18_Part_7':'Javascript',
'Q18_Part_8':'TypeScript',
'Q18_Part_9':'Bash',
'Q18_Part_10':'MATLAB',
'Q18_Part_11':'No Programming Language',
'Q18_Part_12':'Other Prgramming Language',
'Q18_OTHER_TEXT':'Other Prgramming Language_Text',
'Q19':'Programming Lang Recommend.',
'Q19_OTHER_TEXT':'Programming Lang Recommend_Other_Text',
'Q20_Part_1':'Ggplot / ggplot2',
'Q20_Part_2':'Matplotlib',
'Q20_Part_3':'Altair',
'Q20_Part_4':'Shiny',
'Q20_Part_5':'D3.js',
'Q20_Part_6':'Plotly/Plotly Express',
'Q20_Part_7':'Bokeh',
'Q20_Part_8':'Seaborn',
'Q20_Part_9':'Geoplotlib',
'Q20_Part_10':'Leaflet / Folium',
'Q20_Part_11':'No Data Visualization',
'Q20_Part_12':'Other Data Visualization',
'Q20_OTHER_TEXT':'Other Data Visualization_Text',
'Q21_Part_1':'Specialized H/W_CPUs',
'Q21_Part_2':'Specialized H/W_GPUs',
'Q21_Part_3':'Specialized H/W_TPUs',
'Q21_Part_4':'Specialized H/W_None',
'Q21_Part_5':'Specialized H/W_Other',
'Q21_OTHER_TEXT':'Specialized H/W_OtherText',
'Q22':'Have you ever used a TPU (tensor processing unit)?',
'Q23':'Years using ML methods',
'Q24_Part_1':'Linear or Logistic Regression',
'Q24_Part_2':'Decision Trees or Random Forests',
'Q24_Part_3':'Gradient Boosting Machines',
'Q24_Part_4':'Bayesian Approaches',
'Q24_Part_5':'Evolutionary Approaches',
'Q24_Part_6':'Dense Neural Networks',
'Q24_Part_7':'Convolutional Neural Networks',
'Q24_Part_8':'Generative Adversarial Networks',
'Q24_Part_9':'Recurrent Neural Networks',
'Q24_Part_10':'Transformer Networks',
'Q24_Part_11':'No ML algorthms',
'Q24_Part_12':'ML algorthms used_ Other',
'Q24_OTHER_TEXT':'ML algorthms used_ Other_Text',
'Q25_Part_1':'Automated data augmentation',
'Q25_Part_2':'Automated feature engineering/selection',
'Q25_Part_3':'Automated model selection',
'Q25_Part_4':'Automated model architecture searches',
'Q25_Part_5':'Automated hyperparameter tuning',
'Q25_Part_6':'Automation of full ML pipelines',
'Q25_Part_7':'ML Tools_None',
'Q25_Part_8':'ML Tools_Other',
'Q25_OTHER_TEXT':'ML Tools_Other_Text',
'Q26_Part_1':'CV purpose image/video tools',
'Q26_Part_2':'CV Image segmentation methods',
'Q26_Part_3':'CV Object detection methods',
'Q26_Part_4':'CV Image classification/General purpose networks',
'Q26_Part_5':'CV Generative Networks',
'Q26_Part_6':'No Computer Vision Tool',
'Q26_Part_7':'Computer_Vision_Other',
'Q26_OTHER_TEXT':'Computer_Vision_Other_Text',
'Q27_Part_1':'NLP_Word embeddings/vectors',
'Q27_Part_2':'NLP_Encoder-decorder models',
'Q27_Part_3':'NLP_Contextualized embeddings',
'Q27_Part_4':'NLP_Transformer language models',
'Q27_Part_5':'NLP_Method_None',
'Q27_Part_6':'NLP_Method_Other',
'Q27_OTHER_TEXT':'NLP_Method_Other_Text',
'Q28_Part_1':'Scikit-learn',
'Q28_Part_2':'TensorFlow',
'Q28_Part_3':'Keras',
'Q28_Part_4':'RandomForest',
'Q28_Part_5':'Xgboost',
'Q28_Part_6':'PyTorch',
'Q28_Part_7':'Caret',
'Q28_Part_8':'LightGBM',
'Q28_Part_9':'Spark MLib ',
'Q28_Part_10':'Fast.ai',
'Q28_Part_11':'No ML Framework',
'Q28_Part_12':'Other ML Framework',
'Q28_OTHER_TEXT':'ML_Framework_Other_Text',
'Q29_Part_1':'Google Cloud Platform (GCP)',
'Q29_Part_2':'Amazon Web Services (AWS)',
'Q29_Part_3':'Microsoft Azure',
'Q29_Part_4':'IBM Cloud',
'Q29_Part_5':'Alibaba Cloud',
'Q29_Part_6':'Salesforce Cloud',
'Q29_Part_7':'Oracle Cloud',
'Q29_Part_8':'SAP Cloud',
'Q29_Part_9':'VMware Cloud',
'Q29_Part_10':'Red Hat Cloud',
'Q29_Part_11':'No Cloud Computing Platform',
'Q29_Part_12':'Cloud_Computing_Platform_Other',
'Q29_OTHER_TEXT':'Cloud_Computing_Platform_ Other_Text',
'Q30_Part_1':'AWS Elastic Compute Cloud (EC2)',
'Q30_Part_2':'Google Compute Engine (GCE)',
'Q30_Part_3':'AWS Lambda',
'Q30_Part_4':'Azure Virtual Machines',
'Q30_Part_5':'Google App Engine',
'Q30_Part_6':'Google Cloud Functions',
'Q30_Part_7':'AWS Elastic Beanstalk',
'Q30_Part_8':'Google Kubernetes Engine',
'Q30_Part_9':'AWS Batch',
'Q30_Part_10':'Azure Container Service',
'Q30_Part_11':'No Cloud Computing Products',
'Q30_Part_12':'Cloud_Computing_Products_Other',
'Q30_OTHER_TEXT':'Cloud_Computing_Products_Other_Text',
'Q31_Part_1':'Google BigQuery',
'Q31_Part_2':'AWS Redshift',
'Q31_Part_3':'Databricks',
'Q31_Part_4':'AWS Elastic MapReduce',
'Q31_Part_5':'Teradata',
'Q31_Part_6':'Microsoft Analysis Services',
'Q31_Part_7':'Google Cloud Dataflow',
'Q31_Part_8':'AWS Athena',
'Q31_Part_9':'AWS Kinesis',
'Q31_Part_10':'Google Cloud Pub/Sub',
'Q31_Part_11':' No Big Data/Analytics Product',
'Q31_Part_12':'Big Data/Analytics_Product_Other',
'Q31_OTHER_TEXT':'Big Data/Analytics_Product_Other',
'Q32_Part_1':'SAS',
'Q32_Part_2':'Cloudera',
'Q32_Part_3':'Azure Machine Learning Studio',
'Q32_Part_4':'Google Cloud Machine Learning Engine',
'Q32_Part_5':'Google Cloud Vision',
'Q32_Part_6':'Google Cloud Speech-to-Text',
'Q32_Part_7':'Google Cloud Natural Language',
'Q32_Part_8':'RapidMiner',
'Q32_Part_9':'Google Cloud Translation',
'Q32_Part_10':'Amazon SageMaker',
'Q32_Part_11':'No Machine Learning Products',
'Q32_Part_12':'Machine_Learning_Products_Other',
'Q32_OTHER_TEXT':'Machine_Learning_Products_Other_Text',
'Q33_Part_1':'Google AutoML',
'Q33_Part_2':'H20 Driverless AI',
'Q33_Part_3':'Databricks AutoML',
'Q33_Part_4':'DataRobot AutoML',
'Q33_Part_5':'Tpot',
'Q33_Part_6':'Auto-Keras',
'Q33_Part_7':'Auto-Sklearn',
'Q33_Part_8':'Auto_ml',
'Q33_Part_9':'Xcessiv',
'Q33_Part_10':'MLbox',
'Q33_Part_11':'None Automated ML Tool',
'Q33_Part_12':'Automated_Machine_Learning_Tools_Other',
'Q33_OTHER_TEXT':'Automated_Machine_Learning_Tools_Other_Text',
'Q34_Part_1':'MySQL',
'Q34_Part_2':'PostgresSQL',
'Q34_Part_3':'SQLite',
'Q34_Part_4':'Microsoft SQL Server',
'Q34_Part_5':'Oracle Database',
'Q34_Part_6':'Microsoft Access',
'Q34_Part_7':'AWS Relational Database Service',
'Q34_Part_8':'AWS DynamoDB',
'Q34_Part_9':'Azure SQL Database',
'Q34_Part_10':'Google Cloud SQL',
'Q34_Part_11':'No RelationalDatabase',
'Q34_Part_12':'Relational_Database_Other',
'Q34_OTHER_TEXT':'Relational_Database_Other_Text'
}, inplace = True)
#dropped the first column for time taken to finish survey
DF_data=DF_data.drop('Duration (in seconds)',axis = 1)
#created the copy of dataset as backup
DF_data_bkup = DF_data.copy()
#label encoding to convert the numbers
le = preprocessing.LabelEncoder()
DF_data = DF_data.apply(le.fit_transform)
km_cao = KModes(n_clusters=3, init = "Cao", n_init = 1, verbose=1)
fitClusters_cao = km_cao.fit_predict(DF_data)
#Combining the predicted clusters with the original DF.
DF_data=DF_data_bkup.reset_index()
clustersDf = pd.DataFrame(fitClusters_cao)
clustersDf.columns = ['cluster_predicted']
combinedDf = pd.concat([DF_data, clustersDf], axis = 1).reset_index()
combinedDf = combinedDf.drop(['index', 'level_0'], axis = 1)
combinedDf_bkup=combinedDf.copy
cluster_0 = combinedDf[combinedDf['cluster_predicted'] == 0]
cluster_1 = combinedDf[combinedDf['cluster_predicted'] == 1]
cluster_2 = combinedDf[combinedDf['cluster_predicted'] == 2]
#replancing "Not Answered" with Np.Nan for better data visualisation
combinedDF=combinedDf.replace('Not Answered', np.nan,inplace=True)
#cluster_0 Profession and education
f, axs =plt.subplots(2,1,figsize = (30,15))
sns.countplot(x=cluster_0['Title/Profession'],order=cluster_0['Title/Profession'].value_counts(normalize=True).index,ax=axs[0])
sns.countplot(x=cluster_0['Highest level of Education'],order=cluster_0['Highest level of Education'].value_counts(normalize=True).index,ax=axs[1])
plt.tight_layout()
plt.show()
#cluster_1 Profession and education
f, axs =plt.subplots(2,1,figsize = (30,15))
sns.countplot(x=cluster_1['Title/Profession'],order=cluster_1['Title/Profession'].value_counts(normalize=True).index,ax=axs[0])
sns.countplot(x=cluster_1['Highest level of Education'],order=cluster_1['Highest level of Education'].value_counts(normalize=True).index,ax=axs[1])
plt.tight_layout()
plt.show()
#cluster_2 Profession and education
f, axs =plt.subplots(2,1,figsize = (30,15))
sns.countplot(x=cluster_2['Title/Profession'],order=cluster_2['Title/Profession'].value_counts(normalize=True).index,ax=axs[0])
sns.countplot(x=cluster_2['Highest level of Education'],order=cluster_2['Highest level of Education'].value_counts(normalize=True).index,ax=axs[1])
plt.tight_layout()
plt.show()
sns.set(style="white",font_scale=1.3)
5#to show clustering based on profession,education,sex and age
f, axs =plt.subplots(2,1,figsize = (30,15))
sns.countplot(x=combinedDf['Title/Profession'],order=combinedDf['Title/Profession'].value_counts(normalize=True).index,hue=combinedDf['cluster_predicted'],ax=axs[0])
sns.countplot(x=combinedDf['Highest level of Education'],order=combinedDf['Highest level of Education'].value_counts(normalize=True).index,hue=combinedDf['cluster_predicted'],ax=axs[1])
plt.tight_layout()
plt.show()
sns.set(style="white",font_scale=1.3)
f, axs = plt.subplots(2,1,figsize = (15,10))
sns.countplot(x=combinedDf['Gender'],order=combinedDf['Gender'].value_counts(normalize=True).index,hue=combinedDf['cluster_predicted'],ax=axs[0])
sns.countplot(x=combinedDf['Age'],order=combinedDf['Age'].value_counts(normalize=True).index,hue=combinedDf['cluster_predicted'],ax=axs[1])
#Plt.xlable()
plt.tight_layout()
plt.show()
temp = combinedDf['Resident Country'].value_counts().sort_values(ascending=False)[0:10]
top10Country = temp.reset_index()
top10Country.columns = ["Country","NumberOfParticipants"]
sns.set(style="white",font_scale=1.3)
g=sns.catplot(y="NumberOfParticipants", x= "Country",
                 data=top10Country, kind="bar",
                height=5, aspect=2,
                edgecolor=sns.color_palette("dark", 3));
plt.title("Participants Country");
g.set_xticklabels(rotation=90);
#Top 10 country based on clustering
sns.set(style="white",font_scale=1.3)
f, axs = plt.subplots(figsize = (20,15))
#plt.subplots(figsize = (30,25))
sns.countplot(x=combinedDf['Resident Country'],order=combinedDf['Resident Country'].value_counts(normalize=True)[:10].index,hue=combinedDf['cluster_predicted'])

plt.tight_layout()
plt.show()

#IDE's
sns.set(style="white",font_scale=1.3)
f, axs = plt.subplots(2,3,figsize = (15,10))
#f.suptitle('Integrated development environments(IDE)')
sns.countplot(x= combinedDf['IDE_Jupyter'],order=combinedDf['IDE_Jupyter'].value_counts(normalize=True).index,hue= combinedDf['cluster_predicted'],ax=axs[0,0])
sns.countplot(x= combinedDf['IDE_RStudio'],order=combinedDf['IDE_RStudio'].value_counts(normalize=True).index,hue=combinedDf['cluster_predicted'],ax=axs[0,1])
sns.countplot(x= combinedDf['IDE_PyCharm'],order=combinedDf['IDE_PyCharm'].value_counts(normalize=True).index,hue=combinedDf['cluster_predicted'],ax=axs[0,2])
sns.countplot(x= combinedDf['IDE_MATLAB'],order=combinedDf['IDE_MATLAB'].value_counts(normalize=True).index,hue= combinedDf['cluster_predicted'],ax=axs[1,0])
sns.countplot(x= combinedDf['IDE_Visual Studio / Visual Studio Code'],order=combinedDf['IDE_Visual Studio / Visual Studio Code'].value_counts(normalize=True).index,hue= combinedDf['cluster_predicted'],ax=axs[1,1])
sns.countplot(x= combinedDf['IDE_Notepad++'],order= combinedDf['IDE_Notepad++'].value_counts(normalize=True).index,hue= combinedDf['cluster_predicted'],ax=axs[1,2])
plt.tight_layout()
plt.show()
#f.suptitle('Integrated development environments(IDE)')
#to show clustering based on framewok
sns.set(style="white",font_scale=1.3)
f, axs = plt.subplots(2,3,figsize = (15,10))
sns.countplot(x=combinedDf['Scikit-learn'],order=combinedDf['Scikit-learn'].value_counts(normalize=True).index,hue=combinedDf['cluster_predicted'],ax=axs[0,0])
sns.countplot(x=combinedDf['TensorFlow'],order=combinedDf['TensorFlow'].value_counts(normalize=True).index,hue=combinedDf['cluster_predicted'],ax=axs[0,1])
sns.countplot(x=combinedDf['Keras'],order=combinedDf['Keras'].value_counts(normalize=True).index,hue=combinedDf['cluster_predicted'],ax=axs[0,2])
sns.countplot(x=combinedDf['RandomForest'],order=combinedDf['RandomForest'].value_counts(normalize=True).index,hue=combinedDf['cluster_predicted'],ax=axs[1,0])
sns.countplot(x=combinedDf['Xgboost'],order=combinedDf['Xgboost'].value_counts(normalize=True).index,hue=combinedDf['cluster_predicted'],ax=axs[1,1])
sns.countplot(x=combinedDf['PyTorch'],order=combinedDf['PyTorch'].value_counts(normalize=True).index,hue=combinedDf['cluster_predicted'],ax=axs[1,2])
#f.suptitle('Machine Learning Framework')
plt.tight_layout()
plt.show()
#notebooks
sns.set(style="white",font_scale=1.3)
f, axs = plt.subplots(2,3,figsize = (15,10))
#plt.title('Machine Learning Notebooks')
sns.countplot(x=combinedDf['Kaggle Notebooks'],order=combinedDf['Kaggle Notebooks'].value_counts(normalize=True).index,hue=combinedDf['cluster_predicted'],ax=axs[0,0])
sns.countplot(x=combinedDf['Google Colab'],order=combinedDf['Google Colab'].value_counts(normalize=True).index,hue=combinedDf['cluster_predicted'],ax=axs[0,1])
sns.countplot(x=combinedDf['Microsoft Azure Notebooks'],order=combinedDf['Microsoft Azure Notebooks'].value_counts(normalize=True).index,hue=combinedDf['cluster_predicted'],ax=axs[0,2])
sns.countplot(x=combinedDf['Binder/JupyterHub'],order=combinedDf['Binder/JupyterHub'].value_counts(normalize=True).index,hue=combinedDf['cluster_predicted'],ax=axs[1,0])
sns.countplot(x=combinedDf['IBM Watson Studio'],order=combinedDf['IBM Watson Studio'].value_counts(normalize=True).index,hue=combinedDf['cluster_predicted'],ax=axs[1,1])
sns.countplot(x=combinedDf['AWS Notebook Products'],order=combinedDf['AWS Notebook Products'].value_counts(normalize=True).index,hue=combinedDf['cluster_predicted'],ax=axs[1,2])
#f.suptitle('Programming Language Used')
plt.tight_layout()
plt.show()
#Programming language
sns.set(style="white",font_scale=1.3)
f, axs = plt.subplots(3,3,figsize = (15,10))
sns.countplot(x=combinedDf['Python'],order=combinedDf['Python'].value_counts(normalize=True).index,hue=combinedDf['cluster_predicted'],ax=axs[0,0])
sns.countplot(x=combinedDf['R'],order=combinedDf['R'].value_counts(normalize=True).index,hue=combinedDf['cluster_predicted'],ax=axs[0,1])
sns.countplot(x=combinedDf['MATLAB'],order=combinedDf['MATLAB'].value_counts(normalize=True).index,hue=combinedDf['cluster_predicted'],ax=axs[0,2])
sns.countplot(x=combinedDf['C++'],order=combinedDf['C++'].value_counts(normalize=True).index,hue=combinedDf['cluster_predicted'],ax=axs[1,0])
sns.countplot(x=combinedDf['Javascript'],order=combinedDf['Javascript'].value_counts(normalize=True).index,hue=combinedDf['cluster_predicted'],ax=axs[1,1])
sns.countplot(x=combinedDf['TypeScript'],order=combinedDf['TypeScript'].value_counts(normalize=True).index,hue=combinedDf['cluster_predicted'],ax=axs[1,2])
sns.countplot(x=combinedDf['C'],order=combinedDf['C'].value_counts(normalize=True).index,hue=combinedDf['cluster_predicted'],ax=axs[2,0])
sns.countplot(x=combinedDf['Java'],order=combinedDf['Java'].value_counts(normalize=True).index,hue=combinedDf['cluster_predicted'],ax=axs[2,1])
sns.countplot(x=combinedDf['Bash'],order=combinedDf['Bash'].value_counts(normalize=True).index,hue=combinedDf['cluster_predicted'],ax=axs[2,2])
#f.suptitle('Programming Language Used')
plt.tight_layout()
plt.show()
#cloud computing services
sns.set(style="white",font_scale=1.3)
f, axs = plt.subplots(4,3,figsize = (15,20))
sns.countplot(x=combinedDf['Google Cloud Platform (GCP)'],order=combinedDf['Google Cloud Platform (GCP)'].value_counts(normalize=True).index,hue=combinedDf['cluster_predicted'],ax=axs[0,0])
sns.countplot(x=combinedDf['Amazon Web Services (AWS)'],order=combinedDf['Amazon Web Services (AWS)'].value_counts(normalize=True).index,hue=combinedDf['cluster_predicted'],ax=axs[0,1])
sns.countplot(x=combinedDf['Microsoft Azure'],order=combinedDf['Microsoft Azure'].value_counts(normalize=True).index,hue=combinedDf['cluster_predicted'],ax=axs[0,2])
sns.countplot(x=combinedDf['IBM Cloud'],order=combinedDf['IBM Cloud'].value_counts(normalize=True).index,hue=combinedDf['cluster_predicted'],ax=axs[1,0])
sns.countplot(x=combinedDf['Alibaba Cloud'],order=combinedDf['Alibaba Cloud'].value_counts(normalize=True).index,hue=combinedDf['cluster_predicted'],ax=axs[1,1])
sns.countplot(x=combinedDf['Salesforce Cloud'],order=combinedDf['Salesforce Cloud'].value_counts(normalize=True).index,hue=combinedDf['cluster_predicted'],ax=axs[1,2])
sns.countplot(x=combinedDf['Oracle Cloud'],order=combinedDf['Oracle Cloud'].value_counts(normalize=True).index,hue=combinedDf['cluster_predicted'],ax=axs[2,0])
sns.countplot(x=combinedDf['SAP Cloud'],order=combinedDf['SAP Cloud'].value_counts(normalize=True).index,hue=combinedDf['cluster_predicted'],ax=axs[2,1])
sns.countplot(x=combinedDf['VMware Cloud'],order=combinedDf['VMware Cloud'].value_counts(normalize=True).index,hue=combinedDf['cluster_predicted'],ax=axs[2,2])
sns.countplot(x=combinedDf['Red Hat Cloud'],order=combinedDf['Red Hat Cloud'].value_counts(normalize=True).index,hue=combinedDf['cluster_predicted'],ax=axs[3,0])
sns.countplot(x=combinedDf['Red Hat Cloud'],order=combinedDf['Red Hat Cloud'].value_counts(normalize=True).index,hue=combinedDf['cluster_predicted'],ax=axs[3,1])
#f.suptitle('Cloud Computing Services')
#plt.tight_layout()
plt.show()
#Data Visualisation libraries
sns.set(style="white",font_scale=1.3)
f, axs = plt.subplots(4,3,figsize = (15,10))
sns.countplot(x=combinedDf['Ggplot / ggplot2'],order=combinedDf['Ggplot / ggplot2'].value_counts(normalize=True).index,hue=combinedDf['cluster_predicted'],ax=axs[0,0])
sns.countplot(x=combinedDf['Matplotlib'],order=combinedDf['Matplotlib'].value_counts(normalize=True).index,hue=combinedDf['cluster_predicted'],ax=axs[0,1])
sns.countplot(x=combinedDf['Plotly/Plotly Express'],order=combinedDf['Plotly/Plotly Express'].value_counts(normalize=True).index,hue=combinedDf['cluster_predicted'],ax=axs[0,2])
sns.countplot(x=combinedDf['Bokeh'],order=combinedDf['Bokeh'].value_counts(normalize=True).index,hue=combinedDf['cluster_predicted'],ax=axs[1,0])
sns.countplot(x=combinedDf['Seaborn'],order=combinedDf['Seaborn'].value_counts(normalize=True).index,hue=combinedDf['cluster_predicted'],ax=axs[1,1])
sns.countplot(x=combinedDf['Geoplotlib'],order=combinedDf['Geoplotlib'].value_counts(normalize=True).index,hue=combinedDf['cluster_predicted'],ax=axs[1,2])
sns.countplot(x=combinedDf['Altair'],order=combinedDf['Altair'].value_counts(normalize=True).index,hue=combinedDf['cluster_predicted'],ax=axs[2,0])
sns.countplot(x=combinedDf['Shiny'],order=combinedDf['Shiny'].value_counts(normalize=True).index,hue=combinedDf['cluster_predicted'],ax=axs[2,1])
sns.countplot(x=combinedDf['D3.js'],order=combinedDf['D3.js'].value_counts(normalize=True).index,hue=combinedDf['cluster_predicted'],ax=axs[2,2])
sns.countplot(x=combinedDf['Leaflet / Folium'],order=combinedDf['Leaflet / Folium'].value_counts(normalize=True).index,hue=combinedDf['cluster_predicted'],ax=axs[3,0])
sns.countplot(x=combinedDf['No Data Visualization'],order=combinedDf['No Data Visualization'].value_counts(normalize=True).index,hue=combinedDf['cluster_predicted'],ax=axs[3,1])
#f.suptitle('Data Visualisation Libraries')
plt.tight_layout()
plt.show()
