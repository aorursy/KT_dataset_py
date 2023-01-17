import pandas as pd
import numpy as np
import nltk
# nltk.download('stopwords')
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
# %matplotlib inline
response = pd.read_csv('../input/freeFormResponses.csv', low_memory=False)
survey = pd.read_csv('../input/SurveySchema.csv', low_memory=False)
mcr = pd.read_csv('../input/multipleChoiceResponses.csv', low_memory=False)
def clean_data(t_res_Q12):
  res_Q12 = []
  res_Q12_res = 0
  dic_res_Q12 = {}
  for i in t_res_Q12:
    if i.lower() != 'nan':
      res_Q12.append(i.lower().strip())
      res_Q12_res += 1
  del(res_Q12[0])
  temp_split = []
  temp_index = []
  for i in range(len(res_Q12)):
    if len(res_Q12[i].split()) > 1:
      temp_split += res_Q12[i].split()
      temp_index.append(i)
    elif len(res_Q12[i].split(',')) > 1:
      temp_split += res_Q12[i].split(',')
      temp_index.append(i)
    elif len(res_Q12[i].split(', ')) > 1:
      temp_split += res_Q12[i].split(',')
      temp_index.append(i)
    elif len(res_Q12[i].split('/')) > 1:
      temp_split += res_Q12[i].split(',')
      temp_index.append(i)
  for i in sorted(temp_index, reverse=True):
    del(res_Q12[i])
  res_Q12 += temp_split
  stop_words = stopwords.words('english')
  stop_words += ['none', 'nothing','use',' ', ',', '.', 'software', 'tool', 'tools', 'mostly', 'notebook', 'ide', 'studio', 'data']
  for i in res_Q12:
    if i not in stop_words:
      i = i.strip(',')
      i = i.strip()
      if i == 'microsoft' or i == 'ms':
        i = 'excel'
      elif i == 'google' or i =='sheet':
        i = 'sheets'
      elif i == 'power' or i == 'bi':
        i = 'powerbi'
      elif i == 'qlik':
        i = 'qlikview'
      elif i == 'rstudio':
        i = 'r'
      elif i == 'jupyterlab':
        i = 'jupyter'
      elif i == 'watson':
        i = 'ibm'
      elif i == 'pytorch':
        i = 'torch'
      elif i == 'vidhya' or i == 'analytics':
        i = 'analytics vidhya'
      elif i == 'science' or i == 'central':
        i = 'data science central'
      dic_res_Q12[i] = dic_res_Q12.get(i, 0) + 1
  dic_res_Q12 = sorted(dic_res_Q12.items(), key= lambda x: x[1], reverse=True)
  return dic_res_Q12
t_res_Q12 = np.array(list(response['Q12_OTHER_TEXT']))
dic_res_Q12 = clean_data(t_res_Q12)
plt_res_Q12 = dic_res_Q12[:10]
plt.bar(range(len(plt_res_Q12)), [val[1] for val in plt_res_Q12], align='center', label="Recorded user responses")
plt.xticks(range(len(plt_res_Q12)), [val[0] for val in plt_res_Q12])
plt.xticks(rotation=70)
plt.legend()
plt.title('Other analysis tools')
plt.draw()
plt.savefig('foo.png', bbox_inches='tight', dpi=300)
plt.show()
t_res_Q121 = np.array(list(response['Q12_Part_1_TEXT']))
dic_res_Q121 = clean_data(t_res_Q121)
plt_res_Q121 = dic_res_Q121[:10]
plt.bar(range(len(plt_res_Q121)), [val[1] for val in plt_res_Q121], align='center', color='r', label="Recorded user responses")
plt.xticks(range(len(plt_res_Q121)), [val[0] for val in plt_res_Q121])
plt.xticks(rotation=70)
plt.legend()
plt.title('Basic statistical analysis tools')
plt.draw()
plt.savefig('foo1.png', bbox_inches='tight', dpi=300)
plt.show()
t_res_Q122 = np.array(list(response['Q12_Part_2_TEXT']))
dic_res_Q122 = clean_data(t_res_Q122)
plt_res_Q122 = dic_res_Q122[:10]
plt.bar(range(len(plt_res_Q122)), [val[1] for val in plt_res_Q122], align='center', color='g', label="Recorded user responses")
plt.xticks(range(len(plt_res_Q122)), [val[0] for val in plt_res_Q122])
plt.xticks(rotation=70)
plt.legend()
plt.title('Advanced statistical analysis tools')
plt.draw()
plt.savefig('foo2.png', bbox_inches='tight', dpi=300)
plt.show()
t_res_Q123 = np.array(list(response['Q12_Part_3_TEXT']))
dic_res_Q123 = clean_data(t_res_Q123)
plt_res_Q123 = dic_res_Q123[:10]
plt.bar(range(len(plt_res_Q123)), [val[1] for val in plt_res_Q123], align='center', color='orange', label="Recorded user responses")
plt.xticks(range(len(plt_res_Q123)), [val[0] for val in plt_res_Q123])
plt.xticks(rotation=70)
plt.legend()
plt.title('Business intelligence tools')
plt.draw()
plt.savefig('foo3.png', bbox_inches='tight', dpi=300)
plt.show()
t_res_Q124 = np.array(list(response['Q12_Part_4_TEXT']))
dic_res_Q124 = clean_data(t_res_Q124)
plt_res_Q124 = dic_res_Q124[:10]
plt.bar(range(len(plt_res_Q124)), [val[1] for val in plt_res_Q124], align='center', color='pink', label="Recorded user responses")
plt.xticks(range(len(plt_res_Q124)), [val[0] for val in plt_res_Q124])
plt.xticks(rotation=70)
plt.legend()
plt.title('Local or hosted enviroment tools')
plt.draw()
plt.savefig('foo4.png', bbox_inches='tight', dpi=300)
plt.show()
t_res_Q125 = np.array(list(response['Q12_Part_5_TEXT']))
dic_res_Q125 = clean_data(t_res_Q125)
plt_res_Q125 = dic_res_Q125[:10]
plt.bar(range(len(plt_res_Q125)), [val[1] for val in plt_res_Q125], align='center', color='skyblue', label="Recorded user responses")
plt.xticks(range(len(plt_res_Q125)), [val[0] for val in plt_res_Q125])
plt.xticks(rotation=70)
plt.legend()
plt.title('Cloud based tools')
plt.draw()
plt.savefig('foo5.png', bbox_inches='tight', dpi=300)
plt.show()
t_res_Q13 = np.array(list(response['Q16_OTHER_TEXT']))
dic_res_Q13 = clean_data(t_res_Q13)
plt_res_Q13 = dic_res_Q13[:15]
plt.bar(range(len(plt_res_Q13)), [val[1] for val in plt_res_Q13], align='center', color='purple', label="Recorded user responses")
plt.xticks(range(len(plt_res_Q13)), [val[0] for val in plt_res_Q13])
plt.xticks(rotation=70)
plt.legend()
plt.title('Programming language used on a regular basis')
plt.draw()
plt.savefig('foo6.png', bbox_inches='tight', dpi=300)
plt.show()
t_res_Q14 = np.array(list(response['Q19_OTHER_TEXT']))
dic_res_Q14 = clean_data(t_res_Q14)
plt_res_Q14 = dic_res_Q14[:15]
plt.bar(range(len(plt_res_Q14)), [val[1] for val in plt_res_Q14], align='center', color='crimson', label="Recorded user responses")
plt.xticks(range(len(plt_res_Q14)), [val[0] for val in plt_res_Q14])
plt.xticks(rotation=70)
plt.legend()
plt.title('Most used ML frameworks in the past 5 years')
plt.draw()
plt.savefig('foo7.png', bbox_inches='tight', dpi=300)
plt.show()
t_res_Q15 = np.array(list(response['Q21_OTHER_TEXT']))
dic_res_Q15 = clean_data(t_res_Q15)
plt_res_Q15 = dic_res_Q15[:10]
plt.bar(range(len(plt_res_Q15)), [val[1] for val in plt_res_Q15], align='center', color='yellow', label="Recorded user responses")
plt.xticks(range(len(plt_res_Q15)), [val[0] for val in plt_res_Q15])
plt.xticks(rotation=70)
plt.legend()
plt.title('Most used data visualization tools in the past 5 years')
plt.draw()
plt.savefig('foo8.png', bbox_inches='tight', dpi=300)
plt.show()
t_res_Q16 = np.array(list(response['Q33_OTHER_TEXT']))
dic_res_Q16 = clean_data(t_res_Q16)
plt_res_Q16 = dic_res_Q16[:10]
plt.bar(range(len(plt_res_Q16)), [val[1] for val in plt_res_Q16], align='center', color='violet', label="Recorded user responses")
plt.xticks(range(len(plt_res_Q16)), [val[0] for val in plt_res_Q16])
plt.xticks(rotation=70)
plt.legend()
plt.title('Public dataset source')
plt.draw()
plt.savefig('foo9.png', bbox_inches='tight', dpi=300)
plt.show()
t_res_Q17 = np.array(list(response['Q38_OTHER_TEXT']))
dic_res_Q17 = clean_data(t_res_Q17)
plt_res_Q17 = dic_res_Q17[:10]
plt.bar(range(len(plt_res_Q17)), [val[1] for val in plt_res_Q17], align='center', color='gold', label="Recorded user responses")
plt.xticks(range(len(plt_res_Q17)), [val[0] for val in plt_res_Q17])
plt.xticks(rotation=70)
plt.legend()
plt.title('Favourite media sources for ML')
plt.draw()
plt.savefig('foo10.png', bbox_inches='tight', dpi=300)
plt.show()