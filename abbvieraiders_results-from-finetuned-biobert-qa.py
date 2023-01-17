import pandas as pd
from IPython.display import display, HTML

input_path = '../input/covid-drug-target-qa-inputs/COVID Drug Target QA Inputs/'

covid = pd.read_csv(input_path + '388472.txt',encoding='utf-8',sep ='\t',header = None)

covid[[4]] = covid[[4]].apply(pd.to_numeric)

#############################
# pick a Score Threshold
##############################

covid2=covid[covid[4] > 0.76]


targets= covid2.rename(columns={0: "link", 1: "title ", 2: "Abstract",3:"Target", 4:"Score"})

file_path='./first_pass_drug_targets.xlsx'

targets.to_excel(file_path,encoding='UTF-8',index=False,float_format='%g') 

display(HTML(targets.to_html(render_links=True)))
covid = pd.read_csv(input_path + '669533.txt',encoding='utf-8',sep ='\t',header = None)

covid[[4]] = covid[[4]].apply(pd.to_numeric)

##############################
# pick a Score Threshold
##############################

covid2=covid[covid[4] > 0.8]


drugs= covid2.rename(columns={0: "link", 1: "title ", 2: "Abstract",3:"Drug", 4:"Score"})

file_path='./first_pass_drugs_used_to_treat_COVID19.xlsx'

drugs.to_excel(file_path,encoding='UTF-8',index=False,float_format='%g')

display(HTML(drugs.to_html(render_links=True)))
covid = pd.read_csv(input_path + '645033.txt',encoding='utf-8',sep ='\t',header = None)

covid[[4]] = covid[[4]].apply(pd.to_numeric)

##############################
# pick a Score Threshold
##############################

covid2=covid[covid[4] > 0.1]


risks= covid2.rename(columns={0: "link", 1: "title ", 2: "Abstract",3:"Risk Factor(s)", 4:"Score"})

file_path='./first_pass_risks_factors_for_COVID19.xlsx'

risks.to_excel(file_path,encoding='UTF-8',index=False,float_format='%g')  

display(HTML(risks.to_html(render_links=True)))