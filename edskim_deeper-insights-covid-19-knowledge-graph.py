# Utilities to display results.

from IPython.core.display import display, HTML
from pathlib import Path
import pandas as pd
import ast
pd.set_option('display.max_colwidth', -1)
pd.set_option('display.max_rows', 100)

def format_cell(value):
    
    if type(value) is list:
        children=[format_cell(child) for child in value]
        value='; '.join(map(str,value))
    if type(value) is dict:
        children=[f'{format_cell(k)}={format_cell(v)}' for k,v in value.items()]
        value='; '.join(children)
                  
        
    return str(value)
    

def show_results(filename):
    path=Path(f'/kaggle/input/queryoutputs/{filename}.txt')
    line_dicts=[ast.literal_eval(line.strip()) for line in path.read_text(encoding='utf-8').splitlines()]
    df=pd.DataFrame(line_dicts)
    for column in df.columns:
        df[column]=df[column].apply(format_cell)
    return df
    

show_results('asymp_child')
show_results('season')
show_results('persist_body')
show_results('materials')
show_results('models')
show_results('pheno')
show_results('immune')
show_results('ppe')
