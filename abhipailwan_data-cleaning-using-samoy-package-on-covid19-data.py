import pandas as pd
df=pd.read_csv('/kaggle/input/corona-virus-report/covid_19_clean_complete.csv')
df
!pip install samoy
import samoy as sm
help(sm)
help(sm.Handling_Null)
result=sm.swapmissing_lru(df)
result
result = sm.dropmissing(df)
result
result=sm.swapmissing(df, 'mean')
result
result=sm.dropnull(df)
result
result=sm.dropnull_th(df,60)
result
result=sm.swapnull(df, method='mean', num_val=0, char_val='unknown')
result
help(sm.Handling_Duplicate)
result=sm.drop_replicatecols(df)
result
result=sm.drop_replicates(df)
result
help(sm.Case_Conversion)
result=sm.altercase(df, 'lower')
result
