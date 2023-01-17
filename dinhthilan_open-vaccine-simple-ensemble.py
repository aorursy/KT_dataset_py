import pandas as pd

import numpy as np
df1 = pd.read_csv('/kaggle/input/gru-lstm-mix-with-custom-loss-tunning/ensemble_final.csv')

df2 = pd.read_csv('/kaggle/input/mvan-covid-mrna-vaccine-analysis-notebook-268/submission.csv')

df3 = pd.read_csv('/kaggle/input/gru-lstm-mix-with-custom-loss/ensemble_final.csv')
df1.head()
df2.head()
df3.head()
sub = df1

sub['reactivity'] = (df1.reactivity.values + df2.reactivity.values + df3.reactivity.values)/3

sub['deg_Mg_pH10'] = (df1.deg_Mg_pH10.values + df2.deg_Mg_pH10.values + df3.deg_Mg_pH10.values)/3

sub['deg_pH10'] = (df1.deg_pH10.values + df2.deg_pH10.values + df3.deg_pH10.values)/3

sub['deg_Mg_50C'] = (df1.deg_Mg_50C.values + df2.deg_Mg_50C.values +df3.deg_Mg_50C.values)/3

sub['deg_50C'] = (df1.deg_50C.values + df2.deg_50C.values + df3.deg_50C.values)/3
sub.to_csv('submission.csv', index = False)
sub.head()