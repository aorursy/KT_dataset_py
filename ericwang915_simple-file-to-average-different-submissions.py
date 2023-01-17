import pandas as pd

import argparse

parser = argparse.ArgumentParser(description='Merge different results in csv form, ex: python average_submission "1.csv 2.csv"')

parser.add_argument('--submission', type=str,help='several submission')

args = parser.parse_args()

sub=args.submission.split(' ')

df_concat = pd.concat((pd.read_csv(i, header=0).sort_values('image') for i in sub))

df_means=df_concat.mean()

by_row_index = df_concat.groupby('image')

df_means = by_row_index.mean()

df_means.to_csv('mean.csv')