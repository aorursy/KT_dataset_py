import numpy as np
import pandas as pd
import matplotlib.pyplot as pyplot

df = pd.read_csv("../input/anv.csv", index_col="codigo_ocorrencia")
df = pd.DataFrame(df, columns=["aeronave_fase_operacao","aeronave_tipo_operacao","aeronave_tipo_veiculo","aeronave_motor_quantidade","aeronave_pmd_categoria","aeronave_assentos","total_fatalidades"])
df.head(5)
var_clas = [['codigo_ocorrencia' , 'Quantitativa Discreta'],['aeronave_fase_operacao' , 'Qualitativa Nominal'],   
                    ['aeronave_tipo_operacao' , 'Qualitativa Nominal'], ['aeronave_tipo_veiculo' , 'Qualitativa Nominal'], 
                    ['aeronave_motor_quantidade' , 'Quantitativa Discreta'], ['aeronave_pmd_categoria' , 'Qualitativa Ordinal'],
                    ['aeronave_assentos' , 'Quantitativa Discreta'], ['total_fatalidades' , 'Quantitativa Discreta']]
                  
                  
var_clas = pd.DataFrame(var_clas, columns=['Variavel' , 'Classificacao'])
var_clas
fase_operacao = df["aeronave_fase_operacao"].value_counts()
fase_operacao.head()
tipo_operacao = df["aeronave_tipo_operacao"].value_counts()
tipo_operacao.head()
tipo_veiculo = df["aeronave_tipo_veiculo"].value_counts()
tipo_veiculo.head()
categoria = df["aeronave_pmd_categoria"].value_counts()
categoria.head()
ds = df[df.aeronave_motor_quantidade != '***'].groupby(['aeronave_motor_quantidade'])['total_fatalidades'].sum()
ds.plot.pie()
ds = df[df.aeronave_fase_operacao != '***'].groupby(['aeronave_fase_operacao'])['total_fatalidades'].sum()
ds = ds.sort_values()[ds > 0]
ds.plot.bar()
tipo_veiculo = df["aeronave_tipo_veiculo"].value_counts()
tipo_veiculo.plot.bar()
ds = df[df.aeronave_pmd_categoria != '***'].groupby(['aeronave_pmd_categoria'])['total_fatalidades'].sum()
ds = ds.sort_values()[ds > 0]
ds.plot.pie()
ds = df[df.aeronave_tipo_veiculo != '***'].groupby(['aeronave_tipo_veiculo'])['total_fatalidades'].sum()
ds = ds.sort_values()[ds > 0]
ds.plot.bar()
df = pd.read_csv('../input/anv.csv', delimiter=',')
df.head(1)