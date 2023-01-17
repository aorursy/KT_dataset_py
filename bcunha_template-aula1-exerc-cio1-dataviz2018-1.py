import pandas as pd

cols = ['codigo_ocorrencia', 'aeronave_tipo_veiculo', 'aeronave_fabricante', 'aeronave_modelo', 'aeronave_pmd_categoria', 'aeronave_ano_fabricacao', 'aeronave_assentos', 'aeronave_motor_tipo', 'aeronave_motor_quantidade', 'aeronave_nivel_dano', 'aeronave_fase_operacao', 'total_fatalidades']
csv_df = pd.read_csv('../input/anv.csv', encoding='utf8', usecols=cols)

variable_classification = {
    'Variável': cols,
    'Classificação': ['Quantitativa Discreta', 'Qualitativa Ordinal', 'Qualitativa Nominal', 'Qualitativa Nominal', 'Qualitativa Ordinal', 'Quantitativa Discreta', 'Quantitativa Discreta', 'Qualitativa Nominal', 'Quantitativa Discreta', 'Qualitativa Ordinal', 'Qualitativa Nominal', 'Quantitativa Discreta']
}

variable_df = pd.DataFrame(variable_classification)
variable_df

# 1.  codigo_ocorrencia             IDENTIFICADOR ÚNICO DA OCORRÊNCIA          * quantitativa discreta
# 2.  aeronave_tipo_veiculo         TIPO DA AERONAVE                           * qualitativa ordinal
# 3.  aeronave_fabricante           FABRICANTE DA AERONAVE                     * qualitativa nominal
# 4.  aeronave_modelo               MODELO DA AERONAVE                         * qualitativa nominal
# 5.  aeronave_pmd_categoria        PORTE DA AERONAVE                          * qualitativa ordinal
# 6.  aeronave_ano_fabricacao       ANO DE FABRICAÇÃO DA AERONAVE              * quantitativa discreta
# 7.  aeronave_assentos             NÚMERO TOTAL DE ASSENTOS DA AERONAVE       * quantitativa discreta
# 8.  aeronave_motor_tipo           TIPO DO MOTOR DA AERONAVE                  * qualitativa nominal
# 9.  aeronave_motor_quantidade     QUANTIDADE DE MOTOR(ES) DA AERONAVE        * quantitativa discreta
# 10. aeronave_nivel_dano           NÍVEL DE DANO SOFRIDO PELA AERONAVE        * qualitativa ordinal
# 11. aeronave_fase_operacao        FASE DE OPERAÇÃO DA OCORRÊNCIA             * qualitativa nominal
# 12. total_fatalidades             NÚMERO DE VÍTIMAS FATAIS DA OCORRÊNCIA     * quantitativa discreta

def frequency_table(table_name, dataframe):
    table = {
        table_name: dataframe.value_counts().index,
        'Frequência absoluta': dataframe.value_counts().values,
        'Frequência relativa (%)': (dataframe.value_counts(normalize = True).values*100).round(2)
    }
    
    ft = pd.DataFrame(table)
    return ft
frequency_table('Tipo da aeronave', csv_df['aeronave_tipo_veiculo'])
frequency_table('Fabricante da aeronave', csv_df['aeronave_fabricante'])
frequency_table('Modelo da aeronave', csv_df['aeronave_modelo'])
frequency_table('Categoria da aeronave', csv_df['aeronave_pmd_categoria'])
frequency_table('Propulsão da aeronave', csv_df['aeronave_motor_tipo'])
frequency_table('Nível de dano da aeronave', csv_df['aeronave_nivel_dano'])
frequency_table('Fase de operação da ocorrência', csv_df['aeronave_fase_operacao'])
import matplotlib.pyplot as plt
%matplotlib inline

plt.style.use(style='fast')    #plt.style.available
table = {
    'Tipo da aeronave': csv_df['aeronave_tipo_veiculo'].value_counts().index,
    'Total de acidentes': csv_df['aeronave_tipo_veiculo'].value_counts().values,
}

ft = pd.DataFrame(table).plot(x='Tipo da aeronave', kind='bar', legend=False)

for patch in ft.patches: ft.text(patch.get_x(), patch.get_height() + 50, str(round((patch.get_height()), 2)))

plt.title('Acidente aéreo por tipo de aeronave')
plt.ylabel('Total de acidentes')

plt.show()
table = {
    'Fabricante da aeronave': csv_df['aeronave_fabricante'].value_counts().index,
    'Total de acidentes': csv_df['aeronave_fabricante'].value_counts().values,
}

ft = pd.DataFrame(table).plot(x='Fabricante da aeronave', kind='line', legend=False)

#for patch in ft.patches: ft.text(patch.get_x(), patch.get_height() + 50, str(round((patch.get_height()), 2)))

plt.title('Acidente aéreo por fabricante da aeronave')
plt.ylabel('Total de acidentes')

plt.show()
table = {
    'Modelo da aeronave': csv_df['aeronave_modelo'].value_counts().index,
    'Total de acidentes': csv_df['aeronave_modelo'].value_counts().values,
}

ft = pd.DataFrame(table).plot(x='Modelo da aeronave', kind='bar', legend=False)

for patch in ft.patches: ft.text(patch.get_x(), patch.get_height() + 50, str(round((patch.get_height()), 2)))

plt.title('Acidente aéreo por modelo da aeronave')
plt.ylabel('Total de acidentes')

plt.show()
table = {
    'Propulsão da aeronave': csv_df['aeronave_motor_tipo'].value_counts().index,
    'Total de acidentes': csv_df['aeronave_motor_tipo'].value_counts().values,
}

ft = pd.DataFrame(table).plot(x='Propulsão da aeronave', y='Total de acidentes', kind='pie', legend=False)

#for patch in ft.patches: ft.text(patch.get_x(), patch.get_height() + 50, str(round((patch.get_height()), 2)))

plt.title('Acidente aéreo por tipo de propulsão da aeronave')
plt.ylabel('Total de acidentes')

plt.show()
df = pd.read_csv('../input/anv.csv', delimiter=',')
df.head(1)