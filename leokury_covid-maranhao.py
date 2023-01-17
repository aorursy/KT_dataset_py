# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import openpyxl
import requests
import datetime
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
def find_specific_cell(currentSheet, text):
    for row in range(1, currentSheet.max_row + 1):
        for column in "ABCDEFGHIJKL":  # Here you can add or reduce the columns
            cell_name = "{}{}".format(column, row)
            if currentSheet[cell_name].value == text:
                #print("{1} cell is located on {0}" .format(cell_name, currentSheet[cell_name].value))
                print("cell position {} has value {}".format(cell_name, currentSheet[cell_name].value))
                return column, row
            
def get_value_from_cell(currentSheet, column, row):
    cell_name = "{}{}".format(column, row)
    print("cell position {} has value {}".format(cell_name, currentSheet[cell_name].value))
    return currentSheet[cell_name].value

def get_value_from_url(url, columns):
    r = requests.get(url)
    filename = "file.xlsx"
    open(filename, 'wb').write(r.content)

    theFile = openpyxl.load_workbook(filename, data_only=True)
    values = []
    for c in columns:
        column, row = find_specific_cell(theFile.active, c)
        value = get_value_from_cell(theFile.active, chr(ord(column) + 1), row)
        values.append(value)
    return values

# inicio da serie historica
dia = datetime.date(2020, 4, 13)
base_url = "http://www.saude.ma.gov.br/wp-content/uploads/2020/%02d/boletim%02d%02ddadosgraficos.xlsx"

data = {}

while dia != datetime.date.today():
    url = base_url % (dia.month, dia.day, dia.month)
    print(url)
    try:
        uti_ocup = get_value_from_url(url, ["% de ocupação UTI", "% de ocupação", "Total de leitos UTI", "Total de leitos"])
        data[dia] = uti_ocup
        print(f"Ocupação do dia {uti_ocup}")
    except:
        print(f"Dia {dia.day} do mês {dia.month} não encontrado")
        pass
    dia = dia + datetime.timedelta(days=1)
df = pd.DataFrame.from_dict(data, orient='index', columns=['% Ocup. UTI', '% de Ocup. leitos', 'Leitos UTI', 'Leitos'])
df
df[["% Ocup. UTI", "% de Ocup. leitos"]].plot.line(figsize=(15,5), ylim=[0, 1.05], grid=True)
plt.title("Ocupação de leitos em São Luís")
plt.ylabel("Percentual de Ocupação")
df[["Leitos UTI", "Leitos"]].plot.line(figsize=(15,5), ylim=[0, 400], grid=True)
plt.title("Leitos em São Luís")
plt.ylabel("Quantidade de leitos")
